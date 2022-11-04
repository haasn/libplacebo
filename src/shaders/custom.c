/*
 * This file is part of libplacebo.
 *
 * libplacebo is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * libplacebo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libplacebo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <math.h>
#include "gpu.h"
#include "shaders.h"

bool pl_shader_custom(pl_shader sh, const struct pl_custom_shader *params)
{
    if (params->compute) {
        int bw = PL_DEF(params->compute_group_size[0], 16);
        int bh = PL_DEF(params->compute_group_size[1], 16);
        bool flex = !params->compute_group_size[0] ||
                    !params->compute_group_size[1];
        if (!sh_try_compute(sh, bw, bh, flex, params->compute_shmem))
            return false;
    }

    if (!sh_require(sh, params->input, params->output_w, params->output_h))
        return false;

    sh->res.output = params->output;

    // Attach the variables, descriptors etc. directly instead of going via
    // `sh_var` / `sh_desc` etc. to avoid generating fresh names
    for (int i = 0; i < params->num_variables; i++) {
        struct pl_shader_var sv = params->variables[i];
        sv.data = pl_memdup(SH_TMP(sh), sv.data, pl_var_host_layout(0, &sv.var).size);
        sv.var.name = pl_strdup0(SH_TMP(sh), pl_str0(sv.var.name));
        PL_ARRAY_APPEND(sh, sh->vars, sv);
    }

    for (int i = 0; i < params->num_descriptors; i++) {
        struct pl_shader_desc sd = params->descriptors[i];
        size_t bsize = sizeof(sd.buffer_vars[0]) * sd.num_buffer_vars;
        if (bsize)
            sd.buffer_vars = pl_memdup(SH_TMP(sh), sd.buffer_vars, bsize);
        sd.desc.name = pl_strdup0(SH_TMP(sh), pl_str0(sd.desc.name));
        PL_ARRAY_APPEND(sh, sh->descs, sd);
    }

    for (int i = 0; i < params->num_vertex_attribs; i++) {
        struct pl_shader_va sva = params->vertex_attribs[i];
        size_t vsize = sva.attr.fmt->texel_size;
        for (int n = 0; n < PL_ARRAY_SIZE(sva.data); n++)
            sva.data[n] = pl_memdup(SH_TMP(sh), sva.data[n], vsize);
        sva.attr.name = pl_strdup0(SH_TMP(sh), pl_str0(sva.attr.name));
        PL_ARRAY_APPEND(sh, sh->vas, sva);
    }

    for (int i = 0; i < params->num_constants; i++) {
        struct pl_shader_const sc = params->constants[i];
        size_t csize = pl_var_type_size(sc.type);
        sc.data = pl_memdup(SH_TMP(sh), sc.data, csize);
        sc.name = pl_strdup0(SH_TMP(sh), pl_str0(sc.name));
        PL_ARRAY_APPEND(sh, sh->consts, sc);
    }

    if (params->prelude)
        GLSLP("// pl_shader_custom prelude: \n%s\n", params->prelude);
    if (params->header)
        GLSLH("// pl_shader_custom header: \n%s\n", params->header);

    if (params->description)
        sh_describe(sh, pl_strdup0(SH_TMP(sh), pl_str0(params->description)));

    if (params->body) {
        const char *output_decl = "";
        if (params->output != params->input) {
            switch (params->output) {
            case PL_SHADER_SIG_NONE: break;
            case PL_SHADER_SIG_COLOR:
                output_decl = "vec4 color = vec4(0.0);";
                break;

            case PL_SHADER_SIG_SAMPLER:
                pl_unreachable();
            }
        }

        GLSL("// pl_shader_custom \n"
             "%s                  \n"
             "{                   \n"
             "%s                  \n"
             "}                   \n",
             output_decl, params->body);
    }

    return true;
}

// Hard-coded size limits, mainly for convenience (to avoid dynamic memory)
#define SHADER_MAX_HOOKS 16
#define SHADER_MAX_BINDS 16
#define MAX_SZEXP_SIZE 32

enum szexp_op {
    SZEXP_OP_ADD,
    SZEXP_OP_SUB,
    SZEXP_OP_MUL,
    SZEXP_OP_DIV,
    SZEXP_OP_MOD,
    SZEXP_OP_NOT,
    SZEXP_OP_GT,
    SZEXP_OP_LT,
    SZEXP_OP_EQ,
};

enum szexp_tag {
    SZEXP_END = 0, // End of an RPN expression
    SZEXP_CONST, // Push a constant value onto the stack
    SZEXP_VAR_W, // Get the width/height of a named texture (variable)
    SZEXP_VAR_H,
    SZEXP_OP2, // Pop two elements and push the result of a dyadic operation
    SZEXP_OP1, // Pop one element and push the result of a monadic operation
};

struct szexp {
    enum szexp_tag tag;
    union {
        float cval;
        pl_str varname;
        enum szexp_op op;
    } val;
};

struct custom_shader_hook {
    // Variable/literal names of textures
    pl_str pass_desc;
    pl_str hook_tex[SHADER_MAX_HOOKS];
    pl_str bind_tex[SHADER_MAX_BINDS];
    pl_str save_tex;

    // Shader body itself + metadata
    pl_str pass_body;
    float offset[2];
    bool offset_align;
    int comps;

    // Special expressions governing the output size and execution conditions
    struct szexp width[MAX_SZEXP_SIZE];
    struct szexp height[MAX_SZEXP_SIZE];
    struct szexp cond[MAX_SZEXP_SIZE];

    // Special metadata for compute shaders
    bool is_compute;
    int block_w, block_h;       // Block size (each block corresponds to one WG)
    int threads_w, threads_h;   // How many threads form a WG
};

static bool parse_rpn_szexpr(pl_str line, struct szexp out[MAX_SZEXP_SIZE])
{
    int pos = 0;

    while (line.len > 0) {
        pl_str word = pl_str_split_char(line, ' ', &line);
        if (word.len == 0)
            continue;

        if (pos >= MAX_SZEXP_SIZE)
            return false;

        struct szexp *exp = &out[pos++];

        if (pl_str_eatend0(&word, ".w") || pl_str_eatend0(&word, ".width")) {
            exp->tag = SZEXP_VAR_W;
            exp->val.varname = word;
            continue;
        }

        if (pl_str_eatend0(&word, ".h") || pl_str_eatend0(&word, ".height")) {
            exp->tag = SZEXP_VAR_H;
            exp->val.varname = word;
            continue;
        }

        switch (word.buf[0]) {
        case '+': exp->tag = SZEXP_OP2; exp->val.op = SZEXP_OP_ADD; continue;
        case '-': exp->tag = SZEXP_OP2; exp->val.op = SZEXP_OP_SUB; continue;
        case '*': exp->tag = SZEXP_OP2; exp->val.op = SZEXP_OP_MUL; continue;
        case '/': exp->tag = SZEXP_OP2; exp->val.op = SZEXP_OP_DIV; continue;
        case '%': exp->tag = SZEXP_OP2; exp->val.op = SZEXP_OP_MOD; continue;
        case '!': exp->tag = SZEXP_OP1; exp->val.op = SZEXP_OP_NOT; continue;
        case '>': exp->tag = SZEXP_OP2; exp->val.op = SZEXP_OP_GT;  continue;
        case '<': exp->tag = SZEXP_OP2; exp->val.op = SZEXP_OP_LT;  continue;
        case '=': exp->tag = SZEXP_OP2; exp->val.op = SZEXP_OP_EQ;  continue;
        }

        if (word.buf[0] >= '0' && word.buf[0] <= '9') {
            exp->tag = SZEXP_CONST;
            if (!pl_str_parse_float(word, &exp->val.cval))
                return false;
            continue;
        }

        // Some sort of illegal expression
        return false;
    }

    return true;
}

// Evaluate a `szexp`, given a lookup function for named textures
// Returns whether successful. 'result' is left untouched on failure
static bool pl_eval_szexpr(pl_log log, void *priv,
                           bool (*lookup)(void *priv, pl_str var, float size[2]),
                           const struct szexp expr[MAX_SZEXP_SIZE],
                           float *result)
{
    float stack[MAX_SZEXP_SIZE] = {0};
    int idx = 0; // points to next element to push

    for (int i = 0; i < MAX_SZEXP_SIZE; i++) {
        switch (expr[i].tag) {
        case SZEXP_END:
            goto done;

        case SZEXP_CONST:
            // Since our SZEXPs are bound by MAX_SZEXP_SIZE, it should be
            // impossible to overflow the stack
            assert(idx < MAX_SZEXP_SIZE);
            stack[idx++] = expr[i].val.cval;
            continue;

        case SZEXP_OP1:
            if (idx < 1) {
                pl_warn(log, "Stack underflow in RPN expression!");
                return false;
            }

            switch (expr[i].val.op) {
            case SZEXP_OP_NOT: stack[idx-1] = !stack[idx-1]; break;
            default: pl_unreachable();
            }
            continue;

        case SZEXP_OP2:
            if (idx < 2) {
                pl_warn(log, "Stack underflow in RPN expression!");
                return false;
            }

            // Pop the operands in reverse order
            float op2 = stack[--idx];
            float op1 = stack[--idx];
            float res = 0.0;
            switch (expr[i].val.op) {
            case SZEXP_OP_ADD: res = op1 + op2; break;
            case SZEXP_OP_SUB: res = op1 - op2; break;
            case SZEXP_OP_MUL: res = op1 * op2; break;
            case SZEXP_OP_DIV: res = op1 / op2; break;
            case SZEXP_OP_MOD: res = fmodf(op1, op2); break;
            case SZEXP_OP_GT:  res = op1 > op2; break;
            case SZEXP_OP_LT:  res = op1 < op2; break;
            case SZEXP_OP_EQ:  res = fabsf(op1 - op2) <= 1e-6 * fmaxf(op1, op2); break;
            case SZEXP_OP_NOT: pl_unreachable();
            }

            if (!isfinite(res)) {
                pl_warn(log, "Illegal operation in RPN expression!");
                return false;
            }

            stack[idx++] = res;
            continue;

        case SZEXP_VAR_W:
        case SZEXP_VAR_H: {
            pl_str name = expr[i].val.varname;
            float size[2];

            if (!lookup(priv, name, size)) {
                pl_warn(log, "Variable '%.*s' not found in RPN expression!",
                        PL_STR_FMT(name));
                return false;
            }

            stack[idx++] = (expr[i].tag == SZEXP_VAR_W) ? size[0] : size[1];
            continue;
            }
        }
    }

done:
    // Return the single stack element
    if (idx != 1) {
        pl_warn(log, "Malformed stack after RPN expression!");
        return false;
    }

    *result = stack[0];
    return true;
}

static inline pl_str split_magic(pl_str *body)
{
    pl_str ret = pl_str_split_str0(*body, "//!", body);
    if (body->len) {
        // Make sure the separator is included in the remainder
        body->buf -= 3;
        body->len += 3;
    }

    return ret;
}

static bool parse_hook(pl_log log, pl_str *body, struct custom_shader_hook *out)
{
    *out = (struct custom_shader_hook){
        .pass_desc = pl_str0("unknown user shader"),
        .width = {{ SZEXP_VAR_W, { .varname = pl_str0("HOOKED") }}},
        .height = {{ SZEXP_VAR_H, { .varname = pl_str0("HOOKED") }}},
        .cond = {{ SZEXP_CONST, { .cval = 1.0 }}},
    };

    int hook_idx = 0;
    int bind_idx = 0;

    // Parse all headers
    while (true) {
        pl_str rest;
        pl_str line = pl_str_strip(pl_str_getline(*body, &rest));

        // Check for the presence of the magic line beginning
        if (!pl_str_eatstart0(&line, "//!"))
            break;

        *body = rest;

        // Parse the supported commands
        if (pl_str_eatstart0(&line, "HOOK")) {
            if (hook_idx == SHADER_MAX_HOOKS) {
                pl_err(log, "Passes may only hook up to %d textures!",
                       SHADER_MAX_HOOKS);
                return false;
            }
            out->hook_tex[hook_idx++] = pl_str_strip(line);
            continue;
        }

        if (pl_str_eatstart0(&line, "BIND")) {
            if (bind_idx == SHADER_MAX_BINDS) {
                pl_err(log, "Passes may only bind up to %d textures!",
                       SHADER_MAX_BINDS);
                return false;
            }
            out->bind_tex[bind_idx++] = pl_str_strip(line);
            continue;
        }

        if (pl_str_eatstart0(&line, "SAVE")) {
            pl_str save_tex = pl_str_strip(line);
            if (pl_str_equals0(save_tex, "HOOKED")) {
                // This is a special name that means "overwrite existing"
                // texture, which we just signal by not having any `save_tex`
                // name set.
                out->save_tex = (pl_str) {0};
            } else if (pl_str_equals0(save_tex, "MAIN")) {
                // Compatibility alias
                out->save_tex = pl_str0("MAINPRESUB");
            } else {
                out->save_tex = save_tex;
            };
            continue;
        }

        if (pl_str_eatstart0(&line, "DESC")) {
            out->pass_desc = pl_str_strip(line);
            continue;
        }

        if (pl_str_eatstart0(&line, "OFFSET")) {
            line = pl_str_strip(line);
            if (pl_str_equals0(line, "ALIGN")) {
                out->offset_align = true;
            } else {
                if (!pl_str_parse_float(pl_str_split_char(line, ' ', &line), &out->offset[0]) ||
                    !pl_str_parse_float(pl_str_split_char(line, ' ', &line), &out->offset[1]) ||
                    line.len)
                {
                    pl_err(log, "Error while parsing OFFSET!");
                    return false;
                }
            }
            continue;
        }

        if (pl_str_eatstart0(&line, "WIDTH")) {
            if (!parse_rpn_szexpr(line, out->width)) {
                pl_err(log, "Error while parsing WIDTH!");
                return false;
            }
            continue;
        }

        if (pl_str_eatstart0(&line, "HEIGHT")) {
            if (!parse_rpn_szexpr(line, out->height)) {
                pl_err(log, "Error while parsing HEIGHT!");
                return false;
            }
            continue;
        }

        if (pl_str_eatstart0(&line, "WHEN")) {
            if (!parse_rpn_szexpr(line, out->cond)) {
                pl_err(log, "Error while parsing WHEN!");
                return false;
            }
            continue;
        }

        if (pl_str_eatstart0(&line, "COMPONENTS")) {
            if (!pl_str_parse_int(pl_str_strip(line), &out->comps)) {
                pl_err(log, "Error parsing COMPONENTS: '%.*s'", PL_STR_FMT(line));
                return false;
            }
            continue;
        }

        if (pl_str_eatstart0(&line, "COMPUTE")) {
            line = pl_str_strip(line);
            bool ok = pl_str_parse_int(pl_str_split_char(line, ' ', &line), &out->block_w) &&
                      pl_str_parse_int(pl_str_split_char(line, ' ', &line), &out->block_h);

            line = pl_str_strip(line);
            if (ok && line.len) {
                ok = pl_str_parse_int(pl_str_split_char(line, ' ', &line), &out->threads_w) &&
                     pl_str_parse_int(pl_str_split_char(line, ' ', &line), &out->threads_h) &&
                     !line.len;
            } else {
                out->threads_w = out->block_w;
                out->threads_h = out->block_h;
            }

            if (!ok) {
                pl_err(log, "Error while parsing COMPUTE!");
                return false;
            }

            out->is_compute = true;
            continue;
        }

        // Unknown command type
        pl_err(log, "Unrecognized command '%.*s'!", PL_STR_FMT(line));
        return false;
    }

    // The rest of the file up until the next magic line beginning (if any)
    // shall be the shader body
    out->pass_body = split_magic(body);

    // Sanity checking
    if (hook_idx == 0)
        pl_warn(log, "Pass has no hooked textures (will be ignored)!");

    return true;
}

static bool parse_tex(pl_gpu gpu, void *alloc, pl_str *body,
                      struct pl_shader_desc *out)
{
    *out = (struct pl_shader_desc) {
        .desc = {
            .name = "USER_TEX",
            .type = PL_DESC_SAMPLED_TEX,
        },
    };

    struct pl_tex_params params = {
        .w = 1, .h = 1, .d = 0,
        .sampleable = true,
        .debug_tag = PL_DEBUG_TAG,
    };

    while (true) {
        pl_str rest;
        pl_str line = pl_str_strip(pl_str_getline(*body, &rest));

        if (!pl_str_eatstart0(&line, "//!"))
            break;

        *body = rest;

        if (pl_str_eatstart0(&line, "TEXTURE")) {
            out->desc.name = pl_strdup0(alloc, pl_str_strip(line));
            continue;
        }

        if (pl_str_eatstart0(&line, "SIZE")) {
            line = pl_str_strip(line);
            int dims = 0;
            int dim[4]; // extra space to catch invalid extra entries
            while (line.len && dims < PL_ARRAY_SIZE(dim)) {
                if (!pl_str_parse_int(pl_str_split_char(line, ' ', &line), &dim[dims++])) {
                    PL_ERR(gpu, "Error while parsing SIZE!");
                    return false;
                }
            }

            uint32_t lim = dims == 1 ? gpu->limits.max_tex_1d_dim
                         : dims == 2 ? gpu->limits.max_tex_2d_dim
                         : dims == 3 ? gpu->limits.max_tex_3d_dim
                         : 0;

            // Sanity check against GPU size limits
            switch (dims) {
            case 3:
                params.d = dim[2];
                if (params.d < 1 || params.d > lim) {
                    PL_ERR(gpu, "SIZE %d exceeds GPU's texture size limits (%d)!",
                           params.d, lim);
                    return false;
                }
                // fall through
            case 2:
                params.h = dim[1];
                if (params.h < 1 || params.h > lim) {
                    PL_ERR(gpu, "SIZE %d exceeds GPU's texture size limits (%d)!",
                           params.h, lim);
                    return false;
                }
                // fall through
            case 1:
                params.w = dim[0];
                if (params.w < 1 || params.w > lim) {
                    PL_ERR(gpu, "SIZE %d exceeds GPU's texture size limits (%d)!",
                           params.w, lim);
                    return false;
                }
                break;

            default:
                PL_ERR(gpu, "Invalid number of texture dimensions!");
                return false;
            };

            // Clear out the superfluous components
            if (dims < 3)
                params.d = 0;
            if (dims < 2)
                params.h = 0;
            continue;
        }

        if (pl_str_eatstart0(&line, "FORMAT ")) {
            line = pl_str_strip(line);
            params.format = NULL;
            for (int n = 0; n < gpu->num_formats; n++) {
                pl_fmt fmt = gpu->formats[n];
                if (pl_str_equals0(line, fmt->name)) {
                    params.format = fmt;
                    break;
                }
            }

            if (!params.format || params.format->opaque) {
                PL_ERR(gpu, "Unrecognized/unavailable FORMAT name: '%.*s'!",
                       PL_STR_FMT(line));
                return false;
            }

            if (!(params.format->caps & PL_FMT_CAP_SAMPLEABLE)) {
                PL_ERR(gpu, "Chosen FORMAT '%.*s' is not sampleable!",
                       PL_STR_FMT(line));
                return false;
            }
            continue;
        }

        if (pl_str_eatstart0(&line, "FILTER")) {
            line = pl_str_strip(line);
            if (pl_str_equals0(line, "LINEAR")) {
                out->binding.sample_mode = PL_TEX_SAMPLE_LINEAR;
            } else if (pl_str_equals0(line, "NEAREST")) {
                out->binding.sample_mode = PL_TEX_SAMPLE_NEAREST;
            } else {
                PL_ERR(gpu, "Unrecognized FILTER: '%.*s'!", PL_STR_FMT(line));
                return false;
            }
            continue;
        }

        if (pl_str_eatstart0(&line, "BORDER")) {
            line = pl_str_strip(line);
            if (pl_str_equals0(line, "CLAMP")) {
                out->binding.address_mode = PL_TEX_ADDRESS_CLAMP;
            } else if (pl_str_equals0(line, "REPEAT")) {
                out->binding.address_mode = PL_TEX_ADDRESS_REPEAT;
            } else if (pl_str_equals0(line, "MIRROR")) {
                out->binding.address_mode = PL_TEX_ADDRESS_MIRROR;
            } else {
                PL_ERR(gpu, "Unrecognized BORDER: '%.*s'!", PL_STR_FMT(line));
                return false;
            }
            continue;
        }

        if (pl_str_eatstart0(&line, "STORAGE")) {
            params.storable = true;
            out->desc.type = PL_DESC_STORAGE_IMG;
            out->desc.access = PL_DESC_ACCESS_READWRITE;
            out->memory = PL_MEMORY_COHERENT;
            continue;
        }

        PL_ERR(gpu, "Unrecognized command '%.*s'!", PL_STR_FMT(line));
        return false;
    }

    if (!params.format) {
        PL_ERR(gpu, "No FORMAT specified!");
        return false;
    }

    int caps = params.format->caps;
    if (out->binding.sample_mode == PL_TEX_SAMPLE_LINEAR && !(caps & PL_FMT_CAP_LINEAR)) {
        PL_ERR(gpu, "The specified texture format cannot be linear filtered!");
        return false;
    }

    // Decode the rest of the section (up to the next //! marker) as raw hex
    // data for the texture
    pl_str tex, hexdata = split_magic(body);
    if (!pl_str_decode_hex(NULL, pl_str_strip(hexdata), &tex)) {
        PL_ERR(gpu, "Error while parsing TEXTURE body: must be a valid "
                    "hexadecimal sequence!");
        return false;
    }

    int texels = params.w * PL_DEF(params.h, 1) * PL_DEF(params.d, 1);
    size_t expected_len = texels * params.format->texel_size;
    if (tex.len == 0 && params.storable) {
        // In this case, it's okay that the texture has no initial data
        pl_free_ptr(&tex.buf);
    } else if (tex.len != expected_len) {
        PL_ERR(gpu, "Shader TEXTURE size mismatch: got %zu bytes, expected %zu!",
               tex.len, expected_len);
        pl_free(tex.buf);
        return false;
    }

    params.initial_data = tex.buf;
    out->binding.object = pl_tex_create(gpu, &params);
    pl_free(tex.buf);

    if (!out->binding.object) {
        PL_ERR(gpu, "Failed creating custom texture!");
        return false;
    }

    return true;
}

static bool parse_buf(pl_gpu gpu, void *alloc, pl_str *body,
                      struct pl_shader_desc *out)
{
    *out = (struct pl_shader_desc) {
        .desc = {
            .name = "USER_BUF",
            .type = PL_DESC_BUF_UNIFORM,
        },
    };

    // Temporary, to allow deferring variable placement until all headers
    // have been processed (in order to e.g. determine buffer type)
    void *tmp = pl_tmp(alloc); // will be freed automatically on failure
    PL_ARRAY(struct pl_var) vars = {0};

    while (true) {
        pl_str rest;
        pl_str line = pl_str_strip(pl_str_getline(*body, &rest));

        if (!pl_str_eatstart0(&line, "//!"))
            break;

        *body = rest;

        if (pl_str_eatstart0(&line, "BUFFER")) {
            out->desc.name = pl_strdup0(alloc, pl_str_strip(line));
            continue;
        }

        if (pl_str_eatstart0(&line, "STORAGE")) {
            out->desc.type = PL_DESC_BUF_STORAGE;
            out->desc.access = PL_DESC_ACCESS_READWRITE;
            out->memory = PL_MEMORY_COHERENT;
            continue;
        }

        if (pl_str_eatstart0(&line, "VAR")) {
            pl_str type_name = pl_str_split_char(pl_str_strip(line), ' ', &line);
            struct pl_var var = {0};
            for (const struct pl_named_var *nv = pl_var_glsl_types; nv->glsl_name; nv++) {
                if (pl_str_equals0(type_name, nv->glsl_name)) {
                    var = nv->var;
                    break;
                }
            }

            if (!var.type) {
                // No type found
                PL_ERR(gpu, "Unrecognized GLSL type '%.*s'!", PL_STR_FMT(type_name));
                return false;
            }

            pl_str var_name = pl_str_split_char(line, '[', &line);
            if (line.len > 0) {
                // Parse array dimension
                if (!pl_str_parse_int(pl_str_split_char(line, ']', NULL), &var.dim_a)) {
                    PL_ERR(gpu, "Failed parsing array dimension from [%.*s!",
                           PL_STR_FMT(line));
                    return false;
                }

                if (var.dim_a < 1) {
                    PL_ERR(gpu, "Invalid array dimension %d!", var.dim_a);
                    return false;
                }
            }

            var.name = pl_strdup0(alloc, pl_str_strip(var_name));
            PL_ARRAY_APPEND(tmp, vars, var);
            continue;
        }

        PL_ERR(gpu, "Unrecognized command '%.*s'!", PL_STR_FMT(line));
        return false;
    }

    // Try placing all of the buffer variables
    for (int i = 0; i < vars.num; i++) {
        if (!sh_buf_desc_append(alloc, gpu, out, NULL, vars.elem[i])) {
            PL_ERR(gpu, "Custom buffer exceeds GPU limitations!");
            return false;
        }
    }

    // Decode the rest of the section (up to the next //! marker) as raw hex
    // data for the buffer
    pl_str data, hexdata = split_magic(body);
    if (!pl_str_decode_hex(tmp, pl_str_strip(hexdata), &data)) {
        PL_ERR(gpu, "Error while parsing BUFFER body: must be a valid "
                    "hexadecimal sequence!");
        return false;
    }

    size_t buf_size = sh_buf_desc_size(out);
    if (data.len == 0 && out->desc.type == PL_DESC_BUF_STORAGE) {
        // In this case, it's okay that the buffer has no initial data
    } else if (data.len != buf_size) {
        PL_ERR(gpu, "Shader BUFFER size mismatch: got %zu bytes, expected %zu!",
               data.len, buf_size);
        return false;
    }

    out->binding.object = pl_buf_create(gpu, pl_buf_params(
        .size = buf_size,
        .uniform = out->desc.type == PL_DESC_BUF_UNIFORM,
        .storable = out->desc.type == PL_DESC_BUF_STORAGE,
        .initial_data = data.len ? data.buf : NULL,
    ));

    if (!out->binding.object) {
        PL_ERR(gpu, "Failed creating custom buffer!");
        return false;
    }

    pl_free(tmp);
    return true;
}

static enum pl_hook_stage mp_stage_to_pl(pl_str stage)
{
    if (pl_str_equals0(stage, "RGB"))
        return PL_HOOK_RGB_INPUT;
    if (pl_str_equals0(stage, "LUMA"))
        return PL_HOOK_LUMA_INPUT;
    if (pl_str_equals0(stage, "CHROMA"))
        return PL_HOOK_CHROMA_INPUT;
    if (pl_str_equals0(stage, "ALPHA"))
        return PL_HOOK_ALPHA_INPUT;
    if (pl_str_equals0(stage, "XYZ"))
        return PL_HOOK_XYZ_INPUT;

    if (pl_str_equals0(stage, "CHROMA_SCALED"))
        return PL_HOOK_CHROMA_SCALED;
    if (pl_str_equals0(stage, "ALPHA_SCALED"))
        return PL_HOOK_ALPHA_SCALED;

    if (pl_str_equals0(stage, "NATIVE"))
        return PL_HOOK_NATIVE;
    if (pl_str_equals0(stage, "MAINPRESUB"))
        return PL_HOOK_RGB;
    if (pl_str_equals0(stage, "MAIN"))
        return PL_HOOK_RGB; // Note: conflicts with above!

    if (pl_str_equals0(stage, "LINEAR"))
        return PL_HOOK_LINEAR;
    if (pl_str_equals0(stage, "SIGMOID"))
        return PL_HOOK_SIGMOID;
    if (pl_str_equals0(stage, "PREKERNEL"))
        return PL_HOOK_PRE_KERNEL;
    if (pl_str_equals0(stage, "POSTKERNEL"))
        return PL_HOOK_POST_KERNEL;

    if (pl_str_equals0(stage, "SCALED"))
        return PL_HOOK_SCALED;
    if (pl_str_equals0(stage, "OUTPUT"))
        return PL_HOOK_OUTPUT;

    return 0;
}

static pl_str pl_stage_to_mp(enum pl_hook_stage stage)
{
    switch (stage) {
    case PL_HOOK_RGB_INPUT:     return pl_str0("RGB");
    case PL_HOOK_LUMA_INPUT:    return pl_str0("LUMA");
    case PL_HOOK_CHROMA_INPUT:  return pl_str0("CHROMA");
    case PL_HOOK_ALPHA_INPUT:   return pl_str0("ALPHA");
    case PL_HOOK_XYZ_INPUT:     return pl_str0("XYZ");

    case PL_HOOK_CHROMA_SCALED: return pl_str0("CHROMA_SCALED");
    case PL_HOOK_ALPHA_SCALED:  return pl_str0("ALPHA_SCALED");

    case PL_HOOK_NATIVE:        return pl_str0("NATIVE");
    case PL_HOOK_RGB:           return pl_str0("MAINPRESUB");

    case PL_HOOK_LINEAR:        return pl_str0("LINEAR");
    case PL_HOOK_SIGMOID:       return pl_str0("SIGMOID");
    case PL_HOOK_PRE_KERNEL:    return pl_str0("PREKERNEL");
    case PL_HOOK_POST_KERNEL:   return pl_str0("POSTKERNEL");

    case PL_HOOK_SCALED:        return pl_str0("SCALED");
    case PL_HOOK_OUTPUT:        return pl_str0("OUTPUT");
    };

    pl_unreachable();
}

struct hook_pass {
    enum pl_hook_stage exec_stages;
    struct custom_shader_hook hook;
};

struct pass_tex {
    pl_str name;
    pl_tex tex;

    // Metadata
    struct pl_rect2df rect;
    struct pl_color_repr repr;
    struct pl_color_space color;
    int comps;
};

struct hook_priv {
    pl_log log;
    pl_gpu gpu;
    void *alloc;

    PL_ARRAY(struct hook_pass) hook_passes;

    // Fixed (for shader-local resources)
    PL_ARRAY(struct pl_shader_desc) descriptors;

    // Dynamic per pass
    enum pl_hook_stage save_stages;
    PL_ARRAY(struct pass_tex) pass_textures;
    pl_shader trc_helper;

    // State for PRNG/frame count
    int frame_count;
    uint64_t prng_state[4];
};

static void hook_reset(void *priv)
{
    struct hook_priv *p = priv;
    p->pass_textures.num = 0;
}

struct szexp_ctx {
    struct hook_priv *priv;
    const struct pl_hook_params *params;
    struct pass_tex hooked;
};

static bool lookup_tex(void *priv, pl_str var, float size[2])
{
    struct szexp_ctx *ctx = priv;
    struct hook_priv *p = ctx->priv;
    const struct pl_hook_params *params = ctx->params;

    if (pl_str_equals0(var, "HOOKED")) {
        pl_assert(ctx->hooked.tex);
        size[0] = ctx->hooked.tex->params.w;
        size[1] = ctx->hooked.tex->params.h;
        return true;
    }

    if (pl_str_equals0(var, "NATIVE_CROPPED")) {
        size[0] = fabs(pl_rect_w(params->src_rect));
        size[1] = fabs(pl_rect_h(params->src_rect));
        return true;
    }

    if (pl_str_equals0(var, "OUTPUT")) {
        size[0] = abs(pl_rect_w(params->dst_rect));
        size[1] = abs(pl_rect_h(params->dst_rect));
        return true;
    }

    if (pl_str_equals0(var, "MAIN"))
        var = pl_str0("MAINPRESUB");

    for (int i = 0; i < p->pass_textures.num; i++) {
        if (pl_str_equals(var, p->pass_textures.elem[i].name)) {
            pl_tex tex = p->pass_textures.elem[i].tex;
            size[0] = tex->params.w;
            size[1] = tex->params.h;
            return true;
        }
    }

    return false;
}

static double prng_step(uint64_t s[4])
{
    const uint64_t result = s[0] + s[3];
    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = (s[3] << 45) | (s[3] >> (64 - 45));
    return (result >> 11) * 0x1.0p-53;
}

static bool bind_pass_tex(pl_shader sh, pl_str name,
                          const struct pass_tex *ptex,
                          const struct pl_rect2df *rect,
                          bool hooked, bool mainpresub)
{
    ident_t id, pos, size, pt;

    // Compatibility with mpv texture binding semantics
    id = sh_bind(sh, ptex->tex, PL_TEX_ADDRESS_CLAMP, PL_TEX_SAMPLE_LINEAR,
                 "hook_tex", rect, &pos, &size, &pt);
    if (!id)
        return false;

    GLSLH("#define %.*s_raw %s \n", PL_STR_FMT(name), id);
    GLSLH("#define %.*s_pos %s \n", PL_STR_FMT(name), pos);
    GLSLH("#define %.*s_map %s_map \n", PL_STR_FMT(name), pos);
    GLSLH("#define %.*s_size %s \n", PL_STR_FMT(name), size);
    GLSLH("#define %.*s_pt %s \n", PL_STR_FMT(name), pt);

    float off[2] = { ptex->rect.x0, ptex->rect.y0 };
    GLSLH("#define %.*s_off %s \n", PL_STR_FMT(name),
          sh_var(sh, (struct pl_shader_var) {
              .var = pl_var_vec2("offset"),
              .data = off,
    }));

    struct pl_color_repr repr = ptex->repr;
    ident_t scale = SH_FLOAT(pl_color_repr_normalize(&repr));
    GLSLH("#define %.*s_mul %s \n", PL_STR_FMT(name), scale);

    // Compatibility with mpv
    GLSLH("#define %.*s_rot mat2(1.0, 0.0, 0.0, 1.0) \n", PL_STR_FMT(name));

    // Sampling function boilerplate
    GLSLH("#define %.*s_tex(pos) (%s * vec4(%s(%s, pos))) \n",
          PL_STR_FMT(name), scale, sh_tex_fn(sh, ptex->tex->params), id);
    GLSLH("#define %.*s_texOff(off) (%.*s_tex(%s + %s * vec2(off))) \n",
          PL_STR_FMT(name), PL_STR_FMT(name), pos, pt);

    bool can_gather = ptex->tex->params.format->gatherable;
    if (can_gather) {
        GLSLH("#define %.*s_gather(pos, c) (%s * vec4(textureGather(%s, pos, c))) \n",
              PL_STR_FMT(name), scale, id);
    }

    if (hooked) {
        GLSLH("#define HOOKED_raw %.*s_raw \n", PL_STR_FMT(name));
        GLSLH("#define HOOKED_pos %.*s_pos \n", PL_STR_FMT(name));
        GLSLH("#define HOOKED_size %.*s_size \n", PL_STR_FMT(name));
        GLSLH("#define HOOKED_rot %.*s_rot \n", PL_STR_FMT(name));
        GLSLH("#define HOOKED_off %.*s_off \n", PL_STR_FMT(name));
        GLSLH("#define HOOKED_pt %.*s_pt \n", PL_STR_FMT(name));
        GLSLH("#define HOOKED_map %.*s_map \n", PL_STR_FMT(name));
        GLSLH("#define HOOKED_mul %.*s_mul \n", PL_STR_FMT(name));
        GLSLH("#define HOOKED_tex %.*s_tex \n", PL_STR_FMT(name));
        GLSLH("#define HOOKED_texOff %.*s_texOff \n", PL_STR_FMT(name));
        if (can_gather)
            GLSLH("#define HOOKED_gather %.*s_gather \n", PL_STR_FMT(name));
    }

    if (mainpresub) {
        GLSLH("#define MAIN_raw MAINPRESUB_raw \n");
        GLSLH("#define MAIN_pos MAINPRESUB_pos \n");
        GLSLH("#define MAIN_size MAINPRESUB_size \n");
        GLSLH("#define MAIN_rot MAINPRESUB_rot \n");
        GLSLH("#define MAIN_off MAINPRESUB_off \n");
        GLSLH("#define MAIN_pt MAINPRESUB_pt \n");
        GLSLH("#define MAIN_map MAINPRESUB_map \n");
        GLSLH("#define MAIN_mul MAINPRESUB_mul \n");
        GLSLH("#define MAIN_tex MAINPRESUB_tex \n");
        GLSLH("#define MAIN_texOff MAINPRESUB_texOff \n");
        if (can_gather)
            GLSLH("#define MAIN_gather MAINPRESUB_gather \n");
    }

    return true;
}

static void save_pass_tex(struct hook_priv *p, struct pass_tex ptex)
{

    for (int i = 0; i < p->pass_textures.num; i++) {
        if (!pl_str_equals(p->pass_textures.elem[i].name, ptex.name))
            continue;

        p->pass_textures.elem[i] = ptex;
        return;
    }

    // No texture with this name yet, append new one
    PL_ARRAY_APPEND(p->alloc, p->pass_textures, ptex);
}

static struct pl_hook_res hook_hook(void *priv, const struct pl_hook_params *params)
{
    struct hook_priv *p = priv;
    pl_str stage = pl_stage_to_mp(params->stage);
    struct pl_hook_res res = {0};

    pl_shader sh = NULL;
    struct szexp_ctx scope = {
        .priv = p,
        .params = params,
        .hooked = {
            .name = stage,
            .tex = params->tex,
            .rect = params->rect,
            .repr = params->repr,
            .color = params->color,
            .comps = params->components,
        },
    };

    // Save the input texture if needed
    if (p->save_stages & params->stage) {
        PL_TRACE(p, "Saving input texture '%.*s' for binding",
                 PL_STR_FMT(scope.hooked.name));
        save_pass_tex(p, scope.hooked);
    }

    for (int n = 0; n < p->hook_passes.num; n++) {
        const struct hook_pass *pass = &p->hook_passes.elem[n];
        if (!(pass->exec_stages & params->stage))
            continue;

        const struct custom_shader_hook *hook = &pass->hook;
        PL_TRACE(p, "Executing hook pass %d on stage '%.*s': %.*s",
                 n, PL_STR_FMT(stage), PL_STR_FMT(hook->pass_desc));

        // Test for execution condition
        float run = 0;
        if (!pl_eval_szexpr(p->log, &scope, lookup_tex, hook->cond, &run))
            goto error;

        if (!run) {
            PL_TRACE(p, "Skipping hook due to condition");
            continue;
        }

        // Generate a new shader object
        sh = pl_dispatch_begin(params->dispatch);

        // Bind all necessary input textures
        for (int i = 0; i < PL_ARRAY_SIZE(hook->bind_tex); i++) {
            pl_str texname = hook->bind_tex[i];
            if (!texname.len)
                break;

            // Convenience alias, to allow writing shaders that are oblivious
            // of the exact stage they hooked. This simply translates to
            // whatever stage actually fired the hook.
            bool hooked = false, mainpresub = false;
            if (pl_str_equals0(texname, "HOOKED")) {
                // Continue with binding this, under the new name
                texname = stage;
                hooked = true;
            }

            // Compatibility alias, because MAIN and MAINPRESUB mean the same
            // thing to libplacebo, but user shaders are still written as
            // though they can be different concepts.
            if (pl_str_equals0(texname, "MAIN") ||
                pl_str_equals0(texname, "MAINPRESUB"))
            {
                texname = pl_str0("MAINPRESUB");
                mainpresub = true;
            }

            for (int j = 0; j < p->descriptors.num; j++) {
                if (pl_str_equals0(texname, p->descriptors.elem[j].desc.name)) {
                    // Directly bind this, no need to bother with all the
                    // `bind_pass_tex` boilerplate
                    ident_t id = sh_desc(sh, p->descriptors.elem[j]);
                    GLSLH("#define %.*s %s \n", PL_STR_FMT(texname), id);

                    if (p->descriptors.elem[j].desc.type == PL_DESC_SAMPLED_TEX) {
                        pl_tex tex = p->descriptors.elem[j].binding.object;
                        GLSLH("#define %.*s_tex(pos) (%s(%s, pos)) \n",
                              PL_STR_FMT(texname), sh_tex_fn(sh, tex->params), id);
                    }
                    goto next_bind;
                }
            }

            for (int j = 0; j < p->pass_textures.num; j++) {
                if (pl_str_equals(texname, p->pass_textures.elem[j].name)) {
                    // Note: We bind the whole texture, rather than
                    // params->rect, because user shaders in general are not
                    // designed to handle cropped input textures.
                    const struct pass_tex *ptex = &p->pass_textures.elem[j];
                    struct pl_rect2df rect = {
                        0, 0, ptex->tex->params.w, ptex->tex->params.h,
                    };

                    if (hook->offset_align && pl_str_equals(texname, stage)) {
                        float sx = pl_rect_w(params->rect) / pl_rect_w(params->src_rect),
                              sy = pl_rect_h(params->rect) / pl_rect_h(params->src_rect),
                              ox = params->rect.x0 - sx * params->src_rect.x0,
                              oy = params->rect.y0 - sy * params->src_rect.y0;

                        PL_TRACE(p, "Aligning plane with ref: %f %f", ox, oy);
                        pl_rect2df_offset(&rect, ox, oy);
                    }

                    if (!bind_pass_tex(sh, texname, &p->pass_textures.elem[j],
                                       &rect, hooked, mainpresub))
                    {
                        goto error;
                    }
                    goto next_bind;
                }
            }

            // If none of the above matched, this is an unknown texture name,
            // so silently ignore this pass to match the mpv behavior
            PL_TRACE(p, "Skipping hook due to no texture named '%.*s'.",
                     PL_STR_FMT(texname));
            pl_dispatch_abort(params->dispatch, &sh);
            goto next_pass;

    next_bind: ; // outer 'continue'
        }

        // Set up the input variables
        p->frame_count++;
        GLSLH("#define frame %s \n", sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_int("frame"),
            .data = &p->frame_count,
            .dynamic = true,
        }));

        float random = prng_step(p->prng_state);
        GLSLH("#define random %s \n", sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_float("random"),
            .data = &random,
            .dynamic = true,
        }));

        float src_size[2] = { pl_rect_w(params->src_rect), pl_rect_h(params->src_rect) };
        GLSLH("#define input_size %s \n", sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_vec2("input_size"),
            .data = src_size,
        }));

        float dst_size[2] = { pl_rect_w(params->dst_rect), pl_rect_h(params->dst_rect) };
        GLSLH("#define target_size %s \n", sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_vec2("target_size"),
            .data = dst_size,
        }));

        float tex_off[2] = { params->src_rect.x0, params->src_rect.y0 };
        GLSLH("#define tex_offset %s \n", sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_vec2("tex_offset"),
            .data = tex_off,
        }));

        // Helper sub-shaders
        uint64_t sh_id = SH_PARAMS(sh).id;
        pl_shader_reset(p->trc_helper, pl_shader_params(
            .id = ++sh_id,
            .gpu = p->gpu,
        ));
        pl_shader_linearize(p->trc_helper, params->orig_color);
        GLSLH("#define linearize %s \n", sh_subpass(sh, p->trc_helper));

        pl_shader_reset(p->trc_helper, pl_shader_params(
            .id = ++sh_id,
            .gpu = p->gpu,
        ));
        pl_shader_delinearize(p->trc_helper, params->orig_color);
        GLSLH("#define delinearize %s \n", sh_subpass(sh, p->trc_helper));

        // Load and run the user shader itself
        sh_append_str(sh, SH_BUF_HEADER, hook->pass_body);
        sh_describe(sh, pl_strdup0(SH_TMP(sh), hook->pass_desc));

        // Resolve output size and create framebuffer
        float out_size[2] = {0};
        if (!pl_eval_szexpr(p->log, &scope, lookup_tex, hook->width,  &out_size[0]) ||
            !pl_eval_szexpr(p->log, &scope, lookup_tex, hook->height, &out_size[1]))
        {
            goto error;
        }

        int out_w = roundf(out_size[0]),
            out_h = roundf(out_size[1]);

        if (!sh_require(sh, PL_SHADER_SIG_COLOR, out_w, out_h))
            goto error;

        // Generate a new texture to store the render result
        pl_tex fbo;
        fbo = params->get_tex(params->priv, out_w, out_h);
        if (!fbo) {
            PL_ERR(p, "Failed dispatching hook: `get_tex` callback failed?");
            goto error;
        }

        bool ok;
        if (hook->is_compute) {

            if (!sh_try_compute(sh, hook->threads_w, hook->threads_h, false, 0) ||
                !fbo->params.storable)
            {
                PL_ERR(p, "Failed dispatching COMPUTE shader");
                goto error;
            }

            GLSLP("#define out_image %s \n", sh_desc(sh, (struct pl_shader_desc) {
                .binding.object = fbo,
                .desc = {
                    .name = "out_image",
                    .type = PL_DESC_STORAGE_IMG,
                    .access = PL_DESC_ACCESS_WRITEONLY,
                },
            }));

            sh->res.output = PL_SHADER_SIG_NONE;

            GLSL("hook(); \n");
            ok = pl_dispatch_compute(params->dispatch, pl_dispatch_compute_params(
                .shader = &sh,
                .dispatch_size = {
                    // Round up as many blocks as are needed to cover the image
                    (out_w + hook->block_w - 1) / hook->block_w,
                    (out_h + hook->block_h - 1) / hook->block_h,
                    1,
                },
                .width  = out_w,
                .height = out_h,
            ));

        } else {

            // Default non-COMPUTE shaders to explicitly use fragment shaders
            // only, to avoid breaking things like fwidth()
            sh->type = PL_DEF(sh->type, SH_FRAGMENT);

            GLSL("vec4 color = hook(); \n");
            ok = pl_dispatch_finish(params->dispatch, pl_dispatch_params(
                .shader = &sh,
                .target = fbo,
            ));

        }

        if (!ok)
            goto error;

        float sx = (float) out_w / params->tex->params.w,
              sy = (float) out_h / params->tex->params.h,
              x0 = sx * params->rect.x0 + hook->offset[0],
              y0 = sy * params->rect.y0 + hook->offset[1];

        struct pl_rect2df new_rect = {
            x0,
            y0,
            x0 + sx * pl_rect_w(params->rect),
            y0 + sy * pl_rect_h(params->rect),
        };

        if (hook->offset_align) {
            float rx = pl_rect_w(new_rect) / pl_rect_w(params->src_rect),
                  ry = pl_rect_h(new_rect) / pl_rect_h(params->src_rect),
                  ox = rx * params->src_rect.x0 - sx * params->rect.x0,
                  oy = ry * params->src_rect.y0 - sy * params->rect.y0;

            pl_rect2df_offset(&new_rect, ox, oy);
        }

        // Save the result of this shader invocation
        struct pass_tex ptex = {
            .name = hook->save_tex.len ? hook->save_tex : stage,
            .tex = fbo,
            .repr = params->repr,
            .color = params->color,
            .comps  = PL_DEF(hook->comps, params->components),
            .rect = new_rect,
        };

        // It's assumed that users will correctly normalize the input
        pl_color_repr_normalize(&ptex.repr);

        PL_TRACE(p, "Saving output texture '%.*s' from hook execution on '%.*s'",
                 PL_STR_FMT(ptex.name), PL_STR_FMT(stage));

        save_pass_tex(p, ptex);

        // Update the result object, unless we saved to a different name
        if (pl_str_equals(ptex.name, stage)) {
            scope.hooked = ptex;
            res = (struct pl_hook_res) {
                .output = PL_HOOK_SIG_TEX,
                .tex = fbo,
                .repr = ptex.repr,
                .color = ptex.color,
                .components = ptex.comps,
                .rect = new_rect,
            };
        }

next_pass: ;
    }

    return res;

error:
    pl_dispatch_abort(params->dispatch, &sh);
    return (struct pl_hook_res) { .failed = true };
}

const struct pl_hook *pl_mpv_user_shader_parse(pl_gpu gpu,
                                               const char *shader_text,
                                               size_t shader_len)
{
    if (!shader_len)
        return NULL;

    struct pl_hook *hook = pl_alloc_obj(NULL, hook, struct hook_priv);
    struct hook_priv *p = PL_PRIV(hook);

    *hook = (struct pl_hook) {
        .input = PL_HOOK_SIG_TEX,
        .priv = p,
        .reset = hook_reset,
        .hook = hook_hook,
    };

    *p = (struct hook_priv) {
        .log = gpu->log,
        .gpu = gpu,
        .alloc = hook,
        .trc_helper = pl_shader_alloc(gpu->log, NULL),
        .prng_state = {
            // Determined by fair die roll
            0xb76d71f9443c228allu, 0x93a02092fc4807e8llu,
            0x06d81748f838bd07llu, 0x9381ee129dddce6cllu,
        },
    };

    pl_str shader = { (uint8_t *) shader_text, shader_len };
    shader = pl_strdup(hook, shader);

    // Skip all garbage (e.g. comments) before the first header
    int pos = pl_str_find(shader, pl_str0("//!"));
    if (pos < 0) {
        PL_ERR(gpu, "Shader appears to contain no headers?");
        goto error;
    }
    shader = pl_str_drop(shader, pos);

    // Loop over the file
    while (shader.len > 0)
    {
        // Peek at the first header to dispatch the right type
        if (pl_str_startswith0(shader, "//!TEXTURE")) {
            struct pl_shader_desc sd;
            if (!parse_tex(gpu, hook, &shader, &sd))
                goto error;

            PL_INFO(gpu, "Registering named texture '%s'", sd.desc.name);
            PL_ARRAY_APPEND(hook, p->descriptors, sd);
            continue;
        }

        if (pl_str_startswith0(shader, "//!BUFFER")) {
            struct pl_shader_desc sd;
            if (!parse_buf(gpu, hook, &shader, &sd))
                goto error;

            PL_INFO(gpu, "Registering named buffer '%s'", sd.desc.name);
            PL_ARRAY_APPEND(hook, p->descriptors, sd);
            continue;
        }

        struct custom_shader_hook h;
        if (!parse_hook(gpu->log, &shader, &h))
            goto error;

        struct hook_pass pass = {
            .exec_stages = 0,
            .hook = h,
        };

        for (int i = 0; i < PL_ARRAY_SIZE(h.hook_tex); i++)
            pass.exec_stages |= mp_stage_to_pl(h.hook_tex[i]);
        for (int i = 0; i < PL_ARRAY_SIZE(h.bind_tex); i++) {
            p->save_stages |= mp_stage_to_pl(h.bind_tex[i]);
            if (pl_str_equals0(h.bind_tex[i], "HOOKED"))
                p->save_stages |= pass.exec_stages;
        }

        // As an extra precaution, this avoids errors when trying to run
        // conditions against planes that were never hooked. As a sole
        // exception, OUTPUT is special because it's hard-coded to return the
        // dst_rect even before it was hooked. (This is an apparently
        // undocumented mpv quirk, but shaders rely on it in practice)
        enum pl_hook_stage rpn_stages = 0;
        for (int i = 0; i < PL_ARRAY_SIZE(h.width); i++) {
            if (h.width[i].tag == SZEXP_VAR_W || h.width[i].tag == SZEXP_VAR_H)
                rpn_stages |= mp_stage_to_pl(h.width[i].val.varname);
        }
        for (int i = 0; i < PL_ARRAY_SIZE(h.height); i++) {
            if (h.height[i].tag == SZEXP_VAR_W || h.height[i].tag == SZEXP_VAR_H)
                rpn_stages |= mp_stage_to_pl(h.height[i].val.varname);
        }
        for (int i = 0; i < PL_ARRAY_SIZE(h.cond); i++) {
            if (h.cond[i].tag == SZEXP_VAR_W || h.cond[i].tag == SZEXP_VAR_H)
                rpn_stages |= mp_stage_to_pl(h.cond[i].val.varname);
        }

        p->save_stages |= rpn_stages & ~PL_HOOK_OUTPUT;

        PL_INFO(gpu, "Registering hook pass: %.*s", PL_STR_FMT(h.pass_desc));
        PL_ARRAY_APPEND(hook, p->hook_passes, pass);
    }

    // We need to hook on both the exec and save stages, so that we can keep
    // track of any textures we might need
    hook->stages |= p->save_stages;
    for (int i = 0; i < p->hook_passes.num; i++)
        hook->stages |= p->hook_passes.elem[i].exec_stages;

    return hook;

error:
    pl_free(hook);
    return NULL;
}

void pl_mpv_user_shader_destroy(const struct pl_hook **hookp)
{
    const struct pl_hook *hook = *hookp;
    if (!hook)
        return;

    struct hook_priv *p = PL_PRIV(hook);
    for (int i = 0; i < p->descriptors.num; i++) {
        switch (p->descriptors.elem[i].desc.type) {
            case PL_DESC_BUF_UNIFORM:
            case PL_DESC_BUF_STORAGE:
            case PL_DESC_BUF_TEXEL_UNIFORM:
            case PL_DESC_BUF_TEXEL_STORAGE: {
                pl_buf buf = p->descriptors.elem[i].binding.object;
                pl_buf_destroy(p->gpu, &buf);
                break;
            }

            case PL_DESC_SAMPLED_TEX:
            case PL_DESC_STORAGE_IMG: {
                pl_tex tex = p->descriptors.elem[i].binding.object;
                pl_tex_destroy(p->gpu, &tex);
                break;

            case PL_DESC_INVALID:
            case PL_DESC_TYPE_COUNT:
                pl_unreachable();
            }
        }
    }

    pl_shader_free(&p->trc_helper);
    pl_free((void *) hook);
    *hookp = NULL;
}
