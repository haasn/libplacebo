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

// Hard-coded size limits, mainly for convenience (to avoid dynamic memory)
#define SHADER_MAX_HOOKS 16
#define SHADER_MAX_BINDS 16
#define MAX_SZEXP_SIZE 32

enum szexp_op {
    SZEXP_OP_ADD,
    SZEXP_OP_SUB,
    SZEXP_OP_MUL,
    SZEXP_OP_DIV,
    SZEXP_OP_NOT,
    SZEXP_OP_GT,
    SZEXP_OP_LT,
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
        struct bstr varname;
        enum szexp_op op;
    } val;
};

struct custom_shader_hook {
    // Variable/literal names of textures
    struct bstr pass_desc;
    struct bstr hook_tex[SHADER_MAX_HOOKS];
    struct bstr bind_tex[SHADER_MAX_BINDS];
    struct bstr save_tex;

    // Shader body itself + metadata
    struct bstr pass_body;
    float offset[2];
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

struct custom_shader_tex {
    struct bstr name;
    const struct pl_tex *tex;
};

static bool parse_rpn_szexpr(struct bstr line, struct szexp out[MAX_SZEXP_SIZE])
{
    int pos = 0;

    while (line.len > 0) {
        struct bstr word = bstr_strip(bstr_splitchar(line, &line, ' '));
        if (word.len == 0)
            continue;

        if (pos >= MAX_SZEXP_SIZE)
            return false;

        struct szexp *exp = &out[pos++];

        if (bstr_eatend0(&word, ".w") || bstr_eatend0(&word, ".width")) {
            exp->tag = SZEXP_VAR_W;
            exp->val.varname = word;
            continue;
        }

        if (bstr_eatend0(&word, ".h") || bstr_eatend0(&word, ".height")) {
            exp->tag = SZEXP_VAR_H;
            exp->val.varname = word;
            continue;
        }

        switch (word.start[0]) {
        case '+': exp->tag = SZEXP_OP2; exp->val.op = SZEXP_OP_ADD; continue;
        case '-': exp->tag = SZEXP_OP2; exp->val.op = SZEXP_OP_SUB; continue;
        case '*': exp->tag = SZEXP_OP2; exp->val.op = SZEXP_OP_MUL; continue;
        case '/': exp->tag = SZEXP_OP2; exp->val.op = SZEXP_OP_DIV; continue;
        case '!': exp->tag = SZEXP_OP1; exp->val.op = SZEXP_OP_NOT; continue;
        case '>': exp->tag = SZEXP_OP2; exp->val.op = SZEXP_OP_GT;  continue;
        case '<': exp->tag = SZEXP_OP2; exp->val.op = SZEXP_OP_LT;  continue;
        }

        if (word.start[0] >= '0' && word.start[0] <= '9') {
            exp->tag = SZEXP_CONST;
            if (bstr_sscanf(word, "%f", &exp->val.cval) != 1)
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
static bool pl_eval_szexpr(struct pl_context *ctx, void *priv,
                           bool (*lookup)(void *priv, struct bstr var, float size[2]),
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
                pl_warn(ctx, "Stack underflow in RPN expression!");
                return false;
            }

            switch (expr[i].val.op) {
            case SZEXP_OP_NOT: stack[idx-1] = !stack[idx-1]; break;
            default: abort();
            }
            continue;

        case SZEXP_OP2:
            if (idx < 2) {
                pl_warn(ctx, "Stack underflow in RPN expression!");
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
            case SZEXP_OP_GT:  res = op1 > op2; break;
            case SZEXP_OP_LT:  res = op1 < op2; break;
            default: abort();
            }

            if (!isfinite(res)) {
                pl_warn(ctx, "Illegal operation in RPN expression!");
                return false;
            }

            stack[idx++] = res;
            continue;

        case SZEXP_VAR_W:
        case SZEXP_VAR_H: {
            struct bstr name = expr[i].val.varname;
            float size[2];

            if (!lookup(priv, name, size)) {
                pl_warn(ctx, "Variable '%.*s' not found in RPN expression!",
                        BSTR_P(name));
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
        pl_warn(ctx, "Malformed stack after RPN expression!");
        return false;
    }

    *result = stack[0];
    return true;
}

static bool parse_hook(struct pl_context *ctx, struct bstr *body,
                       struct custom_shader_hook *out)
{
    *out = (struct custom_shader_hook){
        .pass_desc = bstr0("(unknown)"),
        .width = {{ SZEXP_VAR_W, { .varname = bstr0("HOOKED") }}},
        .height = {{ SZEXP_VAR_H, { .varname = bstr0("HOOKED") }}},
        .cond = {{ SZEXP_CONST, { .cval = 1.0 }}},
    };

    int hook_idx = 0;
    int bind_idx = 0;

    // Parse all headers
    while (true) {
        struct bstr rest;
        struct bstr line = bstr_strip(bstr_getline(*body, &rest));

        // Check for the presence of the magic line beginning
        if (!bstr_eatstart0(&line, "//!"))
            break;

        *body = rest;

        // Parse the supported commands
        if (bstr_eatstart0(&line, "HOOK")) {
            if (hook_idx == SHADER_MAX_HOOKS) {
                pl_err(ctx, "Passes may only hook up to %d textures!",
                       SHADER_MAX_HOOKS);
                return false;
            }
            out->hook_tex[hook_idx++] = bstr_strip(line);
            continue;
        }

        if (bstr_eatstart0(&line, "BIND")) {
            if (bind_idx == SHADER_MAX_BINDS) {
                pl_err(ctx, "Passes may only bind up to %d textures!",
                       SHADER_MAX_BINDS);
                return false;
            }
            out->bind_tex[bind_idx++] = bstr_strip(line);
            continue;
        }

        if (bstr_eatstart0(&line, "SAVE")) {
            struct bstr save_tex = bstr_strip(line);
            if (bstr_equals0(save_tex, "HOOKED")) {
                // This is a special name that means "overwrite existing"
                // texture, which we just signal by not having any `save_tex`
                // name set.
                out->save_tex = (struct bstr) {0};
            } else {
                out->save_tex = save_tex;
            };
            continue;
        }

        if (bstr_eatstart0(&line, "DESC")) {
            out->pass_desc = bstr_strip(line);
            continue;
        }

        if (bstr_eatstart0(&line, "OFFSET")) {
            float ox, oy;
            if (bstr_sscanf(line, "%f %f", &ox, &oy) != 2) {
                pl_err(ctx, "Error while parsing OFFSET!");
                return false;
            }
            out->offset[0] = ox;
            out->offset[1] = oy;
            continue;
        }

        if (bstr_eatstart0(&line, "WIDTH")) {
            if (!parse_rpn_szexpr(line, out->width)) {
                pl_err(ctx, "Error while parsing WIDTH!");
                return false;
            }
            continue;
        }

        if (bstr_eatstart0(&line, "HEIGHT")) {
            if (!parse_rpn_szexpr(line, out->height)) {
                pl_err(ctx, "Error while parsing HEIGHT!");
                return false;
            }
            continue;
        }

        if (bstr_eatstart0(&line, "WHEN")) {
            if (!parse_rpn_szexpr(line, out->cond)) {
                pl_err(ctx, "Error while parsing WHEN!");
                return false;
            }
            continue;
        }

        if (bstr_eatstart0(&line, "COMPONENTS")) {
            if (bstr_sscanf(line, "%d", &out->comps) != 1) {
                pl_err(ctx, "Error while parsing COMPONENTS!");
                return false;
            }
            continue;
        }

        if (bstr_eatstart0(&line, "COMPUTE")) {
            int num = bstr_sscanf(line, "%d %d %d %d",
                                  &out->block_w, &out->block_h,
                                  &out->threads_w, &out->threads_h);

            if (num == 2 || num == 4) {
                out->is_compute = true;
                if (num == 2) {
                    out->threads_w = out->block_w;
                    out->threads_h = out->block_h;
                }
            } else {
                pl_err(log, "Error while parsing COMPUTE!");
                return false;
            }
            continue;
        }

        // Unknown command type
        pl_err(ctx, "Unrecognized command '%.*s'!", BSTR_P(line));
        return false;
    }

    // The rest of the file up until the next magic line beginning (if any)
    // shall be the shader body
    if (bstr_split_tok(*body, "//!", &out->pass_body, body)) {
        // Make sure the magic line is part of the rest
        body->start -= 3;
        body->len += 3;
    }

    // Sanity checking
    if (hook_idx == 0)
        pl_warn(ctx, "Pass has no hooked textures (will be ignored)!");

    return true;
}

static bool parse_tex(const struct pl_gpu *gpu, struct bstr *body,
                      struct custom_shader_tex *out)
{
    *out = (struct custom_shader_tex) {
        .name = bstr0("USER_TEX"),
    };

    struct pl_tex_params params = {
        .w = 1, .h = 1, .d = 0,
        .sampleable = true,
    };

    while (true) {
        struct bstr rest;
        struct bstr line = bstr_strip(bstr_getline(*body, &rest));

        if (!bstr_eatstart0(&line, "//!"))
            break;

        *body = rest;

        if (bstr_eatstart0(&line, "TEXTURE")) {
            out->name = bstr_strip(line);
            continue;
        }

        if (bstr_eatstart0(&line, "SIZE")) {
            int dims = bstr_sscanf(line, "%d %d %d", &params.w, &params.h, &params.d);
            int lim = dims == 1 ? gpu->limits.max_tex_1d_dim
                    : dims == 2 ? gpu->limits.max_tex_2d_dim
                    : dims == 3 ? gpu->limits.max_tex_3d_dim
                    : 0;

            // Sanity check against GPU size limits
            switch (dims) {
            case 3:
                if (params.d < 1 || params.d > lim) {
                    PL_ERR(gpu, "SIZE %d exceeds GPU's texture size limits (%d)!",
                           params.d, lim);
                    return false;
                }
                // fall through
            case 2:
                if (params.h < 1 || params.h > lim) {
                    PL_ERR(gpu, "SIZE %d exceeds GPU's texture size limits (%d)!",
                           params.h, lim);
                    return false;
                }
                // fall through
            case 1:
                if (params.w < 1 || params.w > lim) {
                    PL_ERR(gpu, "SIZE %d exceeds GPU's texture size limits (%d)!",
                           params.w, lim);
                    return false;
                }
                break;

            default:
                PL_ERR(gpu, "Error while parsing SIZE!");
                return false;
            };

            // Clear out the superfluous components
            if (dims < 3)
                params.d = 0;
            if (dims < 2)
                params.h = 0;
            continue;
        }

        if (bstr_eatstart0(&line, "FORMAT ")) {
            line = bstr_strip(line);
            params.format = NULL;
            for (int n = 0; n < gpu->num_formats; n++) {
                const struct pl_fmt *fmt = gpu->formats[n];
                if (bstr_equals0(line, fmt->name)) {
                    params.format = fmt;
                    break;
                }
            }

            if (!params.format || params.format->opaque) {
                PL_ERR(gpu, "Unrecognized/unavailable FORMAT name: '%.*s'!",
                       BSTR_P(line));
                return false;
            }

            if (!(params.format->caps & PL_FMT_CAP_SAMPLEABLE)) {
                PL_ERR(gpu, "Chosen FORMAT '%.*s' is not sampleable!",
                       BSTR_P(line));
                return false;
            }
            continue;
        }

        if (bstr_eatstart0(&line, "FILTER")) {
            line = bstr_strip(line);
            if (bstr_equals0(line, "LINEAR")) {
                params.sample_mode = PL_TEX_SAMPLE_LINEAR;
            } else if (bstr_equals0(line, "NEAREST")) {
                params.sample_mode = PL_TEX_SAMPLE_NEAREST;
            } else {
                PL_ERR(gpu, "Unrecognized FILTER: '%.*s'!", BSTR_P(line));
                return false;
            }
            continue;
        }

        if (bstr_eatstart0(&line, "BORDER")) {
            line = bstr_strip(line);
            if (bstr_equals0(line, "CLAMP")) {
                params.address_mode = PL_TEX_ADDRESS_CLAMP;
            } else if (bstr_equals0(line, "REPEAT")) {
                params.address_mode = PL_TEX_ADDRESS_REPEAT;
            } else if (bstr_equals0(line, "MIRROR")) {
                params.address_mode = PL_TEX_ADDRESS_MIRROR;
            } else {
                PL_ERR(gpu, "Unrecognized BORDER: '%.*s'!", BSTR_P(line));
                return false;
            }
            continue;
        }

        PL_ERR(gpu, "Unrecognized command '%.*s'!", BSTR_P(line));
        return false;
    }

    if (!params.format) {
        PL_ERR(gpu, "No FORMAT specified!");
        return false;
    }

    int caps = params.format->caps;
    if (params.sample_mode == PL_TEX_SAMPLE_LINEAR && !(caps & PL_FMT_CAP_LINEAR)) {
        PL_ERR(gpu, "The specified texture format cannot be linear filtered!");
        return false;
    }

    // Decode the rest of the section (up to the next //! marker) as raw hex
    // data for the texture
    struct bstr hexdata;
    if (bstr_split_tok(*body, "//!", &hexdata, body)) {
        // Make sure the magic line is part of the rest
        body->start -= 3;
        body->len += 3;
    }

    struct bstr tex;
    if (!bstr_decode_hex(NULL, bstr_strip(hexdata), &tex)) {
        PL_ERR(gpu, "Error while parsing TEXTURE body: must be a valid "
                    "hexadecimal sequence, on a single line!");
        return false;
    }

    int texels = params.w * PL_DEF(params.h, 1) * PL_DEF(params.d, 1);
    size_t expected_len = texels * params.format->texel_size;
    if (tex.len != expected_len) {
        PL_ERR(gpu, "Shader TEXTURE size mismatch: got %zu bytes, expected %zu!",
               tex.len, expected_len);
        talloc_free(tex.start);
        return false;
    }

    params.initial_data = tex.start;
    out->tex = pl_tex_create(gpu, &params);
    talloc_free(tex.start);

    if (!out->tex) {
        PL_ERR(gpu, "Failed uploading custom texture!");
        return false;
    }

    return true;
}

// Parse the next shader block from `body`. The callbacks are invoked on every
// valid shader block parsed.
static bool parse_user_shader(const struct pl_gpu *gpu, struct bstr shader, void *priv,
                              bool (*dohook)(void *p, struct custom_shader_hook hook),
                              bool (*dotex)(void *p, struct custom_shader_tex tex))
{
    if (!shader.len)
        return false;

    // Skip all garbage (e.g. comments) before the first header
    int pos = bstr_find(shader, bstr0("//!"));
    if (pos < 0) {
        PL_ERR(gpu, "Shader appears to contain no headers?");
        return false;
    }
    shader = bstr_cut(shader, pos);

    // Loop over the file
    while (shader.len > 0)
    {
        // Peek at the first header to dispatch the right type
        if (bstr_startswith0(shader, "//!TEXTURE")) {
            struct custom_shader_tex t;
            if (!parse_tex(gpu, &shader, &t) || !dotex(priv, t))
                return false;
            continue;
        }

        struct custom_shader_hook h;
        if (!parse_hook(gpu->ctx, &shader, &h) || !dohook(priv, h))
            return false;
    }

    return true;
}

static enum pl_hook_stage mp_stage_to_pl(struct bstr stage)
{
    if (bstr_equals0(stage, "RGB"))
        return PL_HOOK_RGB_INPUT;
    if (bstr_equals0(stage, "LUMA"))
        return PL_HOOK_LUMA_INPUT;
    if (bstr_equals0(stage, "CHROMA"))
        return PL_HOOK_CHROMA_INPUT;
    if (bstr_equals0(stage, "ALPHA"))
        return PL_HOOK_ALPHA_INPUT;
    if (bstr_equals0(stage, "XYZ"))
        return PL_HOOK_XYZ_INPUT;

    if (bstr_equals0(stage, "CHROMA_SCALED"))
        return PL_HOOK_CHROMA_SCALED;
    if (bstr_equals0(stage, "ALPHA_SCALED"))
        return PL_HOOK_ALPHA_SCALED;

    if (bstr_equals0(stage, "NATIVE"))
        return PL_HOOK_NATIVE;
    if (bstr_equals0(stage, "MAINPRESUB"))
        return PL_HOOK_RGB;
    if (bstr_equals0(stage, "MAIN"))
        return PL_HOOK_RGB; // Note: conflicts with above!

    if (bstr_equals0(stage, "LINEAR"))
        return PL_HOOK_LINEAR;
    if (bstr_equals0(stage, "SIGMOID"))
        return PL_HOOK_SIGMOID;
    if (bstr_equals0(stage, "PREKERNEL"))
        return PL_HOOK_PRE_KERNEL;
    if (bstr_equals0(stage, "POSTKERNEL"))
        return PL_HOOK_POST_KERNEL;

    if (bstr_equals0(stage, "SCALED"))
        return PL_HOOK_SCALED;
    if (bstr_equals0(stage, "OUTPUT"))
        return PL_HOOK_OUTPUT;

    return 0;
}

static struct bstr pl_stage_to_mp(enum pl_hook_stage stage)
{
    switch (stage) {
    case PL_HOOK_RGB_INPUT:     return bstr0("RGB");
    case PL_HOOK_LUMA_INPUT:    return bstr0("LUMA");
    case PL_HOOK_CHROMA_INPUT:  return bstr0("CHROMA");
    case PL_HOOK_ALPHA_INPUT:   return bstr0("ALPHA");
    case PL_HOOK_XYZ_INPUT:     return bstr0("XYZ");

    case PL_HOOK_CHROMA_SCALED: return bstr0("CHROMA_SCALED");
    case PL_HOOK_ALPHA_SCALED:  return bstr0("ALPHA_SCALED");

    case PL_HOOK_NATIVE:        return bstr0("NATIVE");
    case PL_HOOK_RGB:           return bstr0("MAINPRESUB");

    case PL_HOOK_LINEAR:        return bstr0("LINEAR");
    case PL_HOOK_SIGMOID:       return bstr0("SIGMOID");
    case PL_HOOK_PRE_OVERLAY:   return bstr0("PREOVERLAY"); // Note: doesn't exist!
    case PL_HOOK_PRE_KERNEL:    return bstr0("PREKERNEL");
    case PL_HOOK_POST_KERNEL:   return bstr0("POSTKERNEL");

    case PL_HOOK_SCALED:        return bstr0("SCALED");
    case PL_HOOK_OUTPUT:        return bstr0("OUTPUT");

    default:                    return bstr0("UNKNOWN");
    };
}

struct hook_pass {
    enum pl_hook_stage exec_stages;
    struct custom_shader_hook hook;
};

struct pass_tex {
    struct bstr name;
    const struct pl_tex *tex;

    // Metadata
    struct pl_rect2df rect;
    struct pl_color_repr repr;
    struct pl_color_space color;
    int comps;
};

struct hook_priv {
    struct pl_context *ctx;
    const struct pl_gpu *gpu;
    void *tactx;

    struct hook_pass *hook_passes;
    int num_hook_passes;

    // Fixed (for shader-local textures)
    struct custom_shader_tex *lut_textures;
    int num_lut_textures;

    // Dynamic per pass
    enum pl_hook_stage save_stages;
    struct pass_tex *pass_textures;
    int num_pass_textures;

    // State for PRNG/frame count
    int frame_count;
    uint64_t prng_state[4];
};

static void hook_reset(void *priv)
{
    struct hook_priv *p = priv;
    p->num_pass_textures = 0;
}

struct szexp_ctx {
    struct hook_priv *priv;
    const struct pl_hook_params *params;
};

static bool lookup_tex(void *priv, struct bstr var, float size[2])
{
    struct szexp_ctx *ctx = priv;
    struct hook_priv *p = ctx->priv;
    const struct pl_hook_params *params = ctx->params;

    if (bstr_equals0(var, "HOOKED")) {
        pl_assert(params->tex);
        size[0] = params->tex->params.w;
        size[1] = params->tex->params.h;
        return true;
    }

    if (bstr_equals0(var, "NATIVE_CROPPED")) {
        size[0] = pl_rect_w(params->src_rect);
        size[1] = pl_rect_h(params->src_rect);
        return true;
    }

    if (bstr_equals0(var, "OUTPUT")) {
        size[0] = pl_rect_w(params->dst_rect);
        size[1] = pl_rect_h(params->dst_rect);
        return true;
    }

    if (bstr_equals0(var, "MAIN"))
        var = bstr0("MAINPRESUB");

    for (int i = 0; i < p->num_pass_textures; i++) {
        if (bstr_equals(var, p->pass_textures[i].name)) {
            const struct pl_tex *tex = p->pass_textures[i].tex;
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

static bool bind_pass_tex(struct pl_shader *sh, struct bstr name,
                          const struct pass_tex *ptex)
{
    ident_t id, pos, size, pt;
    id = sh_bind(sh, ptex->tex, "hook_tex", &ptex->rect, &pos, &size, &pt);
    if (!id)
        return false;

    GLSLH("#define %.*s_raw %s \n", BSTR_P(name), id);
    GLSLH("#define %.*s_pos %s \n", BSTR_P(name), pos);
    GLSLH("#define %.*s_map %s_map \n", BSTR_P(name), pos);
    GLSLH("#define %.*s_size %s \n", BSTR_P(name), size);
    GLSLH("#define %.*s_pt %s \n", BSTR_P(name), pt);

    float off[2] = { ptex->rect.x0, ptex->rect.y0 };
    GLSLH("#define %.*s_off %s \n", BSTR_P(name), sh_var(sh, (struct pl_shader_var) {
        .var = pl_var_vec2("offset"),
        .data = off,
    }));

    struct pl_color_repr repr = ptex->repr;
    float scale = pl_color_repr_normalize(&repr);
    GLSLH("#define %.*s_mul %f \n", BSTR_P(name), scale);

    // Compatibility with mpv
    GLSLH("#define %.*s_rot mat2(1.0, 0.0, 0.0, 1.0) \n", BSTR_P(name));

    // Sampling function boilerplate
    GLSLH("#define %.*s_tex(pos) (%f * vec4(%s(%s, pos))) \n",
          BSTR_P(name), scale, sh_tex_fn(sh, ptex->tex), id);
    GLSLH("#define %.*s_texOff(off) (%.*s_tex(%s + %s * vec2(off))) \n",
          BSTR_P(name), BSTR_P(name), pos, pt);

    return true;
}

static void save_pass_tex(struct hook_priv *p, struct pass_tex ptex)
{

    for (int i = 0; i < p->num_pass_textures; i++) {
        if (!bstr_equals(p->pass_textures[i].name, ptex.name))
            continue;

        p->pass_textures[i] = ptex;
        return;
    }

    // No texture with this name yet, append new one
    TARRAY_APPEND(p->tactx, p->pass_textures, p->num_pass_textures, ptex);
}

static struct pl_hook_res hook_hook(void *priv, const struct pl_hook_params *params)
{
    struct hook_priv *p = priv;
    struct bstr stage = pl_stage_to_mp(params->stage);
    struct pl_hook_res res = {0};

    // Save the input texture if needed
    if (p->save_stages & params->stage) {
        pl_assert(params->tex);
        struct pass_tex ptex = {
            .name = stage,
            .tex = params->tex,
            .rect = params->rect,
            .repr = params->repr,
            .color = params->color,
            .comps = params->components,
        };

        PL_TRACE(p, "Saving input texture '%.*s' for binding", BSTR_P(ptex.name));
        save_pass_tex(p, ptex);
    }

    struct pl_shader *sh = NULL;
    struct szexp_ctx scope = {
        .priv = p,
        .params = params,
    };

    for (int n = 0; n < p->num_hook_passes; n++) {
        const struct hook_pass *pass = &p->hook_passes[n];
        if (!(pass->exec_stages & params->stage))
            continue;

        const struct custom_shader_hook *hook = &pass->hook;
        PL_TRACE(p, "Executing hook pass %d on stage '%.*s': %.*s",
                 n, BSTR_P(stage), BSTR_P(hook->pass_desc));

        // Test for execution condition
        float run = 0;
        if (!pl_eval_szexpr(p->ctx, &scope, lookup_tex, hook->cond, &run))
            goto error;

        if (!run) {
            PL_TRACE(p, "Skipping hook due to condition");
            continue;
        }

        float out_size[2] = {0};
        if (!pl_eval_szexpr(p->ctx, &scope, lookup_tex, hook->width,  &out_size[0]) ||
            !pl_eval_szexpr(p->ctx, &scope, lookup_tex, hook->height, &out_size[1]))
        {
            goto error;
        }

        // Generate a new texture to store the render result
        const struct pl_tex *fbo;
        fbo = params->get_tex(params->priv, out_size[0], out_size[1]);
        if (!fbo) {
            PL_ERR(p, "Failed dispatching hook: `get_tex` callback failed?");
            goto error;
        }

        // Generate a new shader object
        sh = pl_dispatch_begin(params->dispatch);
        if (!sh_require(sh, PL_SHADER_SIG_NONE, out_size[0], out_size[1]))
            goto error;

        if (hook->is_compute) {
            if (!sh_try_compute(sh, hook->threads_w, hook->threads_h, false, 0) ||
                !fbo->params.storable)
            {
                PL_ERR(p, "Failed dispatching COMPUTE shader");
                goto error;
            }
        }

        // Bind all necessary input textures
        for (int i = 0; i < PL_ARRAY_SIZE(hook->bind_tex); i++) {
            struct bstr texname = hook->bind_tex[i];
            if (!texname.start)
                break;

            // Convenience alias, to allow writing shaders that are oblivious
            // of the exact stage they hooked. This simply translates to
            // whatever stage actually fired the hook.
            if (bstr_equals0(texname, "HOOKED")) {
                GLSLH("#define HOOKED_raw %.*s_raw \n", BSTR_P(stage));
                GLSLH("#define HOOKED_pos %.*s_pos \n", BSTR_P(stage));
                GLSLH("#define HOOKED_size %.*s_size \n", BSTR_P(stage));
                GLSLH("#define HOOKED_rot %.*s_rot \n", BSTR_P(stage));
                GLSLH("#define HOOKED_off %.*s_off \n", BSTR_P(stage));
                GLSLH("#define HOOKED_pt %.*s_pt \n", BSTR_P(stage));
                GLSLH("#define HOOKED_map %.*s_map \n", BSTR_P(stage));
                GLSLH("#define HOOKED_mul %.*s_mul \n", BSTR_P(stage));
                GLSLH("#define HOOKED_tex %.*s_tex \n", BSTR_P(stage));
                GLSLH("#define HOOKED_texOff %.*s_texOff \n", BSTR_P(stage));

                // Continue with binding this, under the new name
                texname = stage;
            }

            // Compatibility alias, because MAIN and MAINPRESUB mean the same
            // thing to libplacebo, but user shaders are still written as
            // though they can be different concepts.
            if (bstr_equals0(texname, "MAIN")) {
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

                texname = bstr0("MAINPRESUB");
            }

            for (int j = 0; j < p->num_lut_textures; j++) {
                if (bstr_equals(texname, p->lut_textures[j].name)) {
                    // Directly bind this, no need to bother with all the
                    // `bind_pass_tex` boilerplate
                    ident_t id = sh_desc(sh, (struct pl_shader_desc) {
                        .desc = {
                            .name = "hook_lut",
                            .type = PL_DESC_SAMPLED_TEX,
                        },
                        .object = p->lut_textures[j].tex,
                    });
                    GLSLH("#define %.*s %s \n", BSTR_P(texname), id);
                    GLSLH("#define %.*s_tex(pos) (%s(%s, pos)) \n",
                          BSTR_P(texname),
                          sh_tex_fn(sh, p->lut_textures[j].tex), id);
                    goto next_bind;
                }
            }

            for (int j = 0; j < p->num_pass_textures; j++) {
                if (bstr_equals(texname, p->pass_textures[j].name)) {
                    if (!bind_pass_tex(sh, texname, &p->pass_textures[j]))
                        goto error;
                    goto next_bind;
                }
            }

            // If none of the above matched, this is a bogus/unknown texture name
            PL_ERR(p, "Tried binding unknown texture '%.*s'!", BSTR_P(texname));
            goto error;

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

        // Load and run the user shader itself
        pl_shader_append_bstr(sh, SH_BUF_HEADER, hook->pass_body);

        if (hook->is_compute) {
            GLSLP("#define out_image %s \n", sh_desc(sh, (struct pl_shader_desc) {
                .desc = {
                    .name = "out_image",
                    .type = PL_DESC_STORAGE_IMG,
                    .access = PL_DESC_ACCESS_WRITEONLY,
                },
                .object = fbo,
            }));

            sh->res.output = PL_SHADER_SIG_NONE;

            int dispatch_size[3] = {
                // Round up as many blocks as are needed to cover the image
                (out_size[0] + hook->block_w - 1) / hook->block_w,
                (out_size[1] + hook->block_h - 1) / hook->block_h,
                1,
            };

            GLSL("hook(); \n");
            if (!pl_dispatch_compute(params->dispatch, &sh, dispatch_size,
                                     out_size[0], out_size[1]))
                goto error;
        } else {
            GLSL("vec4 color = hook(); \n");
            if (!pl_dispatch_finish(params->dispatch, &sh, fbo, NULL, NULL))
                goto error;
        }

        // Update the rendering rect based on the given transform / offset
        float sx = out_size[0] / pl_rect_w(params->rect),
              sy = out_size[1] / pl_rect_h(params->rect),
              x0 = sx * params->rect.x0 + hook->offset[0],
              y0 = sy * params->rect.y0 + hook->offset[1];

        struct pl_rect2df new_rect = {
            x0,
            y0,
            x0 + out_size[0],
            y0 + out_size[1],
        };

        // Save the result of this shader invocation
        struct pass_tex ptex = {
            .name = hook->save_tex.start ? hook->save_tex : stage,
            .tex = fbo,
            .repr = params->repr,
            .color = params->color,
            .comps  = PL_DEF(hook->comps, params->components),
            .rect = new_rect,
        };

        // It's assumed that users will correctly normalize the input
        pl_color_repr_normalize(&ptex.repr);

        PL_TRACE(p, "Saving output texture '%.*s' from hook execution on '%.*s'",
                 BSTR_P(ptex.name), BSTR_P(stage));

        save_pass_tex(p, ptex);

        // Update the result object, unless we saved to a different name
        if (!hook->save_tex.start) {
            res = (struct pl_hook_res) {
                .output = PL_HOOK_SIG_TEX,
                .tex = fbo,
                .repr = ptex.repr,
                .color = ptex.color,
                .components = PL_DEF(hook->comps, params->components),
                .rect = new_rect,
            };
        }
    }

    return res;

error:
    return (struct pl_hook_res) { .failed = true };
}

static bool register_hook(void *priv, struct custom_shader_hook hook)
{
    struct hook_priv *p = priv;
    struct hook_pass pass = {
        .exec_stages = 0,
        .hook = hook,
    };

    for (int i = 0; i < PL_ARRAY_SIZE(hook.hook_tex); i++)
        pass.exec_stages |= mp_stage_to_pl(hook.hook_tex[i]);
    for (int i = 0; i < PL_ARRAY_SIZE(hook.bind_tex); i++) {
        p->save_stages |= mp_stage_to_pl(hook.bind_tex[i]);
        if (bstr_equals0(hook.bind_tex[i], "HOOKED"))
            p->save_stages |= pass.exec_stages;
    }

    PL_INFO(p, "Registering hook pass: %.*s", BSTR_P(hook.pass_desc));
    TARRAY_APPEND(p->tactx, p->hook_passes, p->num_hook_passes, pass);
    return true;
}

static bool register_tex(void *priv, struct custom_shader_tex tex)
{
    struct hook_priv *p = priv;

    PL_INFO(p, "Registering named texture '%.*s'", BSTR_P(tex.name));
    TARRAY_APPEND(p->tactx, p->lut_textures, p->num_lut_textures, tex);
    return true;
}

const struct pl_hook *pl_mpv_user_shader_parse(const struct pl_gpu *gpu,
                                               const char *shader_text,
                                               size_t shader_len)
{
    struct pl_hook *hook = talloc_priv(NULL, struct pl_hook, struct hook_priv);
    struct hook_priv *p = TA_PRIV(hook);

    *hook = (struct pl_hook) {
        .input = PL_HOOK_SIG_TEX,
        .priv = p,
        .reset = hook_reset,
        .hook = hook_hook,
    };

    *p = (struct hook_priv) {
        .ctx = gpu->ctx,
        .gpu = gpu,
        .tactx = hook,
        .prng_state = {
            // Determined by fair die roll
            0xb76d71f9443c228allu, 0x93a02092fc4807e8llu,
            0x06d81748f838bd07llu, 0x9381ee129dddce6cllu,
        },
    };

    struct bstr text = { (char *) shader_text, shader_len };
    text = bstrdup(hook, text);
    if (!parse_user_shader(gpu, text, p, register_hook, register_tex))
        goto error;

    // We need to hook on both the exec and save stages, so that we can keep
    // track of any textures we might need
    hook->stages |= p->save_stages;
    for (int i = 0; i < p->num_hook_passes; i++)
        hook->stages |= p->hook_passes[i].exec_stages;

    return hook;

error:
    talloc_free(hook);
    return NULL;
}

void pl_mpv_user_shader_destroy(const struct pl_hook **hookp)
{
    const struct pl_hook *hook = *hookp;
    if (!hook)
        return;

    struct hook_priv *p = TA_PRIV(hook);
    for (int i = 0; i < p->num_lut_textures; i++)
        pl_tex_destroy(p->gpu, &p->lut_textures[i].tex);

    talloc_free((void *) hook);
}
