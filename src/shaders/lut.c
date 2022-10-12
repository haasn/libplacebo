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
#include <ctype.h>

#include "shaders.h"

static inline bool isnumeric(char c)
{
    return (c >= '0' && c <= '9') || c == '-';
}

void pl_lut_free(struct pl_custom_lut **lut)
{
    pl_free_ptr(lut);
}

struct pl_custom_lut *pl_lut_parse_cube(pl_log log, const char *cstr, size_t cstr_len)
{
    struct pl_custom_lut *lut = pl_zalloc_ptr(NULL, lut);
    pl_str str = (pl_str) { (uint8_t *) cstr, cstr_len };
    lut->signature = pl_str_hash(str);
    int entries = 0;

    float min[3] = { 0.0, 0.0, 0.0 };
    float max[3] = { 1.0, 1.0, 1.0 };

    // Parse header
    while (str.len && !isnumeric(str.buf[0])) {
        pl_str line = pl_str_strip(pl_str_getline(str, &str));
        if (!line.len)
            continue; // skip empty line

        if (pl_str_eatstart0(&line, "TITLE")) {
            pl_info(log, "Loading LUT: %.*s", PL_STR_FMT(pl_str_strip(line)));
            continue;
        }

        if (pl_str_eatstart0(&line, "LUT_3D_SIZE")) {
            line = pl_str_strip(line);
            int size;
            if (!pl_str_parse_int(line, &size)) {
                pl_err(log, "Failed parsing dimension '%.*s'", PL_STR_FMT(line));
                goto error;
            }
            if (size <= 0 || size > 1024) {
                pl_err(log, "Invalid 3DLUT size: %dx%d%x", size, size, size);
                goto error;
            }

            lut->size[0] = lut->size[1] = lut->size[2] = size;
            entries = size * size * size;
            continue;
        }

        if (pl_str_eatstart0(&line, "LUT_1D_SIZE")) {
            line = pl_str_strip(line);
            int size;
            if (!pl_str_parse_int(line, &size)) {
                pl_err(log, "Failed parsing dimension '%.*s'", PL_STR_FMT(line));
                goto error;
            }
            if (size <= 0 || size > 65536) {
                pl_err(log, "Invalid 1DLUT size: %d", size);
                goto error;
            }

            lut->size[0] = size;
            lut->size[1] = lut->size[2] = 0;
            entries = size;
            continue;
        }

        if (pl_str_eatstart0(&line, "DOMAIN_MIN")) {
            line = pl_str_strip(line);
            if (!pl_str_parse_float(pl_str_split_char(line, ' ', &line), &min[0]) ||
                !pl_str_parse_float(pl_str_split_char(line, ' ', &line), &min[1]) ||
                !pl_str_parse_float(line, &min[2]))
            {
                pl_err(log, "Failed parsing domain: '%.*s'", PL_STR_FMT(line));
                goto error;
            }
            continue;
        }

        if (pl_str_eatstart0(&line, "DOMAIN_MAX")) {
            line = pl_str_strip(line);
            if (!pl_str_parse_float(pl_str_split_char(line, ' ', &line), &max[0]) ||
                !pl_str_parse_float(pl_str_split_char(line, ' ', &line), &max[1]) ||
                !pl_str_parse_float(line, &max[2]))
            {
                pl_err(log, "Failed parsing domain: '%.*s'", PL_STR_FMT(line));
                goto error;
            }
            continue;
        }

        if (pl_str_eatstart0(&line, "#")) {
            pl_debug(log, "Unhandled .cube comment: %.*s",
                     PL_STR_FMT(pl_str_strip(line)));
            continue;
        }

        pl_warn(log, "Unhandled .cube line: %.*s", PL_STR_FMT(pl_str_strip(line)));
    }

    if (!entries) {
        pl_err(log, "Missing LUT size specification?");
        goto error;
    }

    for (int i = 0; i < 3; i++) {
        if (max[i] - min[i] < 1e-6) {
            pl_err(log, "Invalid domain range: [%f, %f]", min[i], max[i]);
            goto error;
        }
    }

    float *data = pl_alloc(lut, sizeof(float[3]) * entries);
    lut->data = data;

    // Parse LUT body
    clock_t start = clock();
    for (int n = 0; n < entries; n++) {
        for (int c = 0; c < 3; c++) {
            static const char * const digits = "0123456789.-+e";

            // Extract valid digit sequence
            size_t len = pl_strspn(str, digits);
            pl_str entry = (pl_str) { str.buf, len };
            str.buf += len;
            str.len -= len;

            if (!entry.len) {
                if (!str.len) {
                    pl_err(log, "Failed parsing LUT: Unexpected EOF, expected "
                           "%d entries, got %d", entries * 3, n * 3 + c + 1);
                } else {
                    pl_err(log, "Failed parsing LUT: Unexpected '%c', expected "
                           "digit", str.buf[0]);
                }
                goto error;
            }

            float num;
            if (!pl_str_parse_float(entry, &num)) {
                pl_err(log, "Failed parsing float value '%.*s'", PL_STR_FMT(entry));
                goto error;
            }

            // Rescale to range 0.0 - 1.0
            *data++ = (num - min[c]) / (max[c] - min[c]);

            // Skip whitespace between digits
            str = pl_str_strip(str);
        }
    }

    str = pl_str_strip(str);
    if (str.len)
        pl_warn(log, "Extra data after LUT?... ignoring '%c'", str.buf[0]);

    pl_log_cpu_time(log, start, clock(), "parsing .cube LUT");
    return lut;

error:
    pl_free(lut);
    return NULL;
}

static void fill_lut(void *datap, const struct sh_lut_params *params)
{
    const struct pl_custom_lut *lut = params->priv;

    int dim_r = params->width;
    int dim_g = PL_DEF(params->height, 1);
    int dim_b = PL_DEF(params->depth, 1);

    float *data = datap;
    for (int b = 0; b < dim_b; b++) {
        for (int g = 0; g < dim_g; g++) {
            for (int r = 0; r < dim_r; r++) {
                size_t offset = (b * dim_g + g) * dim_r + r;
                const float *src = &lut->data[offset * 3];
                float *dst = &data[offset * 4];
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
                dst[3] = 0.0f;
            }
        }
    }
}

void pl_shader_custom_lut(pl_shader sh, const struct pl_custom_lut *lut,
                          pl_shader_obj *lut_state)
{
    if (!lut)
        return;

    int dims;
    if (lut->size[0] > 0 && lut->size[1] > 0 && lut->size[2] > 0) {
        dims = 3;
    } else if (lut->size[0] > 0 && !lut->size[1] && !lut->size[2]) {
        dims = 1;
    } else {
        SH_FAIL(sh, "Invalid dimensions %dx%dx%d for pl_custom_lut, must be 1D "
                "or 3D!", lut->size[0], lut->size[1], lut->size[2]);
        return;
    }

    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    ident_t fun = sh_lut(sh, sh_lut_params(
        .object     = lut_state,
        .var_type   = PL_VAR_FLOAT,
        .method     = SH_LUT_TETRAHEDRAL,
        .width      = lut->size[0],
        .height     = lut->size[1],
        .depth      = lut->size[2],
        .comps      = 4, // for better texel alignment
        .signature  = lut->signature,
        .fill       = fill_lut,
        .priv       = (void *) lut,
    ));

    if (!fun) {
        SH_FAIL(sh, "pl_shader_custom_lut: failed generating LUT object");
        return;
    }

    GLSL("// pl_shader_custom_lut \n");

    static const struct pl_matrix3x3 zero = {0};
    if (memcmp(&lut->shaper_in, &zero, sizeof(zero)) != 0) {
        GLSL("color.rgb = %s * color.rgb; \n", sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_mat3("shaper_in"),
            .data = PL_TRANSPOSE_3X3(lut->shaper_in.m),
        }));
    }

    switch (dims) {
    case 1:
        sh_describe(sh, "custom 1DLUT");
        GLSL("color.rgb = vec3(%s(color.r).r, %s(color.g).g, %s(color.b).b); \n",
             fun, fun, fun);
        break;
    case 3:
        sh_describe(sh, "custom 3DLUT");
        GLSL("color.rgb = %s(color.rgb).rgb; \n", fun);
        break;
    }

    if (memcmp(&lut->shaper_out, &zero, sizeof(zero)) != 0) {
        GLSL("color.rgb = %s * color.rgb; \n", sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_mat3("shaper_out"),
            .data = PL_TRANSPOSE_3X3(lut->shaper_out.m),
        }));
    }
}

// Defines a LUT position helper macro. This translates from an absolute texel
// scale (0.0 - 1.0) to the texture coordinate scale for the corresponding
// sample in a texture of dimension `lut_size`.
static ident_t sh_lut_pos(pl_shader sh, int lut_size)
{
    ident_t name = sh_fresh(sh, "LUT_POS");
    GLSLH("#define %s(x) mix(%s, %s, (x)) \n",
          name, SH_FLOAT(0.5 / lut_size), SH_FLOAT(1.0 - 0.5 / lut_size));
    return name;
}

struct sh_lut_obj {
    enum sh_lut_type type;
    enum sh_lut_method method;
    enum pl_var_type vartype;
    pl_fmt fmt;
    int width, height, depth, comps;
    uint64_t signature;
    bool error; // reset if params change

    // weights, depending on the lut type
    pl_tex tex;
    pl_str str;
    void *data;
};

static void sh_lut_uninit(pl_gpu gpu, void *ptr)
{
    struct sh_lut_obj *lut = ptr;
    pl_tex_destroy(gpu, &lut->tex);
    pl_free(lut->str.buf);
    pl_free(lut->data);

    *lut = (struct sh_lut_obj) {0};
}

// Maximum number of floats to embed as a literal array (when using SH_LUT_AUTO)
#define SH_LUT_MAX_LITERAL_SOFT 64
#define SH_LUT_MAX_LITERAL_HARD 256

ident_t sh_lut(pl_shader sh, const struct sh_lut_params *params)
{
    pl_gpu gpu = SH_GPU(sh);
    void *tmp = NULL;

    const enum pl_var_type vartype = params->var_type;
    pl_assert(vartype != PL_VAR_INVALID);
    pl_assert(params->method == SH_LUT_NONE || vartype == PL_VAR_FLOAT);
    pl_assert(params->width > 0 && params->height >= 0 && params->depth >= 0);
    pl_assert(params->comps > 0);

    int sizes[] = { params->width, params->height, params->depth };
    int size = params->width * PL_DEF(params->height, 1) * PL_DEF(params->depth, 1);
    int dims = params->depth ? 3 : params->height ? 2 : 1;
    enum sh_lut_method method = params->method;
    if (method == SH_LUT_TETRAHEDRAL && dims != 3)
        method = SH_LUT_LINEAR;

    int texdim = 0;
    uint32_t max_tex_dim[] = {
        gpu ? gpu->limits.max_tex_1d_dim : 0,
        gpu ? gpu->limits.max_tex_2d_dim : 0,
        (gpu && gpu->glsl.version > 100) ? gpu->limits.max_tex_3d_dim : 0,
    };

    struct sh_lut_obj *lut = SH_OBJ(sh, params->object, PL_SHADER_OBJ_LUT,
                                    struct sh_lut_obj, sh_lut_uninit);

    if (!lut)
        return NULL;

    bool update = params->update || lut->signature != params->signature ||
                  vartype != lut->vartype || params->fmt != lut->fmt ||
                  params->width != lut->width || params->height != lut->height ||
                  params->depth != lut->depth || params->comps != lut->comps;

    if (lut->error && !update)
        return NULL; // suppress error spam until something changes

    // Try picking the right number of dimensions for the texture LUT. This
    // allows e.g. falling back to 2D textures if 1D textures are unsupported.
    for (int d = dims; d <= PL_ARRAY_SIZE(max_tex_dim); d++) {
        // For a given dimension to be compatible, all coordinates need to be
        // within the maximum texture size for that dimension
        for (int i = 0; i < d; i++) {
            if (sizes[i] > max_tex_dim[d - 1])
                goto next_dim;
        }

        // All dimensions are compatible, so pick this texture dimension
        texdim = d;
        break;

next_dim: ; // `continue` out of the inner loop
    }

    static const enum pl_fmt_type fmt_type[PL_VAR_TYPE_COUNT] = {
        [PL_VAR_SINT]   = PL_FMT_SINT,
        [PL_VAR_UINT]   = PL_FMT_UINT,
        [PL_VAR_FLOAT]  = PL_FMT_FLOAT,
    };

    enum pl_fmt_caps texcaps = PL_FMT_CAP_SAMPLEABLE;
    if (method == SH_LUT_LINEAR)
        texcaps |= PL_FMT_CAP_LINEAR;

    pl_fmt texfmt = params->fmt;
    if (texfmt) {
        bool ok;
        switch (texfmt->type) {
        case PL_FMT_SINT: ok = vartype == PL_VAR_SINT; break;
        case PL_FMT_UINT: ok = vartype == PL_VAR_UINT; break;
        default:          ok = vartype == PL_VAR_FLOAT; break;
        }

        if (!ok) {
            PL_ERR(sh, "Specified texture format '%s' does not match LUT "
                   "data type!", texfmt->name);
            goto error;
        }

        if (~texfmt->caps & texcaps) {
            PL_ERR(sh, "Specified texture format '%s' does not match "
                   "required capabilities 0x%x!\n", texfmt->name, texcaps);
            goto error;
        }
    }

    if (texdim && !texfmt) {
        texfmt = pl_find_fmt(gpu, fmt_type[vartype], params->comps,
                             vartype == PL_VAR_FLOAT ? 16 : 32,
                             pl_var_type_size(vartype) * 8,
                             texcaps);
    }

    enum sh_lut_type type = params->lut_type;

    // Integer texture sampling requires GLSL >= 130 (for texelFetch)
    bool can_fetch = sh_glsl(sh).version >= 130;
    bool needs_integer = method != SH_LUT_LINEAR;
    if (needs_integer && !can_fetch) {
        if (method == SH_LUT_TETRAHEDRAL) {
            // Fall back to normal trilinear interpolation
            if (texfmt && (texfmt->caps & PL_FMT_CAP_LINEAR)) {
                method = SH_LUT_LINEAR;
            } else {
                PL_ERR(sh, "Cannot compute tetrahedral interpolation on GLSL "
                       "versions below 130, and no trilinear interpolation "
                       "available!");
                goto error;
            }
        } else {
            // Integer sampling, disable textures
            texfmt = NULL;
        }
    }

    // The linear sampling code currently only supports 1D linear interpolation
    if (method == SH_LUT_LINEAR && dims > 1) {
        if (texfmt) {
            type = SH_LUT_TEXTURE;
        } else {
            PL_ERR(sh, "Can't emulate linear LUTs for 2D/3D LUTs and no "
                  "texture support available!");
            goto error;
        }
    }

    bool can_uniform = gpu && gpu->limits.max_variable_comps >= size * params->comps;
    bool can_literal = sh_glsl(sh).version > 110; // needed for literal arrays
    can_literal &= size <= SH_LUT_MAX_LITERAL_HARD && !params->dynamic;

    // Deselect unsupported methods
    if (type == SH_LUT_UNIFORM && !can_uniform)
        type = SH_LUT_AUTO;
    if (type == SH_LUT_LITERAL && !can_literal)
        type = SH_LUT_AUTO;
    if (type == SH_LUT_TEXTURE && !texfmt)
        type = SH_LUT_AUTO;

    // Sorted by priority
    if (!type && can_literal && !method && size <= SH_LUT_MAX_LITERAL_SOFT)
        type = SH_LUT_LITERAL;
    if (!type && texfmt)
        type = SH_LUT_TEXTURE;
    if (!type && can_uniform)
        type = SH_LUT_UNIFORM;
    if (!type && can_literal)
        type = SH_LUT_LITERAL;

    if (!type) {
        PL_ERR(sh, "Can't generate LUT: no compatible methods!");
        goto error;
    }

    // Reinitialize the existing LUT if needed
    update |= type != lut->type;
    update |= method != lut->method;

    if (update) {
        PL_DEBUG(sh, "LUT cache invalidated, regenerating..");
        size_t buf_size = size * params->comps * pl_var_type_size(vartype);
        tmp = pl_zalloc(NULL, buf_size);
        params->fill(tmp, params);

        switch (type) {
        case SH_LUT_TEXTURE: {
            if (!texdim) {
                PL_ERR(sh, "Texture LUT exceeds texture dimensions!");
                goto error;
            }

            if (!texfmt) {
                PL_ERR(sh, "Found no compatible texture format for LUT!");
                goto error;
            }

            struct pl_tex_params tex_params = {
                .w              = params->width,
                .h              = PL_DEF(params->height, texdim >= 2 ? 1 : 0),
                .d              = PL_DEF(params->depth,  texdim >= 3 ? 1 : 0),
                .format         = texfmt,
                .sampleable     = true,
                .host_writable  = params->dynamic,
                .initial_data   = params->dynamic ? NULL : tmp,
                .debug_tag      = PL_DEBUG_TAG,
            };

            bool ok;
            if (params->dynamic) {
                ok = pl_tex_recreate(gpu, &lut->tex, &tex_params);
                if (ok) {
                    ok = pl_tex_upload(gpu, pl_tex_transfer_params(
                        .tex = lut->tex,
                        .ptr = tmp,
                    ));
                }
            } else {
                // Can't use pl_tex_recreate because of `initial_data`
                pl_tex_destroy(gpu, &lut->tex);
                lut->tex = pl_tex_create(gpu, &tex_params);
                ok = lut->tex;
            }

            if (!ok) {
                PL_ERR(sh, "Failed creating LUT texture!");
                goto error;
            }
            break;
        }

        case SH_LUT_UNIFORM:
            pl_free(lut->data);
            lut->data = tmp; // re-use `tmp`
            tmp = NULL;
            break;

        case SH_LUT_LITERAL: {
            lut->str.len = 0;
            static const char prefix[PL_VAR_TYPE_COUNT] = {
                [PL_VAR_SINT]   = 'i',
                [PL_VAR_UINT]   = 'u',
                [PL_VAR_FLOAT]  = ' ',
            };

            for (int i = 0; i < size * params->comps; i += params->comps) {
                if (i > 0)
                    pl_str_append_asprintf_c(lut, &lut->str, ",");
                if (params->comps > 1) {
                    pl_str_append_asprintf_c(lut, &lut->str, "%cvec%d(",
                                             prefix[vartype], params->comps);
                }
                for (int c = 0; c < params->comps; c++) {
                    switch (vartype) {
                    case PL_VAR_FLOAT:
                        pl_str_append_asprintf_c(lut, &lut->str, "%s%f",
                                                 c > 0 ? "," : "",
                                                 ((float *) tmp)[i+c]);
                        break;
                    case PL_VAR_UINT:
                        pl_str_append_asprintf_c(lut, &lut->str, "%s%u",
                                                 c > 0 ? "," : "",
                                                 ((unsigned int *) tmp)[i+c]);
                        break;
                    case PL_VAR_SINT:
                        pl_str_append_asprintf_c(lut, &lut->str, "%s%d",
                                                 c > 0 ? "," : "",
                                                 ((int *) tmp)[i+c]);
                        break;
                    case PL_VAR_INVALID:
                    case PL_VAR_TYPE_COUNT:
                        pl_unreachable();
                    }
                }
                if (params->comps > 1)
                    pl_str_append_asprintf_c(lut, &lut->str, ")");
            }
            break;
        }

        case SH_LUT_AUTO:
            pl_unreachable();
        }

        lut->type = type;
        lut->method = method;
        lut->vartype = vartype;
        lut->fmt = params->fmt;
        lut->width = params->width;
        lut->height = params->height;
        lut->depth = params->depth;
        lut->comps = params->comps;
        lut->signature = params->signature;
    }

    // Done updating, generate the GLSL
    ident_t name = sh_fresh(sh, "lut");
    ident_t arr_name = NULL;

    static const char * const swizzles[] = {"x", "xy", "xyz", "xyzw"};
    static const char * const vartypes[PL_VAR_TYPE_COUNT][4] = {
        [PL_VAR_SINT] = { "int", "ivec2", "ivec3", "ivec4" },
        [PL_VAR_UINT] = { "uint", "uvec2", "uvec3", "uvec4" },
        [PL_VAR_FLOAT] = { "float", "vec2", "vec3", "vec4" },
    };

    switch (type) {
    case SH_LUT_TEXTURE: {
        assert(texdim);
        ident_t tex = sh_desc(sh, (struct pl_shader_desc) {
            .desc = {
                .name = "weights",
                .type = PL_DESC_SAMPLED_TEX,
            },
            .binding = {
                .object = lut->tex,
                .sample_mode = method == SH_LUT_LINEAR
                                    ? PL_TEX_SAMPLE_LINEAR
                                    : PL_TEX_SAMPLE_NEAREST,
            }
        });

        if (method == SH_LUT_LINEAR) {
            ident_t pos_macros[PL_ARRAY_SIZE(sizes)] = {0};
            for (int i = 0; i < dims; i++)
                pos_macros[i] = sh_lut_pos(sh, sizes[i]);

            GLSLH("#define %s(pos) (%s(%s, %s(\\\n",
                  name, sh_tex_fn(sh, lut->tex->params),
                  tex, vartypes[PL_VAR_FLOAT][texdim - 1]);

            for (int i = 0; i < texdim; i++) {
                char sep = i == 0 ? ' ' : ',';
                if (pos_macros[i]) {
                    if (dims > 1) {
                        GLSLH("   %c%s(%s(pos).%c)\\\n", sep, pos_macros[i],
                              vartypes[PL_VAR_FLOAT][dims - 1], "xyzw"[i]);
                    } else {
                        GLSLH("   %c%s(float(pos))\\\n", sep, pos_macros[i]);
                    }
                } else {
                    GLSLH("   %c%f\\\n", sep, 0.5);
                }
            }
            GLSLH("  )).%s)\n", swizzles[params->comps - 1]);
        } else {
            GLSLH("#define %s(pos) (texelFetch(%s, %s(pos",
                  name, tex, vartypes[PL_VAR_SINT][texdim - 1]);

            // Fill up extra components of the index
            for (int i = dims; i < texdim; i++)
                GLSLH(", 0");

            GLSLH("), 0).%s)\n", swizzles[params->comps - 1]);
        }
        break;
    }

    case SH_LUT_UNIFORM:
        arr_name = sh_var(sh, (struct pl_shader_var) {
            .var = {
                .name = "weights",
                .type = vartype,
                .dim_v = params->comps,
                .dim_m = 1,
                .dim_a = size,
            },
            .data = lut->data,
        });
        break;

    case SH_LUT_LITERAL:
        arr_name = sh_fresh(sh, "weights");
        GLSLH("const %s %s[%d] = %s[](\n  ",
              vartypes[vartype][params->comps - 1], arr_name, size,
              vartypes[vartype][params->comps - 1]);
        pl_str_append(sh, &sh->buffers[SH_BUF_HEADER], lut->str);
        GLSLH(");\n");
        break;

    case SH_LUT_AUTO:
        pl_unreachable();
    }

    if (arr_name) {
        GLSLH("#define %s(pos) (%s[int((pos)%s)\\\n",
              name, arr_name, dims > 1 ? "[0]" : "");
        int shift = params->width;
        for (int i = 1; i < dims; i++) {
            GLSLH("    + %d * int((pos)[%d])\\\n", shift, i);
            shift *= sizes[i];
        }
        GLSLH("  ])\n");

        if (method == SH_LUT_LINEAR) {
            pl_assert(dims == 1);
            pl_assert(vartype == PL_VAR_FLOAT);
            ident_t arr_lut = name;
            name = sh_fresh(sh, "lut_lin");
            GLSLH("%s %s(float fpos) {                              \n"
                  "    fpos = clamp(fpos, 0.0, 1.0) * %d.0;         \n"
                  "    float fbase = floor(fpos);                   \n"
                  "    float fceil = ceil(fpos);                    \n"
                  "    float fcoord = fpos - fbase;                 \n"
                  "    return mix(%s(fbase), %s(fceil), fcoord);    \n"
                  "}                                                \n",
                  vartypes[PL_VAR_FLOAT][params->comps - 1], name,
                  size - 1,
                  arr_lut, arr_lut);
        }
    }

    if (method == SH_LUT_TETRAHEDRAL) {
        ident_t int_lut = name;
        name = sh_fresh(sh, "lut_barycentric");
        GLSLH("%s %s(vec3 pos) {                                            \n"
              // Compute bounding vertices and fractinoal part
              "    pos = clamp(pos, 0.0, 1.0) * vec3(%d.0, %d.0, %d.0);     \n"
              "    vec3 base = floor(pos);                                  \n"
              "    vec3 fpart = pos - base;                                 \n"
              // v0 and v3 are always 'black' and 'white', respectively
              // v1 and v2 are the closest RGB and CMY vertices, respectively
              "    ivec3 v0 = ivec3(base), v3 = ivec3(ceil(pos));           \n"
              "    ivec3 v1 = v0, v2 = v3;                                  \n"
              // Table of boolean checks to simplify following math
              "    bvec3 c = greaterThanEqual(fpart.xyz, fpart.yzx);        \n"
              "    bool c_xy = c.x, c_yx = !c.x,                            \n"
              "       c_yz = c.y, c_zy = !c.y,                              \n"
              "       c_zx = c.z, c_xz = !c.z;                              \n"
              "    vec3 s = fpart.xyz;                                      \n"
              "    bool cond;                                               \n",
              vartypes[PL_VAR_FLOAT][params->comps - 1], name,
              sizes[0] - 1, sizes[1] - 1, sizes[2] - 1);

        // Subdivision of the cube into six congruent tetrahedras
        //
        // For each tetrahedron, test if the point is inside, and if so, update
        // the edge vertices. We test all six, even though only one case will
        // ever be true, because this avoids branches.
        static const char *indices[] = { "xyz", "xzy", "zxy", "zyx", "yzx", "yxz"};
        for (int i = 0; i < PL_ARRAY_SIZE(indices); i++) {
            const char x = indices[i][0], y = indices[i][1], z = indices[i][2];
            GLSLH("cond = c_%c%c && c_%c%c;          \n"
                  "s = cond ? fpart.%c%c%c : s;      \n"
                  "v1.%c = cond ? v3.%c : v1.%c;     \n"
                  "v2.%c = cond ? v0.%c : v2.%c;     \n",
                  x, y, y, z,
                  x, y, z,
                  x, x, x,
                  z, z, z);
        }

        // Interpolate in barycentric coordinates, with four texel fetches
        GLSLH("    return (1.0 - s.x) * %s(v0) +    \n"
              "           (s.x - s.y) * %s(v1) +    \n"
              "           (s.y - s.z) * %s(v2) +    \n"
              "           (s.z)       * %s(v3);     \n"
              "}                                    \n",
              int_lut, int_lut, int_lut, int_lut);
    }

    lut->error = false;
    pl_free(tmp);
    pl_assert(name);
    return name;

error:
    lut->error = true;
    pl_free(tmp);
    return NULL;
}
