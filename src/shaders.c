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

#include <stdio.h>
#include <math.h>

#include "bstr/bstr.h"
#include "common.h"
#include "context.h"
#include "shaders.h"

struct pl_shader *pl_shader_alloc(struct pl_context *ctx, const struct ra *ra,
                                  uint8_t ident, uint8_t index)
{
    pl_assert(ctx);
    struct pl_shader *sh = talloc_ptrtype(ctx, sh);
    *sh = (struct pl_shader) {
        .ctx = ctx,
        .ra = ra,
        .mutable = true,
        .tmp = talloc_ref_new(NULL),
        .ident = ident,
        .index = index,
    };

    return sh;
}

void pl_shader_free(struct pl_shader **psh)
{
    struct pl_shader *sh = *psh;
    if (!sh)
        return;

    talloc_ref_deref(&sh->tmp);
    TA_FREEP(psh);
}

void pl_shader_reset(struct pl_shader *sh, uint8_t ident, uint8_t index)
{
    struct pl_shader new = {
        .ctx = sh->ctx,
        .ra  = sh->ra,
        .tmp = talloc_ref_new(NULL),
        .mutable = true,
        .ident = ident,
        .index = index,

        // Preserve array allocations
        .res = {
            .variables      = sh->res.variables,
            .descriptors    = sh->res.descriptors,
            .vertex_attribs = sh->res.vertex_attribs,
        },
    };

    // Preserve buffer allocations
    for (int i = 0; i < PL_ARRAY_SIZE(new.buffers); i++)
        new.buffers[i] = (struct bstr) { .start = sh->buffers[i].start };

    ta_ref_deref(&sh->tmp);
    *sh = new;
}

bool sh_try_compute(struct pl_shader *sh, int bw, int bh, bool flex, size_t mem)
{
    pl_assert(bw && bh);
    int *sh_bw = &sh->res.compute_group_size[0];
    int *sh_bh = &sh->res.compute_group_size[1];

    if (!sh->ra || !(sh->ra->caps & RA_CAP_COMPUTE)) {
        PL_TRACE(sh, "Disabling compute shader due to missing RA_CAP_COMPUTE");
        return false;
    }

    if (sh->res.compute_shmem + mem > sh->ra->limits.max_shmem_size) {
        PL_TRACE(sh, "Disabling compute shader due to insufficient shmem");
        return false;
    }

    sh->res.compute_shmem += mem;

    // If the current shader is either not a compute shader, or we have no
    // choice but to override the metadata, always do so
    if (!sh->is_compute || (sh->flexible_work_groups && !flex)) {
        *sh_bw = bw;
        *sh_bh = bh;
        sh->is_compute = true;
        return true;
    }

    // If both shaders are flexible, pick the larger of the two
    if (sh->flexible_work_groups && flex) {
        *sh_bw = PL_MAX(*sh_bw, bw);
        *sh_bh = PL_MAX(*sh_bh, bh);
        return true;
    }

    // If the other shader is rigid but this is flexible, change nothing
    if (flex)
        return true;

    // If neither are flexible, make sure the parameters match
    pl_assert(!flex && !sh->flexible_work_groups);
    if (bw != *sh_bw || bh != *sh_bh) {
        PL_TRACE(sh, "Disabling compute shader due to incompatible group "
                 "sizes %dx%d and %dx%d", *sh_bw, *sh_bh, bw, bh);
        sh->res.compute_shmem -= mem;
        return false;
    }

    return true;
}

bool pl_shader_is_compute(const struct pl_shader *sh)
{
    return sh->is_compute;
}

bool pl_shader_output_size(const struct pl_shader *sh, int *w, int *h)
{
    if (!sh->output_w || !sh->output_h)
        return false;

    *w = sh->output_w;
    *h = sh->output_h;
    return true;
}

uint64_t pl_shader_signature(const struct pl_shader *sh)
{
    uint64_t res = 0;
    for (int i = 0; i < PL_ARRAY_SIZE(sh->buffers); i++)
        res ^= bstr_hash64(sh->buffers[i]);

    // FIXME: also hash in the configuration of the descriptors/variables

    return res;
}

ident_t sh_fresh(struct pl_shader *sh, const char *name)
{
    return talloc_asprintf(sh->tmp, "_%s_%d_%u", PL_DEF(name, "var"),
                           sh->fresh++, sh->ident);
}

ident_t sh_var(struct pl_shader *sh, struct pl_shader_var sv)
{
    sv.var.name = sh_fresh(sh, sv.var.name);
    sv.data = talloc_memdup(sh->tmp, sv.data, ra_var_host_layout(0, &sv.var).size);
    TARRAY_APPEND(sh, sh->res.variables, sh->res.num_variables, sv);
    return (ident_t) sv.var.name;
}

ident_t sh_desc(struct pl_shader *sh, struct pl_shader_desc sd)
{
    sd.desc.name = sh_fresh(sh, sd.desc.name);
    TARRAY_APPEND(sh, sh->res.descriptors, sh->res.num_descriptors, sd);
    return (ident_t) sd.desc.name;
}

ident_t sh_attr_vec2(struct pl_shader *sh, const char *name,
                     const struct pl_rect2df *rc)
{
    if (!sh->ra) {
        PL_ERR(sh, "Failed adding vertex attr '%s': No RA available!", name);
        return NULL;
    }

    const struct ra_fmt *fmt = ra_find_vertex_fmt(sh->ra, RA_FMT_FLOAT, 2);
    if (!fmt) {
        PL_ERR(sh, "Failed adding vertex attr '%s': no vertex fmt!", name);
        return NULL;
    }

    float vals[4][2] = {
        { rc->x0, rc->y0 },
        { rc->x1, rc->y0 },
        { rc->x0, rc->y1 },
        { rc->x1, rc->y1 },
    };

    float *data = talloc_memdup(sh->tmp, &vals[0][0], sizeof(vals));
    struct pl_shader_va va = {
        .attr = {
            .name     = sh_fresh(sh, name),
            .fmt      = ra_find_vertex_fmt(sh->ra, RA_FMT_FLOAT, 2),
        },
        .data = { &data[0], &data[2], &data[4], &data[6] },
    };

    TARRAY_APPEND(sh, sh->res.vertex_attribs, sh->res.num_vertex_attribs, va);
    return (ident_t) va.attr.name;
}

ident_t sh_bind(struct pl_shader *sh, const struct ra_tex *tex,
                const char *name, const struct pl_rect2df *rect,
                ident_t *out_pos, ident_t *out_size, ident_t *out_pt)
{
    if (!sh->ra) {
        PL_ERR(sh, "Failed binding texture '%s': No RA available!", name);
        return NULL;
    }

    if (ra_tex_params_dimension(tex->params) != 2 || !tex->params.sampleable) {
        PL_ERR(sh, "Failed binding texture '%s': incompatible params!", name);
        return NULL;
    }

    ident_t itex = sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name = name,
            .type = RA_DESC_SAMPLED_TEX,
        },
        .object = tex,
    });

    if (out_pos) {
        struct pl_rect2df full = {
            .x1 = tex->params.w,
            .y1 = tex->params.h,
        };

        rect = PL_DEF(rect, &full);
        *out_pos = sh_attr_vec2(sh, "tex_coord", &(struct pl_rect2df) {
            .x0 = rect->x0 / tex->params.w, .y0 = rect->y0 / tex->params.h,
            .x1 = rect->x1 / tex->params.w, .y1 = rect->y1 / tex->params.h,
        });
    }

    if (out_size) {
        *out_size = sh_var(sh, (struct pl_shader_var) {
            .var  = ra_var_vec2("tex_size"),
            .data = &(float[2]) {tex->params.w, tex->params.h},
        });
    }

    if (out_pt) {
        *out_pt = sh_var(sh, (struct pl_shader_var) {
            .var  = ra_var_vec2("tex_pt"),
            .data = &(float[2]) {1.0 / tex->params.w, 1.0 / tex->params.h},
        });
    }

    return itex;
}

void pl_shader_append(struct pl_shader *sh, enum pl_shader_buf buf,
                      const char *fmt, ...)
{
    pl_assert(buf >= 0 && buf < SH_BUF_COUNT);

    va_list ap;
    va_start(ap, fmt);
    bstr_xappend_vasprintf(sh, &sh->buffers[buf], fmt, ap);
    va_end(ap);
}

static const char *outsigs[] = {
    [PL_SHADER_SIG_NONE]  = "void",
    [PL_SHADER_SIG_COLOR] = "vec4",
};

static const char *insigs[] = {
    [PL_SHADER_SIG_NONE]  = "",
    [PL_SHADER_SIG_COLOR] = "vec4 color",
};

static const char *retvals[] = {
    [PL_SHADER_SIG_NONE]  = "",
    [PL_SHADER_SIG_COLOR] = "return color;",
};

ident_t sh_subpass(struct pl_shader *sh, const struct pl_shader *sub)
{
    pl_assert(sh->mutable);

    // Check for shader compatibility
    int res_w = PL_DEF(sh->output_w, sub->output_w),
        res_h = PL_DEF(sh->output_h, sub->output_h);

    if ((sub->output_w && res_w != sub->output_w) ||
        (sub->output_h && res_h != sub->output_h))
    {
        PL_ERR(sh, "Failed merging shaders: incompatible sizes: %dx%d and %dx%d",
               sh->output_w, sh->output_h, sub->output_w, sub->output_h);
        return NULL;
    }

    if (sub->is_compute) {
        int subw = sub->res.compute_group_size[0],
            subh = sub->res.compute_group_size[1];
        bool flex = sub->flexible_work_groups;

        if (!sh_try_compute(sh, subw, subh, flex, sub->res.compute_shmem)) {
            PL_ERR(sh, "Failed merging shaders: incompatible block sizes or "
                   "exceeded shared memory resource capabilities");
            return NULL;
        }
    }

    sh->output_w = res_w;
    sh->output_h = res_h;

    // Append the prelude and header
    bstr_xappend(sh, &sh->buffers[SH_BUF_PRELUDE], sub->buffers[SH_BUF_PRELUDE]);
    bstr_xappend(sh, &sh->buffers[SH_BUF_HEADER],  sub->buffers[SH_BUF_HEADER]);

    // Append the body as a new header function
    ident_t name = sh_fresh(sh, "sub");
    GLSLH("%s %s(%s) {\n", outsigs[sub->res.output], name, insigs[sub->res.input]);
    bstr_xappend(sh, &sh->buffers[SH_BUF_HEADER], sub->buffers[SH_BUF_BODY]);
    GLSLH("%s }\n", retvals[sub->res.output]);

    // Copy over all of the descriptors etc.
    talloc_ref_attach(sh->tmp, sub->tmp);
#define COPY(f) TARRAY_CONCAT(sh, sh->res.f, sh->res.num_##f, \
                              sub->res.f, sub->res.num_##f)
    COPY(variables);
    COPY(descriptors);
    COPY(vertex_attribs);
#undef COPY

    return name;
}

// Finish the current shader body and return its function name
static ident_t sh_split(struct pl_shader *sh)
{
    pl_assert(sh->mutable);

    // Concatenate the body onto the head as a new function
    ident_t name = sh_fresh(sh, "main");
    GLSLH("%s %s(%s) {\n", outsigs[sh->res.output], name, insigs[sh->res.input]);

    if (sh->buffers[SH_BUF_BODY].len) {
        bstr_xappend(sh, &sh->buffers[SH_BUF_HEADER], sh->buffers[SH_BUF_BODY]);
        sh->buffers[SH_BUF_BODY].len = 0;
        sh->buffers[SH_BUF_BODY].start[0] = '\0'; // for sanity / efficiency
    }

    GLSLH("%s }\n", retvals[sh->res.output]);
    return name;
}

const struct pl_shader_res *pl_shader_finalize(struct pl_shader *sh)
{
    if (!sh->mutable) {
        PL_WARN(sh, "Attempted to finalize a shader twice?");
        return &sh->res;
    }

    // Split the shader. This finalizes the body and adds it to the header
    sh->res.name = sh_split(sh);

    // Concatenate the header onto the prelude to form the final output
    struct bstr *glsl = &sh->buffers[SH_BUF_PRELUDE];
    bstr_xappend(sh, glsl, sh->buffers[SH_BUF_HEADER]);

    // Update the result pointer and return
    sh->res.glsl = glsl->start;
    sh->mutable = false;
    return &sh->res;
}

bool sh_require(struct pl_shader *sh, enum pl_shader_sig insig, int w, int h)
{
    if (!sh->mutable) {
        PL_ERR(sh, "Attempted to modify an immutable shader!");
        return false;
    }

    if ((w && sh->output_w && sh->output_w != w) ||
        (h && sh->output_h && sh->output_h != h))
    {
        PL_ERR(sh, "Illegal sequence of shader operations: Incompatible "
               "output size requirements %dx%d and %dx%d",
               sh->output_w, sh->output_h, w, h);
        return false;
    }

    static const char *names[] = {
        [PL_SHADER_SIG_NONE]  = "PL_SHADER_SIG_NONE",
        [PL_SHADER_SIG_COLOR] = "PL_SHADER_SIG_COLOR",
    };

    // If we require an input, but there is none available - just get it from
    // the user by turning it into an explicit input signature.
    if (!sh->res.output && insig) {
        pl_assert(!sh->res.input);
        sh->res.input = insig;
    } else if (sh->res.output != insig) {
        PL_ERR(sh, "Illegal sequence of shader operations! Current output "
               "signature is '%s', but called operation expects '%s'!",
               names[sh->res.output], names[insig]);
        return false;
    }

    // All of our shaders end up returning a vec4 color
    sh->res.output = PL_SHADER_SIG_COLOR;
    sh->output_w = PL_DEF(sh->output_w, w);
    sh->output_h = PL_DEF(sh->output_w, h);
    return true;
}

void pl_shader_obj_destroy(struct pl_shader_obj **ptr)
{
    struct pl_shader_obj *obj = *ptr;
    if (!obj)
        return;

    if (obj->uninit)
        obj->uninit(obj->ra, obj->priv);

    *ptr = NULL;
    talloc_free(obj);
}

void *sh_require_obj(struct pl_shader *sh, struct pl_shader_obj **ptr,
                     enum pl_shader_obj_type type, size_t priv_size,
                     void (*uninit)(const struct ra *ra, void *priv))
{
    if (!ptr)
        return NULL;

    struct pl_shader_obj *obj = *ptr;
    if (obj && obj->ra != sh->ra) {
        PL_ERR(sh, "Passed pl_shader_obj belongs to different RA!");
        return NULL;
    }

    if (obj && obj->type != type) {
        PL_ERR(sh, "Passed pl_shader_obj of wrong type! Shader objects must "
               "always be used with the same type of shader.");
        return NULL;
    }

    if (!obj) {
        obj = talloc_zero(NULL, struct pl_shader_obj);
        obj->ra = sh->ra;
        obj->type = type;
        obj->priv = talloc_zero_size(obj, priv_size);
        obj->uninit = uninit;
    }

    *ptr = obj;
    return obj->priv;
}

ident_t sh_prng(struct pl_shader *sh, bool temporal, ident_t *p_state)
{
    // Initialize the PRNG. This is friendly for wide usage and returns in
    // a very pleasant-looking distribution across frames even if the difference
    // between input coordinates is very small. This is based on BlumBlumShub,
    // with some modifications for speed / aesthetics.
    // cf. https://briansharpe.wordpress.com/2011/10/01/gpu-texture-free-noise/
    ident_t randfun = sh_fresh(sh, "random"), permute = sh_fresh(sh, "permute");
    GLSLH("float %s(float x) {                      \n"
          "    x = (34.0 * x + 1.0) * x;            \n"
          "    return fract(x * 1.0/289.0) * 289.0; \n" // (almost) mod 289
          "}                                        \n"
          "float %s(inout float state) {            \n"
          "    state = %s(state);                   \n"
          "    return fract(state * 1.0/41.0);      \n"
          "}\n", permute, randfun, permute);

    const char *seed = "0.0";
    if (temporal) {
        float seedval = modff(M_PI * sh->index, &(float){0});
        seed = sh_var(sh, (struct pl_shader_var) {
            .var  = ra_var_float("seed"),
            .data = &seedval,
            .dynamic = true,
        });
    }

    ident_t state = sh_fresh(sh, "prng");
    GLSL("vec3 %s_m = vec3(gl_FragCoord.xy, %s) + vec3(1.0); \n"
         "float %s = %s(%s(%s(%s_m.x) + %s_m.y) + %s_m.z);   \n",
         state, seed,
         state, permute, permute, permute, state, state, state);

    if (p_state)
        *p_state = state;

    ident_t res = sh_fresh(sh, "RAND");
    GLSLH("#define %s (%s(%s))\n", res, randfun, state);
    return res;
}

// Defines a LUT position helper macro. This translates from an absolute texel
// scale (0.0 - 1.0) to the texture coordinate scale for the corresponding
// sample in a texture of dimension `lut_size`.
static ident_t sh_lut_pos(struct pl_shader *sh, int lut_size)
{
    ident_t name = sh_fresh(sh, "LUT_POS");
    GLSLH("#define %s(x) mix(%f, %f, (x)) \n",
          name, 0.5 / lut_size, 1.0 - 0.5 / lut_size);
    return name;
}

struct sh_lut_obj {
    enum sh_lut_method method;
    int width, height, depth;
    union {
        const struct ra_tex *tex;
        struct bstr str;
        float *data;
    } weights;
};

static void sh_lut_uninit(const struct ra *ra, void *ptr)
{
    struct sh_lut_obj *lut = ptr;
    switch (lut->method) {
    case SH_LUT_TEXTURE:
    case SH_LUT_LINEAR:
        ra_tex_destroy(ra, &lut->weights.tex);
        break;
    case SH_LUT_UNIFORM:
        talloc_free(lut->weights.data);
        break;
    case SH_LUT_LITERAL:
        talloc_free(lut->weights.str.start);
        break;
    default: break;
    }

    *lut = (struct sh_lut_obj) {0};
}

// Maximum number of floats to embed as a literal array (when using SH_LUT_AUTO)
#define SH_LUT_MAX_LITERAL 256

ident_t sh_lut(struct pl_shader *sh, struct pl_shader_obj **obj,
               enum sh_lut_method method, int width, int height, int depth,
               bool update, void *priv,
               void (*fill)(void *priv, float *data, int w, int h, int d))
{
    const struct ra *ra = sh->ra;
    float *tmp = NULL;
    ident_t ret = NULL;

    pl_assert(width > 0 && height >= 0 && depth >= 0);
    int sizes[] = { width, height, depth };
    int size = width * PL_DEF(height, 1) * PL_DEF(depth, 1);
    int dims = depth ? 3 : height ? 2 : 1;

    int texdim = 0;
    int max_tex_dim[] = {
        ra ? ra->limits.max_tex_1d_dim : 0,
        ra ? ra->limits.max_tex_2d_dim : 0,
        ra ? ra->limits.max_tex_3d_dim : 0,
    };

    for (int d = dims; d <= PL_ARRAY_SIZE(max_tex_dim); d++) {
        if (size <= max_tex_dim[d - 1]) {
            texdim = d;
            break;
        }
    }

    struct sh_lut_obj *lut = SH_OBJ(sh, obj, PL_SHADER_OBJ_LUT,
                                    struct sh_lut_obj, sh_lut_uninit);

    if (!lut) {
        PL_ERR(sh, "Failed initializing LUT object!");
        goto error;
    }

    if (!ra && method == SH_LUT_LINEAR) {
        PL_ERR(sh, "Linear LUTs require the use of a RA!");
        goto error;
    }

    if (!ra) {
        PL_TRACE(sh, "No RA available, falling back to literal LUT embedding");
        method = SH_LUT_LITERAL;
    }

    // Pick the best method
    if (!method && size <= SH_LUT_MAX_LITERAL)
        method = SH_LUT_LITERAL;

    if (!method && texdim)
        method = SH_LUT_TEXTURE;

    if (!method && ra && ra->caps & RA_CAP_INPUT_VARIABLES)
        method = SH_LUT_UNIFORM;

    // No other method found
    if (!method) {
        PL_TRACE(sh, "No other LUT method works, falling back to literal "
                 "embedding.. this is most likely a slow path!");
        method = SH_LUT_LITERAL;
    }

    // Forcibly reinitialize the existing LUT if needed
    if (method != lut->method || width != lut->width || height != lut->height
        || depth != lut->depth)
    {
        PL_DEBUG(sh, "LUT method or size changed, reinitializing..");
        sh_lut_uninit(ra, lut);
        update = true;
    }

    if (update) {
        tmp = talloc_zero_size(NULL, size * sizeof(float));
        fill(priv, tmp, width, height, depth);

        switch (method) {
        case SH_LUT_TEXTURE:
        case SH_LUT_LINEAR: {
            if (!texdim) {
                PL_ERR(sh, "Texture LUT exceeds texture dimensions!");
                goto error;
            }

            enum ra_fmt_caps caps = RA_FMT_CAP_SAMPLEABLE;
            enum ra_tex_sample_mode mode = RA_TEX_SAMPLE_NEAREST;

            if (method == SH_LUT_LINEAR) {
                caps |= RA_FMT_CAP_LINEAR;
                mode = RA_TEX_SAMPLE_LINEAR;
            }

            const struct ra_fmt *fmt;
            fmt = ra_find_fmt(ra, RA_FMT_FLOAT, 1, 16, 32, caps);
            if (!fmt) {
                PL_ERR(sh, "Found no compatible texture format for LUT!");
                goto error;
            }

            pl_assert(!lut->weights.tex);
            lut->weights.tex = ra_tex_create(ra, &(struct ra_tex_params) {
                .w              = width,
                .h              = PL_DEF(height, texdim >= 2 ? 1 : 0),
                .d              = PL_DEF(depth,  texdim >= 3 ? 1 : 0),
                .format         = fmt,
                .sampleable     = true,
                .sample_mode    = mode,
                .address_mode   = RA_TEX_ADDRESS_CLAMP,
                .initial_data   = tmp,
            });

            if (!lut->weights.tex) {
                PL_ERR(sh, "Failed creating LUT texture!");
                goto error;
            }
            break;
        }

        case SH_LUT_UNIFORM:
            pl_assert(!lut->weights.data);
            lut->weights.data = tmp; // re-use `tmp`
            tmp = NULL;
            break;

        case SH_LUT_LITERAL: {
            pl_assert(!lut->weights.str.len);
            for (int i = 0; i < size; i++) {
                bstr_xappend_asprintf(lut, &lut->weights.str, "%s%f",
                                      i > 0 ? "," : "", tmp[i]);
            }
            break;
        }

        case SH_LUT_AUTO: abort();
        }

        lut->method = method;
        lut->width = width;
        lut->height = height;
        lut->depth = depth;
    }

    // Done updating, generate the GLSL
    ident_t name = sh_fresh(sh, "lut");
    ident_t arr_name = NULL;

    switch (method) {
    case SH_LUT_TEXTURE:
    case SH_LUT_LINEAR: {
        ident_t tex = sh_desc(sh, (struct pl_shader_desc) {
            .desc = {
                .name = "weights",
                .type = RA_DESC_SAMPLED_TEX,
            },
            .object = lut->weights.tex,
        });

        ident_t pos_macros[PL_ARRAY_SIZE(sizes)] = {0};
        for (int i = 0; i < dims; i++)
            pos_macros[i] = sh_lut_pos(sh, sizes[i]);

        const char *types[] = {"float", "vec2", "vec3", "vec4"};
        GLSLH("#define %s(pos) (texture(%s, %s(\\\n", name, tex, types[texdim-1]);
        for (int i = 0; i < texdim; i++) {
            char sep = i == 0 ? ' ' : ',';
            if (pos_macros[i]) {
                GLSLH("   %c%s((pos).%c)\\\n", sep, pos_macros[i], "xyzw"[i]);
            } else {
                GLSLH("   %c%f\\\n", sep, 0.5);
            }
        }
        GLSLH("  )).r)\n");
        ret = name;
        break;
    }

    case SH_LUT_UNIFORM:
        arr_name = sh_var(sh, (struct pl_shader_var) {
            .var = {
                .name = "weights",
                .type = RA_VAR_FLOAT,
                .dim_v = 1,
                .dim_m = 1,
                .dim_a = size,
            },
            .data = lut->weights.data,
        });
        break;

    case SH_LUT_LITERAL:
        arr_name = sh_fresh(sh, "weights");
        GLSLH("const float %s[%d] = float[](\n  ", arr_name, size);
        bstr_xappend(sh, &sh->buffers[SH_BUF_HEADER], lut->weights.str);
        GLSLH(");\n");
        break;

    default: abort();
    }

    if (arr_name) {
        GLSLH("#define %s(pos) (%s[int(%d * (pos).x)\\\n", name, arr_name, width);
        int shift = width;
        for (int i = 1; i < dims; i++) {
            GLSLH("    + %d * int(%d * (pos)[%d])\\\n", shift, sizes[i], i);
            shift *= sizes[i];
        }
        GLSLH("  ])\n");
        ret = name;
    }

    // fall through
error:
    talloc_free(tmp);
    return ret;
}
