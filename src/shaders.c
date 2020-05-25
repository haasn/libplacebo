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

#include "common.h"
#include "context.h"
#include "shaders.h"

struct pl_shader *pl_shader_alloc(struct pl_context *ctx,
                                  const struct pl_shader_params *params)
{
    pl_assert(ctx);
    struct pl_shader *sh = talloc_ptrtype(ctx, sh);
    *sh = (struct pl_shader) {
        .ctx = ctx,
        .mutable = true,
        .tmp = talloc_ref_new(ctx),
    };

    if (params)
        sh->res.params = *params;

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

void pl_shader_reset(struct pl_shader *sh, const struct pl_shader_params *params)
{
    struct pl_shader new = {
        .ctx = sh->ctx,
        .tmp = talloc_ref_new(sh->ctx),
        .mutable = true,

        // Preserve array allocations
        .variables      = sh->variables,
        .descriptors    = sh->descriptors,
        .vertex_attribs = sh->vertex_attribs,
    };

    if (params)
        new.res.params = *params;

    // Preserve buffer allocations
    for (int i = 0; i < PL_ARRAY_SIZE(new.buffers); i++)
        new.buffers[i] = (struct bstr) { .start = sh->buffers[i].start };

    talloc_ref_deref(&sh->tmp);
    *sh = new;
}

bool pl_shader_is_failed(const struct pl_shader *sh)
{
    return sh->failed;
}

struct pl_glsl_desc sh_glsl(const struct pl_shader *sh)
{
    if (SH_PARAMS(sh).glsl.version)
        return SH_PARAMS(sh).glsl;

    if (SH_GPU(sh))
        return SH_GPU(sh)->glsl;

    return (struct pl_glsl_desc) { .version = 130 };
}

bool sh_try_compute(struct pl_shader *sh, int bw, int bh, bool flex, size_t mem)
{
    pl_assert(bw && bh);
    int *sh_bw = &sh->res.compute_group_size[0];
    int *sh_bh = &sh->res.compute_group_size[1];

    const struct pl_gpu *gpu = SH_GPU(sh);
    if (!gpu || !(gpu->caps & PL_GPU_CAP_COMPUTE)) {
        PL_TRACE(sh, "Disabling compute shader due to missing PL_GPU_CAP_COMPUTE");
        return false;
    }

    if (sh->res.compute_shmem + mem > gpu->limits.max_shmem_size) {
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
                           sh->fresh++, SH_PARAMS(sh).id);
}

ident_t sh_var(struct pl_shader *sh, struct pl_shader_var sv)
{
    sv.var.name = sh_fresh(sh, sv.var.name);
    sv.data = talloc_memdup(sh->tmp, sv.data, pl_var_host_layout(0, &sv.var).size);
    TARRAY_APPEND(sh, sh->variables, sh->res.num_variables, sv);
    return (ident_t) sv.var.name;
}

ident_t sh_desc(struct pl_shader *sh, struct pl_shader_desc sd)
{
    // Skip re-attaching the same buffer desc twice
    // FIXME: define aliases if the variable names differ
    switch (sd.desc.type) {
    case PL_DESC_BUF_UNIFORM:
    case PL_DESC_BUF_STORAGE:
    case PL_DESC_BUF_TEXEL_UNIFORM:
    case PL_DESC_BUF_TEXEL_STORAGE:
        for (int i = 0; i < sh->res.num_descriptors; i++) {
            if (sh->descriptors[i].object == sd.object)
                return (ident_t) sh->descriptors[i].desc.name;
        }

    default: break;
    }

    sd.desc.name = sh_fresh(sh, sd.desc.name);
    TARRAY_APPEND(sh, sh->descriptors, sh->res.num_descriptors, sd);
    return (ident_t) sd.desc.name;
}

ident_t sh_attr_vec2(struct pl_shader *sh, const char *name,
                     const struct pl_rect2df *rc)
{
    const struct pl_gpu *gpu = SH_GPU(sh);
    if (!gpu) {
        SH_FAIL(sh, "Failed adding vertex attr '%s': No GPU available!", name);
        return NULL;
    }

    const struct pl_fmt *fmt = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 2);
    if (!fmt) {
        SH_FAIL(sh, "Failed adding vertex attr '%s': no vertex fmt!", name);
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
            .fmt      = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 2),
        },
        .data = { &data[0], &data[2], &data[4], &data[6] },
    };

    TARRAY_APPEND(sh, sh->vertex_attribs, sh->res.num_vertex_attribs, va);
    return (ident_t) va.attr.name;
}

ident_t sh_bind(struct pl_shader *sh, const struct pl_tex *tex,
                const char *name, const struct pl_rect2df *rect,
                ident_t *out_pos, ident_t *out_size, ident_t *out_pt)
{
    if (!SH_GPU(sh)) {
        SH_FAIL(sh, "Failed binding texture '%s': No GPU available!", name);
        return NULL;
    }

    if (pl_tex_params_dimension(tex->params) != 2 || !tex->params.sampleable) {
        SH_FAIL(sh, "Failed binding texture '%s': incompatible params!", name);
        return NULL;
    }

    ident_t itex = sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name = name,
            .type = PL_DESC_SAMPLED_TEX,
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
            .var  = pl_var_vec2("tex_size"),
            .data = &(float[2]) {tex->params.w, tex->params.h},
        });
    }

    if (out_pt) {
        *out_pt = sh_var(sh, (struct pl_shader_var) {
            .var  = pl_var_vec2("tex_pt"),
            .data = &(float[2]) {1.0 / tex->params.w, 1.0 / tex->params.h},
        });
    }

    return itex;
}

bool sh_buf_desc_append(void *tactx, const struct pl_gpu *gpu,
                        struct pl_shader_desc *buf_desc,
                        struct pl_var_layout *out_layout,
                        const struct pl_var new_var)
{
    struct pl_buffer_var bv = { .var = new_var };
    size_t cur_size = sh_buf_desc_size(buf_desc);

    switch (buf_desc->desc.type) {
    case PL_DESC_BUF_UNIFORM:
        bv.layout = pl_std140_layout(cur_size, &new_var);
        if (bv.layout.offset + bv.layout.size > gpu->limits.max_ubo_size)
            return false;
        break;
    case PL_DESC_BUF_STORAGE:
        bv.layout = pl_std430_layout(cur_size, &new_var);
        if (bv.layout.offset + bv.layout.size > gpu->limits.max_ssbo_size)
            return false;
        break;
    default: abort();
    }

    if (out_layout)
        *out_layout = bv.layout;
    TARRAY_APPEND(tactx, buf_desc->buffer_vars, buf_desc->num_buffer_vars, bv);
    return true;
}

size_t sh_buf_desc_size(const struct pl_shader_desc *buf_desc)
{
    if (!buf_desc->num_buffer_vars)
        return 0;

    const struct pl_buffer_var *last;
    last = &buf_desc->buffer_vars[buf_desc->num_buffer_vars - 1];
    return last->layout.offset + last->layout.size;
}

void pl_shader_append(struct pl_shader *sh, enum pl_shader_buf buf,
                      const char *fmt, ...)
{
    pl_assert(buf >= 0 && buf < SH_BUF_COUNT);

    va_list ap;
    va_start(ap, fmt);
    bstr_xappend_vasprintf_c(sh, &sh->buffers[buf], fmt, ap);
    va_end(ap);
}

void pl_shader_append_bstr(struct pl_shader *sh, enum pl_shader_buf buf,
                           struct bstr str)
{
    pl_assert(buf >= 0 && buf < SH_BUF_COUNT);
    bstr_xappend(sh, &sh->buffers[buf], str);
}

static const char *insigs[] = {
    [PL_SHADER_SIG_NONE]        = "",
    [PL_SHADER_SIG_COLOR]       = "vec4 color",
    [PL_SHADER_SIG_SAMPLER2D]   = "sampler2D src_tex, vec2 tex_coord",
};

static const char *outsigs[] = {
    [PL_SHADER_SIG_NONE]  = "void",
    [PL_SHADER_SIG_COLOR] = "vec4",
};

static const char *retvals[] = {
    [PL_SHADER_SIG_NONE]  = "",
    [PL_SHADER_SIG_COLOR] = "return color;",
};

ident_t sh_subpass(struct pl_shader *sh, const struct pl_shader *sub)
{
    pl_assert(sh->mutable);

    if (SH_PARAMS(sh).id == SH_PARAMS(sub).id) {
        PL_TRACE(sh, "Can't merge shaders: conflicting identifiers!");
        return NULL;
    }

    // Check for shader compatibility
    int res_w = PL_DEF(sh->output_w, sub->output_w),
        res_h = PL_DEF(sh->output_h, sub->output_h);

    if ((sub->output_w && res_w != sub->output_w) ||
        (sub->output_h && res_h != sub->output_h))
    {
        PL_TRACE(sh, "Can't merge shaders: incompatible sizes: %dx%d and %dx%d",
                 sh->output_w, sh->output_h, sub->output_w, sub->output_h);
        return NULL;
    }

    if (sub->is_compute) {
        int subw = sub->res.compute_group_size[0],
            subh = sub->res.compute_group_size[1];
        bool flex = sub->flexible_work_groups;

        if (!sh_try_compute(sh, subw, subh, flex, sub->res.compute_shmem)) {
            PL_TRACE(sh, "Can't merge shaders: incompatible block sizes or "
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
    GLSLH("%s\n}\n\n", retvals[sub->res.output]);

    // Copy over all of the descriptors etc.
    talloc_ref_attach(sh->tmp, sub->tmp);
#define COPY(f) TARRAY_CONCAT(sh, sh->f, sh->res.num_##f, sub->f, sub->res.num_##f)
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

    if (sh->buffers[SH_BUF_FOOTER].len) {
        bstr_xappend(sh, &sh->buffers[SH_BUF_HEADER], sh->buffers[SH_BUF_FOOTER]);
        sh->buffers[SH_BUF_FOOTER].len = 0;
        sh->buffers[SH_BUF_FOOTER].start[0] = '\0';
    }

    GLSLH("%s\n}\n\n", retvals[sh->res.output]);
    return name;
}

const struct pl_shader_res *pl_shader_finalize(struct pl_shader *sh)
{
    if (sh->failed)
        return NULL;

    if (!sh->mutable) {
        PL_WARN(sh, "Attempted to finalize a shader twice?");
        return &sh->res;
    }

    // Split the shader. This finalizes the body and adds it to the header
    sh->res.name = sh_split(sh);

    // Padding for readability
    GLSLP("\n");

    // Concatenate the header onto the prelude to form the final output
    struct bstr *glsl = &sh->buffers[SH_BUF_PRELUDE];
    bstr_xappend(sh, glsl, sh->buffers[SH_BUF_HEADER]);

    // Set the vas/vars/descs
    sh->res.vertex_attribs = sh->vertex_attribs;
    sh->res.variables = sh->variables;
    sh->res.descriptors = sh->descriptors;

    // Update the result pointer and return
    sh->res.glsl = glsl->start;
    sh->mutable = false;
    return &sh->res;
}

bool sh_require(struct pl_shader *sh, enum pl_shader_sig insig, int w, int h)
{
    if (sh->failed) {
        SH_FAIL(sh, "Attempting to modify a failed shader!");
        return false;
    }

    if (!sh->mutable) {
        SH_FAIL(sh, "Attempted to modify an immutable shader!");
        return false;
    }

    if ((w && sh->output_w && sh->output_w != w) ||
        (h && sh->output_h && sh->output_h != h))
    {
        SH_FAIL(sh, "Illegal sequence of shader operations: Incompatible "
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
        SH_FAIL(sh, "Illegal sequence of shader operations! Current output "
                "signature is '%s', but called operation expects '%s'!",
                names[sh->res.output], names[insig]);
        return false;
    }

    // All of our shaders end up returning a vec4 color
    sh->res.output = PL_SHADER_SIG_COLOR;
    sh->output_w = PL_DEF(sh->output_w, w);
    sh->output_h = PL_DEF(sh->output_h, h);
    return true;
}

void pl_shader_obj_destroy(struct pl_shader_obj **ptr)
{
    struct pl_shader_obj *obj = *ptr;
    if (!obj)
        return;

    if (obj->uninit)
        obj->uninit(obj->gpu, obj->priv);

    *ptr = NULL;
    talloc_free(obj);
}

void *sh_require_obj(struct pl_shader *sh, struct pl_shader_obj **ptr,
                     enum pl_shader_obj_type type, size_t priv_size,
                     void (*uninit)(const struct pl_gpu *gpu, void *priv))
{
    if (!ptr)
        return NULL;

    struct pl_shader_obj *obj = *ptr;
    if (obj && obj->gpu != SH_GPU(sh)) {
        SH_FAIL(sh, "Passed pl_shader_obj belongs to different GPU!");
        return NULL;
    }

    if (obj && obj->type != type) {
        SH_FAIL(sh, "Passed pl_shader_obj of wrong type! Shader objects must "
                "always be used with the same type of shader.");
        return NULL;
    }

    if (!obj) {
        obj = talloc_zero(NULL, struct pl_shader_obj);
        obj->gpu = SH_GPU(sh);
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

    // Phi is the most irrational number, so it's a good candidate for
    // generating seed values to the PRNG
    static const double phi = 1.618033988749895;

    const char *seed = "0.0";
    if (temporal) {
        float seedval = modff(phi * SH_PARAMS(sh).index, &(float){0});
        seed = sh_var(sh, (struct pl_shader_var) {
            .var  = pl_var_float("seed"),
            .data = &seedval,
            .dynamic = true,
        });
    }

    ident_t state = sh_fresh(sh, "prng");
    GLSL("vec2 %s_init = fract(gl_FragCoord.xy * vec2(%f)); \n"
         "vec3 %s_m = vec3(%s_init, %s) + vec3(1.0);        \n"
         "float %s = %s(%s(%s(%s_m.x) + %s_m.y) + %s_m.z);  \n",
         state, phi,
         state, state, seed,
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
    int width, height, depth, comps;
    union {
        const struct pl_tex *tex;
        struct bstr str;
        float *data;
    } weights;
};

static void sh_lut_uninit(const struct pl_gpu *gpu, void *ptr)
{
    struct sh_lut_obj *lut = ptr;
    switch (lut->method) {
    case SH_LUT_TEXTURE:
    case SH_LUT_LINEAR:
        pl_tex_destroy(gpu, &lut->weights.tex);
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
               int comps, bool update, bool dynamic, void *priv,
               void (*fill)(void *priv, float *data, int w, int h, int d))
{
    const struct pl_gpu *gpu = SH_GPU(sh);
    float *tmp = NULL;
    ident_t ret = NULL;

    pl_assert(width > 0 && height >= 0 && depth >= 0);
    int sizes[] = { width, height, depth };
    int size = width * PL_DEF(height, 1) * PL_DEF(depth, 1);
    int dims = depth ? 3 : height ? 2 : 1;

    int texdim = 0;
    uint32_t max_tex_dim[] = {
        gpu ? gpu->limits.max_tex_1d_dim : 0,
        gpu ? gpu->limits.max_tex_2d_dim : 0,
        gpu ? gpu->limits.max_tex_3d_dim : 0,
    };

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

    struct sh_lut_obj *lut = SH_OBJ(sh, obj, PL_SHADER_OBJ_LUT,
                                    struct sh_lut_obj, sh_lut_uninit);

    if (!lut) {
        SH_FAIL(sh, "Failed initializing LUT object!");
        goto error;
    }

    if (!gpu && method == SH_LUT_LINEAR) {
        SH_FAIL(sh, "Linear LUTs require the use of a GPU!");
        goto error;
    }

    if (!gpu) {
        PL_TRACE(sh, "No GPU available, falling back to literal LUT embedding");
        method = SH_LUT_LITERAL;
    }

    // Pick the best method
    if (!method && size <= SH_LUT_MAX_LITERAL)
        method = SH_LUT_LITERAL;

    if (!method && texdim)
        method = SH_LUT_TEXTURE;

    if (!method && gpu && gpu->caps & PL_GPU_CAP_INPUT_VARIABLES)
        method = SH_LUT_UNIFORM;

    // texelFetch requires GLSL >= 130, so fall back to the LINEAR code
    if (method == SH_LUT_TEXTURE && gpu->glsl.version < 130)
        method = SH_LUT_LINEAR;

    // No other method found
    if (!method) {
        PL_TRACE(sh, "No other LUT method works, falling back to literal "
                 "embedding.. this is most likely a slow path!");
        method = SH_LUT_LITERAL;
    }

    // Forcibly reinitialize the existing LUT if needed
    if (method != lut->method || width != lut->width || height != lut->height
        || depth != lut->depth || comps != lut->comps)
    {
        PL_DEBUG(sh, "LUT method or size changed, reinitializing..");
        update = true;
    }

    if (update) {
        tmp = talloc_zero_size(NULL, size * comps * sizeof(float));
        fill(priv, tmp, width, height, depth);

        switch (method) {
        case SH_LUT_TEXTURE:
        case SH_LUT_LINEAR: {
            if (!texdim) {
                SH_FAIL(sh, "Texture LUT exceeds texture dimensions!");
                goto error;
            }

            enum pl_fmt_caps caps = PL_FMT_CAP_SAMPLEABLE;
            enum pl_tex_sample_mode mode = PL_TEX_SAMPLE_NEAREST;

            if (method == SH_LUT_LINEAR) {
                caps |= PL_FMT_CAP_LINEAR;
                mode = PL_TEX_SAMPLE_LINEAR;
            }

            const struct pl_fmt *fmt;
            fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, comps, 16, 32, caps);
            if (!fmt) {
                SH_FAIL(sh, "Found no compatible texture format for LUT!");
                goto error;
            }

            struct pl_tex_params params = {
                .w              = width,
                .h              = PL_DEF(height, texdim >= 2 ? 1 : 0),
                .d              = PL_DEF(depth,  texdim >= 3 ? 1 : 0),
                .format         = fmt,
                .sampleable     = true,
                .sample_mode    = mode,
                .address_mode   = PL_TEX_ADDRESS_CLAMP,
                .host_writable  = dynamic,
                .initial_data   = dynamic ? NULL : tmp,
            };

            bool ok;
            if (dynamic) {
                ok = pl_tex_recreate(gpu, &lut->weights.tex, &params);
                if (ok) {
                    ok = pl_tex_upload(gpu, &(struct pl_tex_transfer_params) {
                        .tex = lut->weights.tex,
                        .ptr = tmp,
                    });
                }
            } else {
                pl_tex_destroy(gpu, &lut->weights.tex);
                lut->weights.tex = pl_tex_create(gpu, &params);
                ok = lut->weights.tex;
            }

            if (!ok) {
                SH_FAIL(sh, "Failed creating LUT texture!");
                goto error;
            }
            break;
        }

        case SH_LUT_UNIFORM:
            talloc_free(lut->weights.data);
            lut->weights.data = tmp; // re-use `tmp`
            tmp = NULL;
            break;

        case SH_LUT_LITERAL: {
            lut->weights.str.len = 0;
            for (int i = 0; i < size * comps; i += comps) {
                if (i > 0)
                    bstr_xappend_asprintf_c(lut, &lut->weights.str, ",");
                if (comps > 1)
                    bstr_xappend_asprintf_c(lut, &lut->weights.str, "vec%d(", comps);
                for (int c = 0; c < comps; c++) {
                    bstr_xappend_asprintf_c(lut, &lut->weights.str, "%s%f",
                                            c > 0 ? "," : "",
                                            tmp[i+c]);
                }
                if (comps > 1)
                    bstr_xappend_asprintf_c(lut, &lut->weights.str, ")");
            }
            break;
        }

        case SH_LUT_AUTO: abort();
        }

        lut->method = method;
        lut->width = width;
        lut->height = height;
        lut->depth = depth;
        lut->comps = comps;
    }

    // Done updating, generate the GLSL
    ident_t name = sh_fresh(sh, "lut");
    ident_t arr_name = NULL;

    static const char * const types[] = {"float", "vec2", "vec3", "vec4"};
    static const char * const itypes[] = {"uint", "ivec2", "ivec3", "ivec4"};
    static const char * const swizzles[] = {"x", "xy", "xyz", "xyzw"};

    switch (method) {
    case SH_LUT_TEXTURE: {
        assert(texdim);
        ident_t tex = sh_desc(sh, (struct pl_shader_desc) {
            .desc = {
                .name = "weights",
                .type = PL_DESC_SAMPLED_TEX,
            },
            .object = lut->weights.tex,
        });

        GLSLH("#define %s(pos) (texelFetch(%s, %s(pos",
              name, tex, itypes[texdim - 1]);

        // Fill up extra components of the index
        for (int i = dims; i < texdim; i++)
            GLSLH(", 0");

        GLSLH("), 0).%s)\n", swizzles[comps - 1]);
        ret = name;
        break;
    }

    case SH_LUT_LINEAR: {
        assert(texdim);
        ident_t tex = sh_desc(sh, (struct pl_shader_desc) {
            .desc = {
                .name = "weights",
                .type = PL_DESC_SAMPLED_TEX,
            },
            .object = lut->weights.tex,
        });

        ident_t pos_macros[PL_ARRAY_SIZE(sizes)] = {0};
        for (int i = 0; i < dims; i++)
            pos_macros[i] = sh_lut_pos(sh, sizes[i]);

        GLSLH("#define %s(pos) (%s(%s, %s(\\\n",
              name, sh_tex_fn(sh, lut->weights.tex->params),
              tex, types[texdim - 1]);

        for (int i = 0; i < texdim; i++) {
            char sep = i == 0 ? ' ' : ',';
            if (pos_macros[i]) {
                if (dims > 1) {
                    GLSLH("   %c%s(%s(pos).%c)\\\n", sep, pos_macros[i],
                          types[dims - 1], "xyzw"[i]);
                } else {
                    GLSLH("   %c%s(float(pos))\\\n", sep, pos_macros[i]);
                }
            } else {
                GLSLH("   %c%f\\\n", sep, 0.5);
            }
        }
        GLSLH("  )).%s)\n", swizzles[comps - 1]);
        ret = name;
        break;
    }

    case SH_LUT_UNIFORM:
        arr_name = sh_var(sh, (struct pl_shader_var) {
            .var = {
                .name = "weights",
                .type = PL_VAR_FLOAT,
                .dim_v = comps,
                .dim_m = 1,
                .dim_a = size,
            },
            .data = lut->weights.data,
        });
        break;

    case SH_LUT_LITERAL:
        arr_name = sh_fresh(sh, "weights");
        GLSLH("const %s %s[%d] = float[](\n  ", types[comps - 1], arr_name, size);
        bstr_xappend(sh, &sh->buffers[SH_BUF_HEADER], lut->weights.str);
        GLSLH(");\n");
        break;

    default: abort();
    }

    if (arr_name) {
        GLSLH("#define %s(pos) (%s[int((pos).x)\\\n", name, arr_name);
        int shift = width;
        for (int i = 1; i < dims; i++) {
            GLSLH("    + %d * int((pos)[%d])\\\n", shift, i);
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

const char *sh_bvec(const struct pl_shader *sh, int dims)
{
    static const char *bvecs[] = {
        [1] = "bool",
        [2] = "bvec2",
        [3] = "bvec3",
        [4] = "bvec4",
    };

    static const char *vecs[] = {
        [1] = "float",
        [2] = "vec2",
        [3] = "vec3",
        [4] = "vec4",
    };

    pl_assert(dims > 0 && dims < PL_ARRAY_SIZE(bvecs));
    return sh_glsl(sh).version >= 130 ? bvecs[dims] : vecs[dims];
}
