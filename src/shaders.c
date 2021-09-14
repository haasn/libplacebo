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
#include "log.h"
#include "shaders.h"

pl_shader pl_shader_alloc(pl_log log, const struct pl_shader_params *params)
{
    pl_shader sh = pl_alloc_ptr(NULL, sh);
    *sh = (struct pl_shader) {
        .log = log,
        .mutable = true,
    };

    // Ensure there's always at least one `tmp` object
    PL_ARRAY_APPEND(sh, sh->tmp, pl_ref_new(NULL));

    if (params)
        sh->res.params = *params;

    return sh;
}

void pl_shader_free(pl_shader *psh)
{
    pl_shader sh = *psh;
    if (!sh)
        return;

    for (int i = 0; i < sh->tmp.num; i++)
        pl_ref_deref(&sh->tmp.elem[i]);

    pl_free_ptr(psh);
}

void pl_shader_reset(pl_shader sh, const struct pl_shader_params *params)
{
    for (int i = 0; i < sh->tmp.num; i++)
        pl_ref_deref(&sh->tmp.elem[i]);

    struct pl_shader new = {
        .log = sh->log,
        .mutable = true,

        // Preserve array allocations
        .tmp.elem       = sh->tmp.elem,
        .vas.elem       = sh->vas.elem,
        .vars.elem      = sh->vars.elem,
        .descs.elem     = sh->descs.elem,
        .consts.elem    = sh->consts.elem,
        .steps.elem     = sh->steps.elem,
    };

    if (params)
        new.res.params = *params;

    // Preserve buffer allocations
    for (int i = 0; i < PL_ARRAY_SIZE(new.buffers); i++)
        new.buffers[i] = (pl_str) { .buf = sh->buffers[i].buf };

    *sh = new;
    PL_ARRAY_APPEND(sh, sh->tmp, pl_ref_new(NULL));
}

bool pl_shader_is_failed(const pl_shader sh)
{
    return sh->failed;
}

struct pl_glsl_version sh_glsl(const pl_shader sh)
{
    if (SH_PARAMS(sh).glsl.version)
        return SH_PARAMS(sh).glsl;

    if (SH_GPU(sh))
        return SH_GPU(sh)->glsl;

    return (struct pl_glsl_version) { .version = 130 };
}

bool sh_try_compute(pl_shader sh, int bw, int bh, bool flex, size_t mem)
{
    pl_assert(bw && bh);
    int *sh_bw = &sh->res.compute_group_size[0];
    int *sh_bh = &sh->res.compute_group_size[1];

    struct pl_glsl_version glsl = sh_glsl(sh);
    if (!glsl.compute) {
        PL_TRACE(sh, "Disabling compute shader due to missing `compute` support");
        return false;
    }

    if (sh->res.compute_shmem + mem > glsl.max_shmem_size) {
        PL_TRACE(sh, "Disabling compute shader due to insufficient shmem");
        return false;
    }

    if (bw > glsl.max_group_size[0] ||
        bh > glsl.max_group_size[1] ||
        (bw * bh) > glsl.max_group_threads)
    {
        if (!flex) {
            PL_TRACE(sh, "Disabling compute shader due to exceeded group "
                     "thread count.");
            return false;
        } else {
            // Pick better group sizes
            bw = PL_MIN(bw, glsl.max_group_size[0]);
            bh = glsl.max_group_threads / bw;
        }
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
        pl_assert(*sh_bw * *sh_bh <= glsl.max_group_threads);
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

bool pl_shader_is_compute(const pl_shader sh)
{
    return sh->is_compute;
}

bool pl_shader_output_size(const pl_shader sh, int *w, int *h)
{
    if (!sh->output_w || !sh->output_h)
        return false;

    *w = sh->output_w;
    *h = sh->output_h;
    return true;
}

ident_t sh_fresh(pl_shader sh, const char *name)
{
    return pl_asprintf(SH_TMP(sh), "_%s_%d_%u", PL_DEF(name, "var"),
                       sh->fresh++, SH_PARAMS(sh).id);
}

ident_t sh_var(pl_shader sh, struct pl_shader_var sv)
{
    sv.var.name = sh_fresh(sh, sv.var.name);
    sv.data = pl_memdup(SH_TMP(sh), sv.data, pl_var_host_layout(0, &sv.var).size);
    PL_ARRAY_APPEND(sh, sh->vars, sv);
    return (ident_t) sv.var.name;
}

ident_t sh_desc(pl_shader sh, struct pl_shader_desc sd)
{
    switch (sd.desc.type) {
    case PL_DESC_BUF_UNIFORM:
    case PL_DESC_BUF_STORAGE:
    case PL_DESC_BUF_TEXEL_UNIFORM:
    case PL_DESC_BUF_TEXEL_STORAGE:
        // Skip re-attaching the same buffer desc twice
        // FIXME: define aliases if the variable names differ
        for (int i = 0; i < sh->descs.num; i++) {
            if (sh->descs.elem[i].binding.object == sd.binding.object)
                return (ident_t) sh->descs.elem[i].desc.name;
        }

        size_t bsize = sizeof(sd.buffer_vars[0]) * sd.num_buffer_vars;
        sd.buffer_vars = pl_memdup(SH_TMP(sh), sd.buffer_vars, bsize);
        break;

    case PL_DESC_SAMPLED_TEX:
    case PL_DESC_STORAGE_IMG:
        pl_assert(!sd.num_buffer_vars);
        break;

    case PL_DESC_INVALID:
    case PL_DESC_TYPE_COUNT:
        pl_unreachable();
    }

    sd.desc.name = sh_fresh(sh, sd.desc.name);
    PL_ARRAY_APPEND(sh, sh->descs, sd);
    return (ident_t) sd.desc.name;
}

ident_t sh_const(pl_shader sh, struct pl_shader_const sc)
{
    if (sh->res.params.dynamic_constants && !sc.compile_time) {
        return sh_var(sh, (struct pl_shader_var) {
            .var = {
                .name = sc.name,
                .type = sc.type,
                .dim_v = 1,
                .dim_m = 1,
                .dim_a = 1,
            },
            .data = sc.data,
        });
    }

    sc.name = sh_fresh(sh, sc.name);

    pl_gpu gpu = SH_GPU(sh);
    if (gpu && gpu->limits.max_constants) {
        sc.data = pl_memdup(SH_TMP(sh), sc.data, pl_var_type_size(sc.type));
        PL_ARRAY_APPEND(sh, sh->consts, sc);
        return (ident_t) sc.name;
    }

    // Fallback for GPUs without specialization constants
    switch (sc.type) {
    case PL_VAR_SINT:
        GLSLH("const int %s = %d; \n", sc.name, *(int *) sc.data);
        return (ident_t) sc.name;
    case PL_VAR_UINT:
        GLSLH("const uint %s = %uu; \n", sc.name, *(unsigned int *) sc.data);
        return (ident_t) sc.name;
    case PL_VAR_FLOAT:
        GLSLH("const float %s = %f; \n", sc.name, *(float *) sc.data);
        return (ident_t) sc.name;
    case PL_VAR_INVALID:
    case PL_VAR_TYPE_COUNT:
        break;
    }

    pl_unreachable();
}

ident_t sh_const_int(pl_shader sh, const char *name, int val)
{
    return sh_const(sh, (struct pl_shader_const) {
        .type = PL_VAR_SINT,
        .name = name,
        .data = &val,
    });
}

ident_t sh_const_uint(pl_shader sh, const char *name, unsigned int val)
{
    return sh_const(sh, (struct pl_shader_const) {
        .type = PL_VAR_UINT,
        .name = name,
        .data = &val,
    });
}

ident_t sh_const_float(pl_shader sh, const char *name, float val)
{
    return sh_const(sh, (struct pl_shader_const) {
        .type = PL_VAR_FLOAT,
        .name = name,
        .data = &val,
    });
}


ident_t sh_attr_vec2(pl_shader sh, const char *name,
                     const struct pl_rect2df *rc)
{
    pl_gpu gpu = SH_GPU(sh);
    if (!gpu) {
        SH_FAIL(sh, "Failed adding vertex attr '%s': No GPU available!", name);
        return NULL;
    }

    pl_fmt fmt = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 2);
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

    float *data = pl_memdup(SH_TMP(sh), &vals[0][0], sizeof(vals));
    struct pl_shader_va va = {
        .attr = {
            .name     = sh_fresh(sh, name),
            .fmt      = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 2),
        },
        .data = { &data[0], &data[2], &data[4], &data[6] },
    };

    PL_ARRAY_APPEND(sh, sh->vas, va);
    return (ident_t) va.attr.name;
}

ident_t sh_bind(pl_shader sh, pl_tex tex,
                enum pl_tex_address_mode address_mode,
                enum pl_tex_sample_mode sample_mode,
                const char *name, const struct pl_rect2df *rect,
                ident_t *out_pos, ident_t *out_size, ident_t *out_pt)
{
    if (pl_tex_params_dimension(tex->params) != 2 || !tex->params.sampleable) {
        SH_FAIL(sh, "Failed binding texture '%s': incompatible params!", name);
        return NULL;
    }

    ident_t itex = sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name = name,
            .type = PL_DESC_SAMPLED_TEX,
        },
        .binding = {
            .object = tex,
            .address_mode = address_mode,
            .sample_mode = sample_mode,
        },
    });

    float sx, sy;
    if (tex->sampler_type == PL_SAMPLER_RECT) {
        sx = 1.0;
        sy = 1.0;
    } else {
        sx = 1.0 / tex->params.w;
        sy = 1.0 / tex->params.h;
    }

    if (out_pos) {
        struct pl_rect2df full = {
            .x1 = tex->params.w,
            .y1 = tex->params.h,
        };

        rect = PL_DEF(rect, &full);
        *out_pos = sh_attr_vec2(sh, "tex_coord", &(struct pl_rect2df) {
            .x0 = sx * rect->x0, .y0 = sy * rect->y0,
            .x1 = sx * rect->x1, .y1 = sy * rect->y1,
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
            .data = &(float[2]) {sx, sy},
        });
    }

    return itex;
}

bool sh_buf_desc_append(void *alloc, pl_gpu gpu,
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
    case PL_DESC_INVALID:
    case PL_DESC_SAMPLED_TEX:
    case PL_DESC_STORAGE_IMG:
    case PL_DESC_BUF_TEXEL_UNIFORM:
    case PL_DESC_BUF_TEXEL_STORAGE:
    case PL_DESC_TYPE_COUNT:
        pl_unreachable();
    }

    if (out_layout)
        *out_layout = bv.layout;
    PL_ARRAY_APPEND_RAW(alloc, buf_desc->buffer_vars, buf_desc->num_buffer_vars, bv);
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

void sh_append(pl_shader sh, enum pl_shader_buf buf, const char *fmt, ...)
{
    pl_assert(buf >= 0 && buf < SH_BUF_COUNT);

    va_list ap;
    va_start(ap, fmt);
    pl_str_append_vasprintf_c(sh, &sh->buffers[buf], fmt, ap);
    va_end(ap);
}

void sh_append_str(pl_shader sh, enum pl_shader_buf buf, pl_str str)
{
    pl_assert(buf >= 0 && buf < SH_BUF_COUNT);
    pl_str_append(sh, &sh->buffers[buf], str);
}

static const char *insigs[] = {
    [PL_SHADER_SIG_NONE]  = "",
    [PL_SHADER_SIG_COLOR] = "vec4 color",
};

static const char *outsigs[] = {
    [PL_SHADER_SIG_NONE]  = "void",
    [PL_SHADER_SIG_COLOR] = "vec4",
};

static const char *retvals[] = {
    [PL_SHADER_SIG_NONE]  = "",
    [PL_SHADER_SIG_COLOR] = "return color;",
};

// libplacebo currently only allows 2D samplers for shader signatures
static const char *samplers2D[] = {
    [PL_SAMPLER_NORMAL]     = "sampler2D",
    [PL_SAMPLER_RECT]       = "sampler2DRect",
    [PL_SAMPLER_EXTERNAL]   = "samplerExternalOES",
};

ident_t sh_subpass(pl_shader sh, const pl_shader sub)
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
    pl_str_append(sh, &sh->buffers[SH_BUF_PRELUDE], sub->buffers[SH_BUF_PRELUDE]);
    pl_str_append(sh, &sh->buffers[SH_BUF_HEADER],  sub->buffers[SH_BUF_HEADER]);

    // Append the body as a new header function
    ident_t name = sh_fresh(sh, "sub");
    if (sub->res.input == PL_SHADER_SIG_SAMPLER) {
        pl_assert(sub->sampler_prefix);
        GLSLH("%s %s(%c%s src_tex, vec2 tex_coord) {\n",
              outsigs[sub->res.output], name,
              sub->sampler_prefix, samplers2D[sub->sampler_type]);
    } else {
        GLSLH("%s %s(%s) {\n", outsigs[sub->res.output], name, insigs[sub->res.input]);
    }
    pl_str_append(sh, &sh->buffers[SH_BUF_HEADER], sub->buffers[SH_BUF_BODY]);
    GLSLH("%s\n}\n\n", retvals[sub->res.output]);

    // Copy over all of the descriptors etc.
    for (int i = 0; i < sub->tmp.num; i++)
        PL_ARRAY_APPEND(sh, sh->tmp, pl_ref_dup(sub->tmp.elem[i]));
    PL_ARRAY_CONCAT(sh, sh->vas, sub->vas);
    PL_ARRAY_CONCAT(sh, sh->vars, sub->vars);
    PL_ARRAY_CONCAT(sh, sh->descs, sub->descs);
    PL_ARRAY_CONCAT(sh, sh->consts, sub->consts);
    PL_ARRAY_CONCAT(sh, sh->steps, sub->steps);

    return name;
}

// Finish the current shader body and return its function name
static ident_t sh_split(pl_shader sh)
{
    pl_assert(sh->mutable);

    // Concatenate the body onto the head as a new function
    ident_t name = sh_fresh(sh, "main");
    if (sh->res.input == PL_SHADER_SIG_SAMPLER) {
        pl_assert(sh->sampler_prefix);
        GLSLH("%s %s(%c%s src_tex, vec2 tex_coord) {\n",
              outsigs[sh->res.output], name,
              sh->sampler_prefix, samplers2D[sh->sampler_type]);
    } else {
        GLSLH("%s %s(%s) {\n", outsigs[sh->res.output], name, insigs[sh->res.input]);
    }

    if (sh->buffers[SH_BUF_BODY].len) {
        pl_str_append(sh, &sh->buffers[SH_BUF_HEADER], sh->buffers[SH_BUF_BODY]);
        sh->buffers[SH_BUF_BODY].len = 0;
        sh->buffers[SH_BUF_BODY].buf[0] = '\0'; // for sanity / efficiency
    }

    if (sh->buffers[SH_BUF_FOOTER].len) {
        pl_str_append(sh, &sh->buffers[SH_BUF_HEADER], sh->buffers[SH_BUF_FOOTER]);
        sh->buffers[SH_BUF_FOOTER].len = 0;
        sh->buffers[SH_BUF_FOOTER].buf[0] = '\0';
    }

    GLSLH("%s\n}\n\n", retvals[sh->res.output]);
    return name;
}

const struct pl_shader_res *pl_shader_finalize(pl_shader sh)
{
    if (sh->failed)
        return NULL;

    if (!sh->mutable)
        return &sh->res;

    // Split the shader. This finalizes the body and adds it to the header
    sh->res.name = sh_split(sh);

    // Padding for readability
    GLSLP("\n");

    // Concatenate the header onto the prelude to form the final output
    pl_str *glsl = &sh->buffers[SH_BUF_PRELUDE];
    pl_str_append(sh, glsl, sh->buffers[SH_BUF_HEADER]);

    // Generate the pretty description
    sh->res.description = "(unknown shader)";
    if (sh->steps.num) {
        // Reuse this buffer
        pl_str *desc = &sh->buffers[SH_BUF_BODY];
        desc->len = 0;

        for (int i = 0; i < sh->steps.num; i++) {
            const char *step = sh->steps.elem[i];
            if (!step)
                continue;

            // Group together duplicates. We're okay using a weak equality
            // check here because all pass descriptions are static strings.
            int count = 1;
            for (int j = i+1; j < sh->steps.num; j++) {
                if (sh->steps.elem[j] == step) {
                    sh->steps.elem[j] = NULL;
                    count++;
                }
            }

            if (i > 0)
                pl_str_append(sh, desc, pl_str0(", "));
            pl_str_append(sh, desc, pl_str0(step));
            if (count > 1)
                pl_str_append_asprintf(sh, desc, " x%d", count);
        }

        sh->res.description = desc->buf;
    }

    // Set the vas/vars/descs
    sh->res.vertex_attribs = sh->vas.elem;
    sh->res.num_vertex_attribs = sh->vas.num;
    sh->res.variables = sh->vars.elem;
    sh->res.num_variables = sh->vars.num;
    sh->res.descriptors = sh->descs.elem;
    sh->res.num_descriptors = sh->descs.num;
    sh->res.constants = sh->consts.elem;
    sh->res.num_constants = sh->consts.num;
    sh->res.steps = sh->steps.elem;
    sh->res.num_steps = sh->steps.num;

    // Update the result pointer and return
    sh->res.glsl = glsl->buf;
    sh->mutable = false;
    return &sh->res;
}

bool sh_require(pl_shader sh, enum pl_shader_sig insig, int w, int h)
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

void pl_shader_obj_destroy(pl_shader_obj *ptr)
{
    pl_shader_obj obj = *ptr;
    if (!obj)
        return;

    if (obj->uninit)
        obj->uninit(obj->gpu, obj->priv);

    *ptr = NULL;
    pl_free(obj);
}

void *sh_require_obj(pl_shader sh, pl_shader_obj *ptr,
                     enum pl_shader_obj_type type, size_t priv_size,
                     void (*uninit)(pl_gpu gpu, void *priv))
{
    if (!ptr)
        return NULL;

    pl_shader_obj obj = *ptr;
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
        obj = pl_zalloc_ptr(NULL, obj);
        obj->gpu = SH_GPU(sh);
        obj->type = type;
        obj->priv = pl_zalloc(obj, priv_size);
        obj->uninit = uninit;
    }

    *ptr = obj;
    return obj->priv;
}

ident_t sh_prng(pl_shader sh, bool temporal, ident_t *p_state)
{
    ident_t randfun = sh_fresh(sh, "rand"),
            state = sh_fresh(sh, "state");

    if (sh_glsl(sh).version >= 130) {

        // Based on pcg3d (http://jcgt.org/published/0009/03/02/)
        GLSLP("#define prng_t uvec3\n");
        GLSLH("vec3 %s(inout uvec3 s) {                     \n"
              "    s = 1664525u * s + uvec3(1013904223u);   \n"
              "    s.x += s.y * s.z;                        \n"
              "    s.y += s.z * s.x;                        \n"
              "    s.z += s.x * s.y;                        \n"
              "    s ^= s >> 16u;                           \n"
              "    s.x += s.y * s.z;                        \n"
              "    s.y += s.z * s.x;                        \n"
              "    s.z += s.x * s.y;                        \n"
              "    return vec3(s) * 1.0/float(0xFFFFFFFFu); \n"
              "}                                            \n",
              randfun);

        const char *seed = "0u";
        if (temporal) {
            seed = sh_var(sh, (struct pl_shader_var) {
                .var  = pl_var_uint("seed"),
                .data = &(unsigned int){ SH_PARAMS(sh).index },
                .dynamic = true,
            });
        };

        GLSL("uvec3 %s = uvec3(gl_FragCoord.xy, %s); \n", state, seed);

    } else {

        // Based on SGGP (https://briansharpe.wordpress.com/2011/10/01/gpu-texture-free-noise/)
        ident_t permute = sh_fresh(sh, "permute");
        GLSLP("#define prng_t float\n");
        GLSLH("float %s(float x) {                          \n"
              "    x = (34.0 * x + 1.0) * x;                \n"
              "    return fract(x * 1.0/289.0) * 289.0;     \n"
              "}                                            \n"
              "vec3 %s(inout float s) {                     \n"
              "    vec3 ret;                                \n"
              "    ret.x = %s(s);                           \n"
              "    ret.y = %s(ret.x);                       \n"
              "    ret.z = %s(ret.y);                       \n"
              "    s = ret.z;                               \n"
              "    return fract(ret * 1.0/41.0);            \n"
              "}                                            \n",
              permute, randfun, permute, permute, permute);

        static const double phi = 1.618033988749895;
        const char *seed = "0.0";
        if (temporal) {
            seed = sh_var(sh, (struct pl_shader_var) {
                .var  = pl_var_float("seed"),
                .data = &(float){ modff(phi * SH_PARAMS(sh).index, &(float){0}) },
                .dynamic = true,
            });
        };

        GLSL("vec3 %s_m = vec3(fract(gl_FragCoord.xy * vec2(%f)), %s);  \n"
             "%s_m += vec3(1.0);                                        \n"
             "float %s = %s(%s(%s(%s_m.x) + %s_m.y) + %s_m.z);          \n",
             state, phi, seed,
             state,
             state, permute, permute, permute, state, state, state);

    }

    if (p_state)
        *p_state = state;

    ident_t res = sh_fresh(sh, "RAND");
    GLSLH("#define %s (%s(%s))\n", res, randfun, state);
    return res;
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
    enum sh_lut_method method;
    enum pl_var_type type;
    bool linear;
    int width, height, depth, comps;
    uint64_t signature;

    // weights, depending on the method
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
#define SH_LUT_MAX_LITERAL 256

ident_t sh_lut(pl_shader sh, const struct sh_lut_params *params)
{
    pl_gpu gpu = SH_GPU(sh);
    void *tmp = NULL;
    ident_t ret = NULL;

    pl_assert(params->width > 0 && params->height >= 0 && params->depth >= 0);
    pl_assert(params->comps > 0);
    pl_assert(params->type);
    pl_assert(!params->linear || params->type == PL_VAR_FLOAT);

    int sizes[] = { params->width, params->height, params->depth };
    int size = params->width * PL_DEF(params->height, 1) * PL_DEF(params->depth, 1);
    int dims = params->depth ? 3 : params->height ? 2 : 1;

    int texdim = 0;
    uint32_t max_tex_dim[] = {
        gpu ? gpu->limits.max_tex_1d_dim : 0,
        gpu ? gpu->limits.max_tex_2d_dim : 0,
        (gpu && gpu->glsl.version > 100) ? gpu->limits.max_tex_3d_dim : 0,
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

    static const enum pl_fmt_type fmt_type[PL_VAR_TYPE_COUNT] = {
        [PL_VAR_SINT]   = PL_FMT_SINT,
        [PL_VAR_UINT]   = PL_FMT_UINT,
        [PL_VAR_FLOAT]  = PL_FMT_FLOAT,
    };

    enum pl_fmt_caps texcaps = PL_FMT_CAP_SAMPLEABLE;
    if (params->linear)
        texcaps |= PL_FMT_CAP_LINEAR;

    pl_fmt texfmt = NULL;
    if (texdim) {
        texfmt = pl_find_fmt(gpu, fmt_type[params->type], params->comps,
                             params->type == PL_VAR_FLOAT ? 16 : 32,
                             pl_var_type_size(params->type) * 8,
                             texcaps);
    }

    struct sh_lut_obj *lut = SH_OBJ(sh, params->object, PL_SHADER_OBJ_LUT,
                                    struct sh_lut_obj, sh_lut_uninit);

    if (!lut) {
        SH_FAIL(sh, "Failed initializing LUT object!");
        goto error;
    }

    enum sh_lut_method method = params->method;

    // The linear sampling code currently only supports 1D linear interpolation
    if (params->linear && dims > 1) {
        if (texfmt) {
            method = SH_LUT_TEXTURE;
        } else {
            SH_FAIL(sh, "Can't emulate linear LUTs for 2D/3D LUTs and no "
                    "texture support available!");
            goto error;
        }
    }

    // Older GLSL forbids literal array constructors
    bool can_literal = sh_glsl(sh).version > 110;

    // Pick the best method
    if (!method && size <= SH_LUT_MAX_LITERAL && !params->dynamic && can_literal)
        method = SH_LUT_LITERAL; // use literals for small constant LUTs

    if (!method && texfmt)
        method = SH_LUT_TEXTURE; // use textures if a texfmt exists

    // Use an input variable as a last fallback
    if (!method)
        method = SH_LUT_UNIFORM;

    // Forcibly reinitialize the existing LUT if needed
    bool update = params->update || lut->signature != params->signature;
    if (method != lut->method || params->type != lut->type ||
        params->linear != lut->linear || params->width != lut->width ||
        params->height != lut->height || params->depth != lut->depth ||
        params->comps != lut->comps)
    {
        PL_DEBUG(sh, "LUT cache invalidated, regenerating..");
        update = true;
    }

    if (update) {
        size_t buf_size = size * params->comps * pl_var_type_size(params->type);
        tmp = pl_zalloc(NULL, buf_size);
        params->fill(tmp, params);

        switch (method) {
        case SH_LUT_TEXTURE: {
            if (!texdim) {
                SH_FAIL(sh, "Texture LUT exceeds texture dimensions!");
                goto error;
            }

            if (!texfmt) {
                SH_FAIL(sh, "Found no compatible texture format for LUT!");
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
            };

            bool ok;
            if (params->dynamic) {
                ok = pl_tex_recreate(gpu, &lut->tex, &tex_params);
                if (ok) {
                    ok = pl_tex_upload(gpu, &(struct pl_tex_transfer_params) {
                        .tex = lut->tex,
                        .ptr = tmp,
                    });
                }
            } else {
                // Can't use pl_tex_recreate because of `initial_data`
                pl_tex_destroy(gpu, &lut->tex);
                lut->tex = pl_tex_create(gpu, &tex_params);
                ok = lut->tex;
            }

            if (!ok) {
                SH_FAIL(sh, "Failed creating LUT texture!");
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
                                             prefix[params->type], params->comps);
                }
                for (int c = 0; c < params->comps; c++) {
                    switch (params->type) {
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

        lut->method = method;
        lut->type = params->type;
        lut->linear = params->linear;
        lut->width = params->width;
        lut->height = params->height;
        lut->depth = params->depth;
        lut->comps = params->comps;
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

    switch (method) {
    case SH_LUT_TEXTURE: {
        assert(texdim);
        ident_t tex = sh_desc(sh, (struct pl_shader_desc) {
            .desc = {
                .name = "weights",
                .type = PL_DESC_SAMPLED_TEX,
            },
            .binding = {
                .object = lut->tex,
                .sample_mode = params->linear ? PL_TEX_SAMPLE_LINEAR
                                              : PL_TEX_SAMPLE_NEAREST,
            }
        });

        // texelFetch requires GLSL >= 130, so fall back to the linear code
        if (params->linear || gpu->glsl.version < 130) {
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
                .type = params->type,
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
              vartypes[params->type][params->comps - 1], arr_name, size,
              vartypes[params->type][params->comps - 1]);
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

        if (params->linear) {
            pl_assert(dims == 1);
            pl_assert(params->type == PL_VAR_FLOAT);
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

    pl_assert(name);
    ret = name;
    // fall through
error:
    pl_free(tmp);
    return ret;
}

const char *sh_bvec(const pl_shader sh, int dims)
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
