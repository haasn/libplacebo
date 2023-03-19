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
    *sh = (struct pl_shader_t) {
        .log = log,
        .mutable = true,
    };

    for (int i = 0; i < PL_ARRAY_SIZE(sh->buffers); i++)
        sh->buffers[i] = pl_str_builder_alloc(sh);

    // Ensure there's always at least one `tmp` object
    PL_ARRAY_APPEND(sh, sh->tmp, pl_ref_new(NULL));

    if (params)
        sh->res.params = *params;

    return sh;
}

static void sh_obj_deref(pl_shader_obj obj);

static void sh_deref(pl_shader sh)
{
    for (int i = 0; i < sh->tmp.num; i++)
        pl_ref_deref(&sh->tmp.elem[i]);
    for (int i = 0; i < sh->obj.num; i++)
        sh_obj_deref(sh->obj.elem[i]);
}

void pl_shader_free(pl_shader *psh)
{
    pl_shader sh = *psh;
    if (!sh)
        return;

    sh_deref(sh);
    pl_free_ptr(psh);
}

void pl_shader_reset(pl_shader sh, const struct pl_shader_params *params)
{
    sh_deref(sh);

    struct pl_shader_t new = {
        .log = sh->log,
        .mutable = true,

        // Preserve array allocations
        .tmp.elem       = sh->tmp.elem,
        .obj.elem       = sh->obj.elem,
        .vas.elem       = sh->vas.elem,
        .vars.elem      = sh->vars.elem,
        .descs.elem     = sh->descs.elem,
        .consts.elem    = sh->consts.elem,
        .steps.elem     = sh->steps.elem,
    };

    if (params)
        new.res.params = *params;

    // Preserve buffer allocations
    memcpy(new.buffers, sh->buffers, sizeof(new.buffers));
    for (int i = 0; i < PL_ARRAY_SIZE(new.buffers); i++)
        pl_str_builder_reset(new.buffers[i]);

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

    if (sh->type == SH_FRAGMENT) {
        PL_TRACE(sh, "Disabling compute shader because shader is already marked "
                 "as fragment shader");
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
    if (sh->type != SH_COMPUTE || (sh->flexible_work_groups && !flex)) {
        *sh_bw = bw;
        *sh_bh = bh;
        sh->type = SH_COMPUTE;
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
    return sh->type == SH_COMPUTE;
}

bool pl_shader_output_size(const pl_shader sh, int *w, int *h)
{
    if (!sh->output_w || !sh->output_h)
        return false;

    *w = sh->transpose ? sh->output_h : sh->output_w;
    *h = sh->transpose ? sh->output_w : sh->output_h;
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

ident_t sh_var_int(pl_shader sh, const char *name, int val, bool dynamic)
{
    return sh_var(sh, (struct pl_shader_var) {
        .var     = pl_var_int(name),
        .data    = &val,
        .dynamic = dynamic,
    });
}

ident_t sh_var_uint(pl_shader sh, const char *name, unsigned int val, bool dynamic)
{
    return sh_var(sh, (struct pl_shader_var) {
        .var     = pl_var_uint(name),
        .data    = &val,
        .dynamic = dynamic,
    });
}

ident_t sh_var_float(pl_shader sh, const char *name, float val, bool dynamic)
{
    return sh_var(sh, (struct pl_shader_var) {
        .var     = pl_var_float(name),
        .data    = &val,
        .dynamic = dynamic,
    });
}

static void merge_access(enum pl_desc_access *a, enum pl_desc_access b)
{
    if (*a != b)
        *a = PL_DESC_ACCESS_READWRITE;
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
            if (sh->descs.elem[i].binding.object == sd.binding.object) {
                merge_access(&sh->descs.elem[i].desc.access, sd.desc.access);
                sh->descs.elem[i].memory |= sd.memory;
                return (ident_t) sh->descs.elem[i].desc.name;
            }
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
        if (!sc.compile_time || gpu->limits.array_size_constants) {
            sc.data = pl_memdup(SH_TMP(sh), sc.data, pl_var_type_size(sc.type));
            PL_ARRAY_APPEND(sh, sh->consts, sc);
            return (ident_t) sc.name;
        }
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
    if (pl_tex_params_dimension(tex->params) != 2) {
        SH_FAIL(sh, "Failed binding texture '%s': not a 2D texture!", name);
        return NULL;
    }

    if (!tex->params.sampleable) {
        SH_FAIL(sh, "Failed binding texture '%s': texture not sampleable!", name);
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

void sh_describef(pl_shader sh, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    sh_describe(sh, pl_vasprintf(SH_TMP(sh), fmt, ap));
    va_end(ap);
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

    if (sub->type == SH_COMPUTE) {
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
    pl_str_builder_concat(sh->buffers[SH_BUF_PRELUDE], sub->buffers[SH_BUF_PRELUDE]);
    pl_str_builder_concat(sh->buffers[SH_BUF_HEADER], sub->buffers[SH_BUF_HEADER]);

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
    pl_str_builder_concat(sh->buffers[SH_BUF_HEADER], sub->buffers[SH_BUF_BODY]);
    GLSLH("%s\n}\n\n", retvals[sub->res.output]);

    // Ref all objects
    PL_ARRAY_CONCAT(sh, sh->obj, sub->obj);
    for (int i = 0; i < sub->obj.num; i++)
        pl_rc_ref(&sub->obj.elem[i]->rc);

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

pl_str_builder sh_finalize_internal(pl_shader sh)
{
    pl_assert(sh->mutable);
    if (sh->failed)
        return NULL;

    // Padding for readability
    GLSLP("\n");

    // Concatenate everything onto the prelude to form the final output
    pl_str_builder_concat(sh->buffers[SH_BUF_PRELUDE], sh->buffers[SH_BUF_HEADER]);

    sh->res.name = sh_fresh(sh, "main");
    if (sh->res.input == PL_SHADER_SIG_SAMPLER) {
        pl_assert(sh->sampler_prefix);
        GLSLP("%s %s(%c%s src_tex, vec2 tex_coord) {\n",
              outsigs[sh->res.output], sh->res.name,
              sh->sampler_prefix, samplers2D[sh->sampler_type]);
    } else {
        GLSLP("%s %s(%s) {\n", outsigs[sh->res.output], sh->res.name, insigs[sh->res.input]);
    }

    pl_str_builder_concat(sh->buffers[SH_BUF_PRELUDE], sh->buffers[SH_BUF_BODY]);
    pl_str_builder_concat(sh->buffers[SH_BUF_PRELUDE], sh->buffers[SH_BUF_FOOTER]);
    GLSLP("%s\n}\n\n", retvals[sh->res.output]);

    // Generate the pretty description
    sh->res.description = "(unknown shader)";
    if (sh->steps.num) {
        // Reuse this builder
        pl_str_builder desc = sh->buffers[SH_BUF_BODY];
        pl_str_builder_reset(desc);

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
                pl_str_builder_const_str(desc, ", ");
            pl_str_builder_str0(desc, step);
            if (count > 1)
                pl_str_builder_printf_c(desc, " x%d", count);
        }

        sh->res.description = (char *) pl_str_builder_exec(desc).buf;
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
    sh->mutable = false;
    return sh->buffers[SH_BUF_PRELUDE];
}

const struct pl_shader_res *pl_shader_finalize(pl_shader sh)
{
    pl_str_builder glsl = NULL;
    if (sh->mutable) {
        glsl = sh_finalize_internal(sh);
        if (!glsl)
            return NULL;
    }

    pl_assert(!sh->mutable);
    if (!sh->res.glsl)
        sh->res.glsl = (char *) pl_str_builder_exec(glsl).buf;

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

static void sh_obj_deref(pl_shader_obj obj)
{
    if (!pl_rc_deref(&obj->rc))
        return;

    if (obj->uninit)
        obj->uninit(obj->gpu, obj->priv);

    pl_free(obj);
}

void pl_shader_obj_destroy(pl_shader_obj *ptr)
{
    pl_shader_obj obj = *ptr;
    if (!obj)
        return;

    sh_obj_deref(obj);
    *ptr = NULL;
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
        pl_rc_init(&obj->rc);
        obj->gpu = SH_GPU(sh);
        obj->type = type;
        obj->priv = pl_zalloc(obj, priv_size);
        obj->uninit = uninit;
    }

    PL_ARRAY_APPEND(sh, sh->obj, obj);
    pl_rc_ref(&obj->rc);

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
