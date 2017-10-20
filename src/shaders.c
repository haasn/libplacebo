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
#include "bstr/bstr.h"

#include "common.h"
#include "context.h"
#include "shaders.h"

// Represents a blank placeholder for the purposes of namespace substitution.
// This is picked to be a literal string that is impossible to ever occur in
// valid code.
#define PLACEHOLDER "\1\2\3"

struct pl_shader *pl_shader_alloc(struct pl_context *ctx,
                                  const struct ra *ra)
{
    assert(ctx);
    struct pl_shader *sh = talloc_ptrtype(ctx, sh);
    *sh = (struct pl_shader) {
        .ctx = ctx,
        .ra = ra,
        .mutable = true,
        .tmp = talloc_new(sh),
    };

    if (ra) {
        int num_namespaces = ra_desc_namespace(ra, 0);
        sh->current_binding = talloc_zero_array(sh, int, num_namespaces);
    }

    return sh;
}

void pl_shader_free(struct pl_shader **sh)
{
    TA_FREEP(sh);
}

void pl_shader_reset(struct pl_shader *sh)
{
    struct pl_shader new = {
        .ctx = sh->ctx,
        .ra  = sh->ra,
        .tmp = talloc_new(sh),
        .mutable = true,

        // Preserve array allocations
        .current_binding = sh->current_binding,
        .identifiers = sh->identifiers,
        .res = {
            .variables      = sh->res.variables,
            .descriptors    = sh->res.descriptors,
            .vertex_attribs = sh->res.vertex_attribs,
        },
    };

    // Preserve buffer allocations
    for (int i = 0; i < PL_ARRAY_SIZE(new.buffers); i++)
        new.buffers[i] = (struct bstr) { .start = sh->buffers[i].start };

    // Clear the bindings array
    if (new.current_binding) {
        assert(sh->ra);
        int num_namespaces = ra_desc_namespace(sh->ra, 0);
        for (int i = 0; i < num_namespaces; i++)
            new.current_binding[i] = 0;
    }

    talloc_free(sh->tmp);
    *sh = new;
}

bool sh_try_compute(struct pl_shader *sh, int bw, int bh, bool flex, size_t mem)
{
    assert(bw && bh);
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
    assert(!flex && !sh->flexible_work_groups);
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

    // just in case...
    for (int i = 0; i < sh->num_identifiers; i++)
        res ^= bstr_hash64(bstr0(sh->identifiers[i]));

    return res;
}

ident_t sh_fresh(struct pl_shader *sh, const char *name)
{
    ident_t ident = talloc_asprintf(sh->tmp, "_" PLACEHOLDER "_%s_%d",
                                    PL_DEF(name, "var"), sh->fresh++);
    TARRAY_APPEND(sh, sh->identifiers, sh->num_identifiers, ident);
    return ident;
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
    assert(sh->ra);
    int namespace = ra_desc_namespace(sh->ra, sd.desc.type);

    sd.desc.name = sh_fresh(sh, sd.desc.name);
    sd.desc.binding = sh->current_binding[namespace]++;

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
            .offset   = sh->current_va_offset,
            .location = sh->current_va_location,
        },
        .data = { &data[0], &data[2], &data[4], &data[6] },
    };

    TARRAY_APPEND(sh, sh->res.vertex_attribs, sh->res.num_vertex_attribs, va);
    sh->current_va_offset += sizeof(float[2]);
    sh->current_va_location += 1; // vec2 always consumes one location
    return (ident_t) va.attr.name;
}

ident_t sh_bind(struct pl_shader *sh, const struct ra_tex *tex,
                const char *name, const struct pl_transform2x2 *tf,
                ident_t *out_pos, ident_t *out_size, ident_t *out_pt)
{
    if (!sh->ra) {
        PL_ERR(sh, "Failed binding texture '%s': No RA available!", name);
        return NULL;
    }

    assert(ra_tex_params_dimension(tex->params) == 2);
    ident_t itex = sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name = name,
            .type = RA_DESC_SAMPLED_TEX,
        },
        .object = tex,
    });

    if (out_pos) {
        float xy0[2] = {0};
        float xy1[2] = {tex->params.w, tex->params.h};
        pl_transform2x2_apply(tf, xy0);
        pl_transform2x2_apply(tf, xy1);
        ident_t pos_raw = sh_attr_vec2(sh, "pos_raw", &(struct pl_rect2df) {
            .x0 = xy0[0] / tex->params.w, .y0 = xy0[1] / tex->params.w,
            .x1 = xy1[0] / tex->params.h, .y1 = xy1[1] / tex->params.h,
        });

        *out_pos = sh_fresh(sh, "pos");
        GLSL("vec2 %s = %s;\n", *out_pos, pos_raw);
    }

    if (out_size) {
        *out_size = sh_var(sh, (struct pl_shader_var) {
            .var  = ra_var_vec2("size"),
            .data = &(float[2]) {tex->params.w, tex->params.h},
        });
    }

    if (out_pt) {
        *out_pt = sh_var(sh, (struct pl_shader_var) {
            .var  = ra_var_vec2("pt"),
            .data = &(float[2]) {1.0 / tex->params.w, 1.0 / tex->params.h},
        });
    }

    return itex;
}

void pl_shader_append(struct pl_shader *sh, enum pl_shader_buf buf,
                      const char *fmt, ...)
{
    assert(buf >= 0 && buf < SH_BUF_COUNT);

    va_list ap;
    va_start(ap, fmt);
    bstr_xappend_vasprintf(sh, &sh->buffers[buf], fmt, ap);
    va_end(ap);
}

// Performs the free variable rename. `buf` must point to a buffer with at
// least strlen(PLACEHOLDER) valid replacement characters.
static void rename_str(char *str, const char *buf)
{
    if (!str)
        return;

    while ((str = strstr(str, PLACEHOLDER))) {
        for (int i = 0; i < sizeof(PLACEHOLDER) - 1; i++)
            str[i] = buf[i];
    }
}

// Replace all of the free variables in the glsl and input list by literally
// string replacing it with an encoded representation of the given namespace
static void sh_rename_vars(struct pl_shader *sh, int namespace)
{
    char buf[sizeof(PLACEHOLDER)] = {0};
    int num = snprintf(buf, sizeof(buf), "%d", namespace);

    // Clear out the remainder of `buf` with a safe character
    for (int i = num; i < sizeof(buf) - 1; i++)
        buf[i] = 'z';

    // This is safe, because we never shrink or splice the buffers; and they're
    // always null-terminated (for the same reason we can directly return them)
    for (int i = 0; i < PL_ARRAY_SIZE(sh->buffers); i++)
        rename_str(sh->buffers[i].start, buf);

    // Rename all of the generated identifiers as well. This will also re-name
    // all references to them (in pl_shader_res etc.), which is basically what
    // we want to happen
    for (int i = 0; i < sh->num_identifiers; i++)
        rename_str(sh->identifiers[i], buf);
}

// Finish the current shader body and return its function name
static ident_t sh_split(struct pl_shader *sh)
{
    assert(sh->mutable);

    static const char *outsigs[] = {
        [PL_SHADER_SIG_NONE]  = "void",
        [PL_SHADER_SIG_COLOR] = "vec4",
    };

    static const char *insigs[] = {
        [PL_SHADER_SIG_NONE]  = "",
        [PL_SHADER_SIG_COLOR] = "vec4 color",
    };

    // Concatenate the body onto the head as a new function
    ident_t name = sh_fresh(sh, "main");
    GLSLH("%s %s(%s) {\n", outsigs[sh->res.output], name, insigs[sh->res.input]);

    if (sh->buffers[SH_BUF_BODY].len) {
        bstr_xappend(sh, &sh->buffers[SH_BUF_HEADER], sh->buffers[SH_BUF_BODY]);
        sh->buffers[SH_BUF_BODY].len = 0;
        sh->buffers[SH_BUF_BODY].start[0] = '\0'; // for sanity / efficiency
    }

    switch (sh->res.output) {
    case PL_SHADER_SIG_NONE: break;
    case PL_SHADER_SIG_COLOR:
        GLSLH("return color;\n");
        break;
    }

    GLSLH("}\n");
    sh_rename_vars(sh, sh->namespace);
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
        assert(!sh->res.input);
        sh->res.input = insig;
    } else if (sh->res.output != insig) {
        PL_ERR(sh, "Illegal sequence of shader operations! Current output "
               "signature is '%s', but called operation expects '%s'!",
               names[sh->res.output], names[insig]);
        return false;
    }

    // All of our shaders end up returning a vec4 color
    sh->res.output = PL_SHADER_SIG_COLOR;
    sh->output_w = w;
    sh->output_h = h;
    return true;
}

void pl_shader_obj_destroy(struct pl_shader_obj **ptr)
{
    struct pl_shader_obj *obj = *ptr;
    if (!obj)
        return;

    if (obj->ra) {
        ra_buf_destroy(obj->ra, &obj->buf);
        ra_tex_destroy(obj->ra, &obj->tex);
    }

    *ptr = NULL;
    talloc_free(obj);
}

bool sh_require_obj(struct pl_shader *sh, struct pl_shader_obj **ptr,
                    enum pl_shader_obj_type type)
{
    if (!ptr)
        return false;

    struct pl_shader_obj *obj = *ptr;
    if (obj && obj->ra != sh->ra) {
        PL_ERR(sh, "Passed pl_shader_obj belongs to different RA!");
        return false;
    }

    if (obj && obj->type != type) {
        PL_ERR(sh, "Passed pl_shader_obj of wrong type! Shader objects must "
               "always be used with the same type of shader.");
        return false;
    }

    if (!obj) {
        obj = talloc_zero(NULL, struct pl_shader_obj);
        obj->ra = sh->ra;
        obj->type = type;
    }

    *ptr = obj;
    return true;
}
