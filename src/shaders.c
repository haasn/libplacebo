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
        .buffer_head = { sh->buffer_head.start },
        .buffer_body = { sh->buffer_body.start },
        .current_binding = sh->current_binding,
        .res = {
            .variables      = sh->res.variables,
            .descriptors    = sh->res.descriptors,
            .vertex_attribs = sh->res.vertex_attribs,
        },
    };

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

bool pl_shader_is_compute(const struct pl_shader *sh)
{
    bool ret = true;
    for (int i = 0; i < PL_ARRAY_SIZE(sh->res.compute_work_groups); i++)
        ret &= !!sh->res.compute_work_groups[i];
    return ret;
}

ident_t sh_fresh(struct pl_shader *sh, const char *name)
{
    return talloc_asprintf(sh->tmp, "_" PLACEHOLDER "_%s_%d",
                           PL_DEF(name, "var"), sh->fresh++);
}

ident_t sh_var(struct pl_shader *sh, struct pl_shader_var sv)
{
    sv.var.name = sh_fresh(sh, sv.var.name);
    sv.data = talloc_memdup(sh->tmp, sv.data, ra_var_host_layout(0, &sv.var).size);
    TARRAY_APPEND(sh, sh->res.variables, sh->res.num_variables, sv);
    return sv.var.name;
}

ident_t sh_desc(struct pl_shader *sh, struct pl_shader_desc sd)
{
    assert(sh->ra);
    int namespace = ra_desc_namespace(sh->ra, sd.desc.type);

    sd.desc.name = sh_fresh(sh, sd.desc.name);
    sd.desc.binding = sh->current_binding[namespace]++;

    TARRAY_APPEND(sh, sh->res.descriptors, sh->res.num_descriptors, sd);
    return sd.desc.name;
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
        // Clockwise from top left
        { rc->x0, rc->y0 },
        { rc->x1, rc->y0 },
        { rc->x1, rc->y1 },
        { rc->x0, rc->y1 },
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
    return va.attr.name;
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
        .binding = tex,
    });

    if (out_pos) {
        float xy0[2] = {0};
        float xy1[2] = {tex->params.w, tex->params.h};
        pl_transform2x2_apply(tf, xy0);
        pl_transform2x2_apply(tf, xy1);
        *out_pos = sh_attr_vec2(sh, "pos", &(struct pl_rect2df) {
            .x0 = xy0[0], .y0 = xy0[1],
            .x1 = xy1[0], .y1 = xy1[1],
        });
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

void pl_shader_append(struct pl_shader *sh, struct bstr *buf,
                      const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    bstr_xappend_vasprintf(sh, buf, fmt, ap);
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

void sh_rename_vars(struct pl_shader *sh, int namespace)
{
    char buf[sizeof(PLACEHOLDER)] = {0};
    int num = snprintf(buf, sizeof(buf), "%d", namespace);

    // Clear out the remainder of `buf` with a safe character
    for (int i = num; i < sizeof(buf) - 1; i++)
        buf[i] = 'z';

    // This is safe, because we never shrink or splice the buffers; and they're
    // always null-terminated (for the same reason we can directly return them)
    rename_str(sh->buffer_head.start, buf);
    rename_str(sh->buffer_body.start, buf);

    // These are all safe to directly mutate, because we've allocated all
    // identifiers via `sh_fresh`.
    rename_str((char *) sh->res.name, buf);
    for (int i = 0; i < sh->res.num_vertex_attribs; i++)
        rename_str((char *) sh->res.vertex_attribs[i].attr.name, buf);
    for (int i = 0; i < sh->res.num_variables; i++)
        rename_str((char *) sh->res.variables[i].var.name, buf);
    for (int i = 0; i < sh->res.num_descriptors; i++)
        rename_str((char *) sh->res.descriptors[i].desc.name, buf);
}

const struct pl_shader_res *pl_shader_finalize(struct pl_shader *sh)
{
    if (!sh->mutable) {
        PL_WARN(sh, "Attempted to finalize a shader twice?");
        return &sh->res;
    }

    static const char *outsigs[] = {
        [PL_SHADER_SIG_NONE]  = "void",
        [PL_SHADER_SIG_COLOR] = "vec4",
    };

    static const char *insigs[] = {
        [PL_SHADER_SIG_NONE]  = "",
        [PL_SHADER_SIG_COLOR] = "vec4 color",
    };

    ident_t name = sh->res.name = sh_fresh(sh, "main");
    GLSLH("%s %s(%s) {\n", outsigs[sh->res.output], name, insigs[sh->res.input]);

    if (sh->buffer_body.len) {
        GLSLH("%s", sh->buffer_body.start);
        sh->buffer_body.len = 0;
        sh->buffer_body.start[0] = '\0'; // sanity, and for sh_rename_vars
    }

    switch (sh->res.output) {
    case PL_SHADER_SIG_NONE: break;
    case PL_SHADER_SIG_COLOR:
        GLSLH("return color;\n");
        break;
    }

    GLSLH("}\n");
    sh_rename_vars(sh, sh->namespace);

    // Update the result pointer
    sh->res.glsl = sh->buffer_head.start;
    sh->mutable = false;

    return &sh->res;
}

bool sh_require_input(struct pl_shader *sh, enum pl_shader_sig insig)
{
    if (!sh->mutable) {
        PL_ERR(sh, "Attempted to modify an immutable shader!");
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
    return true;
}
