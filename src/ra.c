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

#include "common.h"
#include "ra.h"

void ra_destroy(const struct ra **ra)
{
    if (!*ra)
        return;

    (*ra)->impl->destroy(*ra);
    *ra = NULL;
}

bool ra_format_is_ordered(const struct ra_format *fmt)
{
    bool ret = true;
    for (int i = 0; i < fmt->num_components; i++)
        ret &= fmt->component_index[i] == i;
    return ret;
}

bool ra_format_is_regular(const struct ra_format *fmt)
{
    int bits = 0;
    for (int i = 0; i < fmt->num_components; i++) {
        if (fmt->component_index[i] != i || fmt->component_pad[i])
            return false;
        bits += fmt->component_depth[i];
    }

    return bits == fmt->texel_size * 8;
}

const struct ra_format *ra_find_texture_format(const struct ra *ra,
                                               enum ra_fmt_type type,
                                               int num_components,
                                               int bits_per_component,
                                               bool regular)
{
    for (int n = 0; n < ra->num_formats; n++) {
        const struct ra_format *fmt = ra->formats[n];
        if (fmt->type != type || fmt->num_components != num_components)
            continue;
        if (regular && !ra_format_is_regular(fmt))
            continue;

        for (int i = 0; i < fmt->num_components; i++) {
            if (fmt->component_depth[i] != bits_per_component)
                goto next_fmt;
        }

        return fmt;

next_fmt: ; // equivalent to `continue`
    }

    // ran out of formats
    return NULL;
}

const struct ra_format *ra_find_named_format(const struct ra *ra,
                                             const char *name)
{
    if (!name)
        return NULL;

    for (int i = 0; i < ra->num_formats; i++) {
        const struct ra_format *fmt = ra->formats[i];
        if (strcmp(name, fmt->name) == 0)
            return fmt;
    }

    // ran out of formats
    return NULL;
}

const struct ra_tex *ra_tex_create(const struct ra *ra,
                                   const struct ra_tex_params *params)
{
    switch (ra_tex_params_dimension(*params)) {
    case 1:
        assert(params->w > 0);
        assert(params->w <= ra->limits.max_tex_1d_dim);
        break;
    case 2:
        assert(params->w > 0 && params->h > 0);
        assert(params->w <= ra->limits.max_tex_2d_dim);
        assert(params->h <= ra->limits.max_tex_2d_dim);
        break;
    case 3:
        assert(params->w > 0 && params->h > 0 && params->d > 0);
        assert(params->w <= ra->limits.max_tex_3d_dim);
        assert(params->h <= ra->limits.max_tex_3d_dim);
        assert(params->d <= ra->limits.max_tex_3d_dim);
        break;
    default: abort();
    }

    const struct ra_format *fmt = params->format;
    assert(fmt);
    assert(fmt->texture_format);
    assert(fmt->sampleable || !params->sampleable);
    assert(fmt->renderable || !params->renderable);
    assert(fmt->storable   || !params->storage_image);
    assert(fmt->blittable  || !params->blit_src);
    assert(fmt->blittable  || !params->blit_dst);
    assert(fmt->linear_filterable || params->sample_mode != RA_TEX_SAMPLE_LINEAR);

    return ra->impl->tex_create(ra, params);
}

void ra_tex_destroy(const struct ra *ra, const struct ra_tex **tex)
{
    if (!*tex)
        return;

    ra->impl->tex_destroy(ra, *tex);
    *tex = NULL;
}

void ra_tex_clear(const struct ra *ra, const struct ra_tex *dst,
                  const float color[4])
{
    assert(dst->params.blit_dst);

    ra->impl->tex_clear(ra, dst, color);
}

void ra_tex_blit(const struct ra *ra,
                 const struct ra_tex *dst, const struct ra_tex *src,
                 struct pl_rect3d dst_rc, struct pl_rect3d src_rc)
{
    assert(src->params.format->texel_size == dst->params.format->texel_size);
    assert(src->params.blit_src);
    assert(dst->params.blit_dst);
    assert(src_rc.x0 >= 0 && src_rc.x0 < src->params.w);
    assert(src_rc.y0 >= 0 && src_rc.y0 < src->params.h);
    assert(src_rc.x1 > 0 && src_rc.x1 <= src->params.w);
    assert(src_rc.y1 > 0 && src_rc.y1 <= src->params.h);
    assert(dst_rc.x0 >= 0 && dst_rc.x0 < dst->params.w);
    assert(dst_rc.y0 >= 0 && dst_rc.y0 < dst->params.h);
    assert(dst_rc.x1 > 0 && dst_rc.x1 <= dst->params.w);
    assert(dst_rc.y1 > 0 && dst_rc.y1 <= dst->params.h);

    ra->impl->tex_blit(ra, dst, src, dst_rc, src_rc);
}

static void check_tex_transfer(const struct ra *ra,
                               const struct ra_tex_transfer_params *params)
{
#ifndef NDEBUG
    struct ra_tex *tex = params->tex;
    struct pl_rect3d rc = params->rc;
    switch (ra_tex_params_dimension(tex->params))
    {
    case 3:
        assert(rc.z1 > rc.z0);
        assert(rc.z0 >= 0 && rc.z0 <  tex->params.d);
        assert(rc.z1 >  0 && rc.z1 <= tex->params.d);
        assert(params->stride_h >= pl_rect_h(rc));
        // fall through
    case 2:
        assert(rc.y1 > rc.y0);
        assert(rc.y0 >= 0 && rc.y0 <  tex->params.h);
        assert(rc.y1 >  0 && rc.y1 <= tex->params.h);
        assert(params->stride_w >= pl_rect_w(rc));
        // fall through
    case 1:
        assert(rc.x1 > rc.x0);
        assert(rc.x0 >= 0 && rc.x0 <  tex->params.w);
        assert(rc.x1 >  0 && rc.x1 <= tex->params.w);
        break;
    default: abort();
    }

    assert(!params->buf ^ !params->src); // exactly one
    if (params->buf) {
        struct ra_buf *buf = params->buf;
        int num;
        assert(buf->params.type == RA_BUF_TEX_TRANSFER);
        switch (ra_tex_params_dimension(tex->params)) {
            case 1: num = pl_rect_w(rc); break;
            case 2: num = pl_rect_h(rc) * params->stride_w; break;
            case 3: num = pl_rect_d(rc) * params->stride_h * params->stride_w; break;
        }
        assert(params->buf_offset == PL_ALIGN2(params->buf_offset, 4));
        assert(params->buf_offset + num <= buf->params.size);
    }
#endif
}

bool ra_tex_upload(const struct ra *ra,
                   const struct ra_tex_transfer_params *params)
{
    struct ra_tex *tex = params->tex;
    assert(tex->params.host_mutable);
    check_tex_transfer(ra, params);

    return ra->impl->tex_upload(ra, params);
}

bool ra_tex_download(const struct ra *ra,
                     const struct ra_tex_transfer_params *params)
{
    struct ra_tex *tex = params->tex;
    assert(tex->params.host_fetchable);
    check_tex_transfer(ra, params);

    return ra->impl->tex_download(ra, params);
}

const struct ra_buf *ra_buf_create(const struct ra *ra,
                                   const struct ra_buf_params *params)
{
    assert(params->size >= 0);
    switch (params->type) {
    case RA_BUF_TEX_TRANSFER:
        assert(ra->limits.max_xfer_size);
        assert(params->size <= ra->limits.max_xfer_size);
        break;
    case RA_BUF_UNIFORM:
        assert(ra->limits.max_ubo_size);
        assert(params->size <= ra->limits.max_ubo_size);
        break;
    case RA_BUF_STORAGE:
        assert(ra->limits.max_ssbo_size);
        assert(params->size <= ra->limits.max_ssbo_size);
        break;
    default: abort();
    }

    const struct ra_buf *buf = ra->impl->buf_create(ra, params);
    assert(buf->data || !params->host_mapped);
    return buf;
}

void ra_buf_destroy(const struct ra *ra, const struct ra_buf **buf)
{
    if (!*buf)
        return;

    ra->impl->buf_destroy(ra, *buf);
    *buf = NULL;
}

void ra_buf_update(const struct ra *ra, const struct ra_buf *buf,
                   size_t buf_offset, const void *data, size_t size)
{
    assert(buf->params.host_mutable);
    assert(buf_offset + size <= buf->params.size);
    assert(buf_offset == PL_ALIGN2(buf_offset, 4));
    ra->impl->buf_update(ra, buf, buf_offset, data, size);
}

bool ra_buf_poll(const struct ra *ra, const struct ra_buf *buf, uint64_t t)
{
    return ra->impl->buf_poll ? ra->impl->buf_poll(ra, buf, t) : false;
}

size_t ra_var_type_size(enum ra_var_type type)
{
    switch (type) {
    case RA_VAR_SINT:  return sizeof(int);
    case RA_VAR_UINT:  return sizeof(unsigned int);
    case RA_VAR_FLOAT: return sizeof(float);
    default: abort();
    }
}

#define MAX_DIM 4

const char *ra_var_glsl_type_name(struct ra_var var)
{
    static const char *types[RA_VAR_TYPE_COUNT][MAX_DIM+1][MAX_DIM+1] = {
    // float vectors
    [RA_VAR_FLOAT][1][1] = "float",
    [RA_VAR_FLOAT][1][2] = "vec2",
    [RA_VAR_FLOAT][1][3] = "vec3",
    [RA_VAR_FLOAT][1][4] = "vec4",
    // float matrices
    [RA_VAR_FLOAT][2][2] = "mat2",
    [RA_VAR_FLOAT][2][3] = "mat2x3",
    [RA_VAR_FLOAT][2][4] = "mat2x4",
    [RA_VAR_FLOAT][3][2] = "mat3x2",
    [RA_VAR_FLOAT][3][3] = "mat3",
    [RA_VAR_FLOAT][3][4] = "mat3x4",
    [RA_VAR_FLOAT][4][2] = "mat4x2",
    [RA_VAR_FLOAT][4][3] = "mat4x3",
    [RA_VAR_FLOAT][4][4] = "mat4",
    // integer vectors
    [RA_VAR_SINT][1][1] = "int",
    [RA_VAR_SINT][1][2] = "ivec2",
    [RA_VAR_SINT][1][3] = "ivec3",
    [RA_VAR_SINT][1][4] = "ivec4",
    // unsigned integer vectors
    [RA_VAR_UINT][1][1] = "uint",
    [RA_VAR_UINT][1][2] = "uvec2",
    [RA_VAR_UINT][1][3] = "uvec3",
    [RA_VAR_UINT][1][4] = "uvec4",
    };

    if (var.dim_v > MAX_DIM || var.dim_m > MAX_DIM)
        return NULL;

    return types[var.type][var.dim_m][var.dim_v];
}

#define RA_VAR_FV(TYPE, M, V)                           \
    struct ra_var ra_var_##TYPE(const char *name) {     \
        return (struct ra_var) {                        \
            .name  = name,                              \
            .type  = RA_VAR_FLOAT,                      \
            .dim_m = M,                                 \
            .dim_v = V,                                 \
        };                                              \
    }

RA_VAR_FV(float, 1, 1)
RA_VAR_FV(vec2,  1, 2)
RA_VAR_FV(vec3,  1, 3)
RA_VAR_FV(vec4,  1, 4)
RA_VAR_FV(mat2,  2, 2)
RA_VAR_FV(mat3,  3, 3)
RA_VAR_FV(mat4,  4, 4)

struct ra_var_layout ra_var_host_layout(size_t offset, struct ra_var var)
{
    size_t col_size = ra_var_type_size(var.type) * var.dim_v;
    return (struct ra_var_layout) {
        .offset = offset,
        .stride = col_size,
        .size   = col_size * var.dim_m,
    };
}

struct ra_var_layout ra_buf_uniform_layout(const struct ra *ra, size_t offset,
                                           const struct ra_var *var)
{
    if (ra->limits.max_ubo_size) {
        return ra->impl->buf_uniform_layout(ra, offset, var);
    } else {
        return (struct ra_var_layout) {0};
    }
}

struct ra_var_layout ra_buf_storage_layout(const struct ra *ra, size_t offset,
                                           const struct ra_var *var)
{
    if (ra->limits.max_ssbo_size) {
        return ra->impl->buf_storage_layout(ra, offset, var);
    } else {
        return (struct ra_var_layout) {0};
    }
}

struct ra_var_layout ra_push_constant_layout(const struct ra *ra, size_t offset,
                                             const struct ra_var *var)
{
    if (ra->limits.max_pushc_size) {
        return ra->impl->push_constant_layout(ra, offset, var);
    } else {
        return (struct ra_var_layout) {0};
    }
}

int ra_desc_namespace(const struct ra *ra, enum ra_desc_type type)
{
    return ra->impl->desc_namespace(ra, type);
}

const char *ra_desc_access_glsl_name(enum ra_desc_access mode)
{
    switch (mode) {
    case RA_DESC_ACCESS_READWRITE: return "";
    case RA_DESC_ACCESS_READONLY:  return "readonly";
    case RA_DESC_ACCESS_WRITEONLY: return "writeonly";
    default: abort();
    }
}

const struct ra_renderpass *ra_renderpass_create(const struct ra *ra,
                                const struct ra_renderpass_params *params)
{
    assert(params->glsl_shader);
    switch(params->type) {
    case RA_RENDERPASS_RASTER:
        assert(params->vertex_shader);
        for (int i = 0; i < params->num_vertex_attribs; i++) {
            struct ra_vertex_attrib va = params->vertex_attribs[i];
            assert(va.name);
            assert(va.fmt);
            assert(va.fmt->vertex_format);
            assert(va.offset + va.fmt->texel_size <= params->vertex_stride);
        }

        assert(params->target_format);
        assert(params->target_format->renderable);
        assert(params->target_format->blendable || !params->enable_blend);
        break;
    case RA_RENDERPASS_COMPUTE:
        assert(ra->caps & RA_CAP_COMPUTE);
        break;
    default: abort();
    }

    for (int i = 0; i < params->num_variables; i++) {
        assert(ra->caps & RA_CAP_INPUT_VARIABLES);
        struct ra_var var = params->variables[i];
        assert(var.name);
        assert(ra_var_glsl_type_name(var));
    }

    for (int i = 0; i < params->num_descriptors; i++) {
        struct ra_desc desc = params->descriptors[i];
        assert(desc.name);
        // TODO: enforce disjoint bindings if possible?
    }

    assert(params->push_constants_size <= ra->limits.max_pushc_size);
    assert(params->push_constants_size == PL_ALIGN2(params->push_constants_size, 4));

    return ra->impl->renderpass_create(ra, params);
}

void ra_renderpass_destroy(const struct ra *ra,
                           const struct ra_renderpass **pass)
{
    if (!*pass)
        return;

    ra->impl->renderpass_destroy(ra, *pass);
    *pass = NULL;
}

void ra_renderpass_run(const struct ra *ra,
                       const struct ra_renderpass_run_params *params)
{
#ifndef NDEBUG
    struct ra_renderpass *pass = params->pass;
    for (int i = 0; i < params->num_desc_updates; i++) {
        struct ra_desc_update du = params->desc_updates[i];
        assert(du.index >= 0 && du.index < pass->params.num_descriptors);

        struct ra_desc desc = pass->params.descriptors[du.index];
        switch (desc.type) {
        case RA_DESC_SAMPLED_TEX: {
            struct ra_tex *tex = du.binding;
            assert(tex->params.sampleable);
            break;
        }
        case RA_DESC_STORAGE_IMG: {
            struct ra_tex *tex = du.binding;
            assert(tex->params.storage_image);
            break;
        }
        case RA_DESC_BUF_UNIFORM: {
            struct ra_buf *buf = du.binding;
            assert(buf->params.type == RA_BUF_UNIFORM);
            break;
        }
        case RA_DESC_BUF_STORAGE: {
            struct ra_buf *buf = du.binding;
            assert(buf->params.type == RA_BUF_STORAGE);
            break;
        }
        default: abort();
        }
    }

    for (int i = 0; i < params->num_var_updates; i++) {
        struct ra_var_update vu = params->var_updates[i];
        assert(vu.index >= 0 && vu.index < pass->params.num_variables);
        assert(vu.data);
    }

    assert(params->push_constants || !pass->params.push_constants_size);

    switch (pass->params.type) {
    case RA_RENDERPASS_RASTER: {
        struct ra_tex *tex = params->target;
        assert(ra_tex_params_dimension(tex->params) == 2);
        assert(tex->params.format == pass->params.target_format);
        assert(tex->params.renderable);
        break;
    }
    case RA_RENDERPASS_COMPUTE:
        for (int i = 0; i < PL_ARRAY_SIZE(params->compute_groups); i++) {
            assert(params->compute_groups[i] >= 0);
            assert(params->compute_groups[i] <= ra->limits.max_dispatch[i]);
        }
        break;
    default: abort();
    }
#endif

    return ra->impl->renderpass_run(ra, params);
}

struct ra_timer *ra_timer_create(const struct ra *ra)
{
    return ra->impl->timer_create ? ra->impl->timer_create(ra) : NULL;
}

void ra_timer_destroy(const struct ra *ra, struct ra_timer **timer)
{
    if (!*timer)
        return;

    ra->impl->timer_destroy(ra, *timer);
    *timer = NULL;
}

void ra_timer_start(const struct ra *ra, struct ra_timer *timer)
{
    if (timer)
        ra->impl->timer_start(ra, timer);
}

uint64_t ra_timer_stop(const struct ra *ra, struct ra_timer *timer)
{
    return timer ? ra->impl->timer_stop(ra, timer) : 0;
}

void ra_flush(const struct ra *ra)
{
    if (ra->impl->flush)
        ra->impl->flush(ra);
}
