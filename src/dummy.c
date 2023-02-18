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

#include <limits.h>
#include <string.h>

#include "gpu.h"

#include <libplacebo/dummy.h>

const struct pl_gpu_dummy_params pl_gpu_dummy_default_params = { PL_GPU_DUMMY_DEFAULTS };
static const struct pl_gpu_fns pl_fns_dummy;

struct priv {
    struct pl_gpu_fns impl;
    struct pl_gpu_dummy_params params;
};

pl_gpu pl_gpu_dummy_create(pl_log log, const struct pl_gpu_dummy_params *params)
{
    params = PL_DEF(params, &pl_gpu_dummy_default_params);

    struct pl_gpu_t *gpu = pl_zalloc_obj(NULL, gpu, struct priv);
    gpu->log = log;
    gpu->glsl = params->glsl;
    gpu->limits = params->limits;

    struct priv *p = PL_PRIV(gpu);
    p->impl = pl_fns_dummy;
    p->params = *params;

    // Forcibly override these, because we know for sure what the values are
    gpu->limits.align_tex_xfer_pitch = 1;
    gpu->limits.align_tex_xfer_offset = 1;
    gpu->limits.align_vertex_stride = 1;

    // Set up the dummy formats, add one for each possible format type that we
    // can represent on the host
    PL_ARRAY(pl_fmt) formats = {0};
    for (enum pl_fmt_type type = 1; type < PL_FMT_TYPE_COUNT; type++) {
        for (int comps = 1; comps <= 4; comps++) {
            for (int depth = 8; depth < 128; depth *= 2) {
                if (type == PL_FMT_FLOAT && depth < 16)
                    continue;

                static const char *cnames[] = {
                    [1] = "r",
                    [2] = "rg",
                    [3] = "rgb",
                    [4] = "rgba",
                };

                static const char *tnames[] = {
                    [PL_FMT_UNORM] = "",
                    [PL_FMT_SNORM] = "s",
                    [PL_FMT_UINT]  = "u",
                    [PL_FMT_SINT]  = "i",
                    [PL_FMT_FLOAT] = "f",
                };

                const char *tname = tnames[type];
                if (type == PL_FMT_FLOAT && depth == 16)
                    tname = "hf";

                struct pl_fmt_t *fmt = pl_alloc_ptr(gpu, fmt);
                *fmt = (struct pl_fmt_t) {
                    .name = pl_asprintf(fmt, "%s%d%s", cnames[comps], depth, tname),
                    .type = type,
                    .num_components = comps,
                    .opaque = false,
                    .gatherable = true,
                    .internal_size = comps * depth / 8,
                    .texel_size = comps * depth / 8,
                    .texel_align = 1,
                    .caps = PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_LINEAR |
                            PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_BLENDABLE |
                            PL_FMT_CAP_VERTEX | PL_FMT_CAP_HOST_READABLE,
                };

                for (int i = 0; i < comps; i++) {
                    fmt->component_depth[i] = depth;
                    fmt->host_bits[i] = depth;
                    fmt->sample_order[i] = i;
                }

                if (gpu->glsl.compute)
                    fmt->caps |= PL_FMT_CAP_STORABLE;
                if (gpu->limits.max_buffer_texels && gpu->limits.max_ubo_size)
                    fmt->caps |= PL_FMT_CAP_TEXEL_UNIFORM;
                if (gpu->limits.max_buffer_texels && gpu->limits.max_ssbo_size)
                    fmt->caps |= PL_FMT_CAP_TEXEL_STORAGE;

                fmt->glsl_type = pl_var_glsl_type_name(pl_var_from_fmt(fmt, ""));
                fmt->glsl_format = pl_fmt_glsl_format(fmt, comps);
                fmt->fourcc = pl_fmt_fourcc(fmt);
                if (!fmt->glsl_format)
                    fmt->caps &= ~(PL_FMT_CAP_STORABLE | PL_FMT_CAP_TEXEL_STORAGE);
                PL_ARRAY_APPEND(gpu, formats, fmt);
            }
        }
    }

    gpu->formats = formats.elem;
    gpu->num_formats = formats.num;
    return pl_gpu_finalize(gpu);
}

static void dumb_destroy(pl_gpu gpu)
{
    pl_free((void *) gpu);
}

void pl_gpu_dummy_destroy(pl_gpu *gpu)
{
    pl_gpu_destroy(*gpu);
    *gpu = NULL;
}

struct buf_priv {
    uint8_t *data;
};

static pl_buf dumb_buf_create(pl_gpu gpu, const struct pl_buf_params *params)
{
    struct pl_buf_t *buf = pl_zalloc_obj(NULL, buf, struct buf_priv);
    buf->params = *params;
    buf->params.initial_data = NULL;

    struct buf_priv *p = PL_PRIV(buf);
    p->data = malloc(params->size);
    if (!p->data) {
        PL_ERR(gpu, "Failed allocating memory for dummy buffer!");
        pl_free(buf);
        return NULL;
    }

    if (params->initial_data)
        memcpy(p->data, params->initial_data, params->size);
    if (params->host_mapped)
        buf->data = p->data;

    return buf;
}

static void dumb_buf_destroy(pl_gpu gpu, pl_buf buf)
{
    struct buf_priv *p = PL_PRIV(buf);
    free(p->data);
    pl_free((void *) buf);
}

uint8_t *pl_buf_dummy_data(pl_buf buf)
{
    struct buf_priv *p = PL_PRIV(buf);
    return p->data;
}

static void dumb_buf_write(pl_gpu gpu, pl_buf buf, size_t buf_offset,
                           const void *data, size_t size)
{
    struct buf_priv *p = PL_PRIV(buf);
    memcpy(p->data + buf_offset, data, size);
}

static bool dumb_buf_read(pl_gpu gpu, pl_buf buf, size_t buf_offset,
                          void *dest, size_t size)
{
    struct buf_priv *p = PL_PRIV(buf);
    memcpy(dest, p->data + buf_offset, size);
    return true;
}

static void dumb_buf_copy(pl_gpu gpu, pl_buf dst, size_t dst_offset,
                          pl_buf src, size_t src_offset, size_t size)
{
    struct buf_priv *dstp = PL_PRIV(dst);
    struct buf_priv *srcp = PL_PRIV(src);
    memcpy(dstp->data + dst_offset, srcp->data + src_offset, size);
}

struct tex_priv {
    void *data;
};

static size_t tex_size(pl_gpu gpu, pl_tex tex)
{
    size_t size = tex->params.format->texel_size * tex->params.w;
    size *= PL_DEF(tex->params.h, 1);
    size *= PL_DEF(tex->params.d, 1);
    return size;
}

static pl_tex dumb_tex_create(pl_gpu gpu, const struct pl_tex_params *params)
{
    struct pl_tex_t *tex = pl_zalloc_obj(NULL, tex, void *);
    tex->params = *params;
    tex->params.initial_data = NULL;

    struct tex_priv *p = PL_PRIV(tex);
    p->data = malloc(tex_size(gpu, tex));
    if (!p->data) {
        PL_ERR(gpu, "Failed allocating memory for dummy texture!");
        pl_free(tex);
        return NULL;
    }

    if (params->initial_data)
        memcpy(p->data, params->initial_data, tex_size(gpu, tex));

    return tex;
}

pl_tex pl_tex_dummy_create(pl_gpu gpu, const struct pl_tex_dummy_params *params)
{
    // Only do minimal sanity checking, since this is just a dummy texture
    pl_assert(params->format && params->w >= 0 && params->h >= 0 && params->d >= 0);

    struct pl_tex_t *tex = pl_zalloc_obj(NULL, tex, struct tex_priv);
    tex->sampler_type = params->sampler_type;
    tex->params = (struct pl_tex_params) {
        .w = params->w,
        .h = params->h,
        .d = params->d,
        .format = params->format,
        .sampleable = true,
        .user_data = params->user_data,
    };

    return tex;
}

static void dumb_tex_destroy(pl_gpu gpu, pl_tex tex)
{
    struct tex_priv *p = PL_PRIV(tex);
    if (p->data)
        free(p->data);
    pl_free((void *) tex);
}

uint8_t *pl_tex_dummy_data(pl_tex tex)
{
    struct tex_priv *p = PL_PRIV(tex);
    return p->data;
}

static bool dumb_tex_upload(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    pl_tex tex = params->tex;
    struct tex_priv *p = PL_PRIV(tex);
    pl_assert(p->data);

    const uint8_t *src = params->ptr;
    uint8_t *dst = p->data;
    if (params->buf) {
        struct buf_priv *bufp = PL_PRIV(params->buf);
        src = (uint8_t *) bufp->data + params->buf_offset;
    }

    size_t texel_size = tex->params.format->texel_size;
    size_t row_size = pl_rect_w(params->rc) * texel_size;
    for (int z = params->rc.z0; z < params->rc.z1; z++) {
        size_t src_plane = z * params->depth_pitch;
        size_t dst_plane = z * tex->params.h * tex->params.w * texel_size;
        for (int y = params->rc.y0; y < params->rc.y1; y++) {
            size_t src_row = src_plane + y * params->row_pitch;
            size_t dst_row = dst_plane + y * tex->params.w * texel_size;
            size_t pos = params->rc.x0 * texel_size;
            memcpy(&dst[dst_row + pos], &src[src_row + pos], row_size);
        }
    }

    return true;
}

static bool dumb_tex_download(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    pl_tex tex = params->tex;
    struct tex_priv *p = PL_PRIV(tex);
    pl_assert(p->data);

    const uint8_t *src = p->data;
    uint8_t *dst = params->ptr;
    if (params->buf) {
        struct buf_priv *bufp = PL_PRIV(params->buf);
        dst = (uint8_t *) bufp->data + params->buf_offset;
    }

    size_t texel_size = tex->params.format->texel_size;
    size_t row_size = pl_rect_w(params->rc) * texel_size;
    for (int z = params->rc.z0; z < params->rc.z1; z++) {
        size_t src_plane = z * tex->params.h * tex->params.w * texel_size;
        size_t dst_plane = z * params->depth_pitch;
        for (int y = params->rc.y0; y < params->rc.y1; y++) {
            size_t src_row = src_plane + y * tex->params.w * texel_size;
            size_t dst_row = dst_plane + y * params->row_pitch;
            size_t pos = params->rc.x0 * texel_size;
            memcpy(&dst[dst_row + pos], &src[src_row + pos], row_size);
        }
    }

    return true;
}

static int dumb_desc_namespace(pl_gpu gpu, enum pl_desc_type type)
{
    return 0; // safest behavior: never alias bindings
}

static pl_pass dumb_pass_create(pl_gpu gpu, const struct pl_pass_params *params)
{
    PL_ERR(gpu, "Creating render passes is not supported for dummy GPUs");
    return NULL;
}

static void dumb_gpu_finish(pl_gpu gpu)
{
    // no-op
}

static const struct pl_gpu_fns pl_fns_dummy = {
    .destroy = dumb_destroy,
    .buf_create = dumb_buf_create,
    .buf_destroy = dumb_buf_destroy,
    .buf_write = dumb_buf_write,
    .buf_read = dumb_buf_read,
    .buf_copy = dumb_buf_copy,
    .tex_create = dumb_tex_create,
    .tex_destroy = dumb_tex_destroy,
    .tex_upload = dumb_tex_upload,
    .tex_download = dumb_tex_download,
    .desc_namespace = dumb_desc_namespace,
    .pass_create = dumb_pass_create,
    .gpu_finish = dumb_gpu_finish,
};
