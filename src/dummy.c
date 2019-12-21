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

const struct pl_gpu_dummy_params pl_gpu_dummy_default_params = {
    .caps = PL_GPU_CAP_COMPUTE | PL_GPU_CAP_INPUT_VARIABLES | PL_GPU_CAP_MAPPED_BUFFERS,
    .glsl = {
        .version    = 450,
        .gles       = false,
        .vulkan     = false,
    },

    .limits = {
        .max_tex_1d_dim     = UINT32_MAX,
        .max_tex_2d_dim     = UINT32_MAX,
        .max_tex_3d_dim     = UINT32_MAX,
        .max_pushc_size     = SIZE_MAX,
        .max_xfer_size      = SIZE_MAX,
        .max_ubo_size       = SIZE_MAX,
        .max_ssbo_size      = SIZE_MAX,
        .max_buffer_texels  = UINT64_MAX,
        .min_gather_offset  = INT16_MIN,
        .max_gather_offset  = INT16_MAX,
        .max_shmem_size     = SIZE_MAX,
        .max_group_threads  = UINT32_MAX,
        .max_group_size     = { UINT32_MAX, UINT32_MAX, UINT32_MAX },
        .max_dispatch       = { UINT32_MAX, UINT32_MAX, UINT32_MAX },
        .align_tex_xfer_stride = 1,
        .align_tex_xfer_offset = 1,
    },
};

static const struct pl_gpu_fns pl_fns_dummy;

struct priv {
    struct pl_gpu_fns impl;
    struct pl_gpu_dummy_params params;
};

const struct pl_gpu *pl_gpu_dummy_create(struct pl_context *ctx,
                                         const struct pl_gpu_dummy_params *params)
{
    params = PL_DEF(params, &pl_gpu_dummy_default_params);

    struct pl_gpu *gpu = talloc_zero_priv(NULL, struct pl_gpu, struct priv);
    gpu->ctx = ctx;
    gpu->caps = params->caps;
    gpu->glsl = params->glsl;
    gpu->limits = params->limits;

    struct priv *p = TA_PRIV(gpu);
    p->impl = pl_fns_dummy;
    p->params = *params;

    // Forcibly override these, because we know for sure what the values are
    gpu->limits.align_tex_xfer_stride = 1;
    gpu->limits.align_tex_xfer_offset = 1;

    // Set up the dummy formats, add one for each possible format type that we
    // can represent on the host
    for (enum pl_fmt_type type = 1; type < PL_FMT_TYPE_COUNT; type++) {
        for (int comps = 1; comps <= 4; comps++) {
            for (int depth = 8; depth < 128; depth *= 2) {
                if (type == PL_FMT_FLOAT && depth < sizeof(float) * 8)
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

                struct pl_fmt *fmt = talloc_ptrtype(gpu, fmt);
                *fmt = (struct pl_fmt) {
                    .name = talloc_asprintf(fmt, "%s%d%s", cnames[comps], depth,
                                            tnames[type]),
                    .type = type,
                    .num_components = comps,
                    .opaque = false,
                    .internal_size = comps * depth / 8,
                    .texel_size = comps * depth / 8,
                    .caps = PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_LINEAR |
                            PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_BLENDABLE |
                            PL_FMT_CAP_VERTEX,
                };

                for (int i = 0; i < comps; i++) {
                    fmt->component_depth[i] = depth;
                    fmt->host_bits[i] = depth;
                    fmt->sample_order[i] = i;
                }

                if (gpu->caps & PL_GPU_CAP_COMPUTE)
                    fmt->caps |= PL_FMT_CAP_STORABLE;
                if (gpu->limits.max_buffer_texels && gpu->limits.max_ubo_size)
                    fmt->caps |= PL_FMT_CAP_TEXEL_UNIFORM;
                if (gpu->limits.max_buffer_texels && gpu->limits.max_ssbo_size)
                    fmt->caps |= PL_FMT_CAP_TEXEL_STORAGE;

                fmt->glsl_type = pl_var_glsl_type_name(pl_var_from_fmt(fmt, ""));
                fmt->glsl_format = pl_fmt_glsl_format(fmt, comps);
                if (!fmt->glsl_format)
                    fmt->caps &= ~(PL_FMT_CAP_STORABLE | PL_FMT_CAP_TEXEL_STORAGE);
                TARRAY_APPEND(gpu, gpu->formats, gpu->num_formats, fmt);
            }
        }
    }

    pl_gpu_sort_formats(gpu);
    pl_gpu_verify_formats(gpu);
    pl_gpu_print_info(gpu, PL_LOG_INFO);
    pl_gpu_print_formats(gpu, PL_LOG_DEBUG);
    return gpu;
}

static void dumb_destroy(const struct pl_gpu *gpu)
{
    talloc_free((void *) gpu);
}

void pl_gpu_dummy_destroy(const struct pl_gpu **gpu)
{
    pl_gpu_destroy(*gpu);
    *gpu = NULL;
}

struct buf_priv {
    void *data;
};

static const struct pl_buf *dumb_buf_create(const struct pl_gpu *gpu,
                                            const struct pl_buf_params *params)
{
    struct pl_buf *buf = talloc_zero_priv(NULL, struct pl_buf, struct buf_priv);
    buf->params = *params;
    buf->params.initial_data = NULL;

    struct buf_priv *p = TA_PRIV(buf);
    p->data = malloc(params->size);
    if (!p->data) {
        PL_ERR(gpu, "Failed allocating memory for dummy buffer!");
        talloc_free(buf);
        return NULL;
    }

    if (params->initial_data)
        memcpy(p->data, params->initial_data, params->size);
    if (params->host_mapped)
        buf->data = p->data;

    return buf;
}

static void dumb_buf_destroy(const struct pl_gpu *gpu, const struct pl_buf *buf)
{
    struct buf_priv *p = TA_PRIV(buf);
    free(p->data);
    talloc_free((void *) buf);
}

uint8_t *pl_buf_dummy_data(const struct pl_buf *buf)
{
    struct buf_priv *p = TA_PRIV(buf);
    return p->data;
}

static void dumb_buf_write(const struct pl_gpu *gpu, const struct pl_buf *buf,
                           size_t buf_offset, const void *data, size_t size)
{
    struct buf_priv *p = TA_PRIV(buf);
    uint8_t *dst = p->data;
    memcpy(&dst[buf_offset], data, size);
}

static bool dumb_buf_read(const struct pl_gpu *gpu, const struct pl_buf *buf,
                          size_t buf_offset, void *dest, size_t size)
{
    struct buf_priv *p = TA_PRIV(buf);
    const uint8_t *src = p->data;
    memcpy(dest, &src[buf_offset], size);
    return true;
}

struct tex_priv {
    void *data;
};

static size_t tex_size(const struct pl_gpu *gpu, const struct pl_tex *tex)
{
    size_t size = tex->params.format->texel_size * tex->params.w;
    size *= PL_DEF(tex->params.h, 1);
    size *= PL_DEF(tex->params.d, 1);
    return size;
}

static const struct pl_tex *dumb_tex_create(const struct pl_gpu *gpu,
                                            const struct pl_tex_params *params)
{
    struct pl_tex *tex = talloc_zero_priv(NULL, struct pl_tex, void *);
    tex->params = *params;
    tex->params.initial_data = NULL;

    struct tex_priv *p = TA_PRIV(tex);
    p->data = malloc(tex_size(gpu, tex));
    if (!p->data) {
        PL_ERR(gpu, "Failed allocating memory for dummy texture!");
        talloc_free(tex);
        return NULL;
    }

    if (params->initial_data)
        memcpy(p->data, params->initial_data, tex_size(gpu, tex));

    return tex;
}

const struct pl_tex *pl_tex_dummy_create(const struct pl_gpu *gpu,
                                         const struct pl_tex_params *params)
{
    // Only do minimal sanity checking, since this is just a dummy texture
    pl_assert(params->format && params->w >= 0 && params->h >= 0 && params->d >= 0);

    struct pl_tex *tex = talloc_zero_priv(NULL, struct pl_tex, struct tex_priv);
    tex->params = (struct pl_tex_params) {
        // Whitelist options
        .w = params->w,
        .h = params->h,
        .d = params->d,
        .format = params->format,
        .sampleable = true,
        .sample_mode = params->sample_mode,
        .address_mode = params->address_mode,
        .user_data = params->user_data,
    };

    return tex;
}

static void dumb_tex_destroy(const struct pl_gpu *gpu, const struct pl_tex *tex)
{
    struct tex_priv *p = TA_PRIV(tex);
    if (p->data)
        free(p->data);
    talloc_free((void *) tex);
}

uint8_t *pl_tex_dummy_data(const struct pl_tex *tex)
{
    struct tex_priv *p = TA_PRIV(tex);
    return p->data;
}

static void dumb_tex_clear(const struct pl_gpu *gpu, const struct pl_tex *tex,
                           const float color[4])
{
    const struct pl_fmt *fmt = tex->params.format;
    struct tex_priv *p = TA_PRIV(tex);
    pl_assert(p->data);

    // Convert from float[4] to whatever internal representation we need
    union {
        uint8_t u8[4];
        uint16_t u16[4];
        uint32_t u32[4];
        uint64_t u64[4];

        int8_t i8[4];
        int16_t i16[4];
        int32_t i32[4];
        int64_t i64[4];

        float f[4];
        double d[4];

        uint8_t bytes[sizeof(double[4])]; // raw data
    } data;

    for (int c = 0; c < fmt->num_components; c++) {
        switch (fmt->type) {
        case PL_FMT_UNORM:
        case PL_FMT_UINT:
            switch (fmt->component_depth[0]) {
            case 8:  data.u8[c]  = UINT8_MAX  * color[c]; break;
            case 16: data.u16[c] = UINT16_MAX * color[c]; break;
            case 32: data.u32[c] = UINT32_MAX * color[c]; break;
            case 64: data.u64[c] = UINT64_MAX * color[c]; break;
            default: abort();
            }
            break;

        case PL_FMT_SNORM:
        case PL_FMT_SINT:
            switch (fmt->component_depth[0]) {
            case 8:  data.i8[c]  = INT8_MAX  * color[c]; break;
            case 16: data.i16[c] = INT16_MAX * color[c]; break;
            case 32: data.i32[c] = INT32_MAX * color[c]; break;
            case 64: data.i64[c] = INT64_MAX * color[c]; break;
            default: abort();
            }
            break;

        case PL_FMT_FLOAT:
            switch (fmt->component_depth[0]) {
            case 32: data.f[c] = color[c]; break;
            case 64: data.d[c] = color[c]; break;
            default: abort();
            }
            break;

        default: abort();
        }
    }

    // Fast path: clearing to a constant value
    bool fast_path = true;
    for (size_t i = 1; i < fmt->texel_size; i++)
        fast_path &= data.bytes[i] == data.bytes[0];

    if (fast_path) {
        memset(p->data, data.bytes[0], tex_size(gpu, tex));
        return;
    }

    uint8_t *dst = p->data;
    for (size_t pos = 0; pos < tex_size(gpu, tex); pos += fmt->texel_size)
        memcpy(&dst[pos], &data.bytes[0], fmt->texel_size);
}

static bool dumb_tex_upload(const struct pl_gpu *gpu,
                            const struct pl_tex_transfer_params *params)
{
    const struct pl_tex *tex = params->tex;
    struct tex_priv *p = TA_PRIV(tex);
    pl_assert(p->data);

    const uint8_t *src = params->ptr;
    uint8_t *dst = p->data;
    if (params->buf) {
        struct buf_priv *bufp = TA_PRIV(params->buf);
        src = (uint8_t *) bufp->data + params->buf_offset;
    }

    size_t texel_size = tex->params.format->texel_size;
    size_t row_size = pl_rect_w(params->rc) * texel_size;
    for (int z = params->rc.z0; z < params->rc.z1; z++) {
        size_t src_plane = z * params->stride_h * params->stride_w * texel_size;
        size_t dst_plane = z * tex->params.h * tex->params.w * texel_size;
        for (int y = params->rc.y0; y < params->rc.y1; y++) {
            size_t src_row = src_plane + y * params->stride_w * texel_size;
            size_t dst_row = dst_plane + y * tex->params.w * texel_size;
            size_t pos = params->rc.x0 * texel_size;
            memcpy(&dst[dst_row + pos], &src[src_row + pos], row_size);
        }
    }

    return true;
}

static bool dumb_tex_download(const struct pl_gpu *gpu,
                              const struct pl_tex_transfer_params *params)
{
    const struct pl_tex *tex = params->tex;
    struct tex_priv *p = TA_PRIV(tex);
    pl_assert(p->data);

    const uint8_t *src = p->data;
    uint8_t *dst = params->ptr;
    if (params->buf) {
        struct buf_priv *bufp = TA_PRIV(params->buf);
        dst = (uint8_t *) bufp->data + params->buf_offset;
    }

    size_t texel_size = tex->params.format->texel_size;
    size_t row_size = pl_rect_w(params->rc) * texel_size;
    for (int z = params->rc.z0; z < params->rc.z1; z++) {
        size_t src_plane = z * tex->params.h * tex->params.w * texel_size;
        size_t dst_plane = z * params->stride_h * params->stride_w * texel_size;
        for (int y = params->rc.y0; y < params->rc.y1; y++) {
            size_t src_row = src_plane + y * tex->params.w * texel_size;
            size_t dst_row = dst_plane + y * params->stride_w * texel_size;
            size_t pos = params->rc.x0 * texel_size;
            memcpy(&dst[dst_row + pos], &src[src_row + pos], row_size);
        }
    }

    return true;
}

static int dumb_desc_namespace(const struct pl_gpu *gpu, enum pl_desc_type type)
{
    return 0; // safest behavior: never alias bindings
}

static const struct pl_pass *dumb_pass_create(const struct pl_gpu *gpu,
                                              const struct pl_pass_params *params)
{
    PL_ERR(gpu, "Creating render passes is not supported for dummy GPUs");
    return NULL;
}

static void dumb_pass_destroy(const struct pl_gpu *gpu, const struct pl_pass *pass)
{
    pl_assert(!"unreachable");
}

static void dumb_pass_run(const struct pl_gpu *gpu,
                          const struct pl_pass_run_params *params)
{
    pl_assert(!"unreachable");
}

static void dumb_gpu_finish(const struct pl_gpu *gpu)
{
    // no-op
}

static const struct pl_gpu_fns pl_fns_dummy = {
    .destroy = dumb_destroy,
    .buf_create = dumb_buf_create,
    .buf_destroy = dumb_buf_destroy,
    .buf_write = dumb_buf_write,
    .buf_read = dumb_buf_read,
    .tex_create = dumb_tex_create,
    .tex_destroy = dumb_tex_destroy,
    .tex_clear = dumb_tex_clear,
    .tex_upload = dumb_tex_upload,
    .tex_download = dumb_tex_download,
    .desc_namespace = dumb_desc_namespace,
    .pass_create = dumb_pass_create,
    .pass_destroy = dumb_pass_destroy,
    .pass_run = dumb_pass_run,
    .gpu_finish = dumb_gpu_finish,
};
