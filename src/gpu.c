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
#include "context.h"
#include "shaders.h"
#include "gpu.h"

#define require(expr)                                           \
  do {                                                          \
      if (!(expr)) {                                            \
          PL_ERR(gpu, "Validation failed: %s (%s:%d)\n",        \
                  #expr, __FILE__, __LINE__);                   \
          goto error;                                           \
      }                                                         \
  } while (0)

int pl_optimal_transfer_stride(const struct pl_gpu *gpu, int dimension)
{
    return PL_ALIGN2(dimension, gpu->limits.align_tex_xfer_stride);
}

void pl_gpu_destroy(const struct pl_gpu *gpu)
{
    if (!gpu)
        return;

    gpu->impl->destroy(gpu);
}

void pl_gpu_print_info(const struct pl_gpu *gpu, enum pl_log_level lev)
{
    PL_MSG(gpu, lev, "GPU information:");
    PL_MSG(gpu, lev, "    GLSL version: %d%s", gpu->glsl.version,
           gpu->glsl.vulkan ? " (vulkan)" : gpu->glsl.gles ? " es" : "");
    PL_MSG(gpu, lev, "    Capabilities: 0x%x", (unsigned int) gpu->caps);
    PL_MSG(gpu, lev, "    Limits:");

#define LOG(fmt, field) \
    PL_MSG(gpu, lev, "      %-26s %" fmt, #field ":", gpu->limits.field)

    LOG(PRIu32, max_tex_1d_dim);
    LOG(PRIu32, max_tex_2d_dim);
    LOG(PRIu32, max_tex_3d_dim);
    LOG("zu", max_pushc_size);
    LOG("zu", max_xfer_size);
    LOG("zu", max_ubo_size);
    LOG("zu", max_ssbo_size);
    LOG(PRIu64, max_buffer_texels);
    LOG(PRId16, min_gather_offset);
    LOG(PRId16, max_gather_offset);

    if (gpu->caps & PL_GPU_CAP_COMPUTE) {
        LOG("zu", max_shmem_size);
        LOG(PRIu32, max_group_threads);
        LOG(PRIu32, max_group_size[0]);
        LOG(PRIu32, max_group_size[1]);
        LOG(PRIu32, max_group_size[2]);
        LOG(PRIu32, max_dispatch[0]);
        LOG(PRIu32, max_dispatch[1]);
        LOG(PRIu32, max_dispatch[2]);
    }

    LOG(PRIu32, align_tex_xfer_stride);
    LOG("zu", align_tex_xfer_offset);
#undef LOG

    if (pl_gpu_supports_interop(gpu)) {
        PL_MSG(gpu, lev, "    External API interop:");

        // Pretty-print the device UUID
        static const char *hexdigits = "0123456789ABCDEF";
        char buf[3 * sizeof(gpu->uuid)];
        for (int i = 0; i < sizeof(gpu->uuid); i++) {
            uint8_t x = gpu->uuid[i];
            buf[3 * i + 0] = hexdigits[x >> 4];
            buf[3 * i + 1] = hexdigits[x & 0xF];
            buf[3 * i + 2] = i == sizeof(gpu->uuid) - 1 ? '\0' : ':';
        }

        PL_MSG(gpu, lev, "      UUID: %s", buf);
        PL_MSG(gpu, lev, "      buf export caps: 0x%x",
               (unsigned int) gpu->export_caps.buf);
        PL_MSG(gpu, lev, "      buf import caps: 0x%x",
               (unsigned int) gpu->import_caps.buf);
        PL_MSG(gpu, lev, "      tex export caps: 0x%x",
               (unsigned int) gpu->export_caps.tex);
        PL_MSG(gpu, lev, "      tex import caps: 0x%x",
               (unsigned int) gpu->import_caps.tex);
        PL_MSG(gpu, lev, "      sync export caps: 0x%x",
               (unsigned int) gpu->export_caps.sync);
        PL_MSG(gpu, lev, "      sync import caps: 0x%x",
               (unsigned int) gpu->import_caps.sync);
    }
}

static int cmp_fmt(const void *pa, const void *pb)
{
    const struct pl_fmt *a = *(const struct pl_fmt **)pa;
    const struct pl_fmt *b = *(const struct pl_fmt **)pb;

    // Always prefer non-opaque formats
    if (a->opaque != b->opaque)
        return PL_CMP(a->opaque, b->opaque);

    // Always prefer non-emulated formats
    if (a->emulated != b->emulated)
        return PL_CMP(a->emulated, b->emulated);

    int ca = __builtin_popcount(a->caps),
        cb = __builtin_popcount(b->caps);
    if (ca != cb)
        return -PL_CMP(ca, cb); // invert to sort higher values first

    // If the population count is the same but the caps are different, prefer
    // the caps with a "lower" value (which tend to be more fundamental caps)
    if (a->caps != b->caps)
        return PL_CMP(a->caps, b->caps);

    // If the capabilities are equal, sort based on the component attributes
    for (int i = 0; i < PL_ARRAY_SIZE(a->component_depth); i++) {
        int da = a->component_depth[i],
            db = b->component_depth[i];
        if (da != db)
            return PL_CMP(da, db);

        int ha = a->host_bits[i],
            hb = b->host_bits[i];
        if (ha != hb)
            return PL_CMP(ha, hb);

        int oa = a->sample_order[i],
            ob = b->sample_order[i];
        if (oa != ob)
            return PL_CMP(oa, ob);
    }

    // Fall back to sorting by the name (for stability)
    return strcmp(a->name, b->name);
}

void pl_gpu_verify_formats(struct pl_gpu *gpu)
{
    for (int n = 0; n < gpu->num_formats; n++) {
        const struct pl_fmt *fmt = gpu->formats[n];
        pl_assert(fmt->name);
        pl_assert(fmt->type);
        pl_assert(fmt->num_components);
        pl_assert(fmt->internal_size);
        pl_assert(fmt->opaque ? !fmt->texel_size : fmt->texel_size);
        for (int i = 0; i < fmt->num_components; i++) {
            pl_assert(fmt->component_depth[i]);
            pl_assert(fmt->opaque ? !fmt->host_bits[i] : fmt->host_bits[i]);
        }

        enum pl_fmt_caps texel_caps = PL_FMT_CAP_VERTEX |
                                      PL_FMT_CAP_TEXEL_UNIFORM |
                                      PL_FMT_CAP_TEXEL_STORAGE;

        if (fmt->caps & texel_caps)
            pl_assert(fmt->glsl_type);
        if (fmt->caps & (PL_FMT_CAP_STORABLE | PL_FMT_CAP_TEXEL_STORAGE))
            pl_assert(fmt->glsl_format);
    }
}

void pl_gpu_sort_formats(struct pl_gpu *gpu)
{
    qsort(gpu->formats, gpu->num_formats, sizeof(struct pl_fmt *), cmp_fmt);
}

void pl_gpu_print_formats(const struct pl_gpu *gpu, enum pl_log_level lev)
{
    if (!pl_msg_test(gpu->ctx, lev))
        return;

    PL_MSG(gpu, lev, "GPU texture formats:");
    PL_MSG(gpu, lev, "    %-10s %-6s %-6s %-4s %-4s %-13s %-13s %-10s %-10s",
           "NAME", "TYPE", "CAPS", "SIZE", "COMP", "DEPTH", "HOST_BITS",
           "GLSL_TYPE", "GLSL_FMT");
    for (int n = 0; n < gpu->num_formats; n++) {
        const struct pl_fmt *fmt = gpu->formats[n];

        static const char *types[] = {
            [PL_FMT_UNKNOWN] = "UNKNOWN",
            [PL_FMT_UNORM]   = "UNORM",
            [PL_FMT_SNORM]   = "SNORM",
            [PL_FMT_UINT]    = "UINT",
            [PL_FMT_SINT]    = "SINT",
            [PL_FMT_FLOAT]   = "FLOAT",
        };

        static const char idx_map[4] = {'R', 'G', 'B', 'A'};
        char indices[4] = {' ', ' ', ' ', ' '};
        if (!fmt->opaque) {
            for (int i = 0; i < fmt->num_components; i++)
                indices[i] = idx_map[fmt->sample_order[i]];
        }

#define IDX4(f) (f)[0], (f)[1], (f)[2], (f)[3]

        PL_MSG(gpu, lev, "    %-10s %-6s 0x%-4x %-4zu %c%c%c%c "
               "{%-2d %-2d %-2d %-2d} {%-2d %-2d %-2d %-2d} %-10s %-10s",
               fmt->name, types[fmt->type], (unsigned int) fmt->caps,
               fmt->texel_size, IDX4(indices), IDX4(fmt->component_depth),
               IDX4(fmt->host_bits), PL_DEF(fmt->glsl_type, ""),
               PL_DEF(fmt->glsl_format, ""));

#undef IDX4
    }
}

bool pl_fmt_is_ordered(const struct pl_fmt *fmt)
{
    bool ret = !fmt->opaque;
    for (int i = 0; i < fmt->num_components; i++)
        ret &= fmt->sample_order[i] == i;
    return ret;
}

struct glsl_fmt {
    enum pl_fmt_type type;
    int num_components;
    int depth[4];
    const char *glsl_format;
};

// List taken from the GLSL specification. (Yes, GLSL supports only exactly
// these formats with exactly these names)
static const struct glsl_fmt pl_glsl_fmts[] = {
    {PL_FMT_FLOAT, 1, {16},             "r16f"},
    {PL_FMT_FLOAT, 1, {32},             "r32f"},
    {PL_FMT_FLOAT, 2, {16, 16},         "rg16f"},
    {PL_FMT_FLOAT, 2, {32, 32},         "rg32f"},
    {PL_FMT_FLOAT, 4, {16, 16, 16, 16}, "rgba16f"},
    {PL_FMT_FLOAT, 4, {32, 32, 32, 32}, "rgba32f"},
    {PL_FMT_FLOAT, 3, {11, 11, 10},     "r11f_g11f_b10f"},

    {PL_FMT_UNORM, 1, {8},              "r8"},
    {PL_FMT_UNORM, 1, {16},             "r16"},
    {PL_FMT_UNORM, 2, {8,  8},          "rg8"},
    {PL_FMT_UNORM, 2, {16, 16},         "rg16"},
    {PL_FMT_UNORM, 4, {8,  8,  8,  8},  "rgba8"},
    {PL_FMT_UNORM, 4, {16, 16, 16, 16}, "rgba16"},
    {PL_FMT_UNORM, 4, {10, 10, 10,  2}, "rgb10_a2"},

    {PL_FMT_SNORM, 1, {8},              "r8_snorm"},
    {PL_FMT_SNORM, 1, {16},             "r16_snorm"},
    {PL_FMT_SNORM, 2, {8,  8},          "rg8_snorm"},
    {PL_FMT_SNORM, 2, {16, 16},         "rg16_snorm"},
    {PL_FMT_SNORM, 4, {8,  8,  8,  8},  "rgba8_snorm"},
    {PL_FMT_SNORM, 4, {16, 16, 16, 16}, "rgba16_snorm"},

    {PL_FMT_UINT,  1, {8},              "r8ui"},
    {PL_FMT_UINT,  1, {16},             "r16ui"},
    {PL_FMT_UINT,  1, {32},             "r32ui"},
    {PL_FMT_UINT,  2, {8,  8},          "rg8ui"},
    {PL_FMT_UINT,  2, {16, 16},         "rg16ui"},
    {PL_FMT_UINT,  2, {32, 32},         "rg32ui"},
    {PL_FMT_UINT,  4, {8,  8,  8,  8},  "rgba8ui"},
    {PL_FMT_UINT,  4, {16, 16, 16, 16}, "rgba16ui"},
    {PL_FMT_UINT,  4, {32, 32, 32, 32}, "rgba32ui"},
    {PL_FMT_UINT,  4, {10, 10, 10,  2}, "rgb10_a2ui"},

    {PL_FMT_SINT,  1, {8},              "r8i"},
    {PL_FMT_SINT,  1, {16},             "r16i"},
    {PL_FMT_SINT,  1, {32},             "r32i"},
    {PL_FMT_SINT,  2, {8,  8},          "rg8i"},
    {PL_FMT_SINT,  2, {16, 16},         "rg16i"},
    {PL_FMT_SINT,  2, {32, 32},         "rg32i"},
    {PL_FMT_SINT,  4, {8,  8,  8,  8},  "rgba8i"},
    {PL_FMT_SINT,  4, {16, 16, 16, 16}, "rgba16i"},
    {PL_FMT_SINT,  4, {32, 32, 32, 32}, "rgba32i"},
};

const char *pl_fmt_glsl_format(const struct pl_fmt *fmt, int components)
{
    if (fmt->opaque)
        return NULL;

    for (int n = 0; n < PL_ARRAY_SIZE(pl_glsl_fmts); n++) {
        const struct glsl_fmt *gfmt = &pl_glsl_fmts[n];

        if (fmt->type != gfmt->type)
            continue;
        if (components != gfmt->num_components)
            continue;

        // The component order is irrelevant, so we need to sort the depth
        // based on the component's index
        int depth[4] = {0};
        for (int i = 0; i < fmt->num_components; i++)
            depth[fmt->sample_order[i]] = fmt->component_depth[i];

        // Copy over any emulated components
        for (int i = fmt->num_components; i < components; i++)
            depth[i] = gfmt->depth[i];

        for (int i = 0; i < PL_ARRAY_SIZE(depth); i++) {
            if (depth[i] != gfmt->depth[i])
                goto next_fmt;
        }

        return gfmt->glsl_format;

next_fmt: ; // equivalent to `continue`
    }

    return NULL;
}

const struct pl_fmt *pl_find_fmt(const struct pl_gpu *gpu, enum pl_fmt_type type,
                                 int num_components, int min_depth,
                                 int host_bits, enum pl_fmt_caps caps)
{
    for (int n = 0; n < gpu->num_formats; n++) {
        const struct pl_fmt *fmt = gpu->formats[n];
        if (fmt->type != type || fmt->num_components != num_components)
            continue;
        if ((fmt->caps & caps) != caps)
            continue;

        // When specifying some particular host representation, ensure the
        // format is non-opaque, ordered and unpadded
        if (host_bits && fmt->opaque)
            continue;
        if (host_bits && fmt->texel_size * 8 != host_bits * num_components)
            continue;
        if (host_bits && !pl_fmt_is_ordered(fmt))
            continue;

        for (int i = 0; i < fmt->num_components; i++) {
            if (fmt->component_depth[i] < min_depth)
                goto next_fmt;
            if (host_bits && fmt->host_bits[i] != host_bits)
                goto next_fmt;
        }

        return fmt;

next_fmt: ; // equivalent to `continue`
    }

    // ran out of formats
    PL_DEBUG(gpu, "No matching format found");
    return NULL;
}

const struct pl_fmt *pl_find_vertex_fmt(const struct pl_gpu *gpu,
                                        enum pl_fmt_type type, int comps)
{
    static const size_t sizes[] = {
        [PL_FMT_FLOAT] = sizeof(float),
        [PL_FMT_UNORM] = sizeof(unsigned),
        [PL_FMT_UINT]  = sizeof(unsigned),
        [PL_FMT_SNORM] = sizeof(int),
        [PL_FMT_SINT]  = sizeof(int),
    };

    return pl_find_fmt(gpu, type, comps, 0, 8 * sizes[type], PL_FMT_CAP_VERTEX);
}

const struct pl_fmt *pl_find_named_fmt(const struct pl_gpu *gpu, const char *name)
{
    if (!name)
        return NULL;

    for (int i = 0; i < gpu->num_formats; i++) {
        const struct pl_fmt *fmt = gpu->formats[i];
        if (strcmp(name, fmt->name) == 0)
            return fmt;
    }

    // ran out of formats
    return NULL;
}

const struct pl_tex *pl_tex_create(const struct pl_gpu *gpu,
                                   const struct pl_tex_params *params)
{
    require(!params->import_handle || !params->export_handle);
    if (params->export_handle) {
        require(params->export_handle & gpu->export_caps.tex);
        require(PL_ISPOT(params->export_handle));
    }
    if (params->import_handle) {
        require(params->import_handle & gpu->import_caps.tex);
        require(PL_ISPOT(params->import_handle));
    }

    switch (pl_tex_params_dimension(*params)) {
    case 1:
        require(params->w > 0);
        require(params->w <= gpu->limits.max_tex_1d_dim);
        require(!params->renderable);
        break;
    case 2:
        require(params->w > 0 && params->h > 0);
        require(params->w <= gpu->limits.max_tex_2d_dim);
        require(params->h <= gpu->limits.max_tex_2d_dim);
        break;
    case 3:
        require(params->w > 0 && params->h > 0 && params->d > 0);
        require(params->w <= gpu->limits.max_tex_3d_dim);
        require(params->h <= gpu->limits.max_tex_3d_dim);
        require(params->d <= gpu->limits.max_tex_3d_dim);
        require(!params->renderable);
        break;
    }

    const struct pl_fmt *fmt = params->format;
    require(fmt);
    require(!params->sampleable || fmt->caps & PL_FMT_CAP_SAMPLEABLE);
    require(!params->renderable || fmt->caps & PL_FMT_CAP_RENDERABLE);
    require(!params->storable   || fmt->caps & PL_FMT_CAP_STORABLE);
    require(!params->blit_src   || fmt->caps & PL_FMT_CAP_BLITTABLE);
    require(!params->blit_dst   || fmt->caps & PL_FMT_CAP_BLITTABLE);
    require(params->sample_mode != PL_TEX_SAMPLE_LINEAR || fmt->caps & PL_FMT_CAP_LINEAR);

    return gpu->impl->tex_create(gpu, params);

error:
    return NULL;
}

static bool pl_tex_params_superset(struct pl_tex_params a, struct pl_tex_params b)
{
    return a.w == b.w && a.h == b.h && a.d == b.d &&
           a.format          == b.format &&
           a.sample_mode     == b.sample_mode &&
           a.address_mode    == b.address_mode &&
           (a.sampleable     || !b.sampleable) &&
           (a.renderable     || !b.renderable) &&
           (a.storable       || !b.storable) &&
           (a.blit_src       || !b.blit_src) &&
           (a.blit_dst       || !b.blit_dst) &&
           (a.host_writable  || !b.host_writable) &&
           (a.host_readable  || !b.host_readable);
}

bool pl_tex_recreate(const struct pl_gpu *gpu, const struct pl_tex **tex,
                     const struct pl_tex_params *params)
{
    if (params->initial_data) {
        PL_ERR(gpu, "pl_tex_recreate may not be used with `initial_data`!");
        return false;
    }

    if (*tex && pl_tex_params_superset((*tex)->params, *params)) {
        pl_tex_invalidate(gpu, *tex);
        return true;
    }

    PL_INFO(gpu, "(Re)creating %dx%dx%d texture", params->w, params->h, params->d);
    pl_tex_destroy(gpu, tex);
    *tex = pl_tex_create(gpu, params);

    return !!*tex;
}

void pl_tex_destroy(const struct pl_gpu *gpu, const struct pl_tex **tex)
{
    if (!*tex)
        return;

    gpu->impl->tex_destroy(gpu, *tex);
    *tex = NULL;
}

void pl_tex_clear(const struct pl_gpu *gpu, const struct pl_tex *dst,
                  const float color[4])
{
    require(dst->params.blit_dst);

    pl_tex_invalidate(gpu, dst);
    gpu->impl->tex_clear(gpu, dst, color);

error:
    return;
}

void pl_tex_invalidate(const struct pl_gpu *gpu, const struct pl_tex *tex)
{
    if (gpu->impl->tex_invalidate)
        gpu->impl->tex_invalidate(gpu, tex);
}

static void strip_coords(const struct pl_tex *tex, struct pl_rect3d *rc)
{
    if (!tex->params.d) {
        rc->z0 = 0;
        rc->z1 = 1;
    }

    if (!tex->params.h) {
        rc->y0 = 0;
        rc->y1 = 1;
    }
}

void pl_tex_blit(const struct pl_gpu *gpu,
                 const struct pl_tex *dst, const struct pl_tex *src,
                 struct pl_rect3d dst_rc, struct pl_rect3d src_rc)
{
    const struct pl_fmt *src_fmt = src->params.format;
    const struct pl_fmt *dst_fmt = dst->params.format;
    require(src_fmt->internal_size == dst_fmt->internal_size);
    require((src_fmt->type == PL_FMT_UINT) == (dst_fmt->type == PL_FMT_UINT));
    require((src_fmt->type == PL_FMT_SINT) == (dst_fmt->type == PL_FMT_SINT));
    require(src->params.blit_src);
    require(dst->params.blit_dst);
    require(src_rc.x0 >= 0 && src_rc.x0 < src->params.w);
    require(src_rc.x1 > 0 && src_rc.x1 <= src->params.w);
    require(dst_rc.x0 >= 0 && dst_rc.x0 < dst->params.w);
    require(dst_rc.x1 > 0 && dst_rc.x1 <= dst->params.w);

    if (src->params.h) {
        require(dst->params.h);
        require(src_rc.y0 >= 0 && src_rc.y0 < src->params.h);
        require(src_rc.y1 > 0 && src_rc.y1 <= src->params.h);
    }
    if (dst->params.h) {
        require(dst_rc.y0 >= 0 && dst_rc.y0 < dst->params.h);
        require(dst_rc.y1 > 0 && dst_rc.y1 <= dst->params.h);
    }
    if (src->params.d) {
        require(dst->params.d);
        require(src_rc.z0 >= 0 && src_rc.z0 < src->params.d);
        require(src_rc.z1 > 0 && src_rc.z1 <= src->params.d);
    }
    if (dst->params.d) {
        require(dst_rc.z0 >= 0 && dst_rc.z0 < dst->params.d);
        require(dst_rc.z1 > 0 && dst_rc.z1 <= dst->params.d);
    }

    strip_coords(src, &src_rc);
    strip_coords(dst, &dst_rc);

    struct pl_rect3d full = {0, 0, 0, dst->params.w, dst->params.h, dst->params.d};
    strip_coords(dst, &full);

    struct pl_rect3d rcnorm = dst_rc;
    pl_rect3d_normalize(&rcnorm);
    if (pl_rect3d_eq(rcnorm, full))
        pl_tex_invalidate(gpu, dst);

    gpu->impl->tex_blit(gpu, dst, src, dst_rc, src_rc);

error:
    return;
}

size_t pl_tex_transfer_size(const struct pl_tex_transfer_params *par)
{
    const struct pl_tex *tex = par->tex;
    int w = pl_rect_w(par->rc), h = pl_rect_h(par->rc), d = pl_rect_d(par->rc);

    // This generates the absolute bare minimum size of a buffer required to
    // hold the data of a texture upload/download, by including stride padding
    // only where strictly necessary.
    int texels = ((d - 1) * par->stride_h + (h - 1)) * par->stride_w + w;
    return texels * tex->params.format->texel_size;
}

static bool fix_tex_transfer(const struct pl_gpu *gpu,
                             struct pl_tex_transfer_params *params)
{
    const struct pl_tex *tex = params->tex;
    struct pl_rect3d rc = params->rc;

    // Infer the default values
    if (!rc.x0 && !rc.x1)
        rc.x1 = tex->params.w;
    if (!rc.y0 && !rc.y1)
        rc.y1 = tex->params.h;
    if (!rc.z0 && !rc.z1)
        rc.z1 = tex->params.d;

    if (!params->stride_w)
        params->stride_w = pl_rect_w(rc);
    if (!params->stride_h)
        params->stride_h = pl_rect_h(rc);

    // Sanitize superfluous coordinates for the benefit of the GPU
    strip_coords(tex, &rc);
    if (!tex->params.w)
        params->stride_w = 1;
    if (!tex->params.h)
        params->stride_h = 1;

    params->rc = rc;

    // Check the parameters for sanity
    switch (pl_tex_params_dimension(tex->params))
    {
    case 3:
        require(rc.z1 > rc.z0);
        require(rc.z0 >= 0 && rc.z0 <  tex->params.d);
        require(rc.z1 >  0 && rc.z1 <= tex->params.d);
        require(params->stride_h >= pl_rect_h(rc));
        // fall through
    case 2:
        require(rc.y1 > rc.y0);
        require(rc.y0 >= 0 && rc.y0 <  tex->params.h);
        require(rc.y1 >  0 && rc.y1 <= tex->params.h);
        require(params->stride_w >= pl_rect_w(rc));
        // fall through
    case 1:
        require(rc.x1 > rc.x0);
        require(rc.x0 >= 0 && rc.x0 <  tex->params.w);
        require(rc.x1 >  0 && rc.x1 <= tex->params.w);
        break;
    }

    require(!params->buf ^ !params->ptr); // exactly one
    if (params->buf) {
        const struct pl_buf *buf = params->buf;
        size_t size = pl_tex_transfer_size(params);
        size_t texel = tex->params.format->texel_size;
        require(params->buf_offset == PL_ALIGN(params->buf_offset, texel));
        require(params->buf_offset + size <= buf->params.size);
    }

    return true;

error:
    return false;
}

bool pl_tex_upload(const struct pl_gpu *gpu,
                   const struct pl_tex_transfer_params *params)
{
    const struct pl_tex *tex = params->tex;
    require(tex);
    require(tex->params.host_writable);

    struct pl_tex_transfer_params fixed = *params;
    if (!fix_tex_transfer(gpu, &fixed))
        goto error;
    return gpu->impl->tex_upload(gpu, &fixed);

error:
    return false;
}

bool pl_tex_download(const struct pl_gpu *gpu,
                     const struct pl_tex_transfer_params *params)
{
    const struct pl_tex *tex = params->tex;
    require(tex);
    require(tex->params.host_readable);

    struct pl_tex_transfer_params fixed = *params;
    if (!fix_tex_transfer(gpu, &fixed))
        goto error;
    return gpu->impl->tex_download(gpu, &fixed);

error:
    return false;
}

const struct pl_buf *pl_buf_create(const struct pl_gpu *gpu,
                                   const struct pl_buf_params *params)
{
    if (params->handle_type) {
        require(params->handle_type & gpu->export_caps.buf);
        require(PL_ISPOT(params->handle_type));
    }

    switch (params->type) {
    case PL_BUF_TEX_TRANSFER:
        require(gpu->limits.max_xfer_size);
        require(params->size <= gpu->limits.max_xfer_size);
        break;
    case PL_BUF_UNIFORM:
        require(gpu->limits.max_ubo_size);
        require(params->size <= gpu->limits.max_ubo_size);
        break;
    case PL_BUF_STORAGE:
        require(gpu->limits.max_ssbo_size);
        require(params->size <= gpu->limits.max_ssbo_size);
        break;
    case PL_BUF_TEXEL_UNIFORM: {
        require(params->format);
        require(params->format->caps & PL_FMT_CAP_TEXEL_UNIFORM);
        size_t limit = gpu->limits.max_buffer_texels * params->format->texel_size;
        require(params->size <= limit);
        break;
    }
    case PL_BUF_TEXEL_STORAGE: {
        require(params->format);
        require(params->format->caps & PL_FMT_CAP_TEXEL_STORAGE);
        size_t limit = gpu->limits.max_buffer_texels * params->format->texel_size;
        require(params->size <= limit);
        break;
    }
    case PL_BUF_PRIVATE: break;
    default: abort();
    }

    const struct pl_buf *buf = gpu->impl->buf_create(gpu, params);
    if (buf)
        require(buf->data || !params->host_mapped);

    return buf;

error:
    return NULL;
}

static bool pl_buf_params_superset(struct pl_buf_params a, struct pl_buf_params b)
{
    return a.type            == b.type &&
           a.format          == b.format &&
           a.size            >= b.size &&
           (a.host_mapped    || !b.host_mapped) &&
           (a.host_writable  || !b.host_writable) &&
           (a.host_readable  || !b.host_readable);
}

bool pl_buf_recreate(const struct pl_gpu *gpu, const struct pl_buf **buf,
                     const struct pl_buf_params *params)
{
    if (params->initial_data) {
        PL_ERR(gpu, "pl_buf_recreate may not be used with `initial_data`!");
        return false;
    }

    if (*buf && pl_buf_params_superset((*buf)->params, *params))
        return true;

    PL_INFO(gpu, "(Re)creating %zu buffer", params->size);
    pl_buf_destroy(gpu, buf);
    *buf = pl_buf_create(gpu, params);

    return !!*buf;
}

void pl_buf_destroy(const struct pl_gpu *gpu, const struct pl_buf **buf)
{
    if (!*buf)
        return;

    gpu->impl->buf_destroy(gpu, *buf);
    *buf = NULL;
}

void pl_buf_write(const struct pl_gpu *gpu, const struct pl_buf *buf,
                  size_t buf_offset, const void *data, size_t size)
{
    require(buf->params.host_writable);
    require(buf_offset + size <= buf->params.size);
    require(buf_offset == PL_ALIGN2(buf_offset, 4));
    gpu->impl->buf_write(gpu, buf, buf_offset, data, size);

error:
    return;
}

bool pl_buf_read(const struct pl_gpu *gpu, const struct pl_buf *buf,
                 size_t buf_offset, void *dest, size_t size)
{
    require(buf->params.host_readable);
    require(buf_offset + size <= buf->params.size);
    require(buf_offset == PL_ALIGN2(buf_offset, 4));
    return gpu->impl->buf_read(gpu, buf, buf_offset, dest, size);

error:
    return false;
}

bool pl_buf_export(const struct pl_gpu *gpu, const struct pl_buf *buf)
{
    require(buf->params.handle_type);
    return gpu->impl->buf_export(gpu, buf);

error:
    return false;
}

bool pl_buf_poll(const struct pl_gpu *gpu, const struct pl_buf *buf, uint64_t t)
{
    return gpu->impl->buf_poll ? gpu->impl->buf_poll(gpu, buf, t) : false;
}

size_t pl_var_type_size(enum pl_var_type type)
{
    switch (type) {
    case PL_VAR_SINT:  return sizeof(int);
    case PL_VAR_UINT:  return sizeof(unsigned int);
    case PL_VAR_FLOAT: return sizeof(float);
    default: abort();
    }
}

#define MAX_DIM 4

const char *pl_var_glsl_type_name(struct pl_var var)
{
    static const char *types[PL_VAR_TYPE_COUNT][MAX_DIM+1][MAX_DIM+1] = {
    // float vectors
    [PL_VAR_FLOAT][1][1] = "float",
    [PL_VAR_FLOAT][1][2] = "vec2",
    [PL_VAR_FLOAT][1][3] = "vec3",
    [PL_VAR_FLOAT][1][4] = "vec4",
    // float matrices
    [PL_VAR_FLOAT][2][2] = "mat2",
    [PL_VAR_FLOAT][2][3] = "mat2x3",
    [PL_VAR_FLOAT][2][4] = "mat2x4",
    [PL_VAR_FLOAT][3][2] = "mat3x2",
    [PL_VAR_FLOAT][3][3] = "mat3",
    [PL_VAR_FLOAT][3][4] = "mat3x4",
    [PL_VAR_FLOAT][4][2] = "mat4x2",
    [PL_VAR_FLOAT][4][3] = "mat4x3",
    [PL_VAR_FLOAT][4][4] = "mat4",
    // integer vectors
    [PL_VAR_SINT][1][1] = "int",
    [PL_VAR_SINT][1][2] = "ivec2",
    [PL_VAR_SINT][1][3] = "ivec3",
    [PL_VAR_SINT][1][4] = "ivec4",
    // unsigned integer vectors
    [PL_VAR_UINT][1][1] = "uint",
    [PL_VAR_UINT][1][2] = "uvec2",
    [PL_VAR_UINT][1][3] = "uvec3",
    [PL_VAR_UINT][1][4] = "uvec4",
    };

    if (var.dim_v > MAX_DIM || var.dim_m > MAX_DIM)
        return NULL;

    return types[var.type][var.dim_m][var.dim_v];
}

#define PL_VAR(TYPE, NAME, M, V)                        \
    struct pl_var pl_var_##NAME(const char *name) {     \
        return (struct pl_var) {                        \
            .name  = name,                              \
            .type  = PL_VAR_##TYPE,                     \
            .dim_m = M,                                 \
            .dim_v = V,                                 \
            .dim_a = 1,                                 \
        };                                              \
    }

PL_VAR(SINT,  int,   1, 1);
PL_VAR(UINT,  uint,  1, 1);
PL_VAR(FLOAT, float, 1, 1);
PL_VAR(FLOAT, vec2,  1, 2);
PL_VAR(FLOAT, vec3,  1, 3);
PL_VAR(FLOAT, vec4,  1, 4);
PL_VAR(FLOAT, mat2,  2, 2);
PL_VAR(FLOAT, mat3,  3, 3);
PL_VAR(FLOAT, mat4,  4, 4);

#undef PL_VAR

struct pl_var pl_var_from_fmt(const struct pl_fmt *fmt, const char *name)
{
    static const enum pl_var_type vartypes[] = {
        [PL_FMT_FLOAT] = PL_VAR_FLOAT,
        [PL_FMT_UNORM] = PL_VAR_FLOAT,
        [PL_FMT_SNORM] = PL_VAR_FLOAT,
        [PL_FMT_UINT]  = PL_VAR_UINT,
        [PL_FMT_SINT]  = PL_VAR_SINT,
    };

    pl_assert(fmt->type < PL_ARRAY_SIZE(vartypes));
    return (struct pl_var) {
        .type  = vartypes[fmt->type],
        .name  = name,
        .dim_v = fmt->num_components,
        .dim_m = 1,
        .dim_a = 1,
    };
}

struct pl_var_layout pl_var_host_layout(size_t offset, const struct pl_var *var)
{
    size_t col_size = pl_var_type_size(var->type) * var->dim_v;
    return (struct pl_var_layout) {
        .offset = offset,
        .stride = col_size,
        .size   = col_size * var->dim_m * var->dim_a,
    };
}

struct pl_var_layout pl_std140_layout(size_t offset, const struct pl_var *var)
{
    size_t el_size = pl_var_type_size(var->type);

    // std140 packing rules:
    // 1. The size of generic values is their size in bytes
    // 2. The size of vectors is the vector length * the base count
    // 3. Matrices are treated like arrays of column vectors
    // 4. The size of array rows is that of the element size rounded up to
    // the nearest multiple of vec4
    // 5. All values are aligned to a multiple of their size (stride for arrays),
    // with the exception of vec3 which is aligned like vec4
    size_t stride = el_size * var->dim_v;
    size_t align = stride;
    if (var->dim_v == 3)
        align += el_size;
    if (var->dim_m * var->dim_a > 1)
        stride = align = PL_ALIGN2(stride, sizeof(float[4]));

    return (struct pl_var_layout) {
        .offset = PL_ALIGN2(offset, align),
        .stride = stride,
        .size   = stride * var->dim_m * var->dim_a,
    };
}

struct pl_var_layout pl_std430_layout(size_t offset, const struct pl_var *var)
{
    size_t el_size = pl_var_type_size(var->type);

    // std430 packing rules: like std140, except arrays/matrices are always
    // "tightly" packed, even arrays/matrices of vec3s
    size_t stride = el_size * var->dim_v;
    size_t align = stride;
    if (var->dim_v == 3 && var->dim_m == 1 && var->dim_a == 1)
        align += el_size;

    return (struct pl_var_layout) {
        .offset = PL_ALIGN2(offset, align),
        .stride = stride,
        .size   = stride * var->dim_m * var->dim_a,
    };
}

void memcpy_layout(void *dst_p, struct pl_var_layout dst_layout,
                   const void *src_p, struct pl_var_layout src_layout)
{
    uintptr_t src = (uintptr_t) src_p + src_layout.offset;
    uintptr_t dst = (uintptr_t) dst_p + dst_layout.offset;

    if (src_layout.stride == dst_layout.stride) {
        memcpy((void *) dst, (const void *) src, src_layout.size);
        return;
    }

    size_t stride = PL_MIN(src_layout.stride, dst_layout.stride);
    uintptr_t end = src + src_layout.size;
    while (src < end) {
        memcpy((void *) dst, (const void *) src, stride);
        src += src_layout.stride;
        dst += dst_layout.stride;
    }
}

int pl_desc_namespace(const struct pl_gpu *gpu, enum pl_desc_type type)
{
    int ret = gpu->impl->desc_namespace(gpu, type);
    pl_assert(ret >= 0 && ret < PL_DESC_TYPE_COUNT);
    return ret;
}

const char *pl_desc_access_glsl_name(enum pl_desc_access mode)
{
    switch (mode) {
    case PL_DESC_ACCESS_READWRITE: return "";
    case PL_DESC_ACCESS_READONLY:  return "readonly";
    case PL_DESC_ACCESS_WRITEONLY: return "writeonly";
    default: abort();
    }
}

const struct pl_pass *pl_pass_create(const struct pl_gpu *gpu,
                                     const struct pl_pass_params *params)
{
    require(params->glsl_shader);
    switch(params->type) {
    case PL_PASS_RASTER:
        require(params->vertex_shader);
        for (int i = 0; i < params->num_vertex_attribs; i++) {
            struct pl_vertex_attrib va = params->vertex_attribs[i];
            require(va.name);
            require(va.fmt);
            require(va.fmt->caps & PL_FMT_CAP_VERTEX);
            require(va.offset + va.fmt->texel_size <= params->vertex_stride);
        }

        const struct pl_fmt *target_fmt = params->target_dummy.params.format;
        require(target_fmt);
        require(target_fmt->caps & PL_FMT_CAP_RENDERABLE);
        require(!params->blend_params || target_fmt->caps & PL_FMT_CAP_BLENDABLE);
        break;
    case PL_PASS_COMPUTE:
        require(gpu->caps & PL_GPU_CAP_COMPUTE);
        break;
    default: abort();
    }

    for (int i = 0; i < params->num_variables; i++) {
        require(gpu->caps & PL_GPU_CAP_INPUT_VARIABLES);
        struct pl_var var = params->variables[i];
        require(var.name);
        require(pl_var_glsl_type_name(var));
    }

    for (int i = 0; i < params->num_descriptors; i++) {
        struct pl_desc desc = params->descriptors[i];
        require(desc.name);
        // TODO: enforce disjoint bindings if possible?
    }

    require(params->push_constants_size <= gpu->limits.max_pushc_size);
    require(params->push_constants_size == PL_ALIGN2(params->push_constants_size, 4));

    return gpu->impl->pass_create(gpu, params);

error:
    return NULL;
}

void pl_pass_destroy(const struct pl_gpu *gpu, const struct pl_pass **pass)
{
    if (!*pass)
        return;

    gpu->impl->pass_destroy(gpu, *pass);
    *pass = NULL;
}

void pl_pass_run(const struct pl_gpu *gpu, const struct pl_pass_run_params *params)
{
    const struct pl_pass *pass = params->pass;
    struct pl_pass_run_params new = *params;

    for (int i = 0; i < pass->params.num_descriptors; i++) {
        struct pl_desc desc = pass->params.descriptors[i];
        struct pl_desc_binding db = params->desc_bindings[i];
        require(db.object);
        switch (desc.type) {
        case PL_DESC_SAMPLED_TEX: {
            const struct pl_tex *tex = db.object;
            require(tex->params.sampleable);
            break;
        }
        case PL_DESC_STORAGE_IMG: {
            const struct pl_tex *tex = db.object;
            require(tex->params.storable);
            break;
        }
        case PL_DESC_BUF_UNIFORM: {
            const struct pl_buf *buf = db.object;
            require(buf->params.type == PL_BUF_UNIFORM);
            break;
        }
        case PL_DESC_BUF_STORAGE: {
            const struct pl_buf *buf = db.object;
            require(buf->params.type == PL_BUF_STORAGE);
            break;
        }
        case PL_DESC_BUF_TEXEL_UNIFORM: {
            const struct pl_buf *buf = db.object;
            require(buf->params.type == PL_BUF_TEXEL_UNIFORM);
            break;
        }
        case PL_DESC_BUF_TEXEL_STORAGE: {
            const struct pl_buf *buf = db.object;
            require(buf->params.type == PL_BUF_TEXEL_STORAGE);
            break;
        }
        default: abort();
        }
    }

    for (int i = 0; i < params->num_var_updates; i++) {
        struct pl_var_update vu = params->var_updates[i];
        require(gpu->caps & PL_GPU_CAP_INPUT_VARIABLES);
        require(vu.index >= 0 && vu.index < pass->params.num_variables);
        require(vu.data);
    }

    require(params->push_constants || !pass->params.push_constants_size);

    switch (pass->params.type) {
    case PL_PASS_RASTER: {
        require(params->vertex_data);
        switch (pass->params.vertex_type) {
        case PL_PRIM_TRIANGLE_LIST:
            require(params->vertex_count % 3 == 0);
            // fall through
        case PL_PRIM_TRIANGLE_STRIP:
        case PL_PRIM_TRIANGLE_FAN:
            require(params->vertex_count >= 3);
            break;
        }

        const struct pl_tex *tex = params->target;
        require(tex);
        require(pl_tex_params_dimension(tex->params) == 2);
        require(tex->params.format == pass->params.target_dummy.params.format);
        require(tex->params.renderable);
        struct pl_rect2d *vp = &new.viewport;
        struct pl_rect2d *sc = &new.scissors;

        // Sanitize viewport/scissors
        if (!vp->x0 && !vp->x1)
            vp->x1 = tex->params.w;
        if (!vp->y0 && !vp->y1)
            vp->y1 = tex->params.h;

        if (!sc->x0 && !sc->x1)
            sc->x1 = tex->params.w;
        if (!sc->y0 && !sc->y1)
            sc->y1 = tex->params.h;

        // Constrain the scissors to the target dimension (to sanitize the
        // underlying graphics API calls)
        sc->x0 = PL_MAX(0, PL_MIN(tex->params.w, sc->x0));
        sc->y0 = PL_MAX(0, PL_MIN(tex->params.h, sc->y0));
        sc->x1 = PL_MAX(0, PL_MIN(tex->params.w, sc->x1));
        sc->y1 = PL_MAX(0, PL_MIN(tex->params.h, sc->y1));

        // Scissors wholly outside target -> silently drop pass (also needed
        // to ensure we don't cause UB by specifying invalid scissors)
        if (!pl_rect_w(*sc) || !pl_rect_h(*sc))
            return;

        require(pl_rect_w(*vp) > 0);
        require(pl_rect_h(*vp) > 0);
        require(pl_rect_w(*sc) > 0);
        require(pl_rect_h(*sc) > 0);
        break;
    }
    case PL_PASS_COMPUTE:
        for (int i = 0; i < PL_ARRAY_SIZE(params->compute_groups); i++) {
            require(params->compute_groups[i] >= 0);
            require(params->compute_groups[i] <= gpu->limits.max_dispatch[i]);
        }
        break;
    default: abort();
    }

    if (params->target && !pass->params.load_target)
        pl_tex_invalidate(gpu, params->target);

    return gpu->impl->pass_run(gpu, &new);

error:
    return;
}

void pl_gpu_flush(const struct pl_gpu *gpu)
{
    if (gpu->impl->gpu_flush)
        gpu->impl->gpu_flush(gpu);
}

void pl_gpu_finish(const struct pl_gpu *gpu)
{
    gpu->impl->gpu_finish(gpu);
}

// GPU-internal helpers

void pl_buf_pool_uninit(const struct pl_gpu *gpu, struct pl_buf_pool *pool)
{
    for (int i = 0; i < pool->num_buffers; i++)
        pl_buf_destroy(gpu, &pool->buffers[i]);

    talloc_free(pool->buffers);
    *pool = (struct pl_buf_pool) {0};
}

static bool pl_buf_pool_grow(const struct pl_gpu *gpu, struct pl_buf_pool *pool)
{
    const struct pl_buf *buf = pl_buf_create(gpu, &pool->current_params);
    if (!buf)
        return false;

    TARRAY_INSERT_AT(NULL, pool->buffers, pool->num_buffers, pool->index, buf);
    PL_DEBUG(gpu, "Resized buffer pool of type %u to size %d",
             pool->current_params.type, pool->num_buffers);
    return true;
}

const struct pl_buf *pl_buf_pool_get(const struct pl_gpu *gpu,
                                     struct pl_buf_pool *pool,
                                     const struct pl_buf_params *params)
{
    require(!params->initial_data);

    if (!pl_buf_params_superset(pool->current_params, *params)) {
        pl_buf_pool_uninit(gpu, pool);
        pool->current_params = *params;
    }

    // Make sure we have at least one buffer available
    if (!pool->buffers && !pl_buf_pool_grow(gpu, pool))
        return NULL;

    bool usable = !pl_buf_poll(gpu, pool->buffers[pool->index], 0);
    if (usable)
        goto done;

    if (pool->num_buffers < PL_BUF_POOL_MAX_BUFFERS) {
        if (pl_buf_pool_grow(gpu, pool))
            goto done;

        // Failed growing the buffer pool, so just error out early
        return NULL;
    }

    // Can't resize any further, so just loop until the buffer is usable
    while (pl_buf_poll(gpu, pool->buffers[pool->index], 1000000000)) // 1s
        PL_TRACE(gpu, "Blocked on buffer pool availability! (slow path)");

done: ;
    const struct pl_buf *buf = pool->buffers[pool->index++];
    pool->index %= pool->num_buffers;

    return buf;

error:
    return NULL;
}

bool pl_tex_upload_pbo(const struct pl_gpu *gpu, struct pl_buf_pool *pbo,
                       const struct pl_tex_transfer_params *params)
{
    if (params->buf)
        return pl_tex_upload(gpu, params);

    struct pl_buf_params bufparams = {
        .type = PL_BUF_TEX_TRANSFER,
        .size = pl_tex_transfer_size(params),
        .host_writable = true,
    };

    const struct pl_buf *buf = pl_buf_pool_get(gpu, pbo, &bufparams);
    if (!buf)
        return false;

    pl_buf_write(gpu, buf, 0, params->ptr, bufparams.size);

    struct pl_tex_transfer_params newparams = *params;
    newparams.buf = buf;
    newparams.ptr = NULL;

    return pl_tex_upload(gpu, &newparams);
}

bool pl_tex_download_pbo(const struct pl_gpu *gpu, struct pl_buf_pool *pbo,
                         const struct pl_tex_transfer_params *params)
{
    if (params->buf)
        return pl_tex_download(gpu, params);

    struct pl_buf_params bufparams = {
        .type = PL_BUF_TEX_TRANSFER,
        .size = pl_tex_transfer_size(params),
        .host_readable = true,
    };

    const struct pl_buf *buf = pl_buf_pool_get(gpu, pbo, &bufparams);
    if (!buf)
        return false;

    struct pl_tex_transfer_params newparams = *params;
    newparams.buf = buf;
    newparams.ptr = NULL;

    if (!pl_tex_download(gpu, &newparams))
        return false;

    if (pl_buf_poll(gpu, buf, 0)) {
        PL_TRACE(gpu, "pl_tex_download without buffer: blocking (slow path)");
        while (pl_buf_poll(gpu, buf, 1000000)) ; // 1 ms
    }

    return pl_buf_read(gpu, buf, 0, params->ptr, bufparams.size);
}

bool pl_tex_upload_texel(const struct pl_gpu *gpu, struct pl_dispatch *dp,
                         const struct pl_tex_transfer_params *params)
{
    const int threads = 256;
    const struct pl_tex *tex = params->tex;
    const struct pl_fmt *fmt = tex->params.format;
    require(params->buf);
    require(params->buf->params.type == PL_BUF_TEXEL_UNIFORM);

    struct pl_shader *sh = pl_dispatch_begin(dp);
    if (!sh_try_compute(sh, threads, 1, true, 0)) {
        PL_ERR(gpu, "Failed emulating texture transfer!");
        pl_dispatch_abort(dp, &sh);
        return false;
    }

    ident_t buf = sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name = "data",
            .type = PL_DESC_BUF_TEXEL_UNIFORM,
        },
        .object = params->buf,
    });

    ident_t img = sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name = "image",
            .type = PL_DESC_STORAGE_IMG,
            .access = PL_DESC_ACCESS_WRITEONLY,
        },
        .object = params->tex,
    });

    GLSL("vec4 color = vec4(0.0);                                       \n"
         "ivec3 pos = ivec3(gl_GlobalInvocationID) + ivec3(%d, %d, %d); \n"
         "int base = ((pos.z * %d + pos.y) * %d + pos.x) * %d;          \n",
         params->rc.x0, params->rc.y0, params->rc.z0,
         params->stride_h, params->stride_w, fmt->num_components);

    for (int i = 0; i < fmt->num_components; i++)
        GLSL("color[%d] = texelFetch(%s, base + %d).r; \n", i, buf, i);

    // If the transfer width is a natural multiple of the thread size, we
    // can skip the bounds check. Otherwise, make sure we aren't blitting out
    // of the range since this would violate semantics
    int groups_x = (pl_rect_w(params->rc) + threads - 1) / threads;
    bool is_crop = params->rc.x1 != params->tex->params.w;
    if (is_crop && groups_x * threads != pl_rect_w(params->rc))
        GLSL("if (gl_GlobalInvocationID.x < %d)\n", pl_rect_w(params->rc));

    int dims = pl_tex_params_dimension(tex->params);
    static const char *coord_types[] = {
        [1] = "int",
        [2] = "ivec2",
        [3] = "ivec3",
    };

    GLSL("imageStore(%s, %s(pos), color);\n", img, coord_types[dims]);
    int groups[3] = { groups_x, pl_rect_h(params->rc), pl_rect_d(params->rc) };
    return pl_dispatch_compute(dp, &sh, groups);

error:
    return false;
}

bool pl_tex_download_texel(const struct pl_gpu *gpu, struct pl_dispatch *dp,
                           const struct pl_tex_transfer_params *params)
{
    const int threads = 256;
    const struct pl_tex *tex = params->tex;
    const struct pl_fmt *fmt = tex->params.format;
    require(params->buf);
    require(params->buf->params.type == PL_BUF_TEXEL_STORAGE);

    struct pl_shader *sh = pl_dispatch_begin(dp);
    if (!sh_try_compute(sh, threads, 1, true, 0)) {
        PL_ERR(gpu, "Failed emulating texture transfer!");
        pl_dispatch_abort(dp, &sh);
        return false;
    }

    ident_t buf = sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name = "data",
            .type = PL_DESC_BUF_TEXEL_STORAGE,
        },
        .object = params->buf,
    });

    ident_t img = sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name = "image",
            .type = PL_DESC_STORAGE_IMG,
            .access = PL_DESC_ACCESS_READONLY,
        },
        .object = params->tex,
    });

    int dims = pl_tex_params_dimension(tex->params);
    static const char *coord_types[] = {
        [1] = "int",
        [2] = "ivec2",
        [3] = "ivec3",
    };

    GLSL("ivec3 pos = ivec3(gl_GlobalInvocationID) + ivec3(%d, %d, %d); \n"
         "int base = ((pos.z * %d + pos.y) * %d + pos.x) * %d;          \n"
         "vec4 color = imageLoad(%s, %s(pos));                          \n",
         params->rc.x0, params->rc.y0, params->rc.z0,
         params->stride_h, params->stride_w, fmt->num_components,
         img, coord_types[dims]);

    int groups_x = (pl_rect_w(params->rc) + threads - 1) / threads;
    if (groups_x * threads != pl_rect_w(params->rc))
        GLSL("if (gl_GlobalInvocationID.x < %d)\n", pl_rect_w(params->rc));

    GLSL("{\n");
    for (int i = 0; i < fmt->num_components; i++)
        GLSL("imageStore(%s, base + %d, vec4(color[%d])); \n", buf, i, i);
    GLSL("}\n");

    int groups[3] = { groups_x, pl_rect_h(params->rc), pl_rect_d(params->rc) };
    return pl_dispatch_compute(dp, &sh, groups);

error:
    return false;
}

struct pl_pass_params pl_pass_params_copy(void *tactx,
                                          const struct pl_pass_params *params)
{
    struct pl_pass_params new = *params;
    new.target_dummy.priv = NULL;
    new.cached_program = NULL;
    new.cached_program_len = 0;

    new.glsl_shader = talloc_strdup(tactx, new.glsl_shader);
    new.vertex_shader = talloc_strdup(tactx, new.vertex_shader);
    if (new.blend_params)
        new.blend_params = talloc_ptrdup(tactx, new.blend_params);

#define DUPSTRS(name, array, num)                                       \
    do {                                                                \
        (array) = TARRAY_DUP(tactx, array, num);                        \
        for (int j = 0; j < num; j++)                                   \
            (array)[j].name = talloc_strdup(tactx, (array)[j].name);    \
    } while (0)

    DUPSTRS(name, new.variables,      new.num_variables);
    DUPSTRS(name, new.descriptors,    new.num_descriptors);
    DUPSTRS(name, new.vertex_attribs, new.num_vertex_attribs);

#undef DUPNAMES

    return new;
}

const struct pl_sync *pl_sync_create(const struct pl_gpu *gpu,
                                     enum pl_handle_type handle_type)
{
    require(handle_type);
    require(handle_type & gpu->export_caps.sync);
    require(PL_ISPOT(handle_type));
    return gpu->impl->sync_create(gpu, handle_type);

error:
    return NULL;
}

void pl_sync_destroy(const struct pl_gpu *gpu,
                     const struct pl_sync **sync)
{
    if (!*sync)
        return;

    gpu->impl->sync_destroy(gpu, *sync);
    *sync = NULL;
}

bool pl_tex_export(const struct pl_gpu *gpu, const struct pl_tex *tex,
                   const struct pl_sync *sync)
{
    return gpu->impl->tex_export(gpu, tex, sync);
}
