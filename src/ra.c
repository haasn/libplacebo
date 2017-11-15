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
#include "ra.h"

int ra_optimal_transfer_stride(const struct ra *ra, int dimension)
{
    return PL_ALIGN2(dimension, ra->limits.align_tex_xfer_stride);
}

void ra_destroy(const struct ra *ra)
{
    if (!ra)
        return;

    ra->impl->destroy(ra);
}

void ra_print_info(const struct ra *ra, enum pl_log_level lev)
{
    PL_MSG(ra, lev, "RA information:");
    PL_MSG(ra, lev, "    GLSL version: %d%s", ra->glsl.version,
           ra->glsl.vulkan ? " (vulkan)" : ra->glsl.gles ? " es" : "");
    PL_MSG(ra, lev, "    Capabilities: 0x%x", (unsigned int) ra->caps);
    PL_MSG(ra, lev, "    Limits:");

#define LOG(fmt, field) \
    PL_MSG(ra, lev, "      %-26s " fmt, #field ":", ra->limits.field)

    LOG("%d", max_tex_1d_dim);
    LOG("%d", max_tex_2d_dim);
    LOG("%d", max_tex_3d_dim);
    LOG("%zu", max_pushc_size);
    LOG("%zu", max_xfer_size);
    LOG("%zu", max_ubo_size);
    LOG("%zu", max_ssbo_size);
    LOG("%d", min_gather_offset);
    LOG("%d", max_gather_offset);

    if (ra->caps & RA_CAP_COMPUTE) {
        LOG("%zu", max_shmem_size);
        LOG("%d", max_group_threads);
        LOG("%d", max_group_size[0]);
        LOG("%d", max_group_size[1]);
        LOG("%d", max_group_size[2]);
        LOG("%d", max_dispatch[0]);
        LOG("%d", max_dispatch[1]);
        LOG("%d", max_dispatch[2]);
    }

    LOG("%d", align_tex_xfer_stride);
    LOG("%zu", align_tex_xfer_offset);
#undef LOG
}

static int cmp_fmt(const void *pa, const void *pb)
{
    const struct ra_fmt *a = *(const struct ra_fmt **)pa;
    const struct ra_fmt *b = *(const struct ra_fmt **)pb;

    // Always prefer non-opaque formats
    if (a->opaque != b->opaque)
        return PL_CMP(a->opaque, b->opaque);

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

void ra_sort_formats(struct ra *ra)
{
    qsort(ra->formats, ra->num_formats, sizeof(struct ra_fmt *), cmp_fmt);
}

void ra_print_formats(const struct ra *ra, enum pl_log_level lev)
{
    if (!pl_msg_test(ra->ctx, lev))
        return;

    PL_MSG(ra, lev, "RA texture formats:");
    PL_MSG(ra, lev, "    %-10s %-6s %-6s %-4s %-4s %-13s %-13s %-10s %-10s",
           "NAME", "TYPE", "CAPS", "SIZE", "COMP", "DEPTH", "BITS",
           "GLSL_TYPE", "GLSL_FMT");
    for (int n = 0; n < ra->num_formats; n++) {
        const struct ra_fmt *fmt = ra->formats[n];

        static const char *types[] = {
            [RA_FMT_UNKNOWN] = "UNKNOWN",
            [RA_FMT_UNORM]   = "UNORM",
            [RA_FMT_SNORM]   = "SNORM",
            [RA_FMT_UINT]    = "UINT",
            [RA_FMT_SINT]    = "SINT",
            [RA_FMT_FLOAT]   = "FLOAT",
        };

        static const char idx_map[4] = {'R', 'G', 'B', 'A'};
        char indices[4] = {' ', ' ', ' ', ' '};
        if (!fmt->opaque) {
            for (int i = 0; i < fmt->num_components; i++)
                indices[i] = idx_map[fmt->sample_order[i]];
        }

#define IDX4(f) (f)[0], (f)[1], (f)[2], (f)[3]

        PL_MSG(ra, lev, "    %-10s %-6s 0x%-4x %-4zu %c%c%c%c "
               "{%-2d %-2d %-2d %-2d} {%-2d %-2d %-2d %-2d} %-10s %-10s",
               fmt->name, types[fmt->type], (unsigned int) fmt->caps,
               fmt->texel_size, IDX4(indices), IDX4(fmt->component_depth),
               IDX4(fmt->host_bits), PL_DEF(fmt->glsl_type, ""),
               PL_DEF(fmt->glsl_format, ""));

#undef IDX4
    }
}

bool ra_fmt_is_ordered(const struct ra_fmt *fmt)
{
    bool ret = !fmt->opaque;
    for (int i = 0; i < fmt->num_components; i++)
        ret &= fmt->sample_order[i] == i;
    return ret;
}

struct glsl_fmt {
    enum ra_fmt_type type;
    int num_components;
    int depth[4];
    const char *glsl_format;
};

// List taken from the GLSL specification. (Yes, GLSL supports only exactly
// these formats with exactly these names)
static const struct glsl_fmt ra_glsl_fmts[] = {
    {RA_FMT_FLOAT, 1, {16},             "r16f"},
    {RA_FMT_FLOAT, 1, {32},             "r32f"},
    {RA_FMT_FLOAT, 2, {16, 16},         "rg16f"},
    {RA_FMT_FLOAT, 2, {32, 32},         "rg32f"},
    {RA_FMT_FLOAT, 4, {16, 16, 16, 16}, "rgba16f"},
    {RA_FMT_FLOAT, 4, {32, 32, 32, 32}, "rgba32f"},
    {RA_FMT_FLOAT, 3, {11, 11, 10},     "r11f_g11f_b10f"},

    {RA_FMT_UNORM, 1, {8},              "r8"},
    {RA_FMT_UNORM, 1, {16},             "r16"},
    {RA_FMT_UNORM, 2, {8,  8},          "rg8"},
    {RA_FMT_UNORM, 2, {16, 16},         "rg16"},
    {RA_FMT_UNORM, 4, {8,  8,  8,  8},  "rgba8"},
    {RA_FMT_UNORM, 4, {16, 16, 16, 16}, "rgba16"},
    {RA_FMT_UNORM, 4, {10, 10, 10,  2}, "rgb10_a2"},

    {RA_FMT_SNORM, 1, {8},              "r8_snorm"},
    {RA_FMT_SNORM, 1, {16},             "r16_snorm"},
    {RA_FMT_SNORM, 2, {8,  8},          "rg8_snorm"},
    {RA_FMT_SNORM, 2, {16, 16},         "rg16_snorm"},
    {RA_FMT_SNORM, 4, {8,  8,  8,  8},  "rgba8_snorm"},
    {RA_FMT_SNORM, 4, {16, 16, 16, 16}, "rgba16_snorm"},

    {RA_FMT_UINT,  1, {8},              "r8ui"},
    {RA_FMT_UINT,  1, {16},             "r16ui"},
    {RA_FMT_UINT,  1, {32},             "r32ui"},
    {RA_FMT_UINT,  2, {8,  8},          "rg8ui"},
    {RA_FMT_UINT,  2, {16, 16},         "rg16ui"},
    {RA_FMT_UINT,  2, {32, 32},         "rg32ui"},
    {RA_FMT_UINT,  4, {8,  8,  8,  8},  "rgba8ui"},
    {RA_FMT_UINT,  4, {16, 16, 16, 16}, "rgba16ui"},
    {RA_FMT_UINT,  4, {32, 32, 32, 32}, "rgba32ui"},
    {RA_FMT_UINT,  4, {10, 10, 10,  2}, "rgb10_a2ui"},

    {RA_FMT_SINT,  1, {8},              "r8i"},
    {RA_FMT_SINT,  1, {16},             "r16i"},
    {RA_FMT_SINT,  1, {32},             "r32i"},
    {RA_FMT_SINT,  2, {8,  8},          "rg8i"},
    {RA_FMT_SINT,  2, {16, 16},         "rg16i"},
    {RA_FMT_SINT,  2, {32, 32},         "rg32i"},
    {RA_FMT_SINT,  4, {8,  8,  8,  8},  "rgba8i"},
    {RA_FMT_SINT,  4, {16, 16, 16, 16}, "rgba16i"},
    {RA_FMT_SINT,  4, {32, 32, 32, 32}, "rgba32i"},
};

const char *ra_fmt_glsl_format(const struct ra_fmt *fmt)
{
    if (fmt->opaque)
        return NULL;

    for (int n = 0; n < PL_ARRAY_SIZE(ra_glsl_fmts); n++) {
        const struct glsl_fmt *gfmt = &ra_glsl_fmts[n];

        if (fmt->type != gfmt->type)
            continue;
        if (fmt->num_components != gfmt->num_components)
            continue;

        // The component order is irrelevant, so we need to sort the depth
        // based on the component's index
        int depth[4] = {0};
        for (int i = 0; i < fmt->num_components; i++)
            depth[fmt->sample_order[i]] = fmt->component_depth[i];

        for (int i = 0; i < PL_ARRAY_SIZE(depth); i++) {
            if (depth[i] != gfmt->depth[i])
                goto next_fmt;
        }

        return gfmt->glsl_format;

next_fmt: ; // equivalent to `continue`
    }

    return NULL;
}

const struct ra_fmt *ra_find_fmt(const struct ra *ra, enum ra_fmt_type type,
                                 int num_components, int min_depth,
                                 int host_bits, enum ra_fmt_caps caps)
{
    for (int n = 0; n < ra->num_formats; n++) {
        const struct ra_fmt *fmt = ra->formats[n];
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
        if (host_bits && !ra_fmt_is_ordered(fmt))
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
    PL_DEBUG(ra, "No matching format found");
    return NULL;
}

const struct ra_fmt *ra_find_vertex_fmt(const struct ra *ra,
                                        enum ra_fmt_type type, int comps)
{
    static const size_t sizes[] = {
        [RA_FMT_FLOAT] = sizeof(float),
        [RA_FMT_UNORM] = sizeof(unsigned),
        [RA_FMT_UINT]  = sizeof(unsigned),
        [RA_FMT_SNORM] = sizeof(int),
        [RA_FMT_SINT]  = sizeof(int),
    };

    return ra_find_fmt(ra, type, comps, 0, 8 * sizes[type], RA_FMT_CAP_VERTEX);
}

const struct ra_fmt *ra_find_named_fmt(const struct ra *ra, const char *name)
{
    if (!name)
        return NULL;

    for (int i = 0; i < ra->num_formats; i++) {
        const struct ra_fmt *fmt = ra->formats[i];
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
        pl_assert(params->w > 0);
        pl_assert(params->w <= ra->limits.max_tex_1d_dim);
        pl_assert(!params->renderable);
        break;
    case 2:
        pl_assert(params->w > 0 && params->h > 0);
        pl_assert(params->w <= ra->limits.max_tex_2d_dim);
        pl_assert(params->h <= ra->limits.max_tex_2d_dim);
        break;
    case 3:
        pl_assert(params->w > 0 && params->h > 0 && params->d > 0);
        pl_assert(params->w <= ra->limits.max_tex_3d_dim);
        pl_assert(params->h <= ra->limits.max_tex_3d_dim);
        pl_assert(params->d <= ra->limits.max_tex_3d_dim);
        pl_assert(!params->renderable);
        break;
    }

    const struct ra_fmt *fmt = params->format;
    pl_assert(fmt);
    pl_assert(!params->sampleable || fmt->caps & RA_FMT_CAP_SAMPLEABLE);
    pl_assert(!params->renderable || fmt->caps & RA_FMT_CAP_RENDERABLE);
    pl_assert(!params->storable   || fmt->caps & RA_FMT_CAP_STORABLE);
    pl_assert(!params->blit_src   || fmt->caps & RA_FMT_CAP_BLITTABLE);
    pl_assert(!params->blit_dst   || fmt->caps & RA_FMT_CAP_BLITTABLE);
    pl_assert(params->sample_mode != RA_TEX_SAMPLE_LINEAR || fmt->caps & RA_FMT_CAP_LINEAR);

    return ra->impl->tex_create(ra, params);
}

static bool ra_tex_params_eq(struct ra_tex_params a, struct ra_tex_params b)
{
    return a.w == b.w && a.h == b.h && a.d == b.d &&
           a.format         == b.format &&
           a.sampleable     == b.sampleable &&
           a.renderable     == b.renderable &&
           a.storable       == b.storable &&
           a.blit_src       == b.blit_src &&
           a.blit_dst       == b.blit_dst &&
           a.host_writable  == b.host_writable &&
           a.host_readable  == b.host_readable &&
           a.sample_mode    == b.sample_mode &&
           a.address_mode   == b.address_mode;
}

bool ra_tex_recreate(const struct ra *ra, const struct ra_tex **tex,
                     const struct ra_tex_params *params)
{
    if (*tex && ra_tex_params_eq((*tex)->params, *params))
        return true;

    PL_DEBUG(ra, "ra_tex_recreate: %dx%dx%d", params->w, params->h,params->d);
    ra_tex_destroy(ra, tex);
    *tex = ra_tex_create(ra, params);

    return !!*tex;
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
    pl_assert(dst->params.blit_dst);

    ra_tex_invalidate(ra, dst);
    ra->impl->tex_clear(ra, dst, color);
}

void ra_tex_invalidate(const struct ra *ra, const struct ra_tex *tex)
{
    ra->impl->tex_invalidate(ra, tex);
}

static void strip_coords(const struct ra_tex *tex, struct pl_rect3d *rc)
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

void ra_tex_blit(const struct ra *ra,
                 const struct ra_tex *dst, const struct ra_tex *src,
                 struct pl_rect3d dst_rc, struct pl_rect3d src_rc)
{
    const struct ra_fmt *src_fmt = src->params.format;
    const struct ra_fmt *dst_fmt = dst->params.format;
    pl_assert(src_fmt->texel_size == dst_fmt->texel_size);
    pl_assert((src_fmt->type == RA_FMT_UINT) == (dst_fmt->type == RA_FMT_UINT));
    pl_assert((src_fmt->type == RA_FMT_SINT) == (dst_fmt->type == RA_FMT_SINT));
    pl_assert(src->params.blit_src);
    pl_assert(dst->params.blit_dst);
    pl_assert(src_rc.x0 >= 0 && src_rc.x0 < src->params.w);
    pl_assert(src_rc.y0 >= 0 && src_rc.y0 < src->params.h);
    pl_assert(src_rc.z0 >= 0 && src_rc.z0 < src->params.d);
    pl_assert(src_rc.x1 > 0 && src_rc.x1 <= src->params.w);
    pl_assert(src_rc.y1 > 0 && src_rc.y1 <= src->params.h);
    pl_assert(src_rc.z1 > 0 && src_rc.z1 <= src->params.d);
    pl_assert(dst_rc.x0 >= 0 && dst_rc.x0 < dst->params.w);
    pl_assert(dst_rc.y0 >= 0 && dst_rc.y0 < dst->params.h);
    pl_assert(dst_rc.z0 >= 0 && dst_rc.z0 < dst->params.d);
    pl_assert(dst_rc.x1 > 0 && dst_rc.x1 <= dst->params.w);
    pl_assert(dst_rc.y1 > 0 && dst_rc.y1 <= dst->params.h);
    pl_assert(dst_rc.z1 > 0 && dst_rc.z1 <= dst->params.d);

    strip_coords(src, &src_rc);
    strip_coords(dst, &dst_rc);

    struct pl_rect3d full = {0, 0, 0, dst->params.w, dst->params.h, dst->params.d};
    strip_coords(dst, &full);

    struct pl_rect3d rcnorm = dst_rc;
    pl_rect3d_normalize(&rcnorm);
    if (pl_rect3d_eq(rcnorm, full))
        ra_tex_invalidate(ra, dst);

    ra->impl->tex_blit(ra, dst, src, dst_rc, src_rc);
}

size_t ra_tex_transfer_size(const struct ra_tex_transfer_params *par)
{
    const struct ra_tex *tex = par->tex;

    int texels;
    switch (ra_tex_params_dimension(tex->params)) {
    case 1: texels = pl_rect_w(par->rc); break;
    case 2: texels = pl_rect_h(par->rc) * par->stride_w; break;
    case 3: texels = pl_rect_d(par->rc) * par->stride_w * par->stride_h; break;
    }

    return texels * tex->params.format->texel_size;
}

static void fix_tex_transfer(const struct ra *ra,
                             struct ra_tex_transfer_params *params)
{
    const struct ra_tex *tex = params->tex;
    struct pl_rect3d rc = params->rc;

    // Infer the default values
    if (!rc.x0 && !rc.x1)
        rc.x1 = tex->params.w;
    if (!rc.y0 && !rc.y1)
        rc.y1 = tex->params.h;
    if (!rc.z0 && !rc.z1)
        rc.z1 = tex->params.d;

    if (!params->stride_w)
        params->stride_w = tex->params.w;
    if (!params->stride_h)
        params->stride_h = tex->params.h;

    // Check the parameters for sanity
#ifndef NDEBUG
    switch (ra_tex_params_dimension(tex->params))
    {
    case 3:
        pl_assert(rc.z1 > rc.z0);
        pl_assert(rc.z0 >= 0 && rc.z0 <  tex->params.d);
        pl_assert(rc.z1 >  0 && rc.z1 <= tex->params.d);
        pl_assert(params->stride_h >= pl_rect_h(rc));
        // fall through
    case 2:
        pl_assert(rc.y1 > rc.y0);
        pl_assert(rc.y0 >= 0 && rc.y0 <  tex->params.h);
        pl_assert(rc.y1 >  0 && rc.y1 <= tex->params.h);
        pl_assert(params->stride_w >= pl_rect_w(rc));
        // fall through
    case 1:
        pl_assert(rc.x1 > rc.x0);
        pl_assert(rc.x0 >= 0 && rc.x0 <  tex->params.w);
        pl_assert(rc.x1 >  0 && rc.x1 <= tex->params.w);
        break;
    }

    pl_assert(!params->buf ^ !params->ptr); // exactly one
    if (params->buf) {
        const struct ra_buf *buf = params->buf;
        size_t size = ra_tex_transfer_size(params);
        pl_assert(params->buf_offset == PL_ALIGN2(params->buf_offset, 4));
        pl_assert(params->buf_offset + size <= buf->params.size);
    }
#endif

    // Sanitize superfluous coordinates for the benefit of the RA
    strip_coords(tex, &rc);
    if (!tex->params.w)
        params->stride_w = 1;
    if (!tex->params.h)
        params->stride_h = 1;

    params->rc = rc;
}

bool ra_tex_upload(const struct ra *ra,
                   const struct ra_tex_transfer_params *params)
{
    const struct ra_tex *tex = params->tex;
    pl_assert(tex);
    pl_assert(tex->params.host_writable);

    struct ra_tex_transfer_params fixed = *params;
    fix_tex_transfer(ra, &fixed);
    return ra->impl->tex_upload(ra, &fixed);
}

bool ra_tex_download(const struct ra *ra,
                     const struct ra_tex_transfer_params *params)
{
    const struct ra_tex *tex = params->tex;
    pl_assert(tex);
    pl_assert(tex->params.host_readable);

    struct ra_tex_transfer_params fixed = *params;
    fix_tex_transfer(ra, &fixed);
    return ra->impl->tex_download(ra, &fixed);
}

const struct ra_buf *ra_buf_create(const struct ra *ra,
                                   const struct ra_buf_params *params)
{
    switch (params->type) {
    case RA_BUF_TEX_TRANSFER:
        pl_assert(ra->limits.max_xfer_size);
        pl_assert(params->size <= ra->limits.max_xfer_size);
        break;
    case RA_BUF_UNIFORM:
        pl_assert(ra->limits.max_ubo_size);
        pl_assert(params->size <= ra->limits.max_ubo_size);
        break;
    case RA_BUF_STORAGE:
        pl_assert(ra->limits.max_ssbo_size);
        pl_assert(params->size <= ra->limits.max_ssbo_size);
        break;
    case RA_BUF_PRIVATE: break;
    default: abort();
    }

    const struct ra_buf *buf = ra->impl->buf_create(ra, params);
    pl_assert(buf->data || !params->host_mapped);
    return buf;
}

void ra_buf_destroy(const struct ra *ra, const struct ra_buf **buf)
{
    if (!*buf)
        return;

    ra->impl->buf_destroy(ra, *buf);
    *buf = NULL;
}

void ra_buf_write(const struct ra *ra, const struct ra_buf *buf,
                  size_t buf_offset, const void *data, size_t size)
{
    pl_assert(buf->params.host_writable);
    pl_assert(buf_offset + size <= buf->params.size);
    pl_assert(buf_offset == PL_ALIGN2(buf_offset, 4));
    ra->impl->buf_write(ra, buf, buf_offset, data, size);
}

bool ra_buf_read(const struct ra *ra, const struct ra_buf *buf,
                 size_t buf_offset, void *dest, size_t size)
{
    pl_assert(buf->params.host_readable);
    pl_assert(buf_offset + size <= buf->params.size);
    pl_assert(buf_offset == PL_ALIGN2(buf_offset, 4));
    return ra->impl->buf_read(ra, buf, buf_offset, dest, size);
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

#define RA_VAR(TYPE, NAME, M, V)                        \
    struct ra_var ra_var_##NAME(const char *name) {     \
        return (struct ra_var) {                        \
            .name  = name,                              \
            .type  = RA_VAR_##TYPE,                     \
            .dim_m = M,                                 \
            .dim_v = V,                                 \
            .dim_a = 1,                                 \
        };                                              \
    }

RA_VAR(UINT,  uint,  1, 1);
RA_VAR(FLOAT, float, 1, 1);
RA_VAR(FLOAT, vec2,  1, 2);
RA_VAR(FLOAT, vec3,  1, 3);
RA_VAR(FLOAT, vec4,  1, 4);
RA_VAR(FLOAT, mat2,  2, 2);
RA_VAR(FLOAT, mat3,  3, 3);
RA_VAR(FLOAT, mat4,  4, 4);

#undef RA_VAR

struct ra_var ra_var_from_fmt(const struct ra_fmt *fmt, const char *name)
{
    static const enum ra_var_type vartypes[] = {
        [RA_FMT_FLOAT] = RA_VAR_FLOAT,
        [RA_FMT_UNORM] = RA_VAR_FLOAT,
        [RA_FMT_SNORM] = RA_VAR_FLOAT,
        [RA_FMT_UINT]  = RA_VAR_UINT,
        [RA_FMT_SINT]  = RA_VAR_SINT,
    };

    pl_assert(fmt->type < PL_ARRAY_SIZE(vartypes));
    return (struct ra_var) {
        .type  = vartypes[fmt->type],
        .name  = name,
        .dim_v = fmt->num_components,
        .dim_m = 1,
        .dim_a = 1,
    };
}

struct ra_var_layout ra_var_host_layout(size_t offset, const struct ra_var *var)
{
    size_t col_size = ra_var_type_size(var->type) * var->dim_v;
    return (struct ra_var_layout) {
        .offset = offset,
        .stride = col_size,
        .size   = col_size * var->dim_m * var->dim_a,
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

bool ra_buf_desc_append(void *tactx, const struct ra *ra,
                        struct ra_desc *buf_desc,
                        struct ra_var_layout *out_layout,
                        const struct ra_var new_var)
{
    struct ra_buffer_var bv = { .var = new_var };
    size_t cur_size = ra_buf_desc_size(buf_desc);

    switch (buf_desc->type) {
    case RA_DESC_BUF_UNIFORM:
        bv.layout = ra_buf_uniform_layout(ra, cur_size, &new_var);
        if (bv.layout.offset + bv.layout.size > ra->limits.max_ubo_size)
            return false;
        break;
    case RA_DESC_BUF_STORAGE:
        bv.layout = ra_buf_storage_layout(ra, cur_size, &new_var);
        if (bv.layout.offset + bv.layout.size > ra->limits.max_ssbo_size)
            return false;
        break;
    default: abort();
    }

    *out_layout = bv.layout;
    TARRAY_APPEND(tactx, buf_desc->buffer_vars, buf_desc->num_buffer_vars, bv);
    return true;
}

size_t ra_buf_desc_size(const struct ra_desc *buf_desc)
{
    if (!buf_desc->num_buffer_vars)
        return 0;

    const struct ra_buffer_var *last;
    last = &buf_desc->buffer_vars[buf_desc->num_buffer_vars - 1];
    return last->layout.offset + last->layout.size;
}

void memcpy_layout(void *dst_p, struct ra_var_layout dst_layout,
                   const void *src_p, struct ra_var_layout src_layout)
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

int ra_desc_namespace(const struct ra *ra, enum ra_desc_type type)
{
    int ret = ra->impl->desc_namespace(ra, type);
    pl_assert(ret >= 0 && ret < RA_DESC_TYPE_COUNT);
    return ret;
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

const struct ra_pass *ra_pass_create(const struct ra *ra,
                                     const struct ra_pass_params *params)
{
    pl_assert(params->glsl_shader);
    switch(params->type) {
    case RA_PASS_RASTER:
        pl_assert(params->vertex_shader);
        for (int i = 0; i < params->num_vertex_attribs; i++) {
            struct ra_vertex_attrib va = params->vertex_attribs[i];
            pl_assert(va.name);
            pl_assert(va.fmt);
            pl_assert(va.fmt->caps & RA_FMT_CAP_VERTEX);
            pl_assert(va.offset + va.fmt->texel_size <= params->vertex_stride);
        }

        const struct ra_fmt *target_fmt = params->target_dummy.params.format;
        pl_assert(target_fmt);
        pl_assert(target_fmt->caps & RA_FMT_CAP_RENDERABLE);
        pl_assert(!params->enable_blend || target_fmt->caps & RA_FMT_CAP_BLENDABLE);
        break;
    case RA_PASS_COMPUTE:
        pl_assert(ra->caps & RA_CAP_COMPUTE);
        break;
    default: abort();
    }

    for (int i = 0; i < params->num_variables; i++) {
        pl_assert(ra->caps & RA_CAP_INPUT_VARIABLES);
        struct ra_var var = params->variables[i];
        pl_assert(var.name);
        pl_assert(ra_var_glsl_type_name(var));
    }

    for (int i = 0; i < params->num_descriptors; i++) {
        struct ra_desc desc = params->descriptors[i];
        pl_assert(desc.name);
        // TODO: enforce disjoint bindings if possible?
    }

    pl_assert(params->push_constants_size <= ra->limits.max_pushc_size);
    pl_assert(params->push_constants_size == PL_ALIGN2(params->push_constants_size, 4));

    return ra->impl->pass_create(ra, params);
}

void ra_pass_destroy(const struct ra *ra, const struct ra_pass **pass)
{
    if (!*pass)
        return;

    ra->impl->pass_destroy(ra, *pass);
    *pass = NULL;
}

static bool ra_tex_params_compat(const struct ra_tex_params a,
                                const struct ra_tex_params b)
{
    return a.format         == b.format &&
           a.sampleable     == b.sampleable &&
           a.renderable     == b.renderable &&
           a.storable       == b.storable &&
           a.blit_src       == b.blit_src &&
           a.blit_dst       == b.blit_dst &&
           a.host_writable  == b.host_writable &&
           a.host_readable  == b.host_readable &&
           a.sample_mode    == b.sample_mode &&
           a.address_mode   == b.address_mode;
}

void ra_pass_run(const struct ra *ra, const struct ra_pass_run_params *params)
{
    const struct ra_pass *pass = params->pass;
    struct ra_pass_run_params new = *params;

    // Sanitize viewport/scissors
    if (!new.viewport.x0 && !new.viewport.x1)
        new.viewport.x1 = params->target->params.w;
    if (!new.viewport.y0 && !new.viewport.y1)
        new.viewport.y1 = params->target->params.h;

    if (!new.scissors.x0 && !new.scissors.x1)
        new.scissors.x1 = params->target->params.w;
    if (!new.scissors.y0 && !new.scissors.y1)
        new.scissors.y1 = params->target->params.h;

    for (int i = 0; i < pass->params.num_descriptors; i++) {
        struct ra_desc desc = pass->params.descriptors[i];
        struct ra_desc_binding db = params->desc_bindings[i];
        pl_assert(db.object);
        switch (desc.type) {
        case RA_DESC_SAMPLED_TEX: {
            const struct ra_tex *tex = db.object;
            pl_assert(tex->params.sampleable);
            break;
        }
        case RA_DESC_STORAGE_IMG: {
            const struct ra_tex *tex = db.object;
            pl_assert(tex->params.storable);
            break;
        }
        case RA_DESC_BUF_UNIFORM: {
            const struct ra_buf *buf = db.object;
            pl_assert(buf->params.type == RA_BUF_UNIFORM);
            break;
        }
        case RA_DESC_BUF_STORAGE: {
            const struct ra_buf *buf = db.object;
            pl_assert(buf->params.type == RA_BUF_STORAGE);
            break;
        }
        default: abort();
        }
    }

    for (int i = 0; i < params->num_var_updates; i++) {
        struct ra_var_update vu = params->var_updates[i];
        pl_assert(ra->caps & RA_CAP_INPUT_VARIABLES);
        pl_assert(vu.index >= 0 && vu.index < pass->params.num_variables);
        pl_assert(vu.data);
    }

    pl_assert(params->push_constants || !pass->params.push_constants_size);

    switch (pass->params.type) {
    case RA_PASS_RASTER: {
        pl_assert(params->vertex_data);
        switch (pass->params.vertex_type) {
        case RA_PRIM_TRIANGLE_LIST:
            pl_assert(params->vertex_count % 3 == 0);
            // fall through
        case RA_PRIM_TRIANGLE_STRIP:
        case RA_PRIM_TRIANGLE_FAN:
            pl_assert(params->vertex_count >= 3);
            break;
        }

        const struct ra_tex *tex = params->target;
        pl_assert(tex);
        pl_assert(ra_tex_params_dimension(tex->params) == 2);
        pl_assert(ra_tex_params_compat(tex->params, pass->params.target_dummy.params));
        pl_assert(tex->params.renderable);
        struct pl_rect2d vp = new.viewport;
        struct pl_rect2d sc = new.scissors;
        pl_assert(pl_rect_w(vp) > 0);
        pl_assert(pl_rect_h(vp) > 0);
        pl_assert(pl_rect_w(sc) > 0);
        pl_assert(pl_rect_h(sc) > 0);
        break;
    }
    case RA_PASS_COMPUTE:
        for (int i = 0; i < PL_ARRAY_SIZE(params->compute_groups); i++) {
            pl_assert(params->compute_groups[i] >= 0);
            pl_assert(params->compute_groups[i] <= ra->limits.max_dispatch[i]);
        }
        break;
    default: abort();
    }

    if (params->target && !pass->params.load_target)
        ra_tex_invalidate(ra, params->target);

    return ra->impl->pass_run(ra, &new);
}

void ra_flush(const struct ra *ra)
{
    if (ra->impl->flush)
        ra->impl->flush(ra);
}

// RA-internal helpers

struct ra_var_layout std140_layout(const struct ra *ra, size_t offset,
                                   const struct ra_var *var)
{
    size_t el_size = ra_var_type_size(var->type);

    // std140 packing rules:
    // 1. The size of generic values is their size in bytes
    // 2. The size of vectors is the vector length * the base count, with the
    // exception of *vec3 which is always the same size as *vec4
    // 3. Matrices are treated like arrays of column vectors
    // 4. The size of array rows is that of the element size rounded up to
    // the nearest multiple of vec4
    // 5. All values are aligned to a multiple of their size (stride for arrays)
    size_t size = el_size * var->dim_v;
    if (var->dim_v == 3)
        size += el_size;
    if (var->dim_m * var->dim_a > 1)
        size = PL_ALIGN2(size, sizeof(float[4]));

    return (struct ra_var_layout) {
        .offset = PL_ALIGN2(offset, size),
        .stride = size,
        .size   = size * var->dim_m * var->dim_a,
    };
}

struct ra_var_layout std430_layout(const struct ra *ra, size_t offset,
                                   const struct ra_var *var)
{
    size_t el_size = ra_var_type_size(var->type);

    // std430 packing rules: like std140, except arrays/matrices are always
    // "tightly" packed, even arrays/matrices of vec3s
    size_t size = el_size * var->dim_v;
    if (var->dim_v == 3 && var->dim_m == 1 && var->dim_a == 1)
        size += el_size;

    return (struct ra_var_layout) {
        .offset = PL_ALIGN2(offset, size),
        .stride = size,
        .size   = size * var->dim_m * var->dim_a,
    };
}

void ra_buf_pool_uninit(const struct ra *ra, struct ra_buf_pool *pool)
{
    for (int i = 0; i < pool->num_buffers; i++)
        ra_buf_destroy(ra, &pool->buffers[i]);

    talloc_free(pool->buffers);
    *pool = (struct ra_buf_pool){0};
}

static bool ra_buf_params_compatible(const struct ra_buf_params *new,
                                     const struct ra_buf_params *old)
{
    return new->type == old->type &&
           new->size <= old->size &&
           new->host_mapped  == old->host_mapped &&
           new->host_writable == old->host_writable &&
           new->host_readable == old->host_readable;
}

static bool ra_buf_pool_grow(const struct ra *ra, struct ra_buf_pool *pool)
{
    const struct ra_buf *buf = ra_buf_create(ra, &pool->current_params);
    if (!buf)
        return false;

    TARRAY_INSERT_AT(NULL, pool->buffers, pool->num_buffers, pool->index, buf);
    PL_DEBUG(ra, "Resized buffer pool of type %u to size %d",
             pool->current_params.type, pool->num_buffers);
    return true;
}

const struct ra_buf *ra_buf_pool_get(const struct ra *ra,
                                     struct ra_buf_pool *pool,
                                     const struct ra_buf_params *params)
{
    pl_assert(!params->initial_data);

    if (!ra_buf_params_compatible(params, &pool->current_params)) {
        ra_buf_pool_uninit(ra, pool);
        pool->current_params = *params;
    }

    // Make sure we have at least one buffer available
    if (!pool->buffers && !ra_buf_pool_grow(ra, pool))
        return NULL;

    // Make sure the next buffer is available for use
    if (ra_buf_poll(ra, pool->buffers[pool->index], 0) &&
        !ra_buf_pool_grow(ra, pool))
    {
        return NULL;
    }

    const struct ra_buf *buf = pool->buffers[pool->index++];
    pool->index %= pool->num_buffers;

    return buf;
}

bool ra_tex_upload_pbo(const struct ra *ra, struct ra_buf_pool *pbo,
                       const struct ra_tex_transfer_params *params)
{
    if (params->buf)
        return ra_tex_upload(ra, params);

    struct ra_buf_params bufparams = {
        .type = RA_BUF_TEX_TRANSFER,
        .size = ra_tex_transfer_size(params),
        .host_writable = true,
    };

    const struct ra_buf *buf = ra_buf_pool_get(ra, pbo, &bufparams);
    if (!buf)
        return false;

    ra_buf_write(ra, buf, 0, params->ptr, bufparams.size);

    struct ra_tex_transfer_params newparams = *params;
    newparams.buf = buf;
    newparams.ptr = NULL;

    return ra_tex_upload(ra, &newparams);
}

bool ra_tex_download_pbo(const struct ra *ra, struct ra_buf_pool *pbo,
                         const struct ra_tex_transfer_params *params)
{
    if (params->buf)
        return ra_tex_download(ra, params);

    struct ra_buf_params bufparams = {
        .type = RA_BUF_TEX_TRANSFER,
        .size = ra_tex_transfer_size(params),
        .host_readable = true,
    };

    const struct ra_buf *buf = ra_buf_pool_get(ra, pbo, &bufparams);
    if (!buf)
        return false;

    struct ra_tex_transfer_params newparams = *params;
    newparams.buf = buf;
    newparams.ptr = NULL;

    if (!ra_tex_download(ra, &newparams))
        return false;

    if (ra_buf_poll(ra, buf, 0)) {
        PL_TRACE(ra, "ra_tex_download without buffer: blocking (slow path)");
        while (ra_buf_poll(ra, buf, 1000000)) ; // 1 ms
    }

    return ra_buf_read(ra, buf, 0, params->ptr, bufparams.size);
}

struct ra_pass_params ra_pass_params_copy(void *tactx,
                                          const struct ra_pass_params *params)
{
    struct ra_pass_params new = *params;
    new.target_dummy.priv = NULL;
    new.cached_program = NULL;
    new.cached_program_len = 0;

    new.glsl_shader = talloc_strdup(tactx, new.glsl_shader);
    new.vertex_shader = talloc_strdup(tactx, new.vertex_shader);

#define DUPSTRS(name, array, num)                                       \
    do {                                                                \
        (array) = TARRAY_DUP(tactx, array, num);                        \
        for (int j = 0; j < num; j++)                                   \
            (array)[j].name = talloc_strdup(tactx, (array)[j].name);    \
    } while (0)

    DUPSTRS(name, new.variables,      new.num_variables);
    DUPSTRS(name, new.descriptors,    new.num_descriptors);
    DUPSTRS(name, new.vertex_attribs, new.num_vertex_attribs);

    for (int i = 0; i < new.num_descriptors; i++) {
        struct ra_desc *desc = &new.descriptors[i];
        DUPSTRS(var.name, desc->buffer_vars, desc->num_buffer_vars);
    }

#undef DUPNAMES

    return new;
}
