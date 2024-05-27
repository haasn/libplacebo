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

#include <math.h>

#include "common.h"
#include "shaders.h"
#include "gpu.h"

// GPU-internal helpers

static int cmp_fmt(const void *pa, const void *pb)
{
    pl_fmt a = *(pl_fmt *)pa;
    pl_fmt b = *(pl_fmt *)pb;

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

#define FMT_BOOL(letter, cap) ((cap) ? (letter) : '-')
#define FMT_IDX4(f) (f)[0], (f)[1], (f)[2], (f)[3]

static void print_formats(pl_gpu gpu)
{
    if (!pl_msg_test(gpu->log, PL_LOG_DEBUG))
        return;

#define CAP_HEADER "%-12s"
#define CAP_FIELDS "%c%c%c%c%c%c%c%c%c%c%c%c"
#define CAP_VALUES \
    FMT_BOOL('S', fmt->caps & PL_FMT_CAP_SAMPLEABLE),       \
    FMT_BOOL('s', fmt->caps & PL_FMT_CAP_STORABLE),         \
    FMT_BOOL('L', fmt->caps & PL_FMT_CAP_LINEAR),           \
    FMT_BOOL('R', fmt->caps & PL_FMT_CAP_RENDERABLE),       \
    FMT_BOOL('b', fmt->caps & PL_FMT_CAP_BLENDABLE),        \
    FMT_BOOL('B', fmt->caps & PL_FMT_CAP_BLITTABLE),        \
    FMT_BOOL('V', fmt->caps & PL_FMT_CAP_VERTEX),           \
    FMT_BOOL('u', fmt->caps & PL_FMT_CAP_TEXEL_UNIFORM),    \
    FMT_BOOL('t', fmt->caps & PL_FMT_CAP_TEXEL_STORAGE),    \
    FMT_BOOL('H', fmt->caps & PL_FMT_CAP_HOST_READABLE),    \
    FMT_BOOL('W', fmt->caps & PL_FMT_CAP_READWRITE),        \
    FMT_BOOL('G', fmt->gatherable)

    PL_DEBUG(gpu,  "GPU texture formats:");
    PL_DEBUG(gpu,  "    %-20s %-6s %-4s %-4s " CAP_HEADER " %-3s %-13s %-13s %-10s %-10s %-6s",
            "NAME", "TYPE", "SIZE", "COMP", "CAPS", "EMU", "DEPTH", "HOST_BITS",
            "GLSL_TYPE", "GLSL_FMT", "FOURCC");
    for (int n = 0; n < gpu->num_formats; n++) {
        pl_fmt fmt = gpu->formats[n];

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


        PL_DEBUG(gpu, "    %-20s %-6s %-4zu %c%c%c%c " CAP_FIELDS " %-3s "
                 "{%-2d %-2d %-2d %-2d} {%-2d %-2d %-2d %-2d} %-10s %-10s %-6s",
                 fmt->name, types[fmt->type], fmt->texel_size,
                 FMT_IDX4(indices), CAP_VALUES, fmt->emulated ? "y" : "n",
                 FMT_IDX4(fmt->component_depth), FMT_IDX4(fmt->host_bits),
                 PL_DEF(fmt->glsl_type, ""), PL_DEF(fmt->glsl_format, ""),
                 PRINT_FOURCC(fmt->fourcc));

#undef CAP_HEADER
#undef CAP_FIELDS
#undef CAP_VALUES

        for (int i = 0; i < fmt->num_modifiers; i++) {
            PL_TRACE(gpu, "        modifiers[%d]: %s",
                     i, PRINT_DRM_MOD(fmt->modifiers[i]));
        }
    }
}

pl_gpu pl_gpu_finalize(struct pl_gpu_t *gpu)
{
    // Sort formats
    qsort(gpu->formats, gpu->num_formats, sizeof(pl_fmt), cmp_fmt);

    // Verification
    pl_assert(gpu->limits.max_tex_2d_dim);
    pl_assert(gpu->limits.max_variable_comps || gpu->limits.max_ubo_size);
    pl_assert(gpu->limits.max_ubo_size    <= gpu->limits.max_buf_size);
    pl_assert(gpu->limits.max_ssbo_size   <= gpu->limits.max_buf_size);
    pl_assert(gpu->limits.max_vbo_size    <= gpu->limits.max_buf_size);
    pl_assert(gpu->limits.max_mapped_size <= gpu->limits.max_buf_size);
    pl_assert(gpu->limits.max_mapped_vram <= gpu->limits.max_mapped_size);

    for (int n = 0; n < gpu->num_formats; n++) {
        pl_fmt fmt = gpu->formats[n];
        pl_assert(fmt->name);
        pl_assert(fmt->type);
        pl_assert(fmt->num_components);
        pl_assert(fmt->internal_size);
        pl_assert(fmt->opaque ? !fmt->texel_size : fmt->texel_size);
        pl_assert(!fmt->gatherable || (fmt->caps & PL_FMT_CAP_SAMPLEABLE));
        for (int i = 0; i < fmt->num_components; i++) {
            pl_assert(fmt->component_depth[i]);
            pl_assert(fmt->opaque ? !fmt->host_bits[i] : fmt->host_bits[i]);
        }
        for (int i = 0; i < fmt->num_planes; i++)
            pl_assert(fmt->planes[i].format);

        enum pl_fmt_caps texel_caps = PL_FMT_CAP_VERTEX |
                                      PL_FMT_CAP_TEXEL_UNIFORM |
                                      PL_FMT_CAP_TEXEL_STORAGE;

        if (fmt->caps & texel_caps) {
            pl_assert(fmt->glsl_type);
            pl_assert(!fmt->opaque);
        }
        if (!fmt->opaque) {
            pl_assert(fmt->texel_size && fmt->texel_align);
            pl_assert((fmt->texel_size % fmt->texel_align) == 0);
            pl_assert(fmt->internal_size == fmt->texel_size || fmt->emulated);
        } else {
            pl_assert(!fmt->texel_size && !fmt->texel_align);
            pl_assert(!(fmt->caps & PL_FMT_CAP_HOST_READABLE));
        }

        // Assert uniqueness of name
        for (int o = n + 1; o < gpu->num_formats; o++)
            pl_assert(strcmp(fmt->name, gpu->formats[o]->name) != 0);
    }

    // Print info
    PL_INFO(gpu, "GPU information:");

#define LOG(fmt, field) \
    PL_INFO(gpu, "      %-26s %" fmt, #field ":", gpu->LOG_STRUCT.field)

#define LOG_STRUCT glsl
    PL_INFO(gpu, "    GLSL version: %d%s", gpu->glsl.version,
           gpu->glsl.vulkan ? " (vulkan)" : gpu->glsl.gles ? " es" : "");
    if (gpu->glsl.compute) {
        LOG("zu", max_shmem_size);
        LOG(PRIu32, max_group_threads);
        LOG(PRIu32, max_group_size[0]);
        LOG(PRIu32, max_group_size[1]);
        LOG(PRIu32, max_group_size[2]);
    }
    LOG(PRIu32, subgroup_size);
    LOG(PRIi16, min_gather_offset);
    LOG(PRIi16, max_gather_offset);
#undef LOG_STRUCT

#define LOG_STRUCT limits
    PL_INFO(gpu, "    Limits:");
    // pl_gpu
    LOG("d", thread_safe);
    LOG("d", callbacks);
    // pl_buf
    LOG("zu", max_buf_size);
    LOG("zu", max_ubo_size);
    LOG("zu", max_ssbo_size);
    LOG("zu", max_vbo_size);
    LOG("zu", max_mapped_size);
    LOG(PRIu64, max_buffer_texels);
    LOG("zu", align_host_ptr);
    LOG("d", host_cached);
    // pl_tex
    LOG(PRIu32, max_tex_1d_dim);
    LOG(PRIu32, max_tex_2d_dim);
    LOG(PRIu32, max_tex_3d_dim);
    LOG("d", blittable_1d_3d);
    LOG("d", buf_transfer);
    LOG("zu", align_tex_xfer_pitch);
    LOG("zu", align_tex_xfer_offset);
    // pl_pass
    LOG("zu", max_variable_comps);
    LOG("zu", max_constants);
    LOG("zu", max_pushc_size);
    LOG("zu", align_vertex_stride);
    if (gpu->glsl.compute) {
        LOG(PRIu32, max_dispatch[0]);
        LOG(PRIu32, max_dispatch[1]);
        LOG(PRIu32, max_dispatch[2]);
    }
    LOG(PRIu32, fragment_queues);
    LOG(PRIu32, compute_queues);
#undef LOG_STRUCT
#undef LOG

    if (pl_gpu_supports_interop(gpu)) {
        PL_INFO(gpu, "    External API interop:");

        PL_INFO(gpu, "      UUID: %s", PRINT_UUID(gpu->uuid));
        PL_INFO(gpu, "      PCI: %04x:%02x:%02x:%x",
                gpu->pci.domain, gpu->pci.bus, gpu->pci.device, gpu->pci.function);
        PL_INFO(gpu, "      buf export caps: 0x%x",
                (unsigned int) gpu->export_caps.buf);
        PL_INFO(gpu, "      buf import caps: 0x%x",
                (unsigned int) gpu->import_caps.buf);
        PL_INFO(gpu, "      tex export caps: 0x%x",
                (unsigned int) gpu->export_caps.tex);
        PL_INFO(gpu, "      tex import caps: 0x%x",
                (unsigned int) gpu->import_caps.tex);
        PL_INFO(gpu, "      sync export caps: 0x%x",
                (unsigned int) gpu->export_caps.sync);
        PL_INFO(gpu, "      sync import caps: 0x%x",
                (unsigned int) gpu->import_caps.sync);
    }

    print_formats(gpu);

    // Finally, create a `pl_dispatch` object for internal operations
    struct pl_gpu_fns *impl = PL_PRIV(gpu);
    atomic_init(&impl->cache, NULL);
    impl->dp = pl_dispatch_create(gpu->log, gpu);
    return gpu;
}

struct glsl_fmt {
    enum pl_fmt_type type;
    int num_components;
    int depth[4];
    const char *format;
    bool glsl;
};

// List taken from the OpenGL specification (4.6, table 8.12).
static const struct glsl_fmt pl_color_fmts[] = {
    {PL_FMT_UNORM, 1, {8}, "r8", true},
    {PL_FMT_SNORM, 1, {8}, "r8_snorm", true},
    {PL_FMT_UNORM, 1, {16}, "r16", true},
    {PL_FMT_SNORM, 1, {16}, "r16_snorm", true},
    {PL_FMT_UNORM, 2, {8, 8}, "rg8", true},
    {PL_FMT_SNORM, 2, {8, 8}, "rg8_snorm", true},
    {PL_FMT_UNORM, 2, {16, 16}, "rg16", true},
    {PL_FMT_SNORM, 2, {16, 16}, "rg16_snorm", true},
    {PL_FMT_UNORM, 3, {3, 3, 2}, "r3_g3_b2"},
    {PL_FMT_UNORM, 3, {4, 4, 4}, "rgb4"},
    {PL_FMT_UNORM, 3, {5, 5, 5}, "rgb5"},
    {PL_FMT_UNORM, 3, {5, 6, 5}, "rgb565"},
    {PL_FMT_UNORM, 3, {8, 8, 8}, "rgb8"},
    {PL_FMT_SNORM, 3, {8, 8, 8}, "rgb8_snorm"},
    {PL_FMT_UNORM, 3, {10, 10, 10}, "rgb10"},
    {PL_FMT_UNORM, 3, {12, 12, 12}, "rgb12"},
    {PL_FMT_UNORM, 3, {16, 16, 16}, "rgb16"},
    {PL_FMT_SNORM, 3, {16, 16, 16}, "rgb16_snorm"},
    {PL_FMT_UNORM, 4, {2, 2, 2, 2}, "rgba2"},
    {PL_FMT_UNORM, 4, {4, 4, 4, 4}, "rgba4"},
    {PL_FMT_UNORM, 4, {5, 5, 5, 1}, "rgb5_a1"},
    {PL_FMT_UNORM, 4, {8, 8, 8, 8}, "rgba8", true},
    {PL_FMT_SNORM, 4, {8, 8, 8, 8}, "rgba8_snorm", true},
    {PL_FMT_UNORM, 4, {10, 10, 10, 2}, "rgb10_a2", true},
    {PL_FMT_UINT, 4, {10, 10, 10, 2}, "rgb10_a2ui", true},
    {PL_FMT_UNORM, 4, {12, 12, 12, 12}, "rgba12"},
    {PL_FMT_UNORM, 4, {16, 16, 16, 16}, "rgba16", true},
    {PL_FMT_SNORM, 4, {16, 16, 16, 16}, "rgba16_snorm", true},
    // SRGB8 is omitted
    // SRGB8_ALPHA8 is omitted
    {PL_FMT_FLOAT, 1, {16}, "r16f", true},
    {PL_FMT_FLOAT, 2, {16, 16}, "rg16f", true},
    {PL_FMT_FLOAT, 3, {16, 16, 16}, "rgb16f"},
    {PL_FMT_FLOAT, 4, {16, 16, 16, 16}, "rgba16f", true},
    {PL_FMT_FLOAT, 1, {32}, "r32f", true},
    {PL_FMT_FLOAT, 2, {32, 32}, "rg32f", true},
    {PL_FMT_FLOAT, 3, {32, 32, 32}, "rgb32f"},
    {PL_FMT_FLOAT, 4, {32, 32, 32, 32}, "rgba32f", true},
    {PL_FMT_FLOAT, 3, {11, 11, 10}, "r11f_g11f_b10f", true},
    // RGB9_E5 is omitted
    {PL_FMT_SINT, 1, {8}, "r8i", true},
    {PL_FMT_UINT, 1, {8}, "r8ui", true},
    {PL_FMT_SINT, 1, {16}, "r16i", true},
    {PL_FMT_UINT, 1, {16}, "r16ui", true},
    {PL_FMT_SINT, 1, {32}, "r32i", true},
    {PL_FMT_UINT, 1, {32}, "r32ui", true},
    {PL_FMT_SINT, 2, {8, 8}, "rg8i", true},
    {PL_FMT_UINT, 2, {8, 8}, "rg8ui", true},
    {PL_FMT_SINT, 2, {16, 16}, "rg16i", true},
    {PL_FMT_UINT, 2, {16, 16}, "rg16ui", true},
    {PL_FMT_SINT, 2, {32, 32}, "rg32i", true},
    {PL_FMT_UINT, 2, {32, 32}, "rg32ui", true},
    {PL_FMT_SINT, 3, {8, 8, 8}, "rgb8i"},
    {PL_FMT_UINT, 3, {8, 8, 8}, "rgb8ui"},
    {PL_FMT_SINT, 3, {16, 16, 16}, "rgb16i"},
    {PL_FMT_UINT, 3, {16, 16, 16}, "rgb16ui"},
    {PL_FMT_SINT, 3, {32, 32, 32}, "rgb32i"},
    {PL_FMT_UINT, 3, {32, 32, 32}, "rgb32ui"},
    {PL_FMT_SINT, 4, {8, 8, 8, 8}, "rgba8i", true},
    {PL_FMT_UINT, 4, {8, 8, 8, 8}, "rgba8ui", true},
    {PL_FMT_SINT, 4, {16, 16, 16, 16}, "rgba16i", true},
    {PL_FMT_UINT, 4, {16, 16, 16, 16}, "rgba16ui", true},
    {PL_FMT_SINT, 4, {32, 32, 32, 32}, "rgba32i", true},
    {PL_FMT_UINT, 4, {32, 32, 32, 32}, "rgba32ui", true},
};

const char *pl_fmt_color_format(pl_fmt fmt, int components, bool only_glsl)
{
    if (fmt->opaque)
        return NULL;

    for (int n = 0; n < PL_ARRAY_SIZE(pl_color_fmts); n++) {
        const struct glsl_fmt *gfmt = &pl_color_fmts[n];

        if (fmt->type != gfmt->type)
            continue;
        if (components != gfmt->num_components)
            continue;
        if (only_glsl && !gfmt->glsl)
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

        return gfmt->format;

next_fmt: ; // equivalent to `continue`
    }

    return NULL;
}

const char *pl_fmt_glsl_format(pl_fmt fmt, int components)
{
    return pl_fmt_color_format(fmt, components, true);
}

#define FOURCC(a,b,c,d) ((uint32_t)(a)        | ((uint32_t)(b) << 8) | \
                        ((uint32_t)(c) << 16) | ((uint32_t)(d) << 24))

struct pl_fmt_fourcc {
    const char *name;
    uint32_t fourcc;
};

static const struct pl_fmt_fourcc pl_fmt_fourccs[] = {
    // 8 bpp red
    {"r8",          FOURCC('R','8',' ',' ')},
    // 16 bpp red
    {"r16",         FOURCC('R','1','6',' ')},
    // 16 bpp rg
    {"rg8",         FOURCC('G','R','8','8')},
    {"gr8",         FOURCC('R','G','8','8')},
    // 32 bpp rg
    {"rg16",        FOURCC('G','R','3','2')},
    {"gr16",        FOURCC('R','G','3','2')},
    // 8 bpp rgb: N/A
    // 16 bpp rgb
    {"argb4",       FOURCC('B','A','1','2')},
    {"abgr4",       FOURCC('R','A','1','2')},
    {"rgba4",       FOURCC('A','B','1','2')},
    {"bgra4",       FOURCC('A','R','1','2')},

    {"a1rgb5",      FOURCC('B','A','1','5')},
    {"a1bgr5",      FOURCC('R','A','1','5')},
    {"rgb5a1",      FOURCC('A','B','1','5')},
    {"bgr5a1",      FOURCC('A','R','1','5')},

    {"rgb565",      FOURCC('B','G','1','6')},
    {"bgr565",      FOURCC('R','G','1','6')},
    // 24 bpp rgb
    {"rgb8",        FOURCC('B','G','2','4')},
    {"bgr8",        FOURCC('R','G','2','4')},
    // 32 bpp rgb
    {"argb8",       FOURCC('B','A','2','4')},
    {"abgr8",       FOURCC('R','A','2','4')},
    {"rgba8",       FOURCC('A','B','2','4')},
    {"bgra8",       FOURCC('A','R','2','4')},

    {"a2rgb10",     FOURCC('B','A','3','0')},
    {"a2bgr10",     FOURCC('R','A','3','0')},
    {"rgb10a2",     FOURCC('A','B','3','0')},
    {"bgr10a2",     FOURCC('A','R','3','0')},
    // 64bpp rgb
    {"rgba16",      FOURCC('A','B','4','8')},
    {"bgra16",      FOURCC('A','R','4','8')},
    {"rgba16hf",    FOURCC('A','B','4','H')},
    {"bgra16hf",    FOURCC('A','R','4','H')},

    // packed 16-bit formats
    // rx10:        N/A
    // rxgx10:      N/A
    {"rxgxbxax10",  FOURCC('A','B','1','0')},
    // rx12:        N/A
    // rxgx12:      N/A
    // rxgxbxax12:  N/A

    // planar formats
    {"g8_b8_r8_420",    FOURCC('Y','U','1','2')},
    {"g8_b8_r8_422",    FOURCC('Y','U','1','6')},
    {"g8_b8_r8_444",    FOURCC('Y','U','2','4')},
    // g16_b18_r8_*:    N/A
    // gx10_bx10_rx10_42*: N/A
    {"gx10_bx10_rx10_444", FOURCC('Q','4','1','0')},
    // gx12_bx12_rx12_*:N/A
    {"g8_br8_420",      FOURCC('N','V','1','2')},
    {"g8_br8_422",      FOURCC('N','V','1','6')},
    {"g8_br8_444",      FOURCC('N','V','2','4')},
    {"g16_br16_420",    FOURCC('P','0','1','6')},
    // g16_br16_422:    N/A
    // g16_br16_444:    N/A
    {"gx10_bxrx10_420", FOURCC('P','0','1','0')},
    {"gx10_bxrx10_422", FOURCC('P','2','1','0')},
    // gx10_bxrx10_444: N/A
    {"gx12_bxrx12_420", FOURCC('P','0','1','2')},
    // gx12_bxrx12_422: N/A
    // gx12_bxrx12_444: N/A
};

uint32_t pl_fmt_fourcc(pl_fmt fmt)
{
    for (int n = 0; n < PL_ARRAY_SIZE(pl_fmt_fourccs); n++) {
        const struct pl_fmt_fourcc *fourcc = &pl_fmt_fourccs[n];
        if (strcmp(fmt->name, fourcc->name) == 0)
            return fourcc->fourcc;
    }

    return 0; // no matching format
}

size_t pl_tex_transfer_size(const struct pl_tex_transfer_params *par)
{
    int w = pl_rect_w(par->rc), h = pl_rect_h(par->rc), d = pl_rect_d(par->rc);
    size_t pixel_pitch = par->tex->params.format->texel_size;

    // This generates the absolute bare minimum size of a buffer required to
    // hold the data of a texture upload/download, by including stride padding
    // only where strictly necessary.
    return (d - 1) * par->depth_pitch + (h - 1) * par->row_pitch + w * pixel_pitch;
}

int pl_tex_transfer_slices(pl_gpu gpu, pl_fmt texel_fmt,
                           const struct pl_tex_transfer_params *params,
                           struct pl_tex_transfer_params **out_slices)
{
    PL_ARRAY(struct pl_tex_transfer_params) slices = {0};
    size_t max_size = params->buf ? gpu->limits.max_buf_size : SIZE_MAX;

    pl_fmt fmt = params->tex->params.format;
    if (fmt->emulated && texel_fmt) {
        size_t max_texel = gpu->limits.max_buffer_texels * texel_fmt->texel_size;
        max_size = PL_MIN(gpu->limits.max_ssbo_size, max_texel);
    }

    int slice_w = pl_rect_w(params->rc);
    int slice_h = pl_rect_h(params->rc);
    int slice_d = pl_rect_d(params->rc);

    slice_d = PL_MIN(slice_d, max_size / params->depth_pitch);
    if (!slice_d) {
        slice_d = 1;
        slice_h = PL_MIN(slice_h, max_size / params->row_pitch);
        if (!slice_h) {
            slice_h = 1;
            slice_w = PL_MIN(slice_w, max_size / fmt->texel_size);
            pl_assert(slice_w);
        }
    }

    for (int z = 0; z < pl_rect_d(params->rc); z += slice_d) {
        for (int y = 0; y < pl_rect_h(params->rc); y += slice_h) {
            for (int x = 0; x < pl_rect_w(params->rc); x += slice_w) {
                struct pl_tex_transfer_params slice = *params;
                slice.callback = NULL;
                slice.rc.x0 = params->rc.x0 + x;
                slice.rc.y0 = params->rc.y0 + y;
                slice.rc.z0 = params->rc.z0 + z;
                slice.rc.x1 = PL_MIN(slice.rc.x0 + slice_w, params->rc.x1);
                slice.rc.y1 = PL_MIN(slice.rc.y0 + slice_h, params->rc.y1);
                slice.rc.z1 = PL_MIN(slice.rc.z0 + slice_d, params->rc.z1);

                const size_t offset = z * params->depth_pitch +
                                      y * params->row_pitch +
                                      x * fmt->texel_size;
                if (slice.ptr) {
                    slice.ptr = (uint8_t *) slice.ptr + offset;
                } else {
                    slice.buf_offset += offset;
                }

                PL_ARRAY_APPEND(NULL, slices, slice);
            }
        }
    }

    *out_slices = slices.elem;
    return slices.num;
}

bool pl_tex_upload_pbo(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    if (params->buf)
        return pl_tex_upload(gpu, params);

    const size_t size = pl_tex_transfer_size(params);
    struct pl_tex_transfer_params fixed = *params;
    fixed.ptr = NULL;

    // If we can import host pointers directly, and the function is being used
    // asynchronously, then we can use host pointer import to skip a memcpy. In
    // the synchronous case, we still force a host memcpy to avoid stalling the
    // host until the GPU memcpy completes.
    bool can_import = gpu->import_caps.buf & PL_HANDLE_HOST_PTR;
    can_import &= !params->no_import;
    can_import &= params->callback != NULL;
    can_import &= size > (32 << 10); // 32 KiB
    if (can_import) {
        // Suppress errors for this test because it may fail, in which case we
        // want to silently fall back.
        pl_log_level_cap(gpu->log, PL_LOG_DEBUG);
        fixed.buf = pl_buf_create(gpu, pl_buf_params(
            .size = size,
            .import_handle = PL_HANDLE_HOST_PTR,
            .shared_mem = (struct pl_shared_mem) {
                .handle.ptr = params->ptr,
                .size = size,
                .offset = 0,
            },
        ));
        pl_log_level_cap(gpu->log, PL_LOG_NONE);
    }

    if (!fixed.buf) {
        fixed.buf = pl_buf_create(gpu, pl_buf_params(
            .size = size,
            .host_writable = true,
        ));
        if (!fixed.buf)
            return false;
        pl_buf_write(gpu, fixed.buf, 0, params->ptr, size);
        if (params->callback)
            params->callback(params->priv);
        fixed.callback = NULL;
    }

    bool ok = pl_tex_upload(gpu, &fixed);
    pl_buf_destroy(gpu, &fixed.buf);
    return ok;
}

struct pbo_cb_ctx {
    pl_gpu gpu;
    pl_buf buf;
    void *ptr;
    void (*callback)(void *priv);
    void *priv;
};

static void pbo_download_cb(void *priv)
{
    struct pbo_cb_ctx *p = priv;
    pl_buf_read(p->gpu, p->buf, 0, p->ptr, p->buf->params.size);
    pl_buf_destroy(p->gpu, &p->buf);

    // Run the original callback
    p->callback(p->priv);
    pl_free(priv);
};

bool pl_tex_download_pbo(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    if (params->buf)
        return pl_tex_download(gpu, params);

    const size_t size = pl_tex_transfer_size(params);
    pl_buf buf = NULL;

    // If we can import host pointers directly, we can avoid an extra memcpy
    // (sometimes). In the cases where it isn't avoidable, the extra memcpy
    // will happen inside VRAM, which is typically faster anyway.
    bool can_import = gpu->import_caps.buf & PL_HANDLE_HOST_PTR;
    can_import &= !params->no_import;
    can_import &= size > (32 << 10); // 32 KiB
    if (can_import) {
        // Suppress errors for this test because it may fail, in which case we
        // want to silently fall back.
        pl_log_level_cap(gpu->log, PL_LOG_DEBUG);
        buf = pl_buf_create(gpu, pl_buf_params(
            .size = size,
            .import_handle = PL_HANDLE_HOST_PTR,
            .shared_mem = (struct pl_shared_mem) {
                .handle.ptr = params->ptr,
                .size = size,
                .offset = 0,
            },
        ));
        pl_log_level_cap(gpu->log, PL_LOG_NONE);
    }

    if (!buf) {
        // Fallback when host pointer import is not supported
        buf = pl_buf_create(gpu, pl_buf_params(
            .size = size,
            .host_readable = true,
        ));
    }

    if (!buf)
        return false;

    struct pl_tex_transfer_params newparams = *params;
    newparams.ptr = NULL;
    newparams.buf = buf;

    bool import_handle = buf->params.import_handle;
    // If the transfer is asynchronous, propagate our host read asynchronously
    if (params->callback && !import_handle) {
        newparams.callback = pbo_download_cb;
        newparams.priv = pl_alloc_struct(NULL, struct pbo_cb_ctx, {
            .gpu = gpu,
            .buf = buf,
            .ptr = params->ptr,
            .callback = params->callback,
            .priv = params->priv,
        });
    }

    if (!pl_tex_download(gpu, &newparams)) {
        pl_buf_destroy(gpu, &buf);
        return false;
    }

    if (!params->callback) {
        while (pl_buf_poll(gpu, buf, 10000000)) // 10 ms
            PL_TRACE(gpu, "pl_tex_download: synchronous/blocking (slow path)");
    }

    bool ok;
    if (import_handle) {
        // Buffer download completion already means the host pointer contains
        // the valid data, no more need to copy. (Note: this applies even for
        // asynchronous downloads)
        ok = true;
        pl_buf_destroy(gpu, &buf);
    } else if (!params->callback) {
        // Synchronous read back to the host pointer
        ok = pl_buf_read(gpu, buf, 0, params->ptr, size);
        pl_buf_destroy(gpu, &buf);
    } else {
        // Nothing left to do here, the rest will be done by pbo_download_cb
        ok = true;
    }

    return ok;
}

bool pl_tex_upload_texel(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    const int threads = PL_MIN(256, pl_rect_w(params->rc));
    pl_tex tex = params->tex;
    pl_fmt fmt = tex->params.format;
    pl_require(gpu, params->buf);

    pl_dispatch dp = pl_gpu_dispatch(gpu);
    pl_shader sh = pl_dispatch_begin(dp);
    if (!sh_try_compute(sh, threads, 1, false, 0)) {
        PL_ERR(gpu, "Failed emulating texture transfer!");
        pl_dispatch_abort(dp, &sh);
        return false;
    }

    ident_t buf = sh_desc(sh, (struct pl_shader_desc) {
        .binding.object = params->buf,
        .desc = {
            .name = "data",
            .type = PL_DESC_BUF_TEXEL_STORAGE,
        },
    });

    ident_t img = sh_desc(sh, (struct pl_shader_desc) {
        .binding.object = params->tex,
        .desc = {
            .name = "image",
            .type = PL_DESC_STORAGE_IMG,
            .access = PL_DESC_ACCESS_WRITEONLY,
        },
    });

    // If the transfer width is a natural multiple of the thread size, we
    // can skip the bounds check. Otherwise, make sure we aren't blitting out
    // of the range since this would read out of bounds.
    int groups_x = PL_DIV_UP(pl_rect_w(params->rc), threads);
    if (groups_x * threads != pl_rect_w(params->rc)) {
        GLSL("if (gl_GlobalInvocationID.x >= %d) \n"
             "    return;                        \n",
             pl_rect_w(params->rc));
    }

    // fmt->texel_align contains the size of an individual color value
    assert(fmt->texel_size == fmt->num_components * fmt->texel_align);
    GLSL("vec4 color = vec4(0.0, 0.0, 0.0, 1.0);                        \n"
         "ivec3 pos = ivec3(gl_GlobalInvocationID);                     \n"
         "ivec3 tex_pos = pos + ivec3("$", "$", "$");                   \n"
         "int base = "$" + pos.z * "$" + pos.y * "$" + pos.x * "$";     \n",
         SH_INT_DYN(params->rc.x0), SH_INT_DYN(params->rc.y0), SH_INT_DYN(params->rc.z0),
         SH_INT_DYN(params->buf_offset),
         SH_INT(params->depth_pitch / fmt->texel_align),
         SH_INT(params->row_pitch / fmt->texel_align),
         SH_INT(fmt->texel_size / fmt->texel_align));

    for (int i = 0; i < fmt->num_components; i++)
        GLSL("color[%d] = imageLoad("$", base + %d).r; \n", i, buf, i);

    int dims = pl_tex_params_dimension(tex->params);
    static const char *coord_types[] = {
        [1] = "int",
        [2] = "ivec2",
        [3] = "ivec3",
    };

    GLSL("imageStore("$", %s(tex_pos), color);\n", img, coord_types[dims]);
    return pl_dispatch_compute(dp, pl_dispatch_compute_params(
        .shader = &sh,
        .dispatch_size = {
            groups_x,
            pl_rect_h(params->rc),
            pl_rect_d(params->rc),
        },
    ));

error:
    return false;
}

bool pl_tex_download_texel(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    const int threads = PL_MIN(256, pl_rect_w(params->rc));
    pl_tex tex = params->tex;
    pl_fmt fmt = tex->params.format;
    pl_require(gpu, params->buf);

    pl_dispatch dp = pl_gpu_dispatch(gpu);
    pl_shader sh = pl_dispatch_begin(dp);
    if (!sh_try_compute(sh, threads, 1, false, 0)) {
        PL_ERR(gpu, "Failed emulating texture transfer!");
        pl_dispatch_abort(dp, &sh);
        return false;
    }

    ident_t buf = sh_desc(sh, (struct pl_shader_desc) {
        .binding.object = params->buf,
        .desc = {
            .name = "data",
            .type = PL_DESC_BUF_TEXEL_STORAGE,
        },
    });

    ident_t img = sh_desc(sh, (struct pl_shader_desc) {
        .binding.object = params->tex,
        .desc = {
            .name = "image",
            .type = PL_DESC_STORAGE_IMG,
            .access = PL_DESC_ACCESS_READONLY,
        },
    });

    int groups_x = PL_DIV_UP(pl_rect_w(params->rc), threads);
    if (groups_x * threads != pl_rect_w(params->rc)) {
        GLSL("if (gl_GlobalInvocationID.x >= %d) \n"
             "    return;                        \n",
             pl_rect_w(params->rc));
    }

    int dims = pl_tex_params_dimension(tex->params);
    static const char *coord_types[] = {
        [1] = "int",
        [2] = "ivec2",
        [3] = "ivec3",
    };

    assert(fmt->texel_size == fmt->num_components * fmt->texel_align);
    GLSL("ivec3 pos = ivec3(gl_GlobalInvocationID);                     \n"
         "ivec3 tex_pos = pos + ivec3("$", "$", "$");                   \n"
         "int base = "$" + pos.z * "$" + pos.y * "$" + pos.x * "$";     \n"
         "vec4 color = imageLoad("$", %s(tex_pos));                     \n",
         SH_INT_DYN(params->rc.x0), SH_INT_DYN(params->rc.y0), SH_INT_DYN(params->rc.z0),
         SH_INT_DYN(params->buf_offset),
         SH_INT(params->depth_pitch / fmt->texel_align),
         SH_INT(params->row_pitch / fmt->texel_align),
         SH_INT(fmt->texel_size / fmt->texel_align),
         img, coord_types[dims]);

    for (int i = 0; i < fmt->num_components; i++)
        GLSL("imageStore("$", base + %d, vec4(color[%d])); \n", buf, i, i);

    return pl_dispatch_compute(dp, pl_dispatch_compute_params(
        .shader = &sh,
        .dispatch_size = {
            groups_x,
            pl_rect_h(params->rc),
            pl_rect_d(params->rc),
        },
    ));

error:
    return false;
}

bool pl_tex_blit_compute(pl_gpu gpu, const struct pl_tex_blit_params *params)
{
    if (!params->dst->params.storable)
        return false;

    // Normalize `dst_rc`, moving all flipping to `src_rc` instead.
    pl_rect3d src_rc = params->src_rc;
    pl_rect3d dst_rc = params->dst_rc;
    if (pl_rect_w(dst_rc) < 0) {
        PL_SWAP(src_rc.x0, src_rc.x1);
        PL_SWAP(dst_rc.x0, dst_rc.x1);
    }
    if (pl_rect_h(dst_rc) < 0) {
        PL_SWAP(src_rc.y0, src_rc.y1);
        PL_SWAP(dst_rc.y0, dst_rc.y1);
    }
    if (pl_rect_d(dst_rc) < 0) {
        PL_SWAP(src_rc.z0, src_rc.z1);
        PL_SWAP(dst_rc.z0, dst_rc.z1);
    }

    bool needs_scaling = false;
    needs_scaling |= pl_rect_w(dst_rc) != abs(pl_rect_w(src_rc));
    needs_scaling |= pl_rect_h(dst_rc) != abs(pl_rect_h(src_rc));
    needs_scaling |= pl_rect_d(dst_rc) != abs(pl_rect_d(src_rc));

    // Exception: fast path for 1-pixel blits, which don't require scaling
    bool is_1pixel = abs(pl_rect_w(src_rc)) == 1 && abs(pl_rect_h(src_rc)) == 1;
    needs_scaling &= !is_1pixel;

    // Manual trilinear interpolation would be too slow to justify
    bool needs_sampling = needs_scaling && params->sample_mode != PL_TEX_SAMPLE_NEAREST;
    needs_sampling |= !params->src->params.storable;
    if (needs_sampling && !params->src->params.sampleable)
        return false;

    const int threads = 256;
    int bw = PL_MIN(32, pl_rect_w(dst_rc));
    int bh = PL_MIN(threads / bw, pl_rect_h(dst_rc));
    pl_dispatch dp = pl_gpu_dispatch(gpu);
    pl_shader sh = pl_dispatch_begin(dp);
    if (!sh_try_compute(sh, bw, bh, false, 0)) {
        pl_dispatch_abort(dp, &sh);
        return false;
    }

    // Avoid over-writing into `dst`
    int groups_x = PL_DIV_UP(pl_rect_w(dst_rc), bw);
    if (groups_x * bw != pl_rect_w(dst_rc)) {
        GLSL("if (gl_GlobalInvocationID.x >= %d) \n"
             "    return;                        \n",
             pl_rect_w(dst_rc));
    }

    int groups_y = PL_DIV_UP(pl_rect_h(dst_rc), bh);
    if (groups_y * bh != pl_rect_h(dst_rc)) {
        GLSL("if (gl_GlobalInvocationID.y >= %d) \n"
             "    return;                        \n",
             pl_rect_h(dst_rc));
    }

    ident_t dst = sh_desc(sh, (struct pl_shader_desc) {
        .binding.object = params->dst,
        .desc = {
            .name   = "dst",
            .type   = PL_DESC_STORAGE_IMG,
            .access = PL_DESC_ACCESS_WRITEONLY,
        },
    });

    static const char *vecs[] = {
        [1] = "float",
        [2] = "vec2",
        [3] = "vec3",
        [4] = "vec4",
    };

    static const char *ivecs[] = {
        [1] = "int",
        [2] = "ivec2",
        [3] = "ivec3",
        [4] = "ivec4",
    };

    int src_dims = pl_tex_params_dimension(params->src->params);
    int dst_dims = pl_tex_params_dimension(params->dst->params);
    GLSL("ivec3 pos = ivec3(gl_GlobalInvocationID); \n"
         "%s dst_pos = %s(pos + ivec3(%d, %d, %d)); \n",
         ivecs[dst_dims], ivecs[dst_dims],
         params->dst_rc.x0, params->dst_rc.y0, params->dst_rc.z0);

    if (needs_sampling || (needs_scaling && params->src->params.sampleable)) {

        ident_t src = sh_desc(sh, (struct pl_shader_desc) {
            .desc = {
                .name = "src",
                .type = PL_DESC_SAMPLED_TEX,
            },
            .binding = {
                .object = params->src,
                .address_mode = PL_TEX_ADDRESS_CLAMP,
                .sample_mode = params->sample_mode,
            }
        });

        if (is_1pixel) {
            GLSL("%s fpos = %s(0.5); \n", vecs[src_dims], vecs[src_dims]);
        } else {
            GLSL("vec3 fpos = (vec3(pos) + vec3(0.5)) / vec3(%d.0, %d.0, %d.0); \n",
                 pl_rect_w(dst_rc), pl_rect_h(dst_rc), pl_rect_d(dst_rc));
        }

        GLSL("%s src_pos = %s(0.5);             \n"
             "src_pos.x = mix(%f, %f, fpos.x);  \n",
             vecs[src_dims], vecs[src_dims],
             (float) src_rc.x0 / params->src->params.w,
             (float) src_rc.x1 / params->src->params.w);

        if (params->src->params.h) {
            GLSL("src_pos.y = mix(%f, %f, fpos.y); \n",
                 (float) src_rc.y0 / params->src->params.h,
                 (float) src_rc.y1 / params->src->params.h);
        }

        if (params->src->params.d) {
            GLSL("src_pos.z = mix(%f, %f, fpos.z); \n",
                 (float) src_rc.z0 / params->src->params.d,
                 (float) src_rc.z1 / params->src->params.d);
        }

        GLSL("imageStore("$", dst_pos, textureLod("$", src_pos, 0.0)); \n",
             dst, src);

    } else {

        ident_t src = sh_desc(sh, (struct pl_shader_desc) {
            .binding.object = params->src,
            .desc = {
                .name   = "src",
                .type   = PL_DESC_STORAGE_IMG,
                .access = PL_DESC_ACCESS_READONLY,
            },
        });

        if (is_1pixel) {
            GLSL("ivec3 src_pos = ivec3(0); \n");
        } else if (needs_scaling) {
            GLSL("ivec3 src_pos = ivec3(vec3(%f, %f, %f) * vec3(pos)); \n",
                 fabs((float) pl_rect_w(src_rc) / pl_rect_w(dst_rc)),
                 fabs((float) pl_rect_h(src_rc) / pl_rect_h(dst_rc)),
                 fabs((float) pl_rect_d(src_rc) / pl_rect_d(dst_rc)));
        } else {
            GLSL("ivec3 src_pos = pos; \n");
        }

        GLSL("src_pos = ivec3(%d, %d, %d) * src_pos + ivec3(%d, %d, %d);    \n"
             "imageStore("$", dst_pos, imageLoad("$", %s(src_pos)));        \n",
             src_rc.x1 < src_rc.x0 ? -1 : 1,
             src_rc.y1 < src_rc.y0 ? -1 : 1,
             src_rc.z1 < src_rc.z0 ? -1 : 1,
             src_rc.x0, src_rc.y0, src_rc.z0,
             dst, src, ivecs[src_dims]);

    }

    return pl_dispatch_compute(dp, pl_dispatch_compute_params(
        .shader = &sh,
        .dispatch_size = {
            groups_x,
            groups_y,
            pl_rect_d(dst_rc),
        },
    ));
}

void pl_tex_blit_raster(pl_gpu gpu, const struct pl_tex_blit_params *params)
{
    enum pl_fmt_type src_type = params->src->params.format->type;
    enum pl_fmt_type dst_type = params->dst->params.format->type;

    // Only for 2D textures
    pl_assert(params->src->params.h && !params->src->params.d);
    pl_assert(params->dst->params.h && !params->dst->params.d);

    // Integer textures are not supported
    pl_assert(src_type != PL_FMT_UINT && src_type != PL_FMT_SINT);
    pl_assert(dst_type != PL_FMT_UINT && dst_type != PL_FMT_SINT);

    pl_rect2df src_rc = {
        .x0 = params->src_rc.x0, .x1 = params->src_rc.x1,
        .y0 = params->src_rc.y0, .y1 = params->src_rc.y1,
    };
    pl_rect2d dst_rc = {
        .x0 = params->dst_rc.x0, .x1 = params->dst_rc.x1,
        .y0 = params->dst_rc.y0, .y1 = params->dst_rc.y1,
    };

    pl_dispatch dp = pl_gpu_dispatch(gpu);
    pl_shader sh = pl_dispatch_begin(dp);
    sh->output = PL_SHADER_SIG_COLOR;

    ident_t pos, src = sh_bind(sh, params->src, PL_TEX_ADDRESS_CLAMP,
        params->sample_mode, "src_tex", &src_rc, &pos, NULL);

    GLSL("vec4 color = textureLod("$", "$", 0.0); \n", src, pos);

    pl_dispatch_finish(dp, pl_dispatch_params(
        .shader = &sh,
        .target = params->dst,
        .rect = dst_rc,
    ));
}

bool pl_buf_copy_swap(pl_gpu gpu, const struct pl_buf_copy_swap_params *params)
{
    pl_buf src = params->src, dst = params->dst;
    pl_require(gpu, src->params.storable && dst->params.storable);
    pl_require(gpu, params->src_offset % sizeof(unsigned) == 0);
    pl_require(gpu, params->dst_offset % sizeof(unsigned) == 0);
    pl_require(gpu, params->src_offset + params->size <= src->params.size);
    pl_require(gpu, params->dst_offset + params->size <= dst->params.size);
    pl_require(gpu, src != dst || params->src_offset == params->dst_offset);
    pl_require(gpu, params->size % sizeof(unsigned) == 0);
    pl_require(gpu, params->wordsize == sizeof(uint16_t) ||
                    params->wordsize == sizeof(uint32_t));

    const size_t words = params->size / sizeof(unsigned);
    const size_t src_off = params->src_offset / sizeof(unsigned);
    const size_t dst_off = params->dst_offset / sizeof(unsigned);

    const int threads = PL_MIN(256, words);
    pl_dispatch dp = pl_gpu_dispatch(gpu);
    pl_shader sh = pl_dispatch_begin(dp);
    if (!sh_try_compute(sh, threads, 1, false, 0)) {
        pl_dispatch_abort(dp, &sh);
        return false;
    }

    const size_t groups = PL_DIV_UP(words, threads);
    if (groups * threads > words) {
        GLSL("if (gl_GlobalInvocationID.x >= %zu) \n"
             "    return;                         \n",
             words);
    }

    sh_desc(sh, (struct pl_shader_desc) {
        .binding.object = src,
        .desc = {
            .name = "SrcBuf",
            .type = PL_DESC_BUF_STORAGE,
            .access = src == dst ? PL_DESC_ACCESS_READWRITE : PL_DESC_ACCESS_READONLY,
        },
        .num_buffer_vars = 1,
        .buffer_vars = &(struct pl_buffer_var) {
            .var = {
                .name = "src",
                .type = PL_VAR_UINT,
                .dim_v = 1,
                .dim_m = 1,
                .dim_a = src_off + words,
            },
        },
    });

    if (src != dst) {
        sh_desc(sh, (struct pl_shader_desc) {
            .binding.object = dst,
            .desc = {
                .name = "DstBuf",
                .type = PL_DESC_BUF_STORAGE,
                .access = PL_DESC_ACCESS_WRITEONLY,
            },
            .num_buffer_vars = 1,
            .buffer_vars = &(struct pl_buffer_var) {
                .var = {
                    .name = "dst",
                    .type = PL_VAR_UINT,
                    .dim_v = 1,
                    .dim_m = 1,
                    .dim_a = dst_off + words,
                },
            },
        });
    } else {
        GLSL("#define dst src \n");
    }

    GLSL("// pl_buf_copy_swap                               \n"
         "{                                                 \n"
         "uint word = src["$" + gl_GlobalInvocationID.x];   \n"
         "word = (word & 0xFF00FF00u) >> 8 |                \n"
         "       (word & 0x00FF00FFu) << 8;                 \n",
         SH_UINT(src_off));
    if (params->wordsize > 2) {
        GLSL("word = (word & 0xFFFF0000u) >> 16 |           \n"
             "       (word & 0x0000FFFFu) << 16;            \n");
    }
    GLSL("dst["$" + gl_GlobalInvocationID.x] = word;        \n"
         "}                                                 \n",
         SH_UINT(dst_off));

    return pl_dispatch_compute(dp, pl_dispatch_compute_params(
        .shader = &sh,
        .dispatch_size = {groups, 1, 1},
    ));

error:
    if (src->params.debug_tag || dst->params.debug_tag) {
        PL_ERR(gpu, "  for buffers: src %s, dst %s",
               src->params.debug_tag, dst->params.debug_tag);
    }
    return false;
}

void pl_pass_run_vbo(pl_gpu gpu, const struct pl_pass_run_params *params)
{
    if (!params->vertex_data && !params->index_data)
        return pl_pass_run(gpu, params);

    struct pl_pass_run_params newparams = *params;
    pl_buf vert = NULL, index = NULL;

    if (params->vertex_data) {
        vert = pl_buf_create(gpu, pl_buf_params(
            .size = pl_vertex_buf_size(params),
            .initial_data = params->vertex_data,
            .drawable = true,
        ));

        if (!vert) {
            PL_ERR(gpu, "Failed allocating vertex buffer!");
            return;
        }

        newparams.vertex_buf = vert;
        newparams.vertex_data = NULL;
    }

    if (params->index_data) {
        index = pl_buf_create(gpu, pl_buf_params(
            .size = pl_index_buf_size(params),
            .initial_data = params->index_data,
            .drawable = true,
        ));

        if (!index) {
            PL_ERR(gpu, "Failed allocating index buffer!");
            return;
        }

        newparams.index_buf = index;
        newparams.index_data = NULL;
    }

    pl_pass_run(gpu, &newparams);
    pl_buf_destroy(gpu, &vert);
    pl_buf_destroy(gpu, &index);
}

struct pl_pass_params pl_pass_params_copy(void *alloc, const struct pl_pass_params *params)
{
    struct pl_pass_params new = *params;

    new.glsl_shader = pl_str0dup0(alloc, new.glsl_shader);
    new.vertex_shader = pl_str0dup0(alloc, new.vertex_shader);
    if (new.blend_params)
        new.blend_params = pl_memdup_ptr(alloc, new.blend_params);

#define DUPNAMES(field)                                                 \
    do {                                                                \
        size_t _size = new.num_##field * sizeof(new.field[0]);          \
        new.field = pl_memdup(alloc, new.field, _size);                 \
        for (int j = 0; j < new.num_##field; j++)                       \
            new.field[j].name = pl_str0dup0(alloc, new.field[j].name);  \
    } while (0)

    DUPNAMES(variables);
    DUPNAMES(descriptors);
    DUPNAMES(vertex_attribs);

#undef DUPNAMES

    new.constant_data = NULL;
    new.constants = pl_memdup(alloc, new.constants,
                              new.num_constants * sizeof(new.constants[0]));

    return new;
}

size_t pl_vertex_buf_size(const struct pl_pass_run_params *params)
{
    if (!params->index_data)
        return params->vertex_count * params->pass->params.vertex_stride;

    int num_vertices = 0;
    const void *idx = params->index_data;
    switch (params->index_fmt) {
    case PL_INDEX_UINT16:
        for (int i = 0; i < params->vertex_count; i++)
            num_vertices = PL_MAX(num_vertices, ((const uint16_t *) idx)[i]);
        break;
    case PL_INDEX_UINT32:
        for (int i = 0; i < params->vertex_count; i++)
            num_vertices = PL_MAX(num_vertices, ((const uint32_t *) idx)[i]);
        break;
    case PL_INDEX_FORMAT_COUNT: pl_unreachable();
    }

    return (num_vertices + 1) * params->pass->params.vertex_stride;
}

const char *print_uuid(char buf[3 * UUID_SIZE], const uint8_t uuid[UUID_SIZE])
{
    static const char *hexdigits = "0123456789ABCDEF";
    for (int i = 0; i < UUID_SIZE; i++) {
        uint8_t x = uuid[i];
        buf[3 * i + 0] = hexdigits[x >> 4];
        buf[3 * i + 1] = hexdigits[x & 0xF];
        buf[3 * i + 2] = i == UUID_SIZE - 1 ? '\0' : ':';
    }

    return buf;
}

const char *print_drm_mod(char buf[DRM_MOD_SIZE], uint64_t mod)
{
    switch (mod) {
    case DRM_FORMAT_MOD_LINEAR: return "LINEAR";
    case DRM_FORMAT_MOD_INVALID: return "INVALID";
    }

    uint8_t vendor = mod >> 56;
    uint64_t val = mod & ((1ULL << 56) - 1);

    const char *name = NULL;
    switch (vendor) {
    case 0x00: name = "NONE"; break;
    case 0x01: name = "INTEL"; break;
    case 0x02: name = "AMD"; break;
    case 0x03: name = "NVIDIA"; break;
    case 0x04: name = "SAMSUNG"; break;
    case 0x08: name = "ARM"; break;
    }

    if (name) {
        snprintf(buf, DRM_MOD_SIZE, "%s 0x%"PRIx64, name, val);
    } else {
        snprintf(buf, DRM_MOD_SIZE, "0x%02x 0x%"PRIx64, vendor, val);
    }

    return buf;
}
