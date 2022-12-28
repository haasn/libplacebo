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
#include "log.h"
#include "shaders.h"
#include "gpu.h"

#define require(expr)                                           \
  do {                                                          \
      if (!(expr)) {                                            \
          PL_ERR(gpu, "Validation failed: %s (%s:%d)",          \
                  #expr, __FILE__, __LINE__);                   \
          pl_log_stack_trace(gpu->log, PL_LOG_ERR);             \
          pl_debug_abort();                                     \
          goto error;                                           \
      }                                                         \
  } while (0)

void pl_gpu_destroy(pl_gpu gpu)
{
    if (!gpu)
        return;

    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    impl->destroy(gpu);
}

bool pl_fmt_is_ordered(pl_fmt fmt)
{
    bool ret = !fmt->opaque;
    for (int i = 0; i < fmt->num_components; i++)
        ret &= fmt->sample_order[i] == i;
    return ret;
}

bool pl_fmt_is_float(pl_fmt fmt)
{
    switch (fmt->type) {
    case PL_FMT_UNKNOWN: // more likely than not
    case PL_FMT_FLOAT:
    case PL_FMT_UNORM:
    case PL_FMT_SNORM:
        return true;

    case PL_FMT_UINT:
    case PL_FMT_SINT:
        return false;

    case PL_FMT_TYPE_COUNT:
        break;
    }

    pl_unreachable();
}

bool pl_fmt_has_modifier(pl_fmt fmt, uint64_t modifier)
{
    if (!fmt)
        return false;

    for (int i = 0; i < fmt->num_modifiers; i++) {
        if (fmt->modifiers[i] == modifier)
            return true;
    }

    return false;
}

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
    PL_DEBUG(gpu,  "    %-10s %-6s %-4s %-4s " CAP_HEADER " %-3s %-13s %-13s %-10s %-10s %-6s",
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


        PL_DEBUG(gpu, "    %-10s %-6s %-4zu %c%c%c%c " CAP_FIELDS " %-3s "
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

    // Set the backwards compatibility fields in `limits`
    gpu->limits.max_variables = gpu->limits.max_variable_comps;

    return gpu;
}

struct glsl_fmt {
    enum pl_fmt_type type;
    int num_components;
    int depth[4];
    const char *glsl_format;
    uint32_t drm_fourcc;
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

const char *pl_fmt_glsl_format(pl_fmt fmt, int components)
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
    {"rgba16hf",    FOURCC('A','B','4','H')},
    {"bgra16hf",    FOURCC('A','R','4','H')},

    // no planar formats yet (tm)
};

uint32_t pl_fmt_fourcc(pl_fmt fmt)
{
    if (fmt->opaque)
        return 0;

    for (int n = 0; n < PL_ARRAY_SIZE(pl_fmt_fourccs); n++) {
        const struct pl_fmt_fourcc *fourcc = &pl_fmt_fourccs[n];
        if (strcmp(fmt->name, fourcc->name) == 0)
            return fourcc->fourcc;
    }

    return 0; // no matching format
}

pl_fmt pl_find_fmt(pl_gpu gpu, enum pl_fmt_type type, int num_components,
                    int min_depth, int host_bits, enum pl_fmt_caps caps)
{
    for (int n = 0; n < gpu->num_formats; n++) {
        pl_fmt fmt = gpu->formats[n];
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
    PL_TRACE(gpu, "No matching format found");
    return NULL;
}

pl_fmt pl_find_vertex_fmt(pl_gpu gpu, enum pl_fmt_type type, int comps)
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

pl_fmt pl_find_named_fmt(pl_gpu gpu, const char *name)
{
    if (!name)
        return NULL;

    for (int i = 0; i < gpu->num_formats; i++) {
        pl_fmt fmt = gpu->formats[i];
        if (strcmp(name, fmt->name) == 0)
            return fmt;
    }

    // ran out of formats
    return NULL;
}

pl_fmt pl_find_fourcc(pl_gpu gpu, uint32_t fourcc)
{
    if (!fourcc)
        return NULL;

    for (int i = 0; i < gpu->num_formats; i++) {
        pl_fmt fmt = gpu->formats[i];
        if (fourcc == fmt->fourcc)
            return fmt;
    }

    // ran out of formats
    return NULL;
}

static inline bool check_mod(pl_gpu gpu, pl_fmt fmt, uint64_t mod)
{
    for (int i = 0; i < fmt->num_modifiers; i++) {
        if (fmt->modifiers[i] == mod)
            return true;
    }


    PL_ERR(gpu, "DRM modifier %s not available for format %s. Available modifiers:",
           PRINT_DRM_MOD(mod), fmt->name);
    for (int i = 0; i < fmt->num_modifiers; i++)
        PL_ERR(gpu, "    %s", PRINT_DRM_MOD(fmt->modifiers[i]));

    return false;
}

pl_tex pl_tex_create(pl_gpu gpu, const struct pl_tex_params *params)
{
    require(!params->import_handle || !params->export_handle);
    require(!params->import_handle || !params->initial_data);
    if (params->export_handle) {
        require(params->export_handle & gpu->export_caps.tex);
        require(PL_ISPOT(params->export_handle));
    }
    if (params->import_handle) {
        require(params->import_handle & gpu->import_caps.tex);
        require(PL_ISPOT(params->import_handle));
        if (params->import_handle == PL_HANDLE_DMA_BUF) {
            if (!check_mod(gpu, params->format, params->shared_mem.drm_format_mod))
                goto error;
            if (params->shared_mem.stride_w)
                require(params->w && params->shared_mem.stride_w >= params->w);
            if (params->shared_mem.stride_h)
                require(params->h && params->shared_mem.stride_h >= params->h);
        } else if (params->import_handle == PL_HANDLE_MTL_TEX) {
            require(params->shared_mem.plane <= 2);
        }
    }

    switch (pl_tex_params_dimension(*params)) {
    case 1:
        require(params->w > 0);
        require(params->w <= gpu->limits.max_tex_1d_dim);
        require(!params->renderable);
        require(!params->blit_src || gpu->limits.blittable_1d_3d);
        require(!params->blit_dst || gpu->limits.blittable_1d_3d);
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
        require(!params->blit_src || gpu->limits.blittable_1d_3d);
        require(!params->blit_dst || gpu->limits.blittable_1d_3d);
        break;
    }

    pl_fmt fmt = params->format;
    require(fmt);
    require(!params->host_readable || fmt->caps & PL_FMT_CAP_HOST_READABLE);
    require(!params->host_readable || !fmt->opaque);
    require(!params->host_writable || !fmt->opaque);
    require(!params->sampleable || fmt->caps & PL_FMT_CAP_SAMPLEABLE);
    require(!params->renderable || fmt->caps & PL_FMT_CAP_RENDERABLE);
    require(!params->storable   || fmt->caps & PL_FMT_CAP_STORABLE);
    require(!params->blit_src   || fmt->caps & PL_FMT_CAP_BLITTABLE);
    require(!params->blit_dst   || fmt->caps & PL_FMT_CAP_BLITTABLE);

    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    return impl->tex_create(gpu, params);

error:
    if (params->debug_tag)
        PL_ERR(gpu, "  for texture: %s", params->debug_tag);
    return NULL;
}

void pl_tex_destroy(pl_gpu gpu, pl_tex *tex)
{
    if (!*tex)
        return;

    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    impl->tex_destroy(gpu, *tex);
    *tex = NULL;
}

static bool pl_tex_params_superset(struct pl_tex_params a, struct pl_tex_params b)
{
    return a.w == b.w && a.h == b.h && a.d == b.d &&
           a.format          == b.format &&
           (a.sampleable     || !b.sampleable) &&
           (a.renderable     || !b.renderable) &&
           (a.storable       || !b.storable) &&
           (a.blit_src       || !b.blit_src) &&
           (a.blit_dst       || !b.blit_dst) &&
           (a.host_writable  || !b.host_writable) &&
           (a.host_readable  || !b.host_readable);
}

bool pl_tex_recreate(pl_gpu gpu, pl_tex *tex, const struct pl_tex_params *params)
{
    if (params->initial_data) {
        PL_ERR(gpu, "pl_tex_recreate may not be used with `initial_data`!");
        return false;
    }

    if (params->import_handle) {
        PL_ERR(gpu, "pl_tex_recreate may not be used with `import_handle`!");
        return false;
    }

    if (*tex && pl_tex_params_superset((*tex)->params, *params)) {
        pl_tex_invalidate(gpu, *tex);
        return true;
    }

    PL_DEBUG(gpu, "(Re)creating %dx%dx%d texture with format %s",
             params->w, params->h, params->d, params->format->name);

    pl_tex_destroy(gpu, tex);
    *tex = pl_tex_create(gpu, params);

    return !!*tex;
}

void pl_tex_clear_ex(pl_gpu gpu, pl_tex dst, const union pl_clear_color color)
{
    require(dst->params.blit_dst);

    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    if (impl->tex_invalidate)
        impl->tex_invalidate(gpu, dst);
    impl->tex_clear_ex(gpu, dst, color);
    return;

error:
    if (dst->params.debug_tag)
        PL_ERR(gpu, "  for texture: %s", dst->params.debug_tag);
}

void pl_tex_clear(pl_gpu gpu, pl_tex dst, const float color[4])
{
    if (!pl_fmt_is_float(dst->params.format)) {
        PL_ERR(gpu, "Cannot call `pl_tex_clear` on integer textures, please "
               "use `pl_tex_clear_ex` instead.");
        return;
    }

    const union pl_clear_color col = {
        .f = { color[0], color[1], color[2], color[3] },
    };

    pl_tex_clear_ex(gpu, dst, col);
}

void pl_tex_invalidate(pl_gpu gpu, pl_tex tex)
{
    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    if (impl->tex_invalidate)
        impl->tex_invalidate(gpu, tex);
}

static void strip_coords(pl_tex tex, struct pl_rect3d *rc)
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

static void infer_rc(pl_tex tex, struct pl_rect3d *rc)
{
    if (!rc->x0 && !rc->x1)
        rc->x1 = tex->params.w;
    if (!rc->y0 && !rc->y1)
        rc->y1 = tex->params.h;
    if (!rc->z0 && !rc->z1)
        rc->z1 = tex->params.d;
}

void pl_tex_blit(pl_gpu gpu, const struct pl_tex_blit_params *params)
{
    pl_tex src = params->src, dst = params->dst;
    require(src && dst);
    pl_fmt src_fmt = src->params.format;
    pl_fmt dst_fmt = dst->params.format;
    require(src_fmt->internal_size == dst_fmt->internal_size);
    require((src_fmt->type == PL_FMT_UINT) == (dst_fmt->type == PL_FMT_UINT));
    require((src_fmt->type == PL_FMT_SINT) == (dst_fmt->type == PL_FMT_SINT));
    require(src->params.blit_src);
    require(dst->params.blit_dst);
    require(params->sample_mode != PL_TEX_SAMPLE_LINEAR || (src_fmt->caps & PL_FMT_CAP_LINEAR));

    struct pl_tex_blit_params fixed = *params;
    infer_rc(src, &fixed.src_rc);
    infer_rc(dst, &fixed.dst_rc);
    strip_coords(src, &fixed.src_rc);
    strip_coords(dst, &fixed.dst_rc);

    require(fixed.src_rc.x0 >= 0 && fixed.src_rc.x0 < src->params.w);
    require(fixed.src_rc.x1 > 0 && fixed.src_rc.x1 <= src->params.w);
    require(fixed.dst_rc.x0 >= 0 && fixed.dst_rc.x0 < dst->params.w);
    require(fixed.dst_rc.x1 > 0 && fixed.dst_rc.x1 <= dst->params.w);

    if (src->params.h) {
        require(fixed.src_rc.y0 >= 0 && fixed.src_rc.y0 < src->params.h);
        require(fixed.src_rc.y1 > 0 && fixed.src_rc.y1 <= src->params.h);
    }

    if (dst->params.h) {
        require(fixed.dst_rc.y0 >= 0 && fixed.dst_rc.y0 < dst->params.h);
        require(fixed.dst_rc.y1 > 0 && fixed.dst_rc.y1 <= dst->params.h);
    }

    if (src->params.d) {
        require(fixed.src_rc.z0 >= 0 && fixed.src_rc.z0 < src->params.d);
        require(fixed.src_rc.z1 > 0 && fixed.src_rc.z1 <= src->params.d);
    }

    if (dst->params.d) {
        require(fixed.dst_rc.z0 >= 0 && fixed.dst_rc.z0 < dst->params.d);
        require(fixed.dst_rc.z1 > 0 && fixed.dst_rc.z1 <= dst->params.d);
    }

    struct pl_rect3d full = {0, 0, 0, dst->params.w, dst->params.h, dst->params.d};
    strip_coords(dst, &full);

    struct pl_rect3d rcnorm = fixed.dst_rc;
    pl_rect3d_normalize(&rcnorm);
    if (pl_rect3d_eq(rcnorm, full))
        pl_tex_invalidate(gpu, dst);

    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    impl->tex_blit(gpu, &fixed);
    return;

error:
    if (src->params.debug_tag || dst->params.debug_tag) {
        PL_ERR(gpu, "  for textures: src %s, dst %s",
               PL_DEF(src->params.debug_tag, "(unknown)"),
               PL_DEF(dst->params.debug_tag, "(unknown)"));
    }
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

static bool fix_tex_transfer(pl_gpu gpu, struct pl_tex_transfer_params *params)
{
    pl_tex tex = params->tex;
    pl_fmt fmt = tex->params.format;
    struct pl_rect3d rc = params->rc;

    // Infer the default values
    infer_rc(tex, &rc);
    strip_coords(tex, &rc);

    if (!params->row_pitch && params->stride_w)
        params->row_pitch = params->stride_w * fmt->texel_size;
    if (!params->row_pitch || !tex->params.w)
        params->row_pitch = pl_rect_w(rc) * fmt->texel_size;
    if (!params->depth_pitch && params->stride_h)
        params->depth_pitch = params->stride_h * params->row_pitch;
    if (!params->depth_pitch || !tex->params.d)
        params->depth_pitch = pl_rect_h(rc) * params->row_pitch;

    params->rc = rc;

    // Check the parameters for sanity
    switch (pl_tex_params_dimension(tex->params))
    {
    case 3:
        require(rc.z1 > rc.z0);
        require(rc.z0 >= 0 && rc.z0 <  tex->params.d);
        require(rc.z1 >  0 && rc.z1 <= tex->params.d);
        require(params->depth_pitch >= pl_rect_h(rc) * params->row_pitch);
        require(params->depth_pitch % params->row_pitch == 0);
        // fall through
    case 2:
        require(rc.y1 > rc.y0);
        require(rc.y0 >= 0 && rc.y0 <  tex->params.h);
        require(rc.y1 >  0 && rc.y1 <= tex->params.h);
        require(params->row_pitch >= pl_rect_w(rc) * fmt->texel_size);
        require(params->row_pitch % fmt->texel_align == 0);
        // fall through
    case 1:
        require(rc.x1 > rc.x0);
        require(rc.x0 >= 0 && rc.x0 <  tex->params.w);
        require(rc.x1 >  0 && rc.x1 <= tex->params.w);
        break;
    }

    require(!params->buf ^ !params->ptr); // exactly one
    if (params->buf) {
        pl_buf buf = params->buf;
        size_t size = pl_tex_transfer_size(params);
        require(params->buf_offset + size >= params->buf_offset); // overflow check
        require(params->buf_offset + size <= buf->params.size);
        require(gpu->limits.buf_transfer);
    }

    require(!params->callback || gpu->limits.callbacks);
    return true;

error:
    if (tex->params.debug_tag)
        PL_ERR(gpu, "  for texture: %s", tex->params.debug_tag);
    return false;
}

bool pl_tex_upload(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    pl_tex tex = params->tex;
    require(tex->params.host_writable);

    struct pl_tex_transfer_params fixed = *params;
    if (!fix_tex_transfer(gpu, &fixed))
        goto error;

    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    return impl->tex_upload(gpu, &fixed);

error:
    if (tex->params.debug_tag)
        PL_ERR(gpu, "  for texture: %s", tex->params.debug_tag);
    return false;
}

bool pl_tex_download(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    pl_tex tex = params->tex;
    require(tex->params.host_readable);

    struct pl_tex_transfer_params fixed = *params;
    if (!fix_tex_transfer(gpu, &fixed))
        goto error;

    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    return impl->tex_download(gpu, &fixed);

error:
    if (tex->params.debug_tag)
        PL_ERR(gpu, "  for texture: %s", tex->params.debug_tag);
    return false;
}

bool pl_tex_poll(pl_gpu gpu, pl_tex tex, uint64_t t)
{
    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    return impl->tex_poll ? impl->tex_poll(gpu, tex, t) : false;
}

static bool warned_rounding = false;

pl_buf pl_buf_create(pl_gpu gpu, const struct pl_buf_params *params)
{
    struct pl_buf_params params_rounded;

    require(!params->import_handle || !params->export_handle);
    if (params->export_handle) {
        require(PL_ISPOT(params->export_handle));
        require(params->export_handle & gpu->export_caps.buf);
    }
    if (params->import_handle) {
        require(PL_ISPOT(params->import_handle));
        require(params->import_handle & gpu->import_caps.buf);
        const struct pl_shared_mem *shmem = &params->shared_mem;
        require(shmem->offset + params->size <= shmem->size);
        require(params->import_handle != PL_HANDLE_DMA_BUF || !shmem->drm_format_mod);

        // Fix misalignment on host pointer imports
        if (params->import_handle == PL_HANDLE_HOST_PTR) {
            uintptr_t page_mask = ~(gpu->limits.align_host_ptr - 1);
            uintptr_t ptr_base = (uintptr_t) shmem->handle.ptr & page_mask;
            size_t ptr_offset = (uintptr_t) shmem->handle.ptr - ptr_base;
            size_t buf_offset = ptr_offset + shmem->offset;
            size_t ptr_size = PL_ALIGN2(ptr_offset + shmem->size,
                                        gpu->limits.align_host_ptr);

            if (ptr_base != (uintptr_t) shmem->handle.ptr || ptr_size > shmem->size) {
                if (!warned_rounding) {
                    warned_rounding = true;
                    PL_WARN(gpu, "Imported host pointer is not page-aligned. "
                            "This should normally be fine on most platforms, "
                            "but may cause issues in some rare circumstances.");
                }

                PL_TRACE(gpu, "Rounding imported host pointer %p + %zu -> %zu to "
                         "nearest page boundaries: %p + %zu -> %zu",
                          shmem->handle.ptr, shmem->offset, shmem->size,
                          (void *) ptr_base, buf_offset, ptr_size);
            }

            params_rounded = *params;
            params_rounded.shared_mem.handle.ptr = (void *) ptr_base;
            params_rounded.shared_mem.offset = buf_offset;
            params_rounded.shared_mem.size = ptr_size;
            params = &params_rounded;
        }
    }

    require(params->size > 0 && params->size <= gpu->limits.max_buf_size);
    require(!params->uniform || params->size <= gpu->limits.max_ubo_size);
    require(!params->storable || params->size <= gpu->limits.max_ssbo_size);
    require(!params->drawable || params->size <= gpu->limits.max_vbo_size);
    require(!params->host_mapped || params->size <= gpu->limits.max_mapped_size);

    if (params->format) {
        pl_fmt fmt = params->format;
        require(params->size <= gpu->limits.max_buffer_texels * fmt->texel_size);
        require(!params->uniform || (fmt->caps & PL_FMT_CAP_TEXEL_UNIFORM));
        require(!params->storable || (fmt->caps & PL_FMT_CAP_TEXEL_STORAGE));
    }

    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    pl_buf buf = impl->buf_create(gpu, params);
    if (buf)
        require(!params->host_mapped || buf->data);

    return buf;

error:
    if (params->debug_tag)
        PL_ERR(gpu, "  for buffer: %s", params->debug_tag);
    return NULL;
}

void pl_buf_destroy(pl_gpu gpu, pl_buf *buf)
{
    if (!*buf)
        return;

    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    impl->buf_destroy(gpu, *buf);
    *buf = NULL;
}

static bool pl_buf_params_superset(struct pl_buf_params a, struct pl_buf_params b)
{
    return a.size            >= b.size &&
           a.memory_type     == b.memory_type &&
           a.format          == b.format &&
           (a.host_writable  || !b.host_writable) &&
           (a.host_readable  || !b.host_readable) &&
           (a.host_mapped    || !b.host_mapped) &&
           (a.uniform        || !b.uniform) &&
           (a.storable       || !b.storable) &&
           (a.drawable       || !b.drawable);
}

bool pl_buf_recreate(pl_gpu gpu, pl_buf *buf, const struct pl_buf_params *params)
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

void pl_buf_write(pl_gpu gpu, pl_buf buf, size_t buf_offset,
                  const void *data, size_t size)
{
    require(buf->params.host_writable);
    require(buf_offset + size <= buf->params.size);
    require(buf_offset == PL_ALIGN2(buf_offset, 4));

    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    impl->buf_write(gpu, buf, buf_offset, data, size);
    return;

error:
    if (buf->params.debug_tag)
        PL_ERR(gpu, "  for buffer: %s", buf->params.debug_tag);
}

bool pl_buf_read(pl_gpu gpu, pl_buf buf, size_t buf_offset,
                 void *dest, size_t size)
{
    require(buf->params.host_readable);
    require(buf_offset + size <= buf->params.size);

    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    return impl->buf_read(gpu, buf, buf_offset, dest, size);

error:
    if (buf->params.debug_tag)
        PL_ERR(gpu, "  for buffer: %s", buf->params.debug_tag);
    return false;
}

void pl_buf_copy(pl_gpu gpu, pl_buf dst, size_t dst_offset,
                 pl_buf src, size_t src_offset, size_t size)
{
    require(src_offset + size <= src->params.size);
    require(dst_offset + size <= dst->params.size);

    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    impl->buf_copy(gpu, dst, dst_offset, src, src_offset, size);
    return;

error:
    if (src->params.debug_tag || dst->params.debug_tag) {
        PL_ERR(gpu, "  for buffers: src %s, dst %s",
               src->params.debug_tag, dst->params.debug_tag);
    }
}

bool pl_buf_export(pl_gpu gpu, pl_buf buf)
{
    require(buf->params.export_handle || buf->params.import_handle);

    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    return impl->buf_export(gpu, buf);

error:
    if (buf->params.debug_tag)
        PL_ERR(gpu, "  for buffer: %s", buf->params.debug_tag);
    return false;
}

bool pl_buf_poll(pl_gpu gpu, pl_buf buf, uint64_t t)
{
    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    return impl->buf_poll ? impl->buf_poll(gpu, buf, t) : false;
}

size_t pl_var_type_size(enum pl_var_type type)
{
    switch (type) {
    case PL_VAR_SINT:  return sizeof(int);
    case PL_VAR_UINT:  return sizeof(unsigned int);
    case PL_VAR_FLOAT: return sizeof(float);
    case PL_VAR_INVALID: // fall through
    case PL_VAR_TYPE_COUNT: break;
    }

    pl_unreachable();
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

PL_VAR(FLOAT, float,    1, 1)
PL_VAR(FLOAT, vec2,     1, 2)
PL_VAR(FLOAT, vec3,     1, 3)
PL_VAR(FLOAT, vec4,     1, 4)
PL_VAR(FLOAT, mat2,     2, 2)
PL_VAR(FLOAT, mat2x3,   2, 3)
PL_VAR(FLOAT, mat2x4,   2, 4)
PL_VAR(FLOAT, mat3,     3, 3)
PL_VAR(FLOAT, mat3x4,   3, 4)
PL_VAR(FLOAT, mat4x2,   4, 2)
PL_VAR(FLOAT, mat4x3,   4, 3)
PL_VAR(FLOAT, mat4,     4, 4)
PL_VAR(SINT,  int,      1, 1)
PL_VAR(SINT,  ivec2,    1, 2)
PL_VAR(SINT,  ivec3,    1, 3)
PL_VAR(SINT,  ivec4,    1, 4)
PL_VAR(UINT,  uint,     1, 1)
PL_VAR(UINT,  uvec2,    1, 2)
PL_VAR(UINT,  uvec3,    1, 3)
PL_VAR(UINT,  uvec4,    1, 4)

#undef PL_VAR

const struct pl_named_var pl_var_glsl_types[] = {
    // float vectors
    { "float",  { .type = PL_VAR_FLOAT, .dim_m = 1, .dim_v = 1, .dim_a = 1, }},
    { "vec2",   { .type = PL_VAR_FLOAT, .dim_m = 1, .dim_v = 2, .dim_a = 1, }},
    { "vec3",   { .type = PL_VAR_FLOAT, .dim_m = 1, .dim_v = 3, .dim_a = 1, }},
    { "vec4",   { .type = PL_VAR_FLOAT, .dim_m = 1, .dim_v = 4, .dim_a = 1, }},
    // float matrices
    { "mat2",   { .type = PL_VAR_FLOAT, .dim_m = 2, .dim_v = 2, .dim_a = 1, }},
    { "mat2x3", { .type = PL_VAR_FLOAT, .dim_m = 2, .dim_v = 3, .dim_a = 1, }},
    { "mat2x4", { .type = PL_VAR_FLOAT, .dim_m = 2, .dim_v = 4, .dim_a = 1, }},
    { "mat3",   { .type = PL_VAR_FLOAT, .dim_m = 3, .dim_v = 3, .dim_a = 1, }},
    { "mat3x4", { .type = PL_VAR_FLOAT, .dim_m = 3, .dim_v = 4, .dim_a = 1, }},
    { "mat4x2", { .type = PL_VAR_FLOAT, .dim_m = 4, .dim_v = 2, .dim_a = 1, }},
    { "mat4x3", { .type = PL_VAR_FLOAT, .dim_m = 4, .dim_v = 3, .dim_a = 1, }},
    { "mat4",   { .type = PL_VAR_FLOAT, .dim_m = 4, .dim_v = 4, .dim_a = 1, }},
    // integer vectors
    { "int",    { .type = PL_VAR_SINT,  .dim_m = 1, .dim_v = 1, .dim_a = 1, }},
    { "ivec2",  { .type = PL_VAR_SINT,  .dim_m = 1, .dim_v = 2, .dim_a = 1, }},
    { "ivec3",  { .type = PL_VAR_SINT,  .dim_m = 1, .dim_v = 3, .dim_a = 1, }},
    { "ivec4",  { .type = PL_VAR_SINT,  .dim_m = 1, .dim_v = 4, .dim_a = 1, }},
    // unsigned integer vectors
    { "uint",   { .type = PL_VAR_UINT,  .dim_m = 1, .dim_v = 1, .dim_a = 1, }},
    { "uvec2",  { .type = PL_VAR_UINT,  .dim_m = 1, .dim_v = 2, .dim_a = 1, }},
    { "uvec3",  { .type = PL_VAR_UINT,  .dim_m = 1, .dim_v = 3, .dim_a = 1, }},
    { "uvec4",  { .type = PL_VAR_UINT,  .dim_m = 1, .dim_v = 4, .dim_a = 1, }},

    {0},
};

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

struct pl_var pl_var_from_fmt(pl_fmt fmt, const char *name)
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
        stride = align = PL_ALIGN2(align, sizeof(float[4]));

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
    if (var->dim_v == 3)
        align += el_size;
    if (var->dim_m * var->dim_a > 1)
        stride = align;

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
        pl_assert(dst_layout.size == src_layout.size);
        memcpy((void *) dst, (const void *) src, src_layout.size);
        return;
    }

    size_t stride = PL_MIN(src_layout.stride, dst_layout.stride);
    uintptr_t end = src + src_layout.size;
    while (src < end) {
        pl_assert(dst < dst + dst_layout.size);
        memcpy((void *) dst, (const void *) src, stride);
        src += src_layout.stride;
        dst += dst_layout.stride;
    }
}

int pl_desc_namespace(pl_gpu gpu, enum pl_desc_type type)
{
    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    int ret = impl->desc_namespace(gpu, type);
    pl_assert(ret >= 0 && ret < PL_DESC_TYPE_COUNT);
    return ret;
}

const char *pl_desc_access_glsl_name(enum pl_desc_access mode)
{
    switch (mode) {
    case PL_DESC_ACCESS_READWRITE: return "";
    case PL_DESC_ACCESS_READONLY:  return "readonly";
    case PL_DESC_ACCESS_WRITEONLY: return "writeonly";
    case PL_DESC_ACCESS_COUNT: break;
    }

    pl_unreachable();
}

const struct pl_blend_params pl_alpha_overlay = {
    .src_rgb    = PL_BLEND_SRC_ALPHA,
    .dst_rgb    = PL_BLEND_ONE_MINUS_SRC_ALPHA,
    .src_alpha  = PL_BLEND_ONE,
    .dst_alpha  = PL_BLEND_ONE_MINUS_SRC_ALPHA,
};

static inline void log_shader_sources(pl_log log, enum pl_log_level level,
                                      const struct pl_pass_params *params)
{
    if (!pl_msg_test(log, level) || !params->glsl_shader)
        return;

    switch (params->type) {
    case PL_PASS_RASTER:
        if (!params->vertex_shader)
            return;
        pl_msg(log, level, "vertex shader source:");
        pl_msg_source(log, level, params->vertex_shader);
        pl_msg(log, level, "fragment shader source:");
        pl_msg_source(log, level, params->glsl_shader);
        return;

    case PL_PASS_COMPUTE:
        pl_msg(log, level, "compute shader source:");
        pl_msg_source(log, level, params->glsl_shader);
        return;

    case PL_PASS_INVALID:
    case PL_PASS_TYPE_COUNT:
        break;
    }

    pl_unreachable();
}

static void log_spec_constants(pl_log log, enum pl_log_level lev,
                               const struct pl_pass_params *params,
                               const void *constant_data)
{
    if (!constant_data || !params->num_constants || !pl_msg_test(log, lev))
        return;

    pl_msg(log, lev, "Specialization constant values:");

    uintptr_t data_base = (uintptr_t) constant_data;
    for (int i = 0; i < params->num_constants; i++) {
        union {
            int i;
            unsigned u;
            float f;
        } *data = (void *) (data_base + params->constants[i].offset);
        int id = params->constants[i].id;

        switch (params->constants[i].type) {
        case PL_VAR_SINT:  pl_msg(log, lev, "  constant_id=%d: %d", id, data->i); break;
        case PL_VAR_UINT:  pl_msg(log, lev, "  constant_id=%d: %u", id, data->u); break;
        case PL_VAR_FLOAT: pl_msg(log, lev, "  constant_id=%d: %f", id, data->f); break;
        default: pl_unreachable();
        }
    }
}

pl_pass pl_pass_create(pl_gpu gpu, const struct pl_pass_params *params)
{
    struct pl_pass_params fixed;

    require(params->glsl_shader);
    switch(params->type) {
    case PL_PASS_RASTER:
        require(params->vertex_shader);
        require(params->vertex_stride % gpu->limits.align_vertex_stride == 0);
        for (int i = 0; i < params->num_vertex_attribs; i++) {
            struct pl_vertex_attrib va = params->vertex_attribs[i];
            require(va.name);
            require(va.fmt);
            require(va.fmt->caps & PL_FMT_CAP_VERTEX);
            require(va.offset + va.fmt->texel_size <= params->vertex_stride);
        }

        if (!params->target_format) {
            // Compatibility with older API
            fixed = *params;
            fixed.target_format = params->target_dummy.params.format;
            params = &fixed;
        }

        require(params->target_format);
        require(params->target_format->caps & PL_FMT_CAP_RENDERABLE);
        require(!params->blend_params || params->target_format->caps & PL_FMT_CAP_BLENDABLE);
        require(!params->blend_params || params->load_target);
        break;
    case PL_PASS_COMPUTE:
        require(gpu->glsl.compute);
        break;
    case PL_PASS_INVALID:
    case PL_PASS_TYPE_COUNT:
        pl_unreachable();
    }

    size_t num_var_comps = 0;
    for (int i = 0; i < params->num_variables; i++) {
        struct pl_var var = params->variables[i];
        num_var_comps += var.dim_v * var.dim_m * var.dim_a;
        require(var.name);
        require(pl_var_glsl_type_name(var));
    }
    require(num_var_comps <= gpu->limits.max_variable_comps);

    require(params->num_constants <= gpu->limits.max_constants);
    for (int i = 0; i < params->num_constants; i++)
        require(params->constants[i].type);

    for (int i = 0; i < params->num_descriptors; i++) {
        struct pl_desc desc = params->descriptors[i];
        require(desc.name);

        // enforce disjoint descriptor bindings for each namespace
        int namespace = pl_desc_namespace(gpu, desc.type);
        for (int j = i+1; j < params->num_descriptors; j++) {
            struct pl_desc other = params->descriptors[j];
            require(desc.binding != other.binding ||
                    namespace != pl_desc_namespace(gpu, other.type));
        }
    }

    require(params->push_constants_size <= gpu->limits.max_pushc_size);
    require(params->push_constants_size == PL_ALIGN2(params->push_constants_size, 4));

    log_shader_sources(gpu->log, PL_LOG_DEBUG, params);
    log_spec_constants(gpu->log, PL_LOG_DEBUG, params, params->constant_data);

    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    pl_pass pass = impl->pass_create(gpu, params);
    if (!pass)
        goto error;

    return pass;

error:
    log_shader_sources(gpu->log, PL_LOG_ERR, params);
    return NULL;
}

void pl_pass_destroy(pl_gpu gpu, pl_pass *pass)
{
    if (!*pass)
        return;

    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    impl->pass_destroy(gpu, *pass);
    *pass = NULL;
}

void pl_pass_run(pl_gpu gpu, const struct pl_pass_run_params *params)
{
    pl_pass pass = params->pass;
    struct pl_pass_run_params new = *params;

    for (int i = 0; i < pass->params.num_descriptors; i++) {
        struct pl_desc desc = pass->params.descriptors[i];
        struct pl_desc_binding db = params->desc_bindings[i];
        require(db.object);
        switch (desc.type) {
        case PL_DESC_SAMPLED_TEX: {
            pl_tex tex = db.object;
            pl_fmt fmt = tex->params.format;
            require(tex->params.sampleable);
            require(db.sample_mode != PL_TEX_SAMPLE_LINEAR || (fmt->caps & PL_FMT_CAP_LINEAR));
            break;
        }
        case PL_DESC_STORAGE_IMG: {
            pl_tex tex = db.object;
            pl_fmt fmt = tex->params.format;
            require(tex->params.storable);
            require(desc.access != PL_DESC_ACCESS_READWRITE || (fmt->caps & PL_FMT_CAP_READWRITE));
            break;
        }
        case PL_DESC_BUF_UNIFORM: {
            pl_buf buf = db.object;
            require(buf->params.uniform);
            break;
        }
        case PL_DESC_BUF_STORAGE: {
            pl_buf buf = db.object;
            require(buf->params.storable);
            break;
        }
        case PL_DESC_BUF_TEXEL_UNIFORM: {
            pl_buf buf = db.object;
            require(buf->params.uniform && buf->params.format);
            break;
        }
        case PL_DESC_BUF_TEXEL_STORAGE: {
            pl_buf buf = db.object;
            pl_fmt fmt = buf->params.format;
            require(buf->params.storable && buf->params.format);
            require(desc.access != PL_DESC_ACCESS_READWRITE || (fmt->caps & PL_FMT_CAP_READWRITE));
            break;
        }
        case PL_DESC_INVALID:
        case PL_DESC_TYPE_COUNT:
            pl_unreachable();
        }
    }

    for (int i = 0; i < params->num_var_updates; i++) {
        struct pl_var_update vu = params->var_updates[i];
        require(vu.index >= 0 && vu.index < pass->params.num_variables);
        require(vu.data);
    }

    require(params->push_constants || !pass->params.push_constants_size);

    switch (pass->params.type) {
    case PL_PASS_RASTER: {
        switch (pass->params.vertex_type) {
        case PL_PRIM_TRIANGLE_LIST:
            require(params->vertex_count % 3 == 0);
            // fall through
        case PL_PRIM_TRIANGLE_STRIP:
            require(params->vertex_count >= 3);
            break;
        case PL_PRIM_TYPE_COUNT:
            pl_unreachable();
        }

        require(!params->vertex_data ^ !params->vertex_buf);
        if (params->vertex_buf) {
            pl_buf vertex_buf = params->vertex_buf;
            require(vertex_buf->params.drawable);
            if (!params->index_data && !params->index_buf) {
                // Cannot bounds check indexed draws
                size_t vert_size = params->vertex_count * pass->params.vertex_stride;
                require(params->buf_offset + vert_size <= vertex_buf->params.size);
            }
        }

        require(!params->index_data || !params->index_buf);
        if (params->index_buf) {
            pl_buf index_buf = params->index_buf;
            require(!params->vertex_data);
            require(index_buf->params.drawable);
            size_t index_size = pl_index_buf_size(params);
            require(params->index_offset + index_size <= index_buf->params.size);
        }

        pl_tex target = params->target;
        require(target);
        require(pl_tex_params_dimension(target->params) == 2);
        require(target->params.format->signature == pass->params.target_format->signature);
        require(target->params.renderable);
        struct pl_rect2d *vp = &new.viewport;
        struct pl_rect2d *sc = &new.scissors;

        // Sanitize viewport/scissors
        if (!vp->x0 && !vp->x1)
            vp->x1 = target->params.w;
        if (!vp->y0 && !vp->y1)
            vp->y1 = target->params.h;

        if (!sc->x0 && !sc->x1)
            sc->x1 = target->params.w;
        if (!sc->y0 && !sc->y1)
            sc->y1 = target->params.h;

        // Constrain the scissors to the target dimension (to sanitize the
        // underlying graphics API calls)
        sc->x0 = PL_CLAMP(sc->x0, 0, target->params.w);
        sc->y0 = PL_CLAMP(sc->y0, 0, target->params.h);
        sc->x1 = PL_CLAMP(sc->x1, 0, target->params.w);
        sc->y1 = PL_CLAMP(sc->y1, 0, target->params.h);

        // Scissors wholly outside target -> silently drop pass (also needed
        // to ensure we don't cause UB by specifying invalid scissors)
        if (!pl_rect_w(*sc) || !pl_rect_h(*sc))
            return;

        require(pl_rect_w(*vp) > 0);
        require(pl_rect_h(*vp) > 0);
        require(pl_rect_w(*sc) > 0);
        require(pl_rect_h(*sc) > 0);

        if (!pass->params.load_target)
            pl_tex_invalidate(gpu, target);
        break;
    }
    case PL_PASS_COMPUTE:
        for (int i = 0; i < PL_ARRAY_SIZE(params->compute_groups); i++) {
            require(params->compute_groups[i] >= 0);
            require(params->compute_groups[i] <= gpu->limits.max_dispatch[i]);
        }
        break;
    case PL_PASS_INVALID:
    case PL_PASS_TYPE_COUNT:
        pl_unreachable();
    }

    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    impl->pass_run(gpu, &new);

error:
    return;
}

void pl_gpu_flush(pl_gpu gpu)
{
    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    if (impl->gpu_flush)
        impl->gpu_flush(gpu);
}

void pl_gpu_finish(pl_gpu gpu)
{
    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    impl->gpu_finish(gpu);
}

bool pl_gpu_is_failed(pl_gpu gpu)
{
    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    if (!impl->gpu_is_failed)
        return false;

    return impl->gpu_is_failed(gpu);
}

// GPU-internal helpers

bool pl_tex_upload_pbo(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    if (params->buf)
        return pl_tex_upload(gpu, params);

    pl_buf buf = NULL;
    struct pl_buf_params bufparams = {
        .size = pl_tex_transfer_size(params),
        .debug_tag = PL_DEBUG_TAG,
    };

    // If we can import host pointers directly, and the function is being used
    // asynchronously, then we can use host pointer import to skip a memcpy. In
    // the synchronous case, we still force a host memcpy to avoid stalling the
    // host until the GPU memcpy completes.
    bool can_import = gpu->import_caps.buf & PL_HANDLE_HOST_PTR;
    if (can_import && params->callback && bufparams.size > 32*1024) { // 32 KiB
        bufparams.import_handle = PL_HANDLE_HOST_PTR;
        bufparams.shared_mem = (struct pl_shared_mem) {
            .handle.ptr = params->ptr,
            .size = bufparams.size,
            .offset = 0,
        };

        // Suppress errors for this test because it may fail, in which case we
        // want to silently fall back.
        pl_log_level_cap(gpu->log, PL_LOG_DEBUG);
        buf = pl_buf_create(gpu, &bufparams);
        pl_log_level_cap(gpu->log, PL_LOG_NONE);
    }

    if (!buf) {
        bufparams.import_handle = 0;
        bufparams.host_writable = true;
        buf = pl_buf_create(gpu, &bufparams);
    }

    if (!buf)
        return false;

    if (!bufparams.import_handle)
        pl_buf_write(gpu, buf, 0, params->ptr, buf->params.size);

    struct pl_tex_transfer_params newparams = *params;
    newparams.buf = buf;
    newparams.ptr = NULL;

    bool ok = pl_tex_upload(gpu, &newparams);
    pl_buf_destroy(gpu, &buf);
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

    pl_buf buf = NULL;
    struct pl_buf_params bufparams = {
        .size = pl_tex_transfer_size(params),
        .debug_tag = PL_DEBUG_TAG,
    };

    // If we can import host pointers directly, we can avoid an extra memcpy
    // (sometimes). In the cases where it isn't avoidable, the extra memcpy
    // will happen inside VRAM, which is typically faster anyway.
    bool can_import = gpu->import_caps.buf & PL_HANDLE_HOST_PTR;
    if (can_import && bufparams.size > 32*1024) { // 32 KiB
        bufparams.import_handle = PL_HANDLE_HOST_PTR;
        bufparams.shared_mem = (struct pl_shared_mem) {
            .handle.ptr = params->ptr,
            .size = bufparams.size,
            .offset = 0,
        };

        // Suppress errors for this test because it may fail, in which case we
        // want to silently fall back.
        pl_log_level_cap(gpu->log, PL_LOG_DEBUG);
        buf = pl_buf_create(gpu, &bufparams);
        pl_log_level_cap(gpu->log, PL_LOG_NONE);
    }

    if (!buf) {
        // Fallback when host pointer import is not supported
        bufparams.import_handle = 0;
        bufparams.host_readable = true;
        buf = pl_buf_create(gpu, &bufparams);
    }

    if (!buf)
        return false;

    struct pl_tex_transfer_params newparams = *params;
    newparams.ptr = NULL;
    newparams.buf = buf;

    // If the transfer is asynchronous, propagate our host read asynchronously
    if (params->callback && !bufparams.import_handle) {
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
    if (bufparams.import_handle) {
        // Buffer download completion already means the host pointer contains
        // the valid data, no more need to copy. (Note: this applies even for
        // asynchronous downloads)
        ok = true;
        pl_buf_destroy(gpu, &buf);
    } else if (!params->callback) {
        // Synchronous read back to the host pointer
        ok = pl_buf_read(gpu, buf, 0, params->ptr, bufparams.size);
        pl_buf_destroy(gpu, &buf);
    } else {
        // Nothing left to do here, the rest will be done by pbo_download_cb
        ok = true;
    }

    return ok;
}

bool pl_tex_upload_texel(pl_gpu gpu, pl_dispatch dp,
                         const struct pl_tex_transfer_params *params)
{
    const int threads = PL_MIN(256, pl_rect_w(params->rc));
    pl_tex tex = params->tex;
    pl_fmt fmt = tex->params.format;
    require(params->buf);

    pl_shader sh = pl_dispatch_begin(dp);
    if (!sh_try_compute(sh, threads, 1, false, 0)) {
        PL_ERR(gpu, "Failed emulating texture transfer!");
        pl_dispatch_abort(dp, &sh);
        return false;
    }

    bool ubo = params->buf->params.uniform;
    ident_t buf = sh_desc(sh, (struct pl_shader_desc) {
        .binding.object = params->buf,
        .desc = {
            .name = "data",
            .type = ubo ? PL_DESC_BUF_TEXEL_UNIFORM : PL_DESC_BUF_TEXEL_STORAGE,
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
    int groups_x = (pl_rect_w(params->rc) + threads - 1) / threads;
    if (groups_x * threads != pl_rect_w(params->rc)) {
        GLSL("if (gl_GlobalInvocationID.x >= %d) \n"
             "    return;                        \n",
             pl_rect_w(params->rc));
    }

    // fmt->texel_align contains the size of an individual color value
    assert(fmt->texel_size == fmt->num_components * fmt->texel_align);
    GLSL("vec4 color = vec4(0.0, 0.0, 0.0, 1.0);                        \n"
         "ivec3 pos = ivec3(gl_GlobalInvocationID) + ivec3(%d, %d, %d); \n"
         "int base = pos.z * %s + pos.y * %s + pos.x * %s;              \n",
         params->rc.x0, params->rc.y0, params->rc.z0,
         SH_INT(params->depth_pitch / fmt->texel_align),
         SH_INT(params->row_pitch / fmt->texel_align),
         SH_INT(fmt->texel_size / fmt->texel_align));

    for (int i = 0; i < fmt->num_components; i++) {
        GLSL("color[%d] = %s(%s, base + %d).r; \n",
             i, ubo ? "texelFetch" : "imageLoad", buf, i);
    }

    int dims = pl_tex_params_dimension(tex->params);
    static const char *coord_types[] = {
        [1] = "int",
        [2] = "ivec2",
        [3] = "ivec3",
    };

    GLSL("imageStore(%s, %s(pos), color);\n", img, coord_types[dims]);
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

bool pl_tex_download_texel(pl_gpu gpu, pl_dispatch dp,
                           const struct pl_tex_transfer_params *params)
{
    const int threads = PL_MIN(256, pl_rect_w(params->rc));
    pl_tex tex = params->tex;
    pl_fmt fmt = tex->params.format;
    require(params->buf);

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

    int groups_x = (pl_rect_w(params->rc) + threads - 1) / threads;
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
    GLSL("ivec3 pos = ivec3(gl_GlobalInvocationID) + ivec3(%d, %d, %d); \n"
         "int base = pos.z * %s + pos.y * %s + pos.x * %s;              \n"
         "vec4 color = imageLoad(%s, %s(pos));                          \n",
         params->rc.x0, params->rc.y0, params->rc.z0,
         SH_INT(params->depth_pitch / fmt->texel_align),
         SH_INT(params->row_pitch / fmt->texel_align),
         SH_INT(fmt->texel_size / fmt->texel_align),
         img, coord_types[dims]);

    for (int i = 0; i < fmt->num_components; i++)
        GLSL("imageStore(%s, base + %d, vec4(color[%d])); \n", buf, i, i);

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

bool pl_tex_blit_compute(pl_gpu gpu, pl_dispatch dp,
                         const struct pl_tex_blit_params *params)
{
    if (!params->src->params.storable || !params->dst->params.storable)
        return false;

    // Normalize `dst_rc`, moving all flipping to `src_rc` instead.
    struct pl_rect3d src_rc = params->src_rc;
    struct pl_rect3d dst_rc = params->dst_rc;
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

    // Manual trilinear interpolation would be too slow to justify
    bool needs_sampling = needs_scaling && params->sample_mode != PL_TEX_SAMPLE_NEAREST;
    if (needs_sampling && !params->src->params.sampleable)
        return false;

    const int threads = 256;
    int bw = PL_MIN(32, pl_rect_w(dst_rc));
    int bh = PL_MIN(threads / bw, pl_rect_h(dst_rc));
    pl_shader sh = pl_dispatch_begin(dp);
    if (!sh_try_compute(sh, bw, bh, false, 0)) {
        pl_dispatch_abort(dp, &sh);
        return false;
    }

    // Avoid over-writing into `dst`
    int groups_x = (pl_rect_w(dst_rc) + bw - 1) / bw;
    if (groups_x * bw != pl_rect_w(dst_rc)) {
        GLSL("if (gl_GlobalInvocationID.x >= %d) \n"
             "    return;                        \n",
             pl_rect_w(dst_rc));
    }

    int groups_y = (pl_rect_h(dst_rc) + bh - 1) / bh;
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
    GLSL("const ivec3 pos = ivec3(gl_GlobalInvocationID);   \n"
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

        GLSL("vec3 fpos = (vec3(pos) + vec3(0.5)) / vec3(%d.0, %d.0, %d.0); \n"
             "%s src_pos = %s(0.5);                                         \n"
             "src_pos.x = mix(%f, %f, fpos.x);                              \n",
             pl_rect_w(dst_rc), pl_rect_h(dst_rc), pl_rect_d(dst_rc),
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

        GLSL("imageStore(%s, dst_pos, %s(%s, src_pos)); \n",
             dst, sh_tex_fn(sh, params->src->params), src);

    } else {

        ident_t src = sh_desc(sh, (struct pl_shader_desc) {
            .binding.object = params->src,
            .desc = {
                .name   = "src",
                .type   = PL_DESC_STORAGE_IMG,
                .access = PL_DESC_ACCESS_READONLY,
            },
        });

        if (needs_scaling) {
            GLSL("ivec3 src_pos = ivec3(round(vec3(%f, %f, %f) * vec3(pos))); \n",
                 fabs((float) pl_rect_w(src_rc) / pl_rect_w(dst_rc)),
                 fabs((float) pl_rect_h(src_rc) / pl_rect_h(dst_rc)),
                 fabs((float) pl_rect_d(src_rc) / pl_rect_d(dst_rc)));
        } else {
            GLSL("ivec3 src_pos = pos; \n");
        }

        GLSL("src_pos = ivec3(%d, %d, %d) * src_pos + ivec3(%d, %d, %d);    \n"
             "imageStore(%s, dst_pos, imageLoad(%s, %s(src_pos)));          \n",
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

void pl_tex_blit_raster(pl_gpu gpu, pl_dispatch dp,
                        const struct pl_tex_blit_params *params)
{
    enum pl_fmt_type src_type = params->src->params.format->type;
    enum pl_fmt_type dst_type = params->dst->params.format->type;

    // Only for 2D textures
    pl_assert(params->src->params.h && !params->src->params.d);
    pl_assert(params->dst->params.h && !params->dst->params.d);

    // Integer textures are not supported
    pl_assert(src_type != PL_FMT_UINT && src_type != PL_FMT_SINT);
    pl_assert(dst_type != PL_FMT_UINT && dst_type != PL_FMT_SINT);

    struct pl_rect2df src_rc = {
        .x0 = params->src_rc.x0, .x1 = params->src_rc.x1,
        .y0 = params->src_rc.y0, .y1 = params->src_rc.y1,
    };
    struct pl_rect2d dst_rc = {
        .x0 = params->dst_rc.x0, .x1 = params->dst_rc.x1,
        .y0 = params->dst_rc.y0, .y1 = params->dst_rc.y1,
    };

    pl_shader sh = pl_dispatch_begin(dp);
    sh->res.output = PL_SHADER_SIG_COLOR;

    ident_t pos, src = sh_bind(sh, params->src, PL_TEX_ADDRESS_CLAMP,
        params->sample_mode, "src_tex", &src_rc, &pos, NULL, NULL);

    GLSL("vec4 color = %s(%s, %s); \n",
         sh_tex_fn(sh, params->src->params), src, pos);

    pl_dispatch_finish(dp, pl_dispatch_params(
        .shader = &sh,
        .target = params->dst,
        .rect = dst_rc,
    ));
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
    new.cached_program = NULL;
    new.cached_program_len = 0;

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

pl_sync pl_sync_create(pl_gpu gpu, enum pl_handle_type handle_type)
{
    require(handle_type);
    require(handle_type & gpu->export_caps.sync);
    require(PL_ISPOT(handle_type));

    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    return impl->sync_create(gpu, handle_type);

error:
    return NULL;
}

void pl_sync_destroy(pl_gpu gpu, pl_sync *sync)
{
    if (!*sync)
        return;

    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    impl->sync_destroy(gpu, *sync);
    *sync = NULL;
}

bool pl_tex_export(pl_gpu gpu, pl_tex tex, pl_sync sync)
{
    require(tex->params.import_handle || tex->params.export_handle);

    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    return impl->tex_export(gpu, tex, sync);

error:
    if (tex->params.debug_tag)
        PL_ERR(gpu, "  for texture: %s", tex->params.debug_tag);
    return false;
}

pl_timer pl_timer_create(pl_gpu gpu)
{
    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    if (!impl->timer_create)
        return NULL;

    return impl->timer_create(gpu);
}

void pl_timer_destroy(pl_gpu gpu, pl_timer *timer)
{
    if (!*timer)
        return;

    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    impl->timer_destroy(gpu, *timer);
    *timer = NULL;
}

uint64_t pl_timer_query(pl_gpu gpu, pl_timer timer)
{
    if (!timer)
        return 0;

    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    return impl->timer_query(gpu, timer);
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
