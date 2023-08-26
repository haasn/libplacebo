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
 * License along with libplacebo.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "formats.h"

#define FMT(_name, num, size, ftype, bits, idx) \
    (struct pl_fmt_t) {                         \
        .name = _name,                          \
        .type = PL_FMT_##ftype,                 \
        .num_components  = num,                 \
        .component_depth = bits,                \
        .internal_size   = size,                \
        .opaque          = false,               \
        .texel_size      = size,                \
        .texel_align     = size,                \
        .host_bits       = bits,                \
        .sample_order    = idx,                 \
    }

#define IDX(...)  {__VA_ARGS__}
#define BITS(...) {__VA_ARGS__}

#define REGFMT(name, num, bits, type)           \
    FMT(name, num, (num) * (bits) / 8, type,    \
        BITS(bits, bits, bits, bits),           \
        IDX(0, 1, 2, 3))

#define EMUFMT(_name, in, en, ib, eb, ftype)    \
    (struct pl_fmt_t) {                         \
        .name = _name,                          \
        .type = PL_FMT_##ftype,                 \
        .num_components  = en,                  \
        .component_depth = BITS(ib, ib, ib, ib),\
        .internal_size   = (in) * (ib) / 8,     \
        .opaque          = false,               \
        .emulated        = true,                \
        .texel_size      = (en) * (eb) / 8,     \
        .texel_align     = (eb) / 8,            \
        .host_bits       = BITS(eb, eb, eb, eb),\
        .sample_order    = IDX(0, 1, 2, 3),     \
    }

#define PACKED16FMT(_name, num, b)              \
    (struct pl_fmt_t) {                         \
        .name            = _name,               \
        .type            = PL_FMT_UNORM,        \
        .num_components  = num,                 \
        .component_depth = BITS(b, b, b, b),    \
        .internal_size   = (num) * 2,           \
        .texel_size      = (num) * 2,           \
        .texel_align     = (num) * 2,           \
        .host_bits       = BITS(16, 16, 16, 16),\
        .sample_order    = IDX(0, 1, 2, 3),     \
    }

#define PLANARFMT(_name, planes, size, bits)    \
    (struct pl_fmt_t) {                         \
        .name            = _name,               \
        .type            = PL_FMT_UNORM,        \
        .num_planes      = planes,              \
        .num_components  = 3,                   \
        .component_depth = {bits, bits, bits},  \
        .internal_size   = size,                \
        .opaque          = true,                \
    }

static const struct vk_format rgb8e = {
    .tfmt   = VK_FORMAT_R8G8B8A8_UNORM,
    .bfmt   = VK_FORMAT_R8G8B8_UNORM,
    .icomps = 4,
    .fmt    = EMUFMT("rgb8", 4, 3, 8, 8, UNORM),
};

static const struct vk_format rgb16e = {
    .tfmt   = VK_FORMAT_R16G16B16A16_UNORM,
    .bfmt   = VK_FORMAT_R16G16B16_UNORM,
    .icomps = 4,
    .fmt    = EMUFMT("rgb16", 4, 3, 16, 16, UNORM),
};

static const struct vk_format vk_formats[] = {
    // Regular, byte-aligned integer formats
    {VK_FORMAT_R8_UNORM,              REGFMT("r8",       1,  8, UNORM)},
    {VK_FORMAT_R8G8_UNORM,            REGFMT("rg8",      2,  8, UNORM)},
    {VK_FORMAT_R8G8B8_UNORM,          REGFMT("rgb8",     3,  8, UNORM), .emufmt = &rgb8e},
    {VK_FORMAT_R8G8B8A8_UNORM,        REGFMT("rgba8",    4,  8, UNORM)},
    {VK_FORMAT_R16_UNORM,             REGFMT("r16",      1, 16, UNORM)},
    {VK_FORMAT_R16G16_UNORM,          REGFMT("rg16",     2, 16, UNORM)},
    {VK_FORMAT_R16G16B16_UNORM,       REGFMT("rgb16",    3, 16, UNORM), .emufmt = &rgb16e},
    {VK_FORMAT_R16G16B16A16_UNORM,    REGFMT("rgba16",   4, 16, UNORM)},

    {VK_FORMAT_R8_SNORM,              REGFMT("r8s",      1,  8, SNORM)},
    {VK_FORMAT_R8G8_SNORM,            REGFMT("rg8s",     2,  8, SNORM)},
    {VK_FORMAT_R8G8B8_SNORM,          REGFMT("rgb8s",    3,  8, SNORM)},
    {VK_FORMAT_R8G8B8A8_SNORM,        REGFMT("rgba8s",   4,  8, SNORM)},
    {VK_FORMAT_R16_SNORM,             REGFMT("r16s",     1, 16, SNORM)},
    {VK_FORMAT_R16G16_SNORM,          REGFMT("rg16s",    2, 16, SNORM)},
    {VK_FORMAT_R16G16B16_SNORM,       REGFMT("rgb16s",   3, 16, SNORM)},
    {VK_FORMAT_R16G16B16A16_SNORM,    REGFMT("rgba16s",  4, 16, SNORM)},

    // Float formats (native formats: hf = half float, df = double float)
    {VK_FORMAT_R16_SFLOAT,            REGFMT("r16hf",    1, 16, FLOAT)},
    {VK_FORMAT_R16G16_SFLOAT,         REGFMT("rg16hf",   2, 16, FLOAT)},
    {VK_FORMAT_R16G16B16_SFLOAT,      REGFMT("rgb16hf",  3, 16, FLOAT)},
    {VK_FORMAT_R16G16B16A16_SFLOAT,   REGFMT("rgba16hf", 4, 16, FLOAT)},
    {VK_FORMAT_R32_SFLOAT,            REGFMT("r32f",     1, 32, FLOAT)},
    {VK_FORMAT_R32G32_SFLOAT,         REGFMT("rg32f",    2, 32, FLOAT)},
    {VK_FORMAT_R32G32B32_SFLOAT,      REGFMT("rgb32f",   3, 32, FLOAT)},
    {VK_FORMAT_R32G32B32A32_SFLOAT,   REGFMT("rgba32f",  4, 32, FLOAT)},

    // Float formats (emulated upload/download)
    {VK_FORMAT_R16_SFLOAT,            EMUFMT("r16f",     1, 1, 16, 32, FLOAT)},
    {VK_FORMAT_R16G16_SFLOAT,         EMUFMT("rg16f",    2, 2, 16, 32, FLOAT)},
    {VK_FORMAT_R16G16B16_SFLOAT,      EMUFMT("rgb16f",   3, 3, 16, 32, FLOAT)},
    {VK_FORMAT_R16G16B16A16_SFLOAT,   EMUFMT("rgba16f",  4, 4, 16, 32, FLOAT)},

    // Integer-sampled formats
    {VK_FORMAT_R8_UINT,               REGFMT("r8u",      1,  8, UINT)},
    {VK_FORMAT_R8G8_UINT,             REGFMT("rg8u",     2,  8, UINT)},
    {VK_FORMAT_R8G8B8_UINT,           REGFMT("rgb8u",    3,  8, UINT)},
    {VK_FORMAT_R8G8B8A8_UINT,         REGFMT("rgba8u",   4,  8, UINT)},
    {VK_FORMAT_R16_UINT,              REGFMT("r16u",     1, 16, UINT)},
    {VK_FORMAT_R16G16_UINT,           REGFMT("rg16u",    2, 16, UINT)},
    {VK_FORMAT_R16G16B16_UINT,        REGFMT("rgb16u",   3, 16, UINT)},
    {VK_FORMAT_R16G16B16A16_UINT,     REGFMT("rgba16u",  4, 16, UINT)},
    {VK_FORMAT_R32_UINT,              REGFMT("r32u",     1, 32, UINT)},
    {VK_FORMAT_R32G32_UINT,           REGFMT("rg32u",    2, 32, UINT)},
    {VK_FORMAT_R32G32B32_UINT,        REGFMT("rgb32u",   3, 32, UINT)},
    {VK_FORMAT_R32G32B32A32_UINT,     REGFMT("rgba32u",  4, 32, UINT)},

    {VK_FORMAT_R8_SINT,               REGFMT("r8i",      1,  8, SINT)},
    {VK_FORMAT_R8G8_SINT,             REGFMT("rg8i",     2,  8, SINT)},
    {VK_FORMAT_R8G8B8_SINT,           REGFMT("rgb8i",    3,  8, SINT)},
    {VK_FORMAT_R8G8B8A8_SINT,         REGFMT("rgba8i",   4,  8, SINT)},
    {VK_FORMAT_R16_SINT,              REGFMT("r16i",     1, 16, SINT)},
    {VK_FORMAT_R16G16_SINT,           REGFMT("rg16i",    2, 16, SINT)},
    {VK_FORMAT_R16G16B16_SINT,        REGFMT("rgb16i",   3, 16, SINT)},
    {VK_FORMAT_R16G16B16A16_SINT,     REGFMT("rgba16i",  4, 16, SINT)},
    {VK_FORMAT_R32_SINT,              REGFMT("r32i",     1, 32, SINT)},
    {VK_FORMAT_R32G32_SINT,           REGFMT("rg32i",    2, 32, SINT)},
    {VK_FORMAT_R32G32B32_SINT,        REGFMT("rgb32i",   3, 32, SINT)},
    {VK_FORMAT_R32G32B32A32_SINT,     REGFMT("rgba32i",  4, 32, SINT)},

    // "Swapped" component order formats
    {VK_FORMAT_B8G8R8_UNORM,             FMT("bgr8",     3,  3, UNORM, BITS(8,  8,  8),     IDX(2, 1, 0))},
    {VK_FORMAT_B8G8R8A8_UNORM,           FMT("bgra8",    4,  4, UNORM, BITS(8,  8,  8,  8), IDX(2, 1, 0, 3))},

    {VK_FORMAT_B8G8R8_UINT,              FMT("bgr8u",    3,  3, UINT,  BITS(8,  8,  8),     IDX(2, 1, 0))},
    {VK_FORMAT_B8G8R8A8_UINT,            FMT("bgra8u",   4,  4, UINT,  BITS(8,  8,  8,  8), IDX(2, 1, 0, 3))},

    {VK_FORMAT_B8G8R8_SINT,              FMT("bgr8i",    3,  3, SINT,  BITS(8,  8,  8),     IDX(2, 1, 0))},
    {VK_FORMAT_B8G8R8A8_SINT,            FMT("bgra8i",   4,  4, SINT,  BITS(8,  8,  8,  8), IDX(2, 1, 0, 3))},

    // "Packed" integer formats
    //
    // Note: These have the component order reversed from what the vulkan name
    // implies, because we order our IDX from LSB to MSB (consistent with the
    // usual ordering from lowest byte to highest byte, on little endian
    // platforms), but Vulkan names them from MSB to LSB.
    {VK_FORMAT_R4G4_UNORM_PACK8,         FMT("gr4",      2,  1, UNORM, BITS(4,  4),         IDX(1, 0))},
    {VK_FORMAT_B4G4R4A4_UNORM_PACK16,    FMT("argb4",    4,  2, UNORM, BITS(4,  4,  4,  4), IDX(3, 0, 1, 2))},
    {VK_FORMAT_R4G4B4A4_UNORM_PACK16,    FMT("abgr4",    4,  2, UNORM, BITS(4,  4,  4,  4), IDX(3, 2, 1, 0))},

    {VK_FORMAT_R5G6B5_UNORM_PACK16,      FMT("bgr565",   3,  2, UNORM, BITS(5,  6,  5),     IDX(2, 1, 0))},
    {VK_FORMAT_B5G6R5_UNORM_PACK16,      FMT("rgb565",   3,  2, UNORM, BITS(5,  6,  5),     IDX(0, 1, 2))},

    {VK_FORMAT_R5G5B5A1_UNORM_PACK16,    FMT("a1bgr5",   4,  2, UNORM, BITS(1,  5,  5,  5), IDX(3, 2, 1, 0))},
    {VK_FORMAT_B5G5R5A1_UNORM_PACK16,    FMT("a1rgb5",   4,  2, UNORM, BITS(1,  5,  5,  5), IDX(3, 0, 1, 2))},
    {VK_FORMAT_A1R5G5B5_UNORM_PACK16,    FMT("bgr5a1",   4,  2, UNORM, BITS(5,  5,  5,  1), IDX(2, 1, 0, 3))},

    {VK_FORMAT_A2B10G10R10_UNORM_PACK32, FMT("rgb10a2",  4,  4, UNORM, BITS(10, 10, 10, 2), IDX(0, 1, 2, 3))},
    {VK_FORMAT_A2R10G10B10_UNORM_PACK32, FMT("bgr10a2",  4,  4, UNORM, BITS(10, 10, 10, 2), IDX(2, 1, 0, 3))},
    {VK_FORMAT_A2B10G10R10_SNORM_PACK32, FMT("rgb10a2s", 4,  4, SNORM, BITS(10, 10, 10, 2), IDX(0, 1, 2, 3))},
    {VK_FORMAT_A2R10G10B10_SNORM_PACK32, FMT("bgr10a2s", 4,  4, SNORM, BITS(10, 10, 10, 2), IDX(2, 1, 0, 3))},
    {VK_FORMAT_A2B10G10R10_UINT_PACK32,  FMT("rgb10a2u", 4,  4, UINT,  BITS(10, 10, 10, 2), IDX(0, 1, 2, 3))},
    {VK_FORMAT_A2R10G10B10_UINT_PACK32,  FMT("bgr10a2u", 4,  4, UINT,  BITS(10, 10, 10, 2), IDX(2, 1, 0, 3))},
    {VK_FORMAT_A2B10G10R10_SINT_PACK32,  FMT("rgb10a2i", 4,  4, SINT,  BITS(10, 10, 10, 2), IDX(0, 1, 2, 3))},
    {VK_FORMAT_A2R10G10B10_SINT_PACK32,  FMT("bgr10a2i", 4,  4, SINT,  BITS(10, 10, 10, 2), IDX(2, 1, 0, 3))},


    // Packed 16 bit formats
    {VK_FORMAT_R10X6_UNORM_PACK16,                  PACKED16FMT("rx10",         1, 10)},
    {VK_FORMAT_R10X6G10X6_UNORM_2PACK16,            PACKED16FMT("rxgx10",       2, 10)},
    {VK_FORMAT_R12X4_UNORM_PACK16,                  PACKED16FMT("rx12",         1, 12)},
    {VK_FORMAT_R12X4G12X4_UNORM_2PACK16,            PACKED16FMT("rxgx12",       2, 12)},

    // FIXME: enabling these requires VK_EXT_rgba10x6_formats or equivalent
    // {VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16,  PACKED16FMT("rxgxbxax10",   4, 10)},
    // {VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16,  PACKED16FMT("rxgxbxax12",   4, 12)},

    // Planar formats
    {VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM, PLANARFMT("g8_b8_r8_420", 3, 12, 8),
        .pfmt = {
            {VK_FORMAT_R8_UNORM},
            {VK_FORMAT_R8_UNORM, .sx = 1, .sy = 1},
            {VK_FORMAT_R8_UNORM, .sx = 1, .sy = 1},
        },
    },
    {VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM, PLANARFMT("g8_b8_r8_422", 3, 16, 8),
        .pfmt = {
            {VK_FORMAT_R8_UNORM},
            {VK_FORMAT_R8_UNORM, .sx = 1},
            {VK_FORMAT_R8_UNORM, .sx = 1},
        },
    },
    {VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM, PLANARFMT("g8_b8_r8_444", 3, 24, 8),
        .pfmt = {
            {VK_FORMAT_R8_UNORM},
            {VK_FORMAT_R8_UNORM},
            {VK_FORMAT_R8_UNORM},
        },
    },

    {VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM, PLANARFMT("g16_b16_r16_420", 3, 24, 16),
        .pfmt = {
            {VK_FORMAT_R16_UNORM},
            {VK_FORMAT_R16_UNORM, .sx = 1, .sy = 1},
            {VK_FORMAT_R16_UNORM, .sx = 1, .sy = 1},
        },
    },
    {VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM, PLANARFMT("g16_b16_r16_422", 3, 32, 16),
        .pfmt = {
            {VK_FORMAT_R16_UNORM},
            {VK_FORMAT_R16_UNORM, .sx = 1},
            {VK_FORMAT_R16_UNORM, .sx = 1},
        },
    },
    {VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM, PLANARFMT("g16_b16_r16_444", 3, 48, 16),
        .pfmt = {
            {VK_FORMAT_R16_UNORM},
            {VK_FORMAT_R16_UNORM},
            {VK_FORMAT_R16_UNORM},
        },
    },

    {VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16, PLANARFMT("gx10_bx10_rx10_420", 3, 24, 10),
        .pfmt = {
            {VK_FORMAT_R10X6_UNORM_PACK16},
            {VK_FORMAT_R10X6_UNORM_PACK16, .sx = 1, .sy = 1},
            {VK_FORMAT_R10X6_UNORM_PACK16, .sx = 1, .sy = 1},
        },
    },
    {VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16, PLANARFMT("gx10_bx10_rx10_422", 3, 32, 10),
        .pfmt = {
            {VK_FORMAT_R10X6_UNORM_PACK16},
            {VK_FORMAT_R10X6_UNORM_PACK16, .sx = 1},
            {VK_FORMAT_R10X6_UNORM_PACK16, .sx = 1},
        },
    },
    {VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16, PLANARFMT("gx10_bx10_rx10_444", 3, 48, 10),
        .pfmt = {
            {VK_FORMAT_R10X6_UNORM_PACK16},
            {VK_FORMAT_R10X6_UNORM_PACK16},
            {VK_FORMAT_R10X6_UNORM_PACK16},
        },
    },

    {VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16, PLANARFMT("gx12_bx12_rx12_420", 3, 24, 12),
        .pfmt = {
            {VK_FORMAT_R12X4_UNORM_PACK16},
            {VK_FORMAT_R12X4_UNORM_PACK16, .sx = 1, .sy = 1},
            {VK_FORMAT_R12X4_UNORM_PACK16, .sx = 1, .sy = 1},
        },
    },
    {VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16, PLANARFMT("gx12_bx12_rx12_422", 3, 32, 12),
        .pfmt = {
            {VK_FORMAT_R12X4_UNORM_PACK16},
            {VK_FORMAT_R12X4_UNORM_PACK16, .sx = 1},
            {VK_FORMAT_R12X4_UNORM_PACK16, .sx = 1},
        },
    },
    {VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16, PLANARFMT("gx12_bx12_rx12_444", 3, 48, 12),
        .pfmt = {
            {VK_FORMAT_R12X4_UNORM_PACK16},
            {VK_FORMAT_R12X4_UNORM_PACK16},
            {VK_FORMAT_R12X4_UNORM_PACK16},
        },
    },

    {VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, PLANARFMT("g8_br8_420", 2, 12, 8),
        .pfmt = {
            {VK_FORMAT_R8_UNORM},
            {VK_FORMAT_R8G8_UNORM, .sx = 1, .sy = 1},
        },
    },
    {VK_FORMAT_G8_B8R8_2PLANE_422_UNORM, PLANARFMT("g8_br8_422", 2, 16, 8),
        .pfmt = {
            {VK_FORMAT_R8_UNORM},
            {VK_FORMAT_R8G8_UNORM, .sx = 1},
        },
    },
    {VK_FORMAT_G8_B8R8_2PLANE_444_UNORM, PLANARFMT("g8_br8_444", 2, 24, 8),
        .min_ver = VK_API_VERSION_1_3,
        .pfmt = {
            {VK_FORMAT_R8_UNORM},
            {VK_FORMAT_R8G8_UNORM},
        },
    },

    {VK_FORMAT_G16_B16R16_2PLANE_420_UNORM, PLANARFMT("g16_br16_420", 2, 24, 16),
        .pfmt = {
            {VK_FORMAT_R16_UNORM},
            {VK_FORMAT_R16G16_UNORM, .sx = 1, .sy = 1},
        },
    },
    {VK_FORMAT_G16_B16R16_2PLANE_422_UNORM, PLANARFMT("g16_br16_422", 2, 32, 16),
        .pfmt = {
            {VK_FORMAT_R16_UNORM},
            {VK_FORMAT_R16G16_UNORM, .sx = 1},
        },
    },
    {VK_FORMAT_G16_B16R16_2PLANE_444_UNORM, PLANARFMT("g16_br16_444", 2, 48, 16),
        .min_ver = VK_API_VERSION_1_3,
        .pfmt = {
            {VK_FORMAT_R16_UNORM},
            {VK_FORMAT_R16G16_UNORM},
        },
    },

    {VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16, PLANARFMT("gx10_bxrx10_420", 2, 24, 10),
        .pfmt = {
            {VK_FORMAT_R10X6_UNORM_PACK16},
            {VK_FORMAT_R10X6G10X6_UNORM_2PACK16, .sx = 1, .sy = 1},
        },
    },
    {VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16, PLANARFMT("gx10_bxrx10_422", 2, 32, 10),
        .pfmt = {
            {VK_FORMAT_R10X6_UNORM_PACK16},
            {VK_FORMAT_R10X6G10X6_UNORM_2PACK16, .sx = 1},
        },
    },
    {VK_FORMAT_G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16, PLANARFMT("gx10_bxrx10_444", 2, 48, 10),
        .min_ver = VK_API_VERSION_1_3,
        .pfmt = {
            {VK_FORMAT_R10X6_UNORM_PACK16},
            {VK_FORMAT_R10X6G10X6_UNORM_2PACK16},
        },
    },

    {VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16, PLANARFMT("gx12_bxrx12_420", 2, 24, 12),
        .pfmt = {
            {VK_FORMAT_R12X4_UNORM_PACK16},
            {VK_FORMAT_R12X4G12X4_UNORM_2PACK16, .sx = 1, .sy = 1},
        },
    },
    {VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16, PLANARFMT("gx12_bxrx12_422", 2, 32, 12),
        .pfmt = {
            {VK_FORMAT_R12X4_UNORM_PACK16},
            {VK_FORMAT_R12X4G12X4_UNORM_2PACK16, .sx = 1},
        },
    },
    {VK_FORMAT_G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16, PLANARFMT("gx12_bxrx12_444", 2, 48, 12),
        .min_ver = VK_API_VERSION_1_3,
        .pfmt = {
            {VK_FORMAT_R12X4_UNORM_PACK16},
            {VK_FORMAT_R12X4G12X4_UNORM_2PACK16},
        },
    },

    {0}
};

#undef BITS
#undef IDX
#undef REGFMT
#undef FMT

void vk_setup_formats(struct pl_gpu_t *gpu)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    PL_ARRAY(pl_fmt) formats = {0};

    // Texture format emulation requires at least support for texel buffers
    bool has_emu = gpu->glsl.compute && gpu->limits.max_buffer_texels;

    for (const struct vk_format *pvk_fmt = vk_formats; pvk_fmt->tfmt; pvk_fmt++) {
        const struct vk_format *vk_fmt = pvk_fmt;

        // Skip formats that require a too new version of Vulkan
        if (vk_fmt->min_ver > vk->api_ver)
            continue;

        // Skip formats with innately emulated representation if unsupported
        if (vk_fmt->fmt.emulated && !has_emu)
            continue;

        // Suppress some errors/warnings spit out by the format probing code
        pl_log_level_cap(vk->log, PL_LOG_INFO);

        bool has_drm_mods = vk->GetImageDrmFormatModifierPropertiesEXT;
        VkDrmFormatModifierPropertiesEXT modifiers[16] = {0};
        VkDrmFormatModifierPropertiesListEXT drm_props = {
            .sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT,
            .drmFormatModifierCount = PL_ARRAY_SIZE(modifiers),
            .pDrmFormatModifierProperties = modifiers,
        };

        VkFormatProperties2KHR prop2 = {
            .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
            .pNext = has_drm_mods ? &drm_props : NULL,
        };

        vk->GetPhysicalDeviceFormatProperties2KHR(vk->physd, vk_fmt->tfmt, &prop2);

        // If wholly unsupported, try falling back to the emulation formats
        // for texture operations
        VkFormatProperties *prop = &prop2.formatProperties;
        while (has_emu && !prop->optimalTilingFeatures && vk_fmt->emufmt) {
            vk_fmt = vk_fmt->emufmt;
            vk->GetPhysicalDeviceFormatProperties2KHR(vk->physd, vk_fmt->tfmt, &prop2);
        }

        VkFormatFeatureFlags texflags = prop->optimalTilingFeatures;
        VkFormatFeatureFlags bufflags = prop->bufferFeatures;
        if (vk_fmt->fmt.emulated) {
            // Emulated formats might have a different buffer representation
            // than their texture representation. If they don't, assume their
            // buffer representation is nonsensical (e.g. r16f)
            if (vk_fmt->bfmt) {
                vk->GetPhysicalDeviceFormatProperties(vk->physd, vk_fmt->bfmt, prop);
                bufflags = prop->bufferFeatures;
            } else {
                bufflags = 0;
            }
        } else if (vk_fmt->fmt.num_planes) {
            // Planar textures cannot be used directly
            texflags = bufflags = 0;
        }

        pl_log_level_cap(vk->log, PL_LOG_NONE);

        struct pl_fmt_t *fmt = pl_alloc_obj(gpu, fmt, struct pl_fmt_vk);
        struct pl_fmt_vk *fmtp = PL_PRIV(fmt);
        *fmt = vk_fmt->fmt;
        *fmtp = (struct pl_fmt_vk) {
            .vk_fmt = vk_fmt
        };

        // Always set the signature to the actual texture format, so we can use
        // it to guarantee renderpass compatibility.
        fmt->signature = (uint64_t) vk_fmt->tfmt;

        // For sanity, clear the superfluous fields
        for (int i = fmt->num_components; i < 4; i++) {
            fmt->component_depth[i] = 0;
            fmt->sample_order[i] = 0;
            fmt->host_bits[i] = 0;
        }

        // We can set this universally
        fmt->fourcc = pl_fmt_fourcc(fmt);

        if (has_drm_mods) {

            if (drm_props.drmFormatModifierCount == PL_ARRAY_SIZE(modifiers)) {
                PL_WARN(gpu, "DRM modifier list for format %s possibly truncated",
                        fmt->name);
            }

            // Query the list of supported DRM modifiers from the driver
            PL_ARRAY(uint64_t) modlist = {0};
            for (int i = 0; i < drm_props.drmFormatModifierCount; i++) {
                if (modifiers[i].drmFormatModifierPlaneCount > 1) {
                    PL_TRACE(gpu, "Ignoring format modifier %s of "
                             "format %s because its plane count %d > 1",
                             PRINT_DRM_MOD(modifiers[i].drmFormatModifier),
                             fmt->name, modifiers[i].drmFormatModifierPlaneCount);
                    continue;
                }

                // Only warn about texture format features relevant to us
                const VkFormatFeatureFlags flag_mask =
                    VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT |
                    VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                    VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                    VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
                    VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                    VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                    VK_FORMAT_FEATURE_BLIT_DST_BIT;


                VkFormatFeatureFlags flags = modifiers[i].drmFormatModifierTilingFeatures;
                if ((flags & flag_mask) != (texflags & flag_mask)) {
                    PL_DEBUG(gpu, "DRM format modifier %s of format %s "
                            "supports fewer caps (0x%"PRIx32") than optimal tiling "
                            "(0x%"PRIx32"), may result in limited capability!",
                            PRINT_DRM_MOD(modifiers[i].drmFormatModifier),
                            fmt->name, flags, texflags);
                }

                PL_ARRAY_APPEND(fmt, modlist, modifiers[i].drmFormatModifier);
            }

            fmt->num_modifiers = modlist.num;
            fmt->modifiers = modlist.elem;

        } else if (gpu->export_caps.tex & PL_HANDLE_DMA_BUF) {

            // Hard-code a list of static mods that we're likely to support
            static const uint64_t static_mods[2] = {
                DRM_FORMAT_MOD_INVALID,
                DRM_FORMAT_MOD_LINEAR,
            };

            fmt->num_modifiers = PL_ARRAY_SIZE(static_mods);
            fmt->modifiers = static_mods;

        }

        struct { VkFormatFeatureFlags flags; enum pl_fmt_caps caps; } bufbits[] = {
            {VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT,        PL_FMT_CAP_VERTEX},
            {VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT, PL_FMT_CAP_TEXEL_UNIFORM},
            {VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT, PL_FMT_CAP_TEXEL_STORAGE},
        };

        for (int i = 0; i < PL_ARRAY_SIZE(bufbits); i++) {
            if ((bufflags & bufbits[i].flags) == bufbits[i].flags)
                fmt->caps |= bufbits[i].caps;
        }

        if (fmt->caps) {
            fmt->glsl_type = pl_var_glsl_type_name(pl_var_from_fmt(fmt, ""));
            pl_assert(fmt->glsl_type);
        }

        struct { VkFormatFeatureFlags flags; enum pl_fmt_caps caps; } bits[] = {
            {VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT,      PL_FMT_CAP_BLENDABLE},
            {VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT, PL_FMT_CAP_LINEAR},
            {VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT,               PL_FMT_CAP_SAMPLEABLE},
            {VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT,               PL_FMT_CAP_STORABLE},
            {VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT,            PL_FMT_CAP_RENDERABLE},

            // We don't distinguish between the two blit modes for pl_fmt_caps
            {VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT,
                PL_FMT_CAP_BLITTABLE},
        };

        for (int i = 0; i < PL_ARRAY_SIZE(bits); i++) {
            if ((texflags & bits[i].flags) == bits[i].flags)
                fmt->caps |= bits[i].caps;
        }

        // For blit emulation via compute shaders
        if (!(fmt->caps & PL_FMT_CAP_BLITTABLE) && (fmt->caps & PL_FMT_CAP_STORABLE)) {
            fmt->caps |= PL_FMT_CAP_BLITTABLE;
            fmtp->blit_emulated = true;
        }

        // This is technically supported for all textures, but the semantics
        // of pl_gpu require it only be listed for non-opaque ones
        if (!fmt->opaque)
            fmt->caps |= PL_FMT_CAP_HOST_READABLE;

        // Vulkan requires a minimum GLSL version that supports textureGather()
        if (fmt->caps & PL_FMT_CAP_SAMPLEABLE)
            fmt->gatherable = true;

        // Disable implied capabilities where the dependencies are unavailable
        enum pl_fmt_caps storable = PL_FMT_CAP_STORABLE | PL_FMT_CAP_TEXEL_STORAGE;
        if (!(fmt->caps & PL_FMT_CAP_SAMPLEABLE))
            fmt->caps &= ~PL_FMT_CAP_LINEAR;
        if (!gpu->glsl.compute)
            fmt->caps &= ~storable;

        bool has_nofmt = vk->features.features.shaderStorageImageReadWithoutFormat &&
                         vk->features.features.shaderStorageImageWriteWithoutFormat;

        if (fmt->caps & storable) {
            int real_comps = PL_DEF(vk_fmt->icomps, fmt->num_components);
            fmt->glsl_format = pl_fmt_glsl_format(fmt, real_comps);
            if (!fmt->glsl_format && !has_nofmt) {
                PL_DEBUG(gpu, "Storable format '%s' has no matching GLSL "
                         "format qualifier but read/write without format "
                         "is not supported.. disabling", fmt->name);
                fmt->caps &= ~storable;
            }
        }

        if (fmt->caps & storable)
            fmt->caps |= PL_FMT_CAP_READWRITE;

        // Pick sub-plane formats for planar formats
        for (int n = 0; n < fmt->num_planes; n++) {
            for (int i = 0; i < formats.num; i++) {
                if (formats.elem[i]->signature == vk_fmt->pfmt[n].fmt) {
                    fmt->planes[n].format = formats.elem[i];
                    fmt->planes[n].shift_x = vk_fmt->pfmt[n].sx;
                    fmt->planes[n].shift_y = vk_fmt->pfmt[n].sy;
                    break;
                }
            }

            pl_assert(fmt->planes[n].format);
        }

        PL_ARRAY_APPEND(gpu, formats, fmt);
    }

    gpu->formats = formats.elem;
    gpu->num_formats = formats.num;
}
