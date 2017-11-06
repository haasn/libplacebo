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
    (struct ra_fmt) {                           \
        .name = _name,                          \
        .type = RA_FMT_##ftype,                 \
        .num_components  = num,                 \
        .component_depth = bits,                \
        .opaque          = false,               \
        .texel_size      = size,                \
        .host_bits       = bits,                \
        .sample_order    = idx,                 \
    }

#define IDX(...)  {__VA_ARGS__}
#define BITS(...) {__VA_ARGS__}

#define REGFMT(name, num, bits, type)           \
    FMT(name, num, (num) * (bits) / 8, type,    \
        BITS(bits, bits, bits, bits),           \
        IDX(0, 1, 2, 3))

const struct vk_format vk_formats[] = {
    // Regular, byte-aligned integer formats
    {VK_FORMAT_R8_UNORM,              REGFMT("r8",       1,  8, UNORM)},
    {VK_FORMAT_R8G8_UNORM,            REGFMT("rg8",      2,  8, UNORM)},
    {VK_FORMAT_R8G8B8_UNORM,          REGFMT("rgb8",     3,  8, UNORM)},
    {VK_FORMAT_R8G8B8A8_UNORM,        REGFMT("rgba8",    4,  8, UNORM)},
    {VK_FORMAT_R16_UNORM,             REGFMT("r16",      1, 16, UNORM)},
    {VK_FORMAT_R16G16_UNORM,          REGFMT("rg16",     2, 16, UNORM)},
    {VK_FORMAT_R16G16B16_UNORM,       REGFMT("rgb16",    3, 16, UNORM)},
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
    {VK_FORMAT_R64_SFLOAT,            REGFMT("r64df",    1, 64, FLOAT)},
    {VK_FORMAT_R64G64_SFLOAT,         REGFMT("rg64df",   2, 64, FLOAT)},
    {VK_FORMAT_R64G64B64_SFLOAT,      REGFMT("rgb64df",  3, 64, FLOAT)},
    {VK_FORMAT_R64G64B64A64_SFLOAT,   REGFMT("rgba64df", 4, 64, FLOAT)},

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
    {VK_FORMAT_R64_UINT,              REGFMT("r64u",     1, 64, UINT)},
    {VK_FORMAT_R64G64_UINT,           REGFMT("rg64u",    2, 64, UINT)},
    {VK_FORMAT_R64G64B64_UINT,        REGFMT("rgb64u",   3, 64, UINT)},
    {VK_FORMAT_R64G64B64A64_UINT,     REGFMT("rgba64u",  4, 64, UINT)},

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
    {VK_FORMAT_R64_SINT,              REGFMT("r64i",     1, 64, SINT)},
    {VK_FORMAT_R64G64_SINT,           REGFMT("rg64i",    2, 64, SINT)},
    {VK_FORMAT_R64G64B64_SINT,        REGFMT("rgb64i",   3, 64, SINT)},
    {VK_FORMAT_R64G64B64A64_SINT,     REGFMT("rgba64i",  4, 64, SINT)},

    // "Swapped" component order formats
    {VK_FORMAT_B8G8R8_UNORM,             FMT("bgr8",     3,  3, UNORM, BITS(8,  8,  8),     IDX(2, 1, 0))},
    {VK_FORMAT_B8G8R8A8_UNORM,           FMT("bgra8",    4,  4, UNORM, BITS(8,  8,  8,  8), IDX(2, 1, 0, 3))},
    {VK_FORMAT_B4G4R4A4_UNORM_PACK16,    FMT("bgra4",    4,  2, UNORM, BITS(4,  4,  4,  4), IDX(2, 1, 0, 3))},
    {VK_FORMAT_B5G6R5_UNORM_PACK16,      FMT("bgr565",   3,  2, UNORM, BITS(5,  6,  5),     IDX(2, 1, 0))},
    {VK_FORMAT_B5G5R5A1_UNORM_PACK16,    FMT("bgr5a1",   4,  2, UNORM, BITS(5,  5,  5,  1), IDX(2, 1, 0, 3))},
    {VK_FORMAT_A1R5G5B5_UNORM_PACK16,    FMT("a1rgb5",   4,  2, UNORM, BITS(1,  5,  5,  5), IDX(3, 0, 1, 2))},
    {VK_FORMAT_A2R10G10B10_UNORM_PACK32, FMT("a2rgb10",  4,  4, UNORM, BITS(2, 10, 10, 10), IDX(3, 0, 1, 2))},
    {VK_FORMAT_A2B10G10R10_UNORM_PACK32, FMT("a2bgr10",  4,  4, UNORM, BITS(2, 10, 10, 10), IDX(3, 2, 1, 0))},
    {VK_FORMAT_A8B8G8R8_UNORM_PACK32,    FMT("abgr8",    4,  4, UNORM, BITS(8,  8,  8,  8), IDX(3, 2, 1, 0))},
    {VK_FORMAT_A2R10G10B10_SNORM_PACK32, FMT("a2rgb10s", 4,  4, SNORM, BITS(2, 10, 10, 10), IDX(3, 0, 1, 2))},
    {VK_FORMAT_A2B10G10R10_SNORM_PACK32, FMT("a2bgr10s", 4,  4, SNORM, BITS(2, 10, 10, 10), IDX(3, 2, 1, 0))},
    {VK_FORMAT_A8B8G8R8_SNORM_PACK32,    FMT("abgr8s",   4,  4, SNORM, BITS(8,  8,  8,  8), IDX(3, 2, 1, 0))},

    {VK_FORMAT_B8G8R8_UINT,              FMT("bgr8u",    3,  3, UINT,  BITS(8,  8,  8),     IDX(2, 1, 0))},
    {VK_FORMAT_B8G8R8A8_UINT,            FMT("bgra8u",   4,  4, UINT,  BITS(8,  8,  8,  8), IDX(2, 1, 0, 3))},
    {VK_FORMAT_A2R10G10B10_UINT_PACK32,  FMT("a2rgb10u", 4,  4, UINT,  BITS(2, 10, 10, 10), IDX(3, 0, 1, 2))},
    {VK_FORMAT_A2B10G10R10_UINT_PACK32,  FMT("a2bgr10u", 4,  4, UINT,  BITS(2, 10, 10, 10), IDX(3, 2, 1, 0))},
    {VK_FORMAT_A8B8G8R8_UINT_PACK32,     FMT("abgr8u",   4,  4, UINT,  BITS(8,  8,  8,  8), IDX(3, 2, 1, 0))},

    {VK_FORMAT_B8G8R8_SINT,              FMT("bgr8i",    3,  3, SINT,  BITS(8,  8,  8),     IDX(2, 1, 0))},
    {VK_FORMAT_B8G8R8A8_SINT,            FMT("bgra8i",   4,  4, SINT,  BITS(8,  8,  8,  8), IDX(2, 1, 0, 3))},
    {VK_FORMAT_A2R10G10B10_SINT_PACK32,  FMT("a2rgb10i", 4,  4, SINT,  BITS(2, 10, 10, 10), IDX(3, 0, 1, 2))},
    {VK_FORMAT_A2B10G10R10_SINT_PACK32,  FMT("a2bgr10i", 4,  4, SINT,  BITS(2, 10, 10, 10), IDX(3, 2, 1, 0))},
    {VK_FORMAT_A8B8G8R8_SINT_PACK32,     FMT("abgr8i",   4,  4, SINT,  BITS(8,  8,  8,  8), IDX(3, 2, 1, 0))},

    // Special, packed integer formats (low bit depth)
    {VK_FORMAT_R4G4_UNORM_PACK8,      REGFMT("rg4",      2,  4, UNORM)},
    {VK_FORMAT_R4G4B4A4_UNORM_PACK16, REGFMT("rgba4",    4,  4, UNORM)},
    {VK_FORMAT_R5G6B5_UNORM_PACK16,   FMT("rgb565",      3,  2, UNORM, BITS(5,  6,  5),     IDX(0, 1, 2))},
    {VK_FORMAT_R5G5B5A1_UNORM_PACK16, FMT("rgb5a1",      4,  2, UNORM, BITS(5,  5,  5,  1), IDX(0, 1, 2, 3))},
    {0}
};

#undef BITS
#undef IDX
#undef REGFMT
#undef FMT
