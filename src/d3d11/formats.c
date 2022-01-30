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

#include "formats.h"
#include "gpu.h"

#define FMT(_minor, _name, _dxfmt, _type, num, size, bits, order)  \
    (struct d3d_format) {                                          \
        .dxfmt = DXGI_FORMAT_##_dxfmt##_##_type,                   \
        .minor = _minor,                                           \
        .fmt = {                                                   \
            .name = _name,                                         \
            .type = PL_FMT_##_type,                                \
            .num_components  = num,                                \
            .component_depth = bits,                               \
            .texel_size      = size,                               \
            .texel_align     = 1,                                  \
            .internal_size   = size,                               \
            .host_bits       = bits,                               \
            .sample_order    = order,                              \
        },                                                         \
    }

#define IDX(...) {__VA_ARGS__}
#define BITS(...) {__VA_ARGS__}

#define REGFMT(name, dxfmt, type, num, bits)            \
    FMT(0, name, dxfmt, type, num, (num) * (bits) / 8,  \
        BITS(bits, bits, bits, bits),                   \
        IDX(0, 1, 2, 3))

const struct d3d_format pl_d3d11_formats[] = {
    REGFMT("r8",       R8,           UNORM, 1,  8),
    REGFMT("rg8",      R8G8,         UNORM, 2,  8),
    REGFMT("rgba8",    R8G8B8A8,     UNORM, 4,  8),
    REGFMT("r16",      R16,          UNORM, 1, 16),
    REGFMT("rg16",     R16G16,       UNORM, 2, 16),
    REGFMT("rgba16",   R16G16B16A16, UNORM, 4, 16),

    REGFMT("r8s",      R8,           SNORM, 1,  8),
    REGFMT("rg8s",     R8G8,         SNORM, 2,  8),
    REGFMT("rgba8s",   R8G8B8A8,     SNORM, 4,  8),
    REGFMT("r16s",     R16,          SNORM, 1, 16),
    REGFMT("rg16s",    R16G16,       SNORM, 2, 16),
    REGFMT("rgba16s",  R16G16B16A16, SNORM, 4, 16),

    REGFMT("r16hf",    R16,          FLOAT, 1, 16),
    REGFMT("rg16hf",   R16G16,       FLOAT, 2, 16),
    REGFMT("rgba16hf", R16G16B16A16, FLOAT, 4, 16),
    REGFMT("r32f",     R32,          FLOAT, 1, 32),
    REGFMT("rg32f",    R32G32,       FLOAT, 2, 32),
    REGFMT("rgb32f",   R32G32B32,    FLOAT, 3, 32),
    REGFMT("rgba32f",  R32G32B32A32, FLOAT, 4, 32),

    REGFMT("r8u",      R8,           UINT,  1,  8),
    REGFMT("rg8u",     R8G8,         UINT,  2,  8),
    REGFMT("rgba8u",   R8G8B8A8,     UINT,  4,  8),
    REGFMT("r16u",     R16,          UINT,  1, 16),
    REGFMT("rg16u",    R16G16,       UINT,  2, 16),
    REGFMT("rgba16u",  R16G16B16A16, UINT,  4, 16),
    REGFMT("r32u",     R32,          UINT,  1, 32),
    REGFMT("rg32u",    R32G32,       UINT,  2, 32),
    REGFMT("rgb32u",   R32G32B32,    UINT,  3, 32),
    REGFMT("rgba32u",  R32G32B32A32, UINT,  4, 32),

    REGFMT("r8i",      R8,           SINT,  1,  8),
    REGFMT("rg8i",     R8G8,         SINT,  2,  8),
    REGFMT("rgba8i",   R8G8B8A8,     SINT,  4,  8),
    REGFMT("r16i",     R16,          SINT,  1, 16),
    REGFMT("rg16i",    R16G16,       SINT,  2, 16),
    REGFMT("rgba16i",  R16G16B16A16, SINT,  4, 16),
    REGFMT("r32i",     R32,          SINT,  1, 32),
    REGFMT("rg32i",    R32G32,       SINT,  2, 32),
    REGFMT("rgb32i",   R32G32B32,    SINT,  3, 32),
    REGFMT("rgba32i",  R32G32B32A32, SINT,  4, 32),

    FMT(0, "rgb10a2",  R10G10B10A2,  UNORM, 4,  4, BITS(10, 10, 10,  2), IDX(0, 1, 2, 3)),
    FMT(0, "rgb10a2u", R10G10B10A2,  UINT,  4,  4, BITS(10, 10, 10,  2), IDX(0, 1, 2, 3)),

    FMT(0, "bgra8",    B8G8R8A8,     UNORM, 4,  4, BITS( 8,  8,  8,  8), IDX(2, 1, 0, 3)),
    FMT(0, "bgrx8",    B8G8R8X8,     UNORM, 3,  4, BITS( 8,  8,  8),     IDX(2, 1, 0)),
    FMT(0, "rg11b10f", R11G11B10,    FLOAT, 3,  4, BITS(11, 11, 10),     IDX(0, 1, 2)),

     // D3D11.1 16-bit formats (resurrected D3D9 formats)
    FMT(1, "bgr565",   B5G6R5,       UNORM, 3,  2, BITS( 5,  6,  5),     IDX(2, 1, 0)),
    FMT(1, "bgr5a1",   B5G5R5A1,     UNORM, 4,  2, BITS( 5,  5,  5,  1), IDX(2, 1, 0, 3)),
    FMT(1, "bgra4",    B4G4R4A4,     UNORM, 4,  2, BITS( 4,  4,  4,  4), IDX(2, 1, 0, 3)),

    {0}
};
#undef BITS
#undef IDX
#undef REGFMT
#undef FMT

void pl_d3d11_setup_formats(struct pl_gpu *gpu)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    PL_ARRAY(pl_fmt) formats = {0};
    HRESULT hr;

    for (int i = 0; pl_d3d11_formats[i].dxfmt; i++) {
        const struct d3d_format *d3d_fmt = &pl_d3d11_formats[i];

        // The Direct3D 11.0 debug layer will segfault if CheckFormatSupport is
        // called on a format it doesn't know about
        if (pl_d3d11_formats[i].minor > p->minor)
            continue;

        UINT sup = 0;
        hr = ID3D11Device_CheckFormatSupport(p->dev, d3d_fmt->dxfmt, &sup);
        if (FAILED(hr))
            continue;

        D3D11_FEATURE_DATA_FORMAT_SUPPORT2 sup2 = { .InFormat = d3d_fmt->dxfmt };
        ID3D11Device_CheckFeatureSupport(p->dev, D3D11_FEATURE_FORMAT_SUPPORT2,
                                         &sup2, sizeof(sup2));

        struct pl_fmt *fmt = pl_alloc_obj(gpu, fmt, struct d3d_fmt *);
        const struct d3d_format **fmtp = PL_PRIV(fmt);
        *fmt = d3d_fmt->fmt;
        *fmtp = d3d_fmt;

        // For sanity, clear the superfluous fields
        for (int j = fmt->num_components; j < 4; j++) {
            fmt->component_depth[j] = 0;
            fmt->sample_order[j] = 0;
            fmt->host_bits[j] = 0;
        }

        static const struct {
            enum pl_fmt_caps caps;
            UINT sup;
            UINT sup2;
        } support[] = {
            {
                .caps = PL_FMT_CAP_SAMPLEABLE,
                .sup = D3D11_FORMAT_SUPPORT_TEXTURE2D,
            },
            {
                .caps = PL_FMT_CAP_STORABLE,
                // SHADER_LOAD is for readonly images, which can use a SRV
                .sup = D3D11_FORMAT_SUPPORT_TEXTURE2D |
                       D3D11_FORMAT_SUPPORT_TYPED_UNORDERED_ACCESS_VIEW |
                       D3D11_FORMAT_SUPPORT_SHADER_LOAD,
                .sup2 = D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE,
            },
            {
                .caps = PL_FMT_CAP_READWRITE,
                .sup = D3D11_FORMAT_SUPPORT_TEXTURE2D |
                       D3D11_FORMAT_SUPPORT_TYPED_UNORDERED_ACCESS_VIEW,
                .sup2 = D3D11_FORMAT_SUPPORT2_UAV_TYPED_LOAD,
            },
            {
                .caps = PL_FMT_CAP_LINEAR,
                .sup = D3D11_FORMAT_SUPPORT_TEXTURE2D |
                       D3D11_FORMAT_SUPPORT_SHADER_SAMPLE,
            },
            {
                .caps = PL_FMT_CAP_RENDERABLE,
                .sup = D3D11_FORMAT_SUPPORT_RENDER_TARGET,
            },
            {
                .caps = PL_FMT_CAP_BLENDABLE,
                .sup = D3D11_FORMAT_SUPPORT_RENDER_TARGET |
                       D3D11_FORMAT_SUPPORT_BLENDABLE,
            },
            {
                .caps = PL_FMT_CAP_VERTEX,
                .sup = D3D11_FORMAT_SUPPORT_IA_VERTEX_BUFFER,
            },
            {
                .caps = PL_FMT_CAP_TEXEL_UNIFORM,
                .sup = D3D11_FORMAT_SUPPORT_BUFFER |
                       D3D11_FORMAT_SUPPORT_SHADER_LOAD,
            },
            {
                .caps = PL_FMT_CAP_TEXEL_STORAGE,
                // SHADER_LOAD is for readonly buffers, which can use a SRV
                .sup = D3D11_FORMAT_SUPPORT_BUFFER |
                       D3D11_FORMAT_SUPPORT_TYPED_UNORDERED_ACCESS_VIEW |
                       D3D11_FORMAT_SUPPORT_SHADER_LOAD,
                .sup2 = D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE,
            },
            {
                .caps = PL_FMT_CAP_HOST_READABLE,
                .sup = D3D11_FORMAT_SUPPORT_CPU_LOCKABLE,
            },
        };

        for (int j = 0; j < PL_ARRAY_SIZE(support); j++) {
            if ((sup & support[j].sup) == support[j].sup &&
                (sup2.OutFormatSupport2 & support[j].sup2) == support[j].sup2)
            {
                fmt->caps |= support[j].caps;
            }
        }

        // PL_FMT_CAP_STORABLE implies compute shaders, so don't set it if we
        // don't have them
        if (!gpu->glsl.compute)
            fmt->caps &= ~PL_FMT_CAP_STORABLE;

        // PL_FMT_CAP_READWRITE implies PL_FMT_CAP_STORABLE
        if (!(fmt->caps & PL_FMT_CAP_STORABLE))
            fmt->caps &= ~PL_FMT_CAP_READWRITE;

        // We can't sample from integer textures
        if (fmt->type == PL_FMT_UINT || fmt->type == PL_FMT_SINT)
            fmt->caps &= ~(PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_LINEAR);

        // `fmt->gatherable` must have PL_FMT_CAP_SAMPLEABLE
        if ((fmt->caps & PL_FMT_CAP_SAMPLEABLE) &&
            (sup & D3D11_FORMAT_SUPPORT_SHADER_GATHER))
        {
            fmt->gatherable = true;
        }

        // PL_FMT_CAP_BLITTABLE implies support for stretching, flipping and
        // loose format conversion, which require a shader pass in D3D11
        if (p->fl >= D3D_FEATURE_LEVEL_11_0) {
            // On >=FL11_0, we use a compute pass, which supports 1D and 3D
            // textures
            if (fmt->caps & PL_FMT_CAP_STORABLE)
                fmt->caps |= PL_FMT_CAP_BLITTABLE;
        } else {
            // On <FL11_0 we use a raster pass
            static const enum pl_fmt_caps req = PL_FMT_CAP_RENDERABLE |
                                                PL_FMT_CAP_SAMPLEABLE;
            if ((fmt->caps & req) == req)
                fmt->caps |= PL_FMT_CAP_BLITTABLE;
        }

        if (fmt->caps & (PL_FMT_CAP_VERTEX | PL_FMT_CAP_TEXEL_UNIFORM |
                                             PL_FMT_CAP_TEXEL_STORAGE)) {
            fmt->glsl_type = pl_var_glsl_type_name(pl_var_from_fmt(fmt, ""));
            pl_assert(fmt->glsl_type);
        }

        if (fmt->caps & (PL_FMT_CAP_STORABLE | PL_FMT_CAP_TEXEL_STORAGE))
            fmt->glsl_format = pl_fmt_glsl_format(fmt, fmt->num_components);

        // If no caps, D3D11 only supports this for things we don't care about
        if (!fmt->caps) {
            pl_free(fmt);
            continue;
        }

        PL_ARRAY_APPEND(gpu, formats, fmt);
    }

    gpu->formats = formats.elem;
    gpu->num_formats = formats.num;
}
