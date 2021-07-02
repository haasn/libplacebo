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

#include "gpu.h"
#include "formats.h"

static inline UINT tex_subresource(pl_tex tex)
{
    struct pl_tex_d3d11 *tex_p = PL_PRIV(tex);
    return tex_p->array_slice >= 0 ? tex_p->array_slice : 0;
}

static bool tex_init(pl_gpu gpu, pl_tex tex)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;
    struct pl_tex_d3d11 *tex_p = PL_PRIV(tex);

    if (tex->params.sampleable || tex->params.storable) {
        // The SRV format may be omitted when it matches the texture format, but
        // for simplicity's sake we always set it. It will match the texture
        // format for textures created with tex_create, but it can be different
        // for video textures wrapped with pl_d3d11_wrap.
        D3D11_SHADER_RESOURCE_VIEW_DESC srvdesc = {
            .Format = fmt_to_dxgi(tex->params.format),
        };
        switch (pl_tex_params_dimension(tex->params)) {
        case 1:
            if (tex_p->array_slice >= 0) {
                srvdesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE1DARRAY;
                srvdesc.Texture1DArray.MipLevels = 1;
                srvdesc.Texture1DArray.FirstArraySlice = tex_p->array_slice;
                srvdesc.Texture1DArray.ArraySize = 1;
            } else {
                srvdesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE1D;
                srvdesc.Texture1D.MipLevels = 1;
            }
            break;
        case 2:
            if (tex_p->array_slice >= 0) {
                srvdesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
                srvdesc.Texture2DArray.MipLevels = 1;
                srvdesc.Texture2DArray.FirstArraySlice = tex_p->array_slice;
                srvdesc.Texture2DArray.ArraySize = 1;
            } else {
                srvdesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
                srvdesc.Texture2D.MipLevels = 1;
            }
            break;
        case 3:
            // D3D11 does not have Texture3D arrays
            srvdesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;
            srvdesc.Texture3D.MipLevels = 1;
            break;
        }
        D3D(ID3D11Device_CreateShaderResourceView(p->dev, tex_p->res, &srvdesc,
                                                  &tex_p->srv));
    }

    if (tex->params.renderable) {
        D3D(ID3D11Device_CreateRenderTargetView(p->dev, tex_p->res, NULL,
                                                &tex_p->rtv));
    }

    if (p->fl >= D3D_FEATURE_LEVEL_11_0 && tex->params.storable) {
        D3D(ID3D11Device_CreateUnorderedAccessView(p->dev, tex_p->res, NULL,
                                                   &tex_p->uav));
    }

    return true;
error:
    return false;
}

void pl_d3d11_tex_destroy(pl_gpu gpu, pl_tex tex)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;
    struct pl_tex_d3d11 *tex_p = PL_PRIV(tex);

    SAFE_RELEASE(tex_p->srv);
    SAFE_RELEASE(tex_p->rtv);
    SAFE_RELEASE(tex_p->uav);
    SAFE_RELEASE(tex_p->res);
    SAFE_RELEASE(tex_p->staging);

    pl_d3d11_flush_message_queue(ctx, "After texture destroy");

    pl_free((void *) tex);
}

pl_tex pl_d3d11_tex_create(pl_gpu gpu, const struct pl_tex_params *params)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;

    struct pl_tex *tex = pl_zalloc_obj(NULL, tex, struct pl_tex_d3d11);
    tex->params = *params;
    tex->params.initial_data = NULL;
    tex->sampler_type = PL_SAMPLER_NORMAL;

    struct pl_tex_d3d11 *tex_p = PL_PRIV(tex);

    DXGI_FORMAT dxfmt = fmt_to_dxgi(params->format);

    D3D11_USAGE usage = D3D11_USAGE_DEFAULT;
    D3D11_BIND_FLAG bind_flags = 0;

    if (p->fl >= D3D_FEATURE_LEVEL_11_0) {
        // On >=FL11_0, blit emulation needs image storage
        tex->params.storable |= params->blit_src || params->blit_dst;

        // Blit emulation can use a sampler for linear filtering during stretch
        if ((tex->params.format->caps & PL_FMT_CAP_LINEAR) && params->blit_src)
            tex->params.sampleable = true;
    } else {
        // On <FL11_0, blit emulation uses a render pass
        tex->params.sampleable |= params->blit_src;
        tex->params.renderable |= params->blit_dst;
    }

    if (tex->params.sampleable)
        bind_flags |= D3D11_BIND_SHADER_RESOURCE;
    if (tex->params.renderable)
        bind_flags |= D3D11_BIND_RENDER_TARGET;
    if (p->fl >= D3D_FEATURE_LEVEL_11_0 && tex->params.storable)
        bind_flags |= D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;

    // Apparently IMMUTABLE textures are efficient, so try to infer whether we
    // can use one
    if (params->initial_data && !tex->params.renderable &&
        !tex->params.storable && !params->host_writable)
        usage = D3D11_USAGE_IMMUTABLE;

    // In FL9_x, resources with only D3D11_BIND_SHADER_RESOURCE can't be copied
    // from GPU-accessible memory to CPU-accessible memory. The only other bind
    // flag we set on this FL is D3D11_BIND_RENDER_TARGET, so set it.
    if (p->fl <= D3D_FEATURE_LEVEL_9_3 && tex->params.host_readable)
        bind_flags |= D3D11_BIND_RENDER_TARGET;

    // In FL9_x, when using DEFAULT or IMMUTABLE, BindFlags cannot be zero
    if (p->fl <= D3D_FEATURE_LEVEL_9_3 && !bind_flags)
        bind_flags |= D3D11_BIND_SHADER_RESOURCE;

    D3D11_SUBRESOURCE_DATA data;
    D3D11_SUBRESOURCE_DATA *pdata = NULL;
    if (params->initial_data) {
        data = (D3D11_SUBRESOURCE_DATA) {
            .pSysMem = params->initial_data,
            .SysMemPitch = params->w * params->format->texel_size,
        };
        if (params->d)
            data.SysMemSlicePitch = data.SysMemPitch * params->h;
        pdata = &data;
    }

    switch (pl_tex_params_dimension(*params)) {
    case 1:;
        D3D11_TEXTURE1D_DESC desc1d = {
            .Width = params->w,
            .MipLevels = 1,
            .ArraySize = 1,
            .Format = dxfmt,
            .Usage = usage,
            .BindFlags = bind_flags,
        };
        D3D(ID3D11Device_CreateTexture1D(p->dev, &desc1d, pdata, &tex_p->tex1d));
        tex_p->res = (ID3D11Resource *)tex_p->tex1d;

        // Create a staging texture with CPU access for pl_tex_download()
        if (params->host_readable) {
            desc1d.BindFlags = 0;
            desc1d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            desc1d.Usage = D3D11_USAGE_STAGING;

            D3D(ID3D11Device_CreateTexture1D(p->dev, &desc1d, NULL,
                                             &tex_p->staging1d));
            tex_p->staging = (ID3D11Resource *) tex_p->staging1d;
        }
        break;
    case 2:;
        D3D11_TEXTURE2D_DESC desc2d = {
            .Width = params->w,
            .Height = params->h,
            .MipLevels = 1,
            .ArraySize = 1,
            .SampleDesc.Count = 1,
            .Format = dxfmt,
            .Usage = usage,
            .BindFlags = bind_flags,
        };
        D3D(ID3D11Device_CreateTexture2D(p->dev, &desc2d, pdata, &tex_p->tex2d));
        tex_p->res = (ID3D11Resource *)tex_p->tex2d;

        // Create a staging texture with CPU access for pl_tex_download()
        if (params->host_readable) {
            desc2d.BindFlags = 0;
            desc2d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            desc2d.Usage = D3D11_USAGE_STAGING;

            D3D(ID3D11Device_CreateTexture2D(p->dev, &desc2d, NULL,
                                             &tex_p->staging2d));
            tex_p->staging = (ID3D11Resource *) tex_p->staging2d;
        }
        break;
    case 3:;
        D3D11_TEXTURE3D_DESC desc3d = {
            .Width = params->w,
            .Height = params->h,
            .Depth = params->d,
            .MipLevels = 1,
            .Format = dxfmt,
            .Usage = usage,
            .BindFlags = bind_flags,
        };
        D3D(ID3D11Device_CreateTexture3D(p->dev, &desc3d, pdata, &tex_p->tex3d));
        tex_p->res = (ID3D11Resource *)tex_p->tex3d;

        // Create a staging texture with CPU access for pl_tex_download()
        if (params->host_readable) {
            desc3d.BindFlags = 0;
            desc3d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            desc3d.Usage = D3D11_USAGE_STAGING;

            D3D(ID3D11Device_CreateTexture3D(p->dev, &desc3d, NULL,
                                             &tex_p->staging3d));
            tex_p->staging = (ID3D11Resource *) tex_p->staging3d;
        }
        break;
    default:
        pl_unreachable();
    }

    tex_p->array_slice = -1;

    if (!tex_init(gpu, tex))
        goto error;

    pl_d3d11_flush_message_queue(ctx, "After texture create");

    return tex;

error:
    pl_d3d11_tex_destroy(gpu, tex);
    return NULL;
}

pl_tex pl_d3d11_wrap(pl_gpu gpu, const struct pl_d3d11_wrap_params *params)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;

    struct pl_tex *tex = pl_zalloc_obj(NULL, tex, struct pl_tex_d3d11);
    tex->sampler_type = PL_SAMPLER_NORMAL;

    struct pl_tex_d3d11 *tex_p = PL_PRIV(tex);

    DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
    D3D11_USAGE usage = D3D11_USAGE_DEFAULT;
    D3D11_BIND_FLAG bind_flags = 0;
    UINT mip_levels = 1;
    UINT array_size = 1;
    UINT sample_count = 1;

    D3D11_RESOURCE_DIMENSION type;
    ID3D11Resource_GetType(params->tex, &type);

    switch (type) {
    case D3D11_RESOURCE_DIMENSION_TEXTURE1D:
        D3D(ID3D11Resource_QueryInterface(params->tex, &IID_ID3D11Texture1D,
                                          (void **) &tex_p->tex1d));
        tex_p->res = (ID3D11Resource *) tex_p->tex1d;

        D3D11_TEXTURE1D_DESC desc1d;
        ID3D11Texture1D_GetDesc(tex_p->tex1d, &desc1d);

        tex->params.w = desc1d.Width;
        mip_levels = desc1d.MipLevels;
        array_size = desc1d.ArraySize;
        fmt = desc1d.Format;
        usage = desc1d.Usage;
        bind_flags = desc1d.BindFlags;
        break;

    case D3D11_RESOURCE_DIMENSION_TEXTURE2D:
        D3D(ID3D11Resource_QueryInterface(params->tex, &IID_ID3D11Texture2D,
                                          (void **) &tex_p->tex2d));
        tex_p->res = (ID3D11Resource *) tex_p->tex2d;

        D3D11_TEXTURE2D_DESC desc2d;
        ID3D11Texture2D_GetDesc(tex_p->tex2d, &desc2d);

        tex->params.w = desc2d.Width;
        tex->params.h = desc2d.Height;
        mip_levels = desc2d.MipLevels;
        array_size = desc2d.ArraySize;
        fmt = desc2d.Format;
        sample_count = desc2d.SampleDesc.Count;
        usage = desc2d.Usage;
        bind_flags = desc2d.BindFlags;

        // Allow the format and size of 2D textures to be overridden to support
        // shader views of video resources
        if (params->fmt) {
            fmt = params->fmt;
            tex->params.w = params->w;
            tex->params.h = params->h;
        }

        break;

    case D3D11_RESOURCE_DIMENSION_TEXTURE3D:
        D3D(ID3D11Resource_QueryInterface(params->tex, &IID_ID3D11Texture3D,
                                          (void **) &tex_p->tex3d));
        tex_p->res = (ID3D11Resource *) tex_p->tex3d;

        D3D11_TEXTURE3D_DESC desc3d;
        ID3D11Texture3D_GetDesc(tex_p->tex3d, &desc3d);

        tex->params.w = desc3d.Width;
        tex->params.h = desc3d.Height;
        tex->params.d = desc3d.Depth;
        mip_levels = desc3d.MipLevels;
        fmt = desc3d.Format;
        usage = desc3d.Usage;
        bind_flags = desc3d.BindFlags;
        break;

    case D3D11_RESOURCE_DIMENSION_UNKNOWN:
    case D3D11_RESOURCE_DIMENSION_BUFFER:
        PL_ERR(gpu, "Resource is not suitable to wrap");
        goto error;
    }

    if (mip_levels != 1) {
        PL_ERR(gpu, "Mipmapped textures not supported for wrapping");
        goto error;
    }
    if (sample_count != 1) {
        PL_ERR(gpu, "Multisampled textures not supported for wrapping");
        goto error;
    }
    if (usage != D3D11_USAGE_DEFAULT) {
        PL_ERR(gpu, "Resource is not D3D11_USAGE_DEFAULT");
        goto error;
    }

    if (array_size > 1) {
        if (params->array_slice < 0 || params->array_slice >= array_size) {
            PL_ERR(gpu, "array_slice out of range");
            goto error;
        }
        tex_p->array_slice = params->array_slice;
    } else {
        tex_p->array_slice = -1;
    }

    if (bind_flags & D3D11_BIND_SHADER_RESOURCE) {
        tex->params.sampleable = true;

        // Blit emulation uses a render pass on <FL11_0
        if (p->fl < D3D_FEATURE_LEVEL_11_0)
            tex->params.blit_src = true;
    }
    if (bind_flags & D3D11_BIND_RENDER_TARGET) {
        tex->params.renderable = true;

        // Blit emulation uses a render pass on <FL11_0
        if (p->fl < D3D_FEATURE_LEVEL_11_0)
            tex->params.blit_dst = true;
    }
    static const D3D11_BIND_FLAG storable_flags =
        D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    if ((bind_flags & storable_flags) == storable_flags) {
        tex->params.storable = true;

        // Blit emulation uses image storage on >=FL11_0. A feature level check
        // isn't required because <FL11_0 doesn't have storable images.
        tex->params.blit_src = tex->params.blit_dst = true;
    }

    for (int i = 0; i < gpu->num_formats; i++) {
        DXGI_FORMAT target_fmt = fmt_to_dxgi(gpu->formats[i]);
        if (fmt == target_fmt) {
            tex->params.format = gpu->formats[i];
            break;
        }
    }
    if (!tex->params.format) {
        PL_ERR(gpu, "Could not find a suitable pl_fmt for wrapped resource");
        goto error;
    }

    if (!tex_init(gpu, tex))
        goto error;

    pl_d3d11_flush_message_queue(ctx, "After texture wrap");

    return tex;

error:
    pl_d3d11_tex_destroy(gpu, tex);
    return NULL;
}

void pl_d3d11_tex_invalidate(pl_gpu gpu, pl_tex tex)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;
    struct pl_tex_d3d11 *tex_p = PL_PRIV(tex);

    // Resource discarding requires D3D11.1
    if (!p->imm1)
        return;

    // Prefer discarding a view to discarding the whole resource. The reason
    // for this is that a pl_tex can refer to a single member of a texture
    // array. Discarding the SRV, RTV or UAV should only discard that member.
    if (tex_p->rtv) {
        ID3D11DeviceContext1_DiscardView(p->imm1, (ID3D11View *) tex_p->rtv);
    } else if (tex_p->uav) {
        ID3D11DeviceContext1_DiscardView(p->imm1, (ID3D11View *) tex_p->uav);
    } else if (tex_p->srv) {
        ID3D11DeviceContext1_DiscardView(p->imm1, (ID3D11View *) tex_p->srv);
    } else if (tex_p->array_slice < 0) {
        // If there are no views, only discard if the ID3D11Resource is not a
        // texture array
        ID3D11DeviceContext1_DiscardResource(p->imm1, tex_p->res);
    }

    pl_d3d11_flush_message_queue(ctx, "After texture invalidate");
}

void pl_d3d11_tex_clear_ex(pl_gpu gpu, pl_tex tex,
                           const union pl_clear_color color)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;
    struct pl_tex_d3d11 *tex_p = PL_PRIV(tex);

    if (tex->params.format->type == PL_FMT_UINT) {
        if (tex_p->uav) {
            ID3D11DeviceContext_ClearUnorderedAccessViewUint(p->imm, tex_p->uav,
                                                             color.u);
        } else {
            float c[4] = { color.u[0], color.u[1], color.u[2], color.u[3] };
            ID3D11DeviceContext_ClearRenderTargetView(p->imm, tex_p->rtv, c);
        }

    } else if (tex->params.format->type == PL_FMT_SINT) {
        if (tex_p->uav) {
            ID3D11DeviceContext_ClearUnorderedAccessViewUint(p->imm, tex_p->uav,
                                                             color.i);
        } else {
            float c[4] = { color.i[0], color.i[1], color.i[2], color.i[3] };
            ID3D11DeviceContext_ClearRenderTargetView(p->imm, tex_p->rtv, c);
        }

    } else if (tex_p->rtv) {
        ID3D11DeviceContext_ClearRenderTargetView(p->imm, tex_p->rtv, color.f);
    } else {
        ID3D11DeviceContext_ClearUnorderedAccessViewFloat(p->imm, tex_p->uav, color.f);
    }

    pl_d3d11_flush_message_queue(ctx, "After texture clear");
}

#define pl_rect3d_to_box(rc)                             \
    ((D3D11_BOX) {                                       \
        .left = rc.x0, .top = rc.y0, .front = rc.z0,     \
        .right = rc.x1, .bottom = rc.y1, .back = rc.z1,  \
    })

void pl_d3d11_tex_blit(pl_gpu gpu, const struct pl_tex_blit_params *params)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;
    struct pl_tex_d3d11 *src_p = PL_PRIV(params->src);
    DXGI_FORMAT src_fmt = fmt_to_dxgi(params->src->params.format);
    struct pl_tex_d3d11 *dst_p = PL_PRIV(params->dst);
    DXGI_FORMAT dst_fmt = fmt_to_dxgi(params->dst->params.format);

    // If the blit operation doesn't require flipping, scaling or format
    // conversion, we can use CopySubresourceRegion
    struct pl_rect3d src_rc = params->src_rc, dst_rc = params->dst_rc;
    if (pl_rect3d_eq(src_rc, dst_rc) && src_fmt == dst_fmt) {
        struct pl_rect3d rc = params->src_rc;
        pl_rect3d_normalize(&rc);

        ID3D11DeviceContext_CopySubresourceRegion(p->imm, dst_p->res,
            tex_subresource(params->dst), rc.x0, rc.y0, rc.z0, src_p->res,
            tex_subresource(params->src), &pl_rect3d_to_box(rc));
    } else if (p->fl >= D3D_FEATURE_LEVEL_11_0) {
        if (!pl_tex_blit_compute(gpu, p->dp, params))
            PL_ERR(gpu, "Failed compute shader fallback blit");
    } else {
        pl_tex_blit_raster(gpu, p->dp, params);
    }

    pl_d3d11_flush_message_queue(ctx, "After texture blit");
}

bool pl_d3d11_tex_upload(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;
    pl_tex tex = params->tex;
    struct pl_tex_d3d11 *tex_p = PL_PRIV(tex);

    UINT row_pitch = params->stride_w * tex->params.format->texel_size;
    UINT depth_pitch = row_pitch * params->stride_h;

    pl_d3d11_timer_start(gpu, params->timer);

    ID3D11DeviceContext_UpdateSubresource(p->imm, tex_p->res,
        tex_subresource(tex), &pl_rect3d_to_box(params->rc), params->ptr,
        row_pitch, depth_pitch);

    pl_d3d11_timer_end(gpu, params->timer);
    pl_d3d11_flush_message_queue(ctx, "After texture upload");

    return true;
}

bool pl_d3d11_tex_download(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;
    const struct pl_tex *tex = params->tex;
    struct pl_tex_d3d11 *tex_p = PL_PRIV(tex);

    if (!tex_p->staging)
        return false;

    pl_d3d11_timer_start(gpu, params->timer);

    ID3D11DeviceContext_CopySubresourceRegion(p->imm,
        (ID3D11Resource *) tex_p->staging, 0, params->rc.x0, params->rc.y0,
        params->rc.z0, tex_p->res, tex_subresource(tex),
        &pl_rect3d_to_box(params->rc));

    UINT row_pitch = params->stride_w * tex->params.format->texel_size;
    UINT depth_pitch = row_pitch * params->stride_h;

    D3D11_MAPPED_SUBRESOURCE lock;
    D3D(ID3D11DeviceContext_Map(p->imm, (ID3D11Resource *) tex_p->staging, 0,
                                D3D11_MAP_READ, 0, &lock));

    char *cdst = params->ptr;
    char *csrc = lock.pData;
    size_t line_size = pl_rect_w(params->rc) * tex->params.format->texel_size;
    for (int z = 0; z < pl_rect_d(params->rc); z++) {
        for (int y = 0; y < pl_rect_h(params->rc); y++) {
            memcpy(cdst + z * depth_pitch + y * row_pitch,
                   csrc + (params->rc.z0 + z) * lock.DepthPitch +
                          (params->rc.y0 + y) * lock.RowPitch + params->rc.x0,
                   line_size);
        }
    }

    ID3D11DeviceContext_Unmap(p->imm, (ID3D11Resource*)tex_p->staging, 0);

    pl_d3d11_timer_end(gpu, params->timer);
    pl_d3d11_flush_message_queue(ctx, "After texture download");

    return true;

error:
    return false;
}
