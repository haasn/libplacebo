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

void pl_d3d11_buf_destroy(pl_gpu gpu, pl_buf buf)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;
    struct pl_buf_d3d11 *buf_p = PL_PRIV(buf);

    SAFE_RELEASE(buf_p->buf);
    SAFE_RELEASE(buf_p->staging);
    SAFE_RELEASE(buf_p->raw_srv);
    SAFE_RELEASE(buf_p->raw_uav);
    SAFE_RELEASE(buf_p->texel_srv);
    SAFE_RELEASE(buf_p->texel_uav);

    pl_d3d11_flush_message_queue(ctx, "After buffer destroy");

    pl_free((void *) buf);
}

pl_buf pl_d3d11_buf_create(pl_gpu gpu, const struct pl_buf_params *params)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;

    struct pl_buf_t *buf = pl_zalloc_obj(NULL, buf, struct pl_buf_d3d11);
    buf->params = *params;
    buf->params.initial_data = NULL;

    struct pl_buf_d3d11 *buf_p = PL_PRIV(buf);

    D3D11_BUFFER_DESC desc = { .ByteWidth = params->size };

    if (params->uniform && !params->format &&
        (params->storable || params->drawable))
    {
        // TODO: Figure out what to do with these
        PL_ERR(gpu, "Uniform buffers cannot share any other buffer type");
        goto error;
    }

    // TODO: Distinguish between uniform buffers and texel uniform buffers.
    // Currently we assume that if uniform and format are set, it's a texel
    // buffer and NOT a uniform buffer.
    if (params->uniform && !params->format) {
        desc.BindFlags |= D3D11_BIND_CONSTANT_BUFFER;
        desc.ByteWidth = PL_ALIGN2(desc.ByteWidth, CBUF_ELEM);
    }
    if (params->uniform && params->format) {
        desc.BindFlags |= D3D11_BIND_SHADER_RESOURCE;
    }
    if (params->storable) {
        desc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS
                        | D3D11_BIND_SHADER_RESOURCE;
        desc.ByteWidth = PL_ALIGN2(desc.ByteWidth, sizeof(float));
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    }
    if (params->drawable) {
        desc.BindFlags |= D3D11_BIND_VERTEX_BUFFER;

        // In FL9_x, a vertex buffer can't also be an index buffer, so index
        // buffers are unsupported in FL9_x for now
        if (p->fl > D3D_FEATURE_LEVEL_9_3)
            desc.BindFlags |= D3D11_BIND_INDEX_BUFFER;
    }

    char *data = NULL;

    // D3D11 doesn't allow partial constant buffer updates without special
    // conditions. To support partial buffer updates, keep a mirror of the
    // buffer data in system memory and upload the whole thing before the buffer
    // is used.
    //
    // Note: We don't use a staging buffer for this because of Intel.
    // https://github.com/mpv-player/mpv/issues/5293
    // https://crbug.com/593024
    if (params->uniform && !params->format && params->host_writable) {
        data = pl_zalloc(buf, desc.ByteWidth);
        buf_p->data = data;
    }

    D3D11_SUBRESOURCE_DATA srdata = { 0 };
    if (params->initial_data) {
        if (desc.ByteWidth != params->size) {
            // If the size had to be rounded-up, uploading from
            // params->initial_data is technically undefined behavior, so copy
            // the initial data to an allocation first
            if (!data)
                data = pl_zalloc(buf, desc.ByteWidth);
            srdata.pSysMem = data;
        } else {
            srdata.pSysMem = params->initial_data;
        }

        if (data)
            memcpy(data, params->initial_data, params->size);
    }

    D3D(ID3D11Device_CreateBuffer(p->dev, &desc,
                                  params->initial_data ? &srdata : NULL,
                                  &buf_p->buf));

    if (!buf_p->data)
        pl_free(data);

    // Create raw views for PL_DESC_BUF_STORAGE
    if (params->storable) {
        // A SRV is used for PL_DESC_ACCESS_READONLY
        D3D11_SHADER_RESOURCE_VIEW_DESC sdesc = {
            .Format = DXGI_FORMAT_R32_TYPELESS,
            .ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX,
            .BufferEx = {
                .NumElements =
                    PL_ALIGN2(buf->params.size, sizeof(float)) / sizeof(float),
                .Flags = D3D11_BUFFEREX_SRV_FLAG_RAW,
            },
        };
        D3D(ID3D11Device_CreateShaderResourceView(p->dev,
            (ID3D11Resource *) buf_p->buf, &sdesc, &buf_p->raw_srv));

        // A UAV is used for all other access modes
        D3D11_UNORDERED_ACCESS_VIEW_DESC udesc = {
            .Format = DXGI_FORMAT_R32_TYPELESS,
            .ViewDimension = D3D11_UAV_DIMENSION_BUFFER,
            .Buffer = {
                .NumElements =
                    PL_ALIGN2(buf->params.size, sizeof(float)) / sizeof(float),
                .Flags = D3D11_BUFFER_UAV_FLAG_RAW,
            },
        };
        D3D(ID3D11Device_CreateUnorderedAccessView(p->dev,
            (ID3D11Resource *) buf_p->buf, &udesc, &buf_p->raw_uav));
    }

    // Create a typed SRV for PL_BUF_TEXEL_UNIFORM and PL_BUF_TEXEL_STORAGE
    if (params->format) {
        if (params->uniform) {
            D3D11_SHADER_RESOURCE_VIEW_DESC sdesc = {
                .Format = fmt_to_dxgi(params->format),
                .ViewDimension = D3D11_SRV_DIMENSION_BUFFER,
                .Buffer = {
                    .NumElements =
                        PL_ALIGN(buf->params.size, buf->params.format->texel_size)
                            / buf->params.format->texel_size,
                },
            };
            D3D(ID3D11Device_CreateShaderResourceView(p->dev,
                (ID3D11Resource *) buf_p->buf, &sdesc, &buf_p->texel_srv));
        }

        // Create a typed UAV for PL_BUF_TEXEL_STORAGE
        if (params->storable) {
            D3D11_UNORDERED_ACCESS_VIEW_DESC udesc = {
                .Format = fmt_to_dxgi(buf->params.format),
                .ViewDimension = D3D11_UAV_DIMENSION_BUFFER,
                .Buffer = {
                    .NumElements =
                        PL_ALIGN(buf->params.size, buf->params.format->texel_size)
                            / buf->params.format->texel_size,
                },
            };
            D3D(ID3D11Device_CreateUnorderedAccessView(p->dev,
                (ID3D11Resource *) buf_p->buf, &udesc, &buf_p->texel_uav));
        }
    }


    if (!buf_p->data) {
        // Create the staging buffer regardless of whether params->host_readable
        // is set or not, so that buf_copy can copy to system-memory-backed
        // buffers
        // TODO: Consider sharing a big staging buffer for this, rather than
        // having one staging buffer per buffer
        desc.BindFlags = 0;
        desc.MiscFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        desc.Usage = D3D11_USAGE_STAGING;
        D3D(ID3D11Device_CreateBuffer(p->dev, &desc, NULL, &buf_p->staging));
    }

    pl_d3d11_flush_message_queue(ctx, "After buffer create");

    return buf;

error:
    pl_d3d11_buf_destroy(gpu, buf);
    return NULL;
}

void pl_d3d11_buf_write(pl_gpu gpu, pl_buf buf, size_t offset, const void *data,
                        size_t size)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct pl_buf_d3d11 *buf_p = PL_PRIV(buf);

    if (buf_p->data) {
        memcpy(buf_p->data + offset, data, size);
        buf_p->dirty = true;
    } else {
        ID3D11DeviceContext_UpdateSubresource(p->imm,
            (ID3D11Resource *) buf_p->buf, 0, (&(D3D11_BOX) {
                .left = offset,
                .top = 0,
                .front = 0,
                .right = offset + size,
                .bottom = 1,
                .back = 1,
            }), data, 0, 0);
    }
}

void pl_d3d11_buf_resolve(pl_gpu gpu, pl_buf buf)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct pl_buf_d3d11 *buf_p = PL_PRIV(buf);

    if (!buf_p->data || !buf_p->dirty)
        return;

    ID3D11DeviceContext_UpdateSubresource(p->imm, (ID3D11Resource *) buf_p->buf,
        0, NULL, buf_p->data, 0, 0);
}

bool pl_d3d11_buf_read(pl_gpu gpu, pl_buf buf, size_t offset, void *dest,
                       size_t size)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;
    struct pl_buf_d3d11 *buf_p = PL_PRIV(buf);

    // If there is a system-memory mirror of the buffer contents, use it
    if (buf_p->data) {
        memcpy(dest, buf_p->data + offset, size);
        return true;
    }

    ID3D11DeviceContext_CopyResource(p->imm, (ID3D11Resource *) buf_p->staging,
        (ID3D11Resource *) buf_p->buf);

    D3D11_MAPPED_SUBRESOURCE lock;
    D3D(ID3D11DeviceContext_Map(p->imm, (ID3D11Resource *) buf_p->staging, 0,
                                D3D11_MAP_READ, 0, &lock));

    char *csrc = lock.pData;
    memcpy(dest, csrc + offset, size);

    ID3D11DeviceContext_Unmap(p->imm, (ID3D11Resource *) buf_p->staging, 0);

    pl_d3d11_flush_message_queue(ctx, "After buffer read");

    return true;

error:
    return false;
}

void pl_d3d11_buf_copy(pl_gpu gpu, pl_buf dst, size_t dst_offset, pl_buf src,
                       size_t src_offset, size_t size)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;
    struct pl_buf_d3d11 *src_p = PL_PRIV(src);
    struct pl_buf_d3d11 *dst_p = PL_PRIV(dst);

    // Handle system memory copies in case one or both of the buffers has a
    // system memory mirror
    if (src_p->data && dst_p->data) {
        memcpy(dst_p->data + dst_offset, src_p->data + src_offset, size);
        dst_p->dirty = true;
    } else if (src_p->data) {
        pl_d3d11_buf_write(gpu, dst, dst_offset, src_p->data + src_offset, size);
    } else if (dst_p->data) {
        if (pl_d3d11_buf_read(gpu, src, src_offset, dst_p->data + dst_offset, size)) {
            dst_p->dirty = true;
        } else {
            PL_ERR(gpu, "Failed to read from GPU during buffer copy");
        }
    } else {
        ID3D11DeviceContext_CopySubresourceRegion(p->imm,
            (ID3D11Resource *) dst_p->buf, 0, dst_offset, 0, 0,
            (ID3D11Resource *) src_p->buf, 0, (&(D3D11_BOX) {
                .left = src_offset,
                .top = 0,
                .front = 0,
                .right = src_offset + size,
                .bottom = 1,
                .back = 1,
            }));
    }

    pl_d3d11_flush_message_queue(ctx, "After buffer copy");
}
