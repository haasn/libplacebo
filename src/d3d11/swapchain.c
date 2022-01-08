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

#include <windows.h>
#include <versionhelpers.h>

#include "gpu.h"
#include "swapchain.h"

struct priv {
    struct d3d11_ctx *ctx;
    IDXGISwapChain *swapchain;
    pl_tex backbuffer;
};

static void d3d11_sw_destroy(pl_swapchain sw)
{
    struct priv *p = PL_PRIV(sw);

    pl_tex_destroy(sw->gpu, &p->backbuffer);
    SAFE_RELEASE(p->swapchain);
    pl_free((void *) sw);
}

static int d3d11_sw_latency(pl_swapchain sw)
{
    struct priv *p = PL_PRIV(sw);
    struct d3d11_ctx *ctx = p->ctx;

    UINT max_latency;
    IDXGIDevice1_GetMaximumFrameLatency(ctx->dxgi_dev, &max_latency);
    return max_latency;
}

static pl_tex get_backbuffer(pl_swapchain sw)
{
    struct priv *p = PL_PRIV(sw);
    struct d3d11_ctx *ctx = p->ctx;
    ID3D11Texture2D *backbuffer = NULL;
    pl_tex tex = NULL;

    D3D(IDXGISwapChain_GetBuffer(p->swapchain, 0, &IID_ID3D11Texture2D,
                                 (void **) &backbuffer));

    tex = pl_d3d11_wrap(sw->gpu, pl_d3d11_wrap_params(
        .tex = (ID3D11Resource *) backbuffer,
    ));

error:
    SAFE_RELEASE(backbuffer);
    return tex;
}

static bool d3d11_sw_resize(pl_swapchain sw, int *width, int *height)
{
    struct priv *p = PL_PRIV(sw);
    struct d3d11_ctx *ctx = p->ctx;

    DXGI_SWAP_CHAIN_DESC desc = {0};
    IDXGISwapChain_GetDesc(p->swapchain, &desc);
    int w = PL_DEF(*width, desc.BufferDesc.Width);
    int h = PL_DEF(*height, desc.BufferDesc.Height);

    if (w != desc.BufferDesc.Width || h != desc.BufferDesc.Height) {
        if (p->backbuffer) {
            PL_ERR(sw, "Tried resizing the swapchain while a frame was in "
                   "progress! Please submit the current frame first.");
            return false;
        }

        D3D(IDXGISwapChain_ResizeBuffers(p->swapchain, 0, w, h,
                                         DXGI_FORMAT_UNKNOWN, desc.Flags));
    }

    *width = w;
    *height = h;
    return true;

error:
    return false;
}

static bool d3d11_sw_start_frame(pl_swapchain sw,
                                 struct pl_swapchain_frame *out_frame)
{
    struct priv *p = PL_PRIV(sw);
    struct d3d11_ctx *ctx = p->ctx;

    if (ctx->is_failed)
        return false;
    if (p->backbuffer) {
        PL_ERR(sw, "Attempted calling `pl_swapchain_start_frame` while a frame "
               "was already in progress! Call `pl_swapchain_submit_frame` first.");
        return false;
    }

    p->backbuffer = get_backbuffer(sw);
    if (!p->backbuffer)
        return false;

    *out_frame = (struct pl_swapchain_frame) {
        .fbo = p->backbuffer,
        .flipped = false,
        .color_repr = {
            .sys = PL_COLOR_SYSTEM_RGB,
            .levels = PL_COLOR_LEVELS_FULL,
            .alpha = PL_ALPHA_UNKNOWN,
            .bits = {
                .sample_depth = 8,
                .color_depth = 8,
            },
        },
        .color_space = pl_color_space_monitor,
    };

    return true;
}

static bool d3d11_sw_submit_frame(pl_swapchain sw)
{
    struct priv *p = PL_PRIV(sw);
    struct d3d11_ctx *ctx = p->ctx;

    // Release the backbuffer. We shouldn't hold onto it unnecessarily, because
    // it prevents external code from resizing the swapchain, which we'd
    // otherwise support just fine.
    pl_tex_destroy(sw->gpu, &p->backbuffer);

    return !ctx->is_failed;
}

static void d3d11_sw_swap_buffers(pl_swapchain sw)
{
    struct priv *p = PL_PRIV(sw);
    struct d3d11_ctx *ctx = p->ctx;

    // Present can fail with a device removed error
    D3D(IDXGISwapChain_Present(p->swapchain, 1, 0));

error:
    return;
}

IDXGISwapChain *pl_d3d11_swapchain_unwrap(pl_swapchain sw)
{
    struct priv *p = PL_PRIV(sw);
    IDXGISwapChain_AddRef(p->swapchain);
    return p->swapchain;
}

static struct pl_sw_fns d3d11_swapchain = {
    .destroy      = d3d11_sw_destroy,
    .latency      = d3d11_sw_latency,
    .resize       = d3d11_sw_resize,
    .start_frame  = d3d11_sw_start_frame,
    .submit_frame = d3d11_sw_submit_frame,
    .swap_buffers = d3d11_sw_swap_buffers,
};

static HRESULT create_swapchain_1_2(struct d3d11_ctx *ctx,
    IDXGIFactory2 *factory, const struct pl_d3d11_swapchain_params *params,
    bool flip, UINT width, UINT height, DXGI_FORMAT format,
    IDXGISwapChain **swapchain_out)
{
    IDXGISwapChain *swapchain = NULL;
    IDXGISwapChain1 *swapchain1 = NULL;
    HRESULT hr;

    DXGI_SWAP_CHAIN_DESC1 desc = {
        .Width = width,
        .Height = height,
        .Format = format,
        .SampleDesc.Count = 1,
        .BufferUsage = DXGI_USAGE_SHADER_INPUT | DXGI_USAGE_RENDER_TARGET_OUTPUT,
        .Flags = params->flags,
    };

    if (ID3D11Device_GetFeatureLevel(ctx->dev) >= D3D_FEATURE_LEVEL_11_0)
        desc.BufferUsage |= DXGI_USAGE_UNORDERED_ACCESS;

    if (flip) {
        UINT max_latency;
        IDXGIDevice1_GetMaximumFrameLatency(ctx->dxgi_dev, &max_latency);

        // Make sure we have at least enough buffers to allow `max_latency`
        // frames in-flight at once, plus one frame for the frontbuffer
        desc.BufferCount = max_latency + 1;

        if (IsWindows10OrGreater()) {
            desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        } else {
            desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
        }

        desc.BufferCount = PL_MIN(desc.BufferCount, DXGI_MAX_SWAP_CHAIN_BUFFERS);
    } else {
        desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
        desc.BufferCount = 1;
    }

    if (params->window) {
        hr = IDXGIFactory2_CreateSwapChainForHwnd(factory, (IUnknown *) ctx->dev,
            params->window, &desc, NULL, NULL, &swapchain1);
    } else if (params->core_window) {
        hr = IDXGIFactory2_CreateSwapChainForCoreWindow(factory,
            (IUnknown *) ctx->dev, params->core_window, &desc, NULL, &swapchain1);
    } else {
        hr = IDXGIFactory2_CreateSwapChainForComposition(factory,
            (IUnknown *) ctx->dev, &desc, NULL, &swapchain1);
    }
    if (FAILED(hr))
        goto done;
    hr = IDXGISwapChain1_QueryInterface(swapchain1, &IID_IDXGISwapChain,
                                        (void **) &swapchain);
    if (FAILED(hr))
        goto done;

    *swapchain_out = swapchain;
    swapchain = NULL;

done:
    SAFE_RELEASE(swapchain1);
    SAFE_RELEASE(swapchain);
    return hr;
}

static HRESULT create_swapchain_1_1(struct d3d11_ctx *ctx,
    IDXGIFactory1 *factory, const struct pl_d3d11_swapchain_params *params,
    UINT width, UINT height, DXGI_FORMAT format, IDXGISwapChain **swapchain_out)
{
    DXGI_SWAP_CHAIN_DESC desc = {
        .BufferDesc = {
            .Width = width,
            .Height = height,
            .Format = format,
        },
        .SampleDesc.Count = 1,
        .BufferUsage = DXGI_USAGE_SHADER_INPUT | DXGI_USAGE_RENDER_TARGET_OUTPUT,
        .BufferCount = 1,
        .OutputWindow = params->window,
        .Windowed = TRUE,
        .SwapEffect = DXGI_SWAP_EFFECT_DISCARD,
        .Flags = params->flags,
    };

    return IDXGIFactory1_CreateSwapChain(factory, (IUnknown *) ctx->dev, &desc,
                                         swapchain_out);
}

static IDXGISwapChain *create_swapchain(struct d3d11_ctx *ctx,
    const struct pl_d3d11_swapchain_params *params)
{
    IDXGIDevice1 *dxgi_dev = NULL;
    IDXGIAdapter1 *adapter = NULL;
    IDXGIFactory1 *factory = NULL;
    IDXGIFactory2 *factory2 = NULL;
    IDXGISwapChain *swapchain = NULL;
    bool success = false;
    HRESULT hr;

    D3D(ID3D11Device_QueryInterface(ctx->dev, &IID_IDXGIDevice1,
                                    (void **) &dxgi_dev));
    D3D(IDXGIDevice1_GetParent(dxgi_dev, &IID_IDXGIAdapter1, (void **) &adapter));
    D3D(IDXGIAdapter1_GetParent(adapter, &IID_IDXGIFactory1, (void **) &factory));

    hr = IDXGIFactory1_QueryInterface(factory, &IID_IDXGIFactory2,
                                      (void **) &factory2);
    if (FAILED(hr))
        factory2 = NULL;

    bool flip = factory2 && !params->blit;
    UINT width = PL_DEF(params->width, 1);
    UINT height = PL_DEF(params->height, 1);

    // If both width and height are unset, the default size is the window size
    if (params->window && params->width == 0 && params->height == 0) {
        RECT rc;
        if (GetClientRect(params->window, &rc)) {
            width = PL_DEF(rc.right - rc.left, 1);
            height = PL_DEF(rc.bottom - rc.top, 1);
        }
    }

    // Return here to retry creating the swapchain
    do {
        if (factory2) {
            // Create a DXGI 1.2+ (Windows 8+) swap chain if possible
            hr = create_swapchain_1_2(ctx, factory2, params, flip, width,
                                      height, DXGI_FORMAT_R8G8B8A8_UNORM,
                                      &swapchain);
        } else {
            // Fall back to DXGI 1.1 (Windows 7)
            hr = create_swapchain_1_1(ctx, factory, params, width, height,
                                      DXGI_FORMAT_R8G8B8A8_UNORM, &swapchain);
        }
        if (SUCCEEDED(hr))
            break;

        if (flip) {
            PL_DEBUG(ctx, "Failed to create flip-model swapchain, trying bitblt");
            flip = false;
            continue;
        }

        PL_FATAL(ctx, "Failed to create swapchain: %s", pl_hresult_to_str(hr));
        goto error;
    } while (true);

    // Prevent DXGI from making changes to the window, otherwise it will hook
    // the Alt+Enter keystroke and make it trigger an ugly transition to
    // legacy exclusive fullscreen mode.
    IDXGIFactory_MakeWindowAssociation(factory, params->window,
        DXGI_MWA_NO_WINDOW_CHANGES | DXGI_MWA_NO_ALT_ENTER |
        DXGI_MWA_NO_PRINT_SCREEN);

    success = true;
error:
    if (!success)
        SAFE_RELEASE(swapchain);
    SAFE_RELEASE(factory2);
    SAFE_RELEASE(factory);
    SAFE_RELEASE(adapter);
    SAFE_RELEASE(dxgi_dev);
    return swapchain;
}

pl_swapchain pl_d3d11_create_swapchain(pl_d3d11 d3d11,
    const struct pl_d3d11_swapchain_params *params)
{
    struct d3d11_ctx *ctx = PL_PRIV(d3d11);
    pl_gpu gpu = d3d11->gpu;
    bool success = false;

    struct pl_swapchain *sw = pl_zalloc_obj(NULL, sw, struct priv);
    struct priv *p = PL_PRIV(sw);
    *sw = (struct pl_swapchain) {
        .impl = &d3d11_swapchain,
        .log = gpu->log,
        .gpu = gpu,
    };
    *p = (struct priv) {
        .ctx = ctx,
    };

    if (params->swapchain) {
        p->swapchain = params->swapchain;
        IDXGISwapChain_AddRef(params->swapchain);
    } else {
        p->swapchain = create_swapchain(ctx, params);
        if (!p->swapchain)
            goto error;
    }

    DXGI_SWAP_CHAIN_DESC scd = {0};
    IDXGISwapChain_GetDesc(p->swapchain, &scd);
    if (scd.SwapEffect == DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL ||
        scd.SwapEffect == DXGI_SWAP_EFFECT_FLIP_DISCARD) {
        PL_INFO(gpu, "Using flip-model presentation");
    } else {
        PL_INFO(gpu, "Using bitblt-model presentation");
    }

    success = true;
error:
    if (!success) {
        PL_FATAL(gpu, "Failed to create Direct3D 11 swapchain");
        d3d11_sw_destroy(sw);
        sw = NULL;
    }
    return sw;
}
