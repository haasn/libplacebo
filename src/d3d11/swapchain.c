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
#include <math.h>

#include "gpu.h"
#include "swapchain.h"
#include "utils.h"

struct d3d11_csp_mapping {
    DXGI_COLOR_SPACE_TYPE d3d11_csp;
    DXGI_FORMAT           d3d11_fmt;
    struct pl_color_space out_csp;
};

static struct d3d11_csp_mapping map_pl_csp_to_d3d11(const struct pl_color_space *hint,
                                                    bool use_8bit_sdr)
{
    if (pl_color_space_is_hdr(hint) &&
        hint->transfer != PL_COLOR_TRC_LINEAR)
    {
        struct pl_color_space pl_csp = pl_color_space_hdr10;
        pl_csp.hdr = (struct pl_hdr_metadata) {
            // Whitelist only values that we support signalling metadata for
            .prim     = hint->hdr.prim,
            .min_luma = hint->hdr.min_luma,
            .max_luma = hint->hdr.max_luma,
            .max_cll  = hint->hdr.max_cll,
            .max_fall = hint->hdr.max_fall,
        };

        return (struct d3d11_csp_mapping){
            .d3d11_csp = DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
            .d3d11_fmt = DXGI_FORMAT_R10G10B10A2_UNORM,
            .out_csp   = pl_csp,
        };
#if 0 // TODO: Add support for scRGB
    } else if (pl_color_primaries_is_wide_gamut(hint->primaries) ||
               hint->transfer == PL_COLOR_TRC_LINEAR)
    {
        // scRGB a la VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT,
        // so could be utilized for HDR/wide gamut content as well
        // with content that goes beyond 0.0-1.0.
        return (struct d3d11_csp_mapping){
            .d3d11_csp = DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709,
            .d3d11_fmt = DXGI_FORMAT_R16G16B16A16_FLOAT,
            .out_csp = {
                .primaries = PL_COLOR_PRIM_BT_709,
                .transfer  = PL_COLOR_TRC_LINEAR,
            }
        };
#endif
    }

    return (struct d3d11_csp_mapping){
        .d3d11_csp = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,
        .d3d11_fmt = use_8bit_sdr ? DXGI_FORMAT_R8G8B8A8_UNORM :
                                    DXGI_FORMAT_R10G10B10A2_UNORM,
        .out_csp = pl_color_space_srgb,
    };
}

struct priv {
    struct pl_sw_fns impl;

    struct d3d11_ctx *ctx;
    IDXGISwapChain *swapchain;
    pl_tex backbuffer;

    // Currently requested or applied swap chain configuration.
    // Affected by received colorspace hints.
    struct d3d11_csp_mapping csp_map;

    // Whether a swapchain backbuffer format reconfiguration has been
    // requested by means of an additional resize action.
    bool update_swapchain_format;

    // Whether 10-bit backbuffer format is disabled for SDR content.
    bool disable_10bit_sdr;

    // Fallback to 8-bit RGB was triggered due to lack of compatiblity
    bool fallback_8bit_rgb;
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
    bool format_changed = p->csp_map.d3d11_fmt != desc.BufferDesc.Format;
    if (format_changed) {
        PL_INFO(ctx, "Attempting to reconfigure swap chain format: %s -> %s",
                pl_get_dxgi_format_name(desc.BufferDesc.Format),
                pl_get_dxgi_format_name(p->csp_map.d3d11_fmt));
    }

    if (w != desc.BufferDesc.Width || h != desc.BufferDesc.Height ||
        format_changed)
    {
        if (p->backbuffer) {
            PL_ERR(sw, "Tried resizing the swapchain while a frame was in "
                   "progress! Please submit the current frame first.");
            return false;
        }

        HRESULT hr = IDXGISwapChain_ResizeBuffers(p->swapchain, 0, w, h,
                                                  p->csp_map.d3d11_fmt, desc.Flags);

        if (hr == E_INVALIDARG && p->csp_map.d3d11_fmt != DXGI_FORMAT_R8G8B8A8_UNORM)
        {
            PL_WARN(sw, "Reconfiguring the swapchain failed, re-trying with R8G8B8A8_UNORM fallback.");
            D3D(IDXGISwapChain_ResizeBuffers(p->swapchain, 0, w, h,
                                             DXGI_FORMAT_R8G8B8A8_UNORM, desc.Flags));

            // re-configure the colorspace to 8-bit RGB SDR fallback
            p->csp_map = map_pl_csp_to_d3d11(&pl_color_space_monitor, true);
            p->fallback_8bit_rgb = true;
        }
        else if (FAILED(hr))
        {
            PL_ERR(sw, "Reconfiguring the swapchain failed with error: %s", pl_hresult_to_str(hr));
            return false;
        }
    }

    *width = w;
    *height = h;
    p->update_swapchain_format = false;
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

    if (p->update_swapchain_format) {
        int w = 0, h = 0;
        if (!d3d11_sw_resize(sw, &w, &h))
            return false;
    }

    p->backbuffer = get_backbuffer(sw);
    if (!p->backbuffer)
        return false;

    int bits = 0;
    pl_fmt fmt = p->backbuffer->params.format;
    for (int i = 0; i < fmt->num_components; i++)
        bits = PL_MAX(bits, fmt->component_depth[i]);

    *out_frame = (struct pl_swapchain_frame) {
        .fbo = p->backbuffer,
        .flipped = false,
        .color_repr = {
            .sys = PL_COLOR_SYSTEM_RGB,
            .levels = PL_COLOR_LEVELS_FULL,
            .alpha = PL_ALPHA_UNKNOWN,
            .bits = {
                .sample_depth = bits,
                .color_depth = bits,
            },
        },
        .color_space = p->csp_map.out_csp,
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

static DXGI_HDR_METADATA_HDR10 set_hdr10_metadata(const struct pl_hdr_metadata *hdr)
{
    return (DXGI_HDR_METADATA_HDR10) {
        .RedPrimary   = { roundf(hdr->prim.red.x   * 50000),
                          roundf(hdr->prim.red.y   * 50000) },
        .GreenPrimary = { roundf(hdr->prim.green.x * 50000),
                          roundf(hdr->prim.green.y * 50000) },
        .BluePrimary  = { roundf(hdr->prim.blue.x  * 50000),
                          roundf(hdr->prim.blue.y  * 50000) },
        .WhitePoint   = { roundf(hdr->prim.white.x * 50000),
                          roundf(hdr->prim.white.y * 50000) },
        .MaxMasteringLuminance     = roundf(hdr->max_luma),
        .MinMasteringLuminance     = roundf(hdr->min_luma * 10000),
        .MaxContentLightLevel      = roundf(hdr->max_cll),
        .MaxFrameAverageLightLevel = roundf(hdr->max_fall),
    };
}

static bool set_swapchain_metadata(struct d3d11_ctx *ctx,
                                   IDXGISwapChain3 *swapchain3,
                                   struct d3d11_csp_mapping *csp_map)
{
    IDXGISwapChain4 *swapchain4 = NULL;
    bool ret = false;
    bool is_hdr = pl_color_space_is_hdr(&csp_map->out_csp);
    DXGI_HDR_METADATA_HDR10 hdr10 = is_hdr ?
        set_hdr10_metadata(&csp_map->out_csp.hdr) : (DXGI_HDR_METADATA_HDR10){ 0 };

    D3D(IDXGISwapChain3_SetColorSpace1(swapchain3, csp_map->d3d11_csp));

    // if we succeeded to set the color space, it's good enough,
    // since older versions of Windows 10 will not have swapchain v4 available.
    ret = true;

    if (FAILED(IDXGISwapChain3_QueryInterface(swapchain3, &IID_IDXGISwapChain4,
                                              (void **)&swapchain4)))
    {
        PL_TRACE(ctx, "v4 swap chain interface is not available, skipping HDR10 "
                      "metadata configuration.");
        goto error;
    }

    D3D(IDXGISwapChain4_SetHDRMetaData(swapchain4,
                                       is_hdr ?
                                       DXGI_HDR_METADATA_TYPE_HDR10 :
                                       DXGI_HDR_METADATA_TYPE_NONE,
                                       is_hdr ? sizeof(hdr10) : 0,
                                       is_hdr ? &hdr10 : NULL));

    goto success;

error:
    csp_map->out_csp.hdr = (struct pl_hdr_metadata) { 0 };
success:
    SAFE_RELEASE(swapchain4);
    return ret;
}

static bool d3d11_format_supported(struct d3d11_ctx *ctx, DXGI_FORMAT fmt)
{
    UINT sup = 0;
    UINT wanted_sup =
        D3D11_FORMAT_SUPPORT_TEXTURE2D | D3D11_FORMAT_SUPPORT_DISPLAY |
        D3D11_FORMAT_SUPPORT_SHADER_SAMPLE | D3D11_FORMAT_SUPPORT_RENDER_TARGET |
        D3D11_FORMAT_SUPPORT_BLENDABLE;

    D3D(ID3D11Device_CheckFormatSupport(ctx->dev, fmt, &sup));

    return (sup & wanted_sup) == wanted_sup;

error:
    return false;
}

static bool d3d11_csp_supported(struct d3d11_ctx *ctx,
                                IDXGISwapChain3 *swapchain3,
                                DXGI_COLOR_SPACE_TYPE color_space)
{
    UINT csp_support_flags = 0;

    D3D(IDXGISwapChain3_CheckColorSpaceSupport(swapchain3,
                                               color_space,
                                               &csp_support_flags));

    return (csp_support_flags & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT);

error:
    return false;
}

static void update_swapchain_color_config(pl_swapchain sw,
                                          const struct pl_color_space *csp,
                                          bool is_internal)
{
    struct priv *p = PL_PRIV(sw);
    struct d3d11_ctx *ctx = p->ctx;
    IDXGISwapChain3 *swapchain3 = NULL;
    struct d3d11_csp_mapping old_map = p->csp_map;

    // ignore config changes in fallback mode
    if (p->fallback_8bit_rgb)
        goto cleanup;

    HRESULT hr = IDXGISwapChain_QueryInterface(p->swapchain, &IID_IDXGISwapChain3,
                                               (void **)&swapchain3);
    if (FAILED(hr)) {
        PL_TRACE(ctx, "v3 swap chain interface is not available, skipping "
                      "color space configuration.");
        swapchain3 = NULL;
    }

    // Lack of swap chain v3 means we cannot control swap chain color space;
    // Only effective formats are the 8 and 10 bit RGB ones.
    struct d3d11_csp_mapping csp_map =
        map_pl_csp_to_d3d11(swapchain3 ? csp : &pl_color_space_unknown,
                            p->disable_10bit_sdr);

    if (p->csp_map.d3d11_fmt == csp_map.d3d11_fmt &&
        p->csp_map.d3d11_csp == csp_map.d3d11_csp &&
        pl_color_space_equal(&p->csp_map.out_csp, &csp_map.out_csp))
        goto cleanup;

    PL_INFO(ctx, "%s swap chain configuration%s: format: %s, color space: %s.",
            is_internal ? "Initial" : "New",
            is_internal ? "" : " received from hint",
            pl_get_dxgi_format_name(csp_map.d3d11_fmt),
            pl_get_dxgi_csp_name(csp_map.d3d11_csp));

    bool fmt_supported = d3d11_format_supported(ctx, csp_map.d3d11_fmt);
    bool csp_supported = swapchain3 ?
        d3d11_csp_supported(ctx, swapchain3, csp_map.d3d11_csp) : true;
    if (!fmt_supported || !csp_supported) {
        PL_ERR(ctx, "New swap chain configuration was deemed not supported: "
                    "format: %s, color space: %s. Failling back to 8bit RGB.",
               fmt_supported ? "supported" : "unsupported",
               csp_supported ? "supported" : "unsupported");
        // fall back to 8bit sRGB if requested configuration is not supported
        csp_map = map_pl_csp_to_d3d11(&pl_color_space_monitor, true);
    }

    p->csp_map = csp_map;
    p->update_swapchain_format = true;

    if (!swapchain3)
        goto cleanup;

    if (!set_swapchain_metadata(ctx, swapchain3, &p->csp_map)) {
        // format succeeded, but color space configuration failed
        p->csp_map = old_map;
        p->csp_map.d3d11_fmt = csp_map.d3d11_fmt;
    }

    pl_d3d11_flush_message_queue(ctx, "After colorspace hint");

cleanup:
    SAFE_RELEASE(swapchain3);
}

static void d3d11_sw_colorspace_hint(pl_swapchain sw,
                                     const struct pl_color_space *csp)
{
    update_swapchain_color_config(sw, csp, false);
}

IDXGISwapChain *pl_d3d11_swapchain_unwrap(pl_swapchain sw)
{
    struct priv *p = PL_PRIV(sw);
    IDXGISwapChain_AddRef(p->swapchain);
    return p->swapchain;
}

static const struct pl_sw_fns d3d11_swapchain = {
    .destroy         = d3d11_sw_destroy,
    .latency         = d3d11_sw_latency,
    .resize          = d3d11_sw_resize,
    .colorspace_hint = d3d11_sw_colorspace_hint,
    .start_frame     = d3d11_sw_start_frame,
    .submit_frame    = d3d11_sw_submit_frame,
    .swap_buffers    = d3d11_sw_swap_buffers,
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
    const struct pl_d3d11_swapchain_params *params, DXGI_FORMAT format)
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
                                      height, format, &swapchain);
        } else {
            // Fall back to DXGI 1.1 (Windows 7)
            hr = create_swapchain_1_1(ctx, factory, params, width, height,
                                      format, &swapchain);
        }
        if (SUCCEEDED(hr))
            break;

        pl_d3d11_after_error(ctx, hr);
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

    struct pl_swapchain_t *sw = pl_zalloc_obj(NULL, sw, struct priv);
    struct priv *p = PL_PRIV(sw);
    *sw = (struct pl_swapchain_t) {
        .log = gpu->log,
        .gpu = gpu,
    };
    *p = (struct priv) {
        .impl = d3d11_swapchain,
        .ctx = ctx,
        // default to standard 8 or 10 bit RGB, unset pl_color_space
        .csp_map = {
            .d3d11_fmt = params->disable_10bit_sdr ?
                DXGI_FORMAT_R8G8B8A8_UNORM :
                (d3d11_format_supported(ctx, DXGI_FORMAT_R10G10B10A2_UNORM) ?
                 DXGI_FORMAT_R10G10B10A2_UNORM : DXGI_FORMAT_R8G8B8A8_UNORM),
        },
        .disable_10bit_sdr = params->disable_10bit_sdr,
    };

    if (params->swapchain) {
        p->swapchain = params->swapchain;
        IDXGISwapChain_AddRef(params->swapchain);
    } else {
        p->swapchain = create_swapchain(ctx, params, p->csp_map.d3d11_fmt);
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

    p->csp_map.d3d11_fmt = scd.BufferDesc.Format;

    update_swapchain_color_config(sw, &pl_color_space_srgb, true);

    success = true;
error:
    if (!success) {
        PL_FATAL(gpu, "Failed to create Direct3D 11 swapchain");
        d3d11_sw_destroy(sw);
        sw = NULL;
    }
    return sw;
}
