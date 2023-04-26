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

#include "gpu.h"

// Windows 8 enum value, not present in mingw-w64 v7
#define DXGI_ADAPTER_FLAG_SOFTWARE (2)

const struct pl_d3d11_params pl_d3d11_default_params = { PL_D3D11_DEFAULTS };

static INIT_ONCE d3d11_once = INIT_ONCE_STATIC_INIT;
static PFN_D3D11_CREATE_DEVICE pD3D11CreateDevice = NULL;
static PFN_CREATE_DXGI_FACTORY pCreateDXGIFactory1 = NULL;
static void d3d11_load(void)
{
    BOOL bPending = FALSE;
    InitOnceBeginInitialize(&d3d11_once, 0, &bPending, NULL);

    if (bPending)
    {
        HMODULE d3d11 = LoadLibraryW(L"d3d11.dll");
        if (d3d11) {
            pD3D11CreateDevice = (void *)
                GetProcAddress(d3d11, "D3D11CreateDevice");
        }

        HMODULE dxgi = LoadLibraryW(L"dxgi.dll");
        if (dxgi) {
            pCreateDXGIFactory1 = (void *)
                GetProcAddress(dxgi, "CreateDXGIFactory1");
        }
    }

    InitOnceComplete(&d3d11_once, 0, NULL);
}

// Get a const array of D3D_FEATURE_LEVELs from max_fl to min_fl (inclusive)
static int get_feature_levels(int max_fl, int min_fl,
                              const D3D_FEATURE_LEVEL **out)
{
    static const D3D_FEATURE_LEVEL levels[] = {
        D3D_FEATURE_LEVEL_12_1,
        D3D_FEATURE_LEVEL_12_0,
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
        D3D_FEATURE_LEVEL_9_3,
        D3D_FEATURE_LEVEL_9_2,
        D3D_FEATURE_LEVEL_9_1,
    };
    static const int levels_len = PL_ARRAY_SIZE(levels);

    int start = 0;
    for (; start < levels_len; start++) {
        if (levels[start] <= max_fl)
            break;
    }
    int len = 0;
    for (; start + len < levels_len; len++) {
        if (levels[start + len] < min_fl)
            break;
    }
    *out = &levels[start];
    return len;
}

static bool is_null_luid(LUID luid)
{
    return luid.LowPart == 0 && luid.HighPart == 0;
}

static IDXGIAdapter *get_adapter(pl_d3d11 d3d11, LUID adapter_luid)
{
    struct d3d11_ctx *ctx = PL_PRIV(d3d11);
    IDXGIFactory1 *factory = NULL;
    IDXGIAdapter1 *adapter1 = NULL;
    IDXGIAdapter *adapter = NULL;
    HRESULT hr;

    if (!pCreateDXGIFactory1) {
        PL_FATAL(ctx, "Failed to load dxgi.dll");
        goto error;
    }
    pCreateDXGIFactory1(&IID_IDXGIFactory1, (void **) &factory);

    for (int i = 0;; i++) {
        hr = IDXGIFactory1_EnumAdapters1(factory, i, &adapter1);
        if (hr == DXGI_ERROR_NOT_FOUND)
            break;
        if (FAILED(hr)) {
            PL_FATAL(ctx, "Failed to enumerate adapters");
            goto error;
        }

        DXGI_ADAPTER_DESC1 desc;
        D3D(IDXGIAdapter1_GetDesc1(adapter1, &desc));
        if (desc.AdapterLuid.LowPart == adapter_luid.LowPart &&
            desc.AdapterLuid.HighPart == adapter_luid.HighPart)
        {
            break;
        }

        SAFE_RELEASE(adapter1);
    }
    if (!adapter1) {
        PL_FATAL(ctx, "Adapter with LUID %08lx%08lx not found",
                 adapter_luid.HighPart, adapter_luid.LowPart);
        goto error;
    }

    D3D(IDXGIAdapter1_QueryInterface(adapter1, &IID_IDXGIAdapter,
                                     (void **) &adapter));

error:
    SAFE_RELEASE(factory);
    SAFE_RELEASE(adapter1);
    return adapter;
}

static bool has_sdk_layers(void)
{
    // This will fail if the SDK layers aren't installed
    return SUCCEEDED(pD3D11CreateDevice(NULL, D3D_DRIVER_TYPE_NULL, NULL,
        D3D11_CREATE_DEVICE_DEBUG, NULL, 0, D3D11_SDK_VERSION, NULL, NULL,
        NULL));
}

static ID3D11Device *create_device(struct pl_d3d11_t *d3d11,
                                   const struct pl_d3d11_params *params)
{
    struct d3d11_ctx *ctx = PL_PRIV(d3d11);
    bool debug = params->debug;
    bool warp = params->force_software;
    int max_fl = params->max_feature_level;
    int min_fl = params->min_feature_level;
    ID3D11Device *dev = NULL;
    IDXGIDevice1 *dxgi_dev = NULL;
    IDXGIAdapter *adapter = NULL;
    bool release_adapter = false;
    HRESULT hr;

    d3d11_load();

    if (!pD3D11CreateDevice) {
        PL_FATAL(ctx, "Failed to load d3d11.dll");
        goto error;
    }

    if (params->adapter) {
        adapter = params->adapter;
    } else if (!is_null_luid(params->adapter_luid)) {
        adapter = get_adapter(d3d11, params->adapter_luid);
        release_adapter = true;
    }

    if (debug && !has_sdk_layers()) {
        PL_INFO(ctx, "Debug layer not available, removing debug flag");
        debug = false;
    }

    // Return here to retry creating the device
    do {
        // Use these default feature levels if they are not set
        max_fl = PL_DEF(max_fl, D3D_FEATURE_LEVEL_12_1);
        min_fl = PL_DEF(min_fl, D3D_FEATURE_LEVEL_9_1);

        // Get a list of feature levels from min_fl to max_fl
        const D3D_FEATURE_LEVEL *levels;
        int levels_len = get_feature_levels(max_fl, min_fl, &levels);
        if (!levels_len) {
            PL_FATAL(ctx, "No suitable Direct3D feature level found");
            goto error;
        }

        D3D_DRIVER_TYPE type = D3D_DRIVER_TYPE_UNKNOWN;
        if (!adapter) {
            if (warp) {
                type = D3D_DRIVER_TYPE_WARP;
            } else {
                type = D3D_DRIVER_TYPE_HARDWARE;
            }
        }

        UINT flags = params->flags;
        if (debug)
            flags |= D3D11_CREATE_DEVICE_DEBUG;

        hr = pD3D11CreateDevice(adapter, type, NULL, flags, levels, levels_len,
                                D3D11_SDK_VERSION, &dev, NULL, NULL);
        if (SUCCEEDED(hr))
            break;

        // Trying to create a D3D_FEATURE_LEVEL_12_0 device on Windows 8.1 or
        // below will not succeed. Try an 11_1 device.
        if (hr == E_INVALIDARG && max_fl >= D3D_FEATURE_LEVEL_12_0 &&
                                  min_fl <= D3D_FEATURE_LEVEL_11_1) {
            PL_DEBUG(ctx, "Failed to create 12_0+ device, trying 11_1");
            max_fl = D3D_FEATURE_LEVEL_11_1;
            continue;
        }

        // Trying to create a D3D_FEATURE_LEVEL_11_1 device on Windows 7
        // without the platform update will not succeed. Try an 11_0 device.
        if (hr == E_INVALIDARG && max_fl >= D3D_FEATURE_LEVEL_11_1 &&
                                  min_fl <= D3D_FEATURE_LEVEL_11_0) {
            PL_DEBUG(ctx, "Failed to create 11_1+ device, trying 11_0");
            max_fl = D3D_FEATURE_LEVEL_11_0;
            continue;
        }

        // Retry with WARP if allowed
        if (!adapter && !warp && params->allow_software) {
            PL_DEBUG(ctx, "Failed to create hardware device, trying WARP: %s",
                     pl_hresult_to_str(hr));
            warp = true;
            max_fl = params->max_feature_level;
            min_fl = params->min_feature_level;
            continue;
        }

        PL_FATAL(ctx, "Failed to create Direct3D 11 device: %s",
                 pl_hresult_to_str(hr));
        goto error;
    } while (true);

    if (params->max_frame_latency) {
        D3D(ID3D11Device_QueryInterface(dev, &IID_IDXGIDevice1,
                                        (void **) &dxgi_dev));
        IDXGIDevice1_SetMaximumFrameLatency(dxgi_dev, params->max_frame_latency);
    }

    d3d11->software = warp;

error:
    if (release_adapter)
        SAFE_RELEASE(adapter);
    SAFE_RELEASE(dxgi_dev);
    return dev;
}

static void init_debug_layer(struct d3d11_ctx *ctx)
{
    D3D(ID3D11Device_QueryInterface(ctx->dev, &IID_ID3D11Debug,
                                    (void **) &ctx->debug));
    D3D(ID3D11Device_QueryInterface(ctx->dev, &IID_ID3D11InfoQueue,
                                    (void **) &ctx->iqueue));

    // Filter some annoying messages
    D3D11_MESSAGE_ID deny_ids[] = {
        // This false-positive error occurs every time we Draw() with a shader
        // that samples from a texture format that only supports point sampling.
        // Since we already use CheckFormatSupport to know which formats can be
        // linearly sampled from, we shouldn't ever bind a non-point sampler to
        // a format that doesn't support it.
        D3D11_MESSAGE_ID_DEVICE_DRAW_RESOURCE_FORMAT_SAMPLE_UNSUPPORTED,
    };
    D3D11_INFO_QUEUE_FILTER filter = {
        .DenyList = {
            .NumIDs = PL_ARRAY_SIZE(deny_ids),
            .pIDList = deny_ids,
        },
    };
    ID3D11InfoQueue_PushStorageFilter(ctx->iqueue, &filter);

    ID3D11InfoQueue_SetMessageCountLimit(ctx->iqueue, -1);

error:
    return;
}

void pl_d3d11_destroy(pl_d3d11 *ptr)
{
    pl_d3d11 d3d11 = *ptr;
    if (!d3d11)
        return;
    struct d3d11_ctx *ctx = PL_PRIV(d3d11);

    pl_gpu_destroy(d3d11->gpu);

    SAFE_RELEASE(ctx->dev);
    SAFE_RELEASE(ctx->dxgi_dev);

    if (ctx->debug) {
        // Report any leaked objects
        pl_d3d11_flush_message_queue(ctx, "After destroy");
        ID3D11Debug_ReportLiveDeviceObjects(ctx->debug, D3D11_RLDO_DETAIL);
        pl_d3d11_flush_message_queue(ctx, "After leak check");
        ID3D11Debug_ReportLiveDeviceObjects(ctx->debug, D3D11_RLDO_SUMMARY);
        pl_d3d11_flush_message_queue(ctx, "After leak summary");
    }

    SAFE_RELEASE(ctx->debug);
    SAFE_RELEASE(ctx->iqueue);

    pl_free_ptr((void **) ptr);
}

pl_d3d11 pl_d3d11_create(pl_log log, const struct pl_d3d11_params *params)
{
    params = PL_DEF(params, &pl_d3d11_default_params);
    IDXGIAdapter1 *adapter = NULL;
    IDXGIAdapter2 *adapter2 = NULL;
    bool success = false;
    HRESULT hr;

    struct pl_d3d11_t *d3d11 = pl_zalloc_obj(NULL, d3d11, struct d3d11_ctx);
    struct d3d11_ctx *ctx = PL_PRIV(d3d11);
    ctx->log = log;
    ctx->d3d11 = d3d11;

    if (params->device) {
        d3d11->device = params->device;
        ID3D11Device_AddRef(d3d11->device);
    } else {
        d3d11->device = create_device(d3d11, params);
        if (!d3d11->device)
            goto error;
    }
    ctx->dev = d3d11->device;

    D3D(ID3D11Device_QueryInterface(d3d11->device, &IID_IDXGIDevice1,
                                    (void **) &ctx->dxgi_dev));
    D3D(IDXGIDevice1_GetParent(ctx->dxgi_dev, &IID_IDXGIAdapter1,
                               (void **) &adapter));

    hr = IDXGIAdapter1_QueryInterface(adapter, &IID_IDXGIAdapter2,
                                      (void **) &adapter2);
    if (FAILED(hr))
        adapter2 = NULL;

    if (adapter2) {
        PL_INFO(ctx, "Using DXGI 1.2+");
    } else {
        PL_INFO(ctx, "Using DXGI 1.1");
    }

    D3D_FEATURE_LEVEL fl = ID3D11Device_GetFeatureLevel(d3d11->device);
    PL_INFO(ctx, "Using Direct3D 11 feature level %u_%u",
            ((unsigned) fl) >> 12, (((unsigned) fl) >> 8) & 0xf);

    char *dev_name = NULL;
    UINT vendor_id, device_id, revision, subsys_id;
    LUID adapter_luid;
    UINT flags;

    if (adapter2) {
        // DXGI 1.2 IDXGIAdapter2::GetDesc2 is preferred over the DXGI 1.1
        // version because it reports the real adapter information when using
        // feature level 9 hardware
        DXGI_ADAPTER_DESC2 desc;
        D3D(IDXGIAdapter2_GetDesc2(adapter2, &desc));

        dev_name = pl_to_utf8(NULL, desc.Description);
        vendor_id = desc.VendorId;
        device_id = desc.DeviceId;
        revision = desc.Revision;
        subsys_id = desc.SubSysId;
        adapter_luid = desc.AdapterLuid;
        flags = desc.Flags;
    } else {
        DXGI_ADAPTER_DESC1 desc;
        D3D(IDXGIAdapter1_GetDesc1(adapter, &desc));

        dev_name = pl_to_utf8(NULL, desc.Description);
        vendor_id = desc.VendorId;
        device_id = desc.DeviceId;
        revision = desc.Revision;
        subsys_id = desc.SubSysId;
        adapter_luid = desc.AdapterLuid;
        flags = desc.Flags;
    }

    PL_INFO(ctx, "Direct3D 11 device properties:");
    PL_INFO(ctx, "    Device Name: %s", dev_name);
    PL_INFO(ctx, "    Device ID: %04x:%04x (rev %02x)",
            vendor_id, device_id, revision);
    PL_INFO(ctx, "    Subsystem ID: %04x:%04x",
            LOWORD(subsys_id), HIWORD(subsys_id));
    PL_INFO(ctx, "    LUID: %08lx%08lx",
            adapter_luid.HighPart, adapter_luid.LowPart);
    pl_free(dev_name);

    LARGE_INTEGER version;
    hr = IDXGIAdapter1_CheckInterfaceSupport(adapter, &IID_IDXGIDevice, &version);
    if (SUCCEEDED(hr)) {
        PL_INFO(ctx, "    Driver version: %u.%u.%u.%u",
                HIWORD(version.HighPart), LOWORD(version.HighPart),
                HIWORD(version.LowPart), LOWORD(version.LowPart));
    }

    // Note: DXGI_ADAPTER_FLAG_SOFTWARE doesn't exist before Windows 8, but we
    // also set d3d11->software in create_device if we pick WARP ourselves
    if (flags & DXGI_ADAPTER_FLAG_SOFTWARE)
        d3d11->software = true;

    // If the primary display adapter is a software adapter, the
    // DXGI_ADAPTER_FLAG_SOFTWARE flag won't be set, but the device IDs should
    // still match the Microsoft Basic Render Driver
    if (vendor_id == 0x1414 && device_id == 0x8c)
        d3d11->software = true;

    if (d3d11->software) {
        bool external_adapter = params->device || params->adapter ||
                                !is_null_luid(params->adapter_luid);

        // The allow_software flag only applies if the API user didn't manually
        // specify an adapter or a device
        if (!params->allow_software && !external_adapter) {
            // If we got this far with allow_software set, the primary adapter
            // must be a software adapter
            PL_ERR(ctx, "Primary adapter is a software adapter");
            goto error;
        }

        // If a software adapter was manually specified, don't show a warning
        enum pl_log_level level = PL_LOG_WARN;
        if (external_adapter || params->force_software)
            level = PL_LOG_INFO;

        PL_MSG(ctx, level, "Using a software adapter");
    }

    // Init debug layer
    if (ID3D11Device_GetCreationFlags(d3d11->device) & D3D11_CREATE_DEVICE_DEBUG)
        init_debug_layer(ctx);

    d3d11->gpu = pl_gpu_create_d3d11(ctx);
    if (!d3d11->gpu)
        goto error;

    success = true;
error:
    if (!success) {
        PL_FATAL(ctx, "Failed initializing Direct3D 11 device");
        pl_d3d11_destroy((pl_d3d11 *) &d3d11);
    }
    SAFE_RELEASE(adapter);
    SAFE_RELEASE(adapter2);
    return d3d11;
}
