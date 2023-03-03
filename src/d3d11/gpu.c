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

#include <initguid.h>
#include <windows.h>
#include <versionhelpers.h>

#include "common.h"
#include "gpu.h"
#include "formats.h"
#include "glsl/spirv.h"

#define DXGI_ADAPTER_FLAG3_SUPPORT_MONITORED_FENCES (0x8)

struct timer_query {
    ID3D11Query *ts_start;
    ID3D11Query *ts_end;
    ID3D11Query *disjoint;
};

struct pl_timer_t {
    // Ring buffer of timer queries to use
    int current;
    int pending;
    struct timer_query queries[16];
};

void pl_d3d11_timer_start(pl_gpu gpu, pl_timer timer)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;

    if (!timer)
        return;
    struct timer_query *query = &timer->queries[timer->current];

    // Create the query objects lazilly
    if (!query->ts_start) {
        D3D(ID3D11Device_CreateQuery(p->dev,
            &(D3D11_QUERY_DESC) { D3D11_QUERY_TIMESTAMP }, &query->ts_start));
        D3D(ID3D11Device_CreateQuery(p->dev,
            &(D3D11_QUERY_DESC) { D3D11_QUERY_TIMESTAMP }, &query->ts_end));

        // Measuring duration in D3D11 requires three queries: start and end
        // timestamp queries, and a disjoint query containing a flag which says
        // whether the timestamps are usable or if a discontinuity occurred
        // between them, like a change in power state or clock speed. The
        // disjoint query also contains the timer frequency, so the timestamps
        // are useless without it.
        D3D(ID3D11Device_CreateQuery(p->dev,
            &(D3D11_QUERY_DESC) { D3D11_QUERY_TIMESTAMP_DISJOINT }, &query->disjoint));
    }

    // Query the start timestamp
    ID3D11DeviceContext_Begin(p->imm, (ID3D11Asynchronous *) query->disjoint);
    ID3D11DeviceContext_End(p->imm, (ID3D11Asynchronous *) query->ts_start);
    return;

error:
    SAFE_RELEASE(query->ts_start);
    SAFE_RELEASE(query->ts_end);
    SAFE_RELEASE(query->disjoint);
}

void pl_d3d11_timer_end(pl_gpu gpu, pl_timer timer)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);

    if (!timer)
        return;
    struct timer_query *query = &timer->queries[timer->current];

    // Even if timer_start and timer_end are called in-order, timer_start might
    // have failed to create the timer objects
    if (!query->ts_start)
        return;

    // Query the end timestamp
    ID3D11DeviceContext_End(p->imm, (ID3D11Asynchronous *) query->ts_end);
    ID3D11DeviceContext_End(p->imm, (ID3D11Asynchronous *) query->disjoint);

    // Advance to the next set of queries, for the next call to timer_start
    timer->current++;
    if (timer->current >= PL_ARRAY_SIZE(timer->queries))
        timer->current = 0; // Wrap around

    // Increment the number of pending queries, unless the ring buffer is full,
    // in which case, timer->current now points to the oldest one, which will be
    // dropped and reused
    if (timer->pending < PL_ARRAY_SIZE(timer->queries))
        timer->pending++;
}

static uint64_t timestamp_to_ns(uint64_t timestamp, uint64_t freq)
{
    static const uint64_t ns_per_s = 1000000000llu;
    return timestamp / freq * ns_per_s + timestamp % freq * ns_per_s / freq;
}

static uint64_t d3d11_timer_query(pl_gpu gpu, pl_timer timer)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;
    HRESULT hr;

    for (; timer->pending > 0; timer->pending--) {
        int index = timer->current - timer->pending;
        if (index < 0)
            index += PL_ARRAY_SIZE(timer->queries);
        struct timer_query *query = &timer->queries[index];

        UINT64 start, end;
        D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj;

        // Fetch the results of each query, or on S_FALSE, return 0 to indicate
        // the queries are still pending
        D3D(hr = ID3D11DeviceContext_GetData(p->imm,
            (ID3D11Asynchronous *) query->disjoint, &dj, sizeof(dj),
            D3D11_ASYNC_GETDATA_DONOTFLUSH));
        if (hr == S_FALSE)
            return 0;
        D3D(hr = ID3D11DeviceContext_GetData(p->imm,
            (ID3D11Asynchronous *) query->ts_end, &end, sizeof(end),
            D3D11_ASYNC_GETDATA_DONOTFLUSH));
        if (hr == S_FALSE)
            return 0;
        D3D(hr = ID3D11DeviceContext_GetData(p->imm,
            (ID3D11Asynchronous *) query->ts_start, &start, sizeof(start),
            D3D11_ASYNC_GETDATA_DONOTFLUSH));
        if (hr == S_FALSE)
            return 0;

        // There was a discontinuity during the queries, so a timestamp can't be
        // produced. Skip it and try the next one.
        if (dj.Disjoint || !dj.Frequency)
            continue;

        // We got a result. Return it to the caller.
        timer->pending--;
        pl_d3d11_flush_message_queue(ctx, "After timer query");

        uint64_t ns = timestamp_to_ns(end - start, dj.Frequency);
        return PL_MAX(ns, 1);

    error:
        // There was an error fetching the timer result, so skip it and try the
        // next one
        continue;
    }

    // No more unprocessed results
    return 0;
}

static void d3d11_timer_destroy(pl_gpu gpu, pl_timer timer)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;

    for (int i = 0; i < PL_ARRAY_SIZE(timer->queries); i++) {
        SAFE_RELEASE(timer->queries[i].ts_start);
        SAFE_RELEASE(timer->queries[i].ts_end);
        SAFE_RELEASE(timer->queries[i].disjoint);
    }

    pl_d3d11_flush_message_queue(ctx, "After timer destroy");

    pl_free(timer);
}

static pl_timer d3d11_timer_create(pl_gpu gpu)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    if (!p->has_timestamp_queries)
        return NULL;

    struct pl_timer_t *timer = pl_alloc_ptr(NULL, timer);
    *timer = (struct pl_timer_t) {0};
    return timer;
}

static int d3d11_desc_namespace(pl_gpu gpu, enum pl_desc_type type)
{
    // Vulkan-style binding, where all descriptors are in the same namespace, is
    // required to use SPIRV-Cross' HLSL resource mapping API, which targets
    // resources by binding number
    return 0;
}

static void d3d11_gpu_flush(pl_gpu gpu)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;
    ID3D11DeviceContext_Flush(p->imm);

    pl_d3d11_flush_message_queue(ctx, "After gpu flush");
}

static void d3d11_gpu_finish(pl_gpu gpu)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;
    HRESULT hr;

    if (p->finish_fence) {
        p->finish_value++;
        D3D(ID3D11Fence_SetEventOnCompletion(p->finish_fence, p->finish_value,
                                             p->finish_event));
        ID3D11DeviceContext4_Signal(p->imm4, p->finish_fence, p->finish_value);
        ID3D11DeviceContext_Flush(p->imm);
        WaitForSingleObject(p->finish_event, INFINITE);
    } else {
        ID3D11DeviceContext_End(p->imm, (ID3D11Asynchronous *) p->finish_query);

        // D3D11 doesn't have blocking queries, but it does have blocking
        // readback. As a performance hack to try to avoid polling, do a dummy
        // copy/readback between two buffers. Hopefully this will block until
        // all prior commands are finished. If it does, the first GetData call
        // will return a result and we won't have to poll.
        pl_buf_copy(gpu, p->finish_buf_dst, 0, p->finish_buf_src, 0, sizeof(uint32_t));
        pl_buf_read(gpu, p->finish_buf_dst, 0, &(uint32_t) {0}, sizeof(uint32_t));

        // Poll the event query until it completes
        for (;;) {
            BOOL idle;
            D3D(hr = ID3D11DeviceContext_GetData(p->imm,
                (ID3D11Asynchronous *) p->finish_query, &idle, sizeof(idle), 0));
            if (hr == S_OK && idle)
                break;
            Sleep(1);
        }
    }

    pl_d3d11_flush_message_queue(ctx, "After gpu finish");

error:
    return;
}

static bool d3d11_gpu_is_failed(pl_gpu gpu)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;

    if (ctx->is_failed)
        return true;

    // GetDeviceRemovedReason returns S_OK if the device isn't removed
    HRESULT hr = ID3D11Device_GetDeviceRemovedReason(p->dev);
    if (FAILED(hr)) {
        ctx->is_failed = true;
        pl_d3d11_after_error(ctx, hr);
    }

    return ctx->is_failed;
}

static void d3d11_gpu_destroy(pl_gpu gpu)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);

    pl_buf_destroy(gpu, &p->finish_buf_src);
    pl_buf_destroy(gpu, &p->finish_buf_dst);

    // Release everything except the immediate context
    SAFE_RELEASE(p->dev);
    SAFE_RELEASE(p->dev1);
    SAFE_RELEASE(p->dev5);
    SAFE_RELEASE(p->imm1);
    SAFE_RELEASE(p->imm4);
    SAFE_RELEASE(p->vbuf.buf);
    SAFE_RELEASE(p->ibuf.buf);
    SAFE_RELEASE(p->rstate);
    SAFE_RELEASE(p->dsstate);
    for (int i = 0; i < PL_TEX_SAMPLE_MODE_COUNT; i++) {
        for (int j = 0; j < PL_TEX_ADDRESS_MODE_COUNT; j++) {
            SAFE_RELEASE(p->samplers[i][j]);
        }
    }
    SAFE_RELEASE(p->finish_fence);
    if (p->finish_event)
        CloseHandle(p->finish_event);
    SAFE_RELEASE(p->finish_query);

    // Destroy the immediate context synchronously so referenced objects don't
    // show up in the leak check
    if (p->imm) {
        ID3D11DeviceContext_ClearState(p->imm);
        ID3D11DeviceContext_Flush(p->imm);
        SAFE_RELEASE(p->imm);
    }

    pl_free((void *) gpu);
}

pl_d3d11 pl_d3d11_get(pl_gpu gpu)
{
    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    if (impl->destroy == d3d11_gpu_destroy) {
        struct pl_gpu_d3d11 *p = (struct pl_gpu_d3d11 *) impl;
        return p->ctx->d3d11;
    }

    return NULL;
}

static bool load_d3d_compiler(pl_gpu gpu)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    HMODULE d3dcompiler = NULL;

    static const struct {
        const wchar_t *name;
        bool inbox;
    } compiler_dlls[] = {
        // Try the inbox D3DCompiler first (Windows 8.1 and up)
        { .name = L"d3dcompiler_47.dll", .inbox = true },
        // Check for a packaged version of d3dcompiler_47.dll
        { .name = L"d3dcompiler_47.dll" },
        // Try d3dcompiler_46.dll from the Windows 8 SDK
        { .name = L"d3dcompiler_46.dll" },
        // Try d3dcompiler_43.dll from the June 2010 DirectX SDK
        { .name = L"d3dcompiler_43.dll" },
    };

    for (int i = 0; i < PL_ARRAY_SIZE(compiler_dlls); i++) {
        if (compiler_dlls[i].inbox) {
            if (!IsWindows8Point1OrGreater())
                continue;
            d3dcompiler = LoadLibraryExW(compiler_dlls[i].name, NULL,
                                         LOAD_LIBRARY_SEARCH_SYSTEM32);
        } else {
            d3dcompiler = LoadLibraryW(compiler_dlls[i].name);
        }
        if (!d3dcompiler)
            continue;

        p->D3DCompile = (void *) GetProcAddress(d3dcompiler, "D3DCompile");
        if (!p->D3DCompile)
            return false;
        p->d3d_compiler_ver = pl_get_dll_version(compiler_dlls[i].name);

        return true;
    }

    return false;
}

static struct pl_gpu_fns pl_fns_d3d11 = {
    .tex_create             = pl_d3d11_tex_create,
    .tex_destroy            = pl_d3d11_tex_destroy,
    .tex_invalidate         = pl_d3d11_tex_invalidate,
    .tex_clear_ex           = pl_d3d11_tex_clear_ex,
    .tex_blit               = pl_d3d11_tex_blit,
    .tex_upload             = pl_d3d11_tex_upload,
    .tex_download           = pl_d3d11_tex_download,
    .buf_create             = pl_d3d11_buf_create,
    .buf_destroy            = pl_d3d11_buf_destroy,
    .buf_write              = pl_d3d11_buf_write,
    .buf_read               = pl_d3d11_buf_read,
    .buf_copy               = pl_d3d11_buf_copy,
    .desc_namespace         = d3d11_desc_namespace,
    .pass_create            = pl_d3d11_pass_create,
    .pass_destroy           = pl_d3d11_pass_destroy,
    .pass_run               = pl_d3d11_pass_run,
    .timer_create           = d3d11_timer_create,
    .timer_destroy          = d3d11_timer_destroy,
    .timer_query            = d3d11_timer_query,
    .gpu_flush              = d3d11_gpu_flush,
    .gpu_finish             = d3d11_gpu_finish,
    .gpu_is_failed          = d3d11_gpu_is_failed,
    .destroy                = d3d11_gpu_destroy,
};

pl_gpu pl_gpu_create_d3d11(struct d3d11_ctx *ctx)
{
    pl_assert(ctx->dev);
    IDXGIDevice1 *dxgi_dev = NULL;
    IDXGIAdapter1 *adapter = NULL;
    IDXGIAdapter4 *adapter4 = NULL;
    bool success = false;
    HRESULT hr;

    struct pl_gpu_t *gpu = pl_zalloc_obj(NULL, gpu, struct pl_gpu_d3d11);
    gpu->log = ctx->log;

    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    uint32_t spirv_ver = PL_MIN(SPV_VERSION, PL_MAX_SPIRV_VER);
    *p = (struct pl_gpu_d3d11) {
        .ctx = ctx,
        .impl = pl_fns_d3d11,
        .dev = ctx->dev,
        .spirv = spirv_compiler_create(ctx->log, &(const struct pl_spirv_version) {
                                          .env_version = pl_spirv_version_to_vulkan(spirv_ver),
                                          .spv_version = spirv_ver,
                                      }),
        .vbuf.bind_flags = D3D11_BIND_VERTEX_BUFFER,
        .ibuf.bind_flags = D3D11_BIND_INDEX_BUFFER,
    };
    if (!p->spirv)
        goto error;

    ID3D11Device_AddRef(p->dev);
    ID3D11Device_GetImmediateContext(p->dev, &p->imm);

    // Check D3D11.1 interfaces
    hr = ID3D11Device_QueryInterface(p->dev, &IID_ID3D11Device1,
                                     (void **) &p->dev1);
    if (SUCCEEDED(hr)) {
        p->minor = 1;
        ID3D11Device1_GetImmediateContext1(p->dev1, &p->imm1);
    }

    // Check D3D11.4 interfaces
    hr = ID3D11Device_QueryInterface(p->dev, &IID_ID3D11Device5,
                                     (void **) &p->dev5);
    if (SUCCEEDED(hr)) {
        // There is no GetImmediateContext4 method
        hr = ID3D11DeviceContext_QueryInterface(p->imm, &IID_ID3D11DeviceContext4,
                                                (void **) &p->imm4);
        if (SUCCEEDED(hr))
            p->minor = 4;
    }

    PL_INFO(gpu, "Using Direct3D 11.%d runtime", p->minor);

    D3D(ID3D11Device_QueryInterface(p->dev, &IID_IDXGIDevice1, (void **) &dxgi_dev));
    D3D(IDXGIDevice1_GetParent(dxgi_dev, &IID_IDXGIAdapter1, (void **) &adapter));

    DXGI_ADAPTER_DESC1 adapter_desc = {0};
    IDXGIAdapter1_GetDesc1(adapter, &adapter_desc);

    // No resource can be larger than max_res_size in bytes
    unsigned int max_res_size = PL_CLAMP(
        D3D11_REQ_RESOURCE_SIZE_IN_MEGABYTES_EXPRESSION_B_TERM * adapter_desc.DedicatedVideoMemory,
        D3D11_REQ_RESOURCE_SIZE_IN_MEGABYTES_EXPRESSION_A_TERM * 1024u * 1024u,
        D3D11_REQ_RESOURCE_SIZE_IN_MEGABYTES_EXPRESSION_C_TERM * 1024u * 1024u);

    gpu->glsl = (struct pl_glsl_version) {
        .version = 450,
        .vulkan = true,
    };

    gpu->limits = (struct pl_gpu_limits) {
        .max_buf_size = max_res_size,
        .max_ssbo_size = max_res_size,
        .max_vbo_size = max_res_size,
        .align_vertex_stride = 1,

        // Make up some values
        .align_tex_xfer_offset = 32,
        .align_tex_xfer_pitch = 1,
        .fragment_queues = 1,
    };

    p->fl = ID3D11Device_GetFeatureLevel(p->dev);

    // If we're not using FL9_x, we can use the same suballocated buffer as a
    // vertex buffer and index buffer
    if (p->fl >= D3D_FEATURE_LEVEL_10_0)
        p->vbuf.bind_flags |= D3D11_BIND_INDEX_BUFFER;

    if (p->fl >= D3D_FEATURE_LEVEL_10_0) {
        gpu->limits.max_ubo_size = D3D11_REQ_CONSTANT_BUFFER_ELEMENT_COUNT * CBUF_ELEM;
    } else {
        // 10level9 restriction:
        // https://docs.microsoft.com/en-us/windows/win32/direct3d11/d3d11-graphics-reference-10level9-context
        gpu->limits.max_ubo_size = 255 * CBUF_ELEM;
    }

    if (p->fl >= D3D_FEATURE_LEVEL_11_0) {
        gpu->limits.max_tex_1d_dim = D3D11_REQ_TEXTURE1D_U_DIMENSION;
        gpu->limits.max_tex_2d_dim = D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION;
        gpu->limits.max_tex_3d_dim = D3D11_REQ_TEXTURE3D_U_V_OR_W_DIMENSION;
    } else if (p->fl >= D3D_FEATURE_LEVEL_10_0) {
        gpu->limits.max_tex_1d_dim = D3D10_REQ_TEXTURE1D_U_DIMENSION;
        gpu->limits.max_tex_2d_dim = D3D10_REQ_TEXTURE2D_U_OR_V_DIMENSION;
        gpu->limits.max_tex_3d_dim = D3D10_REQ_TEXTURE3D_U_V_OR_W_DIMENSION;
    } else if (p->fl >= D3D_FEATURE_LEVEL_9_3) {
        gpu->limits.max_tex_2d_dim = D3D_FL9_3_REQ_TEXTURE2D_U_OR_V_DIMENSION;
        // Same limit as FL9_1
        gpu->limits.max_tex_3d_dim = D3D_FL9_1_REQ_TEXTURE3D_U_V_OR_W_DIMENSION;
    } else {
        gpu->limits.max_tex_2d_dim = D3D_FL9_1_REQ_TEXTURE2D_U_OR_V_DIMENSION;
        gpu->limits.max_tex_3d_dim = D3D_FL9_1_REQ_TEXTURE3D_U_V_OR_W_DIMENSION;
    }

    if (p->fl >= D3D_FEATURE_LEVEL_10_0) {
        gpu->limits.max_buffer_texels =
            1 << D3D11_REQ_BUFFER_RESOURCE_TEXEL_COUNT_2_TO_EXP;
    }

    if (p->fl >= D3D_FEATURE_LEVEL_11_0) {
        gpu->glsl.compute = true;
        gpu->limits.compute_queues = 1;
        // Set `gpu->limits.blittable_1d_3d`, since `pl_tex_blit_compute`, which
        // is used to emulate blits on 11_0 and up, supports 1D and 3D textures
        gpu->limits.blittable_1d_3d = true;

        gpu->glsl.max_shmem_size = D3D11_CS_TGSM_REGISTER_COUNT * sizeof(float);
        gpu->glsl.max_group_threads = D3D11_CS_THREAD_GROUP_MAX_THREADS_PER_GROUP;
        gpu->glsl.max_group_size[0] = D3D11_CS_THREAD_GROUP_MAX_X;
        gpu->glsl.max_group_size[1] = D3D11_CS_THREAD_GROUP_MAX_Y;
        gpu->glsl.max_group_size[2] = D3D11_CS_THREAD_GROUP_MAX_Z;
        gpu->limits.max_dispatch[0] = gpu->limits.max_dispatch[1] =
            gpu->limits.max_dispatch[2] =
            D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;
    }

    if (p->fl >= D3D_FEATURE_LEVEL_11_0) {
        // The offset limits are defined by HLSL:
        // https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/gather4-po--sm5---asm-
        gpu->glsl.min_gather_offset = -32;
        gpu->glsl.max_gather_offset = 31;
    } else if (p->fl >= D3D_FEATURE_LEVEL_10_1) {
        // SM4.1 has no gather4_po, so the offset must be specified by an
        // immediate with a range of [-8, 7]
        // https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/gather4--sm4-1---asm-
        // https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sample--sm4---asm-#address-offset
        gpu->glsl.min_gather_offset = -8;
        gpu->glsl.max_gather_offset = 7;
    }

    if (p->fl >= D3D_FEATURE_LEVEL_10_0) {
        p->max_srvs = D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT;
    } else {
        // 10level9 restriction:
        // https://docs.microsoft.com/en-us/windows/win32/direct3d11/d3d11-graphics-reference-10level9-context
        p->max_srvs = 8;
    }

    if (p->fl >= D3D_FEATURE_LEVEL_11_1) {
        p->max_uavs = D3D11_1_UAV_SLOT_COUNT;
    } else {
        p->max_uavs = D3D11_PS_CS_UAV_REGISTER_COUNT;
    }

    if (!load_d3d_compiler(gpu)) {
        PL_FATAL(gpu, "Could not find D3DCompiler DLL");
        goto error;
    }
    PL_INFO(gpu, "D3DCompiler version: %u.%u.%u.%u",
            p->d3d_compiler_ver.major, p->d3d_compiler_ver.minor,
            p->d3d_compiler_ver.build, p->d3d_compiler_ver.revision);

    // Detect support for timestamp queries. Some FL9_x devices don't support them.
    hr = ID3D11Device_CreateQuery(p->dev,
        &(D3D11_QUERY_DESC) { D3D11_QUERY_TIMESTAMP }, NULL);
    p->has_timestamp_queries = SUCCEEDED(hr);

    pl_d3d11_setup_formats(gpu);

    // The rasterizer state never changes, so create it here
    D3D11_RASTERIZER_DESC rdesc = {
        .FillMode = D3D11_FILL_SOLID,
        .CullMode = D3D11_CULL_NONE,
        .FrontCounterClockwise = FALSE,
        .DepthClipEnable = TRUE, // Required for 10level9
        .ScissorEnable = TRUE,
    };
    D3D(ID3D11Device_CreateRasterizerState(p->dev, &rdesc, &p->rstate));

    // The depth stencil state never changes either, and we only set it to turn
    // depth testing off so the debug layer doesn't complain about an unbound
    // depth buffer
    D3D11_DEPTH_STENCIL_DESC dsdesc = {
        .DepthEnable = FALSE,
        .DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL,
        .DepthFunc = D3D11_COMPARISON_LESS,
        .StencilReadMask = D3D11_DEFAULT_STENCIL_READ_MASK,
        .StencilWriteMask = D3D11_DEFAULT_STENCIL_WRITE_MASK,
        .FrontFace = {
            .StencilFailOp = D3D11_STENCIL_OP_KEEP,
            .StencilDepthFailOp = D3D11_STENCIL_OP_KEEP,
            .StencilPassOp = D3D11_STENCIL_OP_KEEP,
            .StencilFunc = D3D11_COMPARISON_ALWAYS,
        },
        .BackFace = {
            .StencilFailOp = D3D11_STENCIL_OP_KEEP,
            .StencilDepthFailOp = D3D11_STENCIL_OP_KEEP,
            .StencilPassOp = D3D11_STENCIL_OP_KEEP,
            .StencilFunc = D3D11_COMPARISON_ALWAYS,
        },
    };
    D3D(ID3D11Device_CreateDepthStencilState(p->dev, &dsdesc, &p->dsstate));

    // Initialize the samplers
    for (int sample_mode = 0; sample_mode < PL_TEX_SAMPLE_MODE_COUNT; sample_mode++) {
        for (int address_mode = 0; address_mode < PL_TEX_ADDRESS_MODE_COUNT; address_mode++) {
            static const D3D11_TEXTURE_ADDRESS_MODE d3d_address_mode[] = {
                [PL_TEX_ADDRESS_CLAMP] = D3D11_TEXTURE_ADDRESS_CLAMP,
                [PL_TEX_ADDRESS_REPEAT] = D3D11_TEXTURE_ADDRESS_WRAP,
                [PL_TEX_ADDRESS_MIRROR] = D3D11_TEXTURE_ADDRESS_MIRROR,
            };
            static const D3D11_FILTER d3d_filter[] = {
                [PL_TEX_SAMPLE_NEAREST] = D3D11_FILTER_MIN_MAG_MIP_POINT,
                [PL_TEX_SAMPLE_LINEAR] = D3D11_FILTER_MIN_MAG_MIP_LINEAR,
            };

            D3D11_SAMPLER_DESC sdesc = {
                .AddressU = d3d_address_mode[address_mode],
                .AddressV = d3d_address_mode[address_mode],
                .AddressW = d3d_address_mode[address_mode],
                .ComparisonFunc = D3D11_COMPARISON_NEVER,
                .MinLOD = 0,
                .MaxLOD = D3D11_FLOAT32_MAX,
                .MaxAnisotropy = 1,
                .Filter = d3d_filter[sample_mode],
            };
            D3D(ID3D11Device_CreateSamplerState(p->dev, &sdesc,
                &p->samplers[sample_mode][address_mode]));
        }
    }

    hr = IDXGIAdapter1_QueryInterface(adapter, &IID_IDXGIAdapter4,
                                      (void **) &adapter4);
    if (SUCCEEDED(hr)) {
        DXGI_ADAPTER_DESC3 adapter_desc3 = {0};
        IDXGIAdapter4_GetDesc3(adapter4, &adapter_desc3);

        p->has_monitored_fences =
            adapter_desc3.Flags & DXGI_ADAPTER_FLAG3_SUPPORT_MONITORED_FENCES;
    }

    // Try to create a D3D11.4 fence object to wait on in pl_gpu_finish()
    if (p->dev5 && p->has_monitored_fences) {
        hr = ID3D11Device5_CreateFence(p->dev5, 0, D3D11_FENCE_FLAG_NONE,
                                       &IID_ID3D11Fence,
                                       (void **) p->finish_fence);
        if (SUCCEEDED(hr)) {
            p->finish_event = CreateEventW(NULL, FALSE, FALSE, NULL);
            if (!p->finish_event) {
                PL_ERR(gpu, "Failed to create finish() event");
                goto error;
            }
        }
    }

    // If fences are not available, we will have to poll a event query instead
    if (!p->finish_fence) {
        // Buffers for dummy copy/readback (see d3d11_gpu_finish())
        p->finish_buf_src = pl_buf_create(gpu, pl_buf_params(
            .size = sizeof(uint32_t),
            .drawable = true, // Make these vertex buffers for 10level9
            .initial_data = &(uint32_t) {0x11223344},
        ));
        p->finish_buf_dst = pl_buf_create(gpu, pl_buf_params(
            .size = sizeof(uint32_t),
            .host_readable = true,
            .drawable = true,
        ));

        D3D(ID3D11Device_CreateQuery(p->dev,
            &(D3D11_QUERY_DESC) { D3D11_QUERY_EVENT }, &p->finish_query));
    }

    pl_d3d11_flush_message_queue(ctx, "After gpu create");

    success = true;
error:
    SAFE_RELEASE(dxgi_dev);
    SAFE_RELEASE(adapter);
    SAFE_RELEASE(adapter4);
    if (success) {
        return pl_gpu_finalize(gpu);
    } else {
        d3d11_gpu_destroy(gpu);
        return NULL;
    }
}
