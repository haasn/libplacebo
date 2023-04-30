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

#include <string.h>

#include "utils.h"

// D3D11.3 message IDs, not present in mingw-w64 v9
#define D3D11_MESSAGE_ID_CREATE_FENCE (0x300209)
#define D3D11_MESSAGE_ID_DESTROY_FENCE (0x30020b)

static enum pl_log_level log_level_override(unsigned int id)
{
    switch (id) {
        // These warnings can happen when a pl_timer is used too often before a
        // blocking pl_swapchain_swap_buffers() or pl_gpu_finish(), overflowing
        // its internal ring buffer and causing older query objects to be reused
        // before their results are read. This is expected behavior, so reduce
        // the log level to PL_LOG_TRACE to prevent log spam.
    case D3D11_MESSAGE_ID_QUERY_BEGIN_ABANDONING_PREVIOUS_RESULTS:
    case D3D11_MESSAGE_ID_QUERY_END_ABANDONING_PREVIOUS_RESULTS:
        return PL_LOG_TRACE;

        // D3D11 writes log messages every time an object is created or
        // destroyed. That results in a lot of log spam, so force PL_LOG_TRACE.
#define OBJ_LIFETIME_MESSAGES(obj)          \
    case D3D11_MESSAGE_ID_CREATE_ ## obj:   \
    case D3D11_MESSAGE_ID_DESTROY_ ## obj

    OBJ_LIFETIME_MESSAGES(CONTEXT):
    OBJ_LIFETIME_MESSAGES(BUFFER):
    OBJ_LIFETIME_MESSAGES(TEXTURE1D):
    OBJ_LIFETIME_MESSAGES(TEXTURE2D):
    OBJ_LIFETIME_MESSAGES(TEXTURE3D):
    OBJ_LIFETIME_MESSAGES(SHADERRESOURCEVIEW):
    OBJ_LIFETIME_MESSAGES(RENDERTARGETVIEW):
    OBJ_LIFETIME_MESSAGES(DEPTHSTENCILVIEW):
    OBJ_LIFETIME_MESSAGES(VERTEXSHADER):
    OBJ_LIFETIME_MESSAGES(HULLSHADER):
    OBJ_LIFETIME_MESSAGES(DOMAINSHADER):
    OBJ_LIFETIME_MESSAGES(GEOMETRYSHADER):
    OBJ_LIFETIME_MESSAGES(PIXELSHADER):
    OBJ_LIFETIME_MESSAGES(INPUTLAYOUT):
    OBJ_LIFETIME_MESSAGES(SAMPLER):
    OBJ_LIFETIME_MESSAGES(BLENDSTATE):
    OBJ_LIFETIME_MESSAGES(DEPTHSTENCILSTATE):
    OBJ_LIFETIME_MESSAGES(RASTERIZERSTATE):
    OBJ_LIFETIME_MESSAGES(QUERY):
    OBJ_LIFETIME_MESSAGES(PREDICATE):
    OBJ_LIFETIME_MESSAGES(COUNTER):
    OBJ_LIFETIME_MESSAGES(COMMANDLIST):
    OBJ_LIFETIME_MESSAGES(CLASSINSTANCE):
    OBJ_LIFETIME_MESSAGES(CLASSLINKAGE):
    OBJ_LIFETIME_MESSAGES(COMPUTESHADER):
    OBJ_LIFETIME_MESSAGES(UNORDEREDACCESSVIEW):
    OBJ_LIFETIME_MESSAGES(VIDEODECODER):
    OBJ_LIFETIME_MESSAGES(VIDEOPROCESSORENUM):
    OBJ_LIFETIME_MESSAGES(VIDEOPROCESSOR):
    OBJ_LIFETIME_MESSAGES(DECODEROUTPUTVIEW):
    OBJ_LIFETIME_MESSAGES(PROCESSORINPUTVIEW):
    OBJ_LIFETIME_MESSAGES(PROCESSOROUTPUTVIEW):
    OBJ_LIFETIME_MESSAGES(DEVICECONTEXTSTATE):
    OBJ_LIFETIME_MESSAGES(FENCE):
        return PL_LOG_TRACE;

#undef OBJ_LIFETIME_MESSAGES

        // Don't force the log level of any other messages. It will be mapped
        // from the D3D severity code instead.
    default:
        return PL_LOG_NONE;
    }
}

void pl_d3d11_flush_message_queue(struct d3d11_ctx *ctx, const char *header)
{
    if (!ctx->iqueue)
        return;

    static const enum pl_log_level severity_map[] = {
        [DXGI_INFO_QUEUE_MESSAGE_SEVERITY_CORRUPTION] = PL_LOG_FATAL,
        [DXGI_INFO_QUEUE_MESSAGE_SEVERITY_ERROR]      = PL_LOG_ERR,
        [DXGI_INFO_QUEUE_MESSAGE_SEVERITY_WARNING]    = PL_LOG_WARN,
        [DXGI_INFO_QUEUE_MESSAGE_SEVERITY_INFO]       = PL_LOG_DEBUG,
        [DXGI_INFO_QUEUE_MESSAGE_SEVERITY_MESSAGE]    = PL_LOG_DEBUG,
    };

    enum pl_log_level header_printed = PL_LOG_NONE;

    // After the storage limit is reached and ID3D11InfoQueue::ClearStoredMessages
    // is called message counter seems to be initialized to -1 which is quite big
    // number if we read it as uint64_t. Any subsequent call to the
    // ID3D11InfoQueue::GetNumStoredMessages will be off by one.
    // Use ID3D11InfoQueue_GetNumStoredMessagesAllowedByRetrievalFilter without
    // any filter set, which seem to be unaffected by this bug and return correct
    // number of messages.
    // IDXGIInfoQueue seems to be unaffected, but keep the same way of retrival
    uint64_t messages = IDXGIInfoQueue_GetNumStoredMessagesAllowedByRetrievalFilters(ctx->iqueue, DXGI_DEBUG_ALL);

    // Just to be on the safe side, check also for the mentioned -1 value...
    if (!messages || messages == UINT64_C(-1))
        return;

    uint64_t discarded =
        IDXGIInfoQueue_GetNumMessagesDiscardedByMessageCountLimit(ctx->iqueue, DXGI_DEBUG_ALL);
    if (discarded > ctx->last_discarded) {
        PL_WARN(ctx, "%s:", header);
        header_printed = PL_LOG_WARN;

        // Notify number of messages skipped due to the message count limit
        PL_WARN(ctx, "    (skipped %"PRIu64" debug layer messages)",
                discarded - ctx->last_discarded);
        ctx->last_discarded = discarded;
    }

    // Copy debug layer messages to libplacebo's log output
    for (uint64_t i = 0; i < messages; i++) {
        SIZE_T len;
        if (FAILED(IDXGIInfoQueue_GetMessage(ctx->iqueue, DXGI_DEBUG_ALL, i, NULL, &len)))
            goto error;

        pl_grow((void *) ctx->d3d11, &ctx->dxgi_msg, len);
        DXGI_INFO_QUEUE_MESSAGE *d3dmsg = ctx->dxgi_msg;

        if (FAILED(IDXGIInfoQueue_GetMessage(ctx->iqueue, DXGI_DEBUG_ALL, i, d3dmsg, &len)))
            goto error;

        enum pl_log_level level = PL_LOG_NONE;
        if (IsEqualGUID(&d3dmsg->Producer, &DXGI_DEBUG_D3D11))
            level = log_level_override(d3dmsg->ID);
        if (level == PL_LOG_NONE)
            level = severity_map[d3dmsg->Severity];

        if (pl_msg_test(ctx->log, level)) {
            // If the header hasn't been printed, or it was printed for a lower
            // log level than the current message, print it (again)
            if (header_printed == PL_LOG_NONE || header_printed > level) {
                PL_MSG(ctx, level, "%s:", header);
                pl_log_stack_trace(ctx->log, level);
                header_printed = level;
            }

            PL_MSG(ctx, level, " %d: %.*s", (int) d3dmsg->ID,
                   (int) d3dmsg->DescriptionByteLength, d3dmsg->pDescription);
        }

        if (d3dmsg->Severity <= DXGI_INFO_QUEUE_MESSAGE_SEVERITY_ERROR)
            pl_debug_abort();
    }

error:
    IDXGIInfoQueue_ClearStoredMessages(ctx->iqueue, DXGI_DEBUG_ALL);
}

HRESULT pl_d3d11_check_device_removed(struct d3d11_ctx *ctx, HRESULT hr)
{
    // This can be called before we have a device
    if (!ctx->dev)
        return hr;

    switch (hr) {
    case DXGI_ERROR_DEVICE_HUNG:
    case DXGI_ERROR_DEVICE_RESET:
    case DXGI_ERROR_DRIVER_INTERNAL_ERROR:
        ctx->is_failed = true;
        break;
    case D3DDDIERR_DEVICEREMOVED:
    case DXGI_ERROR_DEVICE_REMOVED:
        hr = ID3D11Device_GetDeviceRemovedReason(ctx->dev);
        ctx->is_failed = true;
        break;
    }
    if (ctx->is_failed)
        PL_ERR(ctx, "Device lost!");
    return hr;
}

HRESULT pl_d3d11_after_error(struct d3d11_ctx *ctx, HRESULT hr)
{
    hr = pl_d3d11_check_device_removed(ctx, hr);
    pl_d3d11_flush_message_queue(ctx, "After error");
    return hr;
}

struct dll_version pl_get_dll_version(const wchar_t *name)
{
    void *data = NULL;
    struct dll_version ret = {0};

    DWORD size = GetFileVersionInfoSizeW(name, &(DWORD) {0});
    if (!size)
        goto error;
    data = pl_alloc(NULL, size);

    if (!GetFileVersionInfoW(name, 0, size, data))
        goto error;

    VS_FIXEDFILEINFO *ffi;
    UINT ffi_len;
    if (!VerQueryValueW(data, L"\\", (void**)&ffi, &ffi_len))
        goto error;
    if (ffi_len < sizeof(*ffi))
        goto error;

    ret = (struct dll_version) {
        .major = HIWORD(ffi->dwFileVersionMS),
        .minor = LOWORD(ffi->dwFileVersionMS),
        .build = HIWORD(ffi->dwFileVersionLS),
        .revision = LOWORD(ffi->dwFileVersionLS),
    };

error:
    pl_free(data);
    return ret;
}

wchar_t *pl_from_utf8(void *ctx, const char *str)
{
    int count = MultiByteToWideChar(CP_UTF8, 0, str, -1, NULL, 0);
    pl_assert(count > 0);
    wchar_t *ret = pl_calloc_ptr(ctx, count, ret);
    MultiByteToWideChar(CP_UTF8, 0, str, -1, ret, count);
    return ret;
}

char *pl_to_utf8(void *ctx, const wchar_t *str)
{
    int count = WideCharToMultiByte(CP_UTF8, 0, str, -1, NULL, 0, NULL, NULL);
    pl_assert(count > 0);
    char *ret = pl_calloc_ptr(ctx, count, ret);
    WideCharToMultiByte(CP_UTF8, 0, str, -1, ret, count, NULL, NULL);
    return ret;
}

static const char *hresult_str(HRESULT hr)
{
    switch (hr) {
#define CASE(name) case name: return #name
    CASE(S_OK);
    CASE(S_FALSE);
    CASE(E_ABORT);
    CASE(E_ACCESSDENIED);
    CASE(E_FAIL);
    CASE(E_HANDLE);
    CASE(E_INVALIDARG);
    CASE(E_NOINTERFACE);
    CASE(E_NOTIMPL);
    CASE(E_OUTOFMEMORY);
    CASE(E_POINTER);
    CASE(E_UNEXPECTED);

    CASE(DXGI_ERROR_ACCESS_DENIED);
    CASE(DXGI_ERROR_ACCESS_LOST);
    CASE(DXGI_ERROR_CANNOT_PROTECT_CONTENT);
    CASE(DXGI_ERROR_DEVICE_HUNG);
    CASE(DXGI_ERROR_DEVICE_REMOVED);
    CASE(DXGI_ERROR_DEVICE_RESET);
    CASE(DXGI_ERROR_DRIVER_INTERNAL_ERROR);
    CASE(DXGI_ERROR_FRAME_STATISTICS_DISJOINT);
    CASE(DXGI_ERROR_GRAPHICS_VIDPN_SOURCE_IN_USE);
    CASE(DXGI_ERROR_INVALID_CALL);
    CASE(DXGI_ERROR_MORE_DATA);
    CASE(DXGI_ERROR_NAME_ALREADY_EXISTS);
    CASE(DXGI_ERROR_NONEXCLUSIVE);
    CASE(DXGI_ERROR_NOT_CURRENTLY_AVAILABLE);
    CASE(DXGI_ERROR_NOT_FOUND);
    CASE(DXGI_ERROR_REMOTE_CLIENT_DISCONNECTED);
    CASE(DXGI_ERROR_REMOTE_OUTOFMEMORY);
    CASE(DXGI_ERROR_RESTRICT_TO_OUTPUT_STALE);
    CASE(DXGI_ERROR_SDK_COMPONENT_MISSING);
    CASE(DXGI_ERROR_SESSION_DISCONNECTED);
    CASE(DXGI_ERROR_UNSUPPORTED);
    CASE(DXGI_ERROR_WAIT_TIMEOUT);
    CASE(DXGI_ERROR_WAS_STILL_DRAWING);
#undef CASE

    default:
        return "Unknown error";
    }
}

static char *format_error(void *ctx, DWORD error)
{
    wchar_t *wstr;
    if (!FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                        FORMAT_MESSAGE_FROM_SYSTEM |
                        FORMAT_MESSAGE_IGNORE_INSERTS, NULL, error,
                        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                        (LPWSTR)&wstr, 0, NULL))
    {
        return NULL;
    }

    // Trim any trailing newline from the message
    for (int i = wcslen(wstr) - 1; i >= 0; i--) {
        if (wstr[i] != '\r' && wstr[i] != '\n') {
            wstr[i + 1] = '\0';
            break;
        }
    }

    char *str = pl_to_utf8(ctx, wstr);
    LocalFree(wstr);
    return str;
}

char *pl_hresult_to_str_buf(char *buf, size_t buf_size, HRESULT hr)
{
    char *fmsg = format_error(NULL, hr);
    const char *code = hresult_str(hr);
    if (fmsg) {
        snprintf(buf, buf_size, "%s (%s, 0x%08lx)", fmsg, code, hr);
    } else {
        snprintf(buf, buf_size, "%s, 0x%08lx", code, hr);
    }
    pl_free(fmsg);
    return buf;
}

#define D3D11_DXGI_ENUM(prefix, define) { case prefix ## define: return #define; }

const char *pl_get_dxgi_format_name(DXGI_FORMAT fmt)
{
    switch (fmt) {
    D3D11_DXGI_ENUM(DXGI_FORMAT_, UNKNOWN);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R32G32B32A32_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R32G32B32A32_FLOAT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R32G32B32A32_UINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R32G32B32A32_SINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R32G32B32_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R32G32B32_FLOAT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R32G32B32_UINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R32G32B32_SINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R16G16B16A16_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R16G16B16A16_FLOAT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R16G16B16A16_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R16G16B16A16_UINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R16G16B16A16_SNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R16G16B16A16_SINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R32G32_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R32G32_FLOAT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R32G32_UINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R32G32_SINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R32G8X24_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, D32_FLOAT_S8X24_UINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R32_FLOAT_X8X24_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, X32_TYPELESS_G8X24_UINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R10G10B10A2_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R10G10B10A2_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R10G10B10A2_UINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R11G11B10_FLOAT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R8G8B8A8_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R8G8B8A8_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R8G8B8A8_UNORM_SRGB);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R8G8B8A8_UINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R8G8B8A8_SNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R8G8B8A8_SINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R16G16_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R16G16_FLOAT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R16G16_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R16G16_UINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R16G16_SNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R16G16_SINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R32_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, D32_FLOAT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R32_FLOAT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R32_UINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R32_SINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R24G8_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, D24_UNORM_S8_UINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R24_UNORM_X8_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, X24_TYPELESS_G8_UINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R8G8_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R8G8_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R8G8_UINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R8G8_SNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R8G8_SINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R16_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R16_FLOAT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, D16_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R16_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R16_UINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R16_SNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R16_SINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R8_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R8_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R8_UINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R8_SNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R8_SINT);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, A8_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R1_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R9G9B9E5_SHAREDEXP);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R8G8_B8G8_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, G8R8_G8B8_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, BC1_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, BC1_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, BC1_UNORM_SRGB);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, BC2_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, BC2_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, BC2_UNORM_SRGB);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, BC3_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, BC3_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, BC3_UNORM_SRGB);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, BC4_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, BC4_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, BC4_SNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, BC5_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, BC5_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, BC5_SNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, B5G6R5_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, B5G5R5A1_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, B8G8R8A8_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, B8G8R8X8_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, R10G10B10_XR_BIAS_A2_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, B8G8R8A8_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, B8G8R8A8_UNORM_SRGB);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, B8G8R8X8_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, B8G8R8X8_UNORM_SRGB);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, BC6H_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, BC6H_UF16);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, BC6H_SF16);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, BC7_TYPELESS);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, BC7_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, BC7_UNORM_SRGB);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, AYUV);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, Y410);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, Y416);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, NV12);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, P010);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, P016);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, 420_OPAQUE);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, YUY2);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, Y210);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, Y216);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, NV11);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, AI44);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, IA44);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, P8);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, A8P8);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, B4G4R4A4_UNORM);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, P208);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, V208);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, V408);
    D3D11_DXGI_ENUM(DXGI_FORMAT_, FORCE_UINT);
    }

    return "<unknown>";
}

const char *pl_get_dxgi_csp_name(DXGI_COLOR_SPACE_TYPE csp)
{
    switch ((int) csp) {
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, RGB_FULL_G22_NONE_P709);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, RGB_FULL_G10_NONE_P709);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, RGB_STUDIO_G22_NONE_P709);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, RGB_STUDIO_G22_NONE_P2020);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, RESERVED);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, YCBCR_FULL_G22_NONE_P709_X601);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, YCBCR_STUDIO_G22_LEFT_P601);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, YCBCR_FULL_G22_LEFT_P601);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, YCBCR_STUDIO_G22_LEFT_P709);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, YCBCR_FULL_G22_LEFT_P709);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, YCBCR_STUDIO_G22_LEFT_P2020);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, YCBCR_FULL_G22_LEFT_P2020);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, RGB_FULL_G2084_NONE_P2020);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, YCBCR_STUDIO_G2084_LEFT_P2020);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, RGB_STUDIO_G2084_NONE_P2020);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, YCBCR_STUDIO_G22_TOPLEFT_P2020);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, YCBCR_STUDIO_G2084_TOPLEFT_P2020);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, RGB_FULL_G22_NONE_P2020);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, YCBCR_STUDIO_GHLG_TOPLEFT_P2020);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, YCBCR_FULL_GHLG_TOPLEFT_P2020);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, RGB_STUDIO_G24_NONE_P709);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, RGB_STUDIO_G24_NONE_P2020);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, YCBCR_STUDIO_G24_LEFT_P709);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, YCBCR_STUDIO_G24_LEFT_P2020);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, YCBCR_STUDIO_G24_TOPLEFT_P2020);
    D3D11_DXGI_ENUM(DXGI_COLOR_SPACE_, CUSTOM);
    }

    return "<unknown>";
}
