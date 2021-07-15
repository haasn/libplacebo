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

void pl_d3d11_flush_message_queue(struct d3d11_ctx *ctx, const char *header)
{
    if (!ctx->iqueue)
        return;

    static const enum pl_log_level severity_map[] = {
        [D3D11_MESSAGE_SEVERITY_CORRUPTION] = PL_LOG_FATAL,
        [D3D11_MESSAGE_SEVERITY_ERROR] = PL_LOG_ERR,
        [D3D11_MESSAGE_SEVERITY_WARNING] = PL_LOG_WARN,
        [D3D11_MESSAGE_SEVERITY_INFO] = PL_LOG_DEBUG,
        [D3D11_MESSAGE_SEVERITY_MESSAGE] = PL_LOG_DEBUG,
    };

    enum pl_log_level header_printed = PL_LOG_NONE;
    uint64_t messages = ID3D11InfoQueue_GetNumStoredMessages(ctx->iqueue);
    if (!messages)
        return;

    uint64_t discarded =
        ID3D11InfoQueue_GetNumMessagesDiscardedByMessageCountLimit(ctx->iqueue);
    if (discarded > ctx->last_discarded) {
        PL_WARN(ctx, "%s:", header);
        header_printed = PL_LOG_WARN;

        // Notify number of messages skipped due to the message count limit
        PL_WARN(ctx, "    (skipped %llu debug layer messages)",
                discarded - ctx->last_discarded);
        ctx->last_discarded = discarded;
    }

    // Copy debug layer messages to libplacebo's log output
    D3D11_MESSAGE *d3dmsg = NULL;
    for (uint64_t i = 0; i < messages; i++) {
        SIZE_T len;
        D3D(ID3D11InfoQueue_GetMessage(ctx->iqueue, i, NULL, &len));

        d3dmsg = pl_zalloc(NULL, len);
        D3D(ID3D11InfoQueue_GetMessage(ctx->iqueue, i, d3dmsg, &len));

        enum pl_log_level level = severity_map[d3dmsg->Severity];
        if (pl_msg_test(ctx->log, level)) {
            // If the header hasn't been printed, or it was printed for a lower
            // log level than the current message, print it (again)
            if (header_printed == PL_LOG_NONE || header_printed > level) {
                PL_MSG(ctx, level, "%s:", header);
                header_printed = level;
            }

            PL_MSG(ctx, level, "    %d: %.*s", (int) d3dmsg->ID,
                   (int) d3dmsg->DescriptionByteLength, d3dmsg->pDescription);
        }
        pl_free_ptr(&d3dmsg);
    }

    ID3D11InfoQueue_ClearStoredMessages(ctx->iqueue);
error:
    pl_free_ptr(&d3dmsg);
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
