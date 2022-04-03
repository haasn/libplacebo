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

#pragma once

#include "common.h"

#define DXGI_COLOR_SPACE_RGB_STUDIO_G24_NONE_P709       ((DXGI_COLOR_SPACE_TYPE)20)
#define DXGI_COLOR_SPACE_RGB_STUDIO_G24_NONE_P2020      ((DXGI_COLOR_SPACE_TYPE)21)
#define DXGI_COLOR_SPACE_YCBCR_STUDIO_G24_LEFT_P709     ((DXGI_COLOR_SPACE_TYPE)22)
#define DXGI_COLOR_SPACE_YCBCR_STUDIO_G24_LEFT_P2020    ((DXGI_COLOR_SPACE_TYPE)23)
#define DXGI_COLOR_SPACE_YCBCR_STUDIO_G24_TOPLEFT_P2020 ((DXGI_COLOR_SPACE_TYPE)24)

// Flush debug messages from D3D11's info queue to libplacebo's log output.
// Should be called regularly.
void pl_d3d11_flush_message_queue(struct d3d11_ctx *ctx, const char *header);

// Some D3D11 functions can fail with a set of HRESULT codes which indicate the
// device has been removed. This is equivalent to libplacebo's gpu_is_failed
// state and indicates that the pl_gpu needs to be recreated. This function
// checks for one of those HRESULTs, sets the failed state, and returns a
// specific HRESULT that indicates why the device was removed (eg. GPU hang,
// driver crash, etc.)
HRESULT pl_d3d11_check_device_removed(struct d3d11_ctx *ctx, HRESULT hr);

// Helper function for the D3D() macro, though it can be called directly when
// handling D3D11 errors if the D3D() macro isn't suitable for some reason.
// Calls `pl_d3d11_check_device_removed` and `pl_d3d11_drain_debug_messages` and
// returns the specific HRESULT from `pl_d3d11_check_device_removed` for logging
// purposes.
HRESULT pl_d3d11_after_error(struct d3d11_ctx *ctx, HRESULT hr);

// Convenience macro for running DXGI/D3D11 functions and performing appropriate
// actions on failure. Can also be used for any HRESULT-returning function.
#define D3D(call)                                                         \
    do {                                                                  \
        HRESULT hr_ = (call);                                             \
        if (FAILED(hr_)) {                                                \
            hr_ = pl_d3d11_after_error(ctx, hr_);                         \
            PL_ERR(ctx, "%s: %s (%s:%d)", #call, pl_hresult_to_str(hr_),  \
                   __FILE__, __LINE__);                                   \
            goto error;                                                   \
        }                                                                 \
    } while (0);

// Conditionally release a COM interface and set the pointer to NULL
#define SAFE_RELEASE(iface)                   \
    do {                                      \
        if (iface)                            \
            (iface)->lpVtbl->Release(iface);  \
        (iface) = NULL;                       \
    } while (0)

struct dll_version {
    uint16_t major;
    uint16_t minor;
    uint16_t build;
    uint16_t revision;
};

// Get the version number of a DLL. This calls GetFileVersionInfoW, which should
// call LoadLibraryExW internally, so it should get the same copy of the DLL
// that is loaded into memory if there is a copy in System32 and a copy in the
// %PATH% or application directory.
struct dll_version pl_get_dll_version(const wchar_t *name);

wchar_t *pl_from_utf8(void *ctx, const char *str);
char *pl_to_utf8(void *ctx, const wchar_t *str);

#define pl_hresult_to_str(hr) pl_hresult_to_str_buf((char[256]){0}, 256, (hr))
char *pl_hresult_to_str_buf(char *buf, size_t buf_size, HRESULT hr);

const char *pl_get_dxgi_csp_name(DXGI_COLOR_SPACE_TYPE csp);
const char *pl_get_dxgi_format_name(DXGI_FORMAT fmt);
