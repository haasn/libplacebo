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

#include "../common.h"
#include "../log.h"

// Shared struct used to hold the D3D11 device and associated interfaces
struct d3d11_ctx {
    pl_log log;
    pl_d3d11 d3d11;

    // Copy of the device from pl_d3d11 for convenience. Does not hold an
    // additional reference.
    ID3D11Device *dev;

    // DXGI device. This does hold a reference.
    IDXGIDevice1 *dxgi_dev;

    // Debug interfaces
    ID3D11Debug *debug;
    ID3D11InfoQueue *iqueue;
    uint64_t last_discarded; // Last count of discarded messages

    // pl_gpu_is_failed (We saw a device removed error!)
    bool is_failed;
};

// Pointer to dxgi.dll!CreateDXGIFactory1()
typedef HRESULT (WINAPI *PFN_CREATE_DXGI_FACTORY)(REFIID riid, void **ppFactory);

// DDK value. Apparently some D3D functions can return this instead of the
// proper user-mode error code. See:
// https://docs.microsoft.com/en-us/windows/win32/api/dxgi/nf-dxgi-idxgiswapchain-present
#define D3DDDIERR_DEVICEREMOVED (0x88760870)

#ifndef D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE
#define D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE (0x80)
#endif
