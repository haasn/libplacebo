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

#pragma once

#include "common.h"

struct d3d_format {
    DXGI_FORMAT dxfmt;
    int minor; // The D3D11 minor version number which supports this format
    struct pl_fmt_t fmt;
};

extern const struct d3d_format pl_d3d11_formats[];

static inline DXGI_FORMAT fmt_to_dxgi(pl_fmt fmt)
{
    const struct d3d_format **fmtp = PL_PRIV(fmt);
    return (*fmtp)->dxfmt;
}

void pl_d3d11_setup_formats(struct pl_gpu_t *gpu);
