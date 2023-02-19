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

#include "../common.h"
#include "log.h"

#include <libplacebo/d3d11.h>

const struct pl_d3d11_params pl_d3d11_default_params = { PL_D3D11_DEFAULTS };

pl_d3d11 pl_d3d11_create(pl_log log, const struct pl_d3d11_params *params)
{
    pl_fatal(log, "libplacebo compiled without D3D11 support!");
    return NULL;
}

void pl_d3d11_destroy(pl_d3d11 *pd3d11)
{
    pl_d3d11 d3d11 = *pd3d11;
    pl_assert(!d3d11);
}

pl_d3d11 pl_d3d11_get(pl_gpu gpu)
{
    return NULL;
}

pl_swapchain pl_d3d11_create_swapchain(pl_d3d11 d3d11,
    const struct pl_d3d11_swapchain_params *params)
{
    pl_unreachable();
}

IDXGISwapChain *pl_d3d11_swapchain_unwrap(pl_swapchain sw)
{
    pl_unreachable();
}

pl_tex pl_d3d11_wrap(pl_gpu gpu, const struct pl_d3d11_wrap_params *params)
{
    pl_unreachable();
}
