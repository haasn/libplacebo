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

#define SW_PFN(name) __typeof__(pl_swapchain_##name) *name
struct pl_sw_fns {
    // This destructor follows the same rules as `pl_gpu_fns`
    void (*destroy)(pl_swapchain sw);

    SW_PFN(latency); // optional
    SW_PFN(resize); // optional
    SW_PFN(colorspace_hint); // optional
    SW_PFN(start_frame);
    SW_PFN(submit_frame);
    SW_PFN(swap_buffers);
};
#undef SW_PFN
