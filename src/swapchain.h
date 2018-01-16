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

#define RA_SW_PFN(name) __typeof__(ra_swapchain_##name) *name
struct ra_sw {
    // This destructor follows the same rules as `ra_fns`
    void (*destroy)(const struct ra_swapchain *sw);

    RA_SW_PFN(latency); // optional
    RA_SW_PFN(start_frame);
    RA_SW_PFN(submit_frame);
    RA_SW_PFN(swap_buffers);
};
#undef RA_SW_PFN
