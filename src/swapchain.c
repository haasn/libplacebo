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

#include "common.h"
#include "context.h"
#include "swapchain.h"

void ra_swapchain_destroy(const struct ra_swapchain **ptr)
{
    const struct ra_swapchain *sw = *ptr;
    if (!sw)
        return;

    sw->impl->destroy(sw);
    *ptr = NULL;
}

int ra_swapchain_latency(const struct ra_swapchain *sw)
{
    if (!sw->impl->latency)
        return 0;

    return sw->impl->latency(sw);
}

bool ra_swapchain_start_frame(const struct ra_swapchain *sw,
                              struct ra_swapchain_frame *out_frame)
{
    *out_frame = (struct ra_swapchain_frame) {0}; // sanity
    return sw->impl->start_frame(sw, out_frame);
}

bool ra_swapchain_submit_frame(const struct ra_swapchain *sw)
{
    return sw->impl->submit_frame(sw);
}

void ra_swapchain_swap_buffers(const struct ra_swapchain *sw)
{
    sw->impl->swap_buffers(sw);
}

