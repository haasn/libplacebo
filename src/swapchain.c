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
#include "log.h"
#include "swapchain.h"

void pl_swapchain_destroy(const struct pl_swapchain **ptr)
{
    const struct pl_swapchain *sw = *ptr;
    if (!sw)
        return;

    sw->impl->destroy(sw);
    *ptr = NULL;
}

int pl_swapchain_latency(const struct pl_swapchain *sw)
{
    if (!sw->impl->latency)
        return 0;

    return sw->impl->latency(sw);
}

bool pl_swapchain_resize(const struct pl_swapchain *sw, int *width, int *height)
{
    int dummy[2] = {0};
    width = PL_DEF(width, &dummy[0]);
    height = PL_DEF(height, &dummy[1]);

    if (!sw->impl->resize) {
        *width = *height = 0;
        return true;
    }

    return sw->impl->resize(sw, width, height);
}

bool pl_swapchain_hdr_metadata(const struct pl_swapchain *sw,
                               const struct pl_hdr_metadata *metadata)
{
    if (!sw->impl->hdr_metadata)
        return false;

    return sw->impl->hdr_metadata(sw, metadata);
}

bool pl_swapchain_start_frame(const struct pl_swapchain *sw,
                              struct pl_swapchain_frame *out_frame)
{
    *out_frame = (struct pl_swapchain_frame) {0}; // sanity
    return sw->impl->start_frame(sw, out_frame);
}

bool pl_swapchain_submit_frame(const struct pl_swapchain *sw)
{
    return sw->impl->submit_frame(sw);
}

void pl_swapchain_swap_buffers(const struct pl_swapchain *sw)
{
    sw->impl->swap_buffers(sw);
}
