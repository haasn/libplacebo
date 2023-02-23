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

void pl_swapchain_destroy(pl_swapchain *ptr)
{
    pl_swapchain sw = *ptr;
    if (!sw)
        return;

    const struct pl_sw_fns *impl = PL_PRIV(sw);
    impl->destroy(sw);
    *ptr = NULL;
}

int pl_swapchain_latency(pl_swapchain sw)
{
    const struct pl_sw_fns *impl = PL_PRIV(sw);
    if (!impl->latency)
        return 0;

    return impl->latency(sw);
}

bool pl_swapchain_resize(pl_swapchain sw, int *width, int *height)
{
    int dummy[2] = {0};
    width = PL_DEF(width, &dummy[0]);
    height = PL_DEF(height, &dummy[1]);

    const struct pl_sw_fns *impl = PL_PRIV(sw);
    if (!impl->resize) {
        *width = *height = 0;
        return true;
    }

    return impl->resize(sw, width, height);
}

void pl_swapchain_colorspace_hint(pl_swapchain sw, const struct pl_color_space *csp)
{
    const struct pl_sw_fns *impl = PL_PRIV(sw);
    if (!impl->colorspace_hint)
        return;

    struct pl_swapchain_colors fix = {0};
    if (csp) {
        fix = *csp;

        bool has_metadata = !pl_hdr_metadata_equal(&fix.hdr, &pl_hdr_metadata_empty);
        bool is_hdr = pl_color_transfer_is_hdr(fix.transfer);

        // Ensure consistency of the metadata and requested transfer function
        if (has_metadata && !fix.transfer) {
            fix.transfer = PL_COLOR_TRC_PQ;
        } else if (has_metadata && !is_hdr) {
            fix.hdr = pl_hdr_metadata_empty;
        } else if (!has_metadata && is_hdr) {
            fix.hdr = pl_hdr_metadata_hdr10;
        }

        // Ensure we have valid values set for all the fields
        pl_color_space_infer(&fix);
    }

    impl->colorspace_hint(sw, &fix);
}

bool pl_swapchain_start_frame(pl_swapchain sw,
                              struct pl_swapchain_frame *out_frame)
{
    *out_frame = (struct pl_swapchain_frame) {0}; // sanity

    const struct pl_sw_fns *impl = PL_PRIV(sw);
    return impl->start_frame(sw, out_frame);
}

bool pl_swapchain_submit_frame(pl_swapchain sw)
{
    const struct pl_sw_fns *impl = PL_PRIV(sw);
    return impl->submit_frame(sw);
}

void pl_swapchain_swap_buffers(pl_swapchain sw)
{
    const struct pl_sw_fns *impl = PL_PRIV(sw);
    impl->swap_buffers(sw);
}
