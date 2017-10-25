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
#include "../context.h"

// Vulkan allows the optional use of a custom allocator. We don't need one but
// mark this parameter with a better name in case we ever decide to change this
// in the future. (And to make the code more readable)
#define VK_ALLOC NULL

// Needed to load some extension-specific functions (for whatever reason)
#define VK_LOAD_PFN(name) PFN_##name pfn_##name = (PFN_##name) \
                            vkGetInstanceProcAddr(vk->inst, #name);

// Shared struct used to hold vulkan context information
struct vk_ctx {
    bool external_instance;
    struct pl_context *ctx;
    VkInstance inst;
    VkPhysicalDevice physd;
    VkPhysicalDeviceLimits limits;
    VkDebugReportCallbackEXT dbg;
    VkDevice dev;

    struct vk_cmdpool **pools;    // command pools (one per queue family)
    int num_pools;

    // Pointers into *pools
    struct vk_cmdpool *pool_graphics; // required
    struct vk_cmdpool *pool_compute;  // optional
    struct vk_cmdpool *pool_transfer; // optional

    // Queued/pending commands. These are shared for the entire mpvk_ctx to
    // ensure submission and callbacks are FIFO
    struct vk_cmd **cmds_queued;  // recorded but not yet submitted
    struct vk_cmd **cmds_pending; // submitted but not completed
    int num_cmds_queued;
    int num_cmds_pending;

    // A dynamic reference to the most recently submitted command that has not
    // yet completed. Used to implemented vk_dev_callback. Gets cleared when
    // the command completes.
    struct vk_cmd *last_cmd;

    // Common pool of signals, to avoid having to re-create these objects often
    struct vk_signal **signals;
    int num_signals;
};
