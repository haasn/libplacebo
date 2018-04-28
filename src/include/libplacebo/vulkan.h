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

#ifndef LIBPLACEBO_VULKAN_H_
#define LIBPLACEBO_VULKAN_H_

#include <vulkan/vulkan.h>
#include <libplacebo/gpu.h>
#include <libplacebo/swapchain.h>

// Structure representing a VkInstance. Using this is not required.
struct pl_vk_inst {
    VkInstance instance;
    uint64_t priv;
};

struct pl_vk_inst_params {
    // If set, enable the debugging and validation layers.
    bool debug;

    // Enables extra instance extensions. Instance creation will fail if these
    // extensions are not all supported. The user may use this to enable e.g.
    // windowing system integration.
    const char **extensions;
    int num_extensions;
};

extern const struct pl_vk_inst_params pl_vk_inst_default_params;

// Helper function to simplify instance creation. The user could also bypass
// these helpers and do it manually, but this function is provided as a
// convenience. It also sets up a debug callback which forwards all vulkan
// messages to the `pl_context` log callback.
const struct pl_vk_inst *pl_vk_inst_create(struct pl_context *ctx,
                                           const struct pl_vk_inst_params *params);

void pl_vk_inst_destroy(const struct pl_vk_inst **inst);

// Structure representing the actual vulkan device and associated GPU instance
struct pl_vulkan {
    const struct pl_gpu *gpu;
    void *priv;

    // The vulkan objects in use. The user may use this for their own purposes,
    // but please note that the lifetime is tied to the lifetime of the
    // pl_vulkan object, and must not be destroyed by the user. Note that the
    // created vulkan device may have any number of queues and queue family
    // assignments; so using it for queue submission commands is ill-advised.
    VkInstance instance;
    VkPhysicalDevice phys_device;
    VkDevice device;
};

struct pl_vulkan_params {
    // The vulkan instance. Optional, if NULL then libplacebo will internally
    // create a VkInstance with no extra extensions or layers - but note that
    // this is not useful except for offline rendering.
    //
    // NOTE: The VkInstance provided by the user *MUST* be created with the
    // `VK_KHR_get_physical_device_properties2` extension enabled!
    VkInstance instance;

    // When choosing the device, rule out all devices that don't support
    // presenting to this surface. When creating a device, enable all extensions
    // needed to ensure we can present to this surface. Optional. Only legal
    // when specifying an existing VkInstance to use.
    VkSurfaceKHR surface;

    // --- Physical device selection options

    // The vulkan physical device. May be set by the caller to indicate the
    // physical device to use. Otherwise, libplacebo will pick the "best"
    // available GPU, based on the advertised device type. (i.e., it will
    // prefer discrete GPUs over integrated GPUs). Only legal when specifying
    // an existing VkInstance to use.
    VkPhysicalDevice device;

    // When choosing the device, only choose a device with this exact name.
    // This overrides `allow_software`. No effect if `device` is set. Note: A
    // list of devices and their names are logged at level PL_LOG_INFO.
    const char *device_name;

    // When choosing the device, controls whether or not to also allow software
    // GPUs. No effect if `device` or `device_name` are set.
    bool allow_software;

    // --- Logical device creation options

    // Controls whether or not to allow asynchronous transfers, using transfer
    // queue families, if supported by the device. This can be significantly
    // faster and more power efficient, and also allows streaming uploads in
    // parallel with rendering commands. Enabled by default.
    bool async_transfer;

    // Controls whether or not to allow asynchronous compute, using dedicated
    // compute queue families, if supported by the device. On some devices,
    // these can allow the GPU to schedule compute shaders in parallel with
    // fragment shaders. Enabled by default.
    bool async_compute;

    // Limits the number of queues to request. If left as 0, this will enable
    // as many queues as the device supports. Multiple queues can result in
    // improved efficiency when submitting multiple commands that can entirely
    // or partially execute in parallel. Defaults to 1, since using more queues
    // can actually decrease performance.
    int queue_count;

    // Enables extra device extensions. Device creation will fail if these
    // extensions are not all supported. The user may use this to enable e.g.
    // interop extensions.
    const char **extensions;
    int num_extensions;
};

// Default/recommended parameters. Should generally be safe and efficient.
extern const struct pl_vulkan_params pl_vulkan_default_params;

// Creates a new vulkan device based on the given parameters and initializes
// a new GPU. This function will internally initialize a VkDevice. There is
// currently no way to share a vulkan device with the caller. If `params` is
// left as NULL, it defaults to &pl_vulkan_default_params.
const struct pl_vulkan *pl_vulkan_create(struct pl_context *ctx,
                                         const struct pl_vulkan_params *params);

// Destroys the vulkan device and all associated objects, except for the
// VkInstance provided by the user.
//
// Note that all resources allocated from this vulkan object (e.g. via the
// `vk->ra` or using `pl_vulkan_create_swapchain`) *must* be explicitly
// destroyed by the user before calling this.
void pl_vulkan_destroy(const struct pl_vulkan **vk);

struct pl_vulkan_swapchain_params {
    // The surface to use for rendering. Required, the user is in charge of
    // creating this. Must belong to the same VkInstance as `vk->instance`.
    VkSurfaceKHR surface;

    // The image format and colorspace we should be using. Optional, if left
    // as {0}, libplacebo will pick the best surface format based on what the
    // GPU/surface seems to support.
    VkSurfaceFormatKHR surface_format;

    // The preferred presentation mode. See the vulkan documentation for more
    // information about these. If the device/surface combination does not
    // support this mode, libplacebo will fall back to VK_PRESENT_MODE_FIFO_KHR.
    //
    // Warning: Leaving this zero-initialized is the same as having specified
    // VK_PRESENT_MODE_IMMEDIATE_KHR, which is probably not what the user
    // wants!
    VkPresentModeKHR present_mode;

    // Allow up to N in-flight frames. This essentially controls how many
    // rendering commands may be queued up at the same time. See the
    // documentation for `pl_swapchain_get_latency` for more information. For
    // vulkan specifically, we are only able to wait until the GPU has finished
    // rendering a frame - we are unable to wait until the display has actually
    // finished displaying it. So this only provides a rough guideline.
    // Optional, defaults to 3.
    int swapchain_depth;
};

// Creates a new vulkan swapchain based on an existing VkSurfaceKHR. Using this
// function requires that the vulkan device was created with the
// VK_KHR_swapchain extension. The easiest way of accomplishing this is to set
// the `pl_vulkan_params.surface` explicitly at creation time.
const struct pl_swapchain *pl_vulkan_create_swapchain(const struct pl_vulkan *vk,
                              const struct pl_vulkan_swapchain_params *params);

#endif // LIBPLACEBO_VULKAN_H_
