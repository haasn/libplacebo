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

#include "config.h"
#if !PL_HAVE_VULKAN
#error Included <libplacebo/vulkan.h> but libplacebo built without vulkan!
#endif

#include <vulkan/vulkan.h>
#include <libplacebo/gpu.h>
#include <libplacebo/swapchain.h>

// Structure representing a VkInstance. Using this is not required.
struct pl_vk_inst {
    VkInstance instance;
    uint64_t priv;

    // The instance extensions that were successfully enabled, including
    // extensions enabled by libplacebo internally. May contain duplicates.
    const char **extensions;
    int num_extensions;
};

struct pl_vk_inst_params {
    // If set, enable the debugging and validation layers.
    bool debug;

    // Enables extra instance extensions. Instance creation will fail if these
    // extensions are not all supported. The user may use this to enable e.g.
    // windowing system integration.
    const char **extensions;
    int num_extensions;

    // Enables extra optional instance extensions. These are opportunistically
    // enabled if supported by the device, but otherwise skipped.
    const char **opt_extensions;
    int num_opt_extensions;
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

    // The device extensions that were successfully enabled, including
    // extensions enabled by libplacebo internally. May contain duplicates.
    const char **extensions;
    int num_extensions;

    // The list of enabled queue families and their queue counts. This may
    // include secondary queue families providing compute or transfer
    // capabilities.
    const struct pl_vulkan_queue *queues;
    int num_queues;
};

struct pl_vulkan_queue {
    int index; // Queue family index
    int count; // Queue family count
};

struct pl_vulkan_params {
    // The vulkan instance. Optional, if NULL then libplacebo will internally
    // create a VkInstance with the settings from `instance_params`.
    //
    // NOTE: The VkInstance provided by the user *MUST* be created with the
    // `VK_KHR_get_physical_device_properties2` extension enabled!
    VkInstance instance;

    // Configures the settings used for creating an internal vulkan instance.
    // May be NULL. Ignored if `instance` is set.
    const struct pl_vk_inst_params *instance_params;

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

    // Enables extra optional device extensions. These are opportunistically
    // enabled if supported by the device, but otherwise skipped.
    const char **opt_extensions;
    int num_opt_extensions;
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

// VkImage interop API

// Wraps an external VkImage into a pl_tex abstraction. By default, the image
// is considered "held" by the user and will not be touched or used by
// libplacebo until released (see `pl_vulkan_release`). Releasing an image
// makes libplacebo take ownership of the image until the user calls
// `pl_vulkan_hold` on it again. During this time, the user should not use
// the VkImage in any way.
//
// This wrapper can be destroyed by simply calling `pl_tex_destroy` on it,
// which will not destroy the underlying VkImage. If a pl_tex wrapper is
// destroyed while an image is not currently being held by the user, that
// image is left in an undefined state.
//
// Wrapping the same VkImage multiple times is undefined behavior, as is trying
// to wrap an image belonging to a different VkDevice than the one in use by
// `gpu`.
//
// The VkImage must be usable concurrently by all of the queue family indices
// listed in `pl_vulkan->queues`. Note that this requires the use of
// VK_SHARING_MODE_CONCURRENT if `pl_vulkan->num_queues` is greater than 1. If
// this is difficult to achieve for the user, then `async_transfer` /
// `async_compute` should be turned off, which guarantees the use of only one
// queue family.
//
// This function may fail, for example if the VkFormat specified does not
// map to any corresponding `pl_fmt`.
const struct pl_tex *pl_vulkan_wrap(const struct pl_gpu *gpu,
                                    VkImage image, int w, int h, int d,
                                    VkFormat format, VkImageUsageFlags usage);

// Analogous to `pl_vulkan_wrap`, this function takes any `pl_tex` (including
// one created by libplacebo) and unwraps it to expose the underlying VkImage
// to the user. Unlike `pl_vulkan_wrap`, this `pl_tex` is *not* considered held
// by default - the user must explicitly `pl_vulkan_hold` before accessing the
// VkImage for the first time.
//
// `out_format` and `out_flags` will be updated to hold the VkImage's
// format and usage flags. (Optional)
VkImage pl_vulkan_unwrap(const struct pl_gpu *gpu, const struct pl_tex *tex,
                         VkFormat *out_format, VkImageUsageFlags *out_flags);

// "Hold" a shared image. This will transition the image into the layout and
// access mode specified by the user, and fire the given semaphore (required!)
// when this is done. This marks the image as held. Attempting to perform any
// pl_tex_* operation (except pl_tex_destroy) on a held image is an error.
//
// Returns whether successful.
bool pl_vulkan_hold(const struct pl_gpu *gpu, const struct pl_tex *tex,
                    VkImageLayout layout, VkAccessFlags access,
                    VkSemaphore sem_out);

// "Hold" a shared image for external use. This will transition the image into
// a general layout and access mode, and the external queue. It will fire the
// given semaphore (required!) when this is done. This marks the image as held.
// Attempting to perform any pl_tex_* operation (except pl_tex_destroy) on a
// held image is an error.
//
// Returns whether successful.
bool pl_vulkan_hold_external(const struct pl_gpu *gpu,
                             const struct pl_tex *tex,
                             VkSemaphore sem_out);

// "Release" a shared image, meaning it is no longer held. `layout` and
// `access` describe the current state of the image at the point in time when
// the user is releasing it. Performing any operation on the VkImage underlying
// this `pl_tex` while it is not being held by the user is undefined behavior.
//
// If `sem_in` is specified, it must fire before libplacebo will actually use
// or modify the image. (Optional)
void pl_vulkan_release(const struct pl_gpu *gpu, const struct pl_tex *tex,
                       VkImageLayout layout, VkAccessFlags access,
                       VkSemaphore sem_in);

// "Release" a shared image from external use. This is a convenience wrapper
// around `pl_vulkan_release` that automatically sets the layout and access
// mode to those used for external sharing.
void pl_vulkan_release_external(const struct pl_gpu *gpu,
                                const struct pl_tex *tex,
                                VkSemaphore sem_in);

#endif // LIBPLACEBO_VULKAN_H_
