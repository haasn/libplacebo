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
#include "ra.h"

struct pl_vulkan {
    const struct ra *ra;
    void *priv;

    // The vulkan objects in use. The user may use this for their own purposes,
    // but please note that the lifetime is tied to the lifetime of the
    // pl_vulkan object. The user must not destroy these themselves. Also,
    // the created vulkan device may have any number of queues and queue
    // family assignments; so using it for queue submission is ill-advised.
    VkInstance instance;
    VkPhysicalDevice phys_device;
    VkDevice device;
};

struct pl_vulkan_params {
    // When choosing the device, rule out all devices that don't support
    // presenting to this surface. When creating a device, enable all extensions
    // needed to ensure we can present to this surface. Optional.
    VkSurfaceKHR surface;

    // --- Instance creation options

    // The vulkan instance. May be set by the caller to inherit an existing
    // vulkan instace. If left as NULL, libplacebo will create its own vulkan
    // instance.
    VkInstance instance;

    // When creating a new VkInstance, this controls whether or not to load the
    // debugging and validation layers. No effect if `instance` is set.
    bool debug;

    // --- Physical device selection options

    // The vulkan physical device. May be set by the caller to indicate the
    // physical device to use. Otherwise, libplacebo will pick the "best"
    // available GPU, based on the advertised device type. (i.e., it will
    // prefer discrete GPUs over integrated GPUs)
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
// a new RA. This function will internally initialize a VkDevice. There is
// currently no way to share a vulkan device with the caller. If `params` is
// left as NULL, it defaults to &pl_vulkan_default_params.
const struct pl_vulkan *pl_vulkan_create(struct pl_context *ctx,
                                         const struct pl_vulkan_params *params);

// All resources allocated from the `ra` contained by this pl_vulkan must be
// explicitly destroyed by the user before calling pl_vulkan_destroy.
void pl_vulkan_destroy(const struct pl_vulkan **vk);

#endif // LIBPLACEBO_VULKAN_H_
