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

PL_API_BEGIN

// Structure representing a VkInstance. Using this is not required.
typedef const struct pl_vk_inst_t {
    VkInstance instance;

    // The Vulkan API version supported by this VkInstance.
    uint32_t api_version;

    // The associated vkGetInstanceProcAddr pointer.
    PFN_vkGetInstanceProcAddr get_proc_addr;

    // The instance extensions that were successfully enabled, including
    // extensions enabled by libplacebo internally. May contain duplicates.
    const char * const *extensions;
    int num_extensions;

    // The instance layers that were successfully enabled, including
    // layers enabled by libplacebo internally. May contain duplicates.
    const char * const *layers;
    int num_layers;
} *pl_vk_inst;

struct pl_vk_inst_params {
    // If set, enable the debugging and validation layers. These should
    // generally be lightweight and relatively harmless to enable.
    bool debug;

    // If set, also enable GPU-assisted verification and best practices
    // layers. (Note: May cause substantial slowdown and/or result in lots of
    // false positive spam)
    bool debug_extra;

    // If nonzero, restricts the Vulkan API version to be at most this. This
    // is only really useful for explicitly testing backwards compatibility.
    uint32_t max_api_version;

    // Pointer to a user-provided `vkGetInstanceProcAddr`. If this is NULL,
    // libplacebo will use the directly linked version (if available).
    PFN_vkGetInstanceProcAddr get_proc_addr;

    // Enables extra instance extensions. Instance creation will fail if these
    // extensions are not all supported. The user may use this to enable e.g.
    // windowing system integration.
    const char * const *extensions;
    int num_extensions;

    // Enables extra optional instance extensions. These are opportunistically
    // enabled if supported by the device, but otherwise skipped.
    const char * const *opt_extensions;
    int num_opt_extensions;

    // Enables extra layers. Instance creation will fail if these layers are
    // not all supported.
    //
    // NOTE: Layers needed for required/optional extensions are automatically
    // enabled. The user does not specifically need to enable layers related
    // to extension support.
    const char * const *layers;
    int num_layers;

    // Enables extra optional layers. These are opportunistically enabled if
    // supported by the platform, but otherwise skipped.
    const char * const *opt_layers;
    int num_opt_layers;
};

#define pl_vk_inst_params(...) (&(struct pl_vk_inst_params) { __VA_ARGS__ })
extern const struct pl_vk_inst_params pl_vk_inst_default_params;

// Helper function to simplify instance creation. The user could also bypass
// these helpers and do it manually, but this function is provided as a
// convenience. It also sets up a debug callback which forwards all vulkan
// messages to the `pl_log` callback.
pl_vk_inst pl_vk_inst_create(pl_log log, const struct pl_vk_inst_params *params);
void pl_vk_inst_destroy(pl_vk_inst *inst);

struct pl_vulkan_queue {
    uint32_t index; // Queue family index
    uint32_t count; // Queue family count
};

// Structure representing the actual vulkan device and associated GPU instance
typedef const struct pl_vulkan_t *pl_vulkan;
struct pl_vulkan_t {
    pl_gpu gpu;

    // The vulkan objects in use. The user may use this for their own purposes,
    // but please note that the lifetime is tied to the lifetime of the
    // pl_vulkan object, and must not be destroyed by the user. Note that the
    // created vulkan device may have any number of queues and queue family
    // assignments; so using it for queue submission commands is ill-advised.
    VkInstance instance;
    VkPhysicalDevice phys_device;
    VkDevice device;

    // The associated vkGetInstanceProcAddr pointer.
    PFN_vkGetInstanceProcAddr get_proc_addr;

    // The Vulkan API version supported by this VkPhysicalDevice.
    uint32_t api_version;

    // The device extensions that were successfully enabled, including
    // extensions enabled by libplacebo internally. May contain duplicates.
    const char * const *extensions;
    int num_extensions;

    // The device features that were enabled at device creation time.
    const VkPhysicalDeviceFeatures2 *features;

    // The explicit queue families we are using to provide a given capability.
    struct pl_vulkan_queue queue_graphics; // provides VK_QUEUE_GRAPHICS_BIT
    struct pl_vulkan_queue queue_compute;  // provides VK_QUEUE_COMPUTE_BIT
    struct pl_vulkan_queue queue_transfer; // provides VK_QUEUE_TRANSFER_BIT

    // For convenience, these are the same enabled queue families and their
    // queue counts in list form. This list does not contain duplicates.
    const struct pl_vulkan_queue *queues;
    int num_queues;

    // Functions for locking a queue. These must be used to lock VkQueues for
    // submission or other related operations when sharing the VkDevice between
    // multiple threads, Using this on queue families or indices not contained
    // in `queues` is undefined behavior.
    void (*lock_queue)(pl_vulkan vk, int qf, int qidx);
    void (*unlock_queue)(pl_vulkan vk, int qf, int qidx);
};

struct pl_vulkan_params {
    // The vulkan instance. Optional, if NULL then libplacebo will internally
    // create a VkInstance with the settings from `instance_params`.
    //
    // Note: The VkInstance provided by the user *MUST* be created with a
    // VkApplicationInfo.apiVersion of VK_API_VERSION_1_1 or higher.
    VkInstance instance;

    // Pointer to `vkGetInstanceProcAddr`. If this is NULL, libplacebo will
    // use the directly linked version (if available).
    //
    // Note: This overwrites the same value from `instance_params`.
    PFN_vkGetInstanceProcAddr get_proc_addr;

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

    // When choosing the device, only choose a device with this exact UUID.
    // This overrides `allow_software` and `device_name`. No effect if `device`
    // is set.
    uint8_t device_uuid[16];

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
    const char * const *extensions;
    int num_extensions;

    // Enables extra optional device extensions. These are opportunistically
    // enabled if supported by the device, but otherwise skipped.
    const char * const *opt_extensions;
    int num_opt_extensions;

    // Optional extra features to enable at device creation time. These are
    // opportunistically enabled if supported by the physical device, but
    // otherwise kept disabled. Users may include extra extension-specific
    // features in the pNext chain, however these *must* all be
    // extension-specific structs, i.e. the use of "meta-structs" like
    // VkPhysicalDeviceVulkan11Features is not allowed.
    const VkPhysicalDeviceFeatures2 *features;

    // --- Misc/debugging options

    // Restrict specific features to e.g. work around driver bugs, or simply
    // for testing purposes
    int max_glsl_version;       // limit the maximum GLSL version
    uint32_t max_api_version;   // limit the maximum vulkan API version

    // Removed parameters (no effect)
    bool disable_events PL_DEPRECATED;
};

// Default/recommended parameters. Should generally be safe and efficient.
#define PL_VULKAN_DEFAULTS                              \
    .async_transfer = true,                             \
    .async_compute  = true,                             \
    /* enabling multiple queues often decreases perf */ \
    .queue_count    = 1,

#define pl_vulkan_params(...) (&(struct pl_vulkan_params) { PL_VULKAN_DEFAULTS __VA_ARGS__ })
extern const struct pl_vulkan_params pl_vulkan_default_params;

// Creates a new vulkan device based on the given parameters and initializes
// a new GPU. This function will internally initialize a VkDevice. There is
// currently no way to share a vulkan device with the caller. If `params` is
// left as NULL, it defaults to &pl_vulkan_default_params.
//
// Thread-safety: Safe
pl_vulkan pl_vulkan_create(pl_log log, const struct pl_vulkan_params *params);

// Destroys the vulkan device and all associated objects, except for the
// VkInstance provided by the user.
//
// Note that all resources allocated from this vulkan object (e.g. via the
// `vk->ra` or using `pl_vulkan_create_swapchain`) *must* be explicitly
// destroyed by the user before calling this.
//
// Also note that this function will block until all in-flight GPU commands are
// finished processing. You can avoid this by manually calling `pl_gpu_finish`
// before `pl_vulkan_destroy`.
void pl_vulkan_destroy(pl_vulkan *vk);

// For a `pl_gpu` backed by `pl_vulkan`, this function can be used to retrieve
// the underlying `pl_vulkan`. Returns NULL for any other type of `gpu`.
pl_vulkan pl_vulkan_get(pl_gpu gpu);

struct pl_vulkan_device_params {
    // The instance to use. Required!
    //
    // Note: The VkInstance provided by the user *must* be created with a
    // VkApplicationInfo.apiVersion of VK_API_VERSION_1_1 or higher.
    VkInstance instance;

    // Mirrored from `pl_vulkan_params`. All of these fields are optional.
    PFN_vkGetInstanceProcAddr get_proc_addr;
    VkSurfaceKHR surface;
    const char *device_name;
    uint8_t device_uuid[16];
    bool allow_software;
};

#define pl_vulkan_device_params(...) (&(struct pl_vulkan_device_params) { __VA_ARGS__ })

// Helper function to choose the best VkPhysicalDevice, given a VkInstance.
// This uses the same logic as `pl_vulkan_create` uses internally. If no
// matching device was found, this returns VK_NULL_HANDLE.
VkPhysicalDevice pl_vulkan_choose_device(pl_log log,
                              const struct pl_vulkan_device_params *params);

struct pl_vulkan_swapchain_params {
    // The surface to use for rendering. Required, the user is in charge of
    // creating this. Must belong to the same VkInstance as `vk->instance`.
    VkSurfaceKHR surface;

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

    // This suppresses automatic recreation of the swapchain when any call
    // returns VK_SUBOPTIMAL_KHR. Normally, libplacebo will recreate the
    // swapchain internally on the next `pl_swapchain_start_frame`. If enabled,
    // clients are assumed to take care of swapchain recreations themselves, by
    // calling `pl_swapchain_resize` as appropriate. libplacebo will tolerate
    // the "suboptimal" status indefinitely.
    bool allow_suboptimal;
};

#define pl_vulkan_swapchain_params(...) (&(struct pl_vulkan_swapchain_params) { __VA_ARGS__ })

// Creates a new vulkan swapchain based on an existing VkSurfaceKHR. Using this
// function requires that the vulkan device was created with the
// VK_KHR_swapchain extension. The easiest way of accomplishing this is to set
// the `pl_vulkan_params.surface` explicitly at creation time.
pl_swapchain pl_vulkan_create_swapchain(pl_vulkan vk,
                              const struct pl_vulkan_swapchain_params *params);

// This will return true if the vulkan swapchain is internally detected
// as being suboptimal (VK_SUBOPTIMAL_KHR). This might be of use to clients
// who have `params->allow_suboptimal` enabled.
bool pl_vulkan_swapchain_suboptimal(pl_swapchain sw);

// Vulkan interop API, for sharing a single VkDevice (and associated vulkan
// resources) directly with the API user. The use of this API is a bit sketchy
// and requires careful communication of Vulkan API state.

struct pl_vulkan_import_params {
    // The vulkan instance. Required.
    //
    // Note: The VkInstance provided by the user *must* be created with a
    // VkApplicationInfo.apiVersion of VK_API_VERSION_1_1 or higher.
    VkInstance instance;

    // Pointer to `vkGetInstanceProcAddr`. If this is NULL, libplacebo will
    // use the directly linked version (if available).
    PFN_vkGetInstanceProcAddr get_proc_addr;

    // The physical device selected by the user. Required.
    VkPhysicalDevice phys_device;

    // The logical device created by the user. Required.
    VkDevice device;

    // --- Logical device parameters

    // List of all device-level extensions that were enabled. (Instance-level
    // extensions need not be re-specified here, since it's guaranteed that any
    // instance-level extensions that device-level extensions depend on were
    // enabled at the instance level)
    const char * const *extensions;
    int num_extensions;

    // Enabled queue families. At least `queue_graphics` is required.
    //
    // It's okay for multiple queue families to be specified with the same
    // index, e.g. in the event that a dedicated compute queue also happens to
    // be the dedicated transfer queue.
    //
    // It's also okay to leave the queue struct as {0} in the event that no
    // dedicated queue exists for a given operation type. libplacebo will
    // automatically fall back to using e.g. the graphics queue instead.
    struct pl_vulkan_queue queue_graphics; // must support VK_QUEUE_GRAPHICS_BIT
    struct pl_vulkan_queue queue_compute;  // must support VK_QUEUE_COMPUTE_BIT
    struct pl_vulkan_queue queue_transfer; // must support VK_QUEUE_TRANSFER_BIT

    // Enabled VkPhysicalDeviceFeatures. The VkDevice provided by the user
    // *must* be created with the `timelineSemaphore` feature enabled.
    const VkPhysicalDeviceFeatures2 *features;

    // Functions for locking a queue. If set, these will be used instead of
    // libplacebo's internal functions for `pl_vulkan.(un)lock_queue`.
    void (*lock_queue)(void *ctx, int qf, int qidx);
    void (*unlock_queue)(void *ctx, int qf, int qidx);
    void *queue_ctx;

    // --- Misc/debugging options

    // Restrict specific features to e.g. work around driver bugs, or simply
    // for testing purposes. See `pl_vulkan_params` for a description of these.
    int max_glsl_version;
    uint32_t max_api_version;

    // Removed parameters (no effect)
    bool disable_events PL_DEPRECATED;
};

#define pl_vulkan_import_params(...) (&(struct pl_vulkan_import_params) { __VA_ARGS__ })

// Import an existing VkDevice instead of creating a new one, and wrap it into
// a `pl_vulkan` abstraction. It's safe to `pl_vulkan_destroy` this, which will
// destroy application state related to libplacebo but leave the underlying
// VkDevice intact.
pl_vulkan pl_vulkan_import(pl_log log, const struct pl_vulkan_import_params *params);

struct pl_vulkan_wrap_params {
    // The image itself. It *must* be usable concurrently by all of the queue
    // family indices listed in `pl_vulkan->queues`. Note that this requires
    // the use of VK_SHARING_MODE_CONCURRENT if `pl_vulkan->num_queues` is
    // greater than 1. If this is difficult to achieve for the user, then
    // `async_transfer` / `async_compute` should be turned off, which
    // guarantees the use of only one queue family.
    VkImage image;

    // The image's dimensions (unused dimensions must be 0)
    int width;
    int height;
    int depth;

    // The image's format. libplacebo will try to map this to an equivalent
    // pl_fmt. If no compatible pl_fmt is found, wrapping will fail.
    VkFormat format;

    // The usage flags the image was created with. libplacebo will set the
    // pl_tex capabilities to include whatever it can, as determined by the set
    // of enabled usage flags.
    VkImageUsageFlags usage;

    // See `pl_tex_params`
    void *user_data;
    pl_debug_tag debug_tag;
};

#define pl_vulkan_wrap_params(...) (&(struct pl_vulkan_wrap_params) {   \
        .debug_tag = PL_DEBUG_TAG,                                      \
        __VA_ARGS__                                                     \
    })

// Wraps an external VkImage into a pl_tex abstraction. By default, the image
// is considered "held" by the user and must be released before calling any
// pl_tex_* API calls on it (see `pl_vulkan_release`).
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
// This function may fail, in which case it returns NULL.
pl_tex pl_vulkan_wrap(pl_gpu gpu, const struct pl_vulkan_wrap_params *params);

// For purely informative reasons, this contains a list of extensions and
// device features that libplacebo *can* make use of. These are all strictly
// optional, but provide a hint to the API user as to what might be worth
// enabling at device creation time.
//
// Note: This also includes physical device features provided by extensions.
// They are all provided using extension-specific features structs, rather
// than the more general purpose VkPhysicalDeviceVulkan11Features etc.
extern const char * const pl_vulkan_recommended_extensions[];
extern const int pl_vulkan_num_recommended_extensions;
extern const VkPhysicalDeviceFeatures2 pl_vulkan_recommended_features;

// Analogous to `pl_vulkan_wrap`, this function takes any `pl_tex` (including
// ones created by `pl_tex_create`) and unwraps it to expose the underlying
// VkImage to the user. Unlike `pl_vulkan_wrap`, this `pl_tex` is *not*
// considered held after calling this function - the user must explicitly
// `pl_vulkan_hold` before accessing the VkImage.
//
// `out_format` and `out_flags` will be updated to hold the VkImage's
// format and usage flags. (Optional)
VkImage pl_vulkan_unwrap(pl_gpu gpu, pl_tex tex,
                         VkFormat *out_format, VkImageUsageFlags *out_flags);

// Represents a vulkan semaphore/value pair (for compatibility with timeline
// semaphores). When using normal, binary semaphores, `value` may be ignored.
typedef struct pl_vulkan_sem {
    VkSemaphore sem;
    uint64_t value;
} pl_vulkan_sem;

struct pl_vulkan_hold_params {
    // The Vulkan image to hold. It will be marked as held. Attempting to
    // perform any pl_tex_* operation (except pl_tex_destroy) on a held image
    // is undefined behavior.
    pl_tex tex;

    // The layout to transition the image to when holding. Alternatively, a
    // pointer to receive the current image layout. If `out_layout` is
    // provided, `layout` is ignored.
    VkImageLayout layout;
    VkImageLayout *out_layout;

    // The queue family index to transition the image to. This can be used with
    // VK_QUEUE_FAMILY_EXTERNAL to transition the image to an external API. As
    // a special case, if set to VK_QUEUE_FAMILY_IGNORED, libplacebo will not
    // transition the image, even if this image was not set up for concurrent
    // usage. Ignored for concurrent images.
    uint32_t qf;

    // The semaphore to fire when the image is available for use. (Required)
    pl_vulkan_sem semaphore;
};

#define pl_vulkan_hold_params(...) (&(struct pl_vulkan_hold_params) { __VA_ARGS__ })

// "Hold" a shared image, transferring control over the image to the user.
// Returns whether successful.
bool pl_vulkan_hold_ex(pl_gpu gpu, const struct pl_vulkan_hold_params *params);

struct pl_vulkan_release_params {
    // The image to be released. It must be marked as "held". Performing any
    // operation on the VkImage underlying this `pl_tex` while it is not being
    // held by the user is undefined behavior.
    pl_tex tex;

    // The current layout of the image at the point in time when `semaphore`
    // fires, or if no semaphore is specified, at the time of call.
    VkImageLayout layout;

    // The queue family index to transition the image to. This can be used with
    // VK_QUEUE_FAMILY_EXTERNAL to transition the image rom an external API. As
    // a special case, if set to VK_QUEUE_FAMILY_IGNORED, libplacebo will not
    // transition the image, even if this image was not set up for concurrent
    // usage. Ignored for concurrent images.
    uint32_t qf;

    // The semaphore to wait on before libplacebo will actually use or modify
    // the image. (Optional)
    //
    // Note: the lifetime of `semaphore` is indeterminate, and destroying it
    // while the texture is still depending on that semaphore is undefined
    // behavior.
    //
    // Technically, the only way to be sure that it's safe to free is to use
    // `pl_gpu_finish()` or similar (e.g. `pl_vulkan_destroy` or
    // `vkDeviceWaitIdle`) after another operation involving `tex` has been
    // emitted (or the texture has been destroyed).
    pl_vulkan_sem semaphore;
};

#define pl_vulkan_release_params(...) (&(struct pl_vulkan_release_params) { __VA_ARGS__ })

// "Release" a shared image, transferring control to libplacebo.
void pl_vulkan_release_ex(pl_gpu gpu, const struct pl_vulkan_release_params *params);

struct pl_vulkan_sem_params {
    // The type of semaphore to create.
    VkSemaphoreType type;

    // For VK_SEMAPHORE_TYPE_TIMELINE, sets the initial timeline value.
    uint64_t initial_value;

    // If set, exports this VkSemaphore to the handle given in `out_handle`.
    // The user takes over ownership, and should manually close it before
    // destroying this VkSemaphore (via `pl_vulkan_sem_destroy`).
    enum pl_handle_type export_handle;
    union pl_handle *out_handle;

    // Optional debug tag to identify this semaphore.
    pl_debug_tag debug_tag;
};

#define pl_vulkan_sem_params(...) (&(struct pl_vulkan_sem_params) {     \
        .debug_tag = PL_DEBUG_TAG,                                      \
        __VA_ARGS__                                                     \
    })

// Helper functions to create and destroy vulkan semaphores. Returns
// VK_NULL_HANDLE on failure.
VkSemaphore pl_vulkan_sem_create(pl_gpu gpu, const struct pl_vulkan_sem_params *params);
void pl_vulkan_sem_destroy(pl_gpu gpu, VkSemaphore *semaphore);

// Backwards-compatibility wrappers for older versions of the API.
static inline bool pl_vulkan_hold(pl_gpu gpu, pl_tex tex, VkImageLayout layout,
                                  pl_vulkan_sem sem_out)
{
    return pl_vulkan_hold_ex(gpu, pl_vulkan_hold_params(
        .tex        = tex,
        .layout     = layout,
        .semaphore  = sem_out,
        .qf         = VK_QUEUE_FAMILY_IGNORED,
    ));
}

static inline bool pl_vulkan_hold_raw(pl_gpu gpu, pl_tex tex,
                                      VkImageLayout *out_layout,
                                      pl_vulkan_sem sem_out)
{
    return pl_vulkan_hold_ex(gpu, pl_vulkan_hold_params(
        .tex        = tex,
        .out_layout = out_layout,
        .semaphore  = sem_out,
        .qf         = VK_QUEUE_FAMILY_IGNORED,
    ));
}

static inline void pl_vulkan_release(pl_gpu gpu, pl_tex tex, VkImageLayout layout,
                                     pl_vulkan_sem sem_in)
{
    pl_vulkan_release_ex(gpu, pl_vulkan_release_params(
        .tex        = tex,
        .layout     = layout,
        .semaphore  = sem_in,
        .qf         = VK_QUEUE_FAMILY_IGNORED,
    ));
}

PL_API_END

#endif // LIBPLACEBO_VULKAN_H_
