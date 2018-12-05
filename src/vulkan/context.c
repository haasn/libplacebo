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

#include "common.h"
#include "command.h"
#include "utils.h"
#include "gpu.h"

const struct pl_vk_inst_params pl_vk_inst_default_params = {0};

struct vk_ext_fun {
    const char *name;
    size_t offset;
};

struct vk_ext {
    const char *name;
    struct vk_ext_fun *funs;
};

#define VK_DEV_FUN(N)                       \
    { .name = #N,                           \
      .offset = offsetof(struct vk_ctx, N), \
    }

// Table of vulkan instance extensions
// TODO: add support for loading function pointers
static const char *vk_instance_extensions[] = {
    VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
};

// Table of vulkan device extensions and functions they load
static const struct vk_ext vk_device_extensions[] = {
    {
        .name = VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
        .funs = (struct vk_ext_fun[]) {
            VK_DEV_FUN(vkCmdPushDescriptorSetKHR),
            {0},
        },
    }, {
        .name = VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        .funs = (struct vk_ext_fun[]) {{0}},
    }, {
        .name = VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        .funs = (struct vk_ext_fun[]) {
            VK_DEV_FUN(vkGetMemoryFdKHR),
            {0},
        },
#ifdef VK_HAVE_WIN32
    }, {
        .name = VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        .funs = (struct vk_ext_fun[]) {
            VK_DEV_FUN(vkGetMemoryWin32HandleKHR),
            {0},
        },
#endif
    }, {
        .name = VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        .funs = (struct vk_ext_fun[]) {{0}},
    }, {
        .name = VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        .funs = (struct vk_ext_fun[]) {
            VK_DEV_FUN(vkGetSemaphoreFdKHR),
            {0},
        },
#ifdef VK_HAVE_WIN32
    }, {
        .name = VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
        .funs = (struct vk_ext_fun[]) {
            VK_DEV_FUN(vkGetSemaphoreWin32HandleKHR),
            {0},
        },
#endif
    }
};

void pl_vk_inst_destroy(const struct pl_vk_inst **inst_ptr)
{
    const struct pl_vk_inst *inst = *inst_ptr;
    if (!inst)
        return;

    VkDebugReportCallbackEXT debug = (VkDebugReportCallbackEXT) inst->priv;
    if (debug) {
        VK_LOAD_FUN(inst->instance, vkDestroyDebugReportCallbackEXT)
        vkDestroyDebugReportCallbackEXT(inst->instance, debug, VK_ALLOC);
    }

    vkDestroyInstance(inst->instance, VK_ALLOC);
    TA_FREEP((void **) inst_ptr);
}

static VkBool32 VKAPI_PTR vk_dbg_callback(VkDebugReportFlagsEXT flags,
                                          VkDebugReportObjectTypeEXT objType,
                                          uint64_t obj, size_t loc,
                                          int32_t msgCode, const char *layer,
                                          const char *msg, void *priv)
{
    struct pl_context *ctx = priv;
    enum pl_log_level lev = PL_LOG_INFO;

    switch (flags) {
    case VK_DEBUG_REPORT_ERROR_BIT_EXT:               lev = PL_LOG_ERR;   break;
    case VK_DEBUG_REPORT_WARNING_BIT_EXT:             lev = PL_LOG_WARN;  break;
    case VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT: lev = PL_LOG_WARN;  break;
    case VK_DEBUG_REPORT_DEBUG_BIT_EXT:               lev = PL_LOG_DEBUG; break;
    case VK_DEBUG_REPORT_INFORMATION_BIT_EXT:         lev = PL_LOG_TRACE; break;
    };

    pl_msg(ctx, lev, "vk [%s] %d: %s (obj 0x%llx (%s), loc 0x%zx)",
           layer, (int) msgCode, msg, (unsigned long long) obj,
           vk_obj_str(objType), loc);

    // The return value of this function determines whether the call will
    // be explicitly aborted (to prevent GPU errors) or not. In this case,
    // we generally want this to be on for the errors.
    return !!(flags & VK_DEBUG_REPORT_ERROR_BIT_EXT);
}

const struct pl_vk_inst *pl_vk_inst_create(struct pl_context *ctx,
                                           const struct pl_vk_inst_params *params)
{
    void *tmp = talloc_new(NULL);
    params = PL_DEF(params, &pl_vk_inst_default_params);
    VkInstance inst = NULL;

    const char **exts = NULL;
    int num_exts = 0;

    VkInstanceCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    };

    // Enumerate all supported extensions
    uint32_t num_exts_avail = 0;
    vkEnumerateInstanceExtensionProperties(NULL, &num_exts_avail, NULL);
    VkExtensionProperties *exts_avail = talloc_array(tmp, VkExtensionProperties, num_exts_avail);
    vkEnumerateInstanceExtensionProperties(NULL, &num_exts_avail, exts_avail);

    // Add mandatory extensions
    TARRAY_APPEND(tmp, exts, num_exts, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    // Add optional extensions
    for (int i = 0; i < PL_ARRAY_SIZE(vk_instance_extensions); i++) {
        const char *ext = vk_instance_extensions[i];
        for (int n = 0; n < num_exts_avail; n++) {
            if (strcmp(ext, exts_avail[n].extensionName) == 0) {
                TARRAY_APPEND(tmp, exts, num_exts, ext);
                break;
            }
        }
    }

    // Add extra user extensions
    for (int i = 0; i < params->num_extensions; i++)
        TARRAY_APPEND(tmp, exts, num_exts, params->extensions[i]);

    // Add extra optional user extensions
    for (int i = 0; i < params->num_opt_extensions; i++) {
        const char *ext = params->opt_extensions[i];
        for (int n = 0; n < num_exts_avail; n++) {
            if (strcmp(ext, exts_avail[n].extensionName) == 0) {
                TARRAY_APPEND(tmp, exts, num_exts, ext);
                break;
            }
        }
    }

    if (params->debug) {
        pl_info(ctx, "Enabling vulkan debug layers");
        // Enables the LunarG standard validation layer, which
        // is a meta-layer that loads lots of other validators
        static const char *layers[] = {
            "VK_LAYER_LUNARG_standard_validation",
        };
        info.ppEnabledLayerNames = layers;
        info.enabledLayerCount = PL_ARRAY_SIZE(layers);

        // Enable support for debug callbacks, so we get useful messages
        TARRAY_APPEND(tmp, exts, num_exts, VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    }

    info.ppEnabledExtensionNames = exts;
    info.enabledExtensionCount = num_exts;

    pl_info(ctx, "Creating vulkan instance%s", num_exts ? " with extensions:" : "");
    for (int i = 0; i < num_exts; i++)
        pl_info(ctx, "    %s", exts[i]);

    VkResult res = vkCreateInstance(&info, VK_ALLOC, &inst);
    if (res != VK_SUCCESS) {
        pl_fatal(ctx, "Failed creating instance: %s", vk_res_str(res));
        goto error;
    }

    VkDebugReportCallbackEXT debug = VK_NULL_HANDLE;
    if (params->debug) {
        // Set up a debug callback to catch validation messages
        VkDebugReportCallbackCreateInfoEXT dinfo = {
            .sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
            .flags = VK_DEBUG_REPORT_INFORMATION_BIT_EXT |
                     VK_DEBUG_REPORT_WARNING_BIT_EXT |
                     VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT |
                     VK_DEBUG_REPORT_ERROR_BIT_EXT |
                     VK_DEBUG_REPORT_DEBUG_BIT_EXT,
            .pfnCallback = vk_dbg_callback,
            .pUserData = ctx,
        };

        // Since this is not part of the core spec, we need to load it. This
        // can't fail because we've already successfully created an instance
        // with this extension enabled.
        VK_LOAD_FUN(inst, vkCreateDebugReportCallbackEXT)
        vkCreateDebugReportCallbackEXT(inst, &dinfo, VK_ALLOC, &debug);
    }

    struct pl_vk_inst *pl_vk = talloc_struct(NULL, struct pl_vk_inst, {
        .instance = inst,
        .priv = (uint64_t) debug,
        .extensions = exts,
        .num_extensions = num_exts,
    });

    pl_vk->extensions = talloc_steal(pl_vk, pl_vk->extensions);
    talloc_free(tmp);
    return pl_vk;

error:
    pl_fatal(ctx, "Failed initializing vulkan instance");
    vkDestroyInstance(inst, VK_ALLOC);
    talloc_free(tmp);
    return NULL;
}

const struct pl_vulkan_params pl_vulkan_default_params = {
    .async_transfer = true,
    .async_compute  = true,
    .queue_count    = 1, // enabling multiple queues often decreases perf
};

void pl_vulkan_destroy(const struct pl_vulkan **pl_vk)
{
    if (!*pl_vk)
        return;

    pl_gpu_destroy((*pl_vk)->gpu);

    struct vk_ctx *vk = (*pl_vk)->priv;
    if (vk->dev) {
        PL_DEBUG(vk, "Flushing remaining commands...");
        vk_wait_idle(vk);
        pl_assert(vk->num_cmds_queued == 0);
        pl_assert(vk->num_cmds_pending == 0);
        for (int i = 0; i < vk->num_pools; i++)
            vk_cmdpool_destroy(vk, vk->pools[i]);
        for (int i = 0; i < vk->num_signals; i++)
            vk_signal_destroy(vk, &vk->signals[i]);
        vkDestroyDevice(vk->dev, VK_ALLOC);
    }

    pl_vk_inst_destroy(&vk->internal_instance);
    TA_FREEP((void **) pl_vk);
}

static bool supports_surf(struct vk_ctx *vk, VkPhysicalDevice physd,
                          VkSurfaceKHR surf)
{
    uint32_t qfnum;
    vkGetPhysicalDeviceQueueFamilyProperties(physd, &qfnum, NULL);

    for (int i = 0; i < qfnum; i++) {
        VkBool32 sup;
        VK(vkGetPhysicalDeviceSurfaceSupportKHR(physd, i, surf, &sup));
        if (sup)
            return true;
    }

error:
    return false;
}

static bool find_physical_device(struct vk_ctx *vk,
                                 const struct pl_vulkan_params *params)
{
    PL_INFO(vk, "Probing for vulkan devices:");
    bool ret = false;

    VkPhysicalDevice *devices = NULL;
    uint32_t num = 0;
    VK(vkEnumeratePhysicalDevices(vk->inst, &num, NULL));
    devices = talloc_array(NULL, VkPhysicalDevice, num);
    VK(vkEnumeratePhysicalDevices(vk->inst, &num, devices));

    static const struct { const char *name; int priority; } types[] = {
        [VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU]   = {"discrete",   5},
        [VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU] = {"integrated", 4},
        [VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU]    = {"virtual",    3},
        [VK_PHYSICAL_DEVICE_TYPE_CPU]            = {"software",   2},
        [VK_PHYSICAL_DEVICE_TYPE_OTHER]          = {"other",      1},
        [VK_PHYSICAL_DEVICE_TYPE_END_RANGE+1]    = {0},
    };

    int best = 0;
    for (int i = 0; i < num; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        VkPhysicalDeviceType t = props.deviceType;
        PL_INFO(vk, "    GPU %d: %s (%s)", i, props.deviceName, types[t].name);

        if (params->surface && !supports_surf(vk, devices[i], params->surface)) {
            PL_DEBUG(vk, "      -> excluding due to lack of surface support");
            continue;
        }

        if (params->device_name && params->device_name[0] != '\0') {
            if (strcmp(params->device_name, props.deviceName) == 0) {
                vk->physd = devices[i];
                best = 10; // high number...
            } else {
                PL_DEBUG(vk, "      -> excluding due to name mismatch");
                continue;
            }
        }

        if (!params->allow_software && t == VK_PHYSICAL_DEVICE_TYPE_CPU) {
            PL_DEBUG(vk, "      -> excluding due to params->allow_software");
            continue;
        }

        if (types[t].priority > best) {
            vk->physd = devices[i];
            best = types[t].priority;
        }
    }

    if (!vk->physd) {
        PL_FATAL(vk, "Found no suitable device, giving up.");
        goto error;
    }

    ret = true;

error:
    talloc_free(devices);
    return ret;
}

// Find the most specialized queue supported a combination of flags. In cases
// where there are multiple queue families at the same specialization level,
// this finds the one with the most queues. Returns -1 if no queue was found.
static int find_qf(VkQueueFamilyProperties *qfs, int qfnum, VkQueueFlags flags)
{
    int idx = -1;
    for (int i = 0; i < qfnum; i++) {
        if (!(qfs[i].queueFlags & flags))
            continue;

        // QF is more specialized. Since we don't care about other bits like
        // SPARSE_BIT, mask the ones we're interestew in
        const VkQueueFlags mask = VK_QUEUE_GRAPHICS_BIT |
                                  VK_QUEUE_TRANSFER_BIT |
                                  VK_QUEUE_COMPUTE_BIT;

        if (idx < 0 || (qfs[i].queueFlags & mask) < (qfs[idx].queueFlags & mask))
            idx = i;

        // QF has more queues (at the same specialization level)
        if (qfs[i].queueFlags == qfs[idx].queueFlags &&
            qfs[i].queueCount > qfs[idx].queueCount)
            idx = i;
    }

    return idx;
}

static void add_qinfo(void *tactx, VkDeviceQueueCreateInfo **qinfos,
                      int *num_qinfos, VkQueueFamilyProperties *qfs, int idx,
                      int qcount)
{
    if (idx < 0)
        return;

    // Check to see if we've already added this queue family
    for (int i = 0; i < *num_qinfos; i++) {
        if ((*qinfos)[i].queueFamilyIndex == idx)
            return;
    }

    if (!qcount)
        qcount = qfs[idx].queueCount;

    float *priorities = talloc_zero_array(tactx, float, qcount);
    VkDeviceQueueCreateInfo qinfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = idx,
        .queueCount = PL_MIN(qcount, qfs[idx].queueCount),
        .pQueuePriorities = priorities,
    };

    TARRAY_APPEND(tactx, *qinfos, *num_qinfos, qinfo);
}

static bool device_init(struct vk_ctx *vk, const struct pl_vulkan_params *params)
{
    pl_assert(vk->physd);
    void *tmp = talloc_new(NULL);

    // Enumerate the queue families and find suitable families for each task
    int qfnum;
    vkGetPhysicalDeviceQueueFamilyProperties(vk->physd, &qfnum, NULL);
    VkQueueFamilyProperties *qfs = talloc_array(tmp, VkQueueFamilyProperties, qfnum);
    vkGetPhysicalDeviceQueueFamilyProperties(vk->physd, &qfnum, qfs);

    PL_INFO(vk, "Queue families supported by device:");

    for (int i = 0; i < qfnum; i++) {
        PL_INFO(vk, "    QF %d: flags 0x%x num %d", i,
                (unsigned) qfs[i].queueFlags, (int) qfs[i].queueCount);
    }

    int idx_gfx = -1, idx_comp = -1, idx_tf = -1;
    idx_gfx = find_qf(qfs, qfnum, VK_QUEUE_GRAPHICS_BIT);
    if (params->async_compute)
        idx_comp = find_qf(qfs, qfnum, VK_QUEUE_COMPUTE_BIT);
    if (params->async_transfer)
        idx_tf = find_qf(qfs, qfnum, VK_QUEUE_TRANSFER_BIT);

    // Vulkan requires at least one GRAPHICS queue, so if this fails something
    // is horribly wrong.
    pl_assert(idx_gfx >= 0);
    PL_INFO(vk, "Using graphics queue (QF %d)", idx_gfx);

    // If needed, ensure we can actually present to the surface using this queue
    if (params->surface) {
        VkBool32 sup;
        VK(vkGetPhysicalDeviceSurfaceSupportKHR(vk->physd, idx_gfx,
                                                params->surface, &sup));
        if (!sup) {
            PL_FATAL(vk, "Queue family does not support surface presentation!");
            goto error;
        }
    }

    if (idx_tf >= 0 && idx_tf != idx_gfx)
        PL_INFO(vk, "Using async transfer (QF %d)", idx_tf);
    if (idx_comp >= 0 && idx_comp != idx_gfx)
        PL_INFO(vk, "Using async compute (QF %d)", idx_comp);

    // Fall back to supporting compute shaders via the graphics pool for
    // devices which support compute shaders but not async compute.
    if (idx_comp < 0 && qfs[idx_gfx].queueFlags & VK_QUEUE_COMPUTE_BIT)
        idx_comp = idx_gfx;

    // Cache the transfer queue alignment requirements
    if (idx_tf >= 0)
        vk->transfer_alignment = qfs[idx_tf].minImageTransferGranularity;

    // Now that we know which QFs we want, we can create the logical device
    VkDeviceQueueCreateInfo *qinfos = NULL;
    int num_qinfos = 0;
    add_qinfo(tmp, &qinfos, &num_qinfos, qfs, idx_gfx, params->queue_count);
    add_qinfo(tmp, &qinfos, &num_qinfos, qfs, idx_comp, params->queue_count);
    add_qinfo(tmp, &qinfos, &num_qinfos, qfs, idx_tf, params->queue_count);

    // Enumerate all supported extensions
    uint32_t num_exts_avail = 0;
    VK(vkEnumerateDeviceExtensionProperties(vk->physd, NULL, &num_exts_avail, NULL));
    VkExtensionProperties *exts_avail = talloc_array(tmp, VkExtensionProperties, num_exts_avail);
    VK(vkEnumerateDeviceExtensionProperties(vk->physd, NULL, &num_exts_avail, exts_avail));

    // Add all extensions we need
    const char ***exts = &vk->exts;
    int *num_exts = &vk->num_exts;
    if (params->surface)
        TARRAY_APPEND(vk, *exts, *num_exts, VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    // Add all extensions we can benefit from, and remember their functions
    const struct vk_ext_fun **ext_funs = NULL;
    int num_ext_funs = 0;
    for (int i = 0; i < PL_ARRAY_SIZE(vk_device_extensions); i++) {
        const struct vk_ext *ext = &vk_device_extensions[i];
        for (int n = 0; n < num_exts_avail; n++) {
            if (strcmp(ext->name, exts_avail[n].extensionName) == 0) {
                TARRAY_APPEND(vk, *exts, *num_exts, ext->name);
                for (const struct vk_ext_fun *f = ext->funs; f->name; f++)
                    TARRAY_APPEND(tmp, ext_funs, num_ext_funs, f);
                break;
            }
        }
    }

    // Add extra user extensions
    for (int i = 0; i < params->num_extensions; i++)
        TARRAY_APPEND(vk, *exts, *num_exts, params->extensions[i]);

    // Add optional extra user extensions
    for (int i = 0; i < params->num_opt_extensions; i++) {
        const char *ext = params->opt_extensions[i];
        for (int n = 0; n < num_exts_avail; n++) {
            if (strcmp(ext, exts_avail[n].extensionName) == 0) {
                TARRAY_APPEND(vk, *exts, *num_exts, ext);
                break;
            }
        }
    }

    // Enable all features that we might need (whitelisted)
    vkGetPhysicalDeviceFeatures(vk->physd, &vk->features);
#define FEATURE(name) .name = vk->features.name
    vk->features = (VkPhysicalDeviceFeatures) {
        FEATURE(shaderImageGatherExtended),
        FEATURE(shaderStorageImageExtendedFormats),
    };
#undef FEATURE

    VkDeviceCreateInfo dinfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pQueueCreateInfos = qinfos,
        .queueCreateInfoCount = num_qinfos,
        .ppEnabledExtensionNames = *exts,
        .enabledExtensionCount = *num_exts,
        .pEnabledFeatures = &vk->features,
    };

    PL_INFO(vk, "Creating vulkan device%s", *num_exts ? " with extensions:" : "");
    for (int i = 0; i < *num_exts; i++)
        PL_INFO(vk, "    %s", (*exts)[i]);

    VK(vkCreateDevice(vk->physd, &dinfo, VK_ALLOC, &vk->dev));

    // Load all of the functions from the extensions we enabled
    for (int i = 0; i < num_ext_funs; i++) {
        const struct vk_ext_fun *fun = ext_funs[i];
        void *pfn = (void *) ((uintptr_t) vk + (ptrdiff_t) fun->offset);
        *((PFN_vkVoidFunction *) pfn) = vkGetDeviceProcAddr(vk->dev, fun->name);
    }

    // Create the command pools and memory allocator
    for (int i = 0; i < num_qinfos; i++) {
        int qf = qinfos[i].queueFamilyIndex;
        struct vk_cmdpool *pool = vk_cmdpool_create(vk, qinfos[i], qfs[qf]);
        if (!pool)
            goto error;
        TARRAY_APPEND(vk, vk->pools, vk->num_pools, pool);

        // Update the pool_* pointers based on the corresponding index
        if (qf == idx_gfx)
            vk->pool_graphics = pool;
        if (qf == idx_comp)
            vk->pool_compute = pool;
        if (qf == idx_tf)
            vk->pool_transfer = pool;
    }

    talloc_free(tmp);
    return true;

error:
    PL_FATAL(vk, "Failed creating logical device!");
    talloc_free(tmp);
    return false;
}

const struct pl_vulkan *pl_vulkan_create(struct pl_context *ctx,
                                         const struct pl_vulkan_params *params)
{
    params = PL_DEF(params, &pl_vulkan_default_params);
    struct pl_vulkan *pl_vk = talloc_zero(NULL, struct pl_vulkan);
    struct vk_ctx *vk = pl_vk->priv = talloc_zero(pl_vk, struct vk_ctx);
    vk->ctx = ctx;
    vk->inst = params->instance;

    if (!vk->inst) {
        pl_assert(!params->surface);
        pl_assert(!params->device);
        PL_DEBUG(vk, "No VkInstance provided, creating one...");
        vk->internal_instance = pl_vk_inst_create(ctx, params->instance_params);
        if (!vk->internal_instance)
            goto error;
        vk->inst = vk->internal_instance->instance;
    }

    // Choose the physical device
    if (params->device) {
        PL_DEBUG(vk, "Using specified VkPhysicalDevice");
        vk->physd = params->device;
    } else if (!find_physical_device(vk, params)) {
        goto error;
    }

    VkPhysicalDeviceProperties prop;
    vkGetPhysicalDeviceProperties(vk->physd, &prop);
    vk->limits = prop.limits;

    PL_INFO(vk, "Vulkan device properties:");
    PL_INFO(vk, "    Device Name: %s", prop.deviceName);
    PL_INFO(vk, "    Device ID: %x:%x", (unsigned) prop.vendorID,
            (unsigned) prop.deviceID);
    PL_INFO(vk, "    Driver version: %d", (int) prop.driverVersion);
    PL_INFO(vk, "    API version: %d.%d.%d",
            (int) VK_VERSION_MAJOR(prop.apiVersion),
            (int) VK_VERSION_MINOR(prop.apiVersion),
            (int) VK_VERSION_PATCH(prop.apiVersion));

    // Finally, initialize the logical device and the rest of the vk_ctx
    if (!device_init(vk, params))
        goto error;

    pl_vk->gpu = pl_gpu_create_vk(vk);
    if (!pl_vk->gpu)
        goto error;

    // Expose the resulting vulkan objects
    pl_vk->instance = vk->inst;
    pl_vk->phys_device = vk->physd;
    pl_vk->device = vk->dev;
    pl_vk->extensions = vk->exts;
    pl_vk->num_extensions = vk->num_exts;
    pl_vk->num_queues = vk->num_pools;
    pl_vk->queues = talloc_array(pl_vk, struct pl_vulkan_queue, vk->num_pools);
    for (int i = 0; i < vk->num_pools; i++) {
        struct pl_vulkan_queue *queues = (struct pl_vulkan_queue *) pl_vk->queues;
        queues[i] = (struct pl_vulkan_queue) {
            .index = vk->pools[i]->qf,
            .count = vk->pools[i]->num_queues,
        };
    }

    return pl_vk;

error:
    PL_FATAL(vk, "Failed initializing vulkan device");
    pl_vulkan_destroy((const struct pl_vulkan **) &pl_vk);
    return NULL;
}
