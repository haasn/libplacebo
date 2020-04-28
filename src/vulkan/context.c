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

struct vk_fun {
    const char *name;
    size_t offset;
    bool device_level;
};

struct vk_ext {
    const char *name;
    struct vk_fun *funs;
};

#define VK_INST_FUN(N)                      \
    { .name = "vk" #N,                      \
      .offset = offsetof(struct vk_ctx, N), \
    }

#define VK_DEV_FUN(N)                       \
    { .name = "vk" #N,                      \
      .offset = offsetof(struct vk_ctx, N), \
      .device_level = true,                 \
    }

// Table of optional vulkan instance extensions
static const char *vk_instance_extensions[] = {
    VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
};

// List of mandatory instance-level function pointers, including functions
// associated with mandatory instance extensions
static const struct vk_fun vk_inst_funs[] = {
    VK_INST_FUN(CreateDevice),
    VK_INST_FUN(EnumerateDeviceExtensionProperties),
    VK_INST_FUN(GetDeviceProcAddr),
    VK_INST_FUN(GetPhysicalDeviceFeatures),
    VK_INST_FUN(GetPhysicalDeviceFormatProperties),
    VK_INST_FUN(GetPhysicalDeviceImageFormatProperties2KHR),
    VK_INST_FUN(GetPhysicalDeviceMemoryProperties),
    VK_INST_FUN(GetPhysicalDeviceProperties),
    VK_INST_FUN(GetPhysicalDeviceProperties2KHR),
    VK_INST_FUN(GetPhysicalDeviceQueueFamilyProperties),
    VK_INST_FUN(GetPhysicalDeviceSurfaceCapabilitiesKHR),
    VK_INST_FUN(GetPhysicalDeviceSurfaceFormatsKHR),
    VK_INST_FUN(GetPhysicalDeviceSurfacePresentModesKHR),
    VK_INST_FUN(GetPhysicalDeviceSurfaceSupportKHR),
};

// Table of vulkan device extensions and functions they load, including
// functions exported by dependent instance-level extensions
static const struct vk_ext vk_device_extensions[] = {
    {
        .name = VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
        .funs = (struct vk_fun[]) {
            VK_DEV_FUN(CmdPushDescriptorSetKHR),
            {0},
        },
    }, {
        .name = VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        .funs = (struct vk_fun[]) {
            VK_INST_FUN(GetPhysicalDeviceExternalBufferPropertiesKHR),
            {0}
        },
    }, {
        .name = VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        .funs = (struct vk_fun[]) {
            VK_DEV_FUN(GetMemoryFdKHR),
            {0},
        },
    }, {
        .name = VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
        .funs = (struct vk_fun[]) {
            VK_DEV_FUN(GetMemoryFdPropertiesKHR),
            {0},
        },
#ifdef VK_HAVE_WIN32
    }, {
        .name = VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        .funs = (struct vk_fun[]) {
            VK_DEV_FUN(GetMemoryWin32HandleKHR),
            {0},
        },
#endif
    }, {
        .name = VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        .funs = (struct vk_fun[]) {
            VK_INST_FUN(GetPhysicalDeviceExternalSemaphorePropertiesKHR),
            {0},
        },
    }, {
        .name = VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        .funs = (struct vk_fun[]) {
            VK_DEV_FUN(GetSemaphoreFdKHR),
            {0},
        },
#ifdef VK_HAVE_WIN32
    }, {
        .name = VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
        .funs = (struct vk_fun[]) {
            VK_DEV_FUN(GetSemaphoreWin32HandleKHR),
            {0},
        },
#endif
    }, {
        .name = VK_EXT_PCI_BUS_INFO_EXTENSION_NAME,
        .funs = (struct vk_fun[]) {
            {0}
        },
    }, {
        .name = VK_EXT_HDR_METADATA_EXTENSION_NAME,
        .funs = (struct vk_fun[]) {
            VK_DEV_FUN(SetHdrMetadataEXT),
            {0},
        },
    },
};

// List of mandatory device-level functions
static const struct vk_fun vk_dev_funs[] = {
    VK_DEV_FUN(AcquireNextImageKHR),
    VK_DEV_FUN(AllocateCommandBuffers),
    VK_DEV_FUN(AllocateDescriptorSets),
    VK_DEV_FUN(AllocateMemory),
    VK_DEV_FUN(BeginCommandBuffer),
    VK_DEV_FUN(BindBufferMemory),
    VK_DEV_FUN(BindImageMemory),
    VK_DEV_FUN(CmdBeginRenderPass),
    VK_DEV_FUN(CmdBindDescriptorSets),
    VK_DEV_FUN(CmdBindPipeline),
    VK_DEV_FUN(CmdBindVertexBuffers),
    VK_DEV_FUN(CmdBlitImage),
    VK_DEV_FUN(CmdClearColorImage),
    VK_DEV_FUN(CmdCopyBuffer),
    VK_DEV_FUN(CmdCopyBufferToImage),
    VK_DEV_FUN(CmdCopyImage),
    VK_DEV_FUN(CmdCopyImageToBuffer),
    VK_DEV_FUN(CmdDispatch),
    VK_DEV_FUN(CmdDraw),
    VK_DEV_FUN(CmdEndRenderPass),
    VK_DEV_FUN(CmdPipelineBarrier),
    VK_DEV_FUN(CmdPushConstants),
    VK_DEV_FUN(CmdSetEvent),
    VK_DEV_FUN(CmdSetScissor),
    VK_DEV_FUN(CmdSetViewport),
    VK_DEV_FUN(CmdUpdateBuffer),
    VK_DEV_FUN(CmdWaitEvents),
    VK_DEV_FUN(CreateBuffer),
    VK_DEV_FUN(CreateBufferView),
    VK_DEV_FUN(CreateCommandPool),
    VK_DEV_FUN(CreateComputePipelines),
    VK_DEV_FUN(CreateDebugReportCallbackEXT),
    VK_DEV_FUN(CreateDescriptorPool),
    VK_DEV_FUN(CreateDescriptorSetLayout),
    VK_DEV_FUN(CreateEvent),
    VK_DEV_FUN(CreateFence),
    VK_DEV_FUN(CreateFramebuffer),
    VK_DEV_FUN(CreateGraphicsPipelines),
    VK_DEV_FUN(CreateImage),
    VK_DEV_FUN(CreateImageView),
    VK_DEV_FUN(CreatePipelineCache),
    VK_DEV_FUN(CreatePipelineLayout),
    VK_DEV_FUN(CreateRenderPass),
    VK_DEV_FUN(CreateSampler),
    VK_DEV_FUN(CreateSemaphore),
    VK_DEV_FUN(CreateShaderModule),
    VK_DEV_FUN(CreateSwapchainKHR),
    VK_DEV_FUN(DestroyBuffer),
    VK_DEV_FUN(DestroyBufferView),
    VK_DEV_FUN(DestroyCommandPool),
    VK_DEV_FUN(DestroyDebugReportCallbackEXT),
    VK_DEV_FUN(DestroyDescriptorPool),
    VK_DEV_FUN(DestroyDescriptorSetLayout),
    VK_DEV_FUN(DestroyDevice),
    VK_DEV_FUN(DestroyEvent),
    VK_DEV_FUN(DestroyFence),
    VK_DEV_FUN(DestroyFramebuffer),
    VK_DEV_FUN(DestroyImage),
    VK_DEV_FUN(DestroyImageView),
    VK_DEV_FUN(DestroyInstance),
    VK_DEV_FUN(DestroyPipeline),
    VK_DEV_FUN(DestroyPipelineCache),
    VK_DEV_FUN(DestroyPipelineLayout),
    VK_DEV_FUN(DestroyRenderPass),
    VK_DEV_FUN(DestroySampler),
    VK_DEV_FUN(DestroySemaphore),
    VK_DEV_FUN(DestroyShaderModule),
    VK_DEV_FUN(DestroySwapchainKHR),
    VK_DEV_FUN(EndCommandBuffer),
    VK_DEV_FUN(FlushMappedMemoryRanges),
    VK_DEV_FUN(FreeCommandBuffers),
    VK_DEV_FUN(FreeMemory),
    VK_DEV_FUN(GetBufferMemoryRequirements),
    VK_DEV_FUN(GetDeviceQueue),
    VK_DEV_FUN(GetImageMemoryRequirements),
    VK_DEV_FUN(GetPipelineCacheData),
    VK_DEV_FUN(GetSwapchainImagesKHR),
    VK_DEV_FUN(InvalidateMappedMemoryRanges),
    VK_DEV_FUN(MapMemory),
    VK_DEV_FUN(QueuePresentKHR),
    VK_DEV_FUN(QueueSubmit),
    VK_DEV_FUN(ResetEvent),
    VK_DEV_FUN(ResetFences),
    VK_DEV_FUN(UpdateDescriptorSets),
    VK_DEV_FUN(WaitForFences),
};

// Private struct for pl_vk_inst
struct priv {
    VkDebugReportCallbackEXT debug_cb;
};

void pl_vk_inst_destroy(const struct pl_vk_inst **inst_ptr)
{
    const struct pl_vk_inst *inst = *inst_ptr;
    if (!inst)
        return;

    struct priv *p = TA_PRIV(inst);
    if (p->debug_cb) {
        VK_LOAD_FUN(inst->instance, DestroyDebugReportCallbackEXT, inst->get_proc_addr);
        DestroyDebugReportCallbackEXT(inst->instance, p->debug_cb, VK_ALLOC);
    }

    VK_LOAD_FUN(inst->instance, DestroyInstance, inst->get_proc_addr);
    DestroyInstance(inst->instance, VK_ALLOC);
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

    // We will ignore errors for a designated object, but we need to explicitly
    // handle the case where no object is designated, because errors can have no
    // object associated with them, and we don't want to suppress those errors.
    if (ctx->suppress_errors_for_object != VK_NULL_HANDLE &&
        ctx->suppress_errors_for_object == obj)
        return false;

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

static PFN_vkGetInstanceProcAddr get_proc_addr_fallback(struct pl_context *ctx,
                                    PFN_vkGetInstanceProcAddr get_proc_addr)
{
    if (get_proc_addr)
        return get_proc_addr;

#ifdef VK_HAVE_PROC_ADDR
    return vkGetInstanceProcAddr;
#else
    pl_fatal(ctx, "No `vkGetInstanceProcAddr` function provided, and "
             "libplacebo built without linking against this function!");
    return NULL;
#endif
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

    PFN_vkGetInstanceProcAddr get_addr;
    if (!(get_addr = get_proc_addr_fallback(ctx, params->get_proc_addr)))
        goto error;

    // Enumerate all supported layers
    VK_LOAD_FUN(NULL, EnumerateInstanceLayerProperties, get_addr);
    uint32_t num_layers_avail = 0;
    EnumerateInstanceLayerProperties(&num_layers_avail, NULL);
    VkLayerProperties *layers_avail = talloc_zero_array(tmp, VkLayerProperties, num_layers_avail);
    EnumerateInstanceLayerProperties(&num_layers_avail, layers_avail);

    pl_debug(ctx, "Available layers:");
    for (int i = 0; i < num_layers_avail; i++)
        pl_debug(ctx, "    %s", layers_avail[i].layerName);

    const char **layers = NULL;
    int num_layers = 0;

    // Sorted by priority
    static const char *debug_layers[] = {
        "VK_LAYER_KHRONOS_validation",
        "VK_LAYER_LUNARG_standard_validation",
    };

    // This layer has to be initialized first, otherwise all sorts of weirdness
    // happens (random segfaults, yum)
    bool debug = params->debug;
    if (debug) {
        for (int i = 0; i < PL_ARRAY_SIZE(debug_layers); i++) {
            for (int n = 0; n < num_layers_avail; n++) {
                if (strcmp(debug_layers[i], layers_avail[n].layerName) != 0)
                    continue;

                pl_info(ctx, "Enabling debug meta layer: %s", debug_layers[i]);
                // Enable support for debug callbacks, so we get useful messages
                TARRAY_APPEND(tmp, exts, num_exts, VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
                TARRAY_APPEND(tmp, layers, num_layers, debug_layers[i]);
                goto debug_layers_done;
            }
        }

        // No layer found..
        pl_warn(ctx, "API debugging requested but no debug meta layers present... ignoring");
        debug = false;
    }

debug_layers_done: ;

    for (int i = 0; i < params->num_layers; i++)
        TARRAY_APPEND(tmp, layers, num_layers, params->layers[i]);

    for (int i = 0; i < params->num_opt_layers; i++) {
        const char *layer = params->opt_layers[i];
        for (int n = 0; n < num_layers_avail; n++) {
            if (strcmp(layer, layers_avail[n].layerName) == 0) {
                TARRAY_APPEND(tmp, layers, num_layers, layer);
                break;
            }
        }
    }

    // Enumerate all supported extensions
    VK_LOAD_FUN(NULL, EnumerateInstanceExtensionProperties, get_addr);
    uint32_t num_exts_avail = 0;
    EnumerateInstanceExtensionProperties(NULL, &num_exts_avail, NULL);
    VkExtensionProperties *exts_avail = talloc_zero_array(tmp, VkExtensionProperties, num_exts_avail);
    EnumerateInstanceExtensionProperties(NULL, &num_exts_avail, exts_avail);

    struct {
        VkExtensionProperties *exts;
        int num_exts;
    } *layer_exts = talloc_zero_array(tmp, __typeof__(*layer_exts), num_layers_avail);

    // Enumerate extensions from layers
    for (int i = 0; i < num_layers_avail; i++) {
        EnumerateInstanceExtensionProperties(layers_avail[i].layerName, &layer_exts[i].num_exts, NULL);
        layer_exts[i].exts = talloc_zero_array(tmp, VkExtensionProperties, layer_exts[i].num_exts);
        EnumerateInstanceExtensionProperties(layers_avail[i].layerName,
                                             &layer_exts[i].num_exts,
                                             layer_exts[i].exts);

        // Replace all extensions that are already available globally by {0}
        for (int j = 0; j < layer_exts[i].num_exts; j++) {
            for (int k = 0; k < num_exts_avail; k++) {
                if (strcmp(layer_exts[i].exts[j].extensionName, exts_avail[k].extensionName) == 0)
                    layer_exts[i].exts[j] = (VkExtensionProperties) {0};
            }
        }
    }

    pl_debug(ctx, "Available instance extensions:");
    for (int i = 0; i < num_exts_avail; i++)
        pl_debug(ctx, "    %s", exts_avail[i].extensionName);
    for (int i = 0; i < num_layers_avail; i++) {
        for (int j = 0; j < layer_exts[i].num_exts; j++) {
            if (!layer_exts[i].exts[j].extensionName[0])
                continue;

            pl_debug(ctx, "    %s (via %s)",
                     layer_exts[i].exts[j].extensionName,
                     layers_avail[i].layerName);
        }
    }

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
    for (int i = 0; i < params->num_extensions; i++) {
        const char *ext = params->extensions[i];
        TARRAY_APPEND(tmp, exts, num_exts, ext);

        // Enable any additional layers that are required for this extension
        for (int n = 0; n < num_layers_avail; n++) {
            for (int j = 0; j < layer_exts[n].num_exts; j++) {
                if (!layer_exts[n].exts[j].extensionName[0])
                    continue;
                if (strcmp(ext, layer_exts[n].exts[j].extensionName) == 0) {
                    TARRAY_APPEND(tmp, layers, num_layers, layers_avail[n].layerName);
                    goto next_user_ext;
                }
            }
        }

next_user_ext: ;
    }

    // Add extra optional user extensions
    for (int i = 0; i < params->num_opt_extensions; i++) {
        const char *ext = params->opt_extensions[i];
        for (int n = 0; n < num_exts_avail; n++) {
            if (strcmp(ext, exts_avail[n].extensionName) == 0) {
                TARRAY_APPEND(tmp, exts, num_exts, ext);
                goto next_opt_user_ext;
            }
        }

        for (int n = 0; n < num_layers_avail; n++) {
            for (int j = 0; j < layer_exts[n].num_exts; j++) {
                if (!layer_exts[n].exts[j].extensionName[0])
                    continue;
                if (strcmp(ext, layer_exts[n].exts[j].extensionName) == 0) {
                    TARRAY_APPEND(tmp, exts, num_exts, ext);
                    TARRAY_APPEND(tmp, layers, num_layers, layers_avail[n].layerName);
                    goto next_opt_user_ext;
                }
            }
        }

next_opt_user_ext: ;
    }

    info.ppEnabledExtensionNames = exts;
    info.enabledExtensionCount = num_exts;
    info.ppEnabledLayerNames = layers;
    info.enabledLayerCount = num_layers;

    pl_info(ctx, "Creating vulkan instance%s", num_exts ? " with extensions:" : "");
    for (int i = 0; i < num_exts; i++)
        pl_info(ctx, "    %s", exts[i]);

    if (num_layers) {
        pl_info(ctx, "  and layers:");
        for (int i = 0; i < num_layers; i++)
            pl_info(ctx, "    %s", layers[i]);
    }

    VK_LOAD_FUN(NULL, CreateInstance, get_addr);
    VkResult res = CreateInstance(&info, VK_ALLOC, &inst);
    if (res != VK_SUCCESS) {
        pl_fatal(ctx, "Failed creating instance: %s", vk_res_str(res));
        goto error;
    }

    VkDebugReportCallbackEXT debug_cb = VK_NULL_HANDLE;
    if (debug) {
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
        VK_LOAD_FUN(inst, CreateDebugReportCallbackEXT, get_addr)
        CreateDebugReportCallbackEXT(inst, &dinfo, VK_ALLOC, &debug_cb);
    }

    struct pl_vk_inst *pl_vk = talloc_priv(NULL, struct pl_vk_inst, struct priv);
    *pl_vk = (struct pl_vk_inst) {
        .instance = inst,
        .get_proc_addr = get_addr,
        .extensions = exts,
        .num_extensions = num_exts,
        .layers = layers,
        .num_layers = num_layers,
    };

    struct priv *p = TA_PRIV(pl_vk);
    p->debug_cb = debug_cb;

    pl_vk->extensions = talloc_steal(pl_vk, pl_vk->extensions);
    pl_vk->layers = talloc_steal(pl_vk, pl_vk->layers);
    talloc_free(tmp);
    return pl_vk;

error:
    pl_fatal(ctx, "Failed initializing vulkan instance");
    if (inst) {
        VK_LOAD_FUN(inst, DestroyInstance, get_addr);
        DestroyInstance(inst, VK_ALLOC);
    }
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

    struct vk_ctx *vk = TA_PRIV(*pl_vk);
    if (vk->dev) {
        PL_DEBUG(vk, "Flushing remaining commands...");
        vk_wait_idle(vk);
        pl_assert(vk->num_cmds_queued == 0);
        pl_assert(vk->num_cmds_pending == 0);
        for (int i = 0; i < vk->num_pools; i++)
            vk_cmdpool_destroy(vk, vk->pools[i]);
        for (int i = 0; i < vk->num_signals; i++)
            vk_signal_destroy(vk, &vk->signals[i]);
        vk->DestroyDevice(vk->dev, VK_ALLOC);
    }

    pl_vk_inst_destroy(&vk->internal_instance);
    TA_FREEP((void **) pl_vk);
}

static bool supports_surf(struct pl_context *ctx, VkInstance inst,
                          PFN_vkGetInstanceProcAddr get_addr,
                          VkPhysicalDevice physd, VkSurfaceKHR surf)
{
    // Hack for the VK macro's logging to work
    struct { struct pl_context *ctx; } *vk = (void *) &ctx;

    VK_LOAD_FUN(inst, GetPhysicalDeviceQueueFamilyProperties, get_addr);
    VK_LOAD_FUN(inst, GetPhysicalDeviceSurfaceSupportKHR, get_addr);
    uint32_t qfnum;
    GetPhysicalDeviceQueueFamilyProperties(physd, &qfnum, NULL);

    for (int i = 0; i < qfnum; i++) {
        VkBool32 sup;
        VK(GetPhysicalDeviceSurfaceSupportKHR(physd, i, surf, &sup));
        if (sup)
            return true;
    }

error:
    return false;
}

VkPhysicalDevice pl_vulkan_choose_device(struct pl_context *ctx,
                                         const struct pl_vulkan_device_params *params)
{
    // Hack for the VK macro's logging to work
    struct { struct pl_context *ctx; } *vk = (void *) &ctx;
    PL_INFO(vk, "Probing for vulkan devices:");

    pl_assert(params->instance);
    VkInstance inst = params->instance;
    VkPhysicalDevice dev = VK_NULL_HANDLE;

    PFN_vkGetInstanceProcAddr get_addr;
    if (!(get_addr = get_proc_addr_fallback(ctx, params->get_proc_addr)))
        return NULL;

    VK_LOAD_FUN(inst, EnumeratePhysicalDevices, get_addr);
    VK_LOAD_FUN(inst, GetPhysicalDeviceProperties, get_addr);

    VkPhysicalDevice *devices = NULL;
    uint32_t num = 0;
    VK(EnumeratePhysicalDevices(inst, &num, NULL));
    devices = talloc_zero_array(NULL, VkPhysicalDevice, num);
    VK(EnumeratePhysicalDevices(inst, &num, devices));

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
        VkPhysicalDeviceProperties props = {0};
        GetPhysicalDeviceProperties(devices[i], &props);
        VkPhysicalDeviceType t = props.deviceType;
        PL_INFO(vk, "    GPU %d: %s (%s)", i, props.deviceName, types[t].name);

        if (params->surface) {
            if (!supports_surf(ctx, inst, get_addr, devices[i], params->surface)) {
                PL_DEBUG(vk, "      -> excluding due to lack of surface support");
                continue;
            }
        }

        if (params->device_name && params->device_name[0] != '\0') {
            if (strcmp(params->device_name, props.deviceName) == 0) {
                dev = devices[i];
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
            dev = devices[i];
            best = types[t].priority;
        }
    }

error:
    talloc_free(devices);
    return dev;
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
    int qfnum = 0;
    vk->GetPhysicalDeviceQueueFamilyProperties(vk->physd, &qfnum, NULL);
    VkQueueFamilyProperties *qfs = talloc_zero_array(tmp, VkQueueFamilyProperties, qfnum);
    vk->GetPhysicalDeviceQueueFamilyProperties(vk->physd, &qfnum, qfs);

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
        VK(vk->GetPhysicalDeviceSurfaceSupportKHR(vk->physd, idx_gfx,
                                                  params->surface, &sup));
        if (!sup) {
            PL_FATAL(vk, "Queue family does not support surface presentation!");
            goto error;
        }
    }

    // Fall back to supporting compute shaders via the graphics pool for
    // devices which support compute shaders but not async compute.
    if (idx_comp < 0 && qfs[idx_gfx].queueFlags & VK_QUEUE_COMPUTE_BIT)
        idx_comp = idx_gfx;

    if (params->blacklist_caps & PL_GPU_CAP_COMPUTE) {
        PL_INFO(vk, "Disabling compute shaders (blacklisted)");
        idx_comp = -1;
    }

    if (idx_tf >= 0 && idx_tf != idx_gfx)
        PL_INFO(vk, "Using async transfer (QF %d)", idx_tf);
    if (idx_comp >= 0 && idx_comp != idx_gfx)
        PL_INFO(vk, "Using async compute (QF %d)", idx_comp);

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
    VK(vk->EnumerateDeviceExtensionProperties(vk->physd, NULL, &num_exts_avail, NULL));
    VkExtensionProperties *exts_avail = talloc_zero_array(tmp, VkExtensionProperties, num_exts_avail);
    VK(vk->EnumerateDeviceExtensionProperties(vk->physd, NULL, &num_exts_avail, exts_avail));

    PL_DEBUG(vk, "Available device extensions:");
    for (int i = 0; i < num_exts_avail; i++)
        PL_DEBUG(vk, "    %s", exts_avail[i].extensionName);

    // Add all extensions we need
    const char ***exts = &vk->exts;
    int *num_exts = &vk->num_exts;
    if (params->surface)
        TARRAY_APPEND(vk->ta, *exts, *num_exts, VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    // Keep track of all optional function pointers associated with extensions
    const struct vk_fun **ext_funs = NULL;
    int num_ext_funs = 0;

    // Add all optional device-level extensions extensions
    for (int i = 0; i < PL_ARRAY_SIZE(vk_device_extensions); i++) {
        const struct vk_ext *ext = &vk_device_extensions[i];
        for (int n = 0; n < num_exts_avail; n++) {
            if (strcmp(ext->name, exts_avail[n].extensionName) == 0) {
                TARRAY_APPEND(vk->ta, *exts, *num_exts, ext->name);
                for (const struct vk_fun *f = ext->funs; f->name; f++)
                    TARRAY_APPEND(tmp, ext_funs, num_ext_funs, f);
                break;
            }
        }
    }

    // Add extra user extensions
    for (int i = 0; i < params->num_extensions; i++)
        TARRAY_APPEND(vk->ta, *exts, *num_exts, params->extensions[i]);

    // Add optional extra user extensions
    for (int i = 0; i < params->num_opt_extensions; i++) {
        const char *ext = params->opt_extensions[i];
        for (int n = 0; n < num_exts_avail; n++) {
            if (strcmp(ext, exts_avail[n].extensionName) == 0) {
                TARRAY_APPEND(vk->ta, *exts, *num_exts, ext);
                break;
            }
        }
    }

    // Enable all features that we might need (whitelisted)
    vk->GetPhysicalDeviceFeatures(vk->physd, &vk->features);
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

    VK(vk->CreateDevice(vk->physd, &dinfo, VK_ALLOC, &vk->dev));

    // Load all mandatory device-level functions
    for (int i = 0; i < PL_ARRAY_SIZE(vk_dev_funs); i++) {
        const struct vk_fun *fun = &vk_dev_funs[i];
        PFN_vkVoidFunction *pfn = (void *) ((uintptr_t) vk + (ptrdiff_t) fun->offset);
        pl_assert(fun->device_level);
        *pfn = vk->GetDeviceProcAddr(vk->dev, fun->name);
    }

    // Load all of the optional functions from the extensions we enabled
    for (int i = 0; i < num_ext_funs; i++) {
        const struct vk_fun *fun = ext_funs[i];
        PFN_vkVoidFunction *pfn = (void *) ((uintptr_t) vk + (ptrdiff_t) fun->offset);
        if (fun->device_level) {
            *pfn = vk->GetDeviceProcAddr(vk->dev, fun->name);
        } else {
            *pfn = vk->GetInstanceProcAddr(vk->inst, fun->name);
        };
    }

    // Create the command pools and memory allocator
    for (int i = 0; i < num_qinfos; i++) {
        int qf = qinfos[i].queueFamilyIndex;
        struct vk_cmdpool *pool = vk_cmdpool_create(vk, qinfos[i], qfs[qf]);
        if (!pool)
            goto error;
        TARRAY_APPEND(vk->ta, vk->pools, vk->num_pools, pool);

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
    vk->failed = true;
    return false;
}

const struct pl_vulkan *pl_vulkan_create(struct pl_context *ctx,
                                         const struct pl_vulkan_params *params)
{
    params = PL_DEF(params, &pl_vulkan_default_params);
    struct pl_vulkan *pl_vk = talloc_zero_priv(NULL, struct pl_vulkan, struct vk_ctx);
    struct vk_ctx *vk = TA_PRIV(pl_vk);
    vk->ta = pl_vk;
    vk->ctx = ctx;
    vk->inst = params->instance;

    if (!(vk->GetInstanceProcAddr = get_proc_addr_fallback(ctx, params->get_proc_addr)))
        goto error;

    if (!vk->inst) {
        pl_assert(!params->surface);
        pl_assert(!params->device);
        PL_DEBUG(vk, "No VkInstance provided, creating one...");

        // Mirror the instance params here to set `get_proc_addr` correctly
        struct pl_vk_inst_params iparams;
        iparams = *PL_DEF(params->instance_params, &pl_vk_inst_default_params);
        iparams.get_proc_addr = params->get_proc_addr;
        vk->internal_instance = pl_vk_inst_create(ctx, &iparams);
        if (!vk->internal_instance)
            goto error;
        vk->inst = vk->internal_instance->instance;
    }

    // Directly load all mandatory instance-level function pointers, since
    // these will be required for all further device creation logic
    for (int i = 0; i < PL_ARRAY_SIZE(vk_inst_funs); i++) {
        const struct vk_fun *fun = &vk_inst_funs[i];
        PFN_vkVoidFunction *pfn = (void *) ((uintptr_t) vk + (ptrdiff_t) fun->offset);
        pl_assert(!fun->device_level);
        *pfn = vk->GetInstanceProcAddr(vk->inst, fun->name);
    }

    // Choose the physical device
    if (params->device) {
        PL_DEBUG(vk, "Using specified VkPhysicalDevice");
        vk->physd = params->device;
    } else {
        vk->physd = pl_vulkan_choose_device(ctx, &(struct pl_vulkan_device_params) {
            .instance       = vk->inst,
            .surface        = params->surface,
            .device_name    = params->device_name,
            .allow_software = params->allow_software,
        });

        if (!vk->physd) {
            PL_FATAL(vk, "Found no suitable device, giving up.");
            goto error;
        }
    }

    VkPhysicalDeviceProperties prop = {0};
    vk->GetPhysicalDeviceProperties(vk->physd, &prop);
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

    // Blacklist / restrict features
    if (params->blacklist_caps) {
        pl_gpu_caps *caps = (pl_gpu_caps*) &pl_vk->gpu->caps;
        *caps &= ~(params->blacklist_caps);
        PL_INFO(vk, "Restricting capabilities 0x%x... new caps are 0x%x",
                (unsigned int) params->blacklist_caps, (unsigned int) *caps);
    }

    if (params->max_glsl_version) {
        struct pl_glsl_desc *desc = (struct pl_glsl_desc *) &pl_vk->gpu->glsl;
        desc->version = PL_MIN(desc->version, params->max_glsl_version);
        PL_INFO(vk, "Restricting GLSL version to %d... new version is %d",
                params->max_glsl_version, desc->version);
    }

    vk->disable_events = params->disable_events;

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
    vk->failed = true;
    return NULL;
}
