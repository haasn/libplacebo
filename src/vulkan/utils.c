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

#include "utils.h"

const char *vk_res_str(VkResult res)
{
    switch (res) {
#define CASE(name) case name: return #name
    // success codes
    CASE(VK_SUCCESS);
    CASE(VK_NOT_READY);
    CASE(VK_TIMEOUT);
    CASE(VK_EVENT_SET);
    CASE(VK_EVENT_RESET);
    CASE(VK_INCOMPLETE);

    // error codes
    CASE(VK_ERROR_OUT_OF_HOST_MEMORY);
    CASE(VK_ERROR_OUT_OF_DEVICE_MEMORY);
    CASE(VK_ERROR_INITIALIZATION_FAILED);
    CASE(VK_ERROR_DEVICE_LOST);
    CASE(VK_ERROR_MEMORY_MAP_FAILED);
    CASE(VK_ERROR_LAYER_NOT_PRESENT);
    CASE(VK_ERROR_EXTENSION_NOT_PRESENT);
    CASE(VK_ERROR_FEATURE_NOT_PRESENT);
    CASE(VK_ERROR_INCOMPATIBLE_DRIVER);
    CASE(VK_ERROR_TOO_MANY_OBJECTS);
    CASE(VK_ERROR_FORMAT_NOT_SUPPORTED);
    CASE(VK_ERROR_FRAGMENTED_POOL);

    // Symbols introduced by extensions (explicitly guarded against so we can
    // make this switch exhaustive without requiring bleeding edge versions
    // of vulkan.h)
#ifdef VK_KHR_maintenance1
    CASE(VK_ERROR_OUT_OF_POOL_MEMORY_KHR);
#endif
#ifdef VK_KHR_external_memory
    CASE(VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR);
#endif
#ifdef VK_KHR_surface
    CASE(VK_ERROR_SURFACE_LOST_KHR);
    CASE(VK_ERROR_NATIVE_WINDOW_IN_USE_KHR);
#endif
#ifdef VK_KHR_swapchain
    CASE(VK_SUBOPTIMAL_KHR);
    CASE(VK_ERROR_OUT_OF_DATE_KHR);
#endif
#ifdef VK_KHR_display_swapchain
    CASE(VK_ERROR_INCOMPATIBLE_DISPLAY_KHR);
#endif
#ifdef VK_EXT_debug_report
    CASE(VK_ERROR_VALIDATION_FAILED_EXT);
#endif
#undef CASE

    default: return "unknown error";
    }
}

const char *vk_obj_str(VkDebugReportObjectTypeEXT obj)
{
    switch (obj) {
#define CASE(name, str) case VK_DEBUG_REPORT_OBJECT_TYPE_##name##_EXT: return #str
    CASE(INSTANCE,                          VkInstance);
    CASE(PHYSICAL_DEVICE,                   VkPhysicalDevice);
    CASE(DEVICE,                            VkDevice);
    CASE(QUEUE,                             VkQueue);
    CASE(SEMAPHORE,                         VkSemaphore);
    CASE(COMMAND_BUFFER,                    VkCommandBuffer);
    CASE(FENCE,                             VkFence);
    CASE(DEVICE_MEMORY,                     VkDeviceMemory);
    CASE(BUFFER,                            VkBuffer);
    CASE(IMAGE,                             VkImage);
    CASE(EVENT,                             VkEvent);
    CASE(QUERY_POOL,                        VkQueryPool);
    CASE(BUFFER_VIEW,                       VkBufferView);
    CASE(IMAGE_VIEW,                        VkImageView);
    CASE(SHADER_MODULE,                     VkShaderModule);
    CASE(PIPELINE_CACHE,                    VkPipelineCache);
    CASE(PIPELINE_LAYOUT,                   VkPipelineLayout);
    CASE(RENDER_PASS,                       VkRenderPass);
    CASE(PIPELINE,                          VkPipeline);
    CASE(DESCRIPTOR_SET_LAYOUT,             VkDescriptorSetLayout);
    CASE(SAMPLER,                           VkSampler);
    CASE(DESCRIPTOR_POOL,                   VkDescriptorPool);
    CASE(DESCRIPTOR_SET,                    VkDescriptorSet);
    CASE(FRAMEBUFFER,                       VkFramebuffer);
    CASE(COMMAND_POOL,                      VkCommandPool);

    // Objects introduced by extensions
#ifdef VK_KHR_surface
    CASE(SURFACE_KHR,                       VkSurfaceKHR);
#endif
#ifdef VK_KHR_swapchain
    CASE(SWAPCHAIN_KHR,                     VkSwapchainKHR);
#endif
#ifdef VK_KHR_display
    CASE(DISPLAY_KHR,                       VkDisplayKHR);
    CASE(DISPLAY_MODE_KHR,                  VkDisplayModeKHR);
#endif
#ifdef VK_EXT_debug_report
    CASE(DEBUG_REPORT,                      VkDebugReportCallbackEXT);
#endif
#undef CASE

    default: return "unknown object";
    }
}

VkExternalMemoryHandleTypeFlagBitsKHR
vk_mem_handle_type(enum pl_handle_type handle_type)
{
    if (!handle_type)
        return 0;

    switch (handle_type) {
    case PL_HANDLE_FD:
        return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
    case PL_HANDLE_WIN32:
        return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
    case PL_HANDLE_WIN32_KMT:
        return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR;
    case PL_HANDLE_DMA_BUF:
        return VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    }

    abort();
}

VkExternalSemaphoreHandleTypeFlagBitsKHR
vk_sync_handle_type(enum pl_handle_type handle_type)
{
    if (!handle_type)
        return 0;

    switch (handle_type) {
    case PL_HANDLE_FD:
        return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
    case PL_HANDLE_WIN32:
        return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
    case PL_HANDLE_WIN32_KMT:
        return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR;
    case PL_HANDLE_DMA_BUF: abort();
    }

    abort();
}

bool vk_external_mem_check(const VkExternalMemoryPropertiesKHR *props,
                           enum pl_handle_type handle_type,
                           bool import)
{
    VkExternalMemoryFeatureFlagsKHR flags = props->externalMemoryFeatures;

    // No support for this handle type;
    if (!(props->compatibleHandleTypes & vk_mem_handle_type(handle_type)))
        return false;

    if (import) {
        if (!(flags & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT_KHR))
            return false;
    } else {
        if (!(flags & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT_KHR))
            return false;
    }

    // We can't handle VkMemoryDedicatedAllocateInfo currently. (Maybe soon?)
    if (flags & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT_KHR)
        return false;

    return true;
}

const enum pl_handle_type vk_mem_handle_list[] = {
#ifdef VK_HAVE_UNIX
        PL_HANDLE_FD,
        PL_HANDLE_DMA_BUF,
#endif
#ifdef VK_HAVE_WIN32
        PL_HANDLE_WIN32,
        PL_HANDLE_WIN32_KMT,
#endif
        0
};

const enum pl_handle_type vk_sync_handle_list[] = {
#ifdef VK_HAVE_UNIX
        PL_HANDLE_FD,
#endif
#ifdef VK_HAVE_WIN32
        PL_HANDLE_WIN32,
        PL_HANDLE_WIN32_KMT,
#endif
        0
};
