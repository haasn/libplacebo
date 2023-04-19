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

#define VK_NO_PROTOTYPES
#define VK_ENABLE_BETA_EXTENSIONS // for VK_KHR_portability_subset
#define VK_USE_PLATFORM_METAL_EXT

#include "../common.h"
#include "../log.h"
#include "../pl_thread.h"

#include <libplacebo/vulkan.h>

#ifdef PL_HAVE_WIN32
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#endif

// Vulkan allows the optional use of a custom allocator. We don't need one but
// mark this parameter with a better name in case we ever decide to change this
// in the future. (And to make the code more readable)
#define PL_VK_ALLOC NULL

// Type of a vulkan function that needs to be loaded
#define PL_VK_FUN(name) PFN_vk##name name

// Load a vulkan instance-level extension function directly (on the stack)
#define PL_VK_LOAD_FUN(inst, name, get_addr) \
    PL_VK_FUN(name) = (PFN_vk##name) get_addr(inst, "vk" #name);

#ifndef VK_VENDOR_ID_NVIDIA
#define VK_VENDOR_ID_NVIDIA 0x10DE
#endif

// Shared struct used to hold vulkan context information
struct vk_ctx {
    pl_mutex lock;
    pl_vulkan vulkan;
    void *alloc; // host allocations bound to the lifetime of this vk_ctx
    struct vk_malloc *ma; // VRAM malloc layer
    pl_vk_inst internal_instance;
    pl_log log;
    VkInstance inst;
    VkPhysicalDevice physd;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceFeatures2 features;
    uint32_t api_ver; // device API version
    VkDevice dev;
    bool imported; // device was not created by us

    // Generic error flag for catching "failed" devices
    bool failed;

    // Enabled extensions
    PL_ARRAY(const char *) exts;

    // Command pools (one per queue family)
    PL_ARRAY(struct vk_cmdpool *) pools;

    // Pointers into `pools` (always set)
    struct vk_cmdpool *pool_graphics;
    struct vk_cmdpool *pool_compute;
    struct vk_cmdpool *pool_transfer;

    // Queue locking functions
    PL_ARRAY(PL_ARRAY(pl_mutex)) queue_locks;
    void (*lock_queue)(void *queue_ctx, int qf, int idx);
    void (*unlock_queue)(void *queue_ctx, int qf, int idx);
    void *queue_ctx;

    // Pending commands. These are shared for the entire mpvk_ctx to ensure
    // submission and callbacks are FIFO
    PL_ARRAY(struct vk_cmd *) cmds_pending; // submitted but not completed

    // Pending callbacks that still need to be drained before processing
    // callbacks for the next command (in case commands are recursively being
    // polled from another callback)
    const struct vk_callback *pending_callbacks;
    int num_pending_callbacks;

    // Instance-level function pointers
    PL_VK_FUN(CreateDevice);
    PL_VK_FUN(EnumerateDeviceExtensionProperties);
    PL_VK_FUN(GetDeviceProcAddr);
    PL_VK_FUN(GetInstanceProcAddr);
    PL_VK_FUN(GetPhysicalDeviceExternalBufferProperties);
    PL_VK_FUN(GetPhysicalDeviceExternalSemaphoreProperties);
    PL_VK_FUN(GetPhysicalDeviceFeatures2KHR);
    PL_VK_FUN(GetPhysicalDeviceFormatProperties);
    PL_VK_FUN(GetPhysicalDeviceFormatProperties2KHR);
    PL_VK_FUN(GetPhysicalDeviceImageFormatProperties2KHR);
    PL_VK_FUN(GetPhysicalDeviceMemoryProperties);
    PL_VK_FUN(GetPhysicalDeviceProperties);
    PL_VK_FUN(GetPhysicalDeviceProperties2);
    PL_VK_FUN(GetPhysicalDeviceQueueFamilyProperties);
    PL_VK_FUN(GetPhysicalDeviceSurfaceCapabilitiesKHR);
    PL_VK_FUN(GetPhysicalDeviceSurfaceFormatsKHR);
    PL_VK_FUN(GetPhysicalDeviceSurfacePresentModesKHR);
    PL_VK_FUN(GetPhysicalDeviceSurfaceSupportKHR);

    // Device-level function pointers
    PL_VK_FUN(AcquireNextImageKHR);
    PL_VK_FUN(AllocateCommandBuffers);
    PL_VK_FUN(AllocateDescriptorSets);
    PL_VK_FUN(AllocateMemory);
    PL_VK_FUN(BeginCommandBuffer);
    PL_VK_FUN(BindBufferMemory);
    PL_VK_FUN(BindImageMemory);
    PL_VK_FUN(CmdBeginDebugUtilsLabelEXT);
    PL_VK_FUN(CmdBeginRenderPass);
    PL_VK_FUN(CmdBindDescriptorSets);
    PL_VK_FUN(CmdBindIndexBuffer);
    PL_VK_FUN(CmdBindPipeline);
    PL_VK_FUN(CmdBindVertexBuffers);
    PL_VK_FUN(CmdBlitImage);
    PL_VK_FUN(CmdClearColorImage);
    PL_VK_FUN(CmdCopyBuffer);
    PL_VK_FUN(CmdCopyBufferToImage);
    PL_VK_FUN(CmdCopyImage);
    PL_VK_FUN(CmdCopyImageToBuffer);
    PL_VK_FUN(CmdDispatch);
    PL_VK_FUN(CmdDraw);
    PL_VK_FUN(CmdDrawIndexed);
    PL_VK_FUN(CmdEndDebugUtilsLabelEXT);
    PL_VK_FUN(CmdEndRenderPass);
    PL_VK_FUN(CmdPipelineBarrier);
    PL_VK_FUN(CmdPushConstants);
    PL_VK_FUN(CmdPushDescriptorSetKHR);
    PL_VK_FUN(CmdResetQueryPool);
    PL_VK_FUN(CmdSetEvent);
    PL_VK_FUN(CmdSetScissor);
    PL_VK_FUN(CmdSetViewport);
    PL_VK_FUN(CmdUpdateBuffer);
    PL_VK_FUN(CmdWaitEvents);
    PL_VK_FUN(CmdWriteTimestamp);
    PL_VK_FUN(CreateBuffer);
    PL_VK_FUN(CreateBufferView);
    PL_VK_FUN(CreateCommandPool);
    PL_VK_FUN(CreateComputePipelines);
    PL_VK_FUN(CreateDebugReportCallbackEXT);
    PL_VK_FUN(CreateDescriptorPool);
    PL_VK_FUN(CreateDescriptorSetLayout);
    PL_VK_FUN(CreateEvent);
    PL_VK_FUN(CreateFence);
    PL_VK_FUN(CreateFramebuffer);
    PL_VK_FUN(CreateGraphicsPipelines);
    PL_VK_FUN(CreateImage);
    PL_VK_FUN(CreateImageView);
    PL_VK_FUN(CreatePipelineCache);
    PL_VK_FUN(CreatePipelineLayout);
    PL_VK_FUN(CreateQueryPool);
    PL_VK_FUN(CreateRenderPass);
    PL_VK_FUN(CreateSampler);
    PL_VK_FUN(CreateSemaphore);
    PL_VK_FUN(CreateShaderModule);
    PL_VK_FUN(CreateSwapchainKHR);
    PL_VK_FUN(DestroyBuffer);
    PL_VK_FUN(DestroyBufferView);
    PL_VK_FUN(DestroyCommandPool);
    PL_VK_FUN(DestroyDebugReportCallbackEXT);
    PL_VK_FUN(DestroyDescriptorPool);
    PL_VK_FUN(DestroyDescriptorSetLayout);
    PL_VK_FUN(DestroyDevice);
    PL_VK_FUN(DestroyEvent);
    PL_VK_FUN(DestroyFence);
    PL_VK_FUN(DestroyFramebuffer);
    PL_VK_FUN(DestroyImage);
    PL_VK_FUN(DestroyImageView);
    PL_VK_FUN(DestroyInstance);
    PL_VK_FUN(DestroyPipeline);
    PL_VK_FUN(DestroyPipelineCache);
    PL_VK_FUN(DestroyPipelineLayout);
    PL_VK_FUN(DestroyQueryPool);
    PL_VK_FUN(DestroyRenderPass);
    PL_VK_FUN(DestroySampler);
    PL_VK_FUN(DestroySemaphore);
    PL_VK_FUN(DestroyShaderModule);
    PL_VK_FUN(DestroySwapchainKHR);
    PL_VK_FUN(DeviceWaitIdle);
    PL_VK_FUN(EndCommandBuffer);
    PL_VK_FUN(FlushMappedMemoryRanges);
    PL_VK_FUN(FreeCommandBuffers);
    PL_VK_FUN(FreeMemory);
    PL_VK_FUN(GetBufferMemoryRequirements);
    PL_VK_FUN(GetDeviceQueue);
    PL_VK_FUN(GetImageDrmFormatModifierPropertiesEXT);
    PL_VK_FUN(GetImageMemoryRequirements2);
    PL_VK_FUN(GetImageSubresourceLayout);
    PL_VK_FUN(GetMemoryFdKHR);
    PL_VK_FUN(GetMemoryFdPropertiesKHR);
    PL_VK_FUN(GetMemoryHostPointerPropertiesEXT);
    PL_VK_FUN(GetPipelineCacheData);
    PL_VK_FUN(GetQueryPoolResults);
    PL_VK_FUN(GetSemaphoreFdKHR);
    PL_VK_FUN(GetSwapchainImagesKHR);
    PL_VK_FUN(InvalidateMappedMemoryRanges);
    PL_VK_FUN(MapMemory);
    PL_VK_FUN(QueuePresentKHR);
    PL_VK_FUN(QueueSubmit);
    PL_VK_FUN(QueueWaitIdle);
    PL_VK_FUN(ResetEvent);
    PL_VK_FUN(ResetFences);
    PL_VK_FUN(ResetQueryPoolEXT);
    PL_VK_FUN(SetDebugUtilsObjectNameEXT);
    PL_VK_FUN(SetHdrMetadataEXT);
    PL_VK_FUN(UpdateDescriptorSets);
    PL_VK_FUN(WaitForFences);
    PL_VK_FUN(WaitSemaphoresKHR);

#ifdef PL_HAVE_WIN32
    PL_VK_FUN(GetMemoryWin32HandleKHR);
    PL_VK_FUN(GetSemaphoreWin32HandleKHR);
#endif

#ifdef VK_EXT_metal_objects
    PL_VK_FUN(ExportMetalObjectsEXT);
#endif
#ifdef VK_EXT_full_screen_exclusive
    PL_VK_FUN(AcquireFullScreenExclusiveModeEXT);
#endif
};
