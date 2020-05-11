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

#ifdef __unix__
#define VK_HAVE_UNIX 1
#endif

#ifdef _WIN32
#define VK_HAVE_WIN32 1
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#endif

// Vulkan allows the optional use of a custom allocator. We don't need one but
// mark this parameter with a better name in case we ever decide to change this
// in the future. (And to make the code more readable)
#define VK_ALLOC NULL

// Type of a vulkan function that needs to be loaded
#define VK_FUN(name) PFN_vk##name name

// Load a vulkan instance-level extension function directly (on the stack)
#define VK_LOAD_FUN(inst, name, get_addr) \
    VK_FUN(name) = (PFN_vk##name) get_addr(inst, "vk" #name);

// Hard-coded limit on the number of pending commands, to avoid OOM loops
#define PL_VK_MAX_QUEUED_CMDS 1024
#define PL_VK_MAX_PENDING_CMDS 1024

// Shared struct used to hold vulkan context information
struct vk_ctx {
    void *ta; // allocations bound to the lifetime of this vk_ctx
    const struct pl_vk_inst *internal_instance;
    struct pl_context *ctx;
    VkInstance inst;
    VkPhysicalDevice physd;
    VkPhysicalDeviceLimits limits;
    VkPhysicalDeviceFeatures features;
    VkExtent3D transfer_alignment; // for pool_transfer
    uint32_t api_ver; // device API version
    VkDevice dev;

    // Generic error flag for catching "failed" devices
    bool failed;

    // Enabled extensions
    const char **exts;
    int num_exts;

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
    // yet completed. Used to implement vk_dev_callback. Gets cleared when
    // the command completes.
    struct vk_cmd *last_cmd;

    // Common pool of signals, to avoid having to re-create these objects often
    struct vk_signal **signals;
    int num_signals;
    bool disable_events;

    // Instance-level function pointers
    VK_FUN(CreateDevice);
    VK_FUN(EnumerateDeviceExtensionProperties);
    VK_FUN(GetDeviceProcAddr);
    VK_FUN(GetInstanceProcAddr);
    VK_FUN(GetPhysicalDeviceExternalBufferPropertiesKHR);
    VK_FUN(GetPhysicalDeviceExternalSemaphorePropertiesKHR);
    VK_FUN(GetPhysicalDeviceFeatures);
    VK_FUN(GetPhysicalDeviceFormatProperties);
    VK_FUN(GetPhysicalDeviceImageFormatProperties2KHR);
    VK_FUN(GetPhysicalDeviceMemoryProperties);
    VK_FUN(GetPhysicalDeviceProperties);
    VK_FUN(GetPhysicalDeviceProperties2KHR);
    VK_FUN(GetPhysicalDeviceQueueFamilyProperties);
    VK_FUN(GetPhysicalDeviceSurfaceCapabilitiesKHR);
    VK_FUN(GetPhysicalDeviceSurfaceFormatsKHR);
    VK_FUN(GetPhysicalDeviceSurfacePresentModesKHR);
    VK_FUN(GetPhysicalDeviceSurfaceSupportKHR);

    // Device-level function pointers
    VK_FUN(AcquireNextImageKHR);
    VK_FUN(AllocateCommandBuffers);
    VK_FUN(AllocateDescriptorSets);
    VK_FUN(AllocateMemory);
    VK_FUN(BeginCommandBuffer);
    VK_FUN(BindBufferMemory);
    VK_FUN(BindImageMemory);
    VK_FUN(CmdBeginDebugUtilsLabelEXT);
    VK_FUN(CmdBeginRenderPass);
    VK_FUN(CmdBindDescriptorSets);
    VK_FUN(CmdBindPipeline);
    VK_FUN(CmdBindVertexBuffers);
    VK_FUN(CmdBlitImage);
    VK_FUN(CmdClearColorImage);
    VK_FUN(CmdCopyBuffer);
    VK_FUN(CmdCopyBufferToImage);
    VK_FUN(CmdCopyImage);
    VK_FUN(CmdCopyImageToBuffer);
    VK_FUN(CmdDispatch);
    VK_FUN(CmdDraw);
    VK_FUN(CmdEndDebugUtilsLabelEXT);
    VK_FUN(CmdEndRenderPass);
    VK_FUN(CmdPipelineBarrier);
    VK_FUN(CmdPushConstants);
    VK_FUN(CmdPushDescriptorSetKHR);
    VK_FUN(CmdSetEvent);
    VK_FUN(CmdSetScissor);
    VK_FUN(CmdSetViewport);
    VK_FUN(CmdUpdateBuffer);
    VK_FUN(CmdWaitEvents);
    VK_FUN(CreateBuffer);
    VK_FUN(CreateBufferView);
    VK_FUN(CreateCommandPool);
    VK_FUN(CreateComputePipelines);
    VK_FUN(CreateDebugReportCallbackEXT);
    VK_FUN(CreateDescriptorPool);
    VK_FUN(CreateDescriptorSetLayout);
    VK_FUN(CreateEvent);
    VK_FUN(CreateFence);
    VK_FUN(CreateFramebuffer);
    VK_FUN(CreateGraphicsPipelines);
    VK_FUN(CreateImage);
    VK_FUN(CreateImageView);
    VK_FUN(CreatePipelineCache);
    VK_FUN(CreatePipelineLayout);
    VK_FUN(CreateRenderPass);
    VK_FUN(CreateSampler);
    VK_FUN(CreateSemaphore);
    VK_FUN(CreateShaderModule);
    VK_FUN(CreateSwapchainKHR);
    VK_FUN(DestroyBuffer);
    VK_FUN(DestroyBufferView);
    VK_FUN(DestroyCommandPool);
    VK_FUN(DestroyDebugReportCallbackEXT);
    VK_FUN(DestroyDescriptorPool);
    VK_FUN(DestroyDescriptorSetLayout);
    VK_FUN(DestroyDevice);
    VK_FUN(DestroyEvent);
    VK_FUN(DestroyFence);
    VK_FUN(DestroyFramebuffer);
    VK_FUN(DestroyImage);
    VK_FUN(DestroyImageView);
    VK_FUN(DestroyInstance);
    VK_FUN(DestroyPipeline);
    VK_FUN(DestroyPipelineCache);
    VK_FUN(DestroyPipelineLayout);
    VK_FUN(DestroyRenderPass);
    VK_FUN(DestroySampler);
    VK_FUN(DestroySemaphore);
    VK_FUN(DestroyShaderModule);
    VK_FUN(DestroySwapchainKHR);
    VK_FUN(EndCommandBuffer);
    VK_FUN(FlushMappedMemoryRanges);
    VK_FUN(FreeCommandBuffers);
    VK_FUN(FreeMemory);
    VK_FUN(GetBufferMemoryRequirements);
    VK_FUN(GetDeviceQueue);
    VK_FUN(GetImageMemoryRequirements);
    VK_FUN(GetMemoryFdKHR);
    VK_FUN(GetMemoryFdPropertiesKHR);
    VK_FUN(GetPipelineCacheData);
    VK_FUN(GetSemaphoreFdKHR);
    VK_FUN(GetSwapchainImagesKHR);
    VK_FUN(InvalidateMappedMemoryRanges);
    VK_FUN(MapMemory);
    VK_FUN(QueuePresentKHR);
    VK_FUN(QueueSubmit);
    VK_FUN(ResetEvent);
    VK_FUN(ResetFences);
    VK_FUN(SetDebugUtilsObjectNameEXT);
    VK_FUN(SetHdrMetadataEXT);
    VK_FUN(UpdateDescriptorSets);
    VK_FUN(WaitForFences);

#ifdef VK_HAVE_WIN32
    VK_FUN(GetMemoryWin32HandleKHR);
    VK_FUN(GetSemaphoreWin32HandleKHR);
#endif
};
