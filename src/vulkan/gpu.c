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

#include "gpu.h"
#include "formats.h"
#include "glsl/spirv.h"

#ifdef PL_HAVE_UNIX
#include <unistd.h>
#endif

// Gives us enough queries for 8 results
#define QUERY_POOL_SIZE 16

struct pl_timer_t {
    VkQueryPool qpool; // even=start, odd=stop
    int index_write; // next index to write to
    int index_read; // next index to read from
    uint64_t pending; // bitmask of queries that are still running
};

static inline uint64_t timer_bit(int index)
{
    return 1llu << (index / 2);
}

static void timer_destroy_cb(pl_gpu gpu, pl_timer timer)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    pl_assert(!timer->pending);
    vk->DestroyQueryPool(vk->dev, timer->qpool, PL_VK_ALLOC);
    pl_free(timer);
}

static pl_timer vk_timer_create(pl_gpu gpu)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    pl_timer timer = pl_alloc_ptr(NULL, timer);
    *timer = (struct pl_timer_t) {0};

    struct VkQueryPoolCreateInfo qinfo = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = QUERY_POOL_SIZE,
    };

    VK(vk->CreateQueryPool(vk->dev, &qinfo, PL_VK_ALLOC, &timer->qpool));
    return timer;

error:
    timer_destroy_cb(gpu, timer);
    return NULL;
}

static void vk_timer_destroy(pl_gpu gpu, pl_timer timer)
{
    vk_gpu_idle_callback(gpu, (vk_cb) timer_destroy_cb, gpu, timer);
}

static uint64_t vk_timer_query(pl_gpu gpu, pl_timer timer)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    if (timer->index_read == timer->index_write)
        return 0; // no more unprocessed results

    vk_poll_commands(vk, 0);
    if (timer->pending & timer_bit(timer->index_read))
        return 0; // still waiting for results

    VkResult res;
    uint64_t ts[2] = {0};
    res = vk->GetQueryPoolResults(vk->dev, timer->qpool, timer->index_read, 2,
                                  sizeof(ts), &ts[0], sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT);

    switch (res) {
    case VK_SUCCESS:
        timer->index_read = (timer->index_read + 2) % QUERY_POOL_SIZE;
        return (ts[1] - ts[0]) * vk->props.limits.timestampPeriod;
    case VK_NOT_READY:
        return 0;
    default:
        PL_VK_ASSERT(res, "Retrieving query pool results");
    }

error:
    return 0;
}

static void timer_begin(pl_gpu gpu, struct vk_cmd *cmd, pl_timer timer)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    if (!timer)
        return;

    if (!cmd->pool->props.timestampValidBits) {
        PL_TRACE(gpu, "QF %d does not support timestamp queries", cmd->pool->qf);
        return;
    }

    vk_poll_commands(vk, 0);
    if (timer->pending & timer_bit(timer->index_write))
        return; // next query is still running, skip this timer

    VkQueueFlags reset_flags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
    if (cmd->pool->props.queueFlags & reset_flags) {
        // Use direct command buffer resets
        vk->CmdResetQueryPool(cmd->buf, timer->qpool, timer->index_write, 2);
    } else {
        // Use host query reset
        vk->ResetQueryPool(vk->dev, timer->qpool, timer->index_write, 2);
    }

    vk->CmdWriteTimestamp(cmd->buf, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                          timer->qpool, timer->index_write);

    p->cmd_timer = timer;
}

static inline bool supports_marks(struct vk_cmd *cmd) {
    // Spec says debug markers are only available on graphics/compute queues
    VkQueueFlags flags = cmd->pool->props.queueFlags;
    return flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);
}

struct vk_cmd *_begin_cmd(pl_gpu gpu, enum queue_type type, const char *label,
                          pl_timer timer)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    pl_mutex_lock(&p->recording);

    struct vk_cmdpool *pool;
    switch (type) {
    case ANY:      pool = p->cmd ? p->cmd->pool : vk->pool_graphics; break;
    case GRAPHICS: pool = vk->pool_graphics; break;
    case COMPUTE:  pool = vk->pool_compute;  break;
    case TRANSFER: pool = vk->pool_transfer; break;
    default: pl_unreachable();
    }

    if (!p->cmd || p->cmd->pool != pool) {
        vk_cmd_submit(&p->cmd);
        p->cmd = vk_cmd_begin(pool, label);
        if (!p->cmd) {
            pl_mutex_unlock(&p->recording);
            return NULL;
        }
    }

    if (vk->CmdBeginDebugUtilsLabelEXT && supports_marks(p->cmd)) {
        vk->CmdBeginDebugUtilsLabelEXT(p->cmd->buf, &(VkDebugUtilsLabelEXT) {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
            .pLabelName = label,
        });
    }

    timer_begin(gpu, p->cmd, timer);
    return p->cmd;
}

static void timer_end_cb(void *ptimer, void *pindex)
{
    pl_timer timer = ptimer;
    int index = (uintptr_t) pindex;
    timer->pending &= ~timer_bit(index);
}

bool _end_cmd(pl_gpu gpu, struct vk_cmd **pcmd, bool submit)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    bool ret = true;
    if (!pcmd) {
        if (submit) {
            pl_mutex_lock(&p->recording);
            ret = vk_cmd_submit(&p->cmd);
            pl_mutex_unlock(&p->recording);
        }
        return ret;
    }

    struct vk_cmd *cmd = *pcmd;
    pl_assert(p->cmd == cmd);

    if (p->cmd_timer) {
        pl_timer timer = p->cmd_timer;
        vk->CmdWriteTimestamp(cmd->buf, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                              timer->qpool, timer->index_write + 1);

        timer->pending |= timer_bit(timer->index_write);
        vk_cmd_callback(cmd, (vk_cb) timer_end_cb, timer,
                        (void *) (uintptr_t) timer->index_write);

        timer->index_write = (timer->index_write + 2) % QUERY_POOL_SIZE;
        if (timer->index_write == timer->index_read) {
            // forcibly drop the least recent result to make space
            timer->index_read = (timer->index_read + 2) % QUERY_POOL_SIZE;
        }

        p->cmd_timer = NULL;
    }

    if (vk->CmdEndDebugUtilsLabelEXT && supports_marks(cmd))
        vk->CmdEndDebugUtilsLabelEXT(cmd->buf);

    if (submit)
        ret = vk_cmd_submit(&p->cmd);

    pl_mutex_unlock(&p->recording);
    return ret;
}

void vk_gpu_idle_callback(pl_gpu gpu, vk_cb cb, const void *priv, const void *arg)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    pl_mutex_lock(&p->recording);
    if (p->cmd) {
        vk_cmd_callback(p->cmd, cb, priv, arg);
    } else {
        vk_dev_callback(vk, cb, priv, arg);
    }
    pl_mutex_unlock(&p->recording);
}

static void vk_gpu_destroy(pl_gpu gpu)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    vk_cmd_submit(&p->cmd);
    vk_wait_idle(vk);

    for (enum pl_tex_sample_mode s = 0; s < PL_TEX_SAMPLE_MODE_COUNT; s++) {
        for (enum pl_tex_address_mode a = 0; a < PL_TEX_ADDRESS_MODE_COUNT; a++)
            vk->DestroySampler(vk->dev, p->samplers[s][a], PL_VK_ALLOC);
    }

    pl_spirv_destroy(&p->spirv);
    pl_mutex_destroy(&p->recording);
    pl_free((void *) gpu);
}

pl_vulkan pl_vulkan_get(pl_gpu gpu)
{
    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    if (impl->destroy == vk_gpu_destroy) {
        struct pl_vk *p = (struct pl_vk *) impl;
        return p->vk->vulkan;
    }

    return NULL;
}

static pl_handle_caps vk_sync_handle_caps(struct vk_ctx *vk)
{
    pl_handle_caps caps = 0;

    for (int i = 0; vk_sync_handle_list[i]; i++) {
        enum pl_handle_type type = vk_sync_handle_list[i];

        VkPhysicalDeviceExternalSemaphoreInfo info = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO_KHR,
            .handleType = vk_sync_handle_type(type),
        };

        VkExternalSemaphoreProperties props = {
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES_KHR,
        };

        vk->GetPhysicalDeviceExternalSemaphoreProperties(vk->physd, &info, &props);
        VkExternalSemaphoreFeatureFlags flags = props.externalSemaphoreFeatures;
        if ((props.compatibleHandleTypes & info.handleType) &&
            (flags & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT_KHR))
        {
            caps |= type;
        }
    }

    return caps;
}

static pl_handle_caps vk_tex_handle_caps(struct vk_ctx *vk, bool import)
{
    pl_handle_caps caps = 0;

    for (int i = 0; vk_mem_handle_list[i]; i++) {
        enum pl_handle_type handle_type = vk_mem_handle_list[i];
        if (handle_type == PL_HANDLE_DMA_BUF && !vk->GetImageDrmFormatModifierPropertiesEXT) {
            PL_DEBUG(vk, "Tex caps for %s (0x%x) unsupported: no DRM modifiers",
                     vk_handle_name(vk_mem_handle_type(PL_HANDLE_DMA_BUF)),
                     (unsigned int) PL_HANDLE_DMA_BUF);
            continue;
        }

        // Query whether creation of a "basic" dummy texture would work
        VkPhysicalDeviceImageDrmFormatModifierInfoEXT drm_pinfo = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,
            .drmFormatModifier = DRM_FORMAT_MOD_LINEAR,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };

        VkPhysicalDeviceExternalImageFormatInfoKHR ext_pinfo = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO_KHR,
            .handleType = vk_mem_handle_type(handle_type),
        };

        VkPhysicalDeviceImageFormatInfo2KHR pinfo = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2_KHR,
            .pNext = &ext_pinfo,
            .format = VK_FORMAT_R8_UNORM,
            .type = VK_IMAGE_TYPE_2D,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        };

        if (handle_type == PL_HANDLE_DMA_BUF) {
            vk_link_struct(&pinfo, &drm_pinfo);
            pinfo.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
        }

        VkExternalImageFormatPropertiesKHR ext_props = {
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES_KHR,
        };

        VkImageFormatProperties2KHR props = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2_KHR,
            .pNext = &ext_props,
        };

        VkResult res;
        res = vk->GetPhysicalDeviceImageFormatProperties2KHR(vk->physd, &pinfo, &props);
        if (res != VK_SUCCESS) {
            PL_DEBUG(vk, "Tex caps for %s (0x%x) unsupported: %s",
                     vk_handle_name(ext_pinfo.handleType),
                     (unsigned int) handle_type,
                     vk_res_str(res));
            continue;
        }

        if (vk_external_mem_check(vk, &ext_props.externalMemoryProperties,
                                  handle_type, import))
        {
            caps |= handle_type;
        }
    }

#ifdef VK_EXT_metal_objects
    if (vk->ExportMetalObjectsEXT && import)
        caps |= PL_HANDLE_MTL_TEX | PL_HANDLE_IOSURFACE;
#endif

    return caps;
}

static const VkFilter filters[PL_TEX_SAMPLE_MODE_COUNT] = {
    [PL_TEX_SAMPLE_NEAREST] = VK_FILTER_NEAREST,
    [PL_TEX_SAMPLE_LINEAR]  = VK_FILTER_LINEAR,
};

static inline struct pl_spirv_version get_spirv_version(const struct vk_ctx *vk)
{
    if (vk->api_ver >= VK_API_VERSION_1_3) {
        const VkPhysicalDeviceMaintenance4Features *device_maintenance4;
        device_maintenance4 = vk_find_struct(&vk->features,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES);

        if (device_maintenance4 && device_maintenance4->maintenance4) {
            return (struct pl_spirv_version) {
                .env_version = VK_API_VERSION_1_3,
                .spv_version = PL_SPV_VERSION(1, 6),
            };
        }
    }

    pl_assert(vk->api_ver >= VK_API_VERSION_1_2);
    return (struct pl_spirv_version) {
        .env_version = VK_API_VERSION_1_2,
        .spv_version = PL_SPV_VERSION(1, 5),
    };
}

static const struct pl_gpu_fns pl_fns_vk;

pl_gpu pl_gpu_create_vk(struct vk_ctx *vk)
{
    pl_assert(vk->dev);

    struct pl_gpu_t *gpu = pl_zalloc_obj(NULL, gpu, struct pl_vk);
    gpu->log = vk->log;

    struct pl_vk *p = PL_PRIV(gpu);
    pl_mutex_init(&p->recording);
    p->vk = vk;
    p->impl = pl_fns_vk;
    p->spirv = pl_spirv_create(vk->log, get_spirv_version(vk));
    if (!p->spirv)
        goto error;

    // Query all device properties
    VkPhysicalDevicePCIBusInfoPropertiesEXT pci_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT,
    };

    VkPhysicalDeviceIDPropertiesKHR id_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES_KHR,
        .pNext = &pci_props,
    };

    VkPhysicalDevicePushDescriptorPropertiesKHR pushd_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR,
        .pNext = &id_props,
    };

    VkPhysicalDeviceSubgroupProperties group_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
        .pNext = &pushd_props,
    };

    VkPhysicalDeviceExternalMemoryHostPropertiesEXT host_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT,
        .pNext = &group_props,
    };

    VkPhysicalDeviceProperties2KHR props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR,
        .pNext = &host_props,
    };

    bool is_portability = false;

#ifdef VK_KHR_portability_subset
    VkPhysicalDevicePortabilitySubsetPropertiesKHR port_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_PROPERTIES_KHR,
        .minVertexInputBindingStrideAlignment = 1,
    };

    for (int i = 0; i < vk->exts.num; i++) {
        if (!strcmp(vk->exts.elem[i], VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
            vk_link_struct(&props, &port_props);
            is_portability = true;
            break;
        }
    }
#endif

    vk->GetPhysicalDeviceProperties2(vk->physd, &props);
    VkPhysicalDeviceLimits limits = props.properties.limits;

    // Determine GLSL features and limits
    gpu->glsl = (struct pl_glsl_version) {
        .version = 450,
        .vulkan = true,
        .compute = true,
        .max_shmem_size = limits.maxComputeSharedMemorySize,
        .max_group_threads = limits.maxComputeWorkGroupInvocations,
        .max_group_size = {
            limits.maxComputeWorkGroupSize[0],
            limits.maxComputeWorkGroupSize[1],
            limits.maxComputeWorkGroupSize[2],
        },
    };

    VkShaderStageFlags req_stages = VK_SHADER_STAGE_FRAGMENT_BIT |
                                    VK_SHADER_STAGE_COMPUTE_BIT;
    VkSubgroupFeatureFlags req_flags = VK_SUBGROUP_FEATURE_BASIC_BIT |
                                       VK_SUBGROUP_FEATURE_VOTE_BIT |
                                       VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
                                       VK_SUBGROUP_FEATURE_BALLOT_BIT |
                                       VK_SUBGROUP_FEATURE_SHUFFLE_BIT;

    if ((group_props.supportedStages & req_stages) == req_stages &&
        (group_props.supportedOperations & req_flags) == req_flags)
    {
        gpu->glsl.subgroup_size = group_props.subgroupSize;
    }

    if (vk->features.features.shaderImageGatherExtended) {
        gpu->glsl.min_gather_offset = limits.minTexelGatherOffset;
        gpu->glsl.max_gather_offset = limits.maxTexelGatherOffset;
    }

    const size_t max_size = vk_malloc_avail(vk->ma, 0);
    gpu->limits = (struct pl_gpu_limits) {
        // pl_gpu
        .thread_safe        = true,
        .callbacks          = true,
        // pl_buf
        .max_buf_size       = max_size,
        .max_ubo_size       = PL_MIN(limits.maxUniformBufferRange, max_size),
        .max_ssbo_size      = PL_MIN(limits.maxStorageBufferRange, max_size),
        .max_vbo_size       = vk_malloc_avail(vk->ma, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
        .max_mapped_size    = vk_malloc_avail(vk->ma, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT),
        .max_buffer_texels  = PL_MIN(limits.maxTexelBufferElements, max_size),
        .align_host_ptr     = host_props.minImportedHostPointerAlignment,
        .host_cached        = vk_malloc_avail(vk->ma, VK_MEMORY_PROPERTY_HOST_CACHED_BIT),
        // pl_tex
        .max_tex_1d_dim     = limits.maxImageDimension1D,
        .max_tex_2d_dim     = limits.maxImageDimension2D,
        .max_tex_3d_dim     = limits.maxImageDimension3D,
        .blittable_1d_3d    = true,
        .buf_transfer       = true,
        .align_tex_xfer_pitch  = limits.optimalBufferCopyRowPitchAlignment,
        .align_tex_xfer_offset = pl_lcm(limits.optimalBufferCopyOffsetAlignment, 4),
        // pl_pass
        .max_variable_comps = 0, // vulkan doesn't support these at all
        .max_constants      = SIZE_MAX,
        .array_size_constants = !is_portability,
        .max_pushc_size     = limits.maxPushConstantsSize,
#ifdef VK_KHR_portability_subset
        .align_vertex_stride = port_props.minVertexInputBindingStrideAlignment,
#else
        .align_vertex_stride = 1,
#endif
        .max_dispatch = {
            limits.maxComputeWorkGroupCount[0],
            limits.maxComputeWorkGroupCount[1],
            limits.maxComputeWorkGroupCount[2],
        },
        .fragment_queues    = vk->pool_graphics->num_queues,
        .compute_queues     = vk->pool_compute->num_queues,
    };

    gpu->export_caps.buf = vk_malloc_handle_caps(vk->ma, false);
    gpu->import_caps.buf = vk_malloc_handle_caps(vk->ma, true);
    gpu->export_caps.tex = vk_tex_handle_caps(vk, false);
    gpu->import_caps.tex = vk_tex_handle_caps(vk, true);
    gpu->export_caps.sync = vk_sync_handle_caps(vk);
    gpu->import_caps.sync = 0; // Not supported yet

    if (pl_gpu_supports_interop(gpu)) {
        pl_static_assert(sizeof(gpu->uuid) == VK_UUID_SIZE);
        memcpy(gpu->uuid, id_props.deviceUUID, sizeof(gpu->uuid));

        gpu->pci.domain = pci_props.pciDomain;
        gpu->pci.bus = pci_props.pciBus;
        gpu->pci.device = pci_props.pciDevice;
        gpu->pci.function = pci_props.pciFunction;
    }

    if (vk->CmdPushDescriptorSetKHR)
        p->max_push_descriptors = pushd_props.maxPushDescriptors;

    vk_setup_formats(gpu);

    // Compute the correct minimum texture alignment
    p->min_texel_alignment = 1;
    for (int i = 0; i < gpu->num_formats; i++) {
        if (gpu->formats[i]->emulated || gpu->formats[i]->opaque)
            continue;
        size_t texel_size = gpu->formats[i]->texel_size;
        p->min_texel_alignment = pl_lcm(p->min_texel_alignment, texel_size);
    }
    PL_DEBUG(gpu, "Minimum texel alignment: %zu", p->min_texel_alignment);

    // Initialize the samplers
    for (enum pl_tex_sample_mode s = 0; s < PL_TEX_SAMPLE_MODE_COUNT; s++) {
        for (enum pl_tex_address_mode a = 0; a < PL_TEX_ADDRESS_MODE_COUNT; a++) {
            static const VkSamplerAddressMode modes[PL_TEX_ADDRESS_MODE_COUNT] = {
                [PL_TEX_ADDRESS_CLAMP]  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                [PL_TEX_ADDRESS_REPEAT] = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                [PL_TEX_ADDRESS_MIRROR] = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
            };

            VkSamplerCreateInfo sinfo = {
                .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .magFilter = filters[s],
                .minFilter = filters[s],
                .addressModeU = modes[a],
                .addressModeV = modes[a],
                .addressModeW = modes[a],
                .maxAnisotropy = 1.0,
            };

            VK(vk->CreateSampler(vk->dev, &sinfo, PL_VK_ALLOC, &p->samplers[s][a]));
        }
    }

    return pl_gpu_finalize(gpu);

error:
    vk_gpu_destroy(gpu);
    return NULL;
}

void pl_vulkan_sem_destroy(pl_gpu gpu, VkSemaphore *semaphore)
{
    VkSemaphore sem = *semaphore;
    if (!sem)
        return;

    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    vk->DestroySemaphore(vk->dev, sem, PL_VK_ALLOC);
    *semaphore = VK_NULL_HANDLE;
}

VkSemaphore pl_vulkan_sem_create(pl_gpu gpu, const struct pl_vulkan_sem_params *params)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    pl_assert(PL_ISPOT(params->export_handle));
    if ((params->export_handle & gpu->export_caps.sync) != params->export_handle) {
        PL_ERR(gpu, "Invalid handle type 0x%"PRIx64" specified for "
               "`pl_vulkan_sem_create`!", (uint64_t) params->export_handle);
        return VK_NULL_HANDLE;
    }

    switch (params->export_handle) {
    case PL_HANDLE_FD:
        params->out_handle->fd = -1;
        break;
    case PL_HANDLE_WIN32:
    case PL_HANDLE_WIN32_KMT:
        params->out_handle->handle = NULL;
        break;
    case PL_HANDLE_DMA_BUF:
    case PL_HANDLE_HOST_PTR:
    case PL_HANDLE_MTL_TEX:
    case PL_HANDLE_IOSURFACE:
        pl_unreachable();
    }

    const VkExportSemaphoreCreateInfoKHR einfo = {
        .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR,
        .handleTypes = vk_sync_handle_type(params->export_handle),
    };

    const VkSemaphoreTypeCreateInfo stinfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .pNext = params->export_handle ? &einfo : NULL,
        .semaphoreType = params->type,
        .initialValue = params->initial_value,
    };

    const VkSemaphoreCreateInfo sinfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &stinfo,
    };

    VkSemaphore sem = VK_NULL_HANDLE;
    VK(vk->CreateSemaphore(vk->dev, &sinfo, PL_VK_ALLOC, &sem));
    PL_VK_NAME(SEMAPHORE, sem, PL_DEF(params->debug_tag, "pl_vulkan_sem"));

#ifdef PL_HAVE_UNIX
    if (params->export_handle == PL_HANDLE_FD) {
        VkSemaphoreGetFdInfoKHR finfo = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
            .handleType = einfo.handleTypes,
            .semaphore = sem,
        };

        VK(vk->GetSemaphoreFdKHR(vk->dev, &finfo, &params->out_handle->fd));
    }
#endif

#ifdef PL_HAVE_WIN32
    if (params->export_handle == PL_HANDLE_WIN32 ||
        params->export_handle == PL_HANDLE_WIN32_KMT)
    {
        VkSemaphoreGetWin32HandleInfoKHR handle_info = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR,
            .handleType = einfo.handleTypes,
            .semaphore = sem,
        };

        VK(vk->GetSemaphoreWin32HandleKHR(vk->dev, &handle_info,
                                          &params->out_handle->handle));
    }
#endif

    return sem;

error:
#ifdef PL_HAVE_UNIX
    if (params->export_handle == PL_HANDLE_FD) {
        if (params->out_handle->fd > -1)
            close(params->out_handle->fd);
    }
#endif
#ifdef PL_HAVE_WIN32
    if (params->export_handle == PL_HANDLE_WIN32) {
        if (params->out_handle->handle != NULL)
            CloseHandle(params->out_handle->handle);
    }
    // PL_HANDLE_WIN32_KMT is just an identifier. It doesn't get closed.
#endif
    vk->DestroySemaphore(vk->dev, sem, PL_VK_ALLOC);
    return VK_NULL_HANDLE;
}

static void vk_gpu_flush(pl_gpu gpu)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    CMD_SUBMIT(NULL);
    vk_rotate_queues(vk);
    vk_malloc_garbage_collect(vk->ma);
}

static void vk_gpu_finish(pl_gpu gpu)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    CMD_SUBMIT(NULL);
    vk_wait_idle(vk);
}

static bool vk_gpu_is_failed(pl_gpu gpu)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    return vk->failed;
}

struct vk_cmd *pl_vk_steal_cmd(pl_gpu gpu)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    pl_mutex_lock(&p->recording);
    struct vk_cmd *cmd = p->cmd;
    p->cmd = NULL;
    pl_mutex_unlock(&p->recording);

    struct vk_cmdpool *pool = vk->pool_graphics;
    if (!cmd || cmd->pool != pool) {
        vk_cmd_submit(&cmd);
        cmd = vk_cmd_begin(pool, NULL);
    }

    return cmd;
}

void pl_vk_print_heap(pl_gpu gpu, enum pl_log_level lev)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    vk_malloc_print_stats(vk->ma, lev);
}

static const struct pl_gpu_fns pl_fns_vk = {
    .destroy                = vk_gpu_destroy,
    .tex_create             = vk_tex_create,
    .tex_destroy            = vk_tex_deref,
    .tex_invalidate         = vk_tex_invalidate,
    .tex_clear_ex           = vk_tex_clear_ex,
    .tex_blit               = vk_tex_blit,
    .tex_upload             = vk_tex_upload,
    .tex_download           = vk_tex_download,
    .tex_poll               = vk_tex_poll,
    .buf_create             = vk_buf_create,
    .buf_destroy            = vk_buf_deref,
    .buf_write              = vk_buf_write,
    .buf_read               = vk_buf_read,
    .buf_copy               = vk_buf_copy,
    .buf_export             = vk_buf_export,
    .buf_poll               = vk_buf_poll,
    .desc_namespace         = vk_desc_namespace,
    .pass_create            = vk_pass_create,
    .pass_destroy           = vk_pass_destroy,
    .pass_run               = vk_pass_run,
    .timer_create           = vk_timer_create,
    .timer_destroy          = vk_timer_destroy,
    .timer_query            = vk_timer_query,
    .gpu_flush              = vk_gpu_flush,
    .gpu_finish             = vk_gpu_finish,
    .gpu_is_failed          = vk_gpu_is_failed,
};
