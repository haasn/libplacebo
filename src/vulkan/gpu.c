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
#include "command.h"
#include "formats.h"
#include "malloc.h"
#include "spirv.h"

#ifdef PL_HAVE_UNIX
#include <unistd.h>
#endif

static const struct pl_gpu_fns pl_fns_vk;

enum queue_type {
    GRAPHICS,
    COMPUTE,
    TRANSFER,
    ANY,
};

// For gpu.priv
struct pl_vk {
    struct pl_gpu_fns impl;
    struct vk_ctx *vk;
    struct vk_malloc *alloc;
    struct spirv_compiler *spirv;

    // Some additional cached device limits and features checks
    uint32_t max_push_descriptors;
    size_t min_texel_alignment;
    bool host_query_reset;

    // This is a pl_dispatch used (on ourselves!) for the purposes of
    // dispatching compute shaders for performing various emulation tasks
    // (e.g. partial clears, blits or emulated texture transfers).
    // Warning: Care must be taken to avoid recursive calls.
    struct pl_dispatch *dp;

    // The "currently recording" command. This will be queued and replaced by
    // a new command every time we need to "switch" between queue families.
    pthread_mutex_t recording;
    struct vk_cmd *cmd;

    // Array of VkSamplers for every combination of sample/address modes
    VkSampler samplers[PL_TEX_SAMPLE_MODE_COUNT][PL_TEX_ADDRESS_MODE_COUNT];

    // To avoid spamming warnings
    bool warned_modless;
};

static inline bool supports_marks(struct vk_cmd *cmd) {
    // Spec says debug markers are only available on graphics/compute queues
    VkQueueFlags flags = cmd->pool->props.queueFlags;
    return flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);
}

static inline struct vk_cmd *_begin_cmd(struct pl_vk *p, enum queue_type type,
                                        const char *label)
{
    struct vk_ctx *vk = p->vk;
    pthread_mutex_lock(&p->recording);

    struct vk_cmdpool *pool;
    switch (type) {
    case ANY:      pool = p->cmd ? p->cmd->pool : vk->pool_graphics; break;
    case GRAPHICS: pool = vk->pool_graphics; break;
    case COMPUTE:  pool = vk->pool_compute;  break;

    // GRAPHICS and COMPUTE also imply TRANSFER capability (vulkan spec)
    case TRANSFER:
        pool = vk->pool_transfer;
        if (!pool)
            pool = vk->pool_compute;
        if (!pool)
            pool = vk->pool_graphics;
        break;

    default: abort();
    }

    if (!p->cmd || p->cmd->pool != pool) {
        vk_cmd_queue(vk, &p->cmd);
        p->cmd = vk_cmd_begin(vk, pool);
    }

    if (vk->CmdBeginDebugUtilsLabelEXT && supports_marks(p->cmd)) {
        vk->CmdBeginDebugUtilsLabelEXT(p->cmd->buf, &(VkDebugUtilsLabelEXT) {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
            .pLabelName = label,
        });
    }

    return p->cmd;
}

static inline void _end_cmd(struct pl_vk *p, struct vk_cmd **pcmd, bool submit)
{
    struct vk_ctx *vk = p->vk;
    struct vk_cmd *cmd = *pcmd;
    pl_assert(p->cmd == cmd);

    if (vk->CmdEndDebugUtilsLabelEXT && supports_marks(cmd))
        vk->CmdEndDebugUtilsLabelEXT(cmd->buf);

    if (submit)
        vk_cmd_queue(vk, &p->cmd);

    pthread_mutex_unlock(&p->recording);
}

#define begin_cmd(p, type) _begin_cmd(p, type, __func__)
#define finish_cmd(p, cmd) _end_cmd(p, cmd, false)
#define submit_cmd(p, cmd) _end_cmd(p, cmd, true)

static void flush(struct pl_vk *p)
{
    pthread_mutex_lock(&p->recording);
    vk_cmd_queue(p->vk, &p->cmd);
    pthread_mutex_unlock(&p->recording);
}

#define MAKE_LAZY_DESTRUCTOR(fun, argtype)                                  \
    static void fun##_lazy(const struct pl_gpu *gpu, argtype *arg) {        \
        struct pl_vk *p = PL_PRIV(gpu);                                     \
        struct vk_ctx *vk = p->vk;                                          \
        if (p->cmd) {                                                       \
            vk_cmd_callback(p->cmd, (vk_cb) fun, gpu, (void *) arg);        \
        } else {                                                            \
            vk_dev_callback(vk, (vk_cb) fun, gpu, (void *) arg);            \
        }                                                                   \
    }

static void vk_destroy_gpu(const struct pl_gpu *gpu)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    pl_dispatch_destroy(&p->dp);
    vk_cmd_queue(vk, &p->cmd);
    vk_wait_idle(vk);

    for (enum pl_tex_sample_mode s = 0; s < PL_TEX_SAMPLE_MODE_COUNT; s++) {
        for (enum pl_tex_address_mode a = 0; a < PL_TEX_ADDRESS_MODE_COUNT; a++)
            vk->DestroySampler(vk->dev, p->samplers[s][a], PL_VK_ALLOC);
    }

    vk_malloc_destroy(&p->alloc);
    spirv_compiler_destroy(&p->spirv);

    pthread_mutex_destroy(&p->recording);
    pl_free((void *) gpu);
}

static void vk_setup_formats(struct pl_gpu *gpu)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    PL_ARRAY(const struct pl_fmt *) formats = {0};

    // Texture format emulation requires at least support for texel buffers
    bool has_emu = (gpu->caps & PL_GPU_CAP_COMPUTE) && gpu->limits.max_buffer_texels;

    for (const struct vk_format *pvk_fmt = vk_formats; pvk_fmt->tfmt; pvk_fmt++) {
        const struct vk_format *vk_fmt = pvk_fmt;

        // Skip formats with innately emulated representation if unsupported
        if (vk_fmt->fmt.emulated && !has_emu)
            continue;

        // Suppress some errors/warnings spit out by the format probing code
        pl_log_level_cap(vk->ctx, PL_LOG_INFO);

        bool has_drm_mods = vk->GetImageDrmFormatModifierPropertiesEXT;
        VkDrmFormatModifierPropertiesEXT modifiers[16] = {0};
        VkDrmFormatModifierPropertiesListEXT drm_props = {
            .sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT,
            .drmFormatModifierCount = PL_ARRAY_SIZE(modifiers),
            .pDrmFormatModifierProperties = modifiers,
        };

        VkFormatProperties2KHR prop2 = {
            .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
            .pNext = has_drm_mods ? &drm_props : NULL,
        };

        vk->GetPhysicalDeviceFormatProperties2KHR(vk->physd, vk_fmt->tfmt, &prop2);

        // If wholly unsupported, try falling back to the emulation formats
        // for texture operations
        VkFormatProperties *prop = &prop2.formatProperties;
        while (has_emu && !prop->optimalTilingFeatures && vk_fmt->emufmt) {
            vk_fmt = vk_fmt->emufmt;
            vk->GetPhysicalDeviceFormatProperties2KHR(vk->physd, vk_fmt->tfmt, &prop2);
        }

        VkFormatFeatureFlags texflags = prop->optimalTilingFeatures;
        VkFormatFeatureFlags bufflags = prop->bufferFeatures;
        if (vk_fmt->fmt.emulated) {
            // Emulated formats might have a different buffer representation
            // than their texture representation. If they don't, assume their
            // buffer representation is nonsensical (e.g. r16f)
            if (vk_fmt->bfmt) {
                vk->GetPhysicalDeviceFormatProperties(vk->physd, vk_fmt->bfmt, prop);
                bufflags = prop->bufferFeatures;
            } else {
                bufflags = 0;
            }
        }

        pl_log_level_cap(vk->ctx, PL_LOG_NONE);

        struct pl_fmt *fmt = pl_alloc_ptr_priv(gpu, fmt, vk_fmt);
        const struct vk_format **fmtp = PL_PRIV(fmt);
        *fmt = vk_fmt->fmt;
        *fmtp = vk_fmt;

        // For sanity, clear the superfluous fields
        for (int i = fmt->num_components; i < 4; i++) {
            fmt->component_depth[i] = 0;
            fmt->sample_order[i] = 0;
            fmt->host_bits[i] = 0;
        }

        // We can set this universally
        fmt->fourcc = pl_fmt_fourcc(fmt);

        if (has_drm_mods) {

            if (drm_props.drmFormatModifierCount == PL_ARRAY_SIZE(modifiers)) {
                PL_WARN(gpu, "DRM modifier list for format %s possibly truncated",
                        fmt->name);
            }

            // Query the list of supported DRM modifiers from the driver
            PL_ARRAY(uint64_t) modlist = {0};
            for (int i = 0; i < drm_props.drmFormatModifierCount; i++) {
                if (modifiers[i].drmFormatModifierPlaneCount > 1) {
                    PL_DEBUG(gpu, "Ignoring format modifier %s of "
                             "format %s because its plane count %d > 1",
                             PRINT_DRM_MOD(modifiers[i].drmFormatModifier),
                             fmt->name, modifiers[i].drmFormatModifierPlaneCount);
                    continue;
                }

                // Only warn about texture format features relevant to us
                const VkFormatFeatureFlags flag_mask =
                    VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT |
                    VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                    VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                    VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
                    VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                    VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                    VK_FORMAT_FEATURE_BLIT_DST_BIT;


                VkFormatFeatureFlags flags = modifiers[i].drmFormatModifierTilingFeatures;
                if ((flags & flag_mask) != (texflags & flag_mask)) {
                    PL_INFO(gpu, "DRM format modifier %s of format %s "
                            "supports fewer caps (0x%"PRIx32") than optimal tiling "
                            "(0x%"PRIx32"), may result in limited capability!",
                            PRINT_DRM_MOD(modifiers[i].drmFormatModifier),
                            fmt->name, flags, texflags);
                }

                PL_ARRAY_APPEND(fmt, modlist, modifiers[i].drmFormatModifier);
            }

            fmt->num_modifiers = modlist.num;
            fmt->modifiers = modlist.elem;

        } else if (gpu->export_caps.tex & PL_HANDLE_DMA_BUF) {

            // Hard-code a list of static mods that we're likely to support
            static const uint64_t static_mods[2] = {
                DRM_FORMAT_MOD_INVALID,
                DRM_FORMAT_MOD_LINEAR,
            };

            fmt->num_modifiers = PL_ARRAY_SIZE(static_mods);
            fmt->modifiers = static_mods;

        }

        struct { VkFormatFeatureFlags flags; enum pl_fmt_caps caps; } bufbits[] = {
            {VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT,        PL_FMT_CAP_VERTEX},
            {VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT, PL_FMT_CAP_TEXEL_UNIFORM},
            {VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT, PL_FMT_CAP_TEXEL_STORAGE},
        };

        for (int i = 0; i < PL_ARRAY_SIZE(bufbits); i++) {
            if ((bufflags & bufbits[i].flags) == bufbits[i].flags)
                fmt->caps |= bufbits[i].caps;
        }

        if (fmt->caps) {
            fmt->glsl_type = pl_var_glsl_type_name(pl_var_from_fmt(fmt, ""));
            pl_assert(fmt->glsl_type);
        }

        struct { VkFormatFeatureFlags flags; enum pl_fmt_caps caps; } bits[] = {
            {VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT,      PL_FMT_CAP_BLENDABLE},
            {VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT, PL_FMT_CAP_LINEAR},
            {VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT,               PL_FMT_CAP_SAMPLEABLE},
            {VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT,               PL_FMT_CAP_STORABLE},
            {VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT,            PL_FMT_CAP_RENDERABLE},

            // We don't distinguish between the two blit modes for pl_fmt_caps
            {VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT,
                PL_FMT_CAP_BLITTABLE},
        };

        for (int i = 0; i < PL_ARRAY_SIZE(bits); i++) {
            if ((texflags & bits[i].flags) == bits[i].flags)
                fmt->caps |= bits[i].caps;
        }

        // This is technically supported for all textures, but the semantics
        // of pl_gpu require it only be listed for non-opaque ones
        if (!fmt->opaque)
            fmt->caps |= PL_FMT_CAP_HOST_READABLE;

        // Disable implied capabilities where the dependencies are unavailable
        if (!(fmt->caps & PL_FMT_CAP_SAMPLEABLE))
            fmt->caps &= ~PL_FMT_CAP_LINEAR;
        if (!(gpu->caps & PL_GPU_CAP_COMPUTE))
            fmt->caps &= ~(PL_FMT_CAP_STORABLE | PL_FMT_CAP_TEXEL_STORAGE);

        // Only set this gpu-wide cap if at least one blittable fmt exists
        if (fmt->caps & PL_FMT_CAP_BLITTABLE)
            gpu->caps |= PL_GPU_CAP_BLITTABLE_1D_3D;

        enum pl_fmt_caps storable = PL_FMT_CAP_STORABLE | PL_FMT_CAP_TEXEL_STORAGE;
        if (fmt->caps & storable) {
            int real_comps = PL_DEF(vk_fmt->icomps, fmt->num_components);
            fmt->glsl_format = pl_fmt_glsl_format(fmt, real_comps);
            if (!fmt->glsl_format) {
                if (!vk->features.features.shaderStorageImageReadWithoutFormat ||
                    !vk->features.features.shaderStorageImageWriteWithoutFormat)
                {
                    PL_WARN(gpu, "Storable format '%s' has no matching GLSL "
                            "format qualifier but read/write without format "
                            "is not supported.. disabling", fmt->name);
                    fmt->caps &= ~storable;
                }
            }
        }

        PL_ARRAY_APPEND(gpu, formats, fmt);
    }

    gpu->formats = formats.elem;
    gpu->num_formats = formats.num;
    pl_gpu_sort_formats(gpu);
}

static pl_handle_caps vk_sync_handle_caps(struct vk_ctx *vk)
{
    pl_handle_caps caps = 0;

    if (!vk->GetPhysicalDeviceExternalSemaphorePropertiesKHR)
        return caps;

    for (int i = 0; vk_sync_handle_list[i]; i++) {
        enum pl_handle_type type = vk_sync_handle_list[i];

        VkPhysicalDeviceExternalSemaphoreInfoKHR info = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO_KHR,
            .handleType = vk_sync_handle_type(type),
        };

        VkExternalSemaphorePropertiesKHR props = {
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES_KHR,
        };

        vk->GetPhysicalDeviceExternalSemaphorePropertiesKHR(vk->physd, &info, &props);
        VkExternalSemaphoreFeatureFlagsKHR flags = props.externalSemaphoreFeatures;
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

    if (!vk->GetPhysicalDeviceImageFormatProperties2KHR)
        return caps;

    bool has_drm_mods = vk->GetImageDrmFormatModifierPropertiesEXT;
    for (int i = 0; vk_mem_handle_list[i]; i++) {
        enum pl_handle_type handle_type = vk_mem_handle_list[i];

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

        if (handle_type == PL_HANDLE_DMA_BUF && has_drm_mods) {
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
        if (res != VK_SUCCESS && handle_type == PL_HANDLE_DMA_BUF && !has_drm_mods) {
            // Try again with VK_IMAGE_TILING_LINEAR, as a dumb hack
            pinfo.tiling = VK_IMAGE_TILING_LINEAR;
            res = vk->GetPhysicalDeviceImageFormatProperties2KHR(vk->physd, &pinfo, &props);
        }
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

    return caps;
}

static const VkFilter filters[PL_TEX_SAMPLE_MODE_COUNT] = {
    [PL_TEX_SAMPLE_NEAREST] = VK_FILTER_NEAREST,
    [PL_TEX_SAMPLE_LINEAR]  = VK_FILTER_LINEAR,
};

const struct pl_gpu *pl_gpu_create_vk(struct vk_ctx *vk)
{
    pl_assert(vk->dev);

    struct pl_gpu *gpu = pl_zalloc_priv(NULL, struct pl_gpu, struct pl_vk);
    gpu->ctx = vk->ctx;

    struct pl_vk *p = PL_PRIV(gpu);
    pl_mutex_init(&p->recording);
    p->impl = pl_fns_vk;
    p->vk = vk;

    p->spirv = spirv_compiler_create(vk->ctx, vk->api_ver);
    p->alloc = vk_malloc_create(vk);
    if (!p->alloc || !p->spirv)
        goto error;

    gpu->glsl = p->spirv->glsl;
    gpu->limits = (struct pl_gpu_limits) {
        .max_tex_1d_dim    = vk->limits.maxImageDimension1D,
        .max_tex_2d_dim    = vk->limits.maxImageDimension2D,
        .max_tex_3d_dim    = vk->limits.maxImageDimension3D,
        .max_pushc_size    = vk->limits.maxPushConstantsSize,
        .max_buf_size      = SIZE_MAX, // no limit imposed by vulkan
        .max_ubo_size      = vk->limits.maxUniformBufferRange,
        .max_ssbo_size     = vk->limits.maxStorageBufferRange,
        .max_vbo_size      = SIZE_MAX,
        .max_buffer_texels = vk->limits.maxTexelBufferElements,
        .min_gather_offset = vk->limits.minTexelGatherOffset,
        .max_gather_offset = vk->limits.maxTexelGatherOffset,
        .align_tex_xfer_stride = vk->limits.optimalBufferCopyRowPitchAlignment,
        .align_tex_xfer_offset = pl_lcm(vk->limits.optimalBufferCopyOffsetAlignment, 4),
    };

    gpu->export_caps.buf = vk_malloc_handle_caps(p->alloc, false);
    gpu->import_caps.buf = vk_malloc_handle_caps(p->alloc, true);
    gpu->export_caps.tex = vk_tex_handle_caps(vk, false);
    gpu->import_caps.tex = vk_tex_handle_caps(vk, true);
    gpu->export_caps.sync = vk_sync_handle_caps(vk);
    gpu->import_caps.sync = 0; // Not supported yet

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

    vk->GetPhysicalDeviceProperties2KHR(vk->physd, &props);
    gpu->limits.align_host_ptr = host_props.minImportedHostPointerAlignment;

    if (pl_gpu_supports_interop(gpu)) {
        assert(sizeof(gpu->uuid) == VK_UUID_SIZE);
        memcpy(gpu->uuid, id_props.deviceUUID, sizeof(gpu->uuid));

        gpu->pci.domain = pci_props.pciDomain;
        gpu->pci.bus = pci_props.pciBus;
        gpu->pci.device = pci_props.pciDevice;
        gpu->pci.function = pci_props.pciFunction;
    }

    if (vk->CmdPushDescriptorSetKHR)
        p->max_push_descriptors = pushd_props.maxPushDescriptors;

    if (vk->ResetQueryPoolEXT) {
        const VkPhysicalDeviceHostQueryResetFeaturesEXT *host_query_reset;
        host_query_reset = vk_find_struct(&vk->features,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES_EXT);

        if (host_query_reset)
            p->host_query_reset = host_query_reset->hostQueryReset;
    }

    // Determine GPU capabilities
    gpu->caps |= PL_GPU_CAP_CALLBACKS | PL_GPU_CAP_THREAD_SAFE;

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
        gpu->limits.subgroup_size = group_props.subgroupSize;
        gpu->caps |= PL_GPU_CAP_SUBGROUPS;
    }

    // We ostensibly support this, although it can still fail on buffer
    // creation (for certain combinations of buffers)
    gpu->caps |= PL_GPU_CAP_MAPPED_BUFFERS;

    if (vk->pool_compute) {
        gpu->caps |= PL_GPU_CAP_COMPUTE;
        gpu->limits.max_shmem_size = vk->limits.maxComputeSharedMemorySize;
        gpu->limits.max_group_threads = vk->limits.maxComputeWorkGroupInvocations;
        for (int i = 0; i < 3; i++) {
            gpu->limits.max_group_size[i] = vk->limits.maxComputeWorkGroupSize[i];
            gpu->limits.max_dispatch[i] = vk->limits.maxComputeWorkGroupCount[i];
        }

        // If we have more compute queues than graphics queues, we probably
        // want to be using them. (This seems mostly relevant for AMD)
        if (vk->pool_compute->num_queues > vk->pool_graphics->num_queues)
            gpu->caps |= PL_GPU_CAP_PARALLEL_COMPUTE;
    }

    if (!vk->features.features.shaderImageGatherExtended) {
        gpu->limits.min_gather_offset = 0;
        gpu->limits.max_gather_offset = 0;
    }

    vk_setup_formats(gpu);

    // Compute the correct minimum texture alignment
    p->min_texel_alignment = 1;
    for (int i = 0; i < gpu->num_formats; i++) {
        if (gpu->formats[i]->emulated)
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

    // Create the dispatch last, after any setup of `gpu` is done
    p->dp = pl_dispatch_create(vk->ctx, gpu);
    pl_gpu_print_info(gpu);
    return gpu;

error:
    vk_destroy_gpu(gpu);
    return NULL;
}

// Boilerplate wrapper around vkCreateRenderPass to ensure passes remain
// compatible. The renderpass will automatically transition the image out of
// initialLayout and into finalLayout.
static VkResult vk_create_render_pass(struct vk_ctx *vk, const struct pl_fmt *fmt,
                                      VkAttachmentLoadOp loadOp,
                                      VkImageLayout initialLayout,
                                      VkImageLayout finalLayout,
                                      VkRenderPass *out)
{
    const struct vk_format **vk_fmt = PL_PRIV(fmt);

    VkRenderPassCreateInfo rinfo = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &(VkAttachmentDescription) {
            .format = (*vk_fmt)->tfmt,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = loadOp,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .initialLayout = initialLayout,
            .finalLayout = finalLayout,
        },
        .subpassCount = 1,
        .pSubpasses = &(VkSubpassDescription) {
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &(VkAttachmentReference) {
                .attachment = 0,
                .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            },
        },
    };

    return vk->CreateRenderPass(vk->dev, &rinfo, PL_VK_ALLOC, out);
}

static void vk_cmd_timer_begin(const struct pl_gpu *gpu, struct vk_cmd *cmd,
                               struct pl_timer *timer);

static void vk_cmd_timer_end(const struct pl_gpu *gpu, struct vk_cmd *cmd,
                             struct pl_timer *timer);

// For pl_tex.priv
struct pl_tex_vk {
    pl_rc_t rc;
    bool held;
    bool external_img;
    bool may_invalidate;
    enum queue_type transfer_queue;
    VkImageType type;
    VkImage img;
    struct vk_memslice mem;
    // cached properties
    VkFormat img_fmt;
    VkImageUsageFlags usage_flags;
    // for sampling
    VkImageView view;
    // for rendering
    VkFramebuffer framebuffer;
    // for vk_tex_upload/download fallback code
    const struct pl_fmt *texel_fmt;
    // "current" metadata, can change during the course of execution
    VkImageLayout current_layout;
    VkAccessFlags current_access;
    // the signal guards reuse, and can be NULL
    struct vk_signal *sig;
    VkPipelineStageFlags sig_stage;
    PL_ARRAY(VkSemaphore) ext_deps; // external semaphore, not owned by the pl_tex
    const struct pl_sync *ext_sync; // indicates an exported image
};

void pl_tex_vk_external_dep(const struct pl_gpu *gpu, const struct pl_tex *tex,
                            VkSemaphore external_dep)
{
    if (!external_dep)
        return;

    struct pl_tex_vk *tex_vk = PL_PRIV(tex);
    PL_ARRAY_APPEND((void *) tex, tex_vk->ext_deps, external_dep);
}

static void vk_sync_deref(const struct pl_gpu *gpu, const struct pl_sync *sync);

static void vk_tex_destroy(const struct pl_gpu *gpu, struct pl_tex *tex)
{
    if (!tex)
        return;

    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);

    vk_sync_deref(gpu, tex_vk->ext_sync);
    vk_signal_destroy(vk, &tex_vk->sig);
    vk->DestroyFramebuffer(vk->dev, tex_vk->framebuffer, PL_VK_ALLOC);
    vk->DestroyImageView(vk->dev, tex_vk->view, PL_VK_ALLOC);
    if (!tex_vk->external_img) {
        vk->DestroyImage(vk->dev, tex_vk->img, PL_VK_ALLOC);
        vk_malloc_free(p->alloc, &tex_vk->mem);
    }

    pl_free(tex);
}

static void vk_tex_deref(const struct pl_gpu *gpu, const struct pl_tex *tex)
{
    if (!tex)
        return;

    struct pl_tex_vk *tex_vk = PL_PRIV(tex);
    if (pl_rc_deref(&tex_vk->rc))
        vk_tex_destroy(gpu, (struct pl_tex *) tex);
}


// Small helper to ease image barrier creation. if `discard` is set, the contents
// of the image will be undefined after the barrier
static void tex_barrier(const struct pl_gpu *gpu, struct vk_cmd *cmd,
                        const struct pl_tex *tex, VkPipelineStageFlags stage,
                        VkAccessFlags newAccess, VkImageLayout newLayout,
                        bool export)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);
    pl_rc_ref(&tex_vk->rc);
    pl_assert(!tex_vk->held);

    for (int i = 0; i < tex_vk->ext_deps.num; i++)
        vk_cmd_dep(cmd, tex_vk->ext_deps.elem[i], stage);
    tex_vk->ext_deps.num = 0;

    // CONCURRENT images require transitioning to/from IGNORED, EXCLUSIVE
    // images require transitioning to/from the concrete QF index
    uint32_t qf = vk->pools.num > 1 ? VK_QUEUE_FAMILY_IGNORED : cmd->pool->qf;

    VkImageMemoryBarrier imgBarrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout = tex_vk->current_layout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = qf,
        .dstQueueFamilyIndex = export ? VK_QUEUE_FAMILY_EXTERNAL_KHR : qf,
        .srcAccessMask = tex_vk->current_access,
        .dstAccessMask = newAccess,
        .image = tex_vk->img,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
        },
    };

    if (tex_vk->ext_sync) {
        if (tex_vk->current_layout != VK_IMAGE_LAYOUT_UNDEFINED) {
            imgBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL_KHR;
            pl_assert(!export); // can't re-export exported images
        }
        vk_cmd_callback(cmd, (vk_cb) vk_sync_deref, gpu, tex_vk->ext_sync);
        tex_vk->ext_sync = NULL;
    }

    if (tex_vk->may_invalidate) {
        tex_vk->may_invalidate = false;
        imgBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imgBarrier.srcAccessMask = 0;
    }

    VkEvent event = VK_NULL_HANDLE;
    enum vk_wait_type type = vk_cmd_wait(vk, cmd, &tex_vk->sig, stage, &event);

    bool need_trans = tex_vk->current_layout != newLayout ||
                      tex_vk->current_access != newAccess ||
                      (imgBarrier.srcQueueFamilyIndex !=
                       imgBarrier.dstQueueFamilyIndex);

    // Transitioning to VK_IMAGE_LAYOUT_UNDEFINED is a pseudo-operation
    // that for us means we don't need to perform the actual transition
    if (need_trans && newLayout != VK_IMAGE_LAYOUT_UNDEFINED) {
        switch (type) {
        case VK_WAIT_NONE:
            // No synchronization required, so we can safely transition out of
            // VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            imgBarrier.srcAccessMask = 0;
            vk->CmdPipelineBarrier(cmd->buf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                   stage, 0, 0, NULL, 0, NULL, 1, &imgBarrier);
            break;
        case VK_WAIT_BARRIER:
            // Regular pipeline barrier is required
            vk->CmdPipelineBarrier(cmd->buf, tex_vk->sig_stage, stage, 0, 0, NULL,
                                   0, NULL, 1, &imgBarrier);
            break;
        case VK_WAIT_EVENT:
            // We can/should use the VkEvent for synchronization
            vk->CmdWaitEvents(cmd->buf, 1, &event, tex_vk->sig_stage,
                              stage, 0, NULL, 0, NULL, 1, &imgBarrier);
            break;
        }
    }

    tex_vk->current_layout = newLayout;
    tex_vk->current_access = newAccess;
    vk_cmd_callback(cmd, (vk_cb) vk_tex_deref, gpu, tex);
    vk_cmd_obj(cmd, tex);
}

static void tex_signal(const struct pl_gpu *gpu, struct vk_cmd *cmd,
                       const struct pl_tex *tex, VkPipelineStageFlags stage)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);
    struct vk_ctx *vk = p->vk;
    pl_assert(!tex_vk->sig);

    tex_vk->sig = vk_cmd_signal(vk, cmd, stage);
    tex_vk->sig_stage = stage;
}

// Initializes non-VkImage values like the image view, framebuffers, etc.
static bool vk_init_image(const struct pl_gpu *gpu, const struct pl_tex *tex,
                          const char *name)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    const struct pl_tex_params *params = &tex->params;
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);
    pl_assert(tex_vk->img);
    PL_VK_NAME(IMAGE, tex_vk->img, name);

    pl_rc_init(&tex_vk->rc);
    tex_vk->current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    tex_vk->current_access = 0;
    tex_vk->transfer_queue = GRAPHICS;

    // Always use the transfer pool if available, for efficiency
    if ((params->host_writable || params->host_readable) && vk->pool_transfer)
        tex_vk->transfer_queue = TRANSFER;

    // For emulated formats: force usage of the compute queue, because we
    // can't properly track cross-queue dependencies for buffers (yet?)
    if (params->format->emulated)
        tex_vk->transfer_queue = COMPUTE;

    bool ret = false;
    VkRenderPass dummyPass = VK_NULL_HANDLE;

    if (params->sampleable || params->renderable || params->storable ||
        params->format->emulated)
    {
        static const VkImageViewType viewType[] = {
            [VK_IMAGE_TYPE_1D] = VK_IMAGE_VIEW_TYPE_1D,
            [VK_IMAGE_TYPE_2D] = VK_IMAGE_VIEW_TYPE_2D,
            [VK_IMAGE_TYPE_3D] = VK_IMAGE_VIEW_TYPE_3D,
        };

        VkImageViewCreateInfo vinfo = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = tex_vk->img,
            .viewType = viewType[tex_vk->type],
            .format = tex_vk->img_fmt,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .levelCount = 1,
                .layerCount = 1,
            },
        };

        VK(vk->CreateImageView(vk->dev, &vinfo, PL_VK_ALLOC, &tex_vk->view));
        PL_VK_NAME(IMAGE_VIEW, tex_vk->view, name);
    }

    if (params->renderable) {
        // Framebuffers need to be created against a specific render pass
        // layout, so we need to temporarily create a skeleton/dummy render
        // pass for vulkan to figure out the compatibility
        VK(vk_create_render_pass(vk, params->format,
                                 VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 &dummyPass));

        VkFramebufferCreateInfo finfo = {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = dummyPass,
            .attachmentCount = 1,
            .pAttachments = &tex_vk->view,
            .width = tex->params.w,
            .height = tex->params.h,
            .layers = 1,
        };

        if (finfo.width > vk->limits.maxFramebufferWidth ||
            finfo.height > vk->limits.maxFramebufferHeight)
        {
            PL_ERR(gpu, "Framebuffer of size %dx%d exceeds the maximum allowed "
                   "dimensions: %dx%d", finfo.width, finfo.height,
                   vk->limits.maxFramebufferWidth,
                   vk->limits.maxFramebufferHeight);
            goto error;
        }

        VK(vk->CreateFramebuffer(vk->dev, &finfo, PL_VK_ALLOC,
                                 &tex_vk->framebuffer));
        PL_VK_NAME(FRAMEBUFFER, tex_vk->framebuffer, name);
    }

    ret = true;

error:
    vk->DestroyRenderPass(vk->dev, dummyPass, PL_VK_ALLOC);
    return ret;
}

static const struct pl_tex *vk_tex_create(const struct pl_gpu *gpu,
                                          const struct pl_tex_params *params)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    enum pl_handle_type handle_type = params->export_handle |
                                      params->import_handle;

    struct pl_tex *tex = pl_zalloc_priv(NULL, struct pl_tex, struct pl_tex_vk);
    tex->params = *params;
    tex->params.initial_data = NULL;
    tex->sampler_type = PL_SAMPLER_NORMAL;

    struct pl_tex_vk *tex_vk = PL_PRIV(tex);
    const struct vk_format **vk_fmt = PL_PRIV(params->format);
    tex_vk->img_fmt = (*vk_fmt)->tfmt;

    switch (pl_tex_params_dimension(*params)) {
    case 1: tex_vk->type = VK_IMAGE_TYPE_1D; break;
    case 2: tex_vk->type = VK_IMAGE_TYPE_2D; break;
    case 3: tex_vk->type = VK_IMAGE_TYPE_3D; break;
    default: abort();
    }

    if (params->format->emulated) {
        tex_vk->texel_fmt = pl_find_fmt(gpu, params->format->type, 1, 0,
                                        params->format->host_bits[0],
                                        PL_FMT_CAP_TEXEL_UNIFORM);
        if (!tex_vk->texel_fmt) {
            PL_ERR(gpu, "Failed picking texel format for emulated texture!");
            goto error;
        }

        // Statically check to see if we'd even be able to upload it at all
        // and refuse right away if not. In theory, uploading can still fail
        // based on the size of pl_tex_transfer_params.stride_w, but for now
        // this should be enough.
        uint64_t texels = params->w * PL_DEF(params->h, 1) * PL_DEF(params->d, 1) *
                          params->format->num_components;

        if (texels > gpu->limits.max_buffer_texels) {
            PL_ERR(gpu, "Failed creating texture with emulated texture format: "
                   "texture dimensions exceed maximum texel buffer size! Try "
                   "again with a different (non-emulated) format?");
            goto error;
        }

        // Our format emulation requires storage image support. In order to
        // make a bunch of checks happy, just mark it off as storable (and also
        // enable VK_IMAGE_USAGE_STORAGE_BIT, which we do below)
        tex->params.storable = true;
    }

    VkImageUsageFlags usage = 0;
    if (params->sampleable)
        usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (params->renderable)
        usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (params->storable || params->format->emulated)
        usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    if (params->host_readable || params->blit_src)
        usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (params->host_writable || params->blit_dst || params->initial_data)
        usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    if (!usage) {
        // Vulkan requires images have at least *some* image usage set, but our
        // API is perfectly happy with a (useless) image. So just put
        // VK_IMAGE_USAGE_TRANSFER_DST_BIT since this harmless.
        usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    }

    // FIXME: Since we can't keep track of queue family ownership properly,
    // and we don't know in advance what types of queue families this image
    // will belong to, we're forced to share all of our images between all
    // command pools.
    uint32_t qfs[3] = {0};
    for (int i = 0; i < vk->pools.num; i++)
        qfs[i] = vk->pools.elem[i]->qf;

    VkImageDrmFormatModifierExplicitCreateInfoEXT drm_explicit = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
        .drmFormatModifier = params->shared_mem.drm_format_mod,
        .drmFormatModifierPlaneCount = 1,
        .pPlaneLayouts = &(VkSubresourceLayout) {
            .rowPitch = PL_DEF(params->shared_mem.stride_w, params->w),
            .depthPitch = params->d ? PL_DEF(params->shared_mem.stride_h, params->h) : 0,
        },
    };

    VkImageDrmFormatModifierListCreateInfoEXT drm_list = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
        .drmFormatModifierCount = params->format->num_modifiers,
        .pDrmFormatModifiers = params->format->modifiers,
    };

    VkExternalMemoryImageCreateInfoKHR ext_info = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO_KHR,
        .handleTypes = vk_mem_handle_type(handle_type),
    };

    VkImageCreateInfo iinfo = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = handle_type ? &ext_info : NULL,
        .imageType = tex_vk->type,
        .format = tex_vk->img_fmt,
        .extent = (VkExtent3D) {
            .width  = params->w,
            .height = PL_MAX(1, params->h),
            .depth  = PL_MAX(1, params->d)
        },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .sharingMode = vk->pools.num > 1 ? VK_SHARING_MODE_CONCURRENT
                                         : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = vk->pools.num,
        .pQueueFamilyIndices = qfs,
    };

    bool has_drm_mods = vk->GetImageDrmFormatModifierPropertiesEXT;
    if (handle_type == PL_HANDLE_DMA_BUF && !has_drm_mods && !p->warned_modless) {
        PL_WARN(gpu, "Using legacy hacks for DMA buffers without modifiers. "
                "May result in corruption!");
        p->warned_modless = true;
    }

    if (params->import_handle == PL_HANDLE_DMA_BUF) {
        if (has_drm_mods) {

            // We have VK_EXT_image_drm_format_modifier, so we can use
            // format modifiers properly
            vk_link_struct(&iinfo, &drm_explicit);
            iinfo.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;

        } else {

            // Legacy fallback for older drivers. Based on hacks and guesswork.
            switch (drm_explicit.drmFormatModifier) {
            case DRM_FORMAT_MOD_LINEAR:
                iinfo.tiling = VK_IMAGE_TILING_LINEAR;
                break;
            }

        }
    }

    if (params->export_handle == PL_HANDLE_DMA_BUF && has_drm_mods) {
        vk_link_struct(&iinfo, &drm_list);
        iinfo.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
    }

    // Double-check physical image format limits and fail if invalid
    VkPhysicalDeviceImageDrmFormatModifierInfoEXT drm_pinfo = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,
        .drmFormatModifier = drm_explicit.drmFormatModifier,
        .sharingMode = iinfo.sharingMode,
        .queueFamilyIndexCount = iinfo.queueFamilyIndexCount,
        .pQueueFamilyIndices = iinfo.pQueueFamilyIndices,
    };

    VkPhysicalDeviceExternalImageFormatInfoKHR ext_pinfo = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO_KHR,
        .pNext = handle_type == PL_HANDLE_DMA_BUF ? &drm_pinfo : NULL,
        .handleType = ext_info.handleTypes,
    };

    VkPhysicalDeviceImageFormatInfo2KHR pinfo = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2_KHR,
        .pNext = handle_type ? &ext_pinfo : NULL,
        .format = iinfo.format,
        .type = iinfo.imageType,
        .tiling = iinfo.tiling,
        .usage = iinfo.usage,
        .flags = iinfo.flags,
    };

    VkExternalImageFormatPropertiesKHR ext_props = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES_KHR,
    };

    VkImageFormatProperties2KHR props = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2_KHR,
        .pNext = handle_type ? &ext_props : NULL,
    };

    VkResult res;
    res = vk->GetPhysicalDeviceImageFormatProperties2KHR(vk->physd, &pinfo, &props);
    if (res == VK_ERROR_FORMAT_NOT_SUPPORTED) {
        goto error;
    } else {
        PL_VK_ASSERT(res, "Querying image format properties");
    }

    VkExtent3D max = props.imageFormatProperties.maxExtent;
    if (params->w > max.width || params->h > max.height || params->d > max.depth)
    {
        PL_ERR(gpu, "Requested image size %dx%dx%d exceeds the maximum allowed "
               "dimensions %dx%dx%d for vulkan image format %x",
               params->w, params->h, params->d, max.width, max.height, max.depth,
               (unsigned) iinfo.format);
        goto error;
    }

    // Ensure the handle type is supported
    if (handle_type) {
        bool ok = vk_external_mem_check(vk, &ext_props.externalMemoryProperties,
                                        handle_type, params->import_handle);
        if (!ok) {
            PL_ERR(gpu, "Requested handle type is not compatible with the "
                   "specified combination of image parameters. Possibly the "
                   "handle type is unsupported altogether?");
            goto error;
        }
    }

    VK(vk->CreateImage(vk->dev, &iinfo, PL_VK_ALLOC, &tex_vk->img));
    tex_vk->usage_flags = iinfo.usage;

    struct vk_malloc_params mparams = {
        .optimal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        .export_handle = params->export_handle,
        .import_handle = params->import_handle,
        .shared_mem = params->shared_mem,
    };

    if (vk->GetImageMemoryRequirements2KHR) {
        VkMemoryDedicatedRequirementsKHR ded_reqs = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR,
        };

        VkMemoryRequirements2KHR reqs2 = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR,
            .pNext = &ded_reqs,
        };

        VkImageMemoryRequirementsInfo2KHR req_info2 = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2_KHR,
            .image = tex_vk->img,
        };

        vk->GetImageMemoryRequirements2KHR(vk->dev, &req_info2, &reqs2);
        mparams.reqs = reqs2.memoryRequirements;
        if (ded_reqs.prefersDedicatedAllocation)
            mparams.ded_image = tex_vk->img;
    } else {
        vk->GetImageMemoryRequirements(vk->dev, tex_vk->img, &mparams.reqs);
    }

    struct vk_memslice *mem = &tex_vk->mem;
    if (!vk_malloc_slice(p->alloc, mem, &mparams))
        goto error;

    if (params->import_handle && !has_drm_mods) {
        // Currently, we know that attempting to bind imported memory may generate
        // validation errors because there's no way to communicate the memory
        // layout; the validation layer will rely on the expected Vulkan layout
        // for the image. As long as the driver can handle the image, we'll be ok
        // so we don't want these validation errors to fire and create false
        // positives.
        vk->ctx->suppress_errors_for_object = (uint64_t) tex_vk->img;
    }

    VK(vk->BindImageMemory(vk->dev, tex_vk->img, mem->vkmem, mem->offset));
    if (params->import_handle && !has_drm_mods)
        vk->ctx->suppress_errors_for_object = VK_NULL_HANDLE;

    if (!vk_init_image(gpu, tex, params->import_handle ? "imported" : "created"))
        goto error;

    if (params->export_handle)
        tex->shared_mem = tex_vk->mem.shared_mem;

    if (params->export_handle == PL_HANDLE_DMA_BUF) {
        if (vk->GetImageDrmFormatModifierPropertiesEXT) {

            // Query the DRM format modifier and plane layout from the driver
            VkImageDrmFormatModifierPropertiesEXT mod_props = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT,
            };

            VK(vk->GetImageDrmFormatModifierPropertiesEXT(vk->dev, tex_vk->img, &mod_props));
            tex->shared_mem.drm_format_mod = mod_props.drmFormatModifier;

            VkSubresourceLayout layout;
            VkImageSubresource plane = {
                .aspectMask = VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT,
            };

            vk->GetImageSubresourceLayout(vk->dev, tex_vk->img, &plane, &layout);
            if (layout.offset != 0) {
                PL_ERR(gpu, "Exported DRM plane 0 has nonzero offset %zu, "
                       "this should never happen! Erroring for safety...",
                       (size_t) layout.offset);
                goto error;
            }
            tex->shared_mem.stride_w = layout.rowPitch;
            tex->shared_mem.stride_h = layout.depthPitch;

        } else {

            // Fallback for no modifiers, just do something stupid.
            tex->shared_mem.drm_format_mod = DRM_FORMAT_MOD_INVALID;
            tex->shared_mem.stride_w = params->w;
            tex->shared_mem.stride_h = params->h;

        }
    }

    if (params->initial_data) {
        struct pl_tex_transfer_params ul_params = {
            .tex = tex,
            .ptr = (void *) params->initial_data,
            .rc = { 0, 0, 0, params->w, params->h, params->d },
            .stride_w = params->w,
            .stride_h = params->h,
        };

        // Since we re-use GPU helpers which require writable images, just fake it
        bool writable = tex->params.host_writable;
        tex->params.host_writable = true;
        if (!pl_tex_upload(gpu, &ul_params))
            goto error;
        tex->params.host_writable = writable;
    }

    return tex;

error:
    if (params->import_handle)
        vk->ctx->suppress_errors_for_object = VK_NULL_HANDLE;
    vk_tex_destroy(gpu, tex);
    return NULL;
}

static void vk_tex_invalidate(const struct pl_gpu *gpu, const struct pl_tex *tex)
{
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);
    tex_vk->may_invalidate = true;
}

static void vk_tex_clear(const struct pl_gpu *gpu, const struct pl_tex *tex,
                         const float color[4])
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);

    struct vk_cmd *cmd = begin_cmd(p, GRAPHICS);
    if (!cmd)
        return;

    tex_barrier(gpu, cmd, tex, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                false);

    VkClearColorValue clearColor = {0};
    for (int c = 0; c < 4; c++)
        clearColor.float32[c] = color[c];

    static const VkImageSubresourceRange range = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .levelCount = 1,
        .layerCount = 1,
    };

    vk->CmdClearColorImage(cmd->buf, tex_vk->img, tex_vk->current_layout,
                           &clearColor, 1, &range);

    tex_signal(gpu, cmd, tex, VK_PIPELINE_STAGE_TRANSFER_BIT);
    finish_cmd(p, &cmd);
}

static void vk_tex_blit(const struct pl_gpu *gpu,
                        const struct pl_tex_blit_params *params)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_tex_vk *src_vk = PL_PRIV(params->src);
    struct pl_tex_vk *dst_vk = PL_PRIV(params->dst);

    struct vk_cmd *cmd = begin_cmd(p, GRAPHICS);
    if (!cmd)
        return;

    tex_barrier(gpu, cmd, params->src, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                false);

    tex_barrier(gpu, cmd, params->dst, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                false);

    static const VkImageSubresourceLayers layers = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .layerCount = 1,
    };

    // When the blit operation doesn't require scaling, we can use the more
    // efficient vkCmdCopyImage instead of vkCmdBlitImage
    struct pl_rect3d src_rc = params->src_rc, dst_rc = params->dst_rc;
    if (pl_rect3d_eq(src_rc, dst_rc)) {
        pl_rect3d_normalize(&src_rc);
        pl_rect3d_normalize(&dst_rc);

        VkImageCopy region = {
            .srcSubresource = layers,
            .dstSubresource = layers,
            .srcOffset = {src_rc.x0, src_rc.y0, src_rc.z0},
            .dstOffset = {dst_rc.x0, dst_rc.y0, dst_rc.z0},
            .extent = {
                pl_rect_w(src_rc),
                pl_rect_h(src_rc),
                pl_rect_d(src_rc),
            },
        };

        vk->CmdCopyImage(cmd->buf, src_vk->img, src_vk->current_layout,
                         dst_vk->img, dst_vk->current_layout, 1, &region);
    } else {
        VkImageBlit region = {
            .srcSubresource = layers,
            .dstSubresource = layers,
            .srcOffsets = {{src_rc.x0, src_rc.y0, src_rc.z0},
                           {src_rc.x1, src_rc.y1, src_rc.z1}},
            .dstOffsets = {{dst_rc.x0, dst_rc.y0, dst_rc.z0},
                           {dst_rc.x1, dst_rc.y1, dst_rc.z1}},
        };

        vk->CmdBlitImage(cmd->buf, src_vk->img, src_vk->current_layout,
                         dst_vk->img, dst_vk->current_layout, 1, &region,
                         filters[params->sample_mode]);
    }

    tex_signal(gpu, cmd, params->src, VK_PIPELINE_STAGE_TRANSFER_BIT);
    tex_signal(gpu, cmd, params->dst, VK_PIPELINE_STAGE_TRANSFER_BIT);
    finish_cmd(p, &cmd);
}

static bool vk_tex_poll(const struct pl_gpu *gpu, const struct pl_tex *tex,
                        uint64_t timeout)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);

    // Opportunistically check if we can re-use this texture without flush
    vk_poll_commands(vk, 0);
    if (pl_rc_count(&tex_vk->rc) == 1)
        return false;

    // Otherwise, we're force to submit all queued commands so that the
    // user is guaranteed to see progress eventually, even if they call
    // this in a tight loop
    flush(p);
    vk_flush_obj(vk, tex);
    vk_poll_commands(vk, timeout);

    return pl_rc_count(&tex_vk->rc) > 1;
}

const struct pl_tex *pl_vulkan_wrap(const struct pl_gpu *gpu,
                                    const struct pl_vulkan_wrap_params *params)
{
    struct pl_tex *tex = NULL;

    const struct pl_fmt *format = NULL;
    for (int i = 0; i < gpu->num_formats; i++) {
        const struct vk_format **fmt = PL_PRIV(gpu->formats[i]);
        if ((*fmt)->tfmt == params->format) {
            format = gpu->formats[i];
            break;
        }
    }

    if (!format) {
        PL_ERR(gpu, "Could not find pl_fmt suitable for wrapped image "
               "with format %s", vk_fmt_name(params->format));
        goto error;
    }

    tex = pl_zalloc_priv(NULL, struct pl_tex, struct pl_tex_vk);
    tex->params = (struct pl_tex_params) {
        .format = format,
        .w = params->width,
        .h = params->height,
        .d = params->depth,
        .sampleable  = !!(params->usage & VK_IMAGE_USAGE_SAMPLED_BIT),
        .renderable  = !!(params->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT),
        .storable    = !!(params->usage & VK_IMAGE_USAGE_STORAGE_BIT),
        .blit_src    = !!(params->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT),
        .blit_dst    = !!(params->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT),
        .host_writable = !!(params->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT),
        .host_readable = !!(params->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT),
    };

    // Mask out capabilities not permitted by the `pl_fmt`
#define MASK(field, cap)                                                        \
    do {                                                                        \
        if (tex->params.field && !(format->caps & cap)) {                       \
            PL_WARN(gpu, "Masking `" #field "` from wrapped texture because "   \
                    "the corresponding format '%s' does not support " #cap,     \
                    format->name);                                              \
            tex->params.field = false;                                          \
        }                                                                       \
    } while (0)

    MASK(sampleable, PL_FMT_CAP_SAMPLEABLE);
    MASK(storable,   PL_FMT_CAP_STORABLE);
    MASK(blit_src,   PL_FMT_CAP_BLITTABLE);
    MASK(blit_dst,   PL_FMT_CAP_BLITTABLE);
#undef MASK

    struct pl_tex_vk *tex_vk = PL_PRIV(tex);
    tex_vk->type = VK_IMAGE_TYPE_2D;
    tex_vk->external_img = true;
    tex_vk->held = true;
    tex_vk->img = params->image;
    tex_vk->img_fmt = params->format;
    tex_vk->usage_flags = params->usage;

    if (!vk_init_image(gpu, tex, "wrapped"))
        goto error;

    return tex;

error:
    vk_tex_destroy(gpu, tex);
    return NULL;
}

VkImage pl_vulkan_unwrap(const struct pl_gpu *gpu, const struct pl_tex *tex,
                         VkFormat *out_format, VkImageUsageFlags *out_flags)
{
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);

    if (out_format)
        *out_format = tex_vk->img_fmt;
    if (out_flags)
        *out_flags = tex_vk->usage_flags;

    return tex_vk->img;
}

bool pl_vulkan_hold(const struct pl_gpu *gpu, const struct pl_tex *tex,
                    VkImageLayout layout, VkAccessFlags access,
                    VkSemaphore sem_out)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);
    pl_assert(sem_out);

    if (tex_vk->held) {
        PL_ERR(gpu, "Attempting to hold an already held image!");
        return false;
    }

    struct vk_cmd *cmd = begin_cmd(p, GRAPHICS);
    if (!cmd) {
        PL_ERR(gpu, "Failed holding external image!");
        return false;
    }

    tex_barrier(gpu, cmd, tex, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                access, layout, false);

    vk_cmd_sig(cmd, sem_out);
    submit_cmd(p, &cmd);

    tex_vk->held = vk_flush_commands(vk);
    return tex_vk->held;
}

bool pl_vulkan_hold_raw(const struct pl_gpu *gpu, const struct pl_tex *tex,
                        VkImageLayout *layout, VkAccessFlags *access,
                        VkSemaphore sem_out)
{
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);
    bool user_may_invalidate = tex_vk->may_invalidate;

    if (!pl_vulkan_hold(gpu, tex, tex_vk->current_layout, tex_vk->current_access, sem_out))
        return false;

    if (user_may_invalidate) {
        *layout = VK_IMAGE_LAYOUT_UNDEFINED;
        *access = 0;
    } else {
        *layout = tex_vk->current_layout;
        *access = tex_vk->current_access;
    }

    return true;
}

void pl_vulkan_release(const struct pl_gpu *gpu, const struct pl_tex *tex,
                       VkImageLayout layout, VkAccessFlags access,
                       VkSemaphore sem_in)
{
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);
    if (!tex_vk->held) {
        PL_ERR(gpu, "Attempting to release an unheld image?");
        return;
    }

    pl_tex_vk_external_dep(gpu, tex, sem_in);

    tex_vk->current_layout = layout;
    tex_vk->current_access = access;
    tex_vk->held = false;
}

// For pl_buf.priv
struct pl_buf_vk {
    struct vk_memslice mem;
    pl_rc_t rc;
    int writes; // number of queued write commands
    enum queue_type update_queue;
    VkBufferView view; // for texel buffers
    // "current" metadata, can change during course of execution
    VkAccessFlags current_access;
    bool exported;
    bool needs_flush;
    // the signal guards reuse, and can be NULL
    struct vk_signal *sig;
    VkPipelineStageFlags sig_stage;
};

static void vk_buf_deref(const struct pl_gpu *gpu, const struct pl_buf *buf)
{
    if (!buf)
        return;

    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);

    if (pl_rc_deref(&buf_vk->rc)) {
        vk_signal_destroy(vk, &buf_vk->sig);
        vk->DestroyBufferView(vk->dev, buf_vk->view, PL_VK_ALLOC);
        vk_malloc_free(p->alloc, &buf_vk->mem);
        pl_free((void *) buf);
    }
}

static void vk_buf_finish_write(const struct pl_gpu *gpu, const struct pl_buf *buf)
{
    if (!buf)
        return;

    struct pl_buf_vk *buf_vk = PL_PRIV(buf);
    buf_vk->writes--;
}

enum buffer_op {
    BUF_READ    = (1 << 0),
    BUF_WRITE   = (1 << 1),
    BUF_EXPORT  = (1 << 2),
};

// offset: relative to pl_buf
static void buf_barrier(const struct pl_gpu *gpu, struct vk_cmd *cmd,
                        const struct pl_buf *buf, VkPipelineStageFlags stage,
                        VkAccessFlags newAccess, size_t offset, size_t size,
                        enum buffer_op op)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);
    pl_rc_ref(&buf_vk->rc);

    // CONCURRENT buffers require transitioning to/from IGNORED, EXCLUSIVE
    // buffers require transitioning to/from the concrete QF index
    uint32_t qf = vk->pools.num > 1 ? VK_QUEUE_FAMILY_IGNORED : cmd->pool->qf;

    VkBufferMemoryBarrier buffBarrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcQueueFamilyIndex = buf_vk->exported ? VK_QUEUE_FAMILY_EXTERNAL_KHR : qf,
        .dstQueueFamilyIndex = (op & BUF_EXPORT) ? VK_QUEUE_FAMILY_EXTERNAL_KHR : qf,
        .srcAccessMask = buf_vk->current_access,
        .dstAccessMask = newAccess,
        .buffer = buf_vk->mem.buf,
        .offset = buf_vk->mem.offset + offset,
        .size = size,
    };

    // Can't re-export exported buffers
    pl_assert(!(op & BUF_EXPORT) || !buf_vk->exported);

    VkEvent event = VK_NULL_HANDLE;
    enum vk_wait_type type = vk_cmd_wait(vk, cmd, &buf_vk->sig, stage, &event);
    VkPipelineStageFlags src_stages = 0;

    if (buf_vk->needs_flush || buf->params.host_mapped ||
        buf->params.import_handle == PL_HANDLE_HOST_PTR)
    {
        if (!buf_vk->exported) {
            buffBarrier.srcAccessMask |= VK_ACCESS_HOST_WRITE_BIT;
            src_stages |= VK_PIPELINE_STAGE_HOST_BIT;
        }

        if (buf_vk->mem.data && !buf_vk->mem.coherent) {
            if (buf_vk->exported) {
                // TODO: figure out and clean up the semantics?
                PL_WARN(vk, "Mixing host-mapped or user-writable buffers with "
                        "external APIs is risky and untested. If you run into "
                        "any issues, please try using a non-mapped buffer and "
                        "avoid pl_buf_write.");
            }

            VK(vk->FlushMappedMemoryRanges(vk->dev, 1, &(struct VkMappedMemoryRange) {
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .memory = buf_vk->mem.vkmem,
                .offset = buf_vk->mem.offset,
                .size = buf_vk->mem.size,
            }));

            // Just ignore errors, not much we can do about them other than
            // logging them and moving on...
        error: ;
        }

        // Forcibly degrade to non-event based pipeline barrier, because
        // mixing events with host writes is nonsensical
        if (type == VK_WAIT_EVENT)
            type = VK_WAIT_BARRIER;

        buf_vk->needs_flush = false;
    }

    if (buffBarrier.srcAccessMask != buffBarrier.dstAccessMask ||
        buffBarrier.srcQueueFamilyIndex != buffBarrier.dstQueueFamilyIndex)
    {
        switch (type) {
        case VK_WAIT_NONE:
            // No synchronization required, so we can safely transition out of
            // VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            buffBarrier.srcAccessMask = 0;
            src_stages |= VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            vk->CmdPipelineBarrier(cmd->buf, src_stages, stage, 0, 0, NULL,
                                   1, &buffBarrier, 0, NULL);
            break;
        case VK_WAIT_BARRIER:
            // Regular pipeline barrier is required
            vk->CmdPipelineBarrier(cmd->buf, buf_vk->sig_stage | src_stages,
                                   stage, 0, 0, NULL, 1, &buffBarrier, 0, NULL);
            break;
        case VK_WAIT_EVENT:
            // We can/should use the VkEvent for synchronization
            pl_assert(!src_stages);
            vk->CmdWaitEvents(cmd->buf, 1, &event, buf_vk->sig_stage,
                              stage, 0, NULL, 1, &buffBarrier, 0, NULL);
            break;
        }
    }

    if (op & BUF_WRITE) {
        buf_vk->writes++;
        vk_cmd_callback(cmd, (vk_cb) vk_buf_finish_write, gpu, buf);
    }

    buf_vk->current_access = newAccess;
    buf_vk->exported = (op & BUF_EXPORT);
    vk_cmd_callback(cmd, (vk_cb) vk_buf_deref, gpu, buf);
    vk_cmd_obj(cmd, buf);
}

static void buf_signal(const struct pl_gpu *gpu, struct vk_cmd *cmd,
                       const struct pl_buf *buf, VkPipelineStageFlags stage)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);
    pl_assert(!buf_vk->sig);

    buf_vk->sig = vk_cmd_signal(p->vk, cmd, stage);
    buf_vk->sig_stage = stage;
}

static void invalidate_memslice(struct vk_ctx *vk, const struct vk_memslice *mem)
{
    VK(vk->InvalidateMappedMemoryRanges(vk->dev, 1, &(VkMappedMemoryRange) {
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .memory = mem->vkmem,
        .offset = mem->offset,
        .size = mem->size,
    }));

    // Ignore errors (after logging), nothing useful we can do anyway
error: ;
}

// Flush visible writes to a buffer made by the API
// offset: relative to pl_buf
static void buf_flush(const struct pl_gpu *gpu, struct vk_cmd *cmd,
                      const struct pl_buf *buf, size_t offset, size_t size)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);

    // We need to perform a flush if the host is capable of reading back from
    // the buffer, or if we intend to overwrite it using mapped memory
    bool can_read = buf->params.host_readable;
    bool can_write = buf_vk->mem.data && buf->params.host_writable;
    if (buf->params.host_mapped || buf->params.import_handle == PL_HANDLE_HOST_PTR)
        can_read = can_write = true;

    if (!can_read && !can_write)
        return;

    VkBufferMemoryBarrier buffBarrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .srcAccessMask = buf_vk->current_access,
        .dstAccessMask = (can_read ? VK_ACCESS_HOST_READ_BIT : 0)
                       | (can_write ? VK_ACCESS_HOST_WRITE_BIT : 0),
        .buffer = buf_vk->mem.buf,
        .offset = buf_vk->mem.offset + offset,
        .size = size,
    };

    vk->CmdPipelineBarrier(cmd->buf, buf_vk->sig_stage,
                           VK_PIPELINE_STAGE_HOST_BIT, 0,
                           0, NULL, 1, &buffBarrier, 0, NULL);

    // Invalidate the mapped memory as soon as this barrier completes
    if (buf_vk->mem.data && !buf_vk->mem.coherent)
        vk_cmd_callback(cmd, (vk_cb) invalidate_memslice, vk, &buf_vk->mem);
}

static bool vk_buf_poll(const struct pl_gpu *gpu, const struct pl_buf *buf,
                        uint64_t timeout)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);

    // Opportunistically check if we can re-use this buffer without flush
    vk_poll_commands(vk, 0);
    if (pl_rc_count(&buf_vk->rc) == 1)
        return false;

    // Otherwise, we're force to submit all queued commands so that the
    // user is guaranteed to see progress eventually, even if they call
    // this in a tight loop
    flush(p);
    vk_flush_obj(vk, buf);
    vk_poll_commands(vk, timeout);

    return pl_rc_count(&buf_vk->rc) > 1;
}

static void vk_buf_write(const struct pl_gpu *gpu, const struct pl_buf *buf,
                         size_t offset, const void *data, size_t size)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);

    // For host-mapped buffers, we can just directly memcpy the buffer contents.
    // Otherwise, we can update the buffer from the GPU using a command buffer.
    if (buf_vk->mem.data) {
        // ensure no queued operations
        while (vk_buf_poll(gpu, buf, UINT64_MAX))
            ; // do nothing

        uintptr_t addr = (uintptr_t) buf_vk->mem.data + offset;
        memcpy((void *) addr, data, size);
        buf_vk->needs_flush = true;
    } else {
        struct vk_cmd *cmd = begin_cmd(p, buf_vk->update_queue);
        if (!cmd) {
            PL_ERR(gpu, "Failed updating buffer!");
            return;
        }

        buf_barrier(gpu, cmd, buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT, offset, size, BUF_WRITE);

        // Vulkan requires `size` to be a multiple of 4, so we need to make
        // sure to handle the end separately if the original data is not
        const size_t max_transfer = 64 * 1024;
        size_t size_rem = size % 4;
        size_t size_base = size - size_rem;
        VkDeviceSize buf_offset = buf_vk->mem.offset + offset;

        if (size_base > max_transfer) {
            PL_TRACE(gpu, "Using multiple vkCmdUpdateBuffer calls to upload "
                     "large buffer. Consider using buffer-buffer transfers "
                     "instead!");
        }

        for (size_t xfer = 0; xfer < size_base; xfer += max_transfer) {
            vk->CmdUpdateBuffer(cmd->buf, buf_vk->mem.buf,
                                buf_offset + xfer,
                                PL_MIN(size_base, max_transfer),
                                (void *) ((uint8_t *) data + xfer));
        }

        if (size_rem) {
            uint8_t tail[4] = {0};
            memcpy(tail, data, size_rem);
            vk->CmdUpdateBuffer(cmd->buf, buf_vk->mem.buf, buf_offset + size_base,
                                sizeof(tail), tail);
        }

        pl_assert(!buf->params.host_readable); // no flush needed due to this
        buf_signal(gpu, cmd, buf, VK_PIPELINE_STAGE_TRANSFER_BIT);
        finish_cmd(p, &cmd);
    }
}

static bool vk_buf_read(const struct pl_gpu *gpu, const struct pl_buf *buf,
                        size_t offset, void *dest, size_t size)
{
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);
    pl_assert(buf_vk->mem.data);

    // ensure no more queued writes
    while (buf_vk->writes)
        vk_buf_poll(gpu, buf, UINT64_MAX);

    uintptr_t addr = (uintptr_t) buf_vk->mem.data + (size_t) offset;
    memcpy(dest, (void *) addr, size);
    return true;
}

static void vk_buf_copy(const struct pl_gpu *gpu,
                        const struct pl_buf *dst, size_t dst_offset,
                        const struct pl_buf *src, size_t src_offset,
                        size_t size)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct pl_buf_vk *dst_vk = PL_PRIV(dst);
    struct pl_buf_vk *src_vk = PL_PRIV(src);

    struct vk_cmd *cmd = begin_cmd(p, dst_vk->update_queue);
    if (!cmd) {
        PL_ERR(gpu, "Failed copying buffer!");
        return;
    }

    buf_barrier(gpu, cmd, dst, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, dst_offset, size, BUF_WRITE);
    buf_barrier(gpu, cmd, src, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_READ_BIT, src_offset, size, BUF_READ);

    VkBufferCopy region = {
        .srcOffset = src_vk->mem.offset + src_offset,
        .dstOffset = dst_vk->mem.offset + dst_offset,
        .size = size,
    };

    vkCmdCopyBuffer(cmd->buf, src_vk->mem.buf, dst_vk->mem.buf,
                    1, &region);

    buf_signal(gpu, cmd, src, VK_PIPELINE_STAGE_TRANSFER_BIT);
    buf_signal(gpu, cmd, dst, VK_PIPELINE_STAGE_TRANSFER_BIT);
    buf_flush(gpu, cmd, dst, dst_offset, size);
    finish_cmd(p, &cmd);
}

static bool vk_buf_export(const struct pl_gpu *gpu, const struct pl_buf *buf)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);
    if (buf_vk->exported)
        return true;

    struct vk_cmd *cmd = begin_cmd(p, ANY);
    if (!cmd) {
        PL_ERR(gpu, "Failed exporting buffer!");
        return false;
    }

    // For the queue family ownership transfer, we can ignore all pipeline
    // stages since the synchronization via fences/semaphores is required
    buf_barrier(gpu, cmd, buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                0, buf->params.size, BUF_EXPORT);


    submit_cmd(p, &cmd);
    return vk_flush_commands(vk);
}

static const struct pl_buf *vk_buf_create(const struct pl_gpu *gpu,
                                          const struct pl_buf_params *params)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    struct pl_buf *buf = pl_zalloc_priv(NULL, struct pl_buf, struct pl_buf_vk);
    buf->params = *params;
    buf->params.initial_data = NULL;

    struct pl_buf_vk *buf_vk = PL_PRIV(buf);
    buf_vk->current_access = 0;
    pl_rc_init(&buf_vk->rc);

    struct vk_malloc_params mparams = {
        .reqs = {
            .size = PL_ALIGN2(params->size, 4), // for vk_buf_write
            .memoryTypeBits = UINT32_MAX,
            .alignment = 1,
        },
        // these are always set, because `vk_buf_copy` can always be used
        .buf_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .export_handle = params->export_handle,
        .import_handle = params->import_handle,
        .shared_mem = params->shared_mem,
    };

    // Mandatory/optimal buffer offset alignment
    VkDeviceSize *align = &mparams.reqs.alignment;
    VkDeviceSize extra_align = vk->limits.optimalBufferCopyOffsetAlignment;

    // Try and align all buffers to the minimum texel alignment, to make sure
    // tex_upload/tex_download always gets aligned buffer copies if possible
    extra_align = pl_lcm(extra_align, p->min_texel_alignment);

    enum pl_buf_mem_type mem_type = params->memory_type;
    bool is_texel = false;

    if (params->uniform) {
        mparams.buf_usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        *align = pl_lcm(*align, vk->limits.minUniformBufferOffsetAlignment);
        mem_type = PL_BUF_MEM_DEVICE;
        if (params->format) {
            mparams.buf_usage |= VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
            *align = pl_lcm(*align, vk->limits.minTexelBufferOffsetAlignment);
            is_texel = true;
        }
    }

    if (params->storable) {
        mparams.buf_usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        *align = pl_lcm(*align, vk->limits.minStorageBufferOffsetAlignment);
        buf_vk->update_queue = vk->pool_compute ? COMPUTE : GRAPHICS;
        mem_type = PL_BUF_MEM_DEVICE;
        if (params->format) {
            mparams.buf_usage |= VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
            *align = pl_lcm(*align, vk->limits.minTexelBufferOffsetAlignment);
            is_texel = true;
        }
    }

    if (params->drawable) {
        mparams.buf_usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                             VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        mem_type = PL_BUF_MEM_DEVICE;
    }

    if (params->host_writable || params->initial_data) {
        // Buffers should be written using mapped memory if possible
        mparams.optimal = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        // Use the transfer queue for updates on very large buffers (1 MB)
        if (params->size > 1024*1024)
            buf_vk->update_queue = TRANSFER;
    }

    if (params->host_mapped || params->host_readable) {
        mparams.required |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

        if (params->size > 1024) {
            // Require cached memory for large buffers (1 kB) which may be read
            // from, because uncached reads are extremely slow
            mparams.required |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        }
    }

    if (params->host_writable || params->host_readable) {
        // Prefer buffers requiring frequent host operations in host mem
        mem_type = PL_DEF(mem_type, PL_BUF_MEM_HOST);
    }

    switch (mem_type) {
    case PL_BUF_MEM_AUTO:
        // We generally prefer VRAM since it's faster than RAM, but any number
        // of other requirements could potentially exclude it, so just mark it
        // as optimal by default.
        mparams.optimal |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
    case PL_BUF_MEM_DEVICE:
        // Force device local memory.
        mparams.required |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
    case PL_BUF_MEM_HOST:
        // This isn't a true guarantee, but actually trying to restrict the
        // device-local bit locks out all memory heaps on iGPUs. Requiring
        // the memory be host-mapped is the easiest compromise.
        mparams.required |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        break;
    default: abort();
    }

    if (params->import_handle) {
        size_t offset = params->shared_mem.offset;
        if (PL_ALIGN(offset, *align) != offset) {
            PL_ERR(gpu, "Imported memory offset %zu violates minimum alignment "
                   "requirement of enabled usage flags (%zu)!",
                   offset, (size_t) *align);
            goto error;
        }
    } else {
        *align = pl_lcm(*align, extra_align);
    }

    if (!vk_malloc_slice(p->alloc, &buf_vk->mem, &mparams))
        goto error;

    if (params->host_mapped)
        buf->data = buf_vk->mem.data;

    if (params->export_handle) {
        buf->shared_mem = buf_vk->mem.shared_mem;
        buf->shared_mem.drm_format_mod = DRM_FORMAT_MOD_LINEAR;
        buf_vk->exported = true;
    }

    if (is_texel) {
        const struct vk_format **vk_fmt = PL_PRIV(params->format);
        VkBufferViewCreateInfo vinfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
            .buffer = buf_vk->mem.buf,
            .format = PL_DEF((*vk_fmt)->bfmt, (*vk_fmt)->tfmt),
            .offset = buf_vk->mem.offset,
            .range = buf_vk->mem.size,
        };

        VK(vk->CreateBufferView(vk->dev, &vinfo, PL_VK_ALLOC, &buf_vk->view));
        PL_VK_NAME(BUFFER_VIEW, buf_vk->view, "texel");
    }

    if (params->initial_data)
        vk_buf_write(gpu, buf, 0, params->initial_data, params->size);

    return buf;

error:
    vk_buf_deref(gpu, buf);
    return NULL;
}

static enum queue_type vk_img_copy_queue(const struct pl_gpu *gpu,
                                         const struct VkBufferImageCopy *region,
                                         const struct pl_tex *tex)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    const struct pl_tex_vk *tex_vk = PL_PRIV(tex);
    enum queue_type queue = tex_vk->transfer_queue;
    if (queue != TRANSFER || !vk->pool_transfer)
        return queue;

    VkExtent3D alignment = vk->pool_transfer->props.minImageTransferGranularity;

    enum queue_type fallback = GRAPHICS;
    if (gpu->caps & PL_GPU_CAP_PARALLEL_COMPUTE)
        fallback = COMPUTE; // prefer async compute queue

    int tex_w = PL_DEF(tex->params.w, 1),
        tex_h = PL_DEF(tex->params.h, 1),
        tex_d = PL_DEF(tex->params.d, 1);

    bool full_w = region->imageOffset.x + region->imageExtent.width  == tex_w,
         full_h = region->imageOffset.y + region->imageExtent.height == tex_h,
         full_d = region->imageOffset.z + region->imageExtent.depth  == tex_d;

    if (alignment.width) {

        bool unaligned = false;
        unaligned |= region->imageOffset.x % alignment.width;
        unaligned |= region->imageOffset.y % alignment.height;
        unaligned |= region->imageOffset.z % alignment.depth;
        unaligned |= (region->imageExtent.width  % alignment.width)  && !full_w;
        unaligned |= (region->imageExtent.height % alignment.height) && !full_h;
        unaligned |= (region->imageExtent.depth  % alignment.depth)  && !full_d;

        return unaligned ? fallback : queue;

    } else {

        // an alignment of {0} means the copy must span the entire image
        bool unaligned = false;
        unaligned |= region->imageOffset.x || !full_w;
        unaligned |= region->imageOffset.y || !full_h;
        unaligned |= region->imageOffset.z || !full_d;

        return unaligned ? fallback : queue;

    }
}

static void tex_xfer_cb(void *ctx, void *arg)
{
    void (*fun)(void *priv) = ctx;
    fun(arg);
}

static bool vk_tex_upload(const struct pl_gpu *gpu,
                          const struct pl_tex_transfer_params *params)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    const struct pl_tex *tex = params->tex;
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);

    if (!params->buf)
        return pl_tex_upload_pbo(gpu, params);

    const struct pl_buf *buf = params->buf;
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);
    struct pl_rect3d rc = params->rc;
    size_t size = pl_tex_transfer_size(params);

    size_t buf_offset = buf_vk->mem.offset + params->buf_offset;
    bool emulated = tex->params.format->emulated;
    bool unaligned = buf_offset % tex->params.format->texel_size;
    if (unaligned)
        PL_TRACE(gpu, "vk_tex_upload: unaligned transfer (slow path)");

    if (emulated || unaligned) {

        bool ubo;
        if (emulated) {
            if (size <= gpu->limits.max_ubo_size) {
                ubo = true;
            } else if (size <= gpu->limits.max_ssbo_size) {
                ubo = false;
            } else {
                // TODO: Implement strided upload path if really necessary
                PL_ERR(gpu, "Texel buffer size requirements exceed GPU "
                       "capabilities, failed uploading!");
                goto error;
            }
        }

        // Copy the source data buffer into an intermediate buffer
        const struct pl_buf *tbuf = pl_buf_create(gpu, &(struct pl_buf_params) {
            .uniform = emulated && ubo,
            .storable = emulated && !ubo,
            .size = size,
            .memory_type = PL_BUF_MEM_DEVICE,
            .format = tex_vk->texel_fmt,
        });

        if (!tbuf) {
            PL_ERR(gpu, "Failed creating buffer for tex upload fallback!");
            goto error;
        }

        struct vk_cmd *cmd = begin_cmd(p, tex_vk->transfer_queue);
        if (!cmd)
            goto error;

        vk_cmd_timer_begin(gpu, cmd, params->timer);

        struct pl_buf_vk *tbuf_vk = PL_PRIV(tbuf);
        VkBufferCopy region = {
            .srcOffset = buf_offset,
            .dstOffset = tbuf_vk->mem.offset,
            .size = size,
        };

        buf_barrier(gpu, cmd, buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT, params->buf_offset, size,
                    BUF_READ);
        buf_barrier(gpu, cmd, tbuf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT, 0, size, BUF_WRITE);
        vk->CmdCopyBuffer(cmd->buf, buf_vk->mem.buf, tbuf_vk->mem.buf,
                          1, &region);

        buf_signal(gpu, cmd, buf, VK_PIPELINE_STAGE_TRANSFER_BIT);
        buf_signal(gpu, cmd, tbuf, VK_PIPELINE_STAGE_TRANSFER_BIT);

        if (params->callback)
            vk_cmd_callback(cmd, tex_xfer_cb, params->callback, params->priv);

        vk_cmd_timer_end(gpu, cmd, params->timer);
        finish_cmd(p, &cmd);

        struct pl_tex_transfer_params fixed = *params;
        fixed.buf = tbuf;
        fixed.buf_offset = 0;

        bool ok = emulated ? pl_tex_upload_texel(gpu, p->dp, &fixed)
                           : pl_tex_upload(gpu, &fixed);

        pl_buf_destroy(gpu, &tbuf);
        return ok;

    } else {

        VkBufferImageCopy region = {
            .bufferOffset = buf_offset,
            .bufferRowLength = params->stride_w,
            .bufferImageHeight = params->stride_h,
            .imageOffset = { rc.x0, rc.y0, rc.z0 },
            .imageExtent = { rc.x1, rc.y1, rc.z1 },
            .imageSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .layerCount = 1,
            },
        };

        enum queue_type queue = vk_img_copy_queue(gpu, &region, tex);
        struct vk_cmd *cmd = begin_cmd(p, queue);
        if (!cmd)
            goto error;

        vk_cmd_timer_begin(gpu, cmd, params->timer);

        buf_barrier(gpu, cmd, buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT, params->buf_offset, size,
                    BUF_READ);
        tex_barrier(gpu, cmd, tex, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, false);
        vk->CmdCopyBufferToImage(cmd->buf, buf_vk->mem.buf, tex_vk->img,
                                 tex_vk->current_layout, 1, &region);
        buf_signal(gpu, cmd, buf, VK_PIPELINE_STAGE_TRANSFER_BIT);
        tex_signal(gpu, cmd, tex, VK_PIPELINE_STAGE_TRANSFER_BIT);

        if (params->callback)
            vk_cmd_callback(cmd, tex_xfer_cb, params->callback, params->priv);

        vk_cmd_timer_end(gpu, cmd, params->timer);
        finish_cmd(p, &cmd);
    }

    return true;

error:
    return false;
}

static bool vk_tex_download(const struct pl_gpu *gpu,
                            const struct pl_tex_transfer_params *params)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    const struct pl_tex *tex = params->tex;
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);

    if (!params->buf)
        return pl_tex_download_pbo(gpu, params);

    const struct pl_buf *buf = params->buf;
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);
    struct pl_rect3d rc = params->rc;
    size_t size = pl_tex_transfer_size(params);

    size_t buf_offset = buf_vk->mem.offset + params->buf_offset;
    bool emulated = tex->params.format->emulated;
    bool unaligned = buf_offset % tex->params.format->texel_size;
    if (unaligned)
        PL_TRACE(gpu, "vk_tex_download: unaligned transfer (slow path)");

    if (emulated || unaligned) {

        // Download into an intermediate buffer first
        const struct pl_buf *tbuf = pl_buf_create(gpu, &(struct pl_buf_params) {
            .storable = emulated,
            .size = size,
            .memory_type = PL_BUF_MEM_DEVICE,
            .format = tex_vk->texel_fmt,
        });

        if (!tbuf) {
            PL_ERR(gpu, "Failed creating buffer for tex download fallback!");
            goto error;
        }

        struct pl_tex_transfer_params fixed = *params;
        fixed.buf = tbuf;
        fixed.buf_offset = 0;

        bool ok = emulated ? pl_tex_download_texel(gpu, p->dp, &fixed)
                           : pl_tex_download(gpu, &fixed);
        if (!ok) {
            pl_buf_destroy(gpu, &tbuf);
            goto error;
        }

        struct vk_cmd *cmd = begin_cmd(p, tex_vk->transfer_queue);
        if (!cmd) {
            pl_buf_destroy(gpu, &tbuf);
            goto error;
        }

        vk_cmd_timer_begin(gpu, cmd, params->timer);

        struct pl_buf_vk *tbuf_vk = PL_PRIV(tbuf);
        VkBufferCopy region = {
            .srcOffset = tbuf_vk->mem.offset,
            .dstOffset = buf_offset,
            .size = size,
        };

        buf_barrier(gpu, cmd, tbuf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT, 0, size, BUF_READ);
        buf_barrier(gpu, cmd, buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT, params->buf_offset, size,
                    BUF_WRITE);
        vk->CmdCopyBuffer(cmd->buf, tbuf_vk->mem.buf, buf_vk->mem.buf,
                          1, &region);
        buf_signal(gpu, cmd, tbuf, VK_PIPELINE_STAGE_TRANSFER_BIT);
        buf_signal(gpu, cmd, buf, VK_PIPELINE_STAGE_TRANSFER_BIT);
        buf_flush(gpu, cmd, buf, params->buf_offset, size);

        if (params->callback)
            vk_cmd_callback(cmd, tex_xfer_cb, params->callback, params->priv);


        vk_cmd_timer_end(gpu, cmd, params->timer);
        finish_cmd(p, &cmd);

        pl_buf_destroy(gpu, &tbuf);

    } else {

        VkBufferImageCopy region = {
            .bufferOffset = buf_offset,
            .bufferRowLength = params->stride_w,
            .bufferImageHeight = params->stride_h,
            .imageOffset = { rc.x0, rc.y0, rc.z0 },
            .imageExtent = { rc.x1, rc.y1, rc.z1 },
            .imageSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .layerCount = 1,
            },
        };

        enum queue_type queue = vk_img_copy_queue(gpu, &region, tex);

        struct vk_cmd *cmd = begin_cmd(p, queue);
        if (!cmd)
            goto error;

        vk_cmd_timer_begin(gpu, cmd, params->timer);

        buf_barrier(gpu, cmd, buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT, params->buf_offset, size,
                    BUF_WRITE);
        tex_barrier(gpu, cmd, tex, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, false);
        vk->CmdCopyImageToBuffer(cmd->buf, tex_vk->img, tex_vk->current_layout,
                                 buf_vk->mem.buf, 1, &region);
        buf_signal(gpu, cmd, buf, VK_PIPELINE_STAGE_TRANSFER_BIT);
        tex_signal(gpu, cmd, tex, VK_PIPELINE_STAGE_TRANSFER_BIT);
        buf_flush(gpu, cmd, buf, params->buf_offset, size);

        if (params->callback)
            vk_cmd_callback(cmd, tex_xfer_cb, params->callback, params->priv);


        vk_cmd_timer_end(gpu, cmd, params->timer);
        finish_cmd(p, &cmd);
    }

    return true;

error:
    return false;
}

static int vk_desc_namespace(const struct pl_gpu *gpu, enum pl_desc_type type)
{
    return 0;
}

// For pl_pass.priv
struct pl_pass_vk {
    // Pipeline / render pass
    VkPipeline pipe;
    VkPipelineLayout pipeLayout;
    VkRenderPass renderPass;
    VkImageLayout initialLayout;
    VkImageLayout finalLayout;
    // Descriptor set (bindings)
    bool use_pushd;
    VkDescriptorSetLayout dsLayout;
    VkDescriptorPool dsPool;
    // To keep track of which descriptor sets are and aren't available, we
    // allocate a fixed number and use a bitmask of all available sets.
    VkDescriptorSet dss[16];
    uint16_t dmask;

    // For updating
    VkWriteDescriptorSet *dswrite;
    VkDescriptorImageInfo *dsiinfo;
    VkDescriptorBufferInfo *dsbinfo;
};

static void vk_pass_destroy(const struct pl_gpu *gpu, struct pl_pass *pass)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_pass_vk *pass_vk = PL_PRIV(pass);

    vk->DestroyPipeline(vk->dev, pass_vk->pipe, PL_VK_ALLOC);
    vk->DestroyRenderPass(vk->dev, pass_vk->renderPass, PL_VK_ALLOC);
    vk->DestroyPipelineLayout(vk->dev, pass_vk->pipeLayout, PL_VK_ALLOC);
    vk->DestroyDescriptorPool(vk->dev, pass_vk->dsPool, PL_VK_ALLOC);
    vk->DestroyDescriptorSetLayout(vk->dev, pass_vk->dsLayout, PL_VK_ALLOC);

    pl_free(pass);
}

MAKE_LAZY_DESTRUCTOR(vk_pass_destroy, const struct pl_pass)

static const VkDescriptorType dsType[] = {
    [PL_DESC_SAMPLED_TEX] = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    [PL_DESC_STORAGE_IMG] = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    [PL_DESC_BUF_UNIFORM] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    [PL_DESC_BUF_STORAGE] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    [PL_DESC_BUF_TEXEL_UNIFORM] = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
    [PL_DESC_BUF_TEXEL_STORAGE] = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
};

#define CACHE_MAGIC {'R','A','V','K'}
#define CACHE_VERSION 2
static const char vk_cache_magic[4] = CACHE_MAGIC;

struct vk_cache_header {
    char magic[sizeof(vk_cache_magic)];
    int cache_version;
    char compiler[SPIRV_NAME_MAX_LEN];
    int compiler_version;
    size_t vert_spirv_len;
    size_t frag_spirv_len;
    size_t comp_spirv_len;
    size_t pipecache_len;
};

static bool vk_use_cached_program(const struct pl_pass_params *params,
                                  const struct spirv_compiler *spirv,
                                  pl_str *vert_spirv,
                                  pl_str *frag_spirv,
                                  pl_str *comp_spirv,
                                  pl_str *pipecache)
{
    pl_str cache = {
        .buf = (uint8_t *) params->cached_program,
        .len = params->cached_program_len,
    };

    if (cache.len < sizeof(struct vk_cache_header))
        return false;

    struct vk_cache_header *header = (struct vk_cache_header *) cache.buf;
    cache = pl_str_drop(cache, sizeof(*header));

    if (strncmp(header->magic, vk_cache_magic, sizeof(vk_cache_magic)) != 0)
        return false;
    if (header->cache_version != CACHE_VERSION)
        return false;
    if (strncmp(header->compiler, spirv->name, sizeof(header->compiler)) != 0)
        return false;
    if (header->compiler_version != spirv->compiler_version)
        return false;

#define GET(ptr)                                        \
        if (cache.len < header->ptr##_len)              \
            return false;                               \
        *ptr = pl_str_take(cache, header->ptr##_len);   \
        cache = pl_str_drop(cache, ptr->len);

    GET(vert_spirv);
    GET(frag_spirv);
    GET(comp_spirv);
    GET(pipecache);
    return true;
}

static VkResult vk_compile_glsl(const struct pl_gpu *gpu, void *alloc,
                                enum glsl_shader_stage type, const char *glsl,
                                pl_str *spirv)
{
    struct pl_vk *p = PL_PRIV(gpu);

    static const char *shader_names[] = {
        [GLSL_SHADER_VERTEX]   = "vertex",
        [GLSL_SHADER_FRAGMENT] = "fragment",
        [GLSL_SHADER_COMPUTE]  = "compute",
    };

    PL_DEBUG(gpu, "%s shader source:", shader_names[type]);
    pl_msg_source(gpu->ctx, PL_LOG_DEBUG, glsl);

    if (!p->spirv->impl->compile_glsl(p->spirv, alloc, type, glsl, spirv)) {
        pl_msg_source(gpu->ctx, PL_LOG_ERR, glsl);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    return VK_SUCCESS;
}

static const VkShaderStageFlags stageFlags[] = {
    [PL_PASS_RASTER]  = VK_SHADER_STAGE_FRAGMENT_BIT |
                        VK_SHADER_STAGE_VERTEX_BIT,
    [PL_PASS_COMPUTE] = VK_SHADER_STAGE_COMPUTE_BIT,
};

static const struct pl_pass *vk_pass_create(const struct pl_gpu *gpu,
                                            const struct pl_pass_params *params)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    bool success = false;

    struct pl_pass *pass = pl_zalloc_priv(NULL, struct pl_pass, struct pl_pass_vk);
    pass->params = pl_pass_params_copy(pass, params);

    struct pl_pass_vk *pass_vk = PL_PRIV(pass);
    pass_vk->dmask = -1; // all descriptors available

    // temporary allocations
    void *tmp = pl_tmp(NULL);

    VkPipelineCache pipeCache = VK_NULL_HANDLE;
    VkShaderModule vert_shader = VK_NULL_HANDLE;
    VkShaderModule frag_shader = VK_NULL_HANDLE;
    VkShaderModule comp_shader = VK_NULL_HANDLE;

    int num_desc = params->num_descriptors;
    if (!num_desc)
        goto no_descriptors;

    pass_vk->dswrite = pl_calloc(pass, num_desc, sizeof(VkWriteDescriptorSet));
    pass_vk->dsiinfo = pl_calloc(pass, num_desc, sizeof(VkDescriptorImageInfo));
    pass_vk->dsbinfo = pl_calloc(pass, num_desc, sizeof(VkDescriptorBufferInfo));

#define NUM_DS (PL_ARRAY_SIZE(pass_vk->dss))

    static int dsSize[PL_DESC_TYPE_COUNT] = {0};
    VkDescriptorSetLayoutBinding *bindings = pl_calloc_ptr(tmp, num_desc, bindings);

    for (int i = 0; i < num_desc; i++) {
        struct pl_desc *desc = &params->descriptors[i];

        dsSize[desc->type]++;
        bindings[i] = (VkDescriptorSetLayoutBinding) {
            .binding = desc->binding,
            .descriptorType = dsType[desc->type],
            .descriptorCount = 1,
            .stageFlags = stageFlags[params->type],
        };
    }

    VkDescriptorSetLayoutCreateInfo dinfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pBindings = bindings,
        .bindingCount = num_desc,
    };

    if (p->max_push_descriptors && num_desc <= p->max_push_descriptors) {
        dinfo.flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        pass_vk->use_pushd = true;
    } else if (p->max_push_descriptors) {
        PL_INFO(gpu, "Pass with %d descriptors exceeds the maximum push "
                "descriptor count (%d). Falling back to descriptor sets!",
                num_desc, p->max_push_descriptors);
    }

    VK(vk->CreateDescriptorSetLayout(vk->dev, &dinfo, PL_VK_ALLOC,
                                     &pass_vk->dsLayout));

    if (!pass_vk->use_pushd) {
        PL_ARRAY(VkDescriptorPoolSize) dsPoolSizes = {0};

        for (enum pl_desc_type t = 0; t < PL_DESC_TYPE_COUNT; t++) {
            if (dsSize[t] > 0) {
                PL_ARRAY_APPEND(tmp, dsPoolSizes, (VkDescriptorPoolSize) {
                    .type = dsType[t],
                    .descriptorCount = dsSize[t] * NUM_DS,
                });
            }
        }

        if (dsPoolSizes.num) {
            VkDescriptorPoolCreateInfo pinfo = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = NUM_DS,
                .pPoolSizes = dsPoolSizes.elem,
                .poolSizeCount = dsPoolSizes.num,
            };

            VK(vk->CreateDescriptorPool(vk->dev, &pinfo, PL_VK_ALLOC, &pass_vk->dsPool));

            VkDescriptorSetLayout layouts[NUM_DS];
            for (int i = 0; i < NUM_DS; i++)
                layouts[i] = pass_vk->dsLayout;

            VkDescriptorSetAllocateInfo ainfo = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = pass_vk->dsPool,
                .descriptorSetCount = NUM_DS,
                .pSetLayouts = layouts,
            };

            VK(vk->AllocateDescriptorSets(vk->dev, &ainfo, pass_vk->dss));
        }
    }

no_descriptors: ;

    VkPipelineLayoutCreateInfo linfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = num_desc ? 1 : 0,
        .pSetLayouts = &pass_vk->dsLayout,
        .pushConstantRangeCount = params->push_constants_size ? 1 : 0,
        .pPushConstantRanges = &(VkPushConstantRange){
            .stageFlags = stageFlags[params->type],
            .offset = 0,
            .size = params->push_constants_size,
        },
    };

    VK(vk->CreatePipelineLayout(vk->dev, &linfo, PL_VK_ALLOC,
                                &pass_vk->pipeLayout));

    pl_str vert = {0}, frag = {0}, comp = {0}, pipecache = {0};
    if (vk_use_cached_program(params, p->spirv, &vert, &frag, &comp, &pipecache)) {
        PL_DEBUG(gpu, "Using cached SPIR-V and VkPipeline");
    } else {
        pipecache.len = 0;
        switch (params->type) {
        case PL_PASS_RASTER:
            VK(vk_compile_glsl(gpu, tmp, GLSL_SHADER_VERTEX,
                               params->vertex_shader, &vert));
            VK(vk_compile_glsl(gpu, tmp, GLSL_SHADER_FRAGMENT,
                               params->glsl_shader, &frag));
            comp.len = 0;
            break;
        case PL_PASS_COMPUTE:
            VK(vk_compile_glsl(gpu, tmp, GLSL_SHADER_COMPUTE,
                               params->glsl_shader, &comp));
            frag.len = 0;
            vert.len = 0;
            break;
        default: abort();
        }
    }

    VkPipelineCacheCreateInfo pcinfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        .pInitialData = pipecache.buf,
        .initialDataSize = pipecache.len,
    };

    VK(vk->CreatePipelineCache(vk->dev, &pcinfo, PL_VK_ALLOC, &pipeCache));

    VkShaderModuleCreateInfo sinfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    };

    switch (params->type) {
    case PL_PASS_RASTER: {
        sinfo.pCode = (uint32_t *) vert.buf;
        sinfo.codeSize = vert.len;
        VK(vk->CreateShaderModule(vk->dev, &sinfo, PL_VK_ALLOC, &vert_shader));
        PL_VK_NAME(SHADER_MODULE, vert_shader, "vertex");

        sinfo.pCode = (uint32_t *) frag.buf;
        sinfo.codeSize = frag.len;
        VK(vk->CreateShaderModule(vk->dev, &sinfo, PL_VK_ALLOC, &frag_shader));
        PL_VK_NAME(SHADER_MODULE, frag_shader, "fragment");

        VkVertexInputAttributeDescription *attrs =
            pl_calloc_ptr(tmp, params->num_vertex_attribs, attrs);

        for (int i = 0; i < params->num_vertex_attribs; i++) {
            struct pl_vertex_attrib *va = &params->vertex_attribs[i];
            const struct vk_format **pfmt_vk = PL_PRIV(va->fmt);

            attrs[i] = (VkVertexInputAttributeDescription) {
                .binding  = 0,
                .location = va->location,
                .offset   = va->offset,
                .format   = PL_DEF((*pfmt_vk)->bfmt, (*pfmt_vk)->tfmt),
            };
        }

        pass_vk->finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        // Figure out which case we should try and optimize for based on some
        // dumb heuristics. Extremely naive, but good enough for most cases.
        struct pl_tex_params texparams = params->target_dummy.params;
        if (texparams.sampleable)
            pass_vk->finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        if (texparams.blit_src || texparams.host_readable)
            pass_vk->finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

        // Assume we're ping-ponging between a render pass and some other
        // operation. This is the most likely scenario, or rather, the only one
        // we can really optimize for.
        pass_vk->initialLayout = pass_vk->finalLayout;

        VkAttachmentLoadOp loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;

        // If we're blending, then we need to explicitly load the previous
        // contents of the color attachment
        if (pass->params.blend_params)
            loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;

        // If we're ignoring the FBO, we don't need to load or transition
        if (!pass->params.load_target) {
            pass_vk->initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        }

        VK(vk_create_render_pass(vk, texparams.format, loadOp,
                                 pass_vk->initialLayout, pass_vk->finalLayout,
                                 &pass_vk->renderPass));

        static const VkBlendFactor blendFactors[] = {
            [PL_BLEND_ZERO]                = VK_BLEND_FACTOR_ZERO,
            [PL_BLEND_ONE]                 = VK_BLEND_FACTOR_ONE,
            [PL_BLEND_SRC_ALPHA]           = VK_BLEND_FACTOR_SRC_ALPHA,
            [PL_BLEND_ONE_MINUS_SRC_ALPHA] = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        };

        VkPipelineColorBlendAttachmentState blendState = {
            .colorBlendOp = VK_BLEND_OP_ADD,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                              VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT |
                              VK_COLOR_COMPONENT_A_BIT,
        };

        const struct pl_blend_params *blend = params->blend_params;
        if (blend) {
            blendState.blendEnable = true;
            blendState.srcColorBlendFactor = blendFactors[blend->src_rgb];
            blendState.dstColorBlendFactor = blendFactors[blend->dst_rgb];
            blendState.srcAlphaBlendFactor = blendFactors[blend->src_alpha];
            blendState.dstAlphaBlendFactor = blendFactors[blend->dst_alpha];
        }

        static const VkPrimitiveTopology topologies[] = {
            [PL_PRIM_TRIANGLE_LIST]  = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            [PL_PRIM_TRIANGLE_STRIP] = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
            [PL_PRIM_TRIANGLE_FAN]   = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
        };

        VkGraphicsPipelineCreateInfo cinfo = {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = (VkPipelineShaderStageCreateInfo[]) {
                {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_VERTEX_BIT,
                    .module = vert_shader,
                    .pName = "main",
                }, {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                    .module = frag_shader,
                    .pName = "main",
                }
            },
            .pVertexInputState = &(VkPipelineVertexInputStateCreateInfo) {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &(VkVertexInputBindingDescription) {
                    .binding = 0,
                    .stride = params->vertex_stride,
                    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
                },
                .vertexAttributeDescriptionCount = params->num_vertex_attribs,
                .pVertexAttributeDescriptions = attrs,
            },
            .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo) {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .topology = topologies[params->vertex_type],
            },
            .pViewportState = &(VkPipelineViewportStateCreateInfo) {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .viewportCount = 1,
                .scissorCount = 1,
            },
            .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo) {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = VK_CULL_MODE_NONE,
                .lineWidth = 1.0f,
            },
            .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo) {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            },
            .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo) {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .attachmentCount = 1,
                .pAttachments = &blendState,
            },
            .pDynamicState = &(VkPipelineDynamicStateCreateInfo) {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                .dynamicStateCount = 2,
                .pDynamicStates = (VkDynamicState[]){
                    VK_DYNAMIC_STATE_VIEWPORT,
                    VK_DYNAMIC_STATE_SCISSOR,
                },
            },
            .layout = pass_vk->pipeLayout,
            .renderPass = pass_vk->renderPass,
        };

        VK(vk->CreateGraphicsPipelines(vk->dev, pipeCache, 1, &cinfo,
                                       PL_VK_ALLOC, &pass_vk->pipe));
        break;
    }
    case PL_PASS_COMPUTE: {
        sinfo.pCode = (uint32_t *) comp.buf;
        sinfo.codeSize = comp.len;
        VK(vk->CreateShaderModule(vk->dev, &sinfo, PL_VK_ALLOC, &comp_shader));
        PL_VK_NAME(SHADER_MODULE, comp_shader, "compute");

        VkComputePipelineCreateInfo cinfo = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = comp_shader,
                .pName = "main",
            },
            .layout = pass_vk->pipeLayout,
        };

        VK(vk->CreateComputePipelines(vk->dev, pipeCache, 1, &cinfo,
                                      PL_VK_ALLOC, &pass_vk->pipe));
        break;
    }
    default: abort();
    }

    // Update params->cached_program
    pl_str cache = {0};
    VK(vk->GetPipelineCacheData(vk->dev, pipeCache, &cache.len, NULL));
    cache.buf = pl_alloc(tmp, cache.len);
    VK(vk->GetPipelineCacheData(vk->dev, pipeCache, &cache.len, cache.buf));

    struct vk_cache_header header = {
        .magic = CACHE_MAGIC,
        .cache_version = CACHE_VERSION,
        .compiler_version = p->spirv->compiler_version,
        .vert_spirv_len = vert.len,
        .frag_spirv_len = frag.len,
        .comp_spirv_len = comp.len,
        .pipecache_len = cache.len,
    };

    PL_DEBUG(vk, "Pass statistics: size %zu, SPIR-V: vert %zu frag %zu comp %zu",
             cache.len, vert.len, frag.len, comp.len);

    for (int i = 0; i < sizeof(p->spirv->name); i++)
        header.compiler[i] = p->spirv->name[i];

    pl_str prog = {0};
    pl_str_append(pass, &prog, (pl_str){ (char *) &header, sizeof(header) });
    pl_str_append(pass, &prog, vert);
    pl_str_append(pass, &prog, frag);
    pl_str_append(pass, &prog, comp);
    pl_str_append(pass, &prog, cache);
    pass->params.cached_program = prog.buf;
    pass->params.cached_program_len = prog.len;

    success = true;

error:
    if (!success) {
        vk_pass_destroy(gpu, pass);
        pass = NULL;
    }

#undef NUM_DS

    vk->DestroyShaderModule(vk->dev, vert_shader, PL_VK_ALLOC);
    vk->DestroyShaderModule(vk->dev, frag_shader, PL_VK_ALLOC);
    vk->DestroyShaderModule(vk->dev, comp_shader, PL_VK_ALLOC);
    vk->DestroyPipelineCache(vk->dev, pipeCache, PL_VK_ALLOC);
    pl_free(tmp);
    return pass;
}

static const VkPipelineStageFlags passStages[] = {
    [PL_PASS_RASTER]  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
    [PL_PASS_COMPUTE] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
};

static void vk_update_descriptor(const struct pl_gpu *gpu, struct vk_cmd *cmd,
                                 const struct pl_pass *pass,
                                 struct pl_desc_binding db,
                                 VkDescriptorSet ds, int idx)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct pl_pass_vk *pass_vk = PL_PRIV(pass);
    struct pl_desc *desc = &pass->params.descriptors[idx];

    VkWriteDescriptorSet *wds = &pass_vk->dswrite[idx];
    *wds = (VkWriteDescriptorSet) {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = desc->binding,
        .descriptorCount = 1,
        .descriptorType = dsType[desc->type],
    };

    VkAccessFlags access = 0;
    enum buffer_op buf_op = 0;
    switch (desc->access) {
    case PL_DESC_ACCESS_READONLY:
        access = VK_ACCESS_SHADER_READ_BIT;
        buf_op = BUF_READ;
        break;
    case PL_DESC_ACCESS_WRITEONLY:
        access = VK_ACCESS_SHADER_WRITE_BIT;
        buf_op = BUF_WRITE;
        break;
    case PL_DESC_ACCESS_READWRITE:
        access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        buf_op = BUF_READ | BUF_WRITE;
        break;
    default: abort();
    }

    switch (desc->type) {
    case PL_DESC_SAMPLED_TEX: {
        const struct pl_tex *tex = db.object;
        struct pl_tex_vk *tex_vk = PL_PRIV(tex);

        tex_barrier(gpu, cmd, tex, passStages[pass->params.type],
                    VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, false);

        VkDescriptorImageInfo *iinfo = &pass_vk->dsiinfo[idx];
        *iinfo = (VkDescriptorImageInfo) {
            .sampler = p->samplers[db.sample_mode][db.address_mode],
            .imageView = tex_vk->view,
            .imageLayout = tex_vk->current_layout,
        };

        wds->pImageInfo = iinfo;
        break;
    }
    case PL_DESC_STORAGE_IMG: {
        const struct pl_tex *tex = db.object;
        struct pl_tex_vk *tex_vk = PL_PRIV(tex);

        tex_barrier(gpu, cmd, tex, passStages[pass->params.type], access,
                    VK_IMAGE_LAYOUT_GENERAL, false);

        VkDescriptorImageInfo *iinfo = &pass_vk->dsiinfo[idx];
        *iinfo = (VkDescriptorImageInfo) {
            .imageView = tex_vk->view,
            .imageLayout = tex_vk->current_layout,
        };

        wds->pImageInfo = iinfo;
        break;
    }
    case PL_DESC_BUF_UNIFORM:
    case PL_DESC_BUF_STORAGE: {
        const struct pl_buf *buf = db.object;
        struct pl_buf_vk *buf_vk = PL_PRIV(buf);

        buf_barrier(gpu, cmd, buf, passStages[pass->params.type],
                    access, 0, buf->params.size, buf_op);

        VkDescriptorBufferInfo *binfo = &pass_vk->dsbinfo[idx];
        *binfo = (VkDescriptorBufferInfo) {
            .buffer = buf_vk->mem.buf,
            .offset = buf_vk->mem.offset,
            .range = buf->params.size,
        };

        wds->pBufferInfo = binfo;
        break;
    }
    case PL_DESC_BUF_TEXEL_UNIFORM:
    case PL_DESC_BUF_TEXEL_STORAGE: {
        const struct pl_buf *buf = db.object;
        struct pl_buf_vk *buf_vk = PL_PRIV(buf);

        buf_barrier(gpu, cmd, buf, passStages[pass->params.type],
                    access, 0, buf->params.size, buf_op);

        wds->pTexelBufferView = &buf_vk->view;
        break;
    }
    default: abort();
    }
}

static void vk_release_descriptor(const struct pl_gpu *gpu, struct vk_cmd *cmd,
                                  const struct pl_pass *pass,
                                  struct pl_desc_binding db, int idx)
{
    const struct pl_desc *desc = &pass->params.descriptors[idx];

    switch (desc->type) {
    case PL_DESC_BUF_UNIFORM:
    case PL_DESC_BUF_STORAGE:
    case PL_DESC_BUF_TEXEL_UNIFORM:
    case PL_DESC_BUF_TEXEL_STORAGE: {
        const struct pl_buf *buf = db.object;
        buf_signal(gpu, cmd, buf, passStages[pass->params.type]);
        if (desc->access != PL_DESC_ACCESS_READONLY)
            buf_flush(gpu, cmd, buf, 0, buf->params.size);
        break;
    }
    case PL_DESC_SAMPLED_TEX:
    case PL_DESC_STORAGE_IMG: {
        const struct pl_tex *tex = db.object;
        tex_signal(gpu, cmd, tex, passStages[pass->params.type]);
        break;
    }
    default: break;
    }
}

static void set_ds(struct pl_pass_vk *pass_vk, void *dsbit)
{
    pass_vk->dmask |= (uintptr_t) dsbit;
}

static void vk_pass_run(const struct pl_gpu *gpu,
                        const struct pl_pass_run_params *params)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    const struct pl_pass *pass = params->pass;
    struct pl_pass_vk *pass_vk = PL_PRIV(pass);

    if (params->vertex_data || params->index_data)
        return pl_pass_run_vbo(gpu, params);

    if (!pass_vk->use_pushd) {
        // Wait for a free descriptor set
        while (!pass_vk->dmask) {
            PL_TRACE(gpu, "No free descriptor sets! ...blocking (slow path)");
            vk_flush_obj(vk, pass);
            vk_poll_commands(vk, 10000000); // 10 ms
        }
    }

    static const enum queue_type types[] = {
        [PL_PASS_RASTER]  = GRAPHICS,
        [PL_PASS_COMPUTE] = COMPUTE,
    };

    struct vk_cmd *cmd = begin_cmd(p, types[pass->params.type]);
    if (!cmd)
        goto error;

    vk_cmd_timer_begin(gpu, cmd, params->timer);

    // Find a descriptor set to use
    VkDescriptorSet ds = VK_NULL_HANDLE;
    if (!pass_vk->use_pushd) {
        for (int i = 0; i < PL_ARRAY_SIZE(pass_vk->dss); i++) {
            uint16_t dsbit = 1u << i;
            if (pass_vk->dmask & dsbit) {
                ds = pass_vk->dss[i];
                pass_vk->dmask &= ~dsbit; // unset
                vk_cmd_obj(cmd, pass);
                vk_cmd_callback(cmd, (vk_cb) set_ds, pass_vk,
                                (void *)(uintptr_t) dsbit);
                break;
            }
        }
    }

    // Update the dswrite structure with all of the new values
    for (int i = 0; i < pass->params.num_descriptors; i++)
        vk_update_descriptor(gpu, cmd, pass, params->desc_bindings[i], ds, i);

    if (!pass_vk->use_pushd) {
        vk->UpdateDescriptorSets(vk->dev, pass->params.num_descriptors,
                                 pass_vk->dswrite, 0, NULL);
    }

    // Bind the pipeline, descriptor set, etc.
    static const VkPipelineBindPoint bindPoint[] = {
        [PL_PASS_RASTER]  = VK_PIPELINE_BIND_POINT_GRAPHICS,
        [PL_PASS_COMPUTE] = VK_PIPELINE_BIND_POINT_COMPUTE,
    };

    vk->CmdBindPipeline(cmd->buf, bindPoint[pass->params.type], pass_vk->pipe);

    if (ds) {
        vk->CmdBindDescriptorSets(cmd->buf, bindPoint[pass->params.type],
                                  pass_vk->pipeLayout, 0, 1, &ds, 0, NULL);
    }

    if (pass_vk->use_pushd) {
        vk->CmdPushDescriptorSetKHR(cmd->buf, bindPoint[pass->params.type],
                                    pass_vk->pipeLayout, 0,
                                    pass->params.num_descriptors,
                                    pass_vk->dswrite);
    }

    if (pass->params.push_constants_size) {
        vk->CmdPushConstants(cmd->buf, pass_vk->pipeLayout,
                             stageFlags[pass->params.type], 0,
                             pass->params.push_constants_size,
                             params->push_constants);
    }

    switch (pass->params.type) {
    case PL_PASS_RASTER: {
        const struct pl_tex *tex = params->target;
        struct pl_tex_vk *tex_vk = PL_PRIV(tex);
        const struct pl_buf *vert = params->vertex_buf;
        struct pl_buf_vk *vert_vk = PL_PRIV(vert);
        const struct pl_buf *index = params->index_buf;
        struct pl_buf_vk *index_vk = index ? PL_PRIV(index) : NULL;

        // In the edge case that vert = index buffer, we need to synchronize
        // for both flags simultaneously
        VkAccessFlags vbo_flags = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        if (index == vert)
            vbo_flags |= VK_ACCESS_INDEX_READ_BIT;

        buf_barrier(gpu, cmd, vert, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                    vbo_flags, 0, vert->params.size, BUF_READ);

        VkDeviceSize offset = vert_vk->mem.offset + params->buf_offset;
        vk->CmdBindVertexBuffers(cmd->buf, 0, 1, &vert_vk->mem.buf, &offset);

        if (index) {
            if (index != vert) {
                buf_barrier(gpu, cmd, index, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                            VK_ACCESS_INDEX_READ_BIT, 0, index->params.size,
                            BUF_READ);
            }

            vk->CmdBindIndexBuffer(cmd->buf, index_vk->mem.buf,
                                   index_vk->mem.offset + params->index_offset,
                                   VK_INDEX_TYPE_UINT16);
        }

        tex_barrier(gpu, cmd, tex, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    pass_vk->initialLayout, false);

        VkViewport viewport = {
            .x = params->viewport.x0,
            .y = params->viewport.y0,
            .width  = pl_rect_w(params->viewport),
            .height = pl_rect_h(params->viewport),
        };

        VkRect2D scissor = {
            .offset = {params->scissors.x0, params->scissors.y0},
            .extent = {pl_rect_w(params->scissors), pl_rect_h(params->scissors)},
        };

        vk->CmdSetViewport(cmd->buf, 0, 1, &viewport);
        vk->CmdSetScissor(cmd->buf, 0, 1, &scissor);

        VkRenderPassBeginInfo binfo = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = pass_vk->renderPass,
            .framebuffer = tex_vk->framebuffer,
            .renderArea = (VkRect2D){{0, 0}, {tex->params.w, tex->params.h}},
        };

        vk->CmdBeginRenderPass(cmd->buf, &binfo, VK_SUBPASS_CONTENTS_INLINE);

        if (index) {
            vk->CmdDrawIndexed(cmd->buf, params->vertex_count, 1, 0, 0, 0);
        } else {
            vk->CmdDraw(cmd->buf, params->vertex_count, 1, 0, 0);
        }

        vk->CmdEndRenderPass(cmd->buf);

        buf_signal(gpu, cmd, vert, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
        if (index && index != vert)
            buf_signal(gpu, cmd, index, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

        // The renderPass implicitly transitions the texture to this layout
        tex_vk->current_layout = pass_vk->finalLayout;
        tex_signal(gpu, cmd, tex, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        break;
    }
    case PL_PASS_COMPUTE:
        vk->CmdDispatch(cmd->buf, params->compute_groups[0],
                        params->compute_groups[1],
                        params->compute_groups[2]);
        break;
    default: abort();
    };

    for (int i = 0; i < pass->params.num_descriptors; i++)
        vk_release_descriptor(gpu, cmd, pass, params->desc_bindings[i], i);

    // submit this command buffer for better intra-frame granularity
    vk_cmd_timer_end(gpu, cmd, params->timer);
    submit_cmd(p, &cmd);

error:
    return;
}

struct pl_sync_vk {
    pl_rc_t rc;
    VkSemaphore wait;
    VkSemaphore signal;
};

static void vk_sync_destroy(const struct pl_gpu *gpu, struct pl_sync *sync)
{
    if (!sync)
        return;

    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_sync_vk *sync_vk = PL_PRIV(sync);

#ifdef PL_HAVE_UNIX
    if (sync->handle_type == PL_HANDLE_FD) {
        if (sync->wait_handle.fd > -1)
            close(sync->wait_handle.fd);
        if (sync->signal_handle.fd > -1)
            close(sync->signal_handle.fd);
    }
#endif
#ifdef PL_HAVE_WIN32
    if (sync->handle_type == PL_HANDLE_WIN32) {
        if (sync->wait_handle.handle != NULL)
            CloseHandle(sync->wait_handle.handle);
        if (sync->signal_handle.handle != NULL)
            CloseHandle(sync->signal_handle.handle);
    }
    // PL_HANDLE_WIN32_KMT is just an identifier. It doesn't get closed.
#endif

    vk->DestroySemaphore(vk->dev, sync_vk->wait, PL_VK_ALLOC);
    vk->DestroySemaphore(vk->dev, sync_vk->signal, PL_VK_ALLOC);

    pl_free(sync);
}

static void vk_sync_deref(const struct pl_gpu *gpu, const struct pl_sync *sync)
{
    if (!sync)
        return;

    struct pl_sync_vk *sync_vk = PL_PRIV(sync);
    if (pl_rc_deref(&sync_vk->rc))
        vk_sync_destroy(gpu, (struct pl_sync *) sync);
}

static const struct pl_sync *vk_sync_create(const struct pl_gpu *gpu,
                                            enum pl_handle_type handle_type)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    struct pl_sync *sync = pl_zalloc_priv(NULL, struct pl_sync, struct pl_sync_vk);
    sync->handle_type = handle_type;

    struct pl_sync_vk *sync_vk = PL_PRIV(sync);
    pl_rc_init(&sync_vk->rc);

    VkExportSemaphoreCreateInfoKHR einfo = {
        .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR,
        .handleTypes = vk_sync_handle_type(handle_type),
    };

    switch (handle_type) {
    case PL_HANDLE_FD:
        sync->wait_handle.fd = -1;
        sync->signal_handle.fd = -1;
        break;
    case PL_HANDLE_WIN32:
    case PL_HANDLE_WIN32_KMT:
        sync->wait_handle.handle = NULL;
        sync->signal_handle.handle = NULL;
        break;
    case PL_HANDLE_DMA_BUF:
    case PL_HANDLE_HOST_PTR:
        abort();
    }

    const VkSemaphoreCreateInfo sinfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &einfo,
    };

    VK(vk->CreateSemaphore(vk->dev, &sinfo, PL_VK_ALLOC, &sync_vk->wait));
    VK(vk->CreateSemaphore(vk->dev, &sinfo, PL_VK_ALLOC, &sync_vk->signal));
    PL_VK_NAME(SEMAPHORE, sync_vk->wait, "sync wait");
    PL_VK_NAME(SEMAPHORE, sync_vk->signal, "sync signal");

#ifdef PL_HAVE_UNIX
    if (handle_type == PL_HANDLE_FD) {
        VkSemaphoreGetFdInfoKHR finfo = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
            .semaphore = sync_vk->wait,
            .handleType = einfo.handleTypes,
        };

        VK(vk->GetSemaphoreFdKHR(vk->dev, &finfo, &sync->wait_handle.fd));

        finfo.semaphore = sync_vk->signal;
        VK(vk->GetSemaphoreFdKHR(vk->dev, &finfo, &sync->signal_handle.fd));
    }
#endif

#ifdef PL_HAVE_WIN32
    if (handle_type == PL_HANDLE_WIN32 ||
        handle_type == PL_HANDLE_WIN32_KMT)
    {
        VkSemaphoreGetWin32HandleInfoKHR handle_info = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR,
            .semaphore = sync_vk->wait,
            .handleType = einfo.handleTypes,
        };

        VK(vk->GetSemaphoreWin32HandleKHR(vk->dev, &handle_info,
                                          &sync->wait_handle.handle));

        handle_info.semaphore = sync_vk->signal;
        VK(vk->GetSemaphoreWin32HandleKHR(vk->dev, &handle_info,
                                          &sync->signal_handle.handle));
    }
#endif

    return sync;

error:
    vk_sync_destroy(gpu, sync);
    return NULL;
}

void pl_vk_sync_unwrap(const struct pl_sync *sync, VkSemaphore *out_wait,
                       VkSemaphore *out_signal)
{
    struct pl_sync_vk *sync_vk = PL_PRIV(sync);
    if (out_wait)
        *out_wait = sync_vk->wait;
    if (out_signal)
        *out_signal = sync_vk->signal;
}

static bool vk_tex_export(const struct pl_gpu *gpu, const struct pl_tex *tex,
                          const struct pl_sync *sync)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);
    struct pl_sync_vk *sync_vk = PL_PRIV(sync);

    struct vk_cmd *cmd = begin_cmd(p, ANY);
    if (!cmd)
        goto error;

    tex_barrier(gpu, cmd, tex, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0, VK_IMAGE_LAYOUT_GENERAL, true);

    vk_cmd_sig(cmd, sync_vk->wait);
    submit_cmd(p, &cmd);

    if (!vk_flush_commands(vk))
        goto error;

    // Remember the other dependency and hold on to the sync object
    pl_tex_vk_external_dep(gpu, tex, sync_vk->signal);
    pl_rc_ref(&sync_vk->rc);
    tex_vk->ext_sync = sync;
    return true;

error:
    PL_ERR(gpu, "Failed exporting shared texture!");
    return false;
}

// Gives us enough queries for 8 results
#define QUERY_POOL_SIZE 16

struct pl_timer {
    bool recording; // true between vk_cmd_timer_begin() and vk_cmd_timer_end()
    VkQueryPool qpool; // even=start, odd=stop
    int index_write; // next index to write to
    int index_read; // next index to read from
    uint64_t pending; // bitmask of queries that are still running
};

static inline uint64_t timer_bit(int index)
{
    return 1llu << (index / 2);
}

static void vk_timer_destroy(const struct pl_gpu *gpu, struct pl_timer *timer)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    pl_assert(!timer->pending);
    vk->DestroyQueryPool(vk->dev, timer->qpool, PL_VK_ALLOC);
    pl_free(timer);
}

MAKE_LAZY_DESTRUCTOR(vk_timer_destroy, struct pl_timer)

static struct pl_timer *vk_timer_create(const struct pl_gpu *gpu)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    struct pl_timer *timer = pl_alloc_ptr(NULL, timer);
    *timer = (struct pl_timer) {0};

    struct VkQueryPoolCreateInfo qinfo = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = QUERY_POOL_SIZE,
    };

    VK(vk->CreateQueryPool(vk->dev, &qinfo, PL_VK_ALLOC, &timer->qpool));
    return timer;

error:
    vk_timer_destroy(gpu, timer);
    return NULL;
}

static uint64_t vk_timer_query(const struct pl_gpu *gpu, struct pl_timer *timer)
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
        return (ts[1] - ts[0]) * vk->limits.timestampPeriod;
    case VK_NOT_READY:
        return 0;
    default:
        PL_VK_ASSERT(res, "Retrieving query pool results");
    }

error:
    return 0;
}

static void vk_cmd_timer_begin(const struct pl_gpu *gpu, struct vk_cmd *cmd,
                               struct pl_timer *timer)
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
    } else if (p->host_query_reset) {
        // Use host query resets
        vk->ResetQueryPoolEXT(vk->dev, timer->qpool, timer->index_write, 2);
    } else {
        PL_TRACE(gpu, "QF %d supports no mechanism for resetting queries",
                 cmd->pool->qf);
        return;
    }

    vk->CmdWriteTimestamp(cmd->buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                          timer->qpool, timer->index_write);

    timer->recording = true;
}

static void vk_timer_cb(void *ptimer, void *pindex)
{
    struct pl_timer *timer = ptimer;
    int index = (uintptr_t) pindex;
    timer->pending &= ~timer_bit(index);
}

static void vk_cmd_timer_end(const struct pl_gpu *gpu, struct vk_cmd *cmd,
                             struct pl_timer *timer)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    if (!timer || !timer->recording)
        return;

    vk->CmdWriteTimestamp(cmd->buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                          timer->qpool, timer->index_write + 1);

    timer->recording = false;
    timer->pending |= timer_bit(timer->index_write);
    vk_cmd_callback(cmd, (vk_cb) vk_timer_cb, timer,
                    (void *) (uintptr_t) timer->index_write);

    timer->index_write = (timer->index_write + 2) % QUERY_POOL_SIZE;
    if (timer->index_write == timer->index_read) {
        // forcibly drop the least recent result to make space
        timer->index_read = (timer->index_read + 2) % QUERY_POOL_SIZE;
    }
}

static void vk_gpu_flush(const struct pl_gpu *gpu)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    flush(p);
    vk_flush_commands(vk);
    vk_rotate_queues(vk);
}

static void vk_gpu_finish(const struct pl_gpu *gpu)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    flush(p);
    vk_wait_idle(vk);
}

static bool vk_gpu_is_failed(const struct pl_gpu *gpu)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    return vk->failed;
}

struct vk_cmd *pl_vk_steal_cmd(const struct pl_gpu *gpu)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    pthread_mutex_lock(&p->recording);
    struct vk_cmd *cmd = p->cmd;
    p->cmd = NULL;
    pthread_mutex_unlock(&p->recording);

    struct vk_cmdpool *pool = vk->pool_graphics;
    if (!cmd || cmd->pool != pool) {
        vk_cmd_queue(vk, &cmd);
        cmd = vk_cmd_begin(vk, pool);
    }

    return cmd;
}

static const struct pl_gpu_fns pl_fns_vk = {
    .destroy                = vk_destroy_gpu,
    .tex_create             = vk_tex_create,
    .tex_destroy            = vk_tex_deref,
    .tex_invalidate         = vk_tex_invalidate,
    .tex_clear              = vk_tex_clear,
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
    .pass_destroy           = vk_pass_destroy_lazy,
    .pass_run               = vk_pass_run,
    .sync_create            = vk_sync_create,
    .sync_destroy           = vk_sync_deref,
    .tex_export             = vk_tex_export,
    .timer_create           = vk_timer_create,
    .timer_destroy          = vk_timer_destroy_lazy,
    .timer_query            = vk_timer_query,
    .gpu_flush              = vk_gpu_flush,
    .gpu_finish             = vk_gpu_finish,
    .gpu_is_failed          = vk_gpu_is_failed,
};
