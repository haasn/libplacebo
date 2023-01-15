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

void vk_tex_barrier(pl_gpu gpu, struct vk_cmd *cmd, pl_tex tex,
                    VkPipelineStageFlags stage, VkAccessFlags access,
                    VkImageLayout layout, uint32_t qf)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);
    pl_rc_ref(&tex_vk->rc);
    pl_assert(!tex_vk->held);

    // CONCURRENT images require transitioning to/from IGNORED, EXCLUSIVE
    // images require transitioning to/from the concrete QF index
    if (vk->pools.num == 1) {
        if (tex_vk->qf == VK_QUEUE_FAMILY_IGNORED)
            tex_vk->qf = cmd->pool->qf;
        if (qf == VK_QUEUE_FAMILY_IGNORED)
            qf = cmd->pool->qf;
    }

    struct vk_sync_scope last;
    bool is_trans = layout != tex_vk->layout, is_xfer = qf != tex_vk->qf;
    last = vk_sem_barrier(vk, cmd, &tex_vk->sem, stage, access, is_trans || is_xfer);

    VkImageMemoryBarrier barr = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout = tex_vk->layout,
        .newLayout = layout,
        .srcQueueFamilyIndex = tex_vk->qf,
        .dstQueueFamilyIndex = qf,
        .srcAccessMask = last.access,
        .dstAccessMask = access,
        .image = tex_vk->img,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
        },
    };

    if (tex_vk->may_invalidate) {
        tex_vk->may_invalidate = false;
        barr.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    if (last.access || is_trans || is_xfer) {
        vk->CmdPipelineBarrier(cmd->buf, last.stage, stage, 0, 0, NULL,
                               0, NULL, 1, &barr);
    }

    tex_vk->qf = qf;
    tex_vk->layout = layout;
    vk_cmd_callback(cmd, (vk_cb) vk_tex_deref, gpu, tex);

    for (int i = 0; i < tex_vk->ext_deps.num; i++)
        vk_cmd_dep(cmd, stage, tex_vk->ext_deps.elem[i]);
    tex_vk->ext_deps.num = 0;

    if (tex_vk->ext_sync) {
        vk_cmd_callback(cmd, (vk_cb) vk_sync_deref, gpu, tex_vk->ext_sync);
        tex_vk->ext_sync = NULL;
    }
}

static void vk_tex_destroy(pl_gpu gpu, struct pl_tex_t *tex)
{
    if (!tex)
        return;

    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);

    vk_sync_deref(gpu, tex_vk->ext_sync);
    vk_sem_uninit(vk, &tex_vk->sem);
    vk->DestroyFramebuffer(vk->dev, tex_vk->framebuffer, PL_VK_ALLOC);
    vk->DestroyImageView(vk->dev, tex_vk->view, PL_VK_ALLOC);
    if (!tex_vk->external_img) {
        vk->DestroyImage(vk->dev, tex_vk->img, PL_VK_ALLOC);
        vk_malloc_free(vk->ma, &tex_vk->mem);
    }

    pl_free(tex);
}

void vk_tex_deref(pl_gpu gpu, pl_tex tex)
{
    if (!tex)
        return;

    struct pl_tex_vk *tex_vk = PL_PRIV(tex);
    if (pl_rc_deref(&tex_vk->rc))
        vk_tex_destroy(gpu, (struct pl_tex_t *) tex);
}


// Initializes non-VkImage values like the image view, framebuffers, etc.
static bool vk_init_image(pl_gpu gpu, pl_tex tex, pl_debug_tag debug_tag)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    const struct pl_tex_params *params = &tex->params;
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);
    pl_assert(tex_vk->img);
    PL_VK_NAME(IMAGE, tex_vk->img, debug_tag);

    pl_rc_init(&tex_vk->rc);
    if (!vk_sem_init(vk, &tex_vk->sem, debug_tag))
        return false;
    tex_vk->layout = VK_IMAGE_LAYOUT_UNDEFINED;
    tex_vk->transfer_queue = GRAPHICS;
    tex_vk->qf = VK_QUEUE_FAMILY_IGNORED; // will be set on first use, if needed

    // Always use the transfer pool if available, for efficiency
    if ((params->host_writable || params->host_readable) && vk->pool_transfer)
        tex_vk->transfer_queue = TRANSFER;

    // For emulated formats: force usage of the compute queue, because we
    // can't properly track cross-queue dependencies for buffers (yet?)
    if (params->format->emulated)
        tex_vk->transfer_queue = COMPUTE;

    bool ret = false;
    VkRenderPass dummyPass = VK_NULL_HANDLE;

    if (params->sampleable || params->renderable || params->storable) {
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
        PL_VK_NAME(IMAGE_VIEW, tex_vk->view, debug_tag);
    }

    if (params->renderable) {
        // Framebuffers need to be created against a specific render pass
        // layout, so we need to temporarily create a skeleton/dummy render
        // pass for vulkan to figure out the compatibility
        VkRenderPassCreateInfo rinfo = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &(VkAttachmentDescription) {
                .format = tex_vk->img_fmt,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
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

        VK(vk->CreateRenderPass(vk->dev, &rinfo, PL_VK_ALLOC, &dummyPass));

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
        PL_VK_NAME(FRAMEBUFFER, tex_vk->framebuffer, debug_tag);
    }

    ret = true;

error:
    vk->DestroyRenderPass(vk->dev, dummyPass, PL_VK_ALLOC);
    return ret;
}

pl_tex vk_tex_create(pl_gpu gpu, const struct pl_tex_params *params)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    enum pl_handle_type handle_type = params->export_handle |
                                      params->import_handle;
    VkExternalMemoryHandleTypeFlagBitsKHR vk_handle_type = vk_mem_handle_type(handle_type);

    struct pl_tex_t *tex = pl_zalloc_obj(NULL, tex, struct pl_tex_vk);
    tex->params = *params;
    tex->params.initial_data = NULL;
    tex->sampler_type = PL_SAMPLER_NORMAL;

    struct pl_tex_vk *tex_vk = PL_PRIV(tex);
    struct pl_fmt_vk *fmtp = PL_PRIV(params->format);
    tex_vk->img_fmt = fmtp->vk_fmt->tfmt;

    switch (pl_tex_params_dimension(*params)) {
    case 1: tex_vk->type = VK_IMAGE_TYPE_1D; break;
    case 2: tex_vk->type = VK_IMAGE_TYPE_2D; break;
    case 3: tex_vk->type = VK_IMAGE_TYPE_3D; break;
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
        // based on the size of pl_tex_transfer_params.row_pitch, but for now
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

    if (fmtp->blit_emulated) {
        // Enable what's required for sampling
        tex->params.sampleable = params->format->caps & PL_FMT_CAP_SAMPLEABLE;
        tex->params.storable = true;
    }

    VkImageUsageFlags usage = 0;
    if (tex->params.sampleable)
        usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (tex->params.renderable)
        usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (tex->params.storable)
        usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    if (tex->params.host_readable || tex->params.blit_src)
        usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (tex->params.host_writable || tex->params.blit_dst || params->initial_data)
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
            .offset = params->shared_mem.offset,
        },
    };

#ifdef VK_EXT_metal_objects
    VkImportMetalTextureInfoEXT import_metal_tex = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_METAL_TEXTURE_INFO_EXT,
        .plane = VK_IMAGE_ASPECT_PLANE_0_BIT << params->shared_mem.plane,
    };

    VkImportMetalIOSurfaceInfoEXT import_iosurface = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_METAL_IO_SURFACE_INFO_EXT,
    };
#endif

    VkImageDrmFormatModifierListCreateInfoEXT drm_list = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
        .drmFormatModifierCount = params->format->num_modifiers,
        .pDrmFormatModifiers = params->format->modifiers,
    };

    VkExternalMemoryImageCreateInfoKHR ext_info = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO_KHR,
        .handleTypes = vk_handle_type,
    };

    VkImageCreateInfo iinfo = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = vk_handle_type ? &ext_info : NULL,
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

    struct vk_malloc_params mparams = {
        .optimal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        .export_handle = params->export_handle,
        .import_handle = params->import_handle,
        .shared_mem = params->shared_mem,
    };

    if (params->import_handle == PL_HANDLE_DMA_BUF) {
        vk_link_struct(&iinfo, &drm_explicit);
        iinfo.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
        mparams.shared_mem.offset = 0x0; // handled via plane offsets
    }

#ifdef VK_EXT_metal_objects
    if (params->import_handle == PL_HANDLE_MTL_TEX) {
        vk_link_struct(&iinfo, &import_metal_tex);
        import_metal_tex.mtlTexture = params->shared_mem.handle.handle;
    }

    if (params->import_handle == PL_HANDLE_IOSURFACE) {
        vk_link_struct(&iinfo, &import_iosurface);
        import_iosurface.ioSurface = params->shared_mem.handle.handle;
    }
#endif

    if (params->export_handle == PL_HANDLE_DMA_BUF) {
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
        .handleType = ext_info.handleTypes,
    };

    if (handle_type == PL_HANDLE_DMA_BUF)
        vk_link_struct(&ext_pinfo, &drm_pinfo);

    VkPhysicalDeviceImageFormatInfo2KHR pinfo = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2_KHR,
        .pNext = vk_handle_type ? &ext_pinfo : NULL,
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
        .pNext = vk_handle_type ? &ext_props : NULL,
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
    if (vk_handle_type) {
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

    VkMemoryDedicatedRequirements ded_reqs = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR,
    };

    VkMemoryRequirements2 reqs = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR,
        .pNext = &ded_reqs,
    };

    VkImageMemoryRequirementsInfo2 req_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2_KHR,
        .image = tex_vk->img,
    };

    vk->GetImageMemoryRequirements2(vk->dev, &req_info, &reqs);
    mparams.reqs = reqs.memoryRequirements;
    if (ded_reqs.prefersDedicatedAllocation) {
        mparams.ded_image = tex_vk->img;
        if (vk_mem_handle_type(params->import_handle))
            mparams.shared_mem.size = reqs.memoryRequirements.size;
    }

    const char *debug_tag = params->debug_tag ? params->debug_tag :
                            params->import_handle ? "imported" : "created";

    if (!params->import_handle || vk_mem_handle_type(params->import_handle)) {
        struct vk_memslice *mem = &tex_vk->mem;
        if (!vk_malloc_slice(vk->ma, mem, &mparams))
            goto error;

        VK(vk->BindImageMemory(vk->dev, tex_vk->img, mem->vkmem, mem->offset));
    }

    if (!vk_init_image(gpu, tex, debug_tag))
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

            VkSubresourceLayout layout = {0};
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
    vk_tex_destroy(gpu, tex);
    return NULL;
}

void vk_tex_invalidate(pl_gpu gpu, pl_tex tex)
{
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);
    tex_vk->may_invalidate = true;
}

void vk_tex_clear_ex(pl_gpu gpu, pl_tex tex, const union pl_clear_color color)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);

    struct vk_cmd *cmd = CMD_BEGIN(GRAPHICS);
    if (!cmd)
        return;

    vk_tex_barrier(gpu, cmd, tex, VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   VK_QUEUE_FAMILY_IGNORED);

    pl_static_assert(sizeof(VkClearColorValue) == sizeof(union pl_clear_color));
    const VkClearColorValue *clearColor = (const VkClearColorValue *) &color;

    static const VkImageSubresourceRange range = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .levelCount = 1,
        .layerCount = 1,
    };

    vk->CmdClearColorImage(cmd->buf, tex_vk->img, tex_vk->layout,
                           clearColor, 1, &range);

    CMD_FINISH(&cmd);
}

void vk_tex_blit(pl_gpu gpu, const struct pl_tex_blit_params *params)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_tex_vk *src_vk = PL_PRIV(params->src);
    struct pl_tex_vk *dst_vk = PL_PRIV(params->dst);
    struct pl_fmt_vk *src_fmtp = PL_PRIV(params->src->params.format);
    struct pl_fmt_vk *dst_fmtp = PL_PRIV(params->dst->params.format);
    bool blit_emulated = src_fmtp->blit_emulated || dst_fmtp->blit_emulated;

    struct pl_rect3d src_rc = params->src_rc, dst_rc = params->dst_rc;
    bool requires_scaling = !pl_rect3d_eq(src_rc, dst_rc);
    if (requires_scaling && blit_emulated) {
        if (!pl_tex_blit_compute(gpu, p->dp, params))
            PL_ERR(gpu, "Failed emulating texture blit, incompatible textures?");
        return;
    }

    struct vk_cmd *cmd = CMD_BEGIN(GRAPHICS);
    if (!cmd)
        return;

    vk_tex_barrier(gpu, cmd, params->src, VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_ACCESS_TRANSFER_READ_BIT,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   VK_QUEUE_FAMILY_IGNORED);

    vk_tex_barrier(gpu, cmd, params->dst, VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   VK_QUEUE_FAMILY_IGNORED);

    static const VkImageSubresourceLayers layers = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .layerCount = 1,
    };

    // When the blit operation doesn't require scaling, we can use the more
    // efficient vkCmdCopyImage instead of vkCmdBlitImage
    if (!requires_scaling) {
        pl_rect3d_normalize(&src_rc);

        VkImageCopy region = {
            .srcSubresource = layers,
            .dstSubresource = layers,
            .srcOffset = {src_rc.x0, src_rc.y0, src_rc.z0},
            .dstOffset = {src_rc.x0, src_rc.y0, src_rc.z0},
            .extent = {
                pl_rect_w(src_rc),
                pl_rect_h(src_rc),
                pl_rect_d(src_rc),
            },
        };

        vk->CmdCopyImage(cmd->buf, src_vk->img, src_vk->layout,
                         dst_vk->img, dst_vk->layout, 1, &region);
    } else {
        VkImageBlit region = {
            .srcSubresource = layers,
            .dstSubresource = layers,
            .srcOffsets = {{src_rc.x0, src_rc.y0, src_rc.z0},
                           {src_rc.x1, src_rc.y1, src_rc.z1}},
            .dstOffsets = {{dst_rc.x0, dst_rc.y0, dst_rc.z0},
                           {dst_rc.x1, dst_rc.y1, dst_rc.z1}},
        };

        static const VkFilter filters[PL_TEX_SAMPLE_MODE_COUNT] = {
            [PL_TEX_SAMPLE_NEAREST] = VK_FILTER_NEAREST,
            [PL_TEX_SAMPLE_LINEAR]  = VK_FILTER_LINEAR,
        };

        vk->CmdBlitImage(cmd->buf, src_vk->img, src_vk->layout,
                         dst_vk->img, dst_vk->layout, 1, &region,
                         filters[params->sample_mode]);
    }

    CMD_FINISH(&cmd);
}

// Determine the best queue type to perform a buffer<->image copy on
static enum queue_type vk_img_copy_queue(pl_gpu gpu, pl_tex tex,
                                         const struct VkBufferImageCopy *region)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    const struct pl_tex_vk *tex_vk = PL_PRIV(tex);
    enum queue_type queue = tex_vk->transfer_queue;
    if (queue != TRANSFER)
        return queue;

    VkExtent3D alignment = vk->pool_transfer->props.minImageTransferGranularity;

    enum queue_type fallback = GRAPHICS;
    if (gpu->limits.compute_queues > gpu->limits.fragment_queues)
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

bool vk_tex_upload(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    pl_tex tex = params->tex;
    pl_fmt fmt = tex->params.format;
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);

    if (!params->buf)
        return pl_tex_upload_pbo(gpu, params);

    pl_buf buf = params->buf;
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);
    struct pl_rect3d rc = params->rc;
    size_t size = pl_tex_transfer_size(params);

    size_t buf_offset = buf_vk->mem.offset + params->buf_offset;
    bool unaligned = buf_offset % fmt->texel_size;
    if (unaligned)
        PL_TRACE(gpu, "vk_tex_upload: unaligned transfer (slow path)");

    if (fmt->emulated || unaligned) {

        // Copy the source data buffer into an intermediate buffer
        struct pl_buf_params tbuf_params = {
            .debug_tag = PL_DEBUG_TAG,
            .memory_type = PL_BUF_MEM_DEVICE,
            .format = tex_vk->texel_fmt,
            .size = size,
        };

        if (fmt->emulated) {
            if (size <= gpu->limits.max_ubo_size) {
                tbuf_params.uniform = true;
            } else if (size <= gpu->limits.max_ssbo_size) {
                tbuf_params.storable = true;
            } else {
                // TODO: Implement strided upload path if really necessary
                PL_ERR(gpu, "Texel buffer size requirements exceed GPU "
                       "capabilities, failed uploading!");
                goto error;
            }
        }

        pl_buf tbuf = pl_buf_create(gpu, &tbuf_params);
        if (!tbuf) {
            PL_ERR(gpu, "Failed creating buffer for tex upload fallback!");
            goto error;
        }

        struct vk_cmd *cmd = CMD_BEGIN_TIMED(tex_vk->transfer_queue, params->timer);
        if (!cmd)
            goto error;

        struct pl_buf_vk *tbuf_vk = PL_PRIV(tbuf);
        VkBufferCopy region = {
            .srcOffset = buf_offset,
            .dstOffset = tbuf_vk->mem.offset,
            .size = size,
        };

        vk_buf_barrier(gpu, cmd, buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_ACCESS_TRANSFER_READ_BIT, params->buf_offset, size,
                       false);
        vk_buf_barrier(gpu, cmd, tbuf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_ACCESS_TRANSFER_WRITE_BIT, 0, size, false);
        vk->CmdCopyBuffer(cmd->buf, buf_vk->mem.buf, tbuf_vk->mem.buf,
                          1, &region);

        if (params->callback)
            vk_cmd_callback(cmd, tex_xfer_cb, params->callback, params->priv);

        CMD_FINISH(&cmd);

        struct pl_tex_transfer_params fixed = *params;
        fixed.buf = tbuf;
        fixed.buf_offset = 0;

        bool ok = fmt->emulated ? pl_tex_upload_texel(gpu, p->dp, &fixed)
                                : pl_tex_upload(gpu, &fixed);

        pl_buf_destroy(gpu, &tbuf);
        return ok;

    } else {

        pl_assert(fmt->texel_align == fmt->texel_size);
        VkBufferImageCopy region = {
            .bufferOffset = buf_offset,
            .bufferRowLength = params->row_pitch / fmt->texel_size,
            .bufferImageHeight = params->depth_pitch / params->row_pitch,
            .imageOffset = { rc.x0, rc.y0, rc.z0 },
            .imageExtent = { rc.x1, rc.y1, rc.z1 },
            .imageSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .layerCount = 1,
            },
        };

        enum queue_type queue = vk_img_copy_queue(gpu, tex, &region);
        struct vk_cmd *cmd = CMD_BEGIN_TIMED(queue, params->timer);
        if (!cmd)
            goto error;

        vk_buf_barrier(gpu, cmd, buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_ACCESS_TRANSFER_READ_BIT, params->buf_offset, size,
                       false);
        vk_tex_barrier(gpu, cmd, tex, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       VK_QUEUE_FAMILY_IGNORED);
        vk->CmdCopyBufferToImage(cmd->buf, buf_vk->mem.buf, tex_vk->img,
                                 tex_vk->layout, 1, &region);

        if (params->callback)
            vk_cmd_callback(cmd, tex_xfer_cb, params->callback, params->priv);

        CMD_FINISH(&cmd);
    }

    return true;

error:
    return false;
}

bool vk_tex_download(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    pl_tex tex = params->tex;
    pl_fmt fmt = tex->params.format;
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);

    if (!params->buf)
        return pl_tex_download_pbo(gpu, params);

    pl_buf buf = params->buf;
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);
    struct pl_rect3d rc = params->rc;
    size_t size = pl_tex_transfer_size(params);

    size_t buf_offset = buf_vk->mem.offset + params->buf_offset;
    bool unaligned = buf_offset % fmt->texel_size;
    if (unaligned)
        PL_TRACE(gpu, "vk_tex_download: unaligned transfer (slow path)");

    if (fmt->emulated || unaligned) {

        // Download into an intermediate buffer first
        pl_buf tbuf = pl_buf_create(gpu, pl_buf_params(
            .storable = fmt->emulated,
            .size = size,
            .memory_type = PL_BUF_MEM_DEVICE,
            .format = tex_vk->texel_fmt,
        ));

        if (!tbuf) {
            PL_ERR(gpu, "Failed creating buffer for tex download fallback!");
            goto error;
        }

        struct pl_tex_transfer_params fixed = *params;
        fixed.buf = tbuf;
        fixed.buf_offset = 0;

        bool ok = fmt->emulated ? pl_tex_download_texel(gpu, p->dp, &fixed)
                                : pl_tex_download(gpu, &fixed);
        if (!ok) {
            pl_buf_destroy(gpu, &tbuf);
            goto error;
        }

        struct vk_cmd *cmd = CMD_BEGIN_TIMED(tex_vk->transfer_queue, params->timer);
        if (!cmd) {
            pl_buf_destroy(gpu, &tbuf);
            goto error;
        }

        struct pl_buf_vk *tbuf_vk = PL_PRIV(tbuf);
        VkBufferCopy region = {
            .srcOffset = tbuf_vk->mem.offset,
            .dstOffset = buf_offset,
            .size = size,
        };

        vk_buf_barrier(gpu, cmd, tbuf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_ACCESS_TRANSFER_READ_BIT, 0, size, false);
        vk_buf_barrier(gpu, cmd, buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_ACCESS_TRANSFER_WRITE_BIT, params->buf_offset, size,
                       false);
        vk->CmdCopyBuffer(cmd->buf, tbuf_vk->mem.buf, buf_vk->mem.buf,
                          1, &region);
        vk_buf_flush(gpu, cmd, buf, params->buf_offset, size);

        if (params->callback)
            vk_cmd_callback(cmd, tex_xfer_cb, params->callback, params->priv);


        CMD_FINISH(&cmd);
        pl_buf_destroy(gpu, &tbuf);

    } else {

        pl_assert(fmt->texel_align == fmt->texel_size);
        VkBufferImageCopy region = {
            .bufferOffset = buf_offset,
            .bufferRowLength = params->row_pitch / fmt->texel_size,
            .bufferImageHeight = params->depth_pitch / params->row_pitch,
            .imageOffset = { rc.x0, rc.y0, rc.z0 },
            .imageExtent = { rc.x1, rc.y1, rc.z1 },
            .imageSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .layerCount = 1,
            },
        };

        enum queue_type queue = vk_img_copy_queue(gpu, tex, &region);

        struct vk_cmd *cmd = CMD_BEGIN_TIMED(queue, params->timer);
        if (!cmd)
            goto error;

        vk_buf_barrier(gpu, cmd, buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_ACCESS_TRANSFER_WRITE_BIT, params->buf_offset, size,
                       false);
        vk_tex_barrier(gpu, cmd, tex, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_ACCESS_TRANSFER_READ_BIT,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       VK_QUEUE_FAMILY_IGNORED);
        vk->CmdCopyImageToBuffer(cmd->buf, tex_vk->img, tex_vk->layout,
                                 buf_vk->mem.buf, 1, &region);
        vk_buf_flush(gpu, cmd, buf, params->buf_offset, size);

        if (params->callback)
            vk_cmd_callback(cmd, tex_xfer_cb, params->callback, params->priv);


        CMD_FINISH(&cmd);
    }

    return true;

error:
    return false;
}

bool vk_tex_poll(pl_gpu gpu, pl_tex tex, uint64_t timeout)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);

    // Opportunistically check if we can re-use this texture without flush
    vk_poll_commands(vk, 0);
    if (pl_rc_count(&tex_vk->rc) == 1)
        return false;

    // Otherwise, we're force to submit any queued command so that the user is
    // guaranteed to see progress eventually, even if they call this in a loop
    CMD_SUBMIT(NULL);
    vk_poll_commands(vk, timeout);

    return pl_rc_count(&tex_vk->rc) > 1;
}

bool vk_tex_export(pl_gpu gpu, pl_tex tex, pl_sync sync)
{
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);
    struct pl_sync_vk *sync_vk = PL_PRIV(sync);

    struct vk_cmd *cmd = CMD_BEGIN(ANY);
    if (!cmd)
        goto error;

    vk_tex_barrier(gpu, cmd, tex, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                   0, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_EXTERNAL);

    // Make the next barrier appear as though coming from a different queue
    tex_vk->sem.write.queue = tex_vk->sem.read.queue = NULL;

    vk_cmd_sig(cmd, (pl_vulkan_sem){ sync_vk->wait });
    if (!CMD_SUBMIT(&cmd))
        goto error;

    // Remember the other dependency and hold on to the sync object
    PL_ARRAY_APPEND(tex, tex_vk->ext_deps, (pl_vulkan_sem){ sync_vk->signal });
    pl_rc_ref(&sync_vk->rc);
    tex_vk->ext_sync = sync;
    tex_vk->qf = VK_QUEUE_FAMILY_EXTERNAL;
    return true;

error:
    PL_ERR(gpu, "Failed exporting shared texture!");
    return false;
}

pl_tex pl_vulkan_wrap(pl_gpu gpu, const struct pl_vulkan_wrap_params *params)
{
    pl_fmt format = NULL;
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
        return NULL;
    }

    struct pl_tex_t *tex = pl_zalloc_obj(NULL, tex, struct pl_tex_vk);
    tex->params = (struct pl_tex_params) {
        .format = format,
        .w = params->width,
        .h = params->height,
        .d = params->depth,
        .sampleable = !!(params->usage & VK_IMAGE_USAGE_SAMPLED_BIT),
        .renderable = !!(params->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT),
        .storable   = !!(params->usage & VK_IMAGE_USAGE_STORAGE_BIT),
        .blit_src   = !!(params->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT),
        .blit_dst   = !!(params->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT),
        .host_writable = !!(params->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT),
        .host_readable = !!(params->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT),
        .user_data  = params->user_data,
        .debug_tag  = params->debug_tag,
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

    // For simplicity, explicitly mask out blit emulation for wrapped textures
    struct pl_fmt_vk *fmtp = PL_PRIV(format);
    if (fmtp->blit_emulated) {
        tex->params.blit_src = false;
        tex->params.blit_dst = false;
    }

    struct pl_tex_vk *tex_vk = PL_PRIV(tex);
    tex_vk->type = VK_IMAGE_TYPE_2D;
    tex_vk->external_img = true;
    tex_vk->held = true;
    tex_vk->img = params->image;
    tex_vk->img_fmt = params->format;
    tex_vk->usage_flags = params->usage;

    if (!vk_init_image(gpu, tex, PL_DEF(params->debug_tag, "wrapped")))
        goto error;

    return tex;

error:
    vk_tex_destroy(gpu, tex);
    return NULL;
}

VkImage pl_vulkan_unwrap(pl_gpu gpu, pl_tex tex, VkFormat *out_format,
                         VkImageUsageFlags *out_flags)
{
    struct pl_tex_vk *tex_vk = PL_PRIV(tex);

    if (out_format)
        *out_format = tex_vk->img_fmt;
    if (out_flags)
        *out_flags = tex_vk->usage_flags;

    return tex_vk->img;
}

bool pl_vulkan_hold_ex(pl_gpu gpu, const struct pl_vulkan_hold_params *params)
{
    struct pl_tex_vk *tex_vk = PL_PRIV(params->tex);
    pl_assert(params->semaphore.sem);

    if (tex_vk->held) {
        PL_ERR(gpu, "Attempting to hold an already held image!");
        return false;
    }

    struct vk_cmd *cmd = CMD_BEGIN(GRAPHICS);
    if (!cmd) {
        PL_ERR(gpu, "Failed holding external image!");
        return false;
    }

    VkImageLayout layout = params->out_layout ? tex_vk->layout : params->layout;
    bool may_invalidate = tex_vk->may_invalidate;

    vk_tex_barrier(gpu, cmd, params->tex, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                   0, layout, params->qf);

    vk_cmd_sig(cmd, params->semaphore);

    tex_vk->sem.write.queue = tex_vk->sem.read.queue = NULL;
    tex_vk->held = CMD_SUBMIT(&cmd);
    if (!tex_vk->held)
        return false;
    if (params->out_layout)
        *params->out_layout = may_invalidate ? VK_IMAGE_LAYOUT_UNDEFINED : layout;
    return true;
}

void pl_vulkan_release_ex(pl_gpu gpu, const struct pl_vulkan_release_params *params)
{
    struct pl_tex_vk *tex_vk = PL_PRIV(params->tex);
    if (!tex_vk->held) {
        PL_ERR(gpu, "Attempting to release an unheld image?");
        return;
    }

    if (params->semaphore.sem)
        PL_ARRAY_APPEND(params->tex, tex_vk->ext_deps, params->semaphore);

    tex_vk->qf = params->qf;
    tex_vk->layout = params->layout;
    tex_vk->held = false;
}
