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

#include "ra_vk.h"
#include "command.h"
#include "formats.h"
#include "malloc.h"
#include "spirv.h"

static struct ra_fns ra_fns_vk;

enum queue_type {
    GRAPHICS,
    COMPUTE,
    TRANSFER,
};

// For ra.priv
struct ra_vk {
    struct vk_ctx *vk;
    struct vk_malloc *alloc;
    struct spirv_compiler *spirv;

    // The "currently recording" command. This will be queued and replaced by
    // a new command every time we need to "switch" between queue families.
    struct vk_cmd *cmd;
};

struct vk_ctx *ra_vk_get(const struct ra *ra)
{
    if (ra->impl != &ra_fns_vk)
        return NULL;

    struct ra_vk *p = ra->priv;
    return p->vk;
}

static void vk_submit(const struct ra *ra)
{
    struct ra_vk *p = ra->priv;
    struct vk_ctx *vk = ra_vk_get(ra);

    if (p->cmd) {
        vk_cmd_queue(vk, p->cmd);
        p->cmd = NULL;
    }
}

// Returns a command buffer, or NULL on error
static struct vk_cmd *vk_require_cmd(const struct ra *ra, enum queue_type type)
{
    struct ra_vk *p = ra->priv;
    struct vk_ctx *vk = ra_vk_get(ra);

    struct vk_cmdpool *pool;
    switch (type) {
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

    pl_assert(pool);
    if (p->cmd && p->cmd->pool == pool)
        return p->cmd;

    vk_submit(ra);
    p->cmd = vk_cmd_begin(vk, pool);
    return p->cmd;
}

#define MAKE_LAZY_DESTRUCTOR(fun, argtype)                              \
    static void fun##_lazy(const struct ra *ra, const argtype *arg) {   \
        struct ra_vk *p = ra->priv;                                     \
        struct vk_ctx *vk = ra_vk_get(ra);                              \
        if (p->cmd) {                                                   \
            vk_cmd_callback(p->cmd, (vk_cb) fun, ra, (void *) arg);     \
        } else {                                                        \
            vk_dev_callback(vk, (vk_cb) fun, ra, (void *) arg);         \
        }                                                               \
    }

static void vk_destroy_ra(const struct ra *ra)
{
    struct ra_vk *p = ra->priv;
    struct vk_ctx *vk = ra_vk_get(ra);

    vk_submit(ra);
    vk_wait_idle(vk);

    vk_malloc_destroy(&p->alloc);
    spirv_compiler_destroy(&p->spirv);

    talloc_free((void *) ra);
}

static void vk_setup_formats(struct ra *ra)
{
    struct vk_ctx *vk = ra_vk_get(ra);

    for (const struct vk_format *vk_fmt = vk_formats; vk_fmt->ifmt; vk_fmt++) {
        VkFormatProperties prop;
        vkGetPhysicalDeviceFormatProperties(vk->physd, vk_fmt->ifmt, &prop);

        struct ra_fmt *fmt = talloc_ptrtype(ra, fmt);
        *fmt = vk_fmt->fmt;
        fmt->priv = vk_fmt;

        // For sanity, clear the superfluous fields
        for (int i = fmt->num_components; i < 4; i++) {
            fmt->component_depth[i] = 0;
            fmt->sample_order[i] = 0;
            fmt->host_bits[i] = 0;
        }

        struct { VkFormatFeatureFlags flags; enum ra_fmt_caps caps; } bits[] = {
            {VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT,      RA_FMT_CAP_BLENDABLE},
            {VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT, RA_FMT_CAP_LINEAR},
            {VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT,               RA_FMT_CAP_SAMPLEABLE},
            {VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT,               RA_FMT_CAP_STORABLE},
            {VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT,            RA_FMT_CAP_RENDERABLE},

            // We don't distinguish between the two blit modes for ra_fmt_caps
            {VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT,
                RA_FMT_CAP_BLITTABLE},
        };

        // Disable implied capabilities where the dependencies are unavailable
        if (!(fmt->caps & RA_FMT_CAP_SAMPLEABLE))
            fmt->caps &= ~RA_FMT_CAP_LINEAR;
        if (!(ra->caps & RA_CAP_COMPUTE))
            fmt->caps &= ~RA_FMT_CAP_STORABLE;

        for (int i = 0; i < PL_ARRAY_SIZE(bits); i++) {
            if ((prop.optimalTilingFeatures & bits[i].flags) == bits[i].flags)
                fmt->caps |= bits[i].caps;
        }

        if (prop.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT) {
            fmt->caps |= RA_FMT_CAP_VERTEX;
            fmt->glsl_type = ra_var_glsl_type_name(ra_var_from_fmt(fmt, ""));
            pl_assert(fmt->glsl_type);
        }

        if (fmt->caps & RA_FMT_CAP_STORABLE) {
            fmt->glsl_format = ra_fmt_glsl_format(fmt);
            if (!fmt->glsl_format) {
                PL_WARN(ra, "Storable format '%s' has no matching GLSL format "
                        "qualifier?", fmt->name);
                fmt->caps &= ~RA_FMT_CAP_STORABLE;
            }
        }

        TARRAY_APPEND(ra, ra->formats, ra->num_formats, fmt);
    }

    ra_sort_formats(ra);
}

const struct ra *ra_create_vk(struct vk_ctx *vk)
{
    pl_assert(vk->dev);

    struct ra *ra = talloc_zero(NULL, struct ra);
    ra->ctx = vk->ctx;
    ra->impl = &ra_fns_vk;

    struct ra_vk *p = ra->priv = talloc_zero(ra, struct ra_vk);
    p->vk = vk;

    p->spirv = spirv_compiler_create(vk->ctx);
    p->alloc = vk_malloc_create(vk);
    if (!p->alloc || !p->spirv)
        goto error;

    ra->glsl = p->spirv->glsl;
    ra->limits = (struct ra_limits) {
        .max_tex_1d_dim    = vk->limits.maxImageDimension1D,
        .max_tex_2d_dim    = vk->limits.maxImageDimension2D,
        .max_tex_3d_dim    = vk->limits.maxImageDimension3D,
        .max_pushc_size    = vk->limits.maxPushConstantsSize,
        .max_xfer_size     = SIZE_MAX, // no limit imposed by vulkan
        .max_ubo_size      = vk->limits.maxUniformBufferRange,
        .max_ssbo_size     = vk->limits.maxStorageBufferRange,
        .min_gather_offset = vk->limits.minTexelGatherOffset,
        .max_gather_offset = vk->limits.maxTexelGatherOffset,
        .align_tex_xfer_stride = vk->limits.optimalBufferCopyRowPitchAlignment,
        .align_tex_xfer_offset = vk->limits.optimalBufferCopyOffsetAlignment,
    };

    if (vk->pool_compute) {
        ra->caps |= RA_CAP_COMPUTE;
        ra->limits.max_shmem_size = vk->limits.maxComputeSharedMemorySize;
        ra->limits.max_group_threads = vk->limits.maxComputeWorkGroupInvocations;
        for (int i = 0; i < 3; i++) {
            ra->limits.max_group_size[i] = vk->limits.maxComputeWorkGroupSize[i];
            ra->limits.max_dispatch[i] = vk->limits.maxComputeWorkGroupCount[i];
        }

        // If we have more compute queues than graphics queues, we probably
        // want to be using them. (This seems mostly relevant for AMD)
        if (vk->pool_compute->num_queues > vk->pool_graphics->num_queues)
            ra->caps |= RA_CAP_PARALLEL_COMPUTE;
    }

    vk_setup_formats(ra);

    ra_print_info(ra, PL_LOG_INFO);
    ra_print_formats(ra, PL_LOG_DEBUG);
    return ra;

error:
    vk_destroy_ra(ra);
    return NULL;
}

// Boilerplate wrapper around vkCreateRenderPass to ensure passes remain
// compatible. The renderpass will automatically transition the image out of
// initialLayout and into finalLayout.
static VkResult vk_create_render_pass(VkDevice dev, const struct ra_fmt *fmt,
                                      VkAttachmentLoadOp loadOp,
                                      VkImageLayout initialLayout,
                                      VkImageLayout finalLayout,
                                      VkRenderPass *out)
{
    const struct vk_format *vk_fmt = fmt->priv;

    VkRenderPassCreateInfo rinfo = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &(VkAttachmentDescription) {
            .format = vk_fmt->ifmt,
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

    return vkCreateRenderPass(dev, &rinfo, VK_ALLOC, out);
}

// For ra_tex.priv
struct ra_tex_vk {
    bool external_img;
    bool may_invalidate;
    enum queue_type transfer_queue;
    VkImageType type;
    VkImage img;
    struct vk_memslice mem;
    // for sampling
    VkImageView view;
    VkSampler sampler;
    // for rendering
    VkFramebuffer framebuffer;
    // for transfers
    struct ra_buf_pool pbo_write;
    struct ra_buf_pool pbo_read;
    // "current" metadata, can change during the course of execution
    VkImageLayout current_layout;
    VkAccessFlags current_access;
    // the signal guards reuse, and can be NULL
    struct vk_signal *sig;
    VkPipelineStageFlags sig_stage;
    VkSemaphore *ext_deps; // external semaphore, not owned by the ra_tex
    int num_ext_deps;
};

void ra_tex_vk_external_dep(const struct ra *ra, const struct ra_tex *tex,
                            VkSemaphore external_dep)
{
    struct ra_tex_vk *tex_vk = tex->priv;
    TARRAY_APPEND(tex_vk, tex_vk->ext_deps, tex_vk->num_ext_deps, external_dep);
}

// Small helper to ease image barrier creation. if `discard` is set, the contents
// of the image will be undefined after the barrier
static void tex_barrier(const struct ra *ra, struct vk_cmd *cmd,
                        const struct ra_tex *tex, VkPipelineStageFlags stage,
                        VkAccessFlags newAccess, VkImageLayout newLayout)
{
    struct vk_ctx *vk = ra_vk_get(ra);
    struct ra_tex_vk *tex_vk = tex->priv;

    for (int i = 0; i < tex_vk->num_ext_deps; i++)
        vk_cmd_dep(cmd, tex_vk->ext_deps[i], stage);
    tex_vk->num_ext_deps = 0;

    VkImageMemoryBarrier imgBarrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout = tex_vk->current_layout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .srcAccessMask = tex_vk->current_access,
        .dstAccessMask = newAccess,
        .image = tex_vk->img,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
        },
    };

    if (tex_vk->may_invalidate) {
        tex_vk->may_invalidate = false;
        imgBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imgBarrier.srcAccessMask = 0;
    }

    VkEvent event = NULL;
    enum vk_wait_type type = vk_cmd_wait(vk, cmd, &tex_vk->sig, stage, &event);

    bool need_trans = tex_vk->current_layout != newLayout ||
                      tex_vk->current_access != newAccess;

    // Transitioning to VK_IMAGE_LAYOUT_UNDEFINED is a pseudo-operation
    // that for us means we don't need to perform the actual transition
    if (need_trans && newLayout != VK_IMAGE_LAYOUT_UNDEFINED) {
        switch (type) {
        case VK_WAIT_NONE:
            // No synchronization required, so we can safely transition out of
            // VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            imgBarrier.srcAccessMask = 0;
            vkCmdPipelineBarrier(cmd->buf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 stage, 0, 0, NULL, 0, NULL, 1, &imgBarrier);
            break;
        case VK_WAIT_BARRIER:
            // Regular pipeline barrier is required
            vkCmdPipelineBarrier(cmd->buf, tex_vk->sig_stage, stage, 0, 0, NULL,
                                 0, NULL, 1, &imgBarrier);
            break;
        case VK_WAIT_EVENT:
            // We can/should use the VkEvent for synchronization
            vkCmdWaitEvents(cmd->buf, 1, &event, tex_vk->sig_stage,
                            stage, 0, NULL, 0, NULL, 1, &imgBarrier);
            break;
        }
    }

    tex_vk->current_layout = newLayout;
    tex_vk->current_access = newAccess;
}

static void tex_signal(const struct ra *ra, struct vk_cmd *cmd,
                       const struct ra_tex *tex, VkPipelineStageFlags stage)
{
    struct ra_tex_vk *tex_vk = tex->priv;
    struct vk_ctx *vk = ra_vk_get(ra);
    pl_assert(!tex_vk->sig);

    tex_vk->sig = vk_cmd_signal(vk, cmd, stage);
    tex_vk->sig_stage = stage;
}

static void vk_tex_destroy(const struct ra *ra, struct ra_tex *tex)
{
    if (!tex)
        return;

    struct vk_ctx *vk = ra_vk_get(ra);
    struct ra_tex_vk *tex_vk = tex->priv;
    struct ra_vk *p = ra->priv;

    ra_buf_pool_uninit(ra, &tex_vk->pbo_write);
    ra_buf_pool_uninit(ra, &tex_vk->pbo_read);
    vk_signal_destroy(vk, &tex_vk->sig);
    vkDestroyFramebuffer(vk->dev, tex_vk->framebuffer, VK_ALLOC);
    vkDestroySampler(vk->dev, tex_vk->sampler, VK_ALLOC);
    vkDestroyImageView(vk->dev, tex_vk->view, VK_ALLOC);
    if (!tex_vk->external_img) {
        vkDestroyImage(vk->dev, tex_vk->img, VK_ALLOC);
        vk_free_memslice(p->alloc, tex_vk->mem);
    }

    talloc_free(tex);
}

MAKE_LAZY_DESTRUCTOR(vk_tex_destroy, struct ra_tex);

static const VkFilter filters[] = {
    [RA_TEX_SAMPLE_NEAREST] = VK_FILTER_NEAREST,
    [RA_TEX_SAMPLE_LINEAR]  = VK_FILTER_LINEAR,
};

// Initializes non-VkImage values like the image view, samplers, etc.
static bool vk_init_image(const struct ra *ra, const struct ra_tex *tex)
{
    struct vk_ctx *vk = ra_vk_get(ra);

    const struct ra_tex_params *params = &tex->params;
    struct ra_tex_vk *tex_vk = tex->priv;
    pl_assert(tex_vk->img);

    tex_vk->current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    tex_vk->current_access = 0;
    tex_vk->transfer_queue = GRAPHICS;

    // Always use the transfer pool if available, for efficiency
    if ((params->host_writable || params->host_readable) && vk->pool_transfer)
        tex_vk->transfer_queue = TRANSFER;

    bool ret = false;
    VkRenderPass dummyPass = NULL;

    if (params->sampleable || params->renderable) {
        static const VkImageViewType viewType[] = {
            [VK_IMAGE_TYPE_1D] = VK_IMAGE_VIEW_TYPE_1D,
            [VK_IMAGE_TYPE_2D] = VK_IMAGE_VIEW_TYPE_2D,
            [VK_IMAGE_TYPE_3D] = VK_IMAGE_VIEW_TYPE_3D,
        };

        const struct vk_format *fmt = params->format->priv;
        VkImageViewCreateInfo vinfo = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = tex_vk->img,
            .viewType = viewType[tex_vk->type],
            .format = fmt->ifmt,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .levelCount = 1,
                .layerCount = 1,
            },
        };

        VK(vkCreateImageView(vk->dev, &vinfo, VK_ALLOC, &tex_vk->view));
    }

    if (params->sampleable) {
        static const VkSamplerAddressMode modes[] = {
            [RA_TEX_ADDRESS_CLAMP]  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            [RA_TEX_ADDRESS_REPEAT] = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            [RA_TEX_ADDRESS_MIRROR] = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
        };

        VkSamplerCreateInfo sinfo = {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = filters[params->sample_mode],
            .minFilter = filters[params->sample_mode],
            .addressModeU = modes[params->address_mode],
            .addressModeV = modes[params->address_mode],
            .addressModeW = modes[params->address_mode],
            .maxAnisotropy = 1.0,
        };

        VK(vkCreateSampler(vk->dev, &sinfo, VK_ALLOC, &tex_vk->sampler));
    }

    if (params->renderable) {
        // Framebuffers need to be created against a specific render pass
        // layout, so we need to temporarily create a skeleton/dummy render
        // pass for vulkan to figure out the compatibility
        VK(vk_create_render_pass(vk->dev, params->format,
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
            PL_ERR(ra, "Framebuffer of size %dx%d exceeds the maximum allowed "
                   "dimensions: %dx%d", finfo.width, finfo.height,
                   vk->limits.maxFramebufferWidth,
                   vk->limits.maxFramebufferHeight);
            goto error;
        }

        VK(vkCreateFramebuffer(vk->dev, &finfo, VK_ALLOC,
                               &tex_vk->framebuffer));
    }

    ret = true;

error:
    vkDestroyRenderPass(vk->dev, dummyPass, VK_ALLOC);
    return ret;
}

static const struct ra_tex *vk_tex_create(const struct ra *ra,
                                          const struct ra_tex_params *params)
{
    struct vk_ctx *vk = ra_vk_get(ra);
    struct ra_vk *p = ra->priv;

    struct ra_tex *tex = talloc_zero(NULL, struct ra_tex);
    tex->params = *params;
    tex->params.initial_data = NULL;

    struct ra_tex_vk *tex_vk = tex->priv = talloc_zero(tex, struct ra_tex_vk);

    const struct vk_format *fmt = params->format->priv;
    switch (ra_tex_params_dimension(*params)) {
    case 1: tex_vk->type = VK_IMAGE_TYPE_1D; break;
    case 2: tex_vk->type = VK_IMAGE_TYPE_2D; break;
    case 3: tex_vk->type = VK_IMAGE_TYPE_3D; break;
    default: abort();
    }

    VkImageUsageFlags usage = 0;
    if (params->sampleable)
        usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (params->renderable)
        usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (params->storable)
        usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    if (params->host_readable || params->blit_src)
        usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (params->host_writable || params->blit_dst || params->initial_data)
        usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    // Double-check physical image format limits and fail if invalid
    VkImageFormatProperties iprop;
    VkResult res = vkGetPhysicalDeviceImageFormatProperties(vk->physd, fmt->ifmt,
            tex_vk->type, VK_IMAGE_TILING_OPTIMAL, usage, 0, &iprop);

    if (res == VK_ERROR_FORMAT_NOT_SUPPORTED) {
        return NULL;
    } else {
        VK_ASSERT(res, "Querying image format properties");
    }

    VkExtent3D max = iprop.maxExtent;
    if (params->w > max.width || params->h > max.height || params->d > max.depth)
    {
        PL_ERR(ra, "Requested image size %dx%dx%d exceeds the maximum allowed "
               "dimensions %dx%dx%d for vulkan image format %x",
               params->w, params->h, params->d, max.width, max.height, max.depth,
               (unsigned) fmt->ifmt);
        return NULL;
    }

    // FIXME: Since we can't keep track of queue family ownership properly,
    // and we don't know in advance what types of queue families this image
    // will belong to, we're forced to share all of our images between all
    // command pools.
    uint32_t qfs[3] = {0};
    for (int i = 0; i < vk->num_pools; i++)
        qfs[i] = vk->pools[i]->qf;

    VkImageCreateInfo iinfo = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = tex_vk->type,
        .format = fmt->ifmt,
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
        .sharingMode = vk->num_pools > 1 ? VK_SHARING_MODE_CONCURRENT
                                         : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = vk->num_pools,
        .pQueueFamilyIndices = qfs,
    };

    VK(vkCreateImage(vk->dev, &iinfo, VK_ALLOC, &tex_vk->img));

    VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkMemoryRequirements reqs;
    vkGetImageMemoryRequirements(vk->dev, tex_vk->img, &reqs);

    struct vk_memslice *mem = &tex_vk->mem;
    if (!vk_malloc_generic(p->alloc, reqs, memFlags, mem))
        goto error;

    VK(vkBindImageMemory(vk->dev, tex_vk->img, mem->vkmem, mem->offset));

    if (!vk_init_image(ra, tex))
        goto error;

    if (params->initial_data) {
        struct ra_tex_transfer_params ul_params = {
            .tex = tex,
            .ptr = (void *) params->initial_data,
            .rc = { 0, 0, 0, params->w, params->h, params->d },
            .stride_w = params->w,
            .stride_h = params->h,
        };

        // Since we re-use RA helpers which require writable images, just fake it
        bool writable = tex->params.host_writable;
        tex->params.host_writable = true;
        if (!ra_tex_upload(ra, &ul_params))
            goto error;
        tex->params.host_writable = writable;
    }

    return tex;

error:
    vk_tex_destroy(ra, tex);
    return NULL;
}

static void vk_tex_invalidate(const struct ra *ra, const struct ra_tex *tex)
{
    struct ra_tex_vk *tex_vk = tex->priv;
    tex_vk->may_invalidate = true;
}

static void vk_tex_clear(const struct ra *ra, const struct ra_tex *tex,
                         const float color[4])
{
    struct ra_tex_vk *tex_vk = tex->priv;

    struct vk_cmd *cmd = vk_require_cmd(ra, GRAPHICS);
    if (!cmd)
        return;

    tex_barrier(ra, cmd, tex, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkClearColorValue clearColor = {0};
    for (int c = 0; c < 4; c++)
        clearColor.float32[c] = color[c];

    static const VkImageSubresourceRange range = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .levelCount = 1,
        .layerCount = 1,
    };

    vkCmdClearColorImage(cmd->buf, tex_vk->img, tex_vk->current_layout,
                         &clearColor, 1, &range);

    tex_signal(ra, cmd, tex, VK_PIPELINE_STAGE_TRANSFER_BIT);
}

static void vk_tex_blit(const struct ra *ra,
                        const struct ra_tex *dst, const struct ra_tex *src,
                        struct pl_rect3d dst_rc, struct pl_rect3d src_rc)
{
    struct ra_tex_vk *src_vk = src->priv;
    struct ra_tex_vk *dst_vk = dst->priv;

    struct vk_cmd *cmd = vk_require_cmd(ra, GRAPHICS);
    if (!cmd)
        return;

    tex_barrier(ra, cmd, src, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    tex_barrier(ra, cmd, dst, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    static const VkImageSubresourceLayers layers = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .layerCount = 1,
    };

    // When the blit operation doesn't require scaling, we can use the more
    // efficient vkCmdCopyImage instead of vkCmdBlitImage
    if (pl_rect3d_eq(src_rc, dst_rc)) {
        pl_rect3d_normalize(&src_rc);
        pl_rect3d_normalize(&src_rc);

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

        vkCmdCopyImage(cmd->buf, src_vk->img, src_vk->current_layout,
                       dst_vk->img, dst_vk->current_layout, 1, &region);
    } else {
        VkImageBlit region = {
            .srcSubresource = layers,
            .dstSubresource = layers,
            .srcOffsets = {{src_rc.x0, src_rc.y0, src_rc.z0},
                           {src_rc.x1, src_rc.y1, src_rc.z1}},
            .dstOffsets = {{dst_rc.x0, dst_rc.y0, src_rc.z0},
                           {dst_rc.x1, dst_rc.y1, src_rc.z1}},
        };

        vkCmdBlitImage(cmd->buf, src_vk->img, src_vk->current_layout,
                       dst_vk->img, dst_vk->current_layout, 1, &region,
                       filters[src->params.sample_mode]);
    }

    tex_signal(ra, cmd, src, VK_PIPELINE_STAGE_TRANSFER_BIT);
    tex_signal(ra, cmd, dst, VK_PIPELINE_STAGE_TRANSFER_BIT);
}

const struct ra_tex *ra_vk_wrap_swapchain_img(const struct ra *ra, VkImage vkimg,
                                              VkSwapchainCreateInfoKHR info)
{
    struct ra_tex *tex = NULL;

    const struct ra_fmt *format = NULL;
    for (int i = 0; i < ra->num_formats; i++) {
        const struct vk_format *fmt = ra->formats[i]->priv;
        if (fmt->ifmt == info.imageFormat) {
            format = ra->formats[i];
            break;
        }
    }

    if (!format) {
        PL_ERR(ra, "Could not find ra_fmt suitable for wrapped swchain image "
               "with surface format 0x%x\n", (unsigned) info.imageFormat);
        goto error;
    }

    tex = talloc_zero(NULL, struct ra_tex);
    tex->params = (struct ra_tex_params) {
        .format = format,
        .w = info.imageExtent.width,
        .h = info.imageExtent.height,
        .sampleable  = !!(info.imageUsage & VK_IMAGE_USAGE_SAMPLED_BIT),
        .renderable  = !!(info.imageUsage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT),
        .storable    = !!(info.imageUsage & VK_IMAGE_USAGE_STORAGE_BIT),
        .blit_src    = !!(info.imageUsage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT),
        .blit_dst    = !!(info.imageUsage & VK_IMAGE_USAGE_TRANSFER_DST_BIT),
        .host_writable = !!(info.imageUsage & VK_IMAGE_USAGE_TRANSFER_DST_BIT),
        .host_readable = !!(info.imageUsage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT),
    };

    struct ra_tex_vk *tex_vk = tex->priv = talloc_zero(tex, struct ra_tex_vk);
    tex_vk->type = VK_IMAGE_TYPE_2D;
    tex_vk->external_img = true;
    tex_vk->img = vkimg;

    if (!vk_init_image(ra, tex))
        goto error;

    return tex;

error:
    vk_tex_destroy(ra, tex);
    return NULL;
}

// For ra_buf.priv
struct ra_buf_vk {
    struct vk_bufslice slice;
    int refcount; // 1 = object allocated but not in use, > 1 = in use
    bool needs_flush;
    enum queue_type update_queue;
    // "current" metadata, can change during course of execution
    VkPipelineStageFlags current_stage;
    VkAccessFlags current_access;
};

#define RA_VK_BUF_VERTEX RA_BUF_PRIVATE

static void vk_buf_deref(const struct ra *ra, struct ra_buf *buf)
{
    if (!buf)
        return;

    struct ra_buf_vk *buf_vk = buf->priv;
    struct ra_vk *p = ra->priv;

    if (--buf_vk->refcount == 0) {
        vk_free_memslice(p->alloc, buf_vk->slice.mem);
        talloc_free(buf);
    }
}

// visible_writes: whether or not this buffer access consistites a write to the
// buffer that modifies the contents in a way that the host should be able to
// see. This includes any synchronization necessary to ensure the writes are
// made visible to the host.
static void buf_barrier(const struct ra *ra, struct vk_cmd *cmd,
                        const struct ra_buf *buf, VkPipelineStageFlags newStage,
                        VkAccessFlags newAccess, int offset, size_t size,
                        bool visible_writes)
{
    struct ra_buf_vk *buf_vk = buf->priv;

    VkBufferMemoryBarrier buffBarrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = buf_vk->current_access,
        .dstAccessMask = newAccess,
        .buffer = buf_vk->slice.buf,
        .offset = offset,
        .size = size,
    };

    if (buf_vk->needs_flush || buf->params.host_mapped) {
        buffBarrier.srcAccessMask |= VK_ACCESS_HOST_WRITE_BIT;
        buf_vk->current_stage |= VK_PIPELINE_STAGE_HOST_BIT;
        buf_vk->needs_flush = false;
    }

    if (visible_writes && (buf->params.host_readable || buf->params.host_mapped)) {
        buffBarrier.dstAccessMask |= VK_ACCESS_HOST_READ_BIT;
        newStage |= VK_PIPELINE_STAGE_HOST_BIT;
    }

    if (buffBarrier.srcAccessMask != buffBarrier.dstAccessMask) {
        vkCmdPipelineBarrier(cmd->buf, buf_vk->current_stage, newStage, 0,
                             0, NULL, 1, &buffBarrier, 0, NULL);
    }

    buf_vk->current_stage = newStage;
    buf_vk->current_access = newAccess;
    buf_vk->refcount++;
    vk_cmd_callback(cmd, (vk_cb) vk_buf_deref, ra, buf);
}

#define vk_buf_destroy vk_buf_deref
MAKE_LAZY_DESTRUCTOR(vk_buf_destroy, struct ra_buf);

static void vk_buf_write(const struct ra *ra, const struct ra_buf *buf,
                         size_t offset, const void *data, size_t size)
{
    struct ra_buf_vk *buf_vk = buf->priv;

    // For host-mapped buffers, we can just directly memcpy the buffer contents.
    // Otherwise, we can update the buffer from the GPU using a command buffer.
    if (buf_vk->slice.data) {
        uintptr_t addr = (uintptr_t) buf_vk->slice.data + (ptrdiff_t) offset;
        memcpy((void *) addr, data, size);
        buf_vk->needs_flush = true;
    } else {
        struct vk_cmd *cmd = vk_require_cmd(ra, buf_vk->update_queue);
        if (!cmd) {
            PL_ERR(ra, "Failed updating buffer!");
            return;
        }

        buf_barrier(ra, cmd, buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT, offset, size, true);

        VkDeviceSize bufOffset = buf_vk->slice.mem.offset + offset;
        vkCmdUpdateBuffer(cmd->buf, buf_vk->slice.buf, bufOffset, size, data);
    }
}

static bool vk_buf_read(const struct ra *ra, const struct ra_buf *buf,
                        size_t offset, void *dest, size_t size)
{
    struct ra_buf_vk *buf_vk = buf->priv;
    pl_assert(buf_vk->slice.data);

    uintptr_t addr = (uintptr_t) buf_vk->slice.data + (ptrdiff_t) offset;
    memcpy(dest, (void *) addr, size);
    return true;
}

static const struct ra_buf *vk_buf_create(const struct ra *ra,
                                          const struct ra_buf_params *params)
{
    struct vk_ctx *vk = ra_vk_get(ra);
    struct ra_vk *p = ra->priv;

    struct ra_buf *buf = talloc_zero(NULL, struct ra_buf);
    buf->params = *params;
    buf->params.initial_data = NULL;

    struct ra_buf_vk *buf_vk = buf->priv = talloc_zero(buf, struct ra_buf_vk);
    buf_vk->current_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    buf_vk->current_access = 0;
    buf_vk->refcount = 1;

    VkBufferUsageFlags bufFlags = 0;
    VkMemoryPropertyFlags memFlags = 0;
    VkDeviceSize align = 4; // alignment 4 is needed for buf_update

    switch (params->type) {
    case RA_BUF_TEX_TRANSFER:
        bufFlags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        memFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        // Use TRANSFER-style updates for large enough buffers for efficiency
        if (params->size > 1024*1024) // 1 MB
            buf_vk->update_queue = TRANSFER;
        break;
    case RA_BUF_UNIFORM:
        bufFlags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        memFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        align = PL_ALIGN2(align, vk->limits.minUniformBufferOffsetAlignment);
        break;
    case RA_BUF_STORAGE:
        bufFlags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        memFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        align = PL_ALIGN2(align, vk->limits.minStorageBufferOffsetAlignment);
        buf_vk->update_queue = COMPUTE;
        break;
    case RA_VK_BUF_VERTEX:
        bufFlags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        memFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
    default: abort();
    }

    if (params->host_writable || params->initial_data) {
        bufFlags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        align = PL_ALIGN2(align, vk->limits.optimalBufferCopyOffsetAlignment);
    }

    if (params->host_mapped || params->host_readable) {
        memFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                    VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    }

    if (!vk_malloc_buffer(p->alloc, bufFlags, memFlags, params->size, align,
                          &buf_vk->slice))
        goto error;

    if (params->host_mapped)
        buf->data = buf_vk->slice.data;

    if (params->initial_data)
        vk_buf_write(ra, buf, 0, params->initial_data, params->size);

    return buf;

error:
    vk_buf_destroy(ra, buf);
    return NULL;
}

static bool vk_buf_poll(const struct ra *ra, const struct ra_buf *buf,
                        uint64_t timeout)
{
    struct vk_ctx *vk = ra_vk_get(ra);
    struct ra_buf_vk *buf_vk = buf->priv;

    if (timeout > 0) {
        vk_submit(ra);
        vk_poll_commands(vk, timeout);
    }

    return buf_vk->refcount > 1;
}

static bool vk_tex_upload(const struct ra *ra,
                          const struct ra_tex_transfer_params *params)
{
    const struct ra_tex *tex = params->tex;
    struct ra_tex_vk *tex_vk = tex->priv;

    if (!params->buf)
        return ra_tex_upload_pbo(ra, &tex_vk->pbo_write, params);

    struct vk_cmd *cmd = vk_require_cmd(ra, tex_vk->transfer_queue);
    if (!cmd)
        goto error;

    pl_assert(params->buf);
    const struct ra_buf *buf = params->buf;
    struct ra_buf_vk *buf_vk = buf->priv;
    struct pl_rect3d rc = params->rc;
    size_t size = ra_tex_transfer_size(params);

    VkBufferImageCopy region = {
        .bufferOffset = buf_vk->slice.mem.offset + params->buf_offset,
        .bufferRowLength = params->stride_w,
        .bufferImageHeight = params->stride_h,
        .imageOffset = { rc.x0, rc.y0, rc.z0 },
        .imageExtent = { rc.x1, rc.y1, rc.z1 },
        .imageSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .layerCount = 1,
        },
    };

    buf_barrier(ra, cmd, buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_READ_BIT, region.bufferOffset, size, false);

    tex_barrier(ra, cmd, tex, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    vkCmdCopyBufferToImage(cmd->buf, buf_vk->slice.buf, tex_vk->img,
                           tex_vk->current_layout, 1, &region);

    tex_signal(ra, cmd, tex, VK_PIPELINE_STAGE_TRANSFER_BIT);

    return true;

error:
    return false;
}

static bool vk_tex_download(const struct ra *ra,
                            const struct ra_tex_transfer_params *params)
{
    const struct ra_tex *tex = params->tex;
    struct ra_tex_vk *tex_vk = tex->priv;

    if (!params->buf)
        return ra_tex_download_pbo(ra, &tex_vk->pbo_read, params);

    struct vk_cmd *cmd = vk_require_cmd(ra, tex_vk->transfer_queue);
    if (!cmd)
        goto error;

    pl_assert(params->buf);
    const struct ra_buf *buf = params->buf;
    struct ra_buf_vk *buf_vk = buf->priv;
    struct pl_rect3d rc = params->rc;
    size_t size = ra_tex_transfer_size(params);

    VkBufferImageCopy region = {
        .bufferOffset = buf_vk->slice.mem.offset + params->buf_offset,
        .bufferRowLength = params->stride_w,
        .bufferImageHeight = params->stride_h,
        .imageOffset = { rc.x0, rc.y0, rc.z0 },
        .imageExtent = { rc.x1, rc.y1, rc.z1 },
        .imageSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .layerCount = 1,
        },
    };

    buf_barrier(ra, cmd, buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, region.bufferOffset, size, true);

    tex_barrier(ra, cmd, tex, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    vkCmdCopyImageToBuffer(cmd->buf, tex_vk->img, tex_vk->current_layout,
                           buf_vk->slice.buf, 1, &region);

    tex_signal(ra, cmd, tex, VK_PIPELINE_STAGE_TRANSFER_BIT);

    return true;

error:
    return false;
}

static int vk_desc_namespace(const struct ra *ra, enum ra_desc_type type)
{
    return 0;
}

// For ra_pass.priv
struct ra_pass_vk {
    // Pipeline / render pass
    VkPipeline pipe;
    VkPipelineLayout pipeLayout;
    VkRenderPass renderPass;
    VkImageLayout initialLayout;
    VkImageLayout finalLayout;
    // Descriptor set (bindings)
    VkDescriptorSetLayout dsLayout;
    VkDescriptorPool dsPool;
    // To keep track of which descriptor sets are and aren't available, we
    // allocate a fixed number and use a bitmask of all available sets.
    VkDescriptorSet dss[16];
    uint16_t dmask;
    // Vertex buffers (vertices)
    struct ra_buf_pool vbo;

    // For updating
    VkWriteDescriptorSet *dswrite;
    VkDescriptorImageInfo *dsiinfo;
    VkDescriptorBufferInfo *dsbinfo;
};

static void vk_pass_destroy(const struct ra *ra, struct ra_pass *pass)
{
    if (!pass)
        return;

    struct vk_ctx *vk = ra_vk_get(ra);
    struct ra_pass_vk *pass_vk = pass->priv;

    ra_buf_pool_uninit(ra, &pass_vk->vbo);
    vkDestroyPipeline(vk->dev, pass_vk->pipe, VK_ALLOC);
    vkDestroyRenderPass(vk->dev, pass_vk->renderPass, VK_ALLOC);
    vkDestroyPipelineLayout(vk->dev, pass_vk->pipeLayout, VK_ALLOC);
    vkDestroyDescriptorPool(vk->dev, pass_vk->dsPool, VK_ALLOC);
    vkDestroyDescriptorSetLayout(vk->dev, pass_vk->dsLayout, VK_ALLOC);

    talloc_free(pass);
}

MAKE_LAZY_DESTRUCTOR(vk_pass_destroy, struct ra_pass);

static const VkDescriptorType dsType[] = {
    [RA_DESC_SAMPLED_TEX] = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    [RA_DESC_STORAGE_IMG] = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    [RA_DESC_BUF_UNIFORM] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    [RA_DESC_BUF_STORAGE] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
};

static const char vk_cache_magic[4] = {'R','A','V','K'};
static const int vk_cache_version = 2;

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

static bool vk_use_cached_program(const struct ra_pass_params *params,
                                  const struct spirv_compiler *spirv,
                                  struct bstr *vert_spirv,
                                  struct bstr *frag_spirv,
                                  struct bstr *comp_spirv,
                                  struct bstr *pipecache)
{
    struct bstr cache = {
        .start = (void*) params->cached_program,
        .len   =params->cached_program_len,
    };

    if (cache.len < sizeof(struct vk_cache_header))
        return false;

    struct vk_cache_header *header = (struct vk_cache_header *)cache.start;
    cache = bstr_cut(cache, sizeof(*header));

    if (strncmp(header->magic, vk_cache_magic, sizeof(vk_cache_magic)) != 0)
        return false;
    if (header->cache_version != vk_cache_version)
        return false;
    if (strncmp(header->compiler, spirv->name, sizeof(header->compiler)) != 0)
        return false;
    if (header->compiler_version != spirv->compiler_version)
        return false;

#define GET(ptr) \
    if (cache.len < header->ptr##_len)                      \
            return false;                                   \
        *ptr = bstr_splice(cache, 0, header->ptr##_len);    \
        cache = bstr_cut(cache, ptr->len);

    GET(vert_spirv);
    GET(frag_spirv);
    GET(comp_spirv);
    GET(pipecache);
    return true;
}

static VkResult vk_compile_glsl(const struct ra *ra, void *tactx,
                                enum glsl_shader_stage type, const char *glsl,
                                struct bstr *spirv)
{
    struct ra_vk *p = ra->priv;

    static const char *shader_names[] = {
        [GLSL_SHADER_VERTEX]   = "vertex",
        [GLSL_SHADER_FRAGMENT] = "fragment",
        [GLSL_SHADER_COMPUTE]  = "compute",
    };

    PL_DEBUG(ra, "%s shader source:", shader_names[type]);
    pl_msg_source(ra->ctx, PL_LOG_DEBUG, glsl);

    if (!p->spirv->impl->compile_glsl(p->spirv, tactx, type, glsl, spirv)) {
        pl_msg_source(ra->ctx, PL_LOG_ERR, glsl);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    return VK_SUCCESS;
}

static const VkShaderStageFlags stageFlags[] = {
    [RA_PASS_RASTER]  = VK_SHADER_STAGE_FRAGMENT_BIT,
    [RA_PASS_COMPUTE] = VK_SHADER_STAGE_COMPUTE_BIT,
};

static const struct ra_pass *vk_pass_create(const struct ra *ra,
                                            const struct ra_pass_params *params)
{
    struct vk_ctx *vk = ra_vk_get(ra);
    struct ra_vk *p = ra->priv;
    bool success = false;

    struct ra_pass *pass = talloc_zero(NULL, struct ra_pass);
    pass->params = ra_pass_params_copy(pass, params);

    struct ra_pass_vk *pass_vk = pass->priv =
        talloc_zero(pass, struct ra_pass_vk);
    pass_vk->dmask = -1; // all descriptors available

    int num_desc = params->num_descriptors;
    pass_vk->dswrite = talloc_array(pass, VkWriteDescriptorSet, num_desc);
    pass_vk->dsiinfo = talloc_array(pass, VkDescriptorImageInfo, num_desc);
    pass_vk->dsbinfo = talloc_array(pass, VkDescriptorBufferInfo, num_desc);

    // temporary allocations/objects
    void *tmp = talloc_new(NULL);
    VkPipelineCache pipeCache = NULL;
    VkShaderModule vert_shader = NULL;
    VkShaderModule frag_shader = NULL;
    VkShaderModule comp_shader = NULL;

#define NUM_DS (PL_ARRAY_SIZE(pass_vk->dss))

    static int dsSize[RA_DESC_TYPE_COUNT] = {0};
    VkDescriptorSetLayoutBinding *bindings =
        talloc_array(tmp, VkDescriptorSetLayoutBinding, num_desc);

    for (int i = 0; i < num_desc; i++) {
        struct ra_desc *desc = &params->descriptors[i];

        dsSize[desc->type]++;
        bindings[i] = (VkDescriptorSetLayoutBinding) {
            .binding = desc->binding,
            .descriptorType = dsType[desc->type],
            .descriptorCount = 1,
            .stageFlags = stageFlags[params->type],
        };
    }

    VkDescriptorPoolSize *dsPoolSizes = NULL;
    int poolSizeCount = 0;

    for (enum ra_desc_type t = 0; t < RA_DESC_TYPE_COUNT; t++) {
        if (dsSize[t] > 0) {
            VkDescriptorPoolSize dssize = {
                .type = dsType[t],
                .descriptorCount = dsSize[t] * NUM_DS,
            };

            TARRAY_APPEND(tmp, dsPoolSizes, poolSizeCount, dssize);
        }
    }

    if (poolSizeCount) {
        VkDescriptorPoolCreateInfo pinfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = NUM_DS,
            .pPoolSizes = dsPoolSizes,
            .poolSizeCount = poolSizeCount,
        };

        VK(vkCreateDescriptorPool(vk->dev, &pinfo, VK_ALLOC, &pass_vk->dsPool));
    }

    VkDescriptorSetLayoutCreateInfo dinfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pBindings = bindings,
        .bindingCount = num_desc,
    };

    VK(vkCreateDescriptorSetLayout(vk->dev, &dinfo, VK_ALLOC,
                                   &pass_vk->dsLayout));

    VkDescriptorSetLayout layouts[NUM_DS];
    for (int i = 0; i < NUM_DS; i++)
        layouts[i] = pass_vk->dsLayout;

    if (pass_vk->dsPool) {
        VkDescriptorSetAllocateInfo ainfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = pass_vk->dsPool,
            .descriptorSetCount = NUM_DS,
            .pSetLayouts = layouts,
        };

        VK(vkAllocateDescriptorSets(vk->dev, &ainfo, pass_vk->dss));
    }

    VkPipelineLayoutCreateInfo linfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &pass_vk->dsLayout,
        .pushConstantRangeCount = params->push_constants_size ? 1 : 0,
        .pPushConstantRanges = &(VkPushConstantRange){
            .stageFlags = stageFlags[params->type],
            .offset = 0,
            .size = params->push_constants_size,
        },
    };

    VK(vkCreatePipelineLayout(vk->dev, &linfo, VK_ALLOC,
                              &pass_vk->pipeLayout));

    struct bstr vert = {0}, frag = {0}, comp = {0}, pipecache = {0};
    if (vk_use_cached_program(params, p->spirv, &vert, &frag, &comp, &pipecache)) {
        PL_DEBUG(ra, "Using cached SPIR-V and VkPipeline");
    } else {
        pipecache.len = 0;
        switch (params->type) {
        case RA_PASS_RASTER:
            VK(vk_compile_glsl(ra, tmp, GLSL_SHADER_VERTEX,
                               params->vertex_shader, &vert));
            VK(vk_compile_glsl(ra, tmp, GLSL_SHADER_FRAGMENT,
                               params->glsl_shader, &frag));
            comp.len = 0;
            break;
        case RA_PASS_COMPUTE:
            VK(vk_compile_glsl(ra, tmp, GLSL_SHADER_COMPUTE,
                               params->glsl_shader, &comp));
            frag.len = 0;
            vert.len = 0;
            break;
        default: abort();
        }
    }

    VkPipelineCacheCreateInfo pcinfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        .pInitialData = pipecache.start,
        .initialDataSize = pipecache.len,
    };

    VK(vkCreatePipelineCache(vk->dev, &pcinfo, VK_ALLOC, &pipeCache));

    VkShaderModuleCreateInfo sinfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    };

    switch (params->type) {
    case RA_PASS_RASTER: {
        sinfo.pCode = (uint32_t *) vert.start;
        sinfo.codeSize = vert.len;
        VK(vkCreateShaderModule(vk->dev, &sinfo, VK_ALLOC, &vert_shader));

        sinfo.pCode = (uint32_t *) frag.start;
        sinfo.codeSize = frag.len;
        VK(vkCreateShaderModule(vk->dev, &sinfo, VK_ALLOC, &frag_shader));

        VkVertexInputAttributeDescription *attrs = talloc_array(tmp,
                VkVertexInputAttributeDescription, params->num_vertex_attribs);

        for (int i = 0; i < params->num_vertex_attribs; i++) {
            struct ra_vertex_attrib *va = &params->vertex_attribs[i];
            const struct vk_format *fmt_vk = va->fmt->priv;

            attrs[i] = (VkVertexInputAttributeDescription) {
                .binding  = 0,
                .location = va->location,
                .offset   = va->offset,
                .format   = fmt_vk->ifmt,
            };
        }

        pass_vk->finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        // Figure out which case we should try and optimize for based on some
        // dumb heuristics. Extremely naive, but good enough for most cases.
        struct ra_tex_params texparams = params->target_dummy.params;
        if (texparams.storable)
            pass_vk->finalLayout = VK_IMAGE_LAYOUT_GENERAL;
        if (texparams.blit_src || texparams.host_readable)
            pass_vk->finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        if (texparams.blit_dst || texparams.host_writable)
            pass_vk->finalLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        if (texparams.sampleable)
            pass_vk->finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        struct ra_tex_vk *target_vk = params->target_dummy.priv;
        if (target_vk) {
            // If we have a real texture as the target dummy, we can set the
            // initial layout based on what the texture is actually in at the
            // moment.
            pass_vk->initialLayout = target_vk->current_layout;
        } else {
            // Assume we're ping-ponging between a render pass and some other
            // operation. This is the most likely scenario, or rather, the only
            // one we can really optimize for.
            pass_vk->initialLayout = pass_vk->finalLayout;
        }

        VkAttachmentLoadOp loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;

        // If we're blending, then we need to explicitly load the previous
        // contents of the color attachment
        if (pass->params.enable_blend)
            loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;

        // If we're ignoring the FBO, we don't need to load or transition
        if (!pass->params.load_target) {
            pass_vk->initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        }

        VK(vk_create_render_pass(vk->dev, texparams.format, loadOp,
                                 pass_vk->initialLayout, pass_vk->finalLayout,
                                 &pass_vk->renderPass));

        static const VkBlendFactor blendFactors[] = {
            [RA_BLEND_ZERO]                = VK_BLEND_FACTOR_ZERO,
            [RA_BLEND_ONE]                 = VK_BLEND_FACTOR_ONE,
            [RA_BLEND_SRC_ALPHA]           = VK_BLEND_FACTOR_SRC_ALPHA,
            [RA_BLEND_ONE_MINUS_SRC_ALPHA] = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        };

        static const VkPrimitiveTopology topologies[] = {
            [RA_PRIM_TRIANGLE_LIST]  = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            [RA_PRIM_TRIANGLE_STRIP] = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
            [RA_PRIM_TRIANGLE_FAN]   = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
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
                .pAttachments = &(VkPipelineColorBlendAttachmentState) {
                    .blendEnable = params->enable_blend,
                    .colorBlendOp = VK_BLEND_OP_ADD,
                    .srcColorBlendFactor = blendFactors[params->blend_src_rgb],
                    .dstColorBlendFactor = blendFactors[params->blend_dst_rgb],
                    .alphaBlendOp = VK_BLEND_OP_ADD,
                    .srcAlphaBlendFactor = blendFactors[params->blend_src_alpha],
                    .dstAlphaBlendFactor = blendFactors[params->blend_dst_alpha],
                    .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                      VK_COLOR_COMPONENT_G_BIT |
                                      VK_COLOR_COMPONENT_B_BIT |
                                      VK_COLOR_COMPONENT_A_BIT,
                },
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

        VK(vkCreateGraphicsPipelines(vk->dev, pipeCache, 1, &cinfo,
                                     VK_ALLOC, &pass_vk->pipe));
        break;
    }
    case RA_PASS_COMPUTE: {
        sinfo.pCode = (uint32_t *)comp.start;
        sinfo.codeSize = comp.len;
        VK(vkCreateShaderModule(vk->dev, &sinfo, VK_ALLOC, &comp_shader));

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

        VK(vkCreateComputePipelines(vk->dev, pipeCache, 1, &cinfo,
                                    VK_ALLOC, &pass_vk->pipe));
        break;
    }
    default: abort();
    }

    // Update params->cached_program
    struct bstr cache = {0};
    VK(vkGetPipelineCacheData(vk->dev, pipeCache, &cache.len, NULL));
    cache.start = talloc_size(tmp, cache.len);
    VK(vkGetPipelineCacheData(vk->dev, pipeCache, &cache.len, cache.start));

    struct vk_cache_header header = {
        .cache_version = vk_cache_version,
        .compiler_version = p->spirv->compiler_version,
        .vert_spirv_len = vert.len,
        .frag_spirv_len = frag.len,
        .comp_spirv_len = comp.len,
        .pipecache_len = cache.len,
    };

    for (int i = 0; i < PL_ARRAY_SIZE(header.magic); i++)
        header.magic[i] = vk_cache_magic[i];
    for (int i = 0; i < sizeof(p->spirv->name); i++)
        header.compiler[i] = p->spirv->name[i];

    struct bstr prog = {0};
    bstr_xappend(pass, &prog, (struct bstr){ (char *) &header, sizeof(header) });
    bstr_xappend(pass, &prog, vert);
    bstr_xappend(pass, &prog, frag);
    bstr_xappend(pass, &prog, comp);
    bstr_xappend(pass, &prog, cache);
    pass->params.cached_program = prog.start;
    pass->params.cached_program_len = prog.len;

    success = true;

error:
    if (!success) {
        vk_pass_destroy(ra, pass);
        pass = NULL;
    }

#undef NUM_DS

    vkDestroyShaderModule(vk->dev, vert_shader, VK_ALLOC);
    vkDestroyShaderModule(vk->dev, frag_shader, VK_ALLOC);
    vkDestroyShaderModule(vk->dev, comp_shader, VK_ALLOC);
    vkDestroyPipelineCache(vk->dev, pipeCache, VK_ALLOC);
    talloc_free(tmp);
    return pass;
}

static const VkPipelineStageFlags passStages[] = {
    [RA_PASS_RASTER]  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
    [RA_PASS_COMPUTE] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
};

static void vk_update_descriptor(const struct ra *ra, struct vk_cmd *cmd,
                                 const struct ra_pass *pass,
                                 struct ra_desc_binding db,
                                 VkDescriptorSet ds, int idx)
{
    struct ra_pass_vk *pass_vk = pass->priv;
    struct ra_desc *desc = &pass->params.descriptors[idx];
    pl_assert(ds);

    VkWriteDescriptorSet *wds = &pass_vk->dswrite[idx];
    *wds = (VkWriteDescriptorSet) {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = desc->binding,
        .descriptorCount = 1,
        .descriptorType = dsType[desc->type],
    };

    VkAccessFlags access = 0;
    switch (desc->access) {
    case RA_DESC_ACCESS_READONLY:  access = VK_ACCESS_SHADER_READ_BIT; break;
    case RA_DESC_ACCESS_WRITEONLY: access = VK_ACCESS_SHADER_WRITE_BIT; break;
    case RA_DESC_ACCESS_READWRITE:
        access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        break;
    }

    switch (desc->type) {
    case RA_DESC_SAMPLED_TEX: {
        const struct ra_tex *tex = db.object;
        struct ra_tex_vk *tex_vk = tex->priv;

        tex_barrier(ra, cmd, tex, passStages[pass->params.type],
                    VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        VkDescriptorImageInfo *iinfo = &pass_vk->dsiinfo[idx];
        *iinfo = (VkDescriptorImageInfo) {
            .sampler = tex_vk->sampler,
            .imageView = tex_vk->view,
            .imageLayout = tex_vk->current_layout,
        };

        wds->pImageInfo = iinfo;
        break;
    }
    case RA_DESC_STORAGE_IMG: {
        const struct ra_tex *tex = db.object;
        struct ra_tex_vk *tex_vk = tex->priv;

        tex_barrier(ra, cmd, tex, passStages[pass->params.type], access,
                    VK_IMAGE_LAYOUT_GENERAL);

        VkDescriptorImageInfo *iinfo = &pass_vk->dsiinfo[idx];
        *iinfo = (VkDescriptorImageInfo) {
            .imageView = tex_vk->view,
            .imageLayout = tex_vk->current_layout,
        };

        wds->pImageInfo = iinfo;
        break;
    }
    case RA_DESC_BUF_UNIFORM:
    case RA_DESC_BUF_STORAGE: {
        const struct ra_buf *buf = db.object;
        struct ra_buf_vk *buf_vk = buf->priv;

        buf_barrier(ra, cmd, buf, passStages[pass->params.type],
                    access, buf_vk->slice.mem.offset, buf->params.size,
                    access != RA_DESC_ACCESS_READONLY);

        VkDescriptorBufferInfo *binfo = &pass_vk->dsbinfo[idx];
        *binfo = (VkDescriptorBufferInfo) {
            .buffer = buf_vk->slice.buf,
            .offset = buf_vk->slice.mem.offset,
            .range = buf->params.size,
        };

        wds->pBufferInfo = binfo;
        break;
    }
    default: abort();
    }
}

static void vk_release_descriptor(const struct ra *ra, struct vk_cmd *cmd,
                                  const struct ra_pass *pass,
                                  struct ra_desc_binding db, int idx)
{
    const struct ra_desc *desc = &pass->params.descriptors[idx];

    switch (desc->type) {
    case RA_DESC_SAMPLED_TEX:
    case RA_DESC_STORAGE_IMG: {
        const struct ra_tex *tex = db.object;
        tex_signal(ra, cmd, tex, passStages[pass->params.type]);
        break;
    }
    default: break;
    }
}

static void set_ds(struct ra_pass_vk *pass_vk, uintptr_t dsbit)
{
    pass_vk->dmask |= dsbit;
}

static void vk_pass_run(const struct ra *ra,
                        const struct ra_pass_run_params *params)
{
    struct vk_ctx *vk = ra_vk_get(ra);
    const struct ra_pass *pass = params->pass;
    struct ra_pass_vk *pass_vk = pass->priv;

    static const enum queue_type types[] = {
        [RA_PASS_RASTER]  = GRAPHICS,
        [RA_PASS_COMPUTE] = COMPUTE,
    };

    // Wait for a free descriptor set
    while (!pass_vk->dmask) {
        PL_TRACE(ra, "No free descriptor sets! ...blocking (slow path)");
        vk_submit(ra);
        vk_poll_commands(vk, 1000000); // 1ms
    }

    struct vk_cmd *cmd = vk_require_cmd(ra, types[pass->params.type]);
    if (!cmd)
        goto error;

    static const VkPipelineBindPoint bindPoint[] = {
        [RA_PASS_RASTER]  = VK_PIPELINE_BIND_POINT_GRAPHICS,
        [RA_PASS_COMPUTE] = VK_PIPELINE_BIND_POINT_COMPUTE,
    };

    vkCmdBindPipeline(cmd->buf, bindPoint[pass->params.type], pass_vk->pipe);

    VkDescriptorSet ds = NULL;
    for (int i = 0; i < PL_ARRAY_SIZE(pass_vk->dss); i++) {
        uint16_t dsbit = 1u << i;
        if (pass_vk->dmask & dsbit) {
            ds = pass_vk->dss[i];
            pass_vk->dmask &= ~dsbit; // unset
            vk_cmd_callback(cmd, (vk_cb) set_ds, pass_vk,
                            (void *)(uintptr_t) dsbit);
            break;
        }
    }

    for (int i = 0; i < pass->params.num_descriptors; i++)
        vk_update_descriptor(ra, cmd, pass, params->desc_bindings[i], ds, i);

    if (pass->params.num_descriptors > 0) {
        vkUpdateDescriptorSets(vk->dev, pass->params.num_descriptors,
                               pass_vk->dswrite, 0, NULL);
    }

    if (ds) {
        vkCmdBindDescriptorSets(cmd->buf, bindPoint[pass->params.type],
                                pass_vk->pipeLayout, 0, 1, &ds, 0, NULL);
    }

    if (pass->params.push_constants_size) {
        vkCmdPushConstants(cmd->buf, pass_vk->pipeLayout,
                           stageFlags[pass->params.type], 0,
                           pass->params.push_constants_size,
                           params->push_constants);
    }

    switch (pass->params.type) {
    case RA_PASS_RASTER: {
        const struct ra_tex *tex = params->target;
        struct ra_tex_vk *tex_vk = tex->priv;

        struct ra_buf_params vparams = {
            .type = RA_VK_BUF_VERTEX,
            .size = params->vertex_count * pass->params.vertex_stride,
            .host_writable = true,
        };

        const struct ra_buf *buf = ra_buf_pool_get(ra, &pass_vk->vbo, &vparams);
        if (!buf) {
            PL_ERR(ra, "Failed allocating vertex buffer!");
            goto error;
        }

        struct ra_buf_vk *buf_vk = buf->priv;
        vk_buf_write(ra, buf, 0, params->vertex_data, vparams.size);

        buf_barrier(ra, cmd, buf, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                    VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
                    buf_vk->slice.mem.offset, vparams.size, false);

        vkCmdBindVertexBuffers(cmd->buf, 0, 1, &buf_vk->slice.buf,
                               &buf_vk->slice.mem.offset);

        tex_barrier(ra, cmd, tex, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    pass_vk->initialLayout);

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

        vkCmdSetViewport(cmd->buf, 0, 1, &viewport);
        vkCmdSetScissor(cmd->buf, 0, 1, &scissor);

        VkRenderPassBeginInfo binfo = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = pass_vk->renderPass,
            .framebuffer = tex_vk->framebuffer,
            .renderArea = (VkRect2D){{0, 0}, {tex->params.w, tex->params.h}},
        };

        vkCmdBeginRenderPass(cmd->buf, &binfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdDraw(cmd->buf, params->vertex_count, 1, 0, 0);
        vkCmdEndRenderPass(cmd->buf);

        // The renderPass implicitly transitions the texture to this layout
        tex_vk->current_layout = pass_vk->finalLayout;
        tex_signal(ra, cmd, tex, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        break;
    }
    case RA_PASS_COMPUTE:
        vkCmdDispatch(cmd->buf, params->compute_groups[0],
                      params->compute_groups[1],
                      params->compute_groups[2]);
        break;
    default: abort();
    };

    for (int i = 0; i < pass->params.num_descriptors; i++)
        vk_release_descriptor(ra, cmd, pass, params->desc_bindings[i], i);

    // flush the work so far into its own command buffer, for better
    // intra-frame granularity
    vk_submit(ra);

error:
    return;
}

static void vk_flush(const struct ra *ra)
{
    struct vk_ctx *vk = ra_vk_get(ra);
    vk_submit(ra);
    vk_flush_commands(vk);
}

struct vk_cmd *ra_vk_finish_frame(const struct ra *ra, const struct ra_tex *tex)
{
    struct ra_vk *p = ra->priv;
    struct vk_cmd *cmd = vk_require_cmd(ra, GRAPHICS);
    if (!cmd)
        return NULL;

    struct ra_tex_vk *tex_vk = tex->priv;
    pl_assert(tex_vk->external_img);
    tex_barrier(ra, cmd, tex, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                VK_ACCESS_MEMORY_READ_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    // Return this directly instead of going through vk_submit
    p->cmd = NULL;
    return cmd;
}

static struct ra_fns ra_fns_vk = {
    .destroy                = vk_destroy_ra,
    .tex_create             = vk_tex_create,
    .tex_destroy            = vk_tex_destroy_lazy,
    .tex_invalidate         = vk_tex_invalidate,
    .tex_clear              = vk_tex_clear,
    .tex_blit               = vk_tex_blit,
    .tex_upload             = vk_tex_upload,
    .tex_download           = vk_tex_download,
    .buf_create             = vk_buf_create,
    .buf_destroy            = vk_buf_destroy_lazy,
    .buf_write              = vk_buf_write,
    .buf_read               = vk_buf_read,
    .buf_poll               = vk_buf_poll,
    .buf_uniform_layout     = std140_layout,
    .buf_storage_layout     = std430_layout,
    .push_constant_layout   = std430_layout,
    .desc_namespace         = vk_desc_namespace,
    .pass_create            = vk_pass_create,
    .pass_destroy           = vk_pass_destroy_lazy,
    .pass_run               = vk_pass_run,
    .flush                  = vk_flush,
};
