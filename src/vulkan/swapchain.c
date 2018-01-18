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
#include "formats.h"
#include "utils.h"
#include "ra_vk.h"
#include "swapchain.h"

struct priv {
    struct vk_ctx *vk;
    VkSurfaceKHR surf;

    // current swapchain and metadata:
    VkSwapchainCreateInfoKHR protoInfo; // partially filled-in prototype
    VkSwapchainKHR swapchain;
    VkSwapchainKHR old_swapchain;
    int swapchain_depth;
    int frames_in_flight;   // number of frames currently queued
    struct pl_color_repr color_repr;
    struct pl_color_space color_space;

    // state of the images:
    const struct ra_tex **images; // ra_tex wrappers for the VkImages
    int num_images;         // size of `images`
    VkSemaphore *sems_in;   // pool of semaphores used to synchronize images
    VkSemaphore *sems_out;  // outgoing semaphores (rendering complete)
    int num_sems;           // size of `sems_in` / `sems_out`
    int idx_sems;           // index of next free semaphore pair
    int last_imgidx;        // the image index last acquired (for submit)
};

static const struct ra_sw vulkan_swapchain;

static bool pick_surf_format(const struct ra *ra, const struct vk_ctx *vk,
                             VkSurfaceKHR surf, VkSurfaceFormatKHR *out_format)
{
    VkSurfaceFormatKHR *formats = NULL;
    int num = 0;

    VK(vkGetPhysicalDeviceSurfaceFormatsKHR(vk->physd, surf, &num, NULL));
    formats = talloc_array(NULL, VkSurfaceFormatKHR, num);
    VK(vkGetPhysicalDeviceSurfaceFormatsKHR(vk->physd, surf, &num, formats));

    for (int i = 0; i < num; i++) {
        // A value of VK_FORMAT_UNDEFINED means we can pick anything we want
        if (formats[i].format == VK_FORMAT_UNDEFINED) {
            *out_format = (VkSurfaceFormatKHR) {
                .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
                .format = VK_FORMAT_R16G16B16A16_UNORM,
            };
            talloc_free(formats);
            return true;
        }

        // Try and avoid exotic color spaces like HDR output for now
        // TODO: support these and map the correct `pl_color_space`
        if (formats[i].colorSpace != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            continue;

        // Format whitelist, since we want only >= 8 bit _UNORM formats
        switch (formats[i].format) {
        case VK_FORMAT_R8G8B8_UNORM:
        case VK_FORMAT_B8G8R8_UNORM:
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
        case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        case VK_FORMAT_R16G16B16_UNORM:
        case VK_FORMAT_R16G16B16A16_UNORM:
             break; // accept
        default: continue;
        }

        // Make sure we can wrap this format to a meaningful, renderable ra_format
        for (int n = 0; n < ra->num_formats; n++) {
            const struct ra_fmt *rafmt = ra->formats[n];
            const struct vk_format *vkfmt = rafmt->priv;
            if (vkfmt->ifmt != formats[i].format)
                continue;
            if (!(rafmt->caps & RA_FMT_CAP_RENDERABLE))
                continue;
            // format valid, use it
            *out_format = formats[i];
            talloc_free(formats);
            return true;
        }
    }

    // fall through
error:
    PL_FATAL(vk, "Failed picking any valid, renderable surface format!");
    talloc_free(formats);
    return false;
}

const struct ra_swapchain *pl_vulkan_create_swapchain(const struct pl_vulkan *plvk,
                              const struct pl_vulkan_swapchain_params *params)
{
    struct vk_ctx *vk = plvk->priv;
    const struct ra *ra = plvk->ra;

    VkSurfaceFormatKHR sfmt = params->surface_format;
    if (!sfmt.format && !pick_surf_format(ra, vk, params->surface, &sfmt))
        return NULL;

    struct ra_swapchain *sw = talloc_zero(NULL, struct ra_swapchain);
    sw->impl = &vulkan_swapchain;
    sw->ctx = vk->ctx;
    sw->ra = ra;

    struct priv *p = sw->priv = talloc_zero(sw, struct priv);
    p->vk = vk;
    p->surf = params->surface;
    p->swapchain_depth = PL_DEF(params->swapchain_depth, 3);
    pl_assert(p->swapchain_depth > 0);
    p->protoInfo = (VkSwapchainCreateInfoKHR) {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = p->surf,
        .imageFormat = sfmt.format,
        .imageColorSpace = sfmt.colorSpace,
        .imageArrayLayers = 1, // non-stereoscopic
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .minImageCount = p->swapchain_depth + 1, // +1 for the FB
        .presentMode = params->present_mode,
        .clipped = true,
    };

    p->color_space = pl_color_space_monitor;
    p->color_repr = (struct pl_color_repr) {
        .sys    = PL_COLOR_SYSTEM_RGB,
        .levels = PL_COLOR_LEVELS_PC,
        .alpha  = PL_ALPHA_UNKNOWN, // will be updated by vk_sw_recreate
    };

    // Make sure the swapchain present mode is supported
    VkPresentModeKHR *modes = NULL;
    int num_modes;
    VK(vkGetPhysicalDeviceSurfacePresentModesKHR(vk->physd, p->surf,
                                                 &num_modes, NULL));
    modes = talloc_array(NULL, VkPresentModeKHR, num_modes);
    VK(vkGetPhysicalDeviceSurfacePresentModesKHR(vk->physd, p->surf,
                                                 &num_modes, modes));

    bool supported = false;
    for (int i = 0; i < num_modes; i++)
        supported |= (modes[i] == p->protoInfo.presentMode);
    talloc_free(modes);

    if (!supported) {
        PL_WARN(vk, "Requested swap mode unsupported by this device, falling "
                "back to VK_PRESENT_MODE_FIFO_KHR");
        p->protoInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    }

    return sw;

error:
    talloc_free(modes);
    talloc_free(sw);
    return NULL;
}

static void vk_sw_destroy(const struct ra_swapchain *sw)
{
    const struct ra *ra = sw->ra;
    struct priv *p = sw->priv;
    struct vk_ctx *vk = p->vk;

    ra_flush(ra);
    vk_wait_idle(vk);
    for (int i = 0; i < p->num_images; i++)
        ra_tex_destroy(ra, &p->images[i]);
    for (int i = 0; i < p->num_sems; i++) {
        vkDestroySemaphore(vk->dev, p->sems_in[i], VK_ALLOC);
        vkDestroySemaphore(vk->dev, p->sems_out[i], VK_ALLOC);
    }

    vkDestroySwapchainKHR(vk->dev, p->swapchain, VK_ALLOC);
    talloc_free((void *) sw);
}

static int vk_sw_latency(const struct ra_swapchain *sw)
{
    struct priv *p = sw->priv;
    return p->swapchain_depth;
}

static bool update_swapchain_info(struct priv *p, VkSwapchainCreateInfoKHR *info)
{
    struct vk_ctx *vk = p->vk;

    // Query the supported capabilities and update this struct as needed
    VkSurfaceCapabilitiesKHR caps;
    VK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk->physd, p->surf, &caps));

    // Sorted by preference
    static const struct { VkCompositeAlphaFlagsKHR vk_mode;
                          enum pl_alpha_mode pl_mode;
                        } alphaModes[] = {
        {VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,  PL_ALPHA_PREMULTIPLIED},
        {VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR, PL_ALPHA_INDEPENDENT},
        {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,          PL_ALPHA_UNKNOWN},
    };

    for (int i = 0; i < PL_ARRAY_SIZE(alphaModes); i++) {
        if (caps.supportedCompositeAlpha & alphaModes[i].vk_mode) {
            info->compositeAlpha = alphaModes[i].vk_mode;
            p->color_repr.alpha = alphaModes[i].pl_mode;
            break;
        }
    }

    if (!info->compositeAlpha) {
        PL_ERR(vk, "Failed picking alpha compositing mode (caps: 0x%x)",
               caps.supportedCompositeAlpha);
        goto error;
    }

    // Note: We could probably also allow picking a surface transform that
    // flips the framebuffer and set `ra_swapchain_frame.flipped`, but this
    // doesn't appear to be necessary for any vulkan implementations.
    static const VkSurfaceTransformFlagsKHR rotModes[] = {
        VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR,
    };

    for (int i = 0; i < PL_ARRAY_SIZE(rotModes); i++) {
        if (caps.supportedTransforms & rotModes[i]) {
            info->preTransform = rotModes[i];
            break;
        }
    }

    if (!info->preTransform) {
        PL_ERR(vk, "Failed picking surface transform mode (caps: 0x%x)",
               caps.supportedTransforms);
        goto error;
    }

    // Image count as required
    PL_DEBUG(vk, "Requested image count: %d (min %d max %d)",
             (int) info->minImageCount, (int) caps.minImageCount,
             (int) caps.maxImageCount);

    info->minImageCount = PL_MAX(info->minImageCount, caps.minImageCount);
    if (caps.maxImageCount)
        info->minImageCount = PL_MIN(info->minImageCount, caps.maxImageCount);

    // This seems to be an obscure case, and doesn't make sense anyway. So just
    // ignore it and assume we're using a sane environment where the current
    // window size is known.
    if (caps.currentExtent.width == 0xFFFFFFFF ||
        caps.currentExtent.height == 0xFFFFFFFF)
    {
        PL_ERR(vk, "The swapchain's current extent is reported as unknown. "
               "In other words, we don't know the size of the window. Giving up!");
        goto error;
    }

    // This seems to be an obscure case that should technically violate the spec
    // anyway, but better safe than sorry..
    if (!caps.currentExtent.width || !caps.currentExtent.height) {
        PL_WARN(vk, "Unable to recreate swapchain: image extent is 0, possibly "
                "the window is minimized or hidden?");
        goto error;
    }

    info->imageExtent = caps.currentExtent;

    // We just request whatever usage we can, and let the ra_vk decide what
    // ra_tex_params that translates to. This makes the images as flexible
    // as possible.
    info->imageUsage = caps.supportedUsageFlags;
    return true;

error:
    return false;
}

static void destroy_swapchain(struct vk_ctx *vk, struct priv *p)
{
    assert(p->old_swapchain);
    vkDestroySwapchainKHR(vk->dev, p->old_swapchain, VK_ALLOC);
    p->old_swapchain = NULL;
}

static bool vk_sw_recreate(const struct ra_swapchain *sw)
{
    const struct ra *ra = sw->ra;
    struct priv *p = sw->priv;
    struct vk_ctx *vk = p->vk;

    VkImage *vkimages = NULL;
    int num_images = 0;

    // It's invalid to trigger another swapchain recreation while there's more
    // than one swapchain already active, so we need to flush any pending
    // asynchronous swapchain release operations that may be ongoing
    while (p->old_swapchain)
        vk_poll_commands(vk, 1000000); // 1 ms

    VkSwapchainCreateInfoKHR sinfo = p->protoInfo;
    sinfo.oldSwapchain = p->swapchain;

    if (!update_swapchain_info(p, &sinfo))
        goto error;

    VK(vkCreateSwapchainKHR(vk->dev, &sinfo, VK_ALLOC, &p->swapchain));

    // Freeing the old swapchain while it's still in use is an error, so do it
    // asynchronously once the device is idle
    if (sinfo.oldSwapchain) {
        p->old_swapchain = sinfo.oldSwapchain;
        vk_dev_callback(vk, (vk_cb) destroy_swapchain, vk, p);
    }

    // Get the new swapchain images
    VK(vkGetSwapchainImagesKHR(vk->dev, p->swapchain, &num_images, NULL));
    vkimages = talloc_array(NULL, VkImage, num_images);
    VK(vkGetSwapchainImagesKHR(vk->dev, p->swapchain, &num_images, vkimages));

    // If needed, allocate some more semaphores
    while (num_images > p->num_sems) {
        VkSemaphore sem_in, sem_out;
        static const VkSemaphoreCreateInfo seminfo = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        };
        VK(vkCreateSemaphore(vk->dev, &seminfo, VK_ALLOC, &sem_in));
        VK(vkCreateSemaphore(vk->dev, &seminfo, VK_ALLOC, &sem_out));

        int idx = p->num_sems++;
        TARRAY_GROW(p, p->sems_in, idx);
        TARRAY_GROW(p, p->sems_out, idx);
        p->sems_in[idx] = sem_in;
        p->sems_out[idx] = sem_out;
    }

    // Recreate the ra_tex wrappers
    for (int i = 0; i < p->num_images; i++)
        ra_tex_destroy(ra, &p->images[i]);

    p->num_images = num_images;
    TARRAY_GROW(p, p->images, num_images);
    for (int i = 0; i < num_images; i++) {
        p->images[i] = ra_vk_wrap_swapchain_img(ra, vkimages[i], sinfo);
        if (!p->images[i])
            goto error;
    }

    pl_assert(num_images > 0);
    int bits = 0;

    // The channel with the most bits is probably the most authoritative about
    // the actual color information (consider e.g. a2bgr10). Slight downside
    // in that it results in rounding r/b for e.g. rgb565, but we don't pick
    // surfaces with fewer than 8 bits anyway, so let's not care for now.
    const struct ra_fmt *fmt = p->images[0]->params.format;
    for (int i = 0; i < fmt->num_components; i++)
        bits = PL_MAX(bits, fmt->component_depth[i]);

    p->color_repr.bits.sample_depth = bits;
    p->color_repr.bits.color_depth = bits;

    talloc_free(vkimages);
    return true;

error:
    PL_ERR(vk, "Failed (re)creating swapchain!");
    talloc_free(vkimages);
    vkDestroySwapchainKHR(vk->dev, p->swapchain, VK_ALLOC);
    p->swapchain = NULL;
    return false;
}

static bool vk_sw_start_frame(const struct ra_swapchain *sw,
                              struct ra_swapchain_frame *out_frame)
{
    struct priv *p = sw->priv;
    struct vk_ctx *vk = p->vk;
    if (!p->swapchain && !vk_sw_recreate(sw))
        return false;

    VkSemaphore sem_in = p->sems_in[p->idx_sems];
    PL_TRACE(vk, "vkAcquireNextImageKHR signals %p", (void *) sem_in);

    for (int attempts = 0; attempts < 2; attempts++) {
        uint32_t imgidx = 0;
        VkResult res = vkAcquireNextImageKHR(vk->dev, p->swapchain, UINT64_MAX,
                                             sem_in, NULL, &imgidx);

        switch (res) {
        case VK_SUCCESS:
            p->last_imgidx = imgidx;
            *out_frame = (struct ra_swapchain_frame) {
                .fbo = p->images[imgidx],
                .flipped = false,
                .color_repr = p->color_repr,
                .color_space = p->color_space,
            };
            ra_tex_vk_external_dep(sw->ra, out_frame->fbo, sem_in);
            return true;

        case VK_ERROR_OUT_OF_DATE_KHR: {
            // In these cases try recreating the swapchain
            if (!vk_sw_recreate(sw))
                return false;
            continue;
        }

        default:
            PL_ERR(vk, "Failed acquiring swapchain image: %s", vk_res_str(res));
            return false;
        }
    }

    // If we've exhausted the number of attempts to recreate the swapchain,
    // just give up silently and let the user retry some time later.
    return false;
}

static void present_cb(struct priv *p, void *arg)
{
    p->frames_in_flight--;
}

static bool vk_sw_submit_frame(const struct ra_swapchain *sw)
{
    const struct ra *ra = sw->ra;
    struct priv *p = sw->priv;
    struct vk_ctx *vk = p->vk;
    if (!p->swapchain)
        return false;

    struct vk_cmd *cmd = ra_vk_finish_frame(ra, p->images[p->last_imgidx]);
    if (!cmd)
        return false;

    VkSemaphore sem_out = p->sems_out[p->idx_sems++];
    p->idx_sems %= p->num_sems;
    vk_cmd_sig(cmd, sem_out);

    p->frames_in_flight++;
    vk_cmd_callback(cmd, (vk_cb) present_cb, p, NULL);

    vk_cmd_queue(vk, cmd);
    if (!vk_flush_commands(vk))
        return false;

    // Older nvidia drivers can spontaneously combust when submitting to the
    // same queue as we're rendering from, in a multi-queue scenario. Safest
    // option is to flush the commands first and then submit to the next queue.
    // We can drop this hack in the future, I suppose.
    struct vk_cmdpool *pool = vk->pool_graphics;
    VkQueue queue = pool->queues[pool->idx_queues];

    VkPresentInfoKHR pinfo = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &sem_out,
        .swapchainCount = 1,
        .pSwapchains = &p->swapchain,
        .pImageIndices = &p->last_imgidx,
    };

    PL_TRACE(vk, "vkQueuePresentKHR waits on %p", (void *) sem_out);
    VkResult res = vkQueuePresentKHR(queue, &pinfo);
    switch (res) {
    case VK_SUCCESS:
    case VK_SUBOPTIMAL_KHR:
        return true;

    case VK_ERROR_OUT_OF_DATE_KHR:
        // We can silently ignore this error, since the next start_frame will
        // recreate the swapchain automatically.
        return true;

    default:
        PL_ERR(vk, "Failed presenting to queue %p: %s", (void *) queue,
               vk_res_str(res));
        return false;
    }
}

static void vk_sw_swap_buffers(const struct ra_swapchain *sw)
{
    struct priv *p = sw->priv;

    while (p->frames_in_flight >= p->swapchain_depth)
        vk_poll_commands(p->vk, 1000000); // 1 ms
}

static const struct ra_sw vulkan_swapchain = {
    .destroy      = vk_sw_destroy,
    .latency      = vk_sw_latency,
    .start_frame  = vk_sw_start_frame,
    .submit_frame = vk_sw_submit_frame,
    .swap_buffers = vk_sw_swap_buffers,
};
