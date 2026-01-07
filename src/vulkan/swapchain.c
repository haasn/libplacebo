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
#include "gpu.h"
#include "swapchain.h"
#include "pl_thread.h"

struct vk_swapchain {
    VkSwapchainKHR swapchain;
    // state of the images:
    PL_ARRAY(pl_tex) images;        // pl_tex wrappers for the VkImages
    PL_ARRAY(VkSemaphore) sems_in;  // pool of semaphores used to acquire images
    PL_ARRAY(VkSemaphore) sems_out; // pool of semaphores used to present images
    PL_ARRAY(VkFence) fences_out;   // pool of fences for presented images
    int idx_sems_in;                // index of next free semaphore to acquire
    int last_imgidx;                // the image index last acquired (for submit)
};

struct priv {
    struct pl_sw_fns impl;

    pl_mutex lock;
    struct vk_ctx *vk;
    VkSurfaceKHR surf;
    PL_ARRAY(VkSurfaceFormatKHR) formats;

    // current swapchain and metadata:
    struct pl_vulkan_swapchain_params params;
    VkSwapchainCreateInfoKHR protoInfo; // partially filled-in prototype
    struct vk_swapchain *current;
    PL_ARRAY(struct vk_swapchain*) retired;
    uint32_t queue_families[3];
    int cur_width, cur_height;
    int swapchain_depth;
    pl_rc_t frames_in_flight;       // number of frames currently queued
    bool suboptimal;                // true once VK_SUBOPTIMAL_KHR is returned
    bool needs_recreate;            // swapchain needs to be recreated
    bool has_swapchain_maintenance1;
    struct pl_color_repr color_repr;
    struct pl_color_space color_space;
    struct pl_hdr_metadata hdr_metadata;
};

static const struct pl_sw_fns vulkan_swapchain;


static bool map_color_space(VkColorSpaceKHR space, struct pl_color_space *out)
{
    switch (space) {
    case VK_COLOR_SPACE_SRGB_NONLINEAR_KHR:
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_BT_709,
            .transfer  = PL_COLOR_TRC_SRGB,
        };
        return true;
    case VK_COLOR_SPACE_BT709_NONLINEAR_EXT:
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_BT_709,
            .transfer  = PL_COLOR_TRC_BT_1886,
        };
        return true;
    case VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT:
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_DISPLAY_P3,
            // Actually there are some controversy about Display P3's TRC curve,
            // just like sRGB
            .transfer  = PL_COLOR_TRC_SRGB,
        };
        return true;
    case VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT:
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_DISPLAY_P3,
            .transfer  = PL_COLOR_TRC_LINEAR,
        };
        return true;
    case VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT:
        // This color space is using XYZ color system than RGB
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_DCI_P3,
            .transfer  = PL_COLOR_TRC_ST428,
        };
        return true;
    case VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT:
    case VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT:
        // TODO
        return false;
    case VK_COLOR_SPACE_BT709_LINEAR_EXT:
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_BT_709,
            .transfer  = PL_COLOR_TRC_LINEAR,
        };
        return true;
    case VK_COLOR_SPACE_BT2020_LINEAR_EXT:
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_BT_2020,
            .transfer  = PL_COLOR_TRC_LINEAR,
        };
        return true;
    case VK_COLOR_SPACE_HDR10_ST2084_EXT:
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_BT_2020,
            .transfer  = PL_COLOR_TRC_PQ,
        };
        return true;
    case VK_COLOR_SPACE_DOLBYVISION_EXT:
        // Unlikely to ever be implemented
        return false;
    case VK_COLOR_SPACE_HDR10_HLG_EXT:
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_BT_2020,
            .transfer  = PL_COLOR_TRC_HLG,
        };
        return true;
    case VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT:
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_ADOBE,
            .transfer  = PL_COLOR_TRC_LINEAR,
        };
        return true;
    case VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT:
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_ADOBE,
            .transfer  = PL_COLOR_TRC_GAMMA22,
        };
        return true;
    case VK_COLOR_SPACE_PASS_THROUGH_EXT:
        // On color-managed Wayland compositors, it behaves similar to
        // VK_COLOR_SPACE_SRGB_NONLINEAR_KHR as they would treat un-tagged surface
        // as sRGB, but on other OSes it's behavior is not clearly defined, so
        // don't use it.
        return false;

#ifdef VK_AMD_display_native_hdr
    case VK_COLOR_SPACE_DISPLAY_NATIVE_AMD:
        // TODO
        return false;
#endif

    default: return false;
    }
}

static bool pick_surf_format(pl_swapchain sw, const struct pl_color_space *hint)
{
    struct priv *p = PL_PRIV(sw);
    struct vk_ctx *vk = p->vk;
    pl_gpu gpu = sw->gpu;

    int best_score = 0, best_id;
    bool wide_gamut = pl_color_primaries_is_wide_gamut(hint->primaries);
    bool prefer_hdr = pl_color_transfer_is_hdr(hint->transfer);

    for (int i = 0; i < p->formats.num; i++) {
        // Color space / format whitelist
        struct pl_color_space space;
        if (!map_color_space(p->formats.elem[i].colorSpace, &space))
            continue;

        bool disable10 = !pl_color_transfer_is_hdr(space.transfer) &&
                         p->params.disable_10bit_sdr;

        // Make sure we can wrap this format to a meaningful, valid pl_fmt
        for (int n = 0; n < gpu->num_formats; n++) {
            pl_fmt plfmt = gpu->formats[n];
            const struct vk_format **pvkfmt = PL_PRIV(plfmt);
            if ((*pvkfmt)->tfmt != p->formats.elem[i].format)
                continue;

            enum pl_fmt_caps render_caps = 0;
            render_caps |= PL_FMT_CAP_RENDERABLE;
            render_caps |= PL_FMT_CAP_BLITTABLE;
            if ((plfmt->caps & render_caps) != render_caps)
                continue;

            // format valid, use it if it has a higher score
            int score = 0;
            switch (plfmt->component_depth[0]) {
                case 8:
                    if (pl_color_transfer_is_hdr(space.transfer))
                        score += 10;
                    else if (space.transfer == PL_COLOR_TRC_LINEAR)
                        continue; // avoid 8-bit linear formats
                    else if (disable10)
                        score += 30;
                    else
                        score += 20;
                    break;
                case 10:
                    if (pl_color_transfer_is_hdr(space.transfer))
                        score += 30;
                    else if (space.transfer == PL_COLOR_TRC_LINEAR)
                        continue; // avoid 10-bit linear formats
                    else if (disable10)
                        score += 20;
                    else
                        score += 30;
                    break;
                case 16:
                    if (pl_color_transfer_is_hdr(space.transfer))
                        score += 20;
                    else if (space.transfer == PL_COLOR_TRC_LINEAR)
                        score += 30;
                    else if (disable10)
                        score += 10;
                    else
                        score += 10;
                    break;
                default: // skip any other format
                    continue;
            }
            if (pl_fmt_is_ordered(plfmt))
                score += 500;
            if (pl_color_primaries_is_wide_gamut(space.primaries) == wide_gamut)
                score += 1000;
            if (space.primaries == hint->primaries)
                score += 2000;
            if (pl_color_transfer_is_hdr(space.transfer) == prefer_hdr)
                score += 10000;
            if (space.transfer == hint->transfer)
                score += 20000;

            switch (plfmt->type) {
            case PL_FMT_UNKNOWN: break;
            case PL_FMT_UINT: break;
            case PL_FMT_SINT: break;
            case PL_FMT_UNORM: score += 3; break;
            case PL_FMT_SNORM: score += 2; break;
            case PL_FMT_FLOAT: score += 1; break;
            case PL_FMT_TYPE_COUNT: pl_unreachable();
            };

            if (score > best_score) {
                best_score = score;
                best_id = i;
                break;
            }
        }
    }

    if (!best_score) {
        PL_ERR(vk, "Failed picking any valid, renderable surface format!");
        return false;
    }

    VkSurfaceFormatKHR new_sfmt = p->formats.elem[best_id];
    if (p->protoInfo.imageFormat != new_sfmt.format ||
        p->protoInfo.imageColorSpace != new_sfmt.colorSpace)
    {
        PL_INFO(vk, "Picked surface configuration %d: %s + %s", best_id,
                vk_fmt_name(new_sfmt.format),
                vk_csp_name(new_sfmt.colorSpace));

        p->protoInfo.imageFormat = new_sfmt.format;
        p->protoInfo.imageColorSpace = new_sfmt.colorSpace;
        p->needs_recreate = true;
    }

    return true;
}

static void set_hdr_metadata(struct priv *p, const struct pl_hdr_metadata *metadata)
{
    struct vk_ctx *vk = p->vk;
    if (!vk->SetHdrMetadataEXT)
        return;

    // Whitelist only values that we support signalling metadata for
    struct pl_hdr_metadata fix = {
        .prim     = metadata->prim,
        .min_luma = metadata->min_luma,
        .max_luma = metadata->max_luma,
        .max_cll  = metadata->max_cll,
        .max_fall = metadata->max_fall,
    };

    // Ignore no-op changes
    if (pl_hdr_metadata_equal(&fix, &p->hdr_metadata))
        return;

    // Remember the metadata so we can re-apply it after swapchain recreation
    p->hdr_metadata = fix;

    // Ignore HDR metadata requests for SDR swapchains
    if (!pl_color_transfer_is_hdr(p->color_space.transfer))
        return;

    if (!p->current)
        return;

    vk->SetHdrMetadataEXT(vk->dev, 1, &p->current->swapchain, &(VkHdrMetadataEXT) {
        .sType = VK_STRUCTURE_TYPE_HDR_METADATA_EXT,
        .displayPrimaryRed   = { fix.prim.red.x,   fix.prim.red.y },
        .displayPrimaryGreen = { fix.prim.green.x, fix.prim.green.y },
        .displayPrimaryBlue  = { fix.prim.blue.x,  fix.prim.blue.y },
        .whitePoint = { fix.prim.white.x, fix.prim.white.y },
        .maxLuminance = fix.max_luma,
        .minLuminance = fix.min_luma,
        .maxContentLightLevel = fix.max_cll,
        .maxFrameAverageLightLevel = fix.max_fall,
    });

    // Keep track of applied HDR colorimetry metadata
    p->color_space.hdr = p->hdr_metadata;
}

pl_swapchain pl_vulkan_create_swapchain(pl_vulkan plvk,
                              const struct pl_vulkan_swapchain_params *params)
{
    struct vk_ctx *vk = PL_PRIV(plvk);
    pl_gpu gpu = plvk->gpu;

    if (!vk->CreateSwapchainKHR) {
        PL_ERR(gpu, VK_KHR_SWAPCHAIN_EXTENSION_NAME " not enabled!");
        return NULL;
    }

    struct pl_swapchain_t *sw = pl_zalloc_obj(NULL, sw, struct priv);
    sw->log = vk->log;
    sw->gpu = gpu;

    const VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR *sw_maint_features =
    vk_find_struct(vk->features.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR);

    struct priv *p = PL_PRIV(sw);
    pl_mutex_init(&p->lock);
    p->impl = vulkan_swapchain;
    p->params = *params;
    p->vk = vk;
    p->surf = params->surface;
    p->swapchain_depth = PL_DEF(params->swapchain_depth, 3);
    p->has_swapchain_maintenance1 = sw_maint_features && sw_maint_features->swapchainMaintenance1;
    pl_assert(p->swapchain_depth > 0);
    atomic_init(&p->frames_in_flight, 0);
    p->protoInfo = (VkSwapchainCreateInfoKHR) {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = p->surf,
        .imageArrayLayers = 1, // non-stereoscopic
        .imageSharingMode = vk->pools.num > 1 ? VK_SHARING_MODE_CONCURRENT
                                              : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = vk->pools.num,
        .pQueueFamilyIndices = p->queue_families,
        .minImageCount = p->swapchain_depth + 1, // +1 for the FB
        .presentMode = params->present_mode,
        .clipped = true,
    };

    pl_assert(vk->pools.num <= PL_ARRAY_SIZE(p->queue_families));
    for (int i = 0; i < vk->pools.num; i++)
        p->queue_families[i] = vk->pools.elem[i]->qf;

    // These fields will be updated by `vk_sw_recreate`
    p->color_space = pl_color_space_unknown;
    p->color_repr = (struct pl_color_repr) {
        .sys    = PL_COLOR_SYSTEM_RGB,
        .levels = PL_COLOR_LEVELS_FULL,
        .alpha  = PL_ALPHA_UNKNOWN,
    };

    // Make sure the swapchain present mode is supported
    VkPresentModeKHR *modes = NULL;
    uint32_t num_modes = 0;
    VK(vk->GetPhysicalDeviceSurfacePresentModesKHR(vk->physd, p->surf, &num_modes, NULL));
    modes = pl_calloc_ptr(NULL, num_modes, modes);
    VK(vk->GetPhysicalDeviceSurfacePresentModesKHR(vk->physd, p->surf, &num_modes, modes));

    bool supported = false;
    for (int i = 0; i < num_modes; i++)
        supported |= (modes[i] == p->protoInfo.presentMode);
    pl_free_ptr(&modes);

    if (!supported) {
        PL_WARN(vk, "Requested swap mode unsupported by this device, falling "
                "back to VK_PRESENT_MODE_FIFO_KHR");
        p->protoInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    }

    // Enumerate the supported surface color spaces
    uint32_t num_formats = 0;
    VK(vk->GetPhysicalDeviceSurfaceFormatsKHR(vk->physd, p->surf, &num_formats, NULL));
    PL_ARRAY_RESIZE(sw, p->formats, num_formats);
    VK(vk->GetPhysicalDeviceSurfaceFormatsKHR(vk->physd, p->surf, &num_formats, p->formats.elem));
    p->formats.num = num_formats;

    PL_INFO(gpu, "Available surface configurations:");
    for (int i = 0; i < p->formats.num; i++) {
        PL_INFO(gpu, "    %d: %-40s %s", i,
                vk_fmt_name(p->formats.elem[i].format),
                vk_csp_name(p->formats.elem[i].colorSpace));
    }

    // Ensure there exists at least some valid renderable surface format
    struct pl_color_space hint = pl_color_space_srgb;
    if (!pick_surf_format(sw, &hint))
        goto error;

    return sw;

error:
    pl_free(modes);
    pl_free(sw);
    return NULL;
}

static bool swapchain_destroy(pl_swapchain sw, struct vk_swapchain **ptr,
                                uint64_t timeout)
{
    struct priv *p = PL_PRIV(sw);
    struct vk_ctx *vk = p->vk;
    struct vk_swapchain *vk_sw = *ptr;

    if (vk_sw->fences_out.num > 0) {
        VkResult res = vk->WaitForFences(vk->dev, vk_sw->fences_out.num,
                                         vk_sw->fences_out.elem, VK_TRUE, timeout);
        if (res == VK_NOT_READY || res == VK_TIMEOUT)
            return false;
        for (int i = 0; i < vk_sw->fences_out.num; i++)
            vk->DestroyFence(vk->dev, vk_sw->fences_out.elem[i], PL_VK_ALLOC);
    } else {
        // Vulkan without VK_KHR_swapchain_maintenance1 offers no way to know
        // when a queue presentation command is done using these resources,
        // leading to undefined behavior when destroying resources tied to the
        // swapchain. Use an extra `vkQueueWaitIdle` on all of the queues we may
        // have oustanding presentation calls on, to mitigate this risk.
        for (int i = 0; i < vk->pool_graphics->num_queues; i++)
            vk->QueueWaitIdle(vk->pool_graphics->queues[i]);
    }

    for (int i = 0; i < vk_sw->images.num; i++)
        pl_tex_destroy(sw->gpu, &vk_sw->images.elem[i]);
    for (int i = 0; i < vk_sw->sems_in.num; i++)
        vk->DestroySemaphore(vk->dev, vk_sw->sems_in.elem[i], PL_VK_ALLOC);
    for (int i = 0; i < vk_sw->sems_out.num; i++)
        vk->DestroySemaphore(vk->dev, vk_sw->sems_out.elem[i], PL_VK_ALLOC);

    vk->DestroySwapchainKHR(vk->dev, vk_sw->swapchain, PL_VK_ALLOC);
    pl_free_ptr(ptr);
    return true;
}

static void cleanup_retired_swapchains(pl_swapchain sw, uint64_t timeout)
{
    struct priv *p = PL_PRIV(sw);
    for (int i = 0; i < p->retired.num; i++) {
        if (swapchain_destroy(sw, &p->retired.elem[i], timeout)) {
            PL_ARRAY_REMOVE_AT(p->retired, i);
            i--;
        }
    }
}

static void vk_sw_destroy(pl_swapchain sw)
{
    pl_gpu gpu = sw->gpu;
    struct priv *p = PL_PRIV(sw);
    struct vk_ctx *vk = p->vk;

    pl_gpu_flush(gpu);
    vk_wait_idle(vk);

    cleanup_retired_swapchains(sw, UINT64_MAX);
    swapchain_destroy(sw, &p->current, UINT64_MAX);

    pl_mutex_destroy(&p->lock);
    pl_free((void *) sw);
}

static int vk_sw_latency(pl_swapchain sw)
{
    struct priv *p = PL_PRIV(sw);
    return p->swapchain_depth;
}

static bool update_swapchain_info(struct priv *p, VkSwapchainCreateInfoKHR *info,
                                  int w, int h)
{
    struct vk_ctx *vk = p->vk;

    // Query the supported capabilities and update this struct as needed
    VkSurfaceCapabilitiesKHR caps = {0};
    VK(vk->GetPhysicalDeviceSurfaceCapabilitiesKHR(vk->physd, p->surf, &caps));

    // Check for hidden/invisible window
    if (!caps.currentExtent.width || !caps.currentExtent.height) {
        PL_DEBUG(vk, "maxImageExtent reported as 0x0, hidden window? skipping");
        return false;
    }

    // Sorted by preference
    static const struct { VkCompositeAlphaFlagsKHR vk_mode;
                          enum pl_alpha_mode pl_mode;
                        } alphaModes[] = {
        {VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR, PL_ALPHA_INDEPENDENT},
        {VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,  PL_ALPHA_PREMULTIPLIED},
        {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,          PL_ALPHA_UNKNOWN},
        {VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,         PL_ALPHA_UNKNOWN},
    };

    for (int i = 0; i < PL_ARRAY_SIZE(alphaModes); i++) {
        if (caps.supportedCompositeAlpha & alphaModes[i].vk_mode) {
            info->compositeAlpha = alphaModes[i].vk_mode;
            p->color_repr.alpha = alphaModes[i].pl_mode;
            PL_DEBUG(vk, "Requested alpha compositing mode: %s",
                     vk_alpha_mode(info->compositeAlpha));
            break;
        }
    }

    if (!info->compositeAlpha) {
        PL_ERR(vk, "Failed picking alpha compositing mode (caps: 0x%x)",
               caps.supportedCompositeAlpha);
        goto error;
    }

    // Note: We could probably also allow picking a surface transform that
    // flips the framebuffer and set `pl_swapchain_frame.flipped`, but this
    // doesn't appear to be necessary for any vulkan implementations.
    static const VkSurfaceTransformFlagsKHR rotModes[] = {
        VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR,
    };

    for (int i = 0; i < PL_ARRAY_SIZE(rotModes); i++) {
        if (caps.supportedTransforms & rotModes[i]) {
            info->preTransform = rotModes[i];
            PL_DEBUG(vk, "Requested surface transform: %s",
                     vk_surface_transform(info->preTransform));
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

    PL_DEBUG(vk, "Requested image size: %dx%d (min %dx%d < cur %dx%d < max %dx%d)",
             w, h, caps.minImageExtent.width, caps.minImageExtent.height,
             caps.currentExtent.width, caps.currentExtent.height,
             caps.maxImageExtent.width, caps.maxImageExtent.height);

    // Default the requested size based on the reported extent
    if (caps.currentExtent.width != 0xFFFFFFFF)
        w = PL_DEF(w, caps.currentExtent.width);
    if (caps.currentExtent.height != 0xFFFFFFFF)
        h = PL_DEF(h, caps.currentExtent.height);

    // Otherwise, re-use the existing size if available
    w = PL_DEF(w, info->imageExtent.width);
    h = PL_DEF(h, info->imageExtent.height);

    if (!w || !h) {
        PL_ERR(vk, "Failed resizing swapchain: unknown size?");
        goto error;
    }

    // Clamp the extent based on the supported limits
    w = PL_CLAMP(w, caps.minImageExtent.width,  caps.maxImageExtent.width);
    h = PL_CLAMP(h, caps.minImageExtent.height, caps.maxImageExtent.height);
    info->imageExtent = (VkExtent2D) { w, h };

    // We just request whatever makes sense, and let the pl_vk decide what
    // pl_tex_params that translates to. That said, we still need to intersect
    // the swapchain usage flags with the format usage flags
    VkImageUsageFlags req_flags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkImageUsageFlags opt_flags = VK_IMAGE_USAGE_STORAGE_BIT;

    info->imageUsage = caps.supportedUsageFlags & (req_flags | opt_flags);
    VkFormatProperties fmtprop = {0};
    vk->GetPhysicalDeviceFormatProperties(vk->physd, info->imageFormat, &fmtprop);

#define CHECK(usage, feature) \
    if (!((fmtprop.optimalTilingFeatures & VK_FORMAT_FEATURE_##feature##_BIT))) \
        info->imageUsage &= ~VK_IMAGE_USAGE_##usage##_BIT

    CHECK(COLOR_ATTACHMENT, COLOR_ATTACHMENT);
    CHECK(TRANSFER_DST, TRANSFER_DST);
    CHECK(STORAGE, STORAGE_IMAGE);

    if ((info->imageUsage & req_flags) != req_flags) {
        PL_ERR(vk, "The swapchain doesn't support rendering and blitting!");
        goto error;
    }

    return true;

error:
    return false;
}

static bool vk_sw_recreate(pl_swapchain sw, int w, int h)
{
    pl_gpu gpu = sw->gpu;
    struct priv *p = PL_PRIV(sw);
    struct vk_ctx *vk = p->vk;
    struct vk_swapchain *current = p->current;

    VkImage *vkimages = NULL;
    uint32_t num_images = 0;
    char name[32];

    if (!update_swapchain_info(p, &p->protoInfo, w, h))
        return false;

    VkSwapchainCreateInfoKHR sinfo = p->protoInfo;

    VkSwapchainPresentModesCreateInfoKHR pminfo = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_KHR,
        .presentModeCount = 1,
        .pPresentModes = &p->protoInfo.presentMode,
    };
    if (p->has_swapchain_maintenance1)
        vk_link_struct(&sinfo, &pminfo);

#ifdef VK_EXT_full_screen_exclusive
    // Explicitly disallow full screen exclusive mode if possible
    static const VkSurfaceFullScreenExclusiveInfoEXT fsinfo = {
        .sType = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT,
        .fullScreenExclusive = VK_FULL_SCREEN_EXCLUSIVE_DISALLOWED_EXT,
    };
    if (vk->AcquireFullScreenExclusiveModeEXT)
        vk_link_struct(&sinfo, &fsinfo);
#endif

    p->suboptimal = false;
    p->needs_recreate = false;
    p->cur_width = sinfo.imageExtent.width;
    p->cur_height = sinfo.imageExtent.height;

    PL_DEBUG(sw, "(Re)creating swapchain of size %dx%d",
             sinfo.imageExtent.width,
             sinfo.imageExtent.height);

    // Calling `vkCreateSwapchainKHR` puts sinfo.oldSwapchain into a retired
    // state whether the call succeeds or not, so we always need to garbage
    // collect it afterwards - asynchronously as it may still be in use
    if (current) {
        PL_ARRAY_APPEND(sw, p->retired, current);
        sinfo.oldSwapchain = current->swapchain;
    } else {
        sinfo.oldSwapchain = VK_NULL_HANDLE;
    }
    p->current = current = pl_zalloc_ptr(NULL, p->current);
    current->last_imgidx = -1;
    VkResult res = vk->CreateSwapchainKHR(vk->dev, &sinfo, PL_VK_ALLOC, &current->swapchain);
    PL_VK_ASSERT(res, "vk->CreateSwapchainKHR(...)");

    // Get the new swapchain images
    VK(vk->GetSwapchainImagesKHR(vk->dev, current->swapchain, &num_images, NULL));
    vkimages = pl_calloc_ptr(NULL, num_images, vkimages);
    VK(vk->GetSwapchainImagesKHR(vk->dev, current->swapchain, &num_images, vkimages));

    static const VkSemaphoreCreateInfo seminfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };

    static const VkFenceCreateInfo fenceinfo = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };

    pl_assert(num_images > 0);

    PL_ARRAY_CLEAR(current, current->images, num_images);
    for (int i = 0; i < current->images.num; i++) {
        snprintf(name, sizeof(name), "swapchain #%d", i);
        current->images.elem[i] = pl_vulkan_wrap(gpu, pl_vulkan_wrap_params(
            .image = vkimages[i],
            .width = sinfo.imageExtent.width,
            .height = sinfo.imageExtent.height,
            .format = sinfo.imageFormat,
            .usage = sinfo.imageUsage,
            .debug_tag = name,
        ));
        if (!current->images.elem[i])
            goto error;
    }

    // Without swapchain_maintenance1, we cannot use a fence to know when an
    // acquisition semaphore is safe to reuse. We allocate an extra "spare"
    // to ensure we always have one available for vkAcquireNextImageKHR while
    // the others are potentially still in flight.
    PL_ARRAY_CLEAR(current, current->sems_in, num_images + !p->has_swapchain_maintenance1);
    for (int i = 0; i < current->sems_in.num; i++) {
        VK(vk->CreateSemaphore(vk->dev, &seminfo, PL_VK_ALLOC, &current->sems_in.elem[i]));
        snprintf(name, sizeof(name), "swapchain in #%d", i);
        PL_VK_NAME(SEMAPHORE, current->sems_in.elem[i], name);
    }

    PL_ARRAY_CLEAR(current, current->sems_out, num_images);
    for (int i = 0; i < current->sems_out.num; i++) {
        VK(vk->CreateSemaphore(vk->dev, &seminfo, PL_VK_ALLOC, &current->sems_out.elem[i]));
        snprintf(name, sizeof(name), "swapchain out #%d", i);
        PL_VK_NAME(SEMAPHORE, current->sems_out.elem[i], name);
    }

    for (int i = 0; i < num_images && p->has_swapchain_maintenance1; i++) {
        VkFence fence;
        VK(vk->CreateFence(vk->dev, &fenceinfo, PL_VK_ALLOC, &fence));
        snprintf(name, sizeof(name), "present fence #%d", i);
        PL_VK_NAME(FENCE, fence, name);
        PL_ARRAY_APPEND(current, current->fences_out, fence);
    }

    int bits = 0;

    // The channel with the most bits is probably the most authoritative about
    // the actual color information (consider e.g. a2bgr10). Slight downside
    // in that it results in rounding r/b for e.g. rgb565, but we don't pick
    // surfaces with fewer than 8 bits anyway, so let's not care for now.
    pl_fmt fmt = current->images.elem[0]->params.format;
    for (int i = 0; i < fmt->num_components; i++)
        bits = PL_MAX(bits, fmt->component_depth[i]);

    p->color_repr.bits.sample_depth = bits;
    p->color_repr.bits.color_depth = bits;

    // Note: `p->color_space.hdr` is (re-)applied by `set_hdr_metadata`
    map_color_space(sinfo.imageColorSpace, &p->color_space);

    // To convert to XYZ color system for VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT
    if (p->color_space.transfer == PL_COLOR_TRC_ST428) {
        p->color_repr.sys = PL_COLOR_SYSTEM_XYZ;
    } else {
        p->color_repr.sys = PL_COLOR_SYSTEM_RGB;
    }

    // Forcibly re-apply HDR metadata, bypassing the no-op check
    struct pl_hdr_metadata metadata = p->hdr_metadata;
    p->hdr_metadata = pl_hdr_metadata_empty;
    set_hdr_metadata(p, &metadata);

    pl_free(vkimages);
    return true;

error:
    PL_ERR(vk, "Failed (re)creating swapchain!");
    pl_free(vkimages);
    swapchain_destroy(sw, &p->current, UINT64_MAX);
    p->cur_width = p->cur_height = 0;
    return false;
}

static bool vk_sw_start_frame(pl_swapchain sw,
                              struct pl_swapchain_frame *out_frame)
{
    struct priv *p = PL_PRIV(sw);
    struct vk_ctx *vk = p->vk;
    pl_mutex_lock(&p->lock);

    bool recreate = !p->current || p->needs_recreate;
    if (p->suboptimal && !p->params.allow_suboptimal)
        recreate = true;

    if (recreate && !vk_sw_recreate(sw, 0, 0)) {
        pl_mutex_unlock(&p->lock);
        return false;
    }

    VkSemaphore sem_in = p->current->sems_in.elem[p->current->idx_sems_in];
    p->current->idx_sems_in = (p->current->idx_sems_in + 1) % p->current->sems_in.num;
    PL_TRACE(vk, "vkAcquireNextImageKHR signals 0x%"PRIx64, (uint64_t) sem_in);

    for (int attempts = 0; attempts < 2; attempts++) {
        uint32_t imgidx = 0;
        VkResult res = vk->AcquireNextImageKHR(vk->dev, p->current->swapchain, UINT64_MAX,
                                               sem_in, VK_NULL_HANDLE, &imgidx);

        switch (res) {
        case VK_SUBOPTIMAL_KHR:
            p->suboptimal = true;
            // fall through
        case VK_SUCCESS:
            p->current->last_imgidx = imgidx;
            if (p->current->fences_out.num > 0) {
                VkFence *pfence = &p->current->fences_out.elem[imgidx];
                vk->WaitForFences(vk->dev, 1, pfence, VK_TRUE, UINT64_MAX);
                vk->ResetFences(vk->dev, 1, pfence);
            }
            pl_vulkan_release_ex(sw->gpu, pl_vulkan_release_params(
                .tex        = p->current->images.elem[imgidx],
                .layout     = VK_IMAGE_LAYOUT_UNDEFINED,
                .qf         = VK_QUEUE_FAMILY_IGNORED,
                .semaphore  = { sem_in },
            ));
            *out_frame = (struct pl_swapchain_frame) {
                .fbo = p->current->images.elem[imgidx],
                .flipped = false,
                .color_repr = p->color_repr,
                .color_space = p->color_space,
            };
            // keep lock held
            return true;

        case VK_ERROR_OUT_OF_DATE_KHR: {
            // In these cases try recreating the swapchain
            if (!vk_sw_recreate(sw, 0, 0)) {
                pl_mutex_unlock(&p->lock);
                return false;
            }
            continue;
        }

        default:
            PL_ERR(vk, "Failed acquiring swapchain image: %s", vk_res_str(res));
            pl_mutex_unlock(&p->lock);
            return false;
        }
    }

    // If we've exhausted the number of attempts to recreate the swapchain,
    // just give up silently and let the user retry some time later.
    pl_mutex_unlock(&p->lock);
    return false;
}

static void present_cb(struct priv *p, void *arg)
{
    (void) pl_rc_deref(&p->frames_in_flight);
}

VK_CB_FUNC_DEF(present_cb);

static bool vk_sw_submit_frame(pl_swapchain sw)
{
    pl_gpu gpu = sw->gpu;
    struct priv *p = PL_PRIV(sw);
    struct vk_ctx *vk = p->vk;
    struct vk_swapchain *current = p->current;
    pl_assert(current);
    pl_assert(current->last_imgidx >= 0);
    uint32_t idx = current->last_imgidx;
    VkSemaphore sem_out = current->sems_out.elem[idx];
    current->last_imgidx = -1;

    bool held = pl_vulkan_hold_ex(gpu, pl_vulkan_hold_params(
        .tex        = current->images.elem[idx],
        .layout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        .qf         = VK_QUEUE_FAMILY_IGNORED,
        .semaphore  = { sem_out },
    ));

    if (!held) {
        PL_ERR(gpu, "Failed holding swapchain image for presentation");
        pl_mutex_unlock(&p->lock);
        return false;
    }

    struct vk_cmd *cmd = pl_vk_steal_cmd(gpu);
    if (!cmd) {
        pl_mutex_unlock(&p->lock);
        return false;
    }

    pl_rc_ref(&p->frames_in_flight);
    vk_cmd_callback(cmd, VK_CB_FUNC(present_cb), p, NULL);
    if (!vk_cmd_submit(&cmd)) {
        pl_mutex_unlock(&p->lock);
        return false;
    }

    struct vk_cmdpool *pool = vk->pool_graphics;
    int qidx = pool->idx_queues;
    VkQueue queue = pool->queues[qidx];

    vk_rotate_queues(p->vk);
    vk_malloc_garbage_collect(vk->ma);
    cleanup_retired_swapchains(sw, 0);

    VkSwapchainPresentFenceInfoKHR fenceInfo = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR,
        .swapchainCount = 1,
    };

    VkPresentInfoKHR pinfo = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &sem_out,
        .swapchainCount = 1,
        .pSwapchains = &current->swapchain,
        .pImageIndices = &idx,
    };

    if (current->fences_out.num > 0) {
        VkFence *pfence = &current->fences_out.elem[idx];
        fenceInfo.pFences = pfence;
        pinfo.pNext = &fenceInfo;
    }

    PL_TRACE(vk, "vkQueuePresentKHR waits on 0x%"PRIx64, (uint64_t) sem_out);
    vk->lock_queue(vk->queue_ctx, pool->qf, qidx);
    VkResult res = vk->QueuePresentKHR(queue, &pinfo);
    vk->unlock_queue(vk->queue_ctx, pool->qf, qidx);
    pl_mutex_unlock(&p->lock);

    switch (res) {
    case VK_SUBOPTIMAL_KHR:
        p->suboptimal = true;
        // fall through
    case VK_SUCCESS:
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

static void vk_sw_swap_buffers(pl_swapchain sw)
{
    struct priv *p = PL_PRIV(sw);

    pl_mutex_lock(&p->lock);
    while (pl_rc_count(&p->frames_in_flight) >= p->swapchain_depth) {
        pl_mutex_unlock(&p->lock); // don't hold mutex while blocking
        vk_poll_commands(p->vk, UINT64_MAX);
        pl_mutex_lock(&p->lock);
    }
    pl_mutex_unlock(&p->lock);
}

static bool vk_sw_resize(pl_swapchain sw, int *width, int *height)
{
    struct priv *p = PL_PRIV(sw);
    bool ok = true;

    pl_mutex_lock(&p->lock);

    bool width_changed = *width && *width != p->cur_width,
         height_changed = *height && *height != p->cur_height;

    if (p->suboptimal || p->needs_recreate || width_changed || height_changed)
        ok = vk_sw_recreate(sw, *width, *height);

    *width = p->cur_width;
    *height = p->cur_height;

    pl_mutex_unlock(&p->lock);
    return ok;
}

static void vk_sw_colorspace_hint(pl_swapchain sw, const struct pl_color_space *csp)
{
    struct priv *p = PL_PRIV(sw);
    pl_mutex_lock(&p->lock);

    // This should never fail if the swapchain already exists
    bool ok = pick_surf_format(sw, csp);
    set_hdr_metadata(p, &csp->hdr);
    pl_assert(ok);

    pl_mutex_unlock(&p->lock);
}

bool pl_vulkan_swapchain_suboptimal(pl_swapchain sw)
{
    struct priv *p = PL_PRIV(sw);
    return p->suboptimal;
}

static const struct pl_sw_fns vulkan_swapchain = {
    .destroy            = vk_sw_destroy,
    .latency            = vk_sw_latency,
    .resize             = vk_sw_resize,
    .colorspace_hint    = vk_sw_colorspace_hint,
    .start_frame        = vk_sw_start_frame,
    .submit_frame       = vk_sw_submit_frame,
    .swap_buffers       = vk_sw_swap_buffers,
};
