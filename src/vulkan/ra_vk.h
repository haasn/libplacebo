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

#include "ra.h"

#include "common.h"
#include "utils.h"

const struct ra *ra_create_vk(struct vk_ctx *vk);

// May be called on a struct ra of any type. Returns NULL if the ra is not
// a vulkan ra.
struct vk_ctx *ra_vk_get(const struct ra *ra);

// Allocates a ra_tex that wraps a swapchain image. The contents of the image
// will be invalidated, and access to it will only be internally synchronized.
// So the calling could should not do anything else with the VkImage.
const struct ra_tex *ra_vk_wrap_swapchain_img(const struct ra *ra, VkImage vkimg,
                                              VkSwapchainCreateInfoKHR info);

// Associates an external semaphore (dependency) with a ra_tex, such that this
// ra_tex will not be used by the ra_vk until the external semaphore fires.
void ra_tex_vk_external_dep(const struct ra *ra, const struct ra_tex *tex,
                            VkSemaphore external_dep);

// This function finalizes rendering, transitions `tex` (which must be a
// wrapped swapchain image) into a format suitable for presentation, and returns
// the resulting command buffer (or NULL on error). The caller may add their
// own semaphores to this command buffer, and must submit it afterwards.
struct vk_cmd *ra_vk_finish_frame(const struct ra *ra, const struct ra_tex *tex);
