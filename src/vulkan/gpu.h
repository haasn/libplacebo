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

#include "../gpu.h"

#include "common.h"
#include "utils.h"

const struct pl_gpu *pl_gpu_create_vk(struct vk_ctx *vk);

// May be called on a struct ra of any type. Returns NULL if the ra is not
// a vulkan ra.
struct vk_ctx *pl_vk_get(const struct pl_gpu *gpu);

// Associates an external semaphore (dependency) with a pl_tex, such that this
// pl_tex will not be used by the pl_vk until the external semaphore fires.
void pl_tex_vk_external_dep(const struct pl_gpu *gpu, const struct pl_tex *tex,
                            VkSemaphore external_dep);

// This function takes the current graphics command and steals it from the
// GPU, so the caller can do custom vk_cmd_ calls on it. The caller should
// submit it as well.
struct vk_cmd *pl_vk_steal_cmd(const struct pl_gpu *gpu);
