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

#pragma once

#include "common.h"
#include "gpu.h"

struct vk_format {
    VkFormat tfmt;      // internal vulkan format enum (textures)
    struct pl_fmt_t fmt;// pl_fmt template (features will be auto-detected)
    int icomps;         // internal component count (or 0 to infer from `fmt`)
    VkFormat bfmt;      // vulkan format for use as buffers (or 0 to use `tfmt`)
    const struct vk_format *emufmt; // alternate format for emulation
    struct { VkFormat fmt; int sx, sy; } pfmt[4]; // plane formats (for planar textures)
};

// Add all supported formats to the `pl_gpu` format list
void vk_setup_formats(struct pl_gpu_t *gpu);
