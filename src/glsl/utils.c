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

#include "common.h"
#include "utils.h"

static const struct pl_spirv_version spv_ver_vulkan_1_0 = {
    .vulkan = true,
    .env_version = 1 << 22,
    .spv_version = 1 << 16,
};

static const struct pl_spirv_version spv_ver_vulkan_1_1 = {
    .vulkan = true,
    .env_version = 1 << 22 | 1 << 12,
    .spv_version = 1 << 16 | 3 << 8,
};

struct pl_spirv_version pl_glsl_spv_version(const struct pl_glsl_version *glsl)
{
    // We don't currently use SPIR-V for OpenGL
    pl_assert(glsl->vulkan);

    if (glsl->subgroup_size)
        return spv_ver_vulkan_1_1;
    return spv_ver_vulkan_1_0;
}
