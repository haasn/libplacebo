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

#include <stdbool.h>
#include <stdint.h>

#include <libplacebo/gpu.h>

#define PL_SPV_VERSION(major, minor) ((major) << 16 | (minor) << 8)
#define PL_VLK_VERSION(major, minor) ((major) << 22 | (minor) << 12)

// Max version that can be used
#define PL_MAX_SPIRV_VER PL_SPV_VERSION(1, 6)

struct pl_spirv_version {
    uint32_t env_version;
    uint32_t spv_version;
};

static inline uint32_t pl_spirv_version_to_vulkan(uint32_t spirv_ver)
{
    if (spirv_ver >= PL_SPV_VERSION(1, 6))
        return PL_VLK_VERSION(1, 3);
    if (spirv_ver >= PL_SPV_VERSION(1, 5))
        return PL_VLK_VERSION(1, 2);
    if (spirv_ver >= PL_SPV_VERSION(1, 3))
        return PL_VLK_VERSION(1, 1);
    return PL_VLK_VERSION(1, 0);
}

enum glsl_shader_stage {
    GLSL_SHADER_VERTEX = 0,
    GLSL_SHADER_FRAGMENT,
    GLSL_SHADER_COMPUTE,
};
