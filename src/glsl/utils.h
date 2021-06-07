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

#include "config.h"
#include "include/libplacebo/gpu.h"

struct pl_spirv_version {
    bool vulkan;
    uint32_t env_version;
    uint32_t spv_version;
};

struct pl_spirv_version pl_glsl_spv_version(const struct pl_glsl_version *glsl);

enum glsl_shader_stage {
    GLSL_SHADER_VERTEX = 0,
    GLSL_SHADER_FRAGMENT,
    GLSL_SHADER_COMPUTE,
};
