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

#include "common.h"
#include "log.h"

enum glsl_shader_stage {
    GLSL_SHADER_VERTEX,
    GLSL_SHADER_FRAGMENT,
    GLSL_SHADER_COMPUTE,
};

#define SPIRV_NAME_MAX_LEN 32

struct spirv_compiler {
    char name[SPIRV_NAME_MAX_LEN]; // for cache invalidation
    const struct spirv_compiler_fns *impl;
    pl_log log;

    // implementation-specific fields
    struct pl_glsl_version glsl;   // supported GLSL version
    int compiler_version;          // for cache invalidation, may be left as 0
};

struct spirv_compiler_fns {
    const char *name;

    // Compile GLSL to SPIR-V, under GL_KHR_vulkan_glsl semantics.
    bool (*compile_glsl)(struct spirv_compiler *spirv, void *alloc,
                         enum glsl_shader_stage type, const char *glsl,
                         pl_str *out_spirv);

    // Only needs to initialize the implementation-specific fields
    struct spirv_compiler *(*create)(pl_log log, uint32_t api_ver);
    void (*destroy)(struct spirv_compiler *spirv);
};

// Initialize a SPIR-V compiler instance, or returns NULL on failure.
// `api_version` is the Vulkan API version we're targetting.
struct spirv_compiler *spirv_compiler_create(pl_log log,
                                             uint32_t api_version);

void spirv_compiler_destroy(struct spirv_compiler **spirv);
