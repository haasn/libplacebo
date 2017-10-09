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
#include "context.h"

enum glsl_shader_stage {
    GLSL_SHADER_VERTEX,
    GLSL_SHADER_FRAGMENT,
    GLSL_SHADER_COMPUTE,
};

#define SPIRV_NAME_MAX_LEN 32

struct spirv_compiler {
    char name[SPIRV_NAME_MAX_LEN]; // for cache invalidation
    struct pl_context *ctx;
    const struct spirv_compiler_fns *impl;

    // implementation-specific fields
    void *priv;
    struct ra_glsl_desc glsl;      // supported GLSL capabilities
    int compiler_version;          // for cache invalidation, may be left as 0
};

struct spirv_compiler_fns {
    const char *name;

    // Compile GLSL to SPIR-V, under GL_KHR_vulkan_glsl semantics.
    bool (*compile_glsl)(struct spirv_compiler *spirv, void *tactx,
                         enum glsl_shader_stage type, const char *glsl,
                         struct bstr *out_spirv);

    // Only needs to initialize the implementation-specific fields
    bool (*init)(struct spirv_compiler *spirv);
    void (*uninit)(struct spirv_compiler *spirv);
};

// Initialize a SPIR-V compiler instance, or returns NULL on failure.
struct spirv_compiler *spirv_compiler_create(struct pl_context *ctx);
void spirv_compiler_destroy(struct spirv_compiler **spirv);
