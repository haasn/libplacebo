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
#include "utils.h"

struct spirv_compiler {
    const struct spirv_compiler_impl *impl;
    pl_log log;

    // For cache invalidation, should uniquely identify everything about this
    // spirv compiler and its configuration.
    uint64_t signature;
};

// Initialize a SPIR-V compiler instance, or returns NULL on failure.
struct spirv_compiler *spirv_compiler_create(pl_log log,
                                             const struct pl_spirv_version *spirv_ver);
void spirv_compiler_destroy(struct spirv_compiler **spirv);

// Compile GLSL to SPIR-V. Returns {0} on failure.
pl_str spirv_compile_glsl(struct spirv_compiler *spirv, void *alloc,
                          const struct pl_glsl_version *glsl,
                          enum glsl_shader_stage stage,
                          const char *shader);

struct spirv_compiler_impl {
    const char *name;
    void (*destroy)(struct spirv_compiler *spirv);
    __typeof__(spirv_compiler_create) *create;
    __typeof__(spirv_compile_glsl) *compile;
};
