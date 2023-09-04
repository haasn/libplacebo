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

#include "log.h"
#include "utils.h"

typedef const struct pl_spirv_t {
    const struct spirv_compiler *impl;
    pl_log log;

    // SPIR-V version specified at creation time.
    struct pl_spirv_version version;

    // For cache invalidation, should uniquely identify everything about this
    // spirv compiler and its configuration.
    uint64_t signature;
} *pl_spirv;

// Initialize a SPIR-V compiler instance, or returns NULL on failure.
pl_spirv pl_spirv_create(pl_log log, struct pl_spirv_version spirv_ver);
void pl_spirv_destroy(pl_spirv *spirv);

// Compile GLSL to SPIR-V. Returns {0} on failure.
pl_str pl_spirv_compile_glsl(pl_spirv spirv, void *alloc,
                             struct pl_glsl_version glsl_ver,
                             enum glsl_shader_stage stage,
                             const char *shader);

struct spirv_compiler {
    const char *name;
    void (*destroy)(pl_spirv spirv);
    __typeof__(pl_spirv_create) *create;
    __typeof__(pl_spirv_compile_glsl) *compile;
};
