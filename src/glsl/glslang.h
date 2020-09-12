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

#include <stdlib.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

int pl_glslang_version(void);
bool pl_glslang_init(void);
void pl_glslang_uninit(void);

struct pl_glslang_res {
    // Compilation status
    bool success;
    const char *error_msg;

    // Compiled shader memory, or NULL
    void *data;
    size_t size;
};

enum pl_glslang_stage {
    PL_GLSLANG_VERTEX,
    PL_GLSLANG_FRAGMENT,
    PL_GLSLANG_COMPUTE,
};

// Compile GLSL into a SPIRV stream, if possible. The resulting
// pl_glslang_res can simply be freed with talloc_free() when done.
struct pl_glslang_res *pl_glslang_compile(const char *glsl, uint32_t api_ver,
                                          enum pl_glslang_stage stage);

#ifdef __cplusplus
}
#endif
