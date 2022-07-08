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

#include "spirv.h"
#include "glsl/glslang.h"

// This header contains only preprocessor definitions
#include <glslang/build_info.h>

const struct spirv_compiler_impl pl_spirv_glslang;

static void glslang_destroy(struct spirv_compiler *spirv)
{
    pl_glslang_uninit();
    pl_free(spirv);
}

static struct spirv_compiler *glslang_create(pl_log log)
{
    if (!pl_glslang_init()) {
        pl_fatal(log, "Failed initializing glslang SPIR-V compiler!");
        return NULL;
    }

    struct spirv_compiler *spirv = pl_alloc_ptr(NULL, spirv);
    *spirv = (struct spirv_compiler) {
        .signature = pl_str0_hash(pl_spirv_glslang.name),
        .impl = &pl_spirv_glslang,
        .log = log,
    };

    pl_info(log, "glslang version: %d.%d.%d",
            GLSLANG_VERSION_MAJOR,
            GLSLANG_VERSION_MINOR,
            GLSLANG_VERSION_PATCH);

    pl_hash_merge(&spirv->signature, (GLSLANG_VERSION_MAJOR & 0xFF) << 24 |
                                     (GLSLANG_VERSION_MINOR & 0xFF) << 16 |
                                     (GLSLANG_VERSION_PATCH & 0xFFFF));
    return spirv;
}

static pl_str glslang_compile(struct spirv_compiler *spirv, void *alloc,
                              const struct pl_glsl_version *glsl,
                              enum glsl_shader_stage stage,
                              const char *shader)
{
    struct pl_glslang_res *res = pl_glslang_compile(glsl, stage, shader);
    if (!res || !res->success) {
        PL_ERR(spirv, "glslang failed: %s", res ? res->error_msg : "(null)");
        pl_free(res);
        return (struct pl_str) {0};
    }

    struct pl_str ret = {
        .buf = pl_steal(alloc, res->data),
        .len = res->size,
    };

    pl_free(res);
    return ret;
}

const struct spirv_compiler_impl pl_spirv_glslang = {
    .name       = "glslang",
    .destroy    = glslang_destroy,
    .create     = glslang_create,
    .compile    = glslang_compile,
};
