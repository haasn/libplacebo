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
#include "utils.h"
#include "glsl/glslang.h"

// This header contains only preprocessor definitions
#include <glslang/build_info.h>

// This is awkward, but we cannot use upstream macro, it was fixed in 11.11.0
#define PL_GLSLANG_VERSION_GREATER_THAN(major, minor, patch) \
    ((GLSLANG_VERSION_MAJOR) > (major) || ((major) == GLSLANG_VERSION_MAJOR && \
    ((GLSLANG_VERSION_MINOR) > (minor) || ((minor) == GLSLANG_VERSION_MINOR && \
     (GLSLANG_VERSION_PATCH) > (patch)))))

#if PL_GLSLANG_VERSION_GREATER_THAN(11, 8, 0)
#define GLSLANG_SPV_MAX PL_SPV_VERSION(1, 6)
#elif PL_GLSLANG_VERSION_GREATER_THAN(7, 13, 3496)
#define GLSLANG_SPV_MAX PL_SPV_VERSION(1, 5)
#elif PL_GLSLANG_VERSION_GREATER_THAN(6, 2, 2596)
#define GLSLANG_SPV_MAX PL_SPV_VERSION(1, 3)
#else
#define GLSLANG_SPV_MAX PL_SPV_VERSION(1, 0)
#endif

const struct spirv_compiler pl_spirv_glslang;

static void glslang_destroy(pl_spirv spirv)
{
    pl_glslang_uninit();
    pl_free((void *) spirv);
}

static pl_spirv glslang_create(pl_log log, struct pl_spirv_version spirv_ver)
{
    if (!pl_glslang_init()) {
        pl_fatal(log, "Failed initializing glslang SPIR-V compiler!");
        return NULL;
    }

    struct pl_spirv_t *spirv = pl_alloc_ptr(NULL, spirv);
    *spirv = (struct pl_spirv_t) {
        .signature = pl_str0_hash(pl_spirv_glslang.name),
        .impl      = &pl_spirv_glslang,
        .version   = spirv_ver,
        .log       = log,
    };

    PL_INFO(spirv, "glslang version: %d.%d.%d",
            GLSLANG_VERSION_MAJOR,
            GLSLANG_VERSION_MINOR,
            GLSLANG_VERSION_PATCH);

    // Clamp to supported version by glslang
    if (GLSLANG_SPV_MAX < spirv->version.spv_version) {
        spirv->version.spv_version = GLSLANG_SPV_MAX;
        spirv->version.env_version = pl_spirv_version_to_vulkan(GLSLANG_SPV_MAX);
    }

    pl_hash_merge(&spirv->signature, (uint64_t) spirv->version.spv_version << 32 |
                                                spirv->version.env_version);
    pl_hash_merge(&spirv->signature, (GLSLANG_VERSION_MAJOR & 0xFF) << 24 |
                                     (GLSLANG_VERSION_MINOR & 0xFF) << 16 |
                                     (GLSLANG_VERSION_PATCH & 0xFFFF));
    return spirv;
}

static pl_str glslang_compile(pl_spirv spirv, void *alloc,
                              struct pl_glsl_version glsl_ver,
                              enum glsl_shader_stage stage,
                              const char *shader)
{
    struct pl_glslang_res *res;

    res = pl_glslang_compile(glsl_ver, spirv->version, stage, shader);
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

const struct spirv_compiler pl_spirv_glslang = {
    .name       = "glslang",
    .destroy    = glslang_destroy,
    .create     = glslang_create,
    .compile    = glslang_compile,
};
