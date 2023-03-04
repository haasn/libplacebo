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

const struct spirv_compiler_impl pl_spirv_glslang;

struct priv {
    struct pl_spirv_version spirv_ver;
};

static void glslang_destroy(struct spirv_compiler *spirv)
{
    pl_glslang_uninit();
    pl_free(spirv);
}

static struct spirv_compiler *glslang_create(pl_log log,
                                             const struct pl_spirv_version *spirv_ver)
{
    if (!pl_glslang_init()) {
        pl_fatal(log, "Failed initializing glslang SPIR-V compiler!");
        return NULL;
    }

    struct spirv_compiler *spirv = pl_alloc_obj(NULL, spirv, struct priv);
    *spirv = (struct spirv_compiler) {
        .signature = pl_str0_hash(pl_spirv_glslang.name),
        .impl = &pl_spirv_glslang,
        .log = log,
    };

    struct priv *p = PL_PRIV(spirv);
    p->spirv_ver = *spirv_ver;

    pl_info(log, "glslang version: %d.%d.%d",
            GLSLANG_VERSION_MAJOR,
            GLSLANG_VERSION_MINOR,
            GLSLANG_VERSION_PATCH);

    // Clamp to supported version by glslang
    if (GLSLANG_SPV_MAX < p->spirv_ver.spv_version) {
        p->spirv_ver.spv_version = GLSLANG_SPV_MAX;
        p->spirv_ver.env_version = pl_spirv_version_to_vulkan(GLSLANG_SPV_MAX);
    }

    pl_hash_merge(&spirv->signature, (uint64_t) p->spirv_ver.spv_version << 32 |
                                                p->spirv_ver.env_version);
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
    const struct priv *p = PL_PRIV(spirv);

    struct pl_glslang_res *res = pl_glslang_compile(glsl, &p->spirv_ver, stage, shader);
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
