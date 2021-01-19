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

struct priv {
    uint32_t api_ver;
};

static void glslang_destroy(struct spirv_compiler *spirv)
{
    pl_glslang_uninit();
    pl_free(spirv);
}

static struct spirv_compiler *glslang_create(struct pl_context *ctx,
                                             uint32_t api_version)
{
    if (!pl_glslang_init()) {
        pl_fatal(ctx, "Failed initializing glslang SPIR-V compiler!");
        return NULL;
    }

    struct spirv_compiler *spirv;
    spirv = pl_zalloc_priv(NULL, struct spirv_compiler, struct priv);
    spirv->compiler_version = pl_glslang_version();
    spirv->glsl = (struct pl_glsl_desc) {
        .version = 450,
        .vulkan  = true,
    };

    struct priv *p = PL_PRIV(spirv);
    p->api_ver = api_version;

    return spirv;
}

static bool glslang_compile(struct spirv_compiler *spirv, void *alloc,
                            enum glsl_shader_stage type, const char *glsl,
                            pl_str *out_spirv)
{
    struct priv *p = PL_PRIV(spirv);

    static const enum pl_glslang_stage stages[] = {
        [GLSL_SHADER_VERTEX]   = PL_GLSLANG_VERTEX,
        [GLSL_SHADER_FRAGMENT] = PL_GLSLANG_FRAGMENT,
        [GLSL_SHADER_COMPUTE]  = PL_GLSLANG_COMPUTE,
    };

    struct pl_glslang_res *res = pl_glslang_compile(glsl, p->api_ver, stages[type]);
    if (!res || !res->success) {
        PL_ERR(spirv, "glslang failed: %s", res ? res->error_msg : "(null)");
        pl_free(res);
        return false;
    }

    out_spirv->buf = pl_steal(alloc, res->data);
    out_spirv->len = res->size;
    pl_free(res);
    return true;
}

const struct spirv_compiler_fns pl_spirv_glslang = {
    .name = "glslang",
    .compile_glsl = glslang_compile,
    .create = glslang_create,
    .destroy = glslang_destroy,
};
