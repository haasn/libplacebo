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

extern const struct spirv_compiler_impl pl_spirv_shaderc;
extern const struct spirv_compiler_impl pl_spirv_glslang;

static const struct spirv_compiler_impl *compilers[] = {
#ifdef PL_HAVE_SHADERC
    &pl_spirv_shaderc,
#endif
#ifdef PL_HAVE_GLSLANG
    &pl_spirv_glslang,
#endif
};

struct spirv_compiler *spirv_compiler_create(pl_log log,
                                             const struct pl_spirv_version *spirv_ver)
{
    for (int i = 0; i < PL_ARRAY_SIZE(compilers); i++) {
        struct spirv_compiler *spirv = compilers[i]->create(log, spirv_ver);
        if (!spirv)
            continue;

        pl_info(log, "Initialized SPIR-V compiler '%s'", compilers[i]->name);
        return spirv;
    }

    pl_fatal(log, "Failed initializing any SPIR-V compiler! Maybe libplacebo "
             "was built without support for either libshaderc or glslang?");
    return NULL;
}

void spirv_compiler_destroy(struct spirv_compiler **spirv)
{
    if (!*spirv)
        return;

    (*spirv)->impl->destroy(*spirv);
}

pl_str spirv_compile_glsl(struct spirv_compiler *spirv, void *alloc,
                          const struct pl_glsl_version *glsl,
                          enum glsl_shader_stage stage,
                          const char *shader)
{
    return spirv->impl->compile(spirv, alloc, glsl, stage, shader);
}
