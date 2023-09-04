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

extern const struct spirv_compiler pl_spirv_shaderc;
extern const struct spirv_compiler pl_spirv_glslang;

static const struct spirv_compiler *compilers[] = {
#ifdef PL_HAVE_SHADERC
    &pl_spirv_shaderc,
#endif
#ifdef PL_HAVE_GLSLANG
    &pl_spirv_glslang,
#endif
};

pl_spirv pl_spirv_create(pl_log log, struct pl_spirv_version spirv_ver)
{
    for (int i = 0; i < PL_ARRAY_SIZE(compilers); i++) {
        pl_spirv spirv = compilers[i]->create(log, spirv_ver);
        if (!spirv)
            continue;

        pl_info(log, "Initialized SPIR-V compiler '%s'", compilers[i]->name);
        return spirv;
    }

    pl_fatal(log, "Failed initializing any SPIR-V compiler! Maybe libplacebo "
             "was built without support for either libshaderc or glslang?");
    return NULL;
}

void pl_spirv_destroy(pl_spirv *pspirv)
{
    pl_spirv spirv = *pspirv;
    if (!spirv)
        return;

    spirv->impl->destroy(spirv);
    *pspirv = NULL;
}

pl_str pl_spirv_compile_glsl(pl_spirv spirv, void *alloc,
                             struct pl_glsl_version glsl,
                             enum glsl_shader_stage stage,
                             const char *shader)
{
    return spirv->impl->compile(spirv, alloc, glsl, stage, shader);
}
