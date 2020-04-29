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

extern const struct spirv_compiler_fns pl_spirv_shaderc;
extern const struct spirv_compiler_fns pl_spirv_glslang;

static const struct spirv_compiler_fns *compilers[] = {
#if PL_HAVE_SHADERC
    &pl_spirv_shaderc,
#endif
#if PL_HAVE_GLSLANG
    &pl_spirv_glslang,
#endif
};

struct spirv_compiler *spirv_compiler_create(struct pl_context *ctx,
                                             uint32_t api_version)
{
    for (int i = 0; i < PL_ARRAY_SIZE(compilers); i++) {
        const struct spirv_compiler_fns *impl = compilers[i];
        pl_info(ctx, "Initializing SPIR-V compiler '%s'", impl->name);
        struct spirv_compiler *spirv = impl->create(ctx, api_version);
        if (!spirv)
            continue;

        spirv->ctx = ctx;
        spirv->impl = impl;
        strncpy(spirv->name, impl->name, sizeof(spirv->name) - 1);
        return spirv;
    }

    pl_fatal(ctx, "Failed initializing any SPIR-V compiler! Maybe libplacebo "
             "was built without support for either libshaderc or glslang?");
    return NULL;
}

void spirv_compiler_destroy(struct spirv_compiler **spirv)
{
    if (!*spirv)
        return;

    (*spirv)->impl->destroy(*spirv);
}
