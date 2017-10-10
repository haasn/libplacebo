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

extern const struct spirv_compiler_fns spirv_shaderc;

static const struct spirv_compiler_fns *compilers[] = {
#if PL_HAVE_SHADERC
    &spirv_shaderc,
#endif
};

struct spirv_compiler *spirv_compiler_create(struct pl_context *ctx)
{
    for (int i = 0; i < PL_ARRAY_SIZE(compilers); i++) {
        const struct spirv_compiler_fns *impl = compilers[i];

        struct spirv_compiler *spirv = talloc_zero(NULL, struct spirv_compiler);
        spirv->ctx = ctx;
        spirv->impl = impl;
        strncpy(spirv->name, impl->name, sizeof(spirv->name));

        pl_info(ctx, "Initializing SPIR-V compiler '%s'", impl->name);
        if (impl->init(spirv))
            return spirv;
        talloc_free(spirv);
    }

    pl_fatal(ctx, "Failed initializing any SPIR-V compiler! Maybe "
             "libplacebo was built without support for libshaderc?");
    return NULL;
}

void spirv_compiler_destroy(struct spirv_compiler **spirv)
{
    if (!*spirv)
        return;

    (*spirv)->impl->uninit(*spirv);
    TA_FREEP(spirv);
}
