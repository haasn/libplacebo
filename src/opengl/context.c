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
 * License along with libplacebo.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "common.h"
#include "gpu.h"

const struct pl_opengl_params pl_opengl_default_params = {0};

void pl_opengl_destroy(const struct pl_opengl **pl_gl)
{
    if (!*pl_gl)
        return;

    pl_gpu_destroy((*pl_gl)->gpu);
    TA_FREEP((void **) pl_gl);
}

const struct pl_opengl *pl_opengl_create(struct pl_context *ctx,
                                         const struct pl_opengl_params *params)
{
    params = PL_DEF(params, &pl_opengl_default_params);
    struct pl_opengl *pl_gl = talloc_zero(NULL, struct pl_opengl);

    int ver = epoxy_gl_version();
    if (!ver) {
        pl_fatal(ctx, "No OpenGL version detected - make sure an OpenGL context "
                 "is bound to the current thread!");
        goto error;
    }

    pl_info(ctx, "Detected OpenGL version strings:");
    pl_info(ctx, "    GL_VERSION:  %s", glGetString(GL_VERSION));
    pl_info(ctx, "    GL_VENDOR:   %s", glGetString(GL_VENDOR));
    pl_info(ctx, "    GL_RENDERER: %s", glGetString(GL_RENDERER));

    pl_gl->gpu = pl_gpu_create_gl(ctx);
    if (!pl_gl->gpu)
        goto error;

    return pl_gl;

error:
    pl_fatal(ctx, "Failed initializing opengl context");
    pl_opengl_destroy((const struct pl_opengl **) &pl_gl);
    return NULL;
}
