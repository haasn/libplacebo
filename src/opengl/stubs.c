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

#include "../common.h"
#include "log.h"

const struct pl_opengl_params pl_opengl_default_params = {0};

pl_opengl pl_opengl_create(pl_log log, const struct pl_opengl_params *params)
{
    pl_fatal(log, "libplacebo compiled without OpenGL support!");
    return NULL;
}

void pl_opengl_destroy(pl_opengl *pgl)
{
    pl_opengl gl = *pgl;
    pl_assert(!gl);
}

pl_opengl pl_opengl_get(pl_gpu gpu)
{
    return NULL;
}

pl_swapchain pl_opengl_create_swapchain(pl_opengl gl,
                            const struct pl_opengl_swapchain_params *params)
{
    pl_unreachable();
}

void pl_opengl_swapchain_update_fb(pl_swapchain sw,
                                   const struct pl_opengl_framebuffer *fb)
{
    pl_unreachable();
}

pl_tex pl_opengl_wrap(pl_gpu gpu, const struct pl_opengl_wrap_params *params)
{
    pl_unreachable();
}

unsigned int pl_opengl_unwrap(pl_gpu gpu, pl_tex tex, unsigned int *out_target,
                              int *out_iformat, unsigned int *out_fbo)
{
    pl_unreachable();
}
