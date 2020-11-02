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

#pragma once

#include "../gpu.h"
#include "common.h"

const struct pl_gpu *pl_gpu_create_gl(struct pl_context *ctx,
                                      const struct pl_opengl_params *params);

const struct pl_tex *pl_opengl_wrap_fb(const struct pl_gpu *gpu, GLuint fbo,
                                       int w, int h);
