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

#pragma once

#include "common.h"

struct gl_format {
    GLint ifmt;             // sized internal format (e.g. GL_RGBA16F)
    GLenum fmt;             // base internal format (e.g. GL_RGBA)
    GLenum type;            // host-visible type (e.g. GL_FLOAT)
    struct pl_fmt_t tmpl;   // pl_fmt template
};

typedef void (gl_format_cb)(pl_gpu gpu, const struct gl_format *glfmt);

// Add all supported formats to the `pl_gpu` format list.
bool gl_setup_formats(struct pl_gpu_t *gpu);
