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

#include "../common.h"
#include "../context.h"
#include "../gpu.h"

#include <epoxy/gl.h>

#ifdef EPOXY_HAS_EGL
#include <epoxy/egl.h>
#endif

#ifdef __unix__
#define GL_HAVE_UNIX 1
#endif

// For gpu.priv
struct pl_gl {
    struct pl_gpu_fns impl;
    bool failed;

#ifdef EPOXY_HAS_EGL
    // For import/export
    EGLDisplay egl_dpy;
    EGLContext egl_ctx;
#endif

    // Incrementing counters to keep track of object uniqueness
    int buf_id;

    // Cached capabilities
    int gl_ver;
    int gles_ver;
    bool has_stride;
    bool has_invalidate_fb;
    bool has_invalidate_tex;
    bool has_vao;
    bool has_queries;
    bool has_modifiers;
};
