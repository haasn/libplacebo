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
#include "../log.h"
#include "../gpu.h"
#include "pl_thread.h"

#define GLAD_GL
#define GLAD_GLES2
#include <glad/gl.h>
#include <glad/egl.h>

typedef GladGLContext gl_funcs;

// PL_PRIV(pl_opengl)
struct gl_ctx {
    pl_log log;
    struct pl_opengl_params params;
    bool is_debug;
    bool is_debug_egl;
    bool is_gles;

    // For context locking
    pl_mutex lock;
    int count;

    // Dispatch table
    gl_funcs func;
};

struct gl_cb {
    void (*callback)(void *priv);
    void *priv;
    GLsync sync;
};

struct fbo_format {
    pl_fmt fmt;
    const struct gl_format *glfmt;
};

// For locking/unlocking
bool gl_make_current(pl_opengl gl);
void gl_release_current(pl_opengl gl);
