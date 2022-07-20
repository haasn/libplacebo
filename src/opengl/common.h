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

#include <epoxy/gl.h>

#ifdef EPOXY_HAS_EGL
#include <epoxy/egl.h>
#endif

// Transitional: suppress duplicate import check
#undef __gl_h_
#undef __glext_h_
#define GLAD_GL
#include <glad/gl.h>

// PL_PRIV(pl_opengl)
struct gl_ctx {
    pl_log log;
    struct pl_opengl_params params;
    bool is_debug;
    bool is_debug_egl;

    // For context locking
    pl_mutex lock;
    int count;
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
