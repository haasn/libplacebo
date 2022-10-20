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

#include "common.h"

// Iterate through callbacks attached to the `pl_gl` and execute all of the
// ones that have completed.
//
// Thread-safety: Unsafe
void gl_poll_callbacks(pl_gpu gpu);

// Return a human-readable name for various OpenGL errors
//
// Thread-safety: Safe
const char *gl_err_str(GLenum err);

// Check for errors and log them + return false if detected
//
// Thread-safety: Unsafe
bool gl_check_err(pl_gpu gpu, const char *fun);

// Returns true if the context is a suspected software rasterizer
//
// Thread-safety: Unsafe
bool gl_is_software(pl_opengl gl);

// Returns true if the context is detected as OpenGL ES
//
// Thread-safety: Unsafe
bool gl_is_gles(pl_opengl gl);

// Check for presence of an extension, alternatively a minimum GL version
//
// Thread-safety: Unsafe
bool gl_test_ext(pl_gpu gpu, const char *ext, int gl_ver, int gles_ver);

// Thread-safety: Safe
const char *egl_err_str(EGLenum err);

// Thread-safety: Unsafe
bool egl_check_err(pl_gpu gpu, const char *fun);
