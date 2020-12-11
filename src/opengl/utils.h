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
void gl_poll_callbacks(const struct pl_gpu *gpu);

// Return a human-readable name for various OpenGL errors
const char *gl_err_str(GLenum err);

// Check for errors and log them + return false if detected
bool gl_check_err(const struct pl_gpu *gpu, const char *fun);

// Returns true if the context is a suspected software rasterizer
bool gl_is_software(void);

#ifdef EPOXY_HAS_EGL
const char *egl_err_str(EGLenum err);
bool egl_check_err(const struct pl_gpu *gpu, const char *fun);
#endif
