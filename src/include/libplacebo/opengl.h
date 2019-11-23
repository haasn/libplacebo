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

#ifndef LIBPLACEBO_OPENGL_H_
#define LIBPLACEBO_OPENGL_H_

#include "gpu.h"

struct pl_opengl {
    const struct pl_gpu *gpu;
    void *priv;
};

struct pl_opengl_params {
    bool debug;
};

// Default/recommended parameters
extern const struct pl_opengl_params pl_opengl_default_params;

// Creates a new OpenGL renderer based on the given parameters. This will
// internally use whatever platform-defined mechanism (WGL, X11, EGL) is
// appropriate for loading the OpenGL function calls, so the user doesn't need
// to pass in a `getProcAddress` callback. If `params` is left as NULL, it
// defaults to `&pl_opengl_default_params`.
const struct pl_opengl *pl_opengl_create(struct pl_context *ctx,
                                         const struct pl_opengl_params *params);

// All resources allocated from the `pl_gpu` contained by this `pl_opengl` must
// be explicitly destroyed by the user before calling `pl_opengl_destroy`.
void pl_opengl_destroy(const struct pl_opengl **gl);

struct pl_opengl_framebuffer {
    // ID of the framebuffer, or 0 to use the context's default framebuffer.
    int id;

    // If true, then the framebuffer is assumed to be "flipped" relative to
    // normal GL semantics, i.e. set this to `true` if the first pixel is the
    // top left corner.
    bool flipped;
};

struct pl_opengl_swapchain_params {
    // Set this to the platform-specific function to swap buffers, e.g.
    // glXSwapBuffers, eglSwapBuffers etc. This will be called internally by
    // `pl_swapchain_swap_buffers`. Required, unless you never call that
    // function.
    void (*swap_buffers)(void *priv);

    // Initial framebuffer description. This can be changed later on using
    // `pl_opengl_swapchain_update_fb`.
    struct pl_opengl_framebuffer framebuffer;

    // Attempt forcing a specific latency. If this is nonzero, then
    // `pl_swapchain_swap_buffers` will wait until fewer than N frames are "in
    // flight" before returning. Setting this to a high number generally
    // accomplished nothing, because the OpenGL driver typically limits the
    // number of buffers on its own. But setting it to a low number like 2 or
    // even 1 can reduce latency (at the cost of throughput).
    int max_swapchain_depth;

    // Arbitrary user pointer that gets passed to `swap_buffers` etc.
    void *priv;
};

// Creates an instance of `pl_swapchain` tied to the active context.
// Note: Due to OpenGL semantics, users *must* call `pl_swapchain_resize`
// before attempting to use this swapchain, otherwise calls to
// `pl_swapchain_start_frame` will fail.
const struct pl_swapchain *pl_opengl_create_swapchain(const struct pl_opengl *gl,
                            const struct pl_opengl_swapchain_params *params);

// Update the framebuffer description. After calling this function, users
// *must* call `pl_swapchain_resize` before attempting to use the swapchain
// again, otherwise calls to `pl_swapchain_start_frame` will fail.
void pl_opengl_swapchain_update_fb(const struct pl_swapchain *sw,
                                   const struct pl_opengl_framebuffer *fb);

#endif // LIBPLACEBO_OPENGL_H_
