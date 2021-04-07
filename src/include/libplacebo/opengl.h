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

#include <libplacebo/gpu.h>
#include <libplacebo/swapchain.h>

// Note on thread safety: The thread safety of `pl_opengl` and any associated
// GPU objects follows the same thread safety rules as the underlying OpenGL
// context. In other words, they must only be called from the thread the OpenGL
// context is current on.

struct pl_opengl {
    const struct pl_gpu *gpu;
};

struct pl_opengl_params {
    // Enable OpenGL debug report callbacks. May have little effect depending
    // on whether or not the GL context was initialized with appropriate
    // debugging enabled.
    bool debug;

    // Allow the use of (suspected) software rasterizers and renderers. These
    // can be useful for debugging purposes, but normally, their use is
    // undesirable when GPU-accelerated processing is expected.
    bool allow_software;

    // Restrict specific features to e.g. work around driver bugs, or simply
    // for testing purposes
    pl_gpu_caps blacklist_caps; // capabilities to be excluded
    int max_glsl_version;       // limit the maximum GLSL version

    // Optional. Required when importing/exporting dmabufs as textures.
    void *egl_display;
    void *egl_context;
};

// Default/recommended parameters
extern const struct pl_opengl_params pl_opengl_default_params;

// Creates a new OpenGL renderer based on the given parameters. This will
// internally use whatever platform-defined mechanism (WGL, X11, EGL) is
// appropriate for loading the OpenGL function calls, so the user doesn't need
// to pass in a `getProcAddress` callback. If `params` is left as NULL, it
// defaults to `&pl_opengl_default_params`. The context must be active when
// calling this function, and must remain active whenever calling any
// libplacebo function on the resulting `pl_opengl` or `pl_gpu`.
//
// Note that creating multiple `pl_opengl` instances from the same OpenGL
// context is undefined behavior.
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

struct pl_opengl_wrap_params {
    // The GLuint texture object itself. Optional. If no texture is provided,
    // then only the opaque framebuffer `fbo` will be wrapped, leaving the
    // resulting `pl_tex` object with some operations (such as sampling) being
    // unsupported.
    unsigned int texture;

    // The GLuint associated framebuffer. Optional. If this is not specified,
    // then libplacebo will attempt creating a framebuffer from the provided
    // texture object (if possible).
    //
    // Note: As a special case, if neither a texture nor an FBO are provided,
    // this is equivalent to wrapping the OpenGL default framebuffer (id 0).
    unsigned int framebuffer;

    // The image's dimensions (unused dimensions must be 0)
    int width;
    int height;
    int depth;

    // Texture-specific fields:
    //
    // Note: These are only relevant if `texture` is provided.

    // The GLenum for the texture target to use, e.g. GL_TEXTURE_2D. Optional.
    // If this is left as 0, the target is inferred from the number of
    // dimensions. Users may want to set this to something specific like
    // GL_TEXTURE_EXTERNAL_OES depending on the nature of the texture.
    unsigned int target;

    // The texture's GLint sized internal format (e.g. GL_RGBA16F). Required.
    int iformat;

    // Deprecated fields. These are now ignored completely.
    int filter PL_DEPRECATED;
    int address_mode PL_DEPRECATED;
};

// Wraps an external OpenGL object into a `pl_tex` abstraction. Due to the
// internally synchronized nature of OpenGL, no explicit synchronization
// is needed between libplacebo `pl_tex_` operations, and host accesses to
// the texture. Wrapping the same OpenGL texture multiple times is permitted.
// Note that this function transfers no ownership.
//
// This wrapper can be destroyed by simply calling `pl_tex_destroy` on it,
// which will *not* destroy the user-provided OpenGL texture or framebuffer.
//
// This function may fail, in which case it returns NULL.
const struct pl_tex *pl_opengl_wrap(const struct pl_gpu *gpu,
                                    const struct pl_opengl_wrap_params *params);

// Analogous to `pl_opengl_wrap`, this function takes any `pl_tex` (including
// ones created by `pl_tex_create`) and unwraps it to expose the underlying
// OpenGL texture to the user. Note that this function transfers no ownership,
// i.e. the texture object and framebuffer shall not be destroyed by the user.
//
// Returns the OpenGL texture. `out_target` and `out_iformat` will be updated
// to hold the target type and internal format, respectively. (Optional)
//
// For renderable/blittable textures, `out_fbo` will be updated to the ID of
// the framebuffer attached to this texture, or 0 if there is none. (Optional)
unsigned int pl_opengl_unwrap(const struct pl_gpu *gpu, const struct pl_tex *tex,
                              unsigned int *out_target, int *out_iformat,
                              unsigned int *out_fbo);

#endif // LIBPLACEBO_OPENGL_H_
