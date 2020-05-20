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

#ifndef LIBPLACEBO_SHADERS_CUSTOM_H_
#define LIBPLACEBO_SHADERS_CUSTOM_H_

// Framework for enabling custom user shader hooks, as well as compatibility
// functions for parsing shaders in mpv format.

#include <libplacebo/shaders.h>

// Which "rendering stages" are available for user shader hooking purposes.
// Except where otherwise noted, all stages are "non-resizable", i.e. the
// shaders already have specific output size requirements.
enum pl_hook_stage {
    // Hook stages for the untouched planes, as made available by the source.
    // These are all resizable, i.e. there are no specific output stage
    // requirements.
    PL_HOOK_RGB_INPUT       = 1 << 0,
    PL_HOOK_LUMA_INPUT      = 1 << 1,
    PL_HOOK_CHROMA_INPUT    = 1 << 2,
    PL_HOOK_ALPHA_INPUT     = 1 << 3,
    PL_HOOK_XYZ_INPUT       = 1 << 4,

    // Hook stages for the scaled/aligned planes
    PL_HOOK_CHROMA_SCALED   = 1 << 5,
    PL_HOOK_ALPHA_SCALED    = 1 << 6,

    PL_HOOK_NATIVE          = 1 << 7,  // Combined image in its native color space
    PL_HOOK_RGB             = 1 << 8,  // After conversion to RGB (resizable)
    PL_HOOK_LINEAR          = 1 << 9,  // After linearization but before scaling
    PL_HOOK_SIGMOID         = 1 << 10, // After sigmoidization
    PL_HOOK_PRE_OVERLAY     = 1 << 11, // Before applying on-image overlays
    PL_HOOK_PRE_KERNEL      = 1 << 12, // Immediately before the main scaler kernel (after overlays)
    PL_HOOK_POST_KERNEL     = 1 << 13, // Immediately after the main scaler kernel
    PL_HOOK_SCALED          = 1 << 14, // After scaling, before color management
    PL_HOOK_OUTPUT          = 1 << 15, // After color management, before dithering
};

// Returns true if a given hook stage is resizable
static inline bool pl_hook_stage_resizable(enum pl_hook_stage stage) {
    switch (stage) {
    case PL_HOOK_RGB_INPUT:
    case PL_HOOK_LUMA_INPUT:
    case PL_HOOK_CHROMA_INPUT:
    case PL_HOOK_ALPHA_INPUT:
    case PL_HOOK_XYZ_INPUT:
    case PL_HOOK_NATIVE:
    case PL_HOOK_RGB:
        return true;
    default:
        return false;
    }
}

// The different forms of communicating image data between the renderer and
// the hooks
enum pl_hook_sig {
    PL_HOOK_SIG_NONE,   // No data is passed, no data is received/returned
    PL_HOOK_SIG_COLOR,  // `vec4 color` already pre-sampled in a `pl_shader`
    PL_HOOK_SIG_TEX,    // `pl_tex` containing the image data
};

struct pl_hook_params {
    // GPU objects associated with the `pl_renderer`, which the user may
    // use for their own purposes.
    const struct pl_gpu *gpu;
    struct pl_dispatch *dispatch;

    // Helper function to fetch a new temporary texture, using renderer-backed
    // storage. This is guaranteed to have sane image usage requirements and a
    // 16-bit or floating point format. The user does not need to free/destroy
    // this texture in any way. May return NULL.
    const struct pl_tex *(*get_tex)(void *priv, int width, int height);
    void *priv;

    // Which stage triggered the hook to run.
    enum pl_hook_stage stage;

    // For `PL_HOOK_SIG_COLOR`, this contains the existing shader object with
    // the color already pre-sampled into `vec4 color`. The user may modify
    // this as much as they want, as long as they don't dispatch/finalize/reset
    // it.
    //
    // Note that this shader might have specific output size requirements,
    // depending on the exact shader stage hooked by the user, and may already
    // be a compute shader.
    struct pl_shader *sh;

    // For `PL_HOOK_SIG_TEX`, this contains the texture that the user should
    // sample from.
    //
    // Note: This texture object is owned by the renderer, and users must not
    // modify its contents. It will not be touched for the duration of a frame,
    // but the contents are lost in between frames.
    const struct pl_tex *tex;

    // The effective current rectangle of the image we're rendering in this
    // shader, i.e. the effective rect of the content we're interested in,
    // as a crop of either `sh` or `tex` (depending on the signature).
    //
    // Note: This is still set even for `PL_HOOK_SIG_NONE`!
    struct pl_rect2df rect;

    // The current effective colorspace and representation, of either the
    // pre-sampled color (in `sh`), or the contents of `tex`, respectively.
    //
    // Note: This is still set even for `PL_HOOK_SIG_NONE`!
    struct pl_color_repr repr;
    struct pl_color_space color;
    int components;

    // The (cropped) source and destination rectangles of the overall
    // rendering. These are functionallty equivalent to `pl_image.src_rect` and
    // `pl_target.dst_rect`, respectively, but `src_rect` in particular may
    // change as a result of previous hooks being executed. (e.g. prescalers)
    struct pl_rect2df src_rect;
    struct pl_rect2d dst_rect;
};

struct pl_hook_res {
    // If true, the hook is assumed to have "failed" or errored in some way,
    // and all other fields are ignored.
    bool failed;

    // What type of output this hook is returning.
    // Note: If this is `PL_HOOK_SIG_NONE`, all other fields are ignored.
    enum pl_hook_sig output;

    // For `PL_HOOK_SIG_COLOR`, this *must* be set to a valid `pl_shader`
    // object containing the sampled color value (i.e. with an output signature
    // of `PL_SHADER_SIG_COLOR`), and *should* be allocated from the given
    // `pl_dispatch` object. Ignored otherwise.
    struct pl_shader *sh;

    // For `PL_HOOK_SIG_TEX`, this *must* contain the texture object containing
    // the result of rendering the hook. This *should* be a texture allocated
    // using the given `get_tex` callback, to ensure the format and texture
    // usage flags are compatible with what the renderer expects.
    const struct pl_tex *tex;

    // For shaders that return some sort of output, this contains the
    // new/altered versions of the existing "current texture" metadata.
    struct pl_color_repr repr;
    struct pl_color_space color;
    int components;

    // This contains the new effective rect of the contents. This may be
    // different from the original `rect` for resizable passes. Ignored for
    // non-resizable passes.
    struct pl_rect2df rect;
};

struct pl_hook {
    enum pl_hook_stage stages;  // Which stages to hook on
    enum pl_hook_sig input;     // Which input signature this hook expects
    void *priv;                 // Arbitrary user context

    // Called at the beginning of passes, to reset/initialize the hook. (Optional)
    void (*reset)(void *priv);

    // The hook function itself. Called by the renderer at any of the indicated
    // hook stages. See `pl_hook_res` for more info on the return values.
    struct pl_hook_res (*hook)(void *priv, const struct pl_hook_params *params);
};

// Compatibility layer with `mpv` user shaders. See the mpv man page for more
// information on the format. Will return `NULL` if the shader fails parsing.
//
// The resulting `pl_hook` objects should be destroyed with the corresponding
// destructor when no longer needed.
const struct pl_hook *pl_mpv_user_shader_parse(const struct pl_gpu *gpu,
                                               const char *shader_text,
                                               size_t shader_len);

void pl_mpv_user_shader_destroy(const struct pl_hook **hook);

#endif // LIBPLACEBO_SHADERS_CUSTOM_H_
