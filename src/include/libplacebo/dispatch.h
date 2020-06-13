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

#ifndef LIBPLACEBO_DISPATCH_H_
#define LIBPLACEBO_DISPATCH_H_

#include <libplacebo/shaders.h>
#include <libplacebo/gpu.h>

struct pl_dispatch;

// Creates a new shader dispatch object. This object provides a translation
// layer between generated shaders (pl_shader) and the ra context such that it
// can be used to execute shaders. This dispatch object will also provide
// shader caching (for efficient re-use).
struct pl_dispatch *pl_dispatch_create(struct pl_context *ctx,
                                       const struct pl_gpu *gpu);
void pl_dispatch_destroy(struct pl_dispatch **dp);

// Returns a blank pl_shader object, suitable for recording rendering commands.
// For more information, see the header documentation in `shaders/*.h`.
struct pl_shader *pl_dispatch_begin(struct pl_dispatch *dp);

struct pl_dispatch_params {
    // The shader to execute. The pl_dispatch will take over ownership
    // of this shader, and return it back to the internal pool.
    //
    // This shader must have a compatible signature, i.e. inputs
    // `PL_SHADER_SIG_NONE` and outputs `PL_SHADER_SIG_COLOR`.
    struct pl_shader **shader;

    // The texture to render to. This must have params compatible with the
    // shader, i.e. `target->params.renderable` for fragment shaders and
    // `target->params.storable` for compute shaders.
    //
    // Note: Even when not using compute shaders, users are advised to always
    // set `target->params.storable` if permitted by the `pl_fmt`, since this
    // allows the use of compute shaders instead of full-screen quads, which is
    // faster on some platforms.
    const struct pl_tex *target;

    // The target rect to render to. Optional, if left as {0}, then the
    // entire texture will be rendered to.
    struct pl_rect2d rect;

    // If set, enables and controls the blending for this pass. Optional. When
    // using this with fragment shaders, `target->params.fmt->caps` must
    // include `PL_FMT_CAP_BLENDABLE`.
    const struct pl_blend_params *blend_params;

    // If set, records the execution time of this dispatch into the given
    // timer object. Optional.
    struct pl_timer *timer;
};

// Dispatch a generated shader (via the pl_shader mechanism). Returns whether
// or not the dispatch was successful.
bool pl_dispatch_finish(struct pl_dispatch *dp, const struct pl_dispatch_params *params);

struct pl_dispatch_compute_params {
    // The shader to execute. This must be a compute shader with both the
    // input and output signature set to PL_SHADER_SIG_NONE.
    //
    // Note: There is currently no way to actually construct such a shader with
    // the currently available public APIs. (However, it's still used
    // internally, and may be needed in the future)
    struct pl_shader **shader;

    // The number of work groups to dispatch in each dimension. Must be
    // nonzero for all three dimensions.
    int dispatch_size[3];

    // If set, simulate vertex attributes (similar to `pl_dispatch_finish`)
    // according to the given dimensions. The first two components of the
    // thread's global ID will be interpreted as the X and Y locations.
    //
    // Optional, ignored if either component is left as 0.
    int width, height;

    // If set, records the execution time of this dispatch into the given
    // timer object. Optional.
    struct pl_timer *timer;
};

// A variant of `pl_dispatch_finish`, this one only dispatches a compute shader
// that has no output.
bool pl_dispatch_compute(struct pl_dispatch *dp,
                         const struct pl_dispatch_compute_params *params);

// Cancel an active shader without submitting anything. Useful, for example,
// if the shader was instead merged into a different shader.
void pl_dispatch_abort(struct pl_dispatch *dp, struct pl_shader **sh);

// Serialize the internal state of a `pl_dispatch` into an abstract cache
// object that can be e.g. saved to disk and loaded again later. This function
// will not truncate, so the buffer provided by the user must be large enough
// to contain the entire output. Returns the number of bytes written to
// `out_cache`, or the number of bytes that *would* have been written to
// `out_cache` if it's NULL.
size_t pl_dispatch_save(struct pl_dispatch *dp, uint8_t *out_cache);

// Load the result of a previous `pl_dispatch_save` call. This function will
// never fail. It doesn't forget about any existing shaders, but merely
// initializes an internal state cache needed to more efficiently compile
// shaders that are not yet present in the `pl_dispatch`.
void pl_dispatch_load(struct pl_dispatch *dp, const uint8_t *cache);

#endif // LIBPLACEBO_DISPATCH_H
