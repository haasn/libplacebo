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

// Dispatch a generated shader (via the pl_shader mechanism). The results of
// shader execution will be rendered to `target`. Returns whether or not the
// dispatch was successful. This operation will take over ownership of the
// pl_shader passed to it, and return it back to the internal pool.
//
// If `rc` is NULL, renders to the entire texture.
// If set, `blend_params` enables and controls blending for this pass.
//
// Note: `target` must have params compatible with the shader, i.e.
// `target->params.renderable` for fragment shaders and
// `target->params.storable` for compute shaders. Additionally, for fragment
// shaders only, use of `blend_params` requires the target be created with a
// `pl_fmt` that includes `PL_FMT_CAP_BLENDABLE`.
//
// Note: Even when not using compute shaders, users are advised to always set
// `target->params.storable` if permitted by the `pl_fmt`, for efficiency
// reasons.
bool pl_dispatch_finish(struct pl_dispatch *dp, struct pl_shader **sh,
                        const struct pl_tex *target, const struct pl_rect2d *rc,
                        const struct pl_blend_params *blend_params);

// A variant of `pl_dispatch_finish`, this one only dispatches a compute shader
// that has no output.
//
// Note: As an additonal feature, this function supports simulating vertex
// attributes (in the style of `pl_dispatch_finish`). The use of this
// functionality requires the user specify the effective rendering width/height.
// Leaving these as 0 disables this feature.
//
// Note: There is currently no way to actually construct such a shader with the
// currently available public APIs. (However, it's still used internally, and
// may be needed in the future)
bool pl_dispatch_compute(struct pl_dispatch *dp, struct pl_shader **sh,
                         int dispatch_size[3], int width, int height);

// Cancel an active shader without submitting anything. Useful, for example,
// if the shader was instead merged into a different shader.
void pl_dispatch_abort(struct pl_dispatch *dp, struct pl_shader **sh);

#endif // LIBPLACEBO_DISPATCH_H
