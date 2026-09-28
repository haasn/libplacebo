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

#include "shaders.h"

#include <libplacebo/shaders/motion.h>

// These are internal helpers to bind various buffers or textures to shaders.
// The public API in `include/libplacebo/shaders/motion.h` contain functions
// for getting the `pl_tex` or `pl_buf` directly.

// Binds the global motion SSBO of a single direction to `sh` as a read-only
// storage buffer descriptor, and returns an identifier for a
// `vec3(mv.x, mv.y, confidence)` representing the global motion estimate.
// mv.x and mv.y are integer pixel offsets, confidence is a float in [0,1], with
// 1 representing 100% confidence. Returns NULL_IDENT if the global motion of
// `dir` was not estimated for the current frame pair. May be called once per
// direction per shader.
// NOTE: the confidence is currently always 1.
ident_t sh_motion_global(pl_shader sh, pl_shader_obj state,
                         enum pl_motion_direction dir);

// Binds the top-level flow field of a single direction to `sh` as a read-only
// texture with nearest sampling, and returns an identifier for the texture.
// The texture is not guaranteed to match the dimensions of the frames that it
// estimated motion between, as motion may be stored in block granularity; the
// texture should be sampled with relative coordinates. Each element of the
// texture holds in .xyz the data `vec3(mv.x, mv.y, confidence)`, where mv.x
// and mv.y are integer pixel offsets, confidence a float in [0,1]. Returns
// NULL_IDENT if the most recent call to `pl_motion_estimate` did not compute
//`dir`, or if it was not successfully called prior to this function call.
ident_t sh_motion_flow(pl_shader sh, pl_shader_obj state,
                       enum pl_motion_direction dir);
