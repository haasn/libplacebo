/*
 * This file is part of libplacebo, which is normally licensed under the terms
 * of the LGPL v2.1+. However, this file (av1.h) is also available under the
 * terms of the more permissive MIT license:
 *
 * Copyright (c) 2018-2019 Niklas Haas
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef LIBPLACEBO_SHADERS_AV1_H_
#define LIBPLACEBO_SHADERS_AV1_H_

// Helper shaders for AV1. For now, just film grain.

#include <stdint.h>
#include <stdbool.h>

#include <libplacebo/colorspace.h>
#include <libplacebo/shaders.h>

// AV1 film grain parameters. For the exact meaning of these, see the AV1
// specification (section 6.8.20).
//
// NOTE: These parameters are currently *not* sanity checked. Calling these
// functions with e.g. too large `num_points_y` or negative width/height is UB!
// Please make sure to sanity check the grain parameters signalled by the file
// before calling into these functions.
struct pl_av1_grain_data {
    uint16_t grain_seed;
    int num_points_y;
    uint8_t points_y[14][2];     // [n][0] = value, [n][1] = scaling
    bool chroma_scaling_from_luma;
    int num_points_uv[2];        // should be {0} for grayscale images
    uint8_t points_uv[2][10][2]; // like points_y for points_uv[0, 1] = u, v
    int scaling_shift;
    int ar_coeff_lag;
    int8_t ar_coeffs_y[24];
    int8_t ar_coeffs_uv[2][25];
    int ar_coeff_shift;
    int grain_scale_shift;
    int8_t uv_mult[2];
    int8_t uv_mult_luma[2];
    int16_t uv_offset[2];        // 9-bit value, range [-256, 255]
    bool overlap;
};

// Struct containing extra options for the `pl_shader_av1_grain` call.
struct pl_av1_grain_params {
    struct pl_av1_grain_data data;  // av1 grain metadata itself
    const struct pl_tex *tex;       // texture to sample from
    const struct pl_tex *luma_tex;  // "luma" texture (see notes)
    struct pl_color_repr *repr;     // underlying color representation (see notes)
    int components;
    int component_mapping[4];       // same as `struct pl_plane`
    int luma_comp;                  // index of luma in `luma_tex`

    // Notes for `repr`:
    //  - repr->bits affects the rounding for grain generation
    //  - repr->levels affects whether or not we clip to full range or not
    //  - repr->sys affects the interpretation of channels
    //  - *repr gets normalized by this shader, which is why it's a pointer
    //
    // Notes for `luma_tex`:
    //  - `luma_tex` must be specified if the `tex` does not itself contain the
    //     "luma-like" component. For XYZ systems, the Y channel is the luma
    //     component. For RGB systems, the G channel is.
};

// Test if AV1 film grain needs to be applied. This is a helper function
// that users can use to decide whether or not `pl_shader_av1_grain` needs
// to be called, based on the given grain metadata.
bool pl_needs_av1_grain(const struct pl_av1_grain_params *params);

// Sample from a texture while applying AV1 grain at the same time.
// `grain_state` should be unique for every plane, as it only contains the
// state relevant for this particular plane configuration.
//
// Returns false on any error, or if AV1 grain generation is not supported.
// (Requires GLSL version 130 or newer)
bool pl_shader_av1_grain(struct pl_shader *sh,
                         struct pl_shader_obj **grain_state,
                         const struct pl_av1_grain_params *params);

#endif // LIBPLACEBO_SHADERS_AV1_H_
