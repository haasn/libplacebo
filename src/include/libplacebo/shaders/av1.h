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

// AV1 film grain parameters. For the exact meaning of these, see the
// specification (section 6.8.20).
//
// NOTE: These parameters are currently *not* sanity checked. Calling these
// functions with e.g. too large `num_points_y` or negative width/height is UB!
// Please make sure to sanity check the grain parameters signalled by the file
// before calling into these functions.
struct pl_grain_params {
    int width, height;           // dimensions of the image (luma)
    int sub_x, sub_y;            // subsampling shifts for the chroma planes
    struct pl_color_repr repr;   // underlying color system
    // Some notes apply to `repr`:
    //  - repr.bits affects the rounding for grain generation
    //  - repr.levels affects whether or not we clip to full range or not
    //  - repr.sys affects whether channels 1 and 2 are treated like chroma

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

// Apply AV1 film grain to the channels given in `channel_map`, which maps
// from the component index of the vec4 color to the channel contained in that
// index, or PL_CHANNEL_NONE for unused channels.
//
// For example, if this is the pass for the subsampled Cb and Cr planes, which
// are currently available in color.xy, then `channels` would be:
// {PL_CHANNEL_CB, PL_CHANNEL_CR, PL_CHANNEL_NONE} = {1, 2, -1}
//
// When applying grain to the channels 1 and 2 channels, access to information
// from channel 0 is needed. It's important to take this information from the
// undistorted plane (before applying grain), and must be passed as the texture
// `luma_tex` - unless the channel map already includes channel 0 channel.
//
// So for example, for planar YCbCr content, grain must be added to the chroma
// channels first, then followed by the luma channels. (For packed content like
// rgb24 where all channels are part of the same pass, this is unnecessary)
//
// Note: all of this applies even if params->repr.sys == PL_COLOR_SYSTEM_RGB (!)
void pl_shader_av1_grain(struct pl_shader *sh,
                         struct pl_shader_obj **grain_state,
                         const enum pl_channel channels[3],
                         const struct pl_tex *luma_tex,
                         const struct pl_grain_params *params);

#endif // LIBPLACEBO_SHADERS_AV1_H_
