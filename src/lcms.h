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

// Compute a transformation from one color profile to another, and fill the
// provided array by the resulting 3DLUT. The array must have room for four
// components per sample.
bool pl_lcms_compute_lut(struct pl_context *ctx, enum pl_rendering_intent intent,
                         struct pl_3dlut_profile src, struct pl_3dlut_profile dst,
                         float *out_data, int s_r, int s_g, int s_b,
                         struct pl_3dlut_result *out);
