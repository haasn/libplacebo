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

#pragma once

#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

static inline struct pl_context *pl_test_context()
{
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    return pl_context_create(PL_API_VER, &(struct pl_context_params) {
        .log_cb    = isatty(fileno(stdout)) ? pl_log_color : pl_log_simple,
        .log_level = PL_LOG_DEBUG,
    });
}

static inline void pl_test_set_verbosity(struct pl_context *ctx,
                                         enum pl_log_level level)
{
    pl_context_update(ctx, &(struct pl_context_params) {
        .log_cb    = isatty(fileno(stdout)) ? pl_log_color : pl_log_simple,
        .log_level = level,
    });
}

static inline void require(bool b, const char *msg)
{
    if (!b) {
        fprintf(stderr, "%s", msg);
        exit(1);
    }
}

static inline bool feq(float a, float b, float epsilon)
{
    return fabs(a - b) < epsilon * fmax(1.0, fabs(a));
}

#define REQUIRE(cond) require((cond), #cond)
#define RANDOM (rand() / (float) RAND_MAX)
#define SKIP 77

static const struct pl_av1_grain_data av1_grain_data = {
    .grain_seed = 48476,

    .num_points_y = 6,
    .points_y = {{0, 4}, {27, 33}, {54, 55}, {67, 61}, {108, 71}, {255, 72}},
    .chroma_scaling_from_luma = false,
    .num_points_uv = {2, 2},
    .points_uv = {{{0, 64}, {255, 64}}, {{0, 64}, {255, 64}}},
    .scaling_shift = 11,
    .ar_coeff_lag = 3,
    .ar_coeffs_y = {4,   1, 3,   0,  1, -3,  8, -3,  7, -23, 1, -25,
                    0, -10, 6, -17, -4, 53, 36,  5, -5, -17, 8,  66},
    .ar_coeffs_uv = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127},
    },
    .ar_coeff_shift = 7,
    .grain_scale_shift = 0,
    .uv_mult = {0, 0},
    .uv_mult_luma = {64, 64},
    .uv_offset = {0, 0},
};
