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
#include <time.h>

static void pl_log_timestamp(void *stream, enum pl_log_level level, const char *msg)
{
    static char letter[] = {
        [PL_LOG_FATAL] = 'f',
        [PL_LOG_ERR]   = 'e',
        [PL_LOG_WARN]  = 'w',
        [PL_LOG_INFO]  = 'i',
        [PL_LOG_DEBUG] = 'd',
        [PL_LOG_TRACE] = 't',
    };

    float secs = (float) clock() / CLOCKS_PER_SEC;
    printf("[%2.3f][%c] %s\n", secs, letter[level], msg);

    if (level <= PL_LOG_WARN) {
        // duplicate warnings/errors to stderr
        fprintf(stderr, "[%2.3f][%c] %s\n", secs, letter[level], msg);
        fflush(stderr);
    }
}

static inline pl_log pl_test_logger()
{
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    return pl_log_create(PL_API_VER, &(struct pl_log_params) {
        .log_cb    = isatty(fileno(stdout)) ? pl_log_color : pl_log_timestamp,
        .log_level = PL_LOG_DEBUG,
    });
}

static inline void require(bool b, const char *msg, const char *file, int line)
{
    if (!b) {
        fprintf(stderr, "=== FAILED: '%s' at %s:%d\n\n", msg, file, line);
        exit(1);
    }
}

static inline bool feq(float a, float b, float epsilon)
{
    return fabs(a - b) < epsilon * fmax(1.0, fabs(a));
}

#define REQUIRE(cond) require((cond), #cond, __FILE__, __LINE__)
#define RANDOM (rand() / (float) RAND_MAX)
#define SKIP 77

#define REQUIRE_HANDLE(shmem, type)                                             \
    switch (type) {                                                             \
    case PL_HANDLE_FD:                                                          \
    case PL_HANDLE_DMA_BUF:                                                     \
        REQUIRE(shmem.handle.fd > -1);                                          \
        break;                                                                  \
    case PL_HANDLE_WIN32:                                                       \
    case PL_HANDLE_WIN32_KMT:                                                   \
        REQUIRE(shmem.handle.handle);                                           \
        /* INVALID_HANDLE_VALUE = (-1) */                                       \
        REQUIRE(shmem.handle.handle != (void *)(intptr_t) (-1));                \
        break;                                                                  \
    case PL_HANDLE_HOST_PTR:                                                    \
        REQUIRE(shmem.handle.ptr);                                              \
        break;                                                                  \
    }

static const struct pl_av1_grain_data av1_grain_data = {
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

static const struct pl_h274_grain_data h274_grain_data = {
    .model_id = 0,
    .blending_mode_id = 0,
    .log2_scale_factor = 2,
    .component_model_present = {true},
    .num_intensity_intervals = {1},
    .num_model_values = {3},
    .intensity_interval_lower_bound = {{10}},
    .intensity_interval_upper_bound = {{250}},
    .comp_model_value = {{{16, 12, 14}}},
};
