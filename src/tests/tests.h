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

static inline pl_log pl_test_logger(void)
{
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    return pl_log_create(PL_API_VER, pl_log_params(
        .log_cb    = isatty(fileno(stdout)) ? pl_log_color : pl_log_timestamp,
        .log_level = PL_LOG_DEBUG,
    ));
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

static const uint8_t h274_lower_bound = 10;
static const uint8_t h274_upper_bound = 250;
static const int16_t h274_values[6] = {16, 12, 14};

static const struct pl_h274_grain_data h274_grain_data = {
    .model_id = 0,
    .blending_mode_id = 0,
    .log2_scale_factor = 2,
    .component_model_present = {true},
    .num_intensity_intervals = {1},
    .num_model_values = {3},
    .intensity_interval_lower_bound = {&h274_lower_bound},
    .intensity_interval_upper_bound = {&h274_upper_bound},
    .comp_model_value = {&h274_values},
};

static const struct pl_dovi_metadata dovi_meta = {
    .nonlinear = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
    .linear    = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
    .comp = {
        {
            .num_pivots = 9,
            .pivots = {0.0615835786, 0.129032254, 0.353861183,
                       0.604105592, 0.854349971, 0.890518069,
                       0.906158328, 0.913978517, 0.92082113},
            .method = {0, 0, 0, 0, 0, 0, 0, 0},
            .poly_coeffs = {
                {-0.0488376617, 1.99335372, -2.41716385},
                {-0.0141925812, 1.61829138, -1.53397191},
                { 0.157061458, 0.63640213, -0.11302495},
                {0.25272119, 0.246226311, 0.27281332},
                {0.951621532, -1.35507894, 1.18898678},
                {6.41251612, -13.6188488, 8.07336903},
                {13.467535, -29.1869125, 16.6612244},
                {28.2321472, -61.8516273, 34.7264938}
            },
        }, {
            .num_pivots = 2,
            .pivots = {0.0, 1.0},
            .method = {1},
            .mmr_order = {3},
            .mmr_constant = {-0.500733018},
            .mmr_coeffs = {{
                {1.08411026, 3.80807829, 0.0881733894, -3.23097038, -0.409078479, -1.31310081, 2.71297002},
                {-0.241833091, -3.57880807, -0.108109117, 3.13198471, 0.869203091, 1.96561158, -9.30871677},
                {-0.177356839, 1.48970401, 0.0908923149, -0.510447979, -0.687603354, -0.934977889, 12.3544884},
            }},
        }, {
            .num_pivots = 2,
            .pivots = {0.0, 1.0},
            .method = {1},
            .mmr_order = {3},
            .mmr_constant = {-1.23833287},
            .mmr_coeffs = {{
                {3.52909589, 0.383154511, 5.50820637, -1.02094889, -6.36386824, 0.194121242, 0.64683497},
                {-2.57899785, -0.626081586, -6.05729723, 2.29143763, 9.14653015, -0.0507702827, -4.17724133},
                {0.705404401, 0.341412306, 2.98387456, -1.71712542, -4.91501331, 0.1465137, 6.38665438},
            }},
        },
    },
};
