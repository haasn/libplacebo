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

#include <libplacebo/log.h>
#include <libplacebo/colorspace.h>
#include <libplacebo/shaders/film_grain.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifdef PL_HAVE_WIN32
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

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

    // Log time relative to the first message
    static pl_clock_t base = 0;
    if (!base)
        base = pl_clock_now();

    double secs = pl_clock_diff(pl_clock_now(), base);
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

#define RANDOM (rand() / (float) RAND_MAX)
#define RANDOM_U8 ((uint8_t) (256.0 * rand() / (RAND_MAX + 1.0)))
#define SKIP 77

// Helpers for performing various checks
#define REQUIRE(cond) do                                                        \
{                                                                               \
    if (!(cond)) {                                                              \
        fprintf(stderr, "=== FAILED: '"#cond"' at "__FILE__":%d\n\n", __LINE__);\
        exit(1);                                                                \
    }                                                                           \
} while (0)

#define REQUIRE_CMP(a, op, b, fmt) do                                           \
{                                                                               \
    __typeof__(a) _va = (a), _vb = (b);                                         \
                                                                                \
    if (!(_va op _vb)) {                                                        \
        fprintf(stderr, "=== FAILED: '"#a" "#op" "#b"' at "__FILE__":%d\n"      \
                        " %-31s = %"fmt"\n"                                     \
                        " %-31s = %"fmt"\n\n",                                  \
                __LINE__, #a, _va, #b, _vb);                                    \
        exit(1);                                                                \
    }                                                                           \
} while (0)

#define REQUIRE_FEQ(a, b, epsilon) do                                           \
{                                                                               \
    float _va = (a);                                                            \
    float _vb = (b);                                                            \
    float _delta = (epsilon) * fmax(1.0, fabs(_va));                            \
                                                                                \
    if (fabs(_va - _vb) > _delta) {                                             \
        fprintf(stderr, "=== FAILED: '"#a" ≈ "#b"' at "__FILE__":%d\n"          \
                        " %-31s = %f\n"                                         \
                        " %-31s = %f\n"                                         \
                        " %-31s = %f\n\n",                                      \
                __LINE__, #a, _va, #b, _vb,                                     \
                "epsilon "#epsilon" -> max delta", _delta);                     \
        exit(1);                                                                \
    }                                                                           \
} while (0)

#define REQUIRE_STREQ(a, b) do                                                  \
{                                                                               \
    const char *_a = (a);                                                       \
    const char *_b = (b);                                                       \
    if (strcmp(_a, _b) != 0) {                                                  \
        fprintf(stderr, "=== FAILED: !strcmp("#a", "#b") at "__FILE__":%d\n"    \
                        " %-31s = %s\n"                                         \
                        " %-31s = %s\n\n",                                      \
                __LINE__, #a, _a, #b, _b);                                      \
        exit(1);                                                                \
    }                                                                           \
} while (0)

static inline void log_array(const uint8_t *a, const uint8_t *ref, size_t size)
{
    for (size_t n = 0; n < size; n++) {
        const char *prefix = "", *suffix = "";
        char terminator = ' ';
        if (a[n] != ref[n]) {
            prefix = "\033[31;1m";
            suffix = "\033[0m";
        }
        if (n+1 == size || n % 16 == 15)
            terminator = '\n';
        fprintf(stderr, "%s%02"PRIx8"%s%c", prefix, a[n], suffix, terminator);
    }
}

static inline void require_memeq(const void *aptr, const void *bptr, size_t size,
                                 const char *astr, const char *bstr,
                                 const char *sizestr, const char *file, int line)
{
    const uint8_t *a = aptr, *b = bptr;
    for (size_t i = 0; i < size; i++) {
        if (a[i] == b[i])
            continue;

        fprintf(stderr, "=== FAILED: memcmp(%s, %s, %s) == 0 at %s:%d\n"
                        "at position %zu: 0x%02"PRIx8" != 0x%02"PRIx8"\n\n",
                astr, bstr, sizestr, file, line, i, a[i], b[i]);

        const size_t logsize = PL_MIN(size, PL_MAX(i+2, 512));
        fprintf(stderr, "first %zu bytes of '%s':\n", logsize, astr);
        log_array(a, b, logsize);
        fprintf(stderr, "\nfirst %zu bytes of '%s':\n", logsize, bstr);
        log_array(b, a, logsize);
        exit(1);
    }
}

#define REQUIRE_MEMEQ(a, b, size) require_memeq(a, b, size, #a, #b, #size, __FILE__, __LINE__)

#define REQUIRE_HANDLE(shmem, type)                                             \
    switch (type) {                                                             \
    case PL_HANDLE_FD:                                                          \
    case PL_HANDLE_DMA_BUF:                                                     \
        REQUIRE(shmem.handle.fd > -1);                                          \
        break;                                                                  \
    case PL_HANDLE_WIN32:                                                       \
    case PL_HANDLE_WIN32_KMT:                                                   \
        /* INVALID_HANDLE_VALUE = (-1) */                                       \
        REQUIRE(shmem.handle.handle != (void *)(intptr_t) (-1));                \
        /* fallthrough */                                                       \
    case PL_HANDLE_MTL_TEX:                                                     \
    case PL_HANDLE_IOSURFACE:                                                   \
        REQUIRE(shmem.handle.handle);                                           \
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

static const uint8_t sRGB_v2_nano_icc[] = {
  0x00, 0x00, 0x01, 0x9a, 0x6c, 0x63, 0x6d, 0x73, 0x02, 0x10, 0x00, 0x00,
  0x6d, 0x6e, 0x74, 0x72, 0x52, 0x47, 0x42, 0x20, 0x58, 0x59, 0x5a, 0x20,
  0x07, 0xe2, 0x00, 0x03, 0x00, 0x14, 0x00, 0x09, 0x00, 0x0e, 0x00, 0x1d,
  0x61, 0x63, 0x73, 0x70, 0x4d, 0x53, 0x46, 0x54, 0x00, 0x00, 0x00, 0x00,
  0x73, 0x61, 0x77, 0x73, 0x63, 0x74, 0x72, 0x6c, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf6, 0xd6,
  0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0xd3, 0x2d, 0x68, 0x61, 0x6e, 0x64,
  0xeb, 0x77, 0x1f, 0x3c, 0xaa, 0x53, 0x51, 0x02, 0xe9, 0x3e, 0x28, 0x6c,
  0x91, 0x46, 0xae, 0x57, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09,
  0x64, 0x65, 0x73, 0x63, 0x00, 0x00, 0x00, 0xf0, 0x00, 0x00, 0x00, 0x5f,
  0x77, 0x74, 0x70, 0x74, 0x00, 0x00, 0x01, 0x0c, 0x00, 0x00, 0x00, 0x14,
  0x72, 0x58, 0x59, 0x5a, 0x00, 0x00, 0x01, 0x20, 0x00, 0x00, 0x00, 0x14,
  0x67, 0x58, 0x59, 0x5a, 0x00, 0x00, 0x01, 0x34, 0x00, 0x00, 0x00, 0x14,
  0x62, 0x58, 0x59, 0x5a, 0x00, 0x00, 0x01, 0x48, 0x00, 0x00, 0x00, 0x14,
  0x72, 0x54, 0x52, 0x43, 0x00, 0x00, 0x01, 0x5c, 0x00, 0x00, 0x00, 0x34,
  0x67, 0x54, 0x52, 0x43, 0x00, 0x00, 0x01, 0x5c, 0x00, 0x00, 0x00, 0x34,
  0x62, 0x54, 0x52, 0x43, 0x00, 0x00, 0x01, 0x5c, 0x00, 0x00, 0x00, 0x34,
  0x63, 0x70, 0x72, 0x74, 0x00, 0x00, 0x01, 0x90, 0x00, 0x00, 0x00, 0x0a,
  0x64, 0x65, 0x73, 0x63, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05,
  0x6e, 0x52, 0x47, 0x42, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x58, 0x59, 0x5a, 0x20, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0xf3, 0x54, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x16, 0xc9,
  0x58, 0x59, 0x5a, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6f, 0xa0,
  0x00, 0x00, 0x38, 0xf2, 0x00, 0x00, 0x03, 0x8f, 0x58, 0x59, 0x5a, 0x20,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x62, 0x96, 0x00, 0x00, 0xb7, 0x89,
  0x00, 0x00, 0x18, 0xda, 0x58, 0x59, 0x5a, 0x20, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x24, 0xa0, 0x00, 0x00, 0x0f, 0x85, 0x00, 0x00, 0xb6, 0xc4,
  0x63, 0x75, 0x72, 0x76, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x14,
  0x00, 0x00, 0x01, 0x07, 0x02, 0xb5, 0x05, 0x6b, 0x09, 0x36, 0x0e, 0x50,
  0x14, 0xb1, 0x1c, 0x80, 0x25, 0xc8, 0x30, 0xa1, 0x3d, 0x19, 0x4b, 0x40,
  0x5b, 0x27, 0x6c, 0xdb, 0x80, 0x6b, 0x95, 0xe3, 0xad, 0x50, 0xc6, 0xc2,
  0xe2, 0x31, 0xff, 0xff, 0x74, 0x65, 0x78, 0x74, 0x00, 0x00, 0x00, 0x00,
  0x30, 0x00
};

#define TEST_PROFILE(arr) ((struct pl_icc_profile) {    \
    .data = (arr),                                      \
    .len = PL_ARRAY_SIZE(arr),                          \
    .signature = (uintptr_t) (arr),                     \
})
