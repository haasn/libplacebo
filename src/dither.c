/*
 * Generate a noise texture for dithering images.
 * Copyright © 2013  Wessel Dankers <wsl@fruit.je>
 *
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
 *
 * The original code is taken from mpv, under the same license.
 */


#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "common.h"

void pl_generate_bayer_matrix(float *data, int size)
{
    pl_assert(size >= 0);

    // Start with a single entry of 0
    data[0] = 0;

    for (int sz = 1; sz < size; sz *= 2) {
        // Make three copies of the current, appropriately shifted and scaled
        for (int y = 0; y < sz; y ++) {
            for (int x = 0; x < sz; x++) {
                int offsets[] = {0, sz * size + sz, sz, sz * size};
                int pos = y * size + x;

                for (int i = 1; i < 4; i++)
                    data[pos + offsets[i]] = data[pos] + i / (4.0 * sz * sz);
            }
        }
    }
}

#define MAX_SIZEB 8
#define MAX_SIZE (1 << MAX_SIZEB)
#define MAX_SIZE2 (MAX_SIZE * MAX_SIZE)

typedef uint_fast32_t index_t;

#define WRAP_SIZE2(k, x) ((index_t)((index_t)(x) & ((k)->size2 - 1)))
#define XY(k, x, y) ((index_t)(((x) | ((y) << (k)->sizeb))))

struct ctx {
    unsigned int sizeb, size, size2;
    unsigned int gauss_radius;
    unsigned int gauss_middle;
    uint64_t gauss[MAX_SIZE2];
    index_t randomat[MAX_SIZE2];
    bool calcmat[MAX_SIZE2];
    uint64_t gaussmat[MAX_SIZE2];
    index_t unimat[MAX_SIZE2];
};

static void makegauss(struct ctx *k, unsigned int sizeb)
{
    pl_assert(sizeb >= 1 && sizeb <= MAX_SIZEB);

    k->sizeb = sizeb;
    k->size = 1 << k->sizeb;
    k->size2 = k->size * k->size;

    k->gauss_radius = k->size / 2 - 1;
    k->gauss_middle = XY(k, k->gauss_radius, k->gauss_radius);

    unsigned int gauss_size = k->gauss_radius * 2 + 1;
    unsigned int gauss_size2 = gauss_size * gauss_size;

    for (index_t c = 0; c < k->size2; c++)
        k->gauss[c] = 0;

    double sigma = -log(1.5 / (double) UINT64_MAX * gauss_size2) / k->gauss_radius;

    for (index_t gy = 0; gy <= k->gauss_radius; gy++) {
        for (index_t gx = 0; gx <= gy; gx++) {
            int cx = (int)gx - k->gauss_radius;
            int cy = (int)gy - k->gauss_radius;
            int sq = cx * cx + cy * cy;
            double e = exp(-sqrt(sq) * sigma);
            uint64_t v = e / gauss_size2 * (double) UINT64_MAX;
            k->gauss[XY(k, gx, gy)] =
                k->gauss[XY(k, gy, gx)] =
                k->gauss[XY(k, gx, gauss_size - 1 - gy)] =
                k->gauss[XY(k, gy, gauss_size - 1 - gx)] =
                k->gauss[XY(k, gauss_size - 1 - gx, gy)] =
                k->gauss[XY(k, gauss_size - 1 - gy, gx)] =
                k->gauss[XY(k, gauss_size - 1 - gx, gauss_size - 1 - gy)] =
                k->gauss[XY(k, gauss_size - 1 - gy, gauss_size - 1 - gx)] = v;
        }
    }

#ifndef NDEBUG
    uint64_t total = 0;
    for (index_t c = 0; c < k->size2; c++) {
        uint64_t oldtotal = total;
        total += k->gauss[c];
        assert(total >= oldtotal);
    }
#endif
}

static void setbit(struct ctx *k, index_t c)
{
    if (k->calcmat[c])
        return;
    k->calcmat[c] = true;
    uint64_t *m = k->gaussmat;
    uint64_t *me = k->gaussmat + k->size2;
    uint64_t *g = k->gauss + WRAP_SIZE2(k, k->gauss_middle + k->size2 - c);
    uint64_t *ge = k->gauss + k->size2;
    while (g < ge)
        *m++ += *g++;
    g = k->gauss;
    while (m < me)
        *m++ += *g++;
}

static index_t getmin(struct ctx *k)
{
    uint64_t min = UINT64_MAX;
    index_t resnum = 0;
    unsigned int size2 = k->size2;
    for (index_t c = 0; c < size2; c++) {
        if (k->calcmat[c])
            continue;
        uint64_t total = k->gaussmat[c];
        if (total <= min) {
            if (total != min) {
                min = total;
                resnum = 0;
            }
            k->randomat[resnum++] = c;
        }
    }
    assert(resnum > 0);
    if (resnum == 1)
        return k->randomat[0];
    if (resnum == size2)
        return size2 / 2;
    return k->randomat[rand() % resnum];
}

static void makeuniform(struct ctx *k)
{
    unsigned int size2 = k->size2;
    for (index_t c = 0; c < size2; c++) {
        index_t r = getmin(k);
        setbit(k, r);
        k->unimat[r] = c;
    }
}

void pl_generate_blue_noise(float *data, int size)
{
    pl_assert(size > 0);
    int shift = PL_LOG2(size);

    pl_assert((1 << shift) == size);
    struct ctx *k = pl_zalloc_ptr(NULL, k);
    makegauss(k, shift);
    makeuniform(k);
    float invscale = k->size2;
    for(index_t y = 0; y < k->size; y++) {
        for(index_t x = 0; x < k->size; x++)
            data[x + y * k->size] = k->unimat[XY(k, x, y)] / invscale;
    }
    pl_free(k);
}

const struct pl_error_diffusion_kernel pl_error_diffusion_simple = {
    .name = "simple",
    .description = "Simple error diffusion",
    .shift = 1,
    .pattern = {{0, 0, 0, 1, 0},
                {0, 0, 1, 0, 0},
                {0, 0, 0, 0, 0}},
    .divisor = 2,
};

const struct pl_error_diffusion_kernel pl_error_diffusion_false_fs = {
    .name = "false-fs",
    .description = "False Floyd-Steinberg kernel",
    .shift = 1,
    .pattern = {{0, 0, 0, 3, 0},
                {0, 0, 3, 2, 0},
                {0, 0, 0, 0, 0}},
    .divisor = 8,
};

const struct pl_error_diffusion_kernel pl_error_diffusion_sierra_lite = {
    .name = "sierra-lite",
    .description = "Sierra Lite kernel",
    .shift = 2,
    .pattern = {{0, 0, 0, 2, 0},
                {0, 1, 1, 0, 0},
                {0, 0, 0, 0, 0}},
    .divisor = 4,
};

const struct pl_error_diffusion_kernel pl_error_diffusion_floyd_steinberg = {
    .name = "floyd-steinberg",
    .description = "Floyd Steinberg kernel",
    .shift = 2,
    .pattern = {{0, 0, 0, 7, 0},
                {0, 3, 5, 1, 0},
                {0, 0, 0, 0, 0}},
    .divisor = 16,
};

const struct pl_error_diffusion_kernel pl_error_diffusion_atkinson = {
    .name = "atkinson",
    .description = "Atkinson kernel",
    .shift = 2,
    .pattern = {{0, 0, 0, 1, 1},
                {0, 1, 1, 1, 0},
                {0, 0, 1, 0, 0}},
    .divisor = 8,
};

const struct pl_error_diffusion_kernel pl_error_diffusion_jarvis_judice_ninke = {
    .name = "jarvis-judice-ninke",
    .description = "Jarvis, Judice & Ninke kernel",
    .shift = 3,
    .pattern = {{0, 0, 0, 7, 5},
                {3, 5, 7, 5, 3},
                {1, 3, 5, 3, 1}},
    .divisor = 48,
};

const struct pl_error_diffusion_kernel pl_error_diffusion_stucki = {
    .name = "stucki",
    .description = "Stucki kernel",
    .shift = 3,
    .pattern = {{0, 0, 0, 8, 4},
                {2, 4, 8, 4, 2},
                {1, 2, 4, 2, 1}},
    .divisor = 42,
};

const struct pl_error_diffusion_kernel pl_error_diffusion_burkes = {
    .name = "burkes",
    .description = "Burkes kernel",
    .shift = 3,
    .pattern = {{0, 0, 0, 8, 4},
                {2, 4, 8, 4, 2},
                {0, 0, 0, 0, 0}},
    .divisor = 32,
};

const struct pl_error_diffusion_kernel pl_error_diffusion_sierra2 = {
    .name = "sierra-2",
    .description = "Two-row Sierra",
    .shift = 3,
    .pattern = {{0, 0, 0, 4, 3},
                {1, 2, 3, 2, 1},
                {0, 0, 0, 0, 0}},
    .divisor = 16,
};

const struct pl_error_diffusion_kernel pl_error_diffusion_sierra3 = {
    .name = "sierra-3",
    .description = "Three-row Sierra",
    .shift = 3,
    .pattern = {{0, 0, 0, 5, 3},
                {2, 4, 5, 4, 2},
                {0, 2, 3, 2, 0}},
    .divisor = 32,
};

const struct pl_error_diffusion_kernel * const pl_error_diffusion_kernels[] = {
    &pl_error_diffusion_simple,
    &pl_error_diffusion_false_fs,
    &pl_error_diffusion_sierra_lite,
    &pl_error_diffusion_floyd_steinberg,
    &pl_error_diffusion_atkinson,
    &pl_error_diffusion_jarvis_judice_ninke,
    &pl_error_diffusion_stucki,
    &pl_error_diffusion_burkes,
    &pl_error_diffusion_sierra2,
    &pl_error_diffusion_sierra3,
    NULL
};

const int pl_num_error_diffusion_kernels = PL_ARRAY_SIZE(pl_error_diffusion_kernels) - 1;

// Find the error diffusion kernel with the given name, or NULL on failure.
const struct pl_error_diffusion_kernel *pl_find_error_diffusion_kernel(const char *name)
{
    for (int i = 0; i < pl_num_error_diffusion_kernels; i++) {
        if (strcmp(name, pl_error_diffusion_kernels[i]->name) == 0)
            return pl_error_diffusion_kernels[i];
    }

    return NULL;
}
