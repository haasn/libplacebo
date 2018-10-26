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

    double sigma = -log(1.5 / UINT64_MAX * gauss_size2) / k->gauss_radius;

    for (index_t gy = 0; gy <= k->gauss_radius; gy++) {
        for (index_t gx = 0; gx <= gy; gx++) {
            int cx = (int)gx - k->gauss_radius;
            int cy = (int)gy - k->gauss_radius;
            int sq = cx * cx + cy * cy;
            double e = exp(-sqrt(sq) * sigma);
            uint64_t v = e / gauss_size2 * UINT64_MAX;
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
    uint64_t total = 0;
    for (index_t c = 0; c < k->size2; c++) {
        uint64_t oldtotal = total;
        total += k->gauss[c];
        assert(total >= oldtotal);
    }
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
    struct ctx *k = talloc_zero(NULL, struct ctx);
    makegauss(k, shift);
    makeuniform(k);
    float invscale = k->size2;
    for(index_t y = 0; y < k->size; y++) {
        for(index_t x = 0; x < k->size; x++)
            data[x + y * k->size] = k->unimat[XY(k, x, y)] / invscale;
    }
    talloc_free(k);
}
