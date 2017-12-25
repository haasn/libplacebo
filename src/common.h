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

#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

#include "ta/talloc.h"
#include "osdep/printf.h"
#include "config.h"
#include "pl_assert.h"

// Include all of the symbols that should be public in a way that marks them
// as being externally visible. (Otherwise, all symbols are hidden by default)
#pragma GCC visibility push(default)

#include "include/libplacebo/colorspace.h"
#include "include/libplacebo/common.h"
#include "include/libplacebo/context.h"
#include "include/libplacebo/dispatch.h"
#include "include/libplacebo/dither.h"
#include "include/libplacebo/filters.h"
#include "include/libplacebo/ra.h"
#include "include/libplacebo/renderer.h"
#include "include/libplacebo/shaders.h"
#include "include/libplacebo/shaders/colorspace.h"
#include "include/libplacebo/shaders/sampling.h"

#if PL_HAVE_VULKAN
#include "include/libplacebo/vulkan.h"
#endif

#pragma GCC visibility pop

// Align up to the nearest multiple of an arbitrary alignment, which may also
// be 0 to signal no alignment requirements.
#define PL_ALIGN(x, align) ((align) ? ((x) + (align) - 1) / (align) * (align) : (x))

// This is faster but must only be called on positive powers of two.
#define PL_ALIGN2(x, align) (((x) + (align) - 1) & ~((align) - 1))

// Returns the log base 2 of an unsigned long long
#define PL_LOG2(x) ((unsigned) (8*sizeof (unsigned long long) - __builtin_clzll((x)) - 1))

// Returns the size of a static array with known size.
#define PL_ARRAY_SIZE(s) (sizeof(s) / sizeof((s)[0]))

// Swaps two variables
#define PL_SWAP(a, b)           \
    do {                        \
        __typeof__ tmp = (a);   \
        (a) = (b);              \
        (b) = tmp;              \
    } while (0)

// Helper functions for transposing a matrix in-place.
#define PL_TRANSPOSE_DIM(d, m) \
    pl_transpose((d), (float[(d)*(d)]){0}, (const float *)(m))

#define PL_TRANSPOSE_2X2(m) PL_TRANSPOSE_DIM(2, m)
#define PL_TRANSPOSE_3X3(m) PL_TRANSPOSE_DIM(3, m)
#define PL_TRANSPOSE_4X4(m) PL_TRANSPOSE_DIM(4, m)

static inline float *pl_transpose(int dim, float *out, const float *in)
{
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++)
            out[i * dim + j] = in[j * dim + i];
    }

    return out;
}

// Helper functions for some common numeric operations (careful: double-eval)
#define PL_MAX(x, y) ((x) > (y) ? (x) : (y))
#define PL_MIN(x, y) ((x) < (y) ? (x) : (y))
#define PL_CMP(a, b) ((a) < (b) ? -1 : (a) > (b) ? 1 : 0)
#define PL_DEF(x, d) ((x) ? (x) : (d))
