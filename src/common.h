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

#define __STDC_FORMAT_MACROS
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <inttypes.h>

#if defined(__MINGW32__) && !defined(__clang__)
#define PL_PRINTF(fmt, va) __attribute__ ((format(gnu_printf, fmt, va))) \
                           __attribute__ ((nonnull))
#elif defined(__GNUC__)
#define PL_PRINTF(fmt, va) __attribute__ ((format(printf, fmt, va))) \
                           __attribute__ ((nonnull))
#else
#define PL_PRINTF(fmt, va)
#endif

#define PL_NOINLINE __attribute__((noinline))

#include "os.h"

#include "config_internal.h"
#include "pl_assert.h"
#include "pl_alloc.h"
#include "pl_string.h"

#define PL_DEPRECATED

#include <libplacebo/config.h>

#if PL_API_VER != BUILD_API_VER
#error Header mismatch? <libplacebo/config.h> pulled from elsewhere!
#endif

// Divide a number while rounding up (careful: double-eval)
#define PL_DIV_UP(x, y) (((x) + (y) - 1) / (y))

// Align up to the nearest multiple of an arbitrary alignment, which may also
// be 0 to signal no alignment requirements.
#define PL_ALIGN(x, align) ((align) ? PL_DIV_UP(x, align) * (align) : (x))

// This is faster but must only be called on positive powers of two.
#define PL_ALIGN2(x, align) (((x) + (align) - 1) & ~((align) - 1))

// Returns the log base 2 of an unsigned long long
#define PL_LOG2(x) ((unsigned) (8*sizeof (unsigned long long) - __builtin_clzll((x)) - 1))

// Rounds a number up to the nearest power of two
#define PL_ALIGN_POT(x) (0x1LLU << (PL_LOG2((x) - 1) + 1))

// Right shift a number while rounding up
#define PL_RSHIFT_UP(x, s) -((-(x)) >> (s))

// Returns whether or not a number is a power of two (or zero)
#define PL_ISPOT(x) (((x) & ((x) - 1)) == 0)

// Returns the size of a static array with known size.
#define PL_ARRAY_SIZE(s) (sizeof(s) / sizeof((s)[0]))

// Swaps two variables
#define PL_SWAP(a, b)              \
    do {                           \
        __typeof__ (a) _tmp = (a); \
        (a) = (b);                 \
        (b) = _tmp;                \
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
#define PL_MAX3(x, y, z) PL_MAX(PL_MAX(x, y), z)
#define PL_MIN(x, y) ((x) < (y) ? (x) : (y))
#define PL_CLAMP(x, l, h) ((x) < (l) ? (l) : (x) > (h) ? (h) : (x))
#define PL_CMP(a, b) (((a) > (b)) - ((a) < (b)))
#define PL_DEF(x, d) ((x) ? (x) : (d))
#define PL_SQUARE(x) ((x) * (x))
#define PL_CUBE(x) ((x) * (x) * (x))
#define PL_MIX(a, b, x) ((x) * (b) + (1 - (x)) * (a))

static inline float pl_smoothstep(float edge0, float edge1, float x)
{
    x = (x - edge0) / (edge1 - edge0);
    x = PL_CLAMP(x, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

// Helpers for doing alignment calculations
static inline size_t pl_gcd(size_t x, size_t y)
{
    assert(x && y);
    while (y) {
        size_t tmp = y;
        y = x % y;
        x = tmp;
    }

    return x;
}

static inline size_t pl_lcm(size_t x, size_t y)
{
    assert(x && y);
    return x * (y / pl_gcd(x, y));
}

// Conditional abort() macro that depends on the configuration option
#ifdef PL_DEBUG_ABORT
# define pl_debug_abort() do {                          \
    fprintf(stderr, "pl_debug_abort() triggered!\n");   \
    abort();                                            \
} while (0)
#else
# define pl_debug_abort() do {} while (0)
#endif

// Refcounting helpers
typedef _Atomic uint32_t pl_rc_t;
#define pl_rc_init(rc)  atomic_init(rc, 1)
#define pl_rc_ref(rc)   ((void) atomic_fetch_add_explicit(rc, 1, memory_order_acquire))
#define pl_rc_deref(rc) (atomic_fetch_sub_explicit(rc, 1, memory_order_release) == 1)
#define pl_rc_count(rc)  atomic_load(rc)

#define pl_unreachable() (assert(!"unreachable"), __builtin_unreachable())

// Helper for parameter validation
#define pl_require(ctx, expr)                                   \
  do {                                                          \
      if (!(expr)) {                                            \
          PL_ERR(ctx, "Validation failed: %s (%s:%d)",          \
                  #expr, __FILE__, __LINE__);                   \
          pl_log_stack_trace(ctx->log, PL_LOG_ERR);             \
          pl_debug_abort();                                     \
          goto error;                                           \
      }                                                         \
  } while (0)
