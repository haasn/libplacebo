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

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

#include "ta/talloc.h"
#include "config.h"

// Include all of the symbols that should be public in a way that marks them
// as being externally visible. (Otherwise, all symbols are hidden by default)
#pragma GCC visibility push(default)

#include "public/colorspace.h"
#include "public/common.h"
#include "public/context.h"
#include "public/filters.h"
#include "public/ra.h"
#include "public/shaders.h"

#pragma GCC visibility pop

// Align up to the nearest multiple of an arbitrary alignment, which may also
// be 0 to signal no alignment requirements.
#define PL_ALIGN(x, align) ((align) ? ((x) + (align) - 1) / (align) * (align) : (x))

// This is faster but must only be called on positive powers of two.
#define PL_ALIGN2(x, align) (((x) + (align) - 1) & ~((align) - 1))

// Returns the size of a static array with known size.
#define PL_ARRAY_SIZE(s) (sizeof(s) / sizeof((s)[0]))

// Swaps two variables
#define PL_SWAP(a, b)           \
    do {                        \
        __typeof__ tmp = (a);   \
        (a) = (b);              \
        (b) = tmp;              \
    } while (0)
