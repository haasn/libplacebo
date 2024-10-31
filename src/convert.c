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

#include <limits>
#include <string.h>
#include "pl_string.h"

// Function prototypes
static int ccStrPrintDouble(char *str, int bufsize, int decimals, double value);
static bool fast_float_from_chars(const char *str, double *n); // Declaration for fast_float

// Function to print a double as a string
static int ccStrPrintDouble(char *str, int bufsize, int decimals, double value) {
    int size = 0;
    int offset = 0;
    uint32_t u32;
    uint64_t u64;

    if (value < 0.0) {
        size = 1;
        *str++ = '-';
        bufsize--;
        value = -value;
    }

    if (value < 4294967296.0) {
        u32 = (uint32_t)value;
        offset = pl_str_print_uint(str, bufsize, u32);
        if (!offset) goto error;
        size += offset;
        bufsize -= size;
        value -= (double)u32;
    } else if (value < 18446744073709551616.0) {
        u64 = (uint64_t)value;
        offset = pl_str_print_uint64(str, bufsize, u64);
        if (!offset) goto error;
        size += offset;
        bufsize -= size;
        value -= (double)u64;
    } else {
        goto error;
    }

    if (decimals > bufsize - 2) decimals = bufsize - 2;
    if (decimals <= 0) return size;

    double muldec = 10.0;
    int32_t accumsub = 0;
    str += offset;

    for (int index = 0; index < decimals; index++) {
        if (value * muldec - accumsub <= std::numeric_limits<double>::epsilon()) break;
        if (index == 0) {
            size += 1;
            *str++ = '.';
        }
        int32_t frac = (int32_t)(value * muldec) - accumsub;
        frac = (frac < 0) ? 0 : (frac > 9) ? 9 : frac; // Clamp frac between 0 and 9
        str[index] = '0' + (char)frac;
        accumsub += frac;
        accumsub = (accumsub << 3) + (accumsub << 1);
        if (muldec < 10000000) {
            muldec *= 10.0;
        } else {
            value *= 10000000.0;
            value -= (int32_t)value;
            muldec = 10.0;
            accumsub = 0;
        }
    }
    if (str[index - 1] < '9' && (int32_t)(value * muldec) - accumsub >= 5) {
        str[index - 1]++;
    }
    str[index] = '\0'; // Null-terminate the string
    size += index;
    return size;

error:
    if (bufsize < 4) *str = '\0';
    else {
        str[0] = 'E';
        str[1] = 'R';
        str[2] = 'R';
        str[3] = '\0';
    }
    return 0;
}

// Function to parse from string to double using fast_float
static bool from_chars(pl_str str, double *n) {
#if __has_include(<fast_float/fast_float.h>)
    return fast_float_from_chars((const char *)str.buf, n); // Use fast_float
#else
    // Handle error when fast_float is not available
    return false;
#endif
}

// Function to print values to a string
#define CHAR_CONVERT(name, type)                         \
    int pl_str_print_##name(char *buf, size_t len, type n) { \
        if (sizeof(type) == sizeof(double)) {           \
            return ccStrPrintDouble(buf, len, 10, n);   \
        }                                               \
        return 0;                                       \
    }                                                   \
    bool pl_str_parse_##name(pl_str str, type *n) {     \
        return from_chars(str, n);                      \
    }

CHAR_CONVERT(hex, unsigned short)
CHAR_CONVERT(int, int)
CHAR_CONVERT(uint, unsigned int)
CHAR_CONVERT(int64, int64_t)
CHAR_CONVERT(uint64, uint64_t)
CHAR_CONVERT(float, float)
CHAR_CONVERT(double, double)

/* *****************************************************************************
 *
 * Copyright (c) 2007-2016 Alexis Naveros.
 * Modified for use with libplacebo by Niklas Haas
 * Changes include:
 *  - Removed a CC_MIN macro dependency by equivalent logic
 *  - Removed CC_ALWAYSINLINE
 *  - Fixed (!seq) check to (!seqlength)
 *  - Added support for scientific notation (e.g. 1.0e10) in ccSeqParseDouble
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 * claim that you wrote the original software. If you use this software
 * in a product, an acknowledgment in the product documentation would be appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 * ----------------------------------------------------------------------------- 
 */
