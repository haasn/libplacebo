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

#include <stdio.h>
#include <stdarg.h>

#define PRINTF_WRAP(name)   \
    __typeof__(name) name##_c;

// These printf wrappers (named printf_c etc.) must perform locale-invariant
// versions of their equivalent functions without the prefix. (The _c suffix
// stands for the "C" locale)
PRINTF_WRAP(printf);
PRINTF_WRAP(fprintf);
PRINTF_WRAP(sprintf);
PRINTF_WRAP(snprintf);

PRINTF_WRAP(vprintf);
PRINTF_WRAP(vfprintf);
PRINTF_WRAP(vsprintf);
PRINTF_WRAP(vsnprintf);

// Initialization/uninitialization functions. (Not thread safe!)
void printf_c_init(void);
void printf_c_uninit(void);
