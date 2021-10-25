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
#include <assert.h>

#ifndef NDEBUG
# define pl_assert assert
#else
# define pl_assert(expr)                                        \
  do {                                                          \
      if (!(expr)) {                                            \
          fprintf(stderr, "Assertion failed: %s in %s:%d\n",    \
                  #expr, __FILE__, __LINE__);                   \
          abort();                                              \
      }                                                         \
  } while (0)
#endif

// In C11, static asserts must have a string message
#define pl_static_assert(expr) static_assert(expr, #expr)
