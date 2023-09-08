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

#include "common.h"

static inline void pl_hash_merge(uint64_t *accum, uint64_t hash) {
    *accum ^= hash + 0x9e3779b9 + (*accum << 6) + (*accum >> 2);
}

uint64_t pl_mem_hash(const void *mem, size_t size);
#define pl_var_hash(x) pl_mem_hash(&(x), sizeof(x))

static inline uint64_t pl_str_hash(pl_str str)
{
    return pl_mem_hash(str.buf, str.len);
}

static inline uint64_t pl_str0_hash(const char *str)
{
    return pl_mem_hash(str, str ? strlen(str) : 0);
}
