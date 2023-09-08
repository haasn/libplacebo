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

static inline uint64_t pl_mem_hash(const void *mem, size_t size);
#define pl_var_hash(x) pl_mem_hash(&(x), sizeof(x))

static inline uint64_t pl_str_hash(pl_str str)
{
    return pl_mem_hash(str.buf, str.len);
}

static inline uint64_t pl_str0_hash(const char *str)
{
    return pl_mem_hash(str, str ? strlen(str) : 0);
}

/*
   SipHash reference C implementation
   Modified for use by libplacebo:
    - Hard-coded a fixed key (k0 and k1)
    - Hard-coded the output size to 64 bits
    - Return the result vector directly

   Copyright (c) 2012-2016 Jean-Philippe Aumasson
   <jeanphilippe.aumasson@gmail.com>
   Copyright (c) 2012-2014 Daniel J. Bernstein <djb@cr.yp.to>

   To the extent possible under law, the author(s) have dedicated all copyright
   and related and neighboring rights to this software to the public domain
   worldwide. This software is distributed without any warranty.

   <http://creativecommons.org/publicdomain/zero/1.0/>.
 */

/* default: SipHash-2-4 */
#define cROUNDS 2
#define dROUNDS 4

#define ROTL(x, b) (uint64_t)(((x) << (b)) | ((x) >> (64 - (b))))

#define U8TO64_LE(p)                                                           \
    (((uint64_t)((p)[0])) | ((uint64_t)((p)[1]) << 8) |                        \
     ((uint64_t)((p)[2]) << 16) | ((uint64_t)((p)[3]) << 24) |                 \
     ((uint64_t)((p)[4]) << 32) | ((uint64_t)((p)[5]) << 40) |                 \
     ((uint64_t)((p)[6]) << 48) | ((uint64_t)((p)[7]) << 56))

#define SIPROUND                                                               \
    do {                                                                       \
        v0 += v1;                                                              \
        v1 = ROTL(v1, 13);                                                     \
        v1 ^= v0;                                                              \
        v0 = ROTL(v0, 32);                                                     \
        v2 += v3;                                                              \
        v3 = ROTL(v3, 16);                                                     \
        v3 ^= v2;                                                              \
        v0 += v3;                                                              \
        v3 = ROTL(v3, 21);                                                     \
        v3 ^= v0;                                                              \
        v2 += v1;                                                              \
        v1 = ROTL(v1, 17);                                                     \
        v1 ^= v2;                                                              \
        v2 = ROTL(v2, 32);                                                     \
    } while (0)

static inline uint64_t pl_mem_hash(const void *mem, size_t size)
{
    if (!size)
        return 0x8533321381b8254bULL;

    uint64_t v0 = 0x736f6d6570736575ULL;
    uint64_t v1 = 0x646f72616e646f6dULL;
    uint64_t v2 = 0x6c7967656e657261ULL;
    uint64_t v3 = 0x7465646279746573ULL;
    uint64_t k0 = 0xfe9f075098ddb0faULL;
    uint64_t k1 = 0x68f7f03510e5285cULL;
    uint64_t m;
    int i;
    const uint8_t *buf = mem;
    const uint8_t *end = buf + size - (size % sizeof(uint64_t));
    const int left = size & 7;
    uint64_t b = ((uint64_t) size) << 56;
    v3 ^= k1;
    v2 ^= k0;
    v1 ^= k1;
    v0 ^= k0;

    for (; buf != end; buf += 8) {
        m = U8TO64_LE(buf);
        v3 ^= m;

        for (i = 0; i < cROUNDS; ++i)
            SIPROUND;

        v0 ^= m;
    }

    switch (left) {
    case 7: b |= ((uint64_t) buf[6]) << 48; // fall through
    case 6: b |= ((uint64_t) buf[5]) << 40; // fall through
    case 5: b |= ((uint64_t) buf[4]) << 32; // fall through
    case 4: b |= ((uint64_t) buf[3]) << 24; // fall through
    case 3: b |= ((uint64_t) buf[2]) << 16; // fall through
    case 2: b |= ((uint64_t) buf[1]) << 8;  // fall through
    case 1: b |= ((uint64_t) buf[0]); break;
    case 0: break;
    }

    v3 ^= b;

    for (i = 0; i < cROUNDS; ++i)
        SIPROUND;

    v0 ^= b;

    v2 ^= 0xff;

    for (i = 0; i < dROUNDS; ++i)
        SIPROUND;

    b = v0 ^ v1 ^ v2 ^ v3;
    return b;
}
