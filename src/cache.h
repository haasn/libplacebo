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

#include <libplacebo/cache.h>

// Convenience wrapper around pl_cache_set
static inline void pl_cache_str(pl_cache cache, uint64_t key, pl_str *str)
{
    pl_cache_set(cache, &(pl_cache_obj) {
        .key  = key,
        .data = pl_steal(NULL, str->buf),
        .size = str->len,
        .free = pl_free,
    });
    *str = (pl_str) {0};
}

// Steal and insert a cache object
static inline void pl_cache_steal(pl_cache cache, pl_cache_obj *obj)
{
    pl_assert(!obj->data || obj->free == pl_free);
    obj->data = pl_steal(NULL, obj->data);
    pl_cache_set(cache, obj);
}

// Resize `obj->data` to a given size, re-using allocated buffers where possible
static inline void pl_cache_obj_resize(void *alloc, pl_cache_obj *obj, size_t size)
{
    if (obj->free != pl_free) {
        if (obj->free)
            obj->free(obj->data);
        obj->data = pl_alloc(alloc, size);
        obj->free = pl_free;
    } else if (pl_get_size(obj->data) < size) {
        obj->data = pl_steal(alloc, obj->data);
        obj->data = pl_realloc(alloc, obj->data, size);
    }
    obj->size = size;
}

// Internal list of base seeds for different object types, randomly generated

enum {
    CACHE_KEY_SH_LUT    = UINT64_C(0x2206183d320352c6), // sh_lut cache
    CACHE_KEY_ICC_3DLUT = UINT64_C(0xff703a6dd8a996f6), // ICC 3dlut
    CACHE_KEY_DITHER    = UINT64_C(0x6fed75eb6dce86cb), // dither matrix
    CACHE_KEY_H274      = UINT64_C(0x2fb9adca04b42c4d), // H.274 film grain DB
    CACHE_KEY_GAMUT_LUT = UINT64_C(0x41bbe0c35ea24b2e), // gamut mapping 3DLUT
    CACHE_KEY_SPIRV     = UINT64_C(0x32352f6605ff60a7), // bare SPIR-V module
    CACHE_KEY_VK_PIPE   = UINT64_C(0x4bdab2817ad02ad4), // VkPipelineCache
};
