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

#include <stdalign.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

// Unlike standard malloc, `size` may be 0, in which case this returns an empty
// allocation which can still be used as a parent for other allocations.
void *pl_alloc(void *parent, size_t size);
void *pl_zalloc(void *parent, size_t size);
void *pl_realloc(void *parent, void *ptr, size_t size);

static inline void *pl_calloc(void *parent, size_t count, size_t size)
{
    return pl_zalloc(parent, count * size);
}

#define pl_tmp(parent) pl_alloc(parent, 0)

// Variants of the above which resolve to sizeof(*ptr)
#define pl_alloc_ptr(parent, ptr) \
    (__typeof__(ptr)) pl_alloc(parent, sizeof(*(ptr)))
#define pl_zalloc_ptr(parent, ptr) \
    (__typeof__(ptr)) pl_zalloc(parent, sizeof(*(ptr)))
#define pl_calloc_ptr(parent, num, ptr) \
    (__typeof__(ptr)) pl_calloc(parent, num, sizeof(*(ptr)))

// Helper function to allocate a struct and immediately assign it
#define pl_alloc_struct(parent, type, ...) \
    (type *) pl_memdup(parent, &(type) __VA_ARGS__, sizeof(type))

// Free an allocation and its children (recursively)
void pl_free(void *ptr);
void pl_free_children(void *ptr);

#define pl_free_ptr(ptr)    \
    do {                    \
        pl_free(*(ptr));    \
        *(ptr) = NULL;      \
    } while (0)

// Get the current size of an allocation.
size_t pl_get_size(void *ptr);

#define pl_grow(parent, ptr, size)                      \
    do {                                                \
        size_t _size = (size);                          \
        if (_size > pl_get_size(*(ptr)))                \
            *(ptr) = pl_realloc(parent, *(ptr), _size); \
    } while (0)

// Reparent an allocation onto a new parent
void *pl_steal(void *parent, void *ptr);

// Wrapper functions around common string utilities
void *pl_memdup(void *parent, const void *ptr, size_t size);
char *pl_str0dup0(void *parent, const char *str);
char *pl_strndup0(void *parent, const char *str, size_t size);

#define pl_memdup_ptr(parent, ptr) \
    (__typeof__(ptr)) pl_memdup(parent, ptr, sizeof(*(ptr)))

// Helper functions for allocating public/private pairs, done by allocating
// `priv` at the address of `pub` + sizeof(pub), rounded up to the maximum
// alignment requirements.

#define PL_ALIGN_MEM(size) \
    (((size) + alignof(max_align_t) - 1) & ~(alignof(max_align_t) - 1))

#define PL_PRIV(pub) \
    (void *) ((uintptr_t) (pub) + PL_ALIGN_MEM(sizeof(*(pub))))

#define pl_alloc_obj(parent, ptr, priv) \
    (__typeof__(ptr)) pl_alloc(parent, PL_ALIGN_MEM(sizeof(*(ptr))) + sizeof(priv))

#define pl_zalloc_obj(parent, ptr, priv) \
    (__typeof__(ptr)) pl_zalloc(parent, PL_ALIGN_MEM(sizeof(*(ptr))) + sizeof(priv))

// Refcounting helper

struct pl_ref;

// pl_ref_deref will free the ref and all of its children as soon as the
// internal refcount reaches 0
struct pl_ref *pl_ref_new(void *parent);
struct pl_ref *pl_ref_dup(struct pl_ref *ref);
void pl_ref_deref(struct pl_ref **ref);

// Helper functions for dealing with arrays

#define PL_ARRAY(type) struct { type *elem; int num; }

#define PL_ARRAY_RESIZE(parent, arr, len)                                       \
    do {                                                                        \
        size_t _new_size = (len) * sizeof((arr).elem[0]);                       \
        (arr).elem = pl_realloc((void *) parent, (arr).elem, _new_size);        \
    } while (0)

#define PL_ARRAY_GROW(parent, arr)                                              \
    do {                                                                        \
        size_t _avail = pl_get_size((arr).elem) / sizeof((arr).elem[0]);        \
        if (_avail < 10) {                                                      \
            PL_ARRAY_RESIZE(parent, arr, 10);                                   \
        } else if ((arr).num == _avail) {                                       \
            PL_ARRAY_RESIZE(parent, arr, (arr).num * 1.5);                      \
        } else {                                                                \
            assert((arr).elem);                                                 \
        }                                                                       \
    } while (0)

#define PL_ARRAY_APPEND(parent, arr, ...)                                       \
    do {                                                                        \
        PL_ARRAY_GROW(parent, arr);                                             \
        (arr).elem[(arr).num++] = __VA_ARGS__;                                  \
    } while (0)

#define PL_ARRAY_CONCAT(parent, to, from)                                       \
    do {                                                                        \
        if ((from).num) {                                                       \
            PL_ARRAY_RESIZE(parent, to, (to).num + (from).num);                 \
            memmove(&(to).elem[(to).num], (from).elem,                          \
                    (from).num * sizeof((from).elem[0]));                       \
            (to).num += (from).num;                                             \
        }                                                                       \
    } while (0)

#define PL_ARRAY_REMOVE_RANGE(arr, idx, count)                                  \
    do {                                                                        \
        size_t _idx = (idx);                                                    \
        size_t _count = (count);                                                \
        assert(_idx + _count <= (arr).num);                                     \
        memmove(&(arr).elem[_idx], &(arr).elem[_idx + _count],                  \
                ((arr).num - _idx - _count) * sizeof((arr).elem[0]));           \
        (arr).num -= _count;                                                    \
    } while (0)

#define PL_ARRAY_REMOVE_AT(arr, idx) PL_ARRAY_REMOVE_RANGE(arr, idx, 1)

#define PL_ARRAY_INSERT_AT(parent, arr, idx, ...)                               \
    do {                                                                        \
        size_t _idx = (idx);                                                    \
        assert(_idx < (arr).num);                                               \
        PL_ARRAY_GROW(parent, arr);                                             \
        memmove(&(arr).elem[_idx + 1], &(arr).elem[_idx],                       \
                ((arr).num++ - _idx) * sizeof((arr).elem[0]));                  \
        (arr).elem[_idx] = __VA_ARGS__;                                         \
    } while (0)

// Returns whether or not there was any element to pop
#define PL_ARRAY_POP(arr, out)                                                  \
    ((arr).num > 0                                                              \
        ? (*(out) = (arr).elem[--(arr).num], true)                              \
        : false                                                                 \
    )

// Wrapper for dealing with non-PL_ARRAY arrays
#define PL_ARRAY_APPEND_RAW(parent, arr, idxvar, ...)                           \
    do {                                                                        \
        PL_ARRAY(__typeof__((arr)[0])) _arr = { (arr), (idxvar) };              \
        PL_ARRAY_APPEND(parent, _arr, __VA_ARGS__);                             \
        (arr) = _arr.elem;                                                      \
        (idxvar) = _arr.num;                                                    \
    } while (0)
