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

#include "common.h"

struct header {
#ifndef NDEBUG
#define MAGIC 0x20210119LU
    uint32_t magic;
#endif
    size_t size;
    struct header *parent;
    struct ext *ext;

    // Pointer to actual data, for alignment purposes
    intmax_t data[1];
};

// Lazily allocated, to save space for leaf allocations and allocations which
// don't need fancy requirements
struct ext {
    size_t num_children;
    size_t children_size; // total allocated size of `children`
    struct header *children[];
};

#define PTR_OFFSET offsetof(struct header, data)
#define MAX_ALLOC (SIZE_MAX - PTR_OFFSET)
#define MINIMUM_CHILDREN 4

static inline struct header *get_header(void *ptr)
{
    if (!ptr)
        return NULL;

    struct header *hdr = (struct header *) ((uintptr_t) ptr - PTR_OFFSET);
#ifndef NDEBUG
    assert(hdr->magic == MAGIC);
#endif

    return hdr;
}

static inline void *oom(void)
{
    fprintf(stderr, "out of memory\n");
    abort();
}

static inline struct ext *alloc_ext(struct header *h)
{
    if (!h)
        return NULL;

    if (!h->ext) {
        h->ext = malloc(sizeof(struct ext) + MINIMUM_CHILDREN * sizeof(void *));
        if (!h->ext)
            oom();
        h->ext->num_children = 0;
        h->ext->children_size = MINIMUM_CHILDREN;
    }

    return h->ext;
}

static inline void attach_child(struct header *parent, struct header *child)
{
    child->parent = parent;
    if (!parent)
        return;


    struct ext *ext = alloc_ext(parent);
    if (ext->num_children == ext->children_size) {
        size_t new_size = ext->children_size * 2;
        ext = realloc(ext, sizeof(struct ext) + new_size * sizeof(void *));
        if (!ext)
            oom();
        ext->children_size = new_size;
        parent->ext = ext;
    }

    ext->children[ext->num_children++] = child;
}

static inline void unlink_child(struct header *parent, struct header *child)
{
    child->parent = NULL;
    if (!parent)
        return;

    struct ext *ext = parent->ext;
    for (size_t i = 0; i < ext->num_children; i++) {
        if (ext->children[i] == child) {
            memmove(&ext->children[i], &ext->children[i + 1],
                    (--ext->num_children - i) * sizeof(ext->children[0]));
            return;
        }
    }

    assert(!"unlinking orphaned child?");
}

void *pl_alloc(void *parent, size_t size)
{
    if (size >= MAX_ALLOC)
        return oom();

    struct header *h = malloc(PTR_OFFSET + size);
    if (!h)
        return oom();

#ifndef NDEBUG
    h->magic = MAGIC;
#endif
    h->size = size;
    h->ext = NULL;

    attach_child(get_header(parent), h);
    return h->data;
}

void *pl_zalloc(void *parent, size_t size)
{
    if (size >= MAX_ALLOC)
        return oom();

    struct header *h = calloc(1, PTR_OFFSET + size);
    if (!h)
        return oom();

#ifndef NDEBUG
    h->magic = MAGIC;
#endif
    h->size = size;

    attach_child(get_header(parent), h);
    return h->data;
}

void *pl_realloc(void *parent, void *ptr, size_t size)
{
    if (size >= MAX_ALLOC)
        return oom();
    if (!ptr)
        return pl_alloc(parent, size);

    struct header *h = get_header(ptr);
    assert(get_header(parent) == h->parent);
    if (h->size == size)
        return ptr;

    struct header *old_h = h;
    h = realloc(h, PTR_OFFSET + size);
    if (!h)
        return oom();

    h->size = size;

    if (h != old_h) {
        if (h->parent) {
            struct ext *ext = h->parent->ext;
            for (size_t i = 0; i < ext->num_children; i++) {
                if (ext->children[i] == old_h) {
                    ext->children[i] = h;
                    goto done_reparenting;
                }
            }
            assert(!"reallocating orphaned child?");
        }
done_reparenting:

        if (h->ext) {
            for (size_t i = 0; i < h->ext->num_children; i++)
                h->ext->children[i]->parent = h;
        }
    }

    return h->data;
}

void pl_free(void *ptr)
{
    struct header *h = get_header(ptr);
    if (!h)
        return;

    pl_free_children(ptr);
    unlink_child(h->parent, h);

    free(h->ext);
    free(h);
}

void pl_free_children(void *ptr)
{
    struct header *h = get_header(ptr);
    if (!h || !h->ext)
        return;

#ifndef NDEBUG
    // this detects recursive hierarchies
    h->magic = 0;
#endif

    for (size_t i = 0; i < h->ext->num_children; i++) {
        h->ext->children[i]->parent = NULL; // prevent recursive access
        pl_free(h->ext->children[i]->data);
    }

#ifndef NDEBUG
    h->magic = MAGIC;
#endif
}

size_t pl_get_size(void *ptr)
{
    struct header *h = get_header(ptr);
    return h ? h->size : 0;
}

void *pl_steal(void *parent, void *ptr)
{
    struct header *h = get_header(ptr);
    if (!h)
        return NULL;

    struct header *new_par = get_header(parent);
    if (new_par != h->parent) {
        unlink_child(h->parent, h);
        attach_child(new_par, h);
    }

    return h->data;
}

void *pl_memdup(void *parent, const void *ptr, size_t size)
{
    if (!size)
        return NULL;

    void *new = pl_alloc(parent, size);
    if (!new)
        return oom();

    assert(ptr);
    memcpy(new, ptr, size);
    return new;
}

char *pl_str0dup0(void *parent, const char *str)
{
    if (!str)
        return NULL;

    return pl_memdup(parent, str, strlen(str) + 1);
}

char *pl_strndup0(void *parent, const char *str, size_t size)
{
    if (!str)
        return NULL;

    size_t str_size = strnlen(str, size);
    char *new = pl_alloc(parent, str_size + 1);
    if (!new)
        return oom();
    memcpy(new, str, str_size);
    new[str_size] = '\0';
    return new;
}

struct pl_ref {
    pl_rc_t rc;
};

struct pl_ref *pl_ref_new(void *parent)
{
    struct pl_ref *ref = pl_zalloc_ptr(parent, ref);
    if (!ref)
        return oom();

    pl_rc_init(&ref->rc);
    return ref;
}

struct pl_ref *pl_ref_dup(struct pl_ref *ref)
{
    if (!ref)
        return NULL;

    pl_rc_ref(&ref->rc);
    return ref;
}

void pl_ref_deref(struct pl_ref **refp)
{
    struct pl_ref *ref = *refp;
    if (!ref)
        return;

    if (pl_rc_deref(&ref->rc)) {
        pl_free(ref);
        *refp = NULL;
    }
}

char *pl_asprintf(void *parent, const char *fmt, ...)
{
    char *str;
    va_list ap;
    va_start(ap, fmt);
    str = pl_vasprintf(parent, fmt, ap);
    va_end(ap);
    return str;
}

char *pl_vasprintf(void *parent, const char *fmt, va_list ap)
{
    // First, we need to determine the size that will be required for
    // printing the entire string. Do this by making a copy of the va_list
    // and printing it to a null buffer.
    va_list copy;
    va_copy(copy, ap);
    int size = vsnprintf(NULL, 0, fmt, copy);
    va_end(copy);
    if (size < 0)
        return NULL;

    char *str = pl_alloc(parent, size + 1);
    vsnprintf(str, size + 1, fmt, ap);
    return str;
}
