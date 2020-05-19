/* Copyright (C) 2017 the mpv developers
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#define TA_NO_WRAPPERS
#include "ta.h"

// Return element_size * count. If it overflows, return (size_t)-1 (SIZE_MAX).
// I.e. this returns the equivalent of: MIN(element_size * count, SIZE_MAX).
// The idea is that every real memory allocator will reject (size_t)-1, thus
// this is a valid way to handle too large array allocation requests.
size_t xta_calc_array_size(size_t element_size, size_t count)
{
    if (count > (((size_t)-1) / element_size))
        return (size_t)-1;
    return element_size * count;
}

// This is used when an array has to be enlarged for appending new elements.
// Return a "good" size for the new array (in number of elements). This returns
// a value > nextidx, unless the calculation overflows, in which case SIZE_MAX
// is returned.
size_t xta_calc_prealloc_elems(size_t nextidx)
{
    if (nextidx >= ((size_t)-1) / 2 - 1)
        return (size_t)-1;
    return (nextidx + 1) * 2;
}

static void dummy_dtor(void *p){}

/* Create an empty (size 0) TA allocation, which is prepared in a way such that
 * using it as parent with xta_set_parent() always succeed. Calling
 * xta_set_destructor() on it will always succeed as well.
 */
void *xta_new_context(void *xta_parent)
{
    void *new = xta_alloc_size(xta_parent, 0);
    // Force it to allocate an extended header.
    if (!xta_set_destructor(new, dummy_dtor)) {
        xta_free(new);
        new = NULL;
    }
    return new;
}

/* Set parent of ptr to xta_parent, return the ptr.
 * Note that xta_parent==NULL will simply unset the current parent of ptr.
 * If the operation fails (on OOM), return NULL. (That's pretty bad behavior,
 * but the only way to signal failure.)
 */
void *xta_steal_(void *xta_parent, void *ptr)
{
    if (!xta_set_parent(ptr, xta_parent))
        return NULL;
    return ptr;
}

/* Duplicate the memory at ptr with the given size.
 */
void *xta_memdup(void *xta_parent, const void *ptr, size_t size)
{
    if (!ptr) {
        assert(!size);
        return NULL;
    }
    void *res = xta_alloc_size(xta_parent, size);
    if (!res)
        return NULL;
    memcpy(res, ptr, size);
    return res;
}

// *str = *str[0..at] + append[0..append_len]
// (append_len being a maximum length; shorter if embedded \0s are encountered)
static bool strndup_append_at(char **str, size_t at, const char *append,
                              size_t append_len)
{
    assert(xta_get_size(*str) >= at);

    if (!*str && !append)
        return true; // stays NULL, but not an OOM condition

    size_t real_len = append ? strnlen(append, append_len) : 0;
    if (append_len > real_len)
        append_len = real_len;

    if (xta_get_size(*str) < at + append_len + 1) {
        char *t = xta_realloc_size(NULL, *str, at + append_len + 1);
        if (!t)
            return false;
        *str = t;
    }

    assert(*str);
    if (append_len)
        memcpy(*str + at, append, append_len);

    (*str)[at + append_len] = '\0';

    xta_dbg_mark_as_string(*str);

    return true;
}

/* Return a copy of str.
 * Returns NULL on OOM.
 */
char *xta_strdup(void *xta_parent, const char *str)
{
    return xta_strndup(xta_parent, str, str ? strlen(str) : 0);
}

/* Return a copy of str. If the string is longer than n, copy only n characters
 * (the returned allocation will be n+1 bytes and contain a terminating '\0').
 * The returned string will have the length MIN(strlen(str), n)
 * If str==NULL, return NULL. Returns NULL on OOM as well.
 */
char *xta_strndup(void *xta_parent, const char *str, size_t n)
{
    if (!str)
        return NULL;
    char *new = NULL;
    strndup_append_at(&new, 0, str, n);
    if (!xta_set_parent(new, xta_parent)) {
        xta_free(new);
        new = NULL;
    }
    return new;
}

/* Append a to *str. If *str is NULL, the string is newly allocated, otherwise
 * xta_realloc() is used on *str as needed.
 * Return success or failure (it can fail due to OOM only).
 */
bool xta_strdup_append(char **str, const char *a)
{
    return strndup_append_at(str, *str ? strlen(*str) : 0, a, (size_t)-1);
}

/* Like xta_strdup_append(), but use xta_get_size(*str)-1 instead of strlen(*str).
 * (See also: xta_asprintf_append_buffer())
 */
bool xta_strdup_append_buffer(char **str, const char *a)
{
    size_t size = xta_get_size(*str);
    if (size > 0)
        size -= 1;
    return strndup_append_at(str, size, a, (size_t)-1);
}

/* Like xta_strdup_append(), but limit the length of a with n.
 * (See also: xta_strndup())
 */
bool xta_strndup_append(char **str, const char *a, size_t n)
{
    return strndup_append_at(str, *str ? strlen(*str) : 0, a, n);
}

/* Like xta_strdup_append_buffer(), but limit the length of a with n.
 * (See also: xta_strndup())
 */
bool xta_strndup_append_buffer(char **str, const char *a, size_t n)
{
    size_t size = xta_get_size(*str);
    if (size > 0)
        size -= 1;
    return strndup_append_at(str, size, a, n);
}

static bool xta_vasprintf_append_at(char **str, size_t at, const char *fmt,
                                   va_list ap)
{
    assert(xta_get_size(*str) >= at);

    int size;
    va_list copy;
    va_copy(copy, ap);
    char c;
    size = vsnprintf(&c, 1, fmt, copy);
    va_end(copy);

    if (size < 0)
        return false;

    if (xta_get_size(*str) < at + size + 1) {
        char *t = xta_realloc_size(NULL, *str, at + size + 1);
        if (!t)
            return false;
        *str = t;
    }
    vsnprintf(*str + at, size + 1, fmt, ap);

    xta_dbg_mark_as_string(*str);

    return true;
}

/* Like snprintf(); returns the formatted string as allocation (or NULL on OOM
 * or snprintf() errors).
 */
char *xta_asprintf(void *xta_parent, const char *fmt, ...)
{
    char *res;
    va_list ap;
    va_start(ap, fmt);
    res = xta_vasprintf(xta_parent, fmt, ap);
    va_end(ap);
    return res;
}

char *xta_vasprintf(void *xta_parent, const char *fmt, va_list ap)
{
    char *res = NULL;
    xta_vasprintf_append_at(&res, 0, fmt, ap);
    if (!res || !xta_set_parent(res, xta_parent)) {
        xta_free(res);
        return NULL;
    }
    return res;
}

/* Append the formatted string to *str (after strlen(*str)). The allocation is
 * xta_realloced if needed.
 * Returns false on OOM or snprintf() errors, with *str left untouched.
 */
bool xta_asprintf_append(char **str, const char *fmt, ...)
{
    bool res;
    va_list ap;
    va_start(ap, fmt);
    res = xta_vasprintf_append(str, fmt, ap);
    va_end(ap);
    return res;
}

bool xta_vasprintf_append(char **str, const char *fmt, va_list ap)
{
    return xta_vasprintf_append_at(str, *str ? strlen(*str) : 0, fmt, ap);
}

/* Append the formatted string at the end of the allocation of *str. It
 * overwrites the last byte of the allocation too (which is assumed to be the
 * '\0' terminating the string). Compared to xta_asprintf_append(), this is
 * useful if you know that the string ends with the allocation, so that the
 * extra strlen() can be avoided for better performance.
 * Returns false on OOM or snprintf() errors, with *str left untouched.
 */
bool xta_asprintf_append_buffer(char **str, const char *fmt, ...)
{
    bool res;
    va_list ap;
    va_start(ap, fmt);
    res = xta_vasprintf_append_buffer(str, fmt, ap);
    va_end(ap);
    return res;
}

bool xta_vasprintf_append_buffer(char **str, const char *fmt, va_list ap)
{
    size_t size = xta_get_size(*str);
    if (size > 0)
        size -= 1;
    return xta_vasprintf_append_at(str, size, fmt, ap);
}

static inline void oom_abort()
{
    fprintf(stderr, "out of memory\n");
    abort();
}

void *xta_oom_p(void *p)
{
    if (!p)
        oom_abort();
    return p;
}

void xta_oom_b(bool b)
{
    if (!b)
        oom_abort();
}

char *xta_oom_s(char *s)
{
    if (!s)
        oom_abort();
    return s;
}

void *xta_xsteal_(void *xta_parent, void *ptr)
{
    xta_oom_b(xta_set_parent(ptr, xta_parent));
    return ptr;
}

void *xta_xmemdup(void *xta_parent, const void *ptr, size_t size)
{
    void *new = xta_memdup(xta_parent, ptr, size);
    xta_oom_b(new || !ptr);
    return new;
}

void *xta_xrealloc_size(void *xta_parent, void *ptr, size_t size)
{
    ptr = xta_realloc_size(xta_parent, ptr, size);
    xta_oom_b(ptr || !size);
    return ptr;
}

char *xta_xstrdup(void *xta_parent, const char *str)
{
    char *res = xta_strdup(xta_parent, str);
    xta_oom_b(res || !str);
    return res;
}

char *xta_xstrndup(void *xta_parent, const char *str, size_t n)
{
    char *res = xta_strndup(xta_parent, str, n);
    xta_oom_b(res || !str);
    return res;
}
