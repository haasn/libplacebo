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
#include <pthread.h>

#include <xtalloc.h>

char *xta_talloc_strdup_append(char *s, const char *a)
{
    xta_xstrdup_append(&s, a);
    return s;
}

char *xta_talloc_strdup_append_buffer(char *s, const char *a)
{
    xta_xstrdup_append_buffer(&s, a);
    return s;
}

char *xta_talloc_strndup_append(char *s, const char *a, size_t n)
{
    xta_xstrndup_append(&s, a, n);
    return s;
}

char *xta_talloc_strndup_append_buffer(char *s, const char *a, size_t n)
{
    xta_xstrndup_append_buffer(&s, a, n);
    return s;
}

char *xta_talloc_vasprintf_append(char *s, const char *fmt, va_list ap)
{
    xta_xvasprintf_append(&s, fmt, ap);
    return s;
}

char *xta_talloc_vasprintf_append_buffer(char *s, const char *fmt, va_list ap)
{
    xta_xvasprintf_append_buffer(&s, fmt, ap);
    return s;
}

char *xta_talloc_asprintf_append(char *s, const char *fmt, ...)
{
    char *res;
    va_list ap;
    va_start(ap, fmt);
    res = talloc_vasprintf_append(s, fmt, ap);
    va_end(ap);
    return res;
}

char *xta_talloc_asprintf_append_buffer(char *s, const char *fmt, ...)
{
    char *res;
    va_list ap;
    va_start(ap, fmt);
    res = talloc_vasprintf_append_buffer(s, fmt, ap);
    va_end(ap);
    return res;
}

struct xta_ref {
    pthread_mutex_t lock;
    int refcount;
};

struct xta_ref *xta_ref_new(void *t)
{
    struct xta_ref *ref = xta_znew(t, struct xta_ref);
    if (!ref)
        return NULL;

    *ref = (struct xta_ref) {
        .lock = PTHREAD_MUTEX_INITIALIZER,
        .refcount = 1,
    };

    return ref;
}

struct xta_ref *xta_ref_dup(struct xta_ref *ref)
{
    if (!ref)
        return NULL;

    pthread_mutex_lock(&ref->lock);
    ref->refcount++;
    pthread_mutex_unlock(&ref->lock);
    return ref;
}

void xta_ref_deref(struct xta_ref **refp)
{
    struct xta_ref *ref = *refp;
    if (!ref)
        return;

    pthread_mutex_lock(&ref->lock);
    if (--ref->refcount > 0) {
        pthread_mutex_unlock(&ref->lock);
        return;
    }

    pthread_mutex_unlock(&ref->lock);
    pthread_mutex_destroy(&ref->lock);
    xta_free(ref);
    *refp = NULL;
}

// Indirection object, used to associate the destructor with a xta_ref_deref
struct xta_ref_indirect {
    struct xta_ref *ref;
};

static void xta_ref_indir_dtor(void *p)
{
    struct xta_ref_indirect *indir = p;
    xta_ref_deref(&indir->ref);
}

bool xta_ref_attach(void *t, struct xta_ref *ref)
{
    if (!ref)
        return true;

    struct xta_ref_indirect *indir = xta_new_ptrtype(t, indir);
    if (!indir)
        return false;

    indir->ref = xta_ref_dup(ref);
    xta_set_destructor(indir, xta_ref_indir_dtor);
    return true;
}
