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

#ifndef TA_XTALLOC_H_
#define TA_XTALLOC_H_

#include <string.h>
#include <assert.h>
#include <stdint.h>

#include "../ta.h"

// Note: all talloc wrappers are wired to the "x" functions, which abort on OOM.
//       libtalloc doesn't do that, but the mplayer2/mpv internal copies of it did.

#define talloc                          xta_xnew
#define talloc_zero                     xta_xznew

#define talloc_array                    xta_xnew_array
#define talloc_zero_array               xta_xznew_array

#define talloc_array_size               xta_xnew_array_size
#define talloc_realloc                  xta_xrealloc
#define talloc_ptrtype                  xta_xnew_ptrtype
#define talloc_array_ptrtype            xta_xnew_array_ptrtype

#define talloc_steal                    xta_xsteal
#define talloc_realloc_size             xta_xrealloc_size
#define talloc_new                      xta_xnew_context
#define talloc_set_destructor           xta_xset_destructor
#define talloc_parent                   xta_find_parent
#define talloc_enable_leak_report       xta_enable_leak_report
#define talloc_print_leak_report        xta_print_leak_report
#define talloc_size                     xta_xalloc_size
#define talloc_zero_size                xta_xzalloc_size
#define talloc_get_size                 xta_get_size
#define talloc_free_children            xta_free_children
#define talloc_free                     xta_free
#define talloc_memdup                   xta_xmemdup
#define talloc_strdup                   xta_xstrdup
#define talloc_strndup                  xta_xstrndup
#define talloc_ptrdup                   xta_xdup_ptrtype
#define talloc_asprintf                 xta_xasprintf
#define talloc_vasprintf                xta_xvasprintf

// Don't define linker-level symbols, as that would clash with real libtalloc.
#define talloc_strdup_append            xta_talloc_strdup_append
#define talloc_strdup_append_buffer     xta_talloc_strdup_append_buffer
#define talloc_strndup_append           xta_talloc_strndup_append
#define talloc_strndup_append_buffer    xta_talloc_strndup_append_buffer
#define talloc_vasprintf_append         xta_talloc_vasprintf_append
#define talloc_vasprintf_append_buffer  xta_talloc_vasprintf_append_buffer
#define talloc_asprintf_append          xta_talloc_asprintf_append
#define talloc_asprintf_append_buffer   xta_talloc_asprintf_append_buffer

char *xta_talloc_strdup(void *t, const char *p);
char *xta_talloc_strdup_append(char *s, const char *a);
char *xta_talloc_strdup_append_buffer(char *s, const char *a);

char *xta_talloc_strndup(void *t, const char *p, size_t n);
char *xta_talloc_strndup_append(char *s, const char *a, size_t n);
char *xta_talloc_strndup_append_buffer(char *s, const char *a, size_t n);

char *xta_talloc_vasprintf_append(char *s, const char *fmt, va_list ap) TA_PRF(2, 0);
char *xta_talloc_vasprintf_append_buffer(char *s, const char *fmt, va_list ap) TA_PRF(2, 0);

char *xta_talloc_asprintf_append(char *s, const char *fmt, ...) TA_PRF(2, 3);
char *xta_talloc_asprintf_append_buffer(char *s, const char *fmt, ...) TA_PRF(2, 3);

// Talloc refcounting

struct xta_ref;

// xta_ref_deref will free the ref and all of its children as soon as the
// internal refcount reaches 0
struct xta_ref *xta_ref_new(void *t);
struct xta_ref *xta_ref_dup(struct xta_ref *ref);
void xta_ref_deref(struct xta_ref **ref);

// Attaches a reference as a child of another talloc ctx, such that freeing
// `t` is like dereferencing the xta_ref.
bool xta_ref_attach(void *t, struct xta_ref *ref);

#define talloc_ref_new(...)             xta_oom_p(xta_ref_new(__VA_ARGS__))
#define talloc_ref_dup(...)             xta_oom_p(xta_ref_dup(__VA_ARGS__))
#define talloc_ref_deref(...)           xta_ref_deref(__VA_ARGS__)
#define talloc_ref_attach(...)          xta_oom_b(xta_ref_attach(__VA_ARGS__))

// Talloc public/private struct helpers
#define talloc_alignment (offsetof(struct { char c; intmax_t x; }, x))
#define talloc_align(size) \
    (((size) + talloc_alignment - 1) & ~(talloc_alignment - 1))

#define TA_PRIV(pub) \
    ((void *) ((uintptr_t) (pub) + talloc_align(sizeof(*(pub)))))

#define talloc_priv(ta, pub, priv) \
    ((pub *) talloc_size((ta), talloc_align(sizeof(pub)) + sizeof(priv)))

#define talloc_zero_priv(ta, pub, priv) \
    ((pub *) talloc_zero_size((ta), talloc_align(sizeof(pub)) + sizeof(priv)))

#define talloc_ptrtype_priv(ta, ptr, priv) \
    ((TA_TYPEOF(ptr)) talloc_size((ta), talloc_align(sizeof(*ptr)) + sizeof(priv)))

// Utility functions (ported from mpv)

#define TA_FREEP(pctx) do {talloc_free(*(pctx)); *(pctx) = NULL;} while(0)

#define TA_EXPAND_ARGS(...) __VA_ARGS__

#define TALLOC_AVAIL(p) (talloc_get_size(p) / sizeof((p)[0]))

#define TARRAY_RESIZE(ctx, p, count)                            \
    do {                                                        \
        (p) = xta_xrealloc_size((void*) ctx, p,                 \
                    xta_calc_array_size(sizeof((p)[0]), count));\
    } while (0)

#define TARRAY_GROW(ctx, p, nextidx)                \
    do {                                            \
        size_t nextidx_ = (nextidx);                \
        if (nextidx_ >= TALLOC_AVAIL(p))            \
            TARRAY_RESIZE(ctx, p, xta_calc_prealloc_elems(nextidx_)); \
        assert(p);                                  \
    } while (0)

#define TARRAY_APPEND(ctx, p, idxvar, ...)          \
    do {                                            \
        TARRAY_GROW(ctx, p, idxvar);                \
        (p)[(idxvar)] = (TA_EXPAND_ARGS(__VA_ARGS__));\
        (idxvar)++;                                 \
    } while (0)

#define TARRAY_INSERT_AT(ctx, p, idxvar, at, ...)   \
    do {                                            \
        size_t at_ = (at);                          \
        assert(at_ <= (idxvar));                    \
        TARRAY_GROW(ctx, p, idxvar);                \
        memmove((p) + at_ + 1, (p) + at_,           \
                ((idxvar) - at_) * sizeof((p)[0])); \
        (idxvar)++;                                 \
        (p)[at_] = (TA_EXPAND_ARGS(__VA_ARGS__));   \
    } while (0)

// Appends all of `op` to `p`
#define TARRAY_CONCAT(ctx, p, idxvar, op, oidxvar)  \
    do {                                            \
        TARRAY_GROW(ctx, p, (idxvar) + (oidxvar));  \
        if ((oidxvar)) {                            \
            memmove((p) + (idxvar), (op),           \
                    (oidxvar) * sizeof((op)[0]));   \
            (idxvar) += (oidxvar);                  \
        }                                           \
    } while (0)

// Doesn't actually free any memory, or do any other talloc calls.
#define TARRAY_REMOVE_AT(p, idxvar, at)             \
    do {                                            \
        size_t at_ = (at);                          \
        assert(at_ <= (idxvar));                    \
        memmove((p) + at_, (p) + at_ + 1,           \
                ((idxvar) - at_ - 1) * sizeof((p)[0])); \
        (idxvar)--;                                 \
    } while (0)

// Returns whether or not there was any element to pop.
#define TARRAY_POP(p, idxvar, out)                  \
    ((idxvar) > 0                                   \
        ? (*(out) = (p)[--(idxvar)], true)          \
        : false                                     \
    )

#define TARRAY_DUP(ctx, p, count) \
    talloc_memdup(ctx, p, (count) * sizeof((p)[0]))

#define talloc_struct(ctx, type, ...) \
    talloc_memdup(ctx, &(type) TA_EXPAND_ARGS(__VA_ARGS__), sizeof(type))

#endif
