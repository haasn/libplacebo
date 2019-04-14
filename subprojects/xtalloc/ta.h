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

#ifndef TA_H_
#define TA_H_

#include <stddef.h>
#include <stdbool.h>
#include <stdarg.h>

#ifdef __GNUC__
#define TA_PRF(a1, a2) __attribute__ ((format(printf, a1, a2)))
#define TA_TYPEOF(t) __typeof__(t)
#else
#define TA_PRF(a1, a2)
#define TA_TYPEOF(t) void *
#endif

// Broken crap with __USE_MINGW_ANSI_STDIO
#if defined(__MINGW32__) && defined(__GNUC__) && !defined(__clang__)
#undef TA_PRF
#define TA_PRF(a1, a2) __attribute__ ((format (gnu_printf, a1, a2)))
#endif

#define TA_STRINGIFY_(x) # x
#define TA_STRINGIFY(x) TA_STRINGIFY_(x)

#ifdef NDEBUG
#define TA_LOC ""
#else
#define TA_LOC __FILE__ ":" TA_STRINGIFY(__LINE__)
#endif

// Core functions
void *xta_alloc_size(void *xta_parent, size_t size);
void *xta_zalloc_size(void *xta_parent, size_t size);
void *xta_realloc_size(void *xta_parent, void *ptr, size_t size);
size_t xta_get_size(void *ptr);
void xta_free(void *ptr);
void xta_free_children(void *ptr);
bool xta_set_destructor(void *ptr, void (*destructor)(void *));
bool xta_set_parent(void *ptr, void *xta_parent);
void *xta_find_parent(void *ptr);

// Utility functions
size_t xta_calc_array_size(size_t element_size, size_t count);
size_t xta_calc_prealloc_elems(size_t nextidx);
void *xta_new_context(void *xta_parent);
void *xta_steal_(void *xta_parent, void *ptr);
void *xta_memdup(void *xta_parent, const void *ptr, size_t size);
char *xta_strdup(void *xta_parent, const char *str);
bool xta_strdup_append(char **str, const char *a);
bool xta_strdup_append_buffer(char **str, const char *a);
char *xta_strndup(void *xta_parent, const char *str, size_t n);
bool xta_strndup_append(char **str, const char *a, size_t n);
bool xta_strndup_append_buffer(char **str, const char *a, size_t n);
char *xta_asprintf(void *xta_parent, const char *fmt, ...) TA_PRF(2, 3);
char *xta_vasprintf(void *xta_parent, const char *fmt, va_list ap) TA_PRF(2, 0);
bool xta_asprintf_append(char **str, const char *fmt, ...) TA_PRF(2, 3);
bool xta_vasprintf_append(char **str, const char *fmt, va_list ap) TA_PRF(2, 0);
bool xta_asprintf_append_buffer(char **str, const char *fmt, ...) TA_PRF(2, 3);
bool xta_vasprintf_append_buffer(char **str, const char *fmt, va_list ap) TA_PRF(2, 0);

#define xta_new(xta_parent, type)  (type *)xta_alloc_size(xta_parent, sizeof(type))
#define xta_znew(xta_parent, type) (type *)xta_zalloc_size(xta_parent, sizeof(type))

#define xta_new_array(xta_parent, type, count) \
    (type *)xta_alloc_size(xta_parent, xta_calc_array_size(sizeof(type), count))

#define xta_znew_array(xta_parent, type, count) \
    (type *)xta_zalloc_size(xta_parent, xta_calc_array_size(sizeof(type), count))

#define xta_new_array_size(xta_parent, element_size, count) \
    xta_alloc_size(xta_parent, xta_calc_array_size(element_size, count))

#define xta_realloc(xta_parent, ptr, type, count) \
    (type *)xta_realloc_size(xta_parent, ptr, xta_calc_array_size(sizeof(type), count))

#define xta_new_ptrtype(xta_parent, ptr) \
    (TA_TYPEOF(ptr))xta_alloc_size(xta_parent, sizeof(*ptr))

#define xta_new_array_ptrtype(xta_parent, ptr, count) \
    (TA_TYPEOF(ptr))xta_new_array_size(xta_parent, sizeof(*(ptr)), count)

#define xta_steal(xta_parent, ptr) (TA_TYPEOF(ptr))xta_steal_(xta_parent, ptr)

#define xta_dup_ptrtype(xta_parent, ptr) \
    (TA_TYPEOF(ptr))xta_memdup(xta_parent, (void*) (ptr), sizeof(*(ptr)))

// Ugly macros that crash on OOM.
// All of these mirror real functions (with a 'x' added after the 'xta_'
// prefix), and the only difference is that they will call abort() on allocation
// failures (such as out of memory conditions), instead of returning an error
// code.
#define xta_xalloc_size(...)             xta_oom_p(xta_alloc_size(__VA_ARGS__))
#define xta_xzalloc_size(...)            xta_oom_p(xta_zalloc_size(__VA_ARGS__))
#define xta_xset_destructor(...)         xta_oom_b(xta_set_destructor(__VA_ARGS__))
#define xta_xset_parent(...)             xta_oom_b(xta_set_parent(__VA_ARGS__))
#define xta_xnew_context(...)            xta_oom_p(xta_new_context(__VA_ARGS__))
#define xta_xstrdup_append(...)          xta_oom_b(xta_strdup_append(__VA_ARGS__))
#define xta_xstrdup_append_buffer(...)   xta_oom_b(xta_strdup_append_buffer(__VA_ARGS__))
#define xta_xstrndup_append(...)         xta_oom_b(xta_strndup_append(__VA_ARGS__))
#define xta_xstrndup_append_buffer(...)  xta_oom_b(xta_strndup_append_buffer(__VA_ARGS__))
#define xta_xasprintf(...)               xta_oom_s(xta_asprintf(__VA_ARGS__))
#define xta_xvasprintf(...)              xta_oom_s(xta_vasprintf(__VA_ARGS__))
#define xta_xasprintf_append(...)        xta_oom_b(xta_asprintf_append(__VA_ARGS__))
#define xta_xvasprintf_append(...)       xta_oom_b(xta_vasprintf_append(__VA_ARGS__))
#define xta_xasprintf_append_buffer(...) xta_oom_b(xta_asprintf_append_buffer(__VA_ARGS__))
#define xta_xvasprintf_append_buffer(...) xta_oom_b(xta_vasprintf_append_buffer(__VA_ARGS__))
#define xta_xnew(...)                    xta_oom_g(xta_new(__VA_ARGS__))
#define xta_xznew(...)                   xta_oom_g(xta_znew(__VA_ARGS__))
#define xta_xnew_array(...)              xta_oom_g(xta_new_array(__VA_ARGS__))
#define xta_xznew_array(...)             xta_oom_g(xta_znew_array(__VA_ARGS__))
#define xta_xnew_array_size(...)         xta_oom_p(xta_new_array_size(__VA_ARGS__))
#define xta_xnew_ptrtype(...)            xta_oom_g(xta_new_ptrtype(__VA_ARGS__))
#define xta_xnew_array_ptrtype(...)      xta_oom_g(xta_new_array_ptrtype(__VA_ARGS__))
#define xta_xdup_ptrtype(...)            xta_oom_g(xta_dup_ptrtype(__VA_ARGS__))

#define xta_xsteal(xta_parent, ptr) (TA_TYPEOF(ptr))xta_xsteal_(xta_parent, ptr)
#define xta_xrealloc(xta_parent, ptr, type, count) \
    (type *)xta_xrealloc_size(xta_parent, ptr, xta_calc_array_size(sizeof(type), count))

// Can't be macros, because the OOM logic is slightly less trivial.
char *xta_xstrdup(void *xta_parent, const char *str);
char *xta_xstrndup(void *xta_parent, const char *str, size_t n);
void *xta_xsteal_(void *xta_parent, void *ptr);
void *xta_xmemdup(void *xta_parent, const void *ptr, size_t size);
void *xta_xrealloc_size(void *xta_parent, void *ptr, size_t size);

#ifndef TA_NO_WRAPPERS
#define xta_alloc_size(...)      xta_dbg_set_loc(xta_alloc_size(__VA_ARGS__), TA_LOC)
#define xta_zalloc_size(...)     xta_dbg_set_loc(xta_zalloc_size(__VA_ARGS__), TA_LOC)
#define xta_realloc_size(...)    xta_dbg_set_loc(xta_realloc_size(__VA_ARGS__), TA_LOC)
#define xta_memdup(...)          xta_dbg_set_loc(xta_memdup(__VA_ARGS__), TA_LOC)
#define xta_xmemdup(...)         xta_dbg_set_loc(xta_xmemdup(__VA_ARGS__), TA_LOC)
#define xta_xrealloc_size(...)   xta_dbg_set_loc(xta_xrealloc_size(__VA_ARGS__), TA_LOC)
#endif

void xta_oom_b(bool b);
char *xta_oom_s(char *s);
void *xta_oom_p(void *p);
// Generic pointer
#define xta_oom_g(ptr) (TA_TYPEOF(ptr))xta_oom_p((void*) ptr)

void xta_enable_leak_report(void);
void xta_print_leak_report(void); // no-op when disabled
void *xta_dbg_set_loc(void *ptr, const char *name);
void *xta_dbg_mark_as_string(void *ptr);

#endif
