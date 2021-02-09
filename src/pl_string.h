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

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct pl_str {
    uint8_t *buf;
    size_t len;
} pl_str;

// For formatting with "%.*s"
#define PL_STR_FMT(str) (int)((str).len), ((str).buf ? (char *)((str).buf) : "")

static inline pl_str pl_str0(const char *str)
{
    return (pl_str) {
        .buf = (uint8_t *) str,
        .len = str ? strlen(str) : 0,
    };
}

// Macro version of pl_str0, for constants
#define PL_STR0(str) ((pl_str) { (uint8_t *) (str), (str) ? strlen(str) : 0 })

static inline pl_str pl_strdup(void *alloc, pl_str str)
{
    return (pl_str) {
        .buf = str.len ? pl_memdup(alloc, str.buf, str.len) : NULL,
        .len = str.len,
    };
}

// Always returns a valid string
static inline char *pl_strdup0(void *alloc, pl_str str)
{
    return pl_strndup0(alloc, str.len ? (char *) str.buf : "", str.len);
}

void pl_str_append(void *alloc, pl_str *str, pl_str append);

// Locale-sensitive string functions
char *pl_asprintf(void *parent, const char *fmt, ...)
    PL_PRINTF(2, 3);
char *pl_vasprintf(void *parent, const char *fmt, va_list ap)
    PL_PRINTF(2, 0);
void pl_str_append_asprintf(void *alloc, pl_str *str, const char *fmt, ...)
    PL_PRINTF(3, 4);
void pl_str_append_vasprintf(void *alloc, pl_str *str, const char *fmt, va_list va)
    PL_PRINTF(3, 0);
int pl_str_sscanf(pl_str str, const char *fmt, ...);

// Locale-invariant versions of append_(v)asprintf
//
// NOTE: These only support a small handful of modifiers. Check `format.c`
// for a list. Calling them on an invalid string will abort!
void pl_str_append_asprintf_c(void *alloc, pl_str *str, const char *fmt, ...)
    PL_PRINTF(3, 4);
void pl_str_append_vasprintf_c(void *alloc, pl_str *str, const char *fmt, va_list va)
    PL_PRINTF(3, 0);

// Locale-invariant number parsing
bool pl_str_parse_double(pl_str str, double *out);
bool pl_str_parse_int64(pl_str str, int64_t *out);

static inline bool pl_str_parse_float(pl_str str, float *out)
{
    double dbl;
    bool ret = pl_str_parse_double(str, &dbl);
    *out = (float) dbl;
    return ret;
}

static inline bool pl_str_parse_int(pl_str str, int *out)
{
    int64_t i64;
    bool ret = pl_str_parse_int64(str, &i64);
    *out = (int) i64;
    return ret;
}

// Variants of string.h functions
int pl_strchr(pl_str str, int c);
size_t pl_strspn(pl_str str, const char *accept);
size_t pl_strcspn(pl_str str, const char *reject);

// Strip leading/trailing whitespace
pl_str pl_str_strip(pl_str str);

// Generic functions for cutting up strings
static inline pl_str pl_str_take(pl_str str, size_t len)
{
    if (len < str.len)
        str.len = len;
    return str;
}

static inline pl_str pl_str_drop(pl_str str, size_t len)
{
    if (len >= str.len)
        return (pl_str) {0};

    str.buf += len;
    str.len -= len;
    return str;
}

// Find a substring in another string, and return its index (or -1)
int pl_str_find(pl_str haystack, pl_str needle);

// String splitting functions. These return the part of the string before
// the separator, and optionally the rest (in `out_rest`).
//
// Note that the separator is not included as part of either string.
pl_str pl_str_split_char(pl_str str, char sep, pl_str *out_rest);
pl_str pl_str_split_str(pl_str str, pl_str sep, pl_str *out_rest);

static inline pl_str pl_str_getline(pl_str str, pl_str *out_rest)
{
    return pl_str_split_char(str, '\n', out_rest);
}

// Decode a string containing hexadecimal data. All whitespace will be silently
// ignored. When successful, this allocates a new array to store the output.
bool pl_str_decode_hex(void *alloc, pl_str hex, pl_str *out);

// Return a 64-bit hash of a string's contents
uint64_t pl_str_hash(pl_str str);

static inline bool pl_str_equals(pl_str str1, pl_str str2)
{
    if (str1.len != str2.len)
        return false;
    if (str1.buf == str2.buf || !str1.len)
        return true;
    return memcmp(str1.buf, str2.buf, str1.len) == 0;
}

static inline bool pl_str_startswith(pl_str str, pl_str prefix)
{
    if (!prefix.len)
        return true;
    if (str.len < prefix.len)
        return false;
    return memcmp(str.buf, prefix.buf, prefix.len) == 0;
}

static inline bool pl_str_endswith(pl_str str, pl_str suffix)
{
    if (!suffix.len)
        return true;
    if (str.len < suffix.len)
        return false;
    return memcmp(str.buf + str.len - suffix.len, suffix.buf, suffix.len) == 0;
}

static inline bool pl_str_eatstart(pl_str *str, pl_str prefix)
{
    if (!pl_str_startswith(*str, prefix))
        return false;

    str->buf += prefix.len;
    str->len -= prefix.len;
    return true;
}

static inline bool pl_str_eatend(pl_str *str, pl_str suffix)
{
    if (!pl_str_endswith(*str, suffix))
        return false;

    str->len -= suffix.len;
    return true;
}

// Convenience wrappers for the above which save the use of a pl_str0
static inline pl_str pl_str_split_str0(pl_str str, const char *sep, pl_str *out_rest)
{
    return pl_str_split_str(str, pl_str0(sep), out_rest);
}

static inline bool pl_str_startswith0(pl_str str, const char *prefix)
{
    return pl_str_startswith(str, pl_str0(prefix));
}

static inline bool pl_str_endswith0(pl_str str, const char *suffix)
{
    return pl_str_endswith(str, pl_str0(suffix));
}

static inline bool pl_str_equals0(pl_str str1, const char *str2)
{
    return pl_str_equals(str1, pl_str0(str2));
}

static inline bool pl_str_eatstart0(pl_str *str, const char *prefix)
{
    return pl_str_eatstart(str, pl_str0(prefix));
}

static inline bool pl_str_eatend0(pl_str *str, const char *prefix)
{
    return pl_str_eatend(str, pl_str0(prefix));
}
