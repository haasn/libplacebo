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

PL_API_BEGIN

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
        .buf = (uint8_t *) (str.len ? pl_memdup(alloc, str.buf, str.len) : NULL),
        .len = str.len,
    };
}

// Always returns a valid string
static inline char *pl_strdup0(void *alloc, pl_str str)
{
    return pl_strndup0(alloc, str.len ? (char *) str.buf : "", str.len);
}

// Adds a trailing \0 for convenience, even if `append` is an empty string
void pl_str_append(void *alloc, pl_str *str, pl_str append);

// Like `pl_str_append` but for raw memory, omits trailing \0
void pl_str_append_raw(void *alloc, pl_str *str, const void *ptr, size_t size);

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

// Variant of the above which takes arguments directly from a pointer in memory,
// reading them incrementally (tightly packed). Returns the amount of bytes
// read from `args`, as determined by the following table:
//
// %c: sizeof(char)
// %d, %u: sizeof(int)
// %f: sizeof(double)
// %lld, %llu: sizeof(long long int)
// %zu: sizeof(size_t)
// %s: \0 terminated string
// %.*s: sizeof(int) + that many bytes (no \0 terminator)
size_t pl_str_append_memprintf_c(void *alloc, pl_str *str, const char *fmt,
                                 const void *args)
    PL_PRINTF(3, 0);

// Locale-invariant number printing
int pl_str_print_hex(char *buf, size_t len, unsigned short n);
int pl_str_print_int(char *buf, size_t len, int n);
int pl_str_print_uint(char *buf, size_t len, unsigned int n);
int pl_str_print_int64(char *buf, size_t len, int64_t n);
int pl_str_print_uint64(char *buf, size_t len, uint64_t n);
int pl_str_print_float(char *buf, size_t len, float n);
int pl_str_print_double(char *buf, size_t len, double n);

// Locale-invariant number parsing
bool pl_str_parse_hex(pl_str str, unsigned short *out);
bool pl_str_parse_int(pl_str str, int *out);
bool pl_str_parse_uint(pl_str str, unsigned int *out);
bool pl_str_parse_int64(pl_str str, int64_t *out);
bool pl_str_parse_uint64(pl_str str, uint64_t *out);
bool pl_str_parse_float(pl_str str, float *out);
bool pl_str_parse_double(pl_str str, double *out);

int print_hex(char *buf, unsigned int x);
int ccStrPrintInt32( char *str, int32_t n );
int ccStrPrintUint32( char *str, uint32_t n );
int ccStrPrintInt64( char *str, int64_t n );
int ccStrPrintUint64( char *str, uint64_t n );
int ccStrPrintDouble( char *str, int bufsize, int decimals, double value );
int ccSeqParseInt64( char *seq, int seqlength, int64_t *retint );
int ccSeqParseUint64( char *seq, int seqlength, uint64_t *retint );
int ccSeqParseDouble( char *seq, int seqlength, double *retdouble );

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
        return (pl_str) { .buf = NULL, .len = 0 };

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

// Like `pl_str_split_char`, but splits on any char in `seps`
pl_str pl_str_split_chars(pl_str str, const char *seps, pl_str *out_rest);

static inline pl_str pl_str_getline(pl_str str, pl_str *out_rest)
{
    return pl_str_split_char(str, '\n', out_rest);
}

// Decode a string containing hexadecimal data. All whitespace will be silently
// ignored. When successful, this allocates a new array to store the output.
bool pl_str_decode_hex(void *alloc, pl_str hex, pl_str *out);

// Compute a fast 64-bit hash
uint64_t pl_mem_hash(const void *mem, size_t size);
static inline void pl_hash_merge(uint64_t *accum, uint64_t hash) {
    *accum ^= hash + 0x9e3779b9 + (*accum << 6) + (*accum >> 2);
}

static inline uint64_t pl_str_hash(pl_str str)
{
    return pl_mem_hash(str.buf, str.len);
}

static inline uint64_t pl_str0_hash(const char *str)
{
    return pl_mem_hash(str, str ? strlen(str) : 0);
}

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

// String building helpers, used to lazily construct a string by appending a
// series of string templates which can be executed on-demand into a final
// output buffer.
typedef struct pl_str_builder_t *pl_str_builder;

// Returns the number of bytes consumed from `args`. Be warned that the pointer
// given will not necessarily be aligned to the type you need it as, so make
// sure to use `memcpy` or some other method of safely loading arbitrary data
// from memory.
typedef size_t (*pl_str_template)(void *alloc, pl_str *buf, const uint8_t *args);

pl_str_builder pl_str_builder_alloc(void *alloc);
void pl_str_builder_free(pl_str_builder *builder);

// Resets string builder without destroying buffer
void pl_str_builder_reset(pl_str_builder builder);

// Returns a representative hash of the string builder's output, without
// actually executing it. Note that this is *not* the same as a pl_str_hash of
// the string builder's output.
//
// Note also that the output of this may not survive a process restart because
// of position-independent code and address randomization moving around the
// locatons of template functions, so special care must be taken not to
// compare such hashes across process invocations.
uint64_t pl_str_builder_hash(const pl_str_builder builder);

// Executes a string builder, dispatching all templates. The resulting string
// is guaranteed to be \0-terminated, as a minor convenience.
//
// Calling any other `pl_str_builder_*` function on this builder causes the
// contents of the returned string to become undefined.
pl_str pl_str_builder_exec(pl_str_builder builder);

// Append a template and its arguments to a string builder
void pl_str_builder_append(pl_str_builder builder, pl_str_template tmpl,
                           const void *args, size_t args_size);

// Append an entire other `pl_str_builder` onto `builder`
void pl_str_builder_concat(pl_str_builder builder, const pl_str_builder append);

// Append a constant string. This will only record &str into the buffer, which
// may have a number of unwanted consequences if the memory pointed at by
// `str` mutates at any point in time in the future, or if `str` is not
// at a stable location in memory.
//
// This is intended for strings which are compile-time constants.
void pl_str_builder_const_str(pl_str_builder builder, const char *str);

// Append a string. This will make a full copy of `str`
void pl_str_builder_str(pl_str_builder builder, const pl_str str);
#define pl_str_builder_str0(b, str) pl_str_builder_str(b, pl_str0(str))

// Append a string printf-style. This will preprocess `fmt` to determine the
// number and type of arguments. Supports the same format conversion characters
// as `pl_str_append_asprintf_c`.
void pl_str_builder_printf_c(pl_str_builder builder, const char *fmt, ...)
    PL_PRINTF(2, 3);

void pl_str_builder_vprintf_c(pl_str_builder builder, const char *fmt, va_list ap)
    PL_PRINTF(2, 0);

// Helper macro to specialize `pl_str_builder_printf_c` to
// `pl_str_builder_const_str` if it contains no format characters.
#define pl_str_builder_addf(builder, ...) do                                    \
{                                                                               \
    if (_contains_fmt_chars(__VA_ARGS__)) {                                     \
        pl_str_builder_printf_c(builder, __VA_ARGS__);                          \
    } else {                                                                    \
        pl_str_builder_const_str(builder, _get_fmt(__VA_ARGS__));               \
    }                                                                           \
} while (0)

// Helper macros to deal with the non-portability of __VA_OPT__(,)
#define _contains_fmt_chars(fmt, ...)   (strchr(fmt, '%'))
#define _get_fmt(fmt, ...)              fmt

PL_API_END
