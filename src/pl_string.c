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

static void grow_str(void *alloc, pl_str *str, size_t len)
{
    // Like pl_grow, but with some extra headroom
    if (len > pl_get_size(str->buf))
        str->buf = pl_realloc(alloc, str->buf, len * 1.5);
}

void pl_str_append(void *alloc, pl_str *str, pl_str append)
{
    if (!append.len)
        return;

    // Also append an extra \0 for convenience, since a lot of the time
    // this function will be used to generate a string buffer
    grow_str(alloc, str, str->len + append.len + 1);
    memcpy(str->buf + str->len, append.buf, append.len);
    str->len += append.len;
    str->buf[str->len] = '\0';
}

void pl_str_append_asprintf(void *alloc, pl_str *str, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    pl_str_append_vasprintf(alloc, str, fmt, ap);
    va_end(ap);
}

void pl_str_append_vasprintf(void *alloc, pl_str *str, const char *fmt, va_list ap)
{
    // First, we need to determine the size that will be required for
    // printing the entire string. Do this by making a copy of the va_list
    // and printing it to a null buffer.
    va_list copy;
    va_copy(copy, ap);
    int size = vsnprintf(NULL, 0, fmt, copy);
    va_end(copy);
    if (size < 0)
        return;

    // Make room in `str` and format to there directly
    grow_str(alloc, str, str->len + size + 1);
    str->len += vsnprintf(str->buf + str->len, size + 1, fmt, ap);
}

int pl_str_sscanf(pl_str str, const char *fmt, ...)
{
    char *tmp = pl_strdup0(NULL, str);
    va_list va;
    va_start(va, fmt);
    int ret = vsscanf(tmp, fmt, va);
    va_end(va);
    pl_free(tmp);
    return ret;
}

int pl_strchr(pl_str str, int c)
{
    if (!str.len)
        return -1;

    void *pos = memchr(str.buf, c, str.len);
    if (pos)
        return (intptr_t) pos - (intptr_t) str.buf;
    return -1;
}

size_t pl_strspn(pl_str str, const char *accept)
{
    for (size_t i = 0; i < str.len; i++) {
        if (!strchr(accept, str.buf[i]))
            return i;
    }

    return str.len;
}

size_t pl_strcspn(pl_str str, const char *reject)
{
    for (size_t i = 0; i < str.len; i++) {
        if (strchr(reject, str.buf[i]))
            return i;
    }

    return str.len;
}

static inline bool pl_isspace(char c)
{
    switch (c) {
    case ' ':
    case '\n':
    case '\r':
    case '\t':
    case '\v':
    case '\f':
        return true;
    default:
        return false;
    }
}

pl_str pl_str_strip(pl_str str)
{
    while (str.len && pl_isspace(str.buf[0])) {
        str.buf++;
        str.len--;
    }
    while (str.len && pl_isspace(str.buf[str.len - 1]))
        str.len--;
    return str;
}

int pl_str_find(pl_str haystack, pl_str needle)
{
    if (!needle.len)
        return 0;

    for (size_t i = 0; i + needle.len <= haystack.len; i++) {
        if (memcmp(&haystack.buf[i], needle.buf, needle.len) == 0)
            return i;
    }

    return -1;
}

pl_str pl_str_split_char(pl_str str, char sep, pl_str *out_rest)
{
    int pos = pl_strchr(str, sep);
    if (pos < 0) {
        if (out_rest)
            *out_rest = (pl_str) {0};
        return str;
    } else {
        if (out_rest)
            *out_rest = pl_str_drop(str, pos + 1);
        return pl_str_take(str, pos);
    }
}

pl_str pl_str_split_str(pl_str str, pl_str sep, pl_str *out_rest)
{
    int pos = pl_str_find(str, sep);
    if (pos < 0) {
        if (out_rest)
            *out_rest = (pl_str) {0};
        return str;
    } else {
        if (out_rest)
            *out_rest = pl_str_drop(str, pos + sep.len);
        return pl_str_take(str, pos);
    }
}

static bool get_hexdigit(pl_str *str, int *digit)
{
    while (str->len && pl_isspace(str->buf[0])) {
        str->buf++;
        str->len--;
    }

    if (!str->len) {
        *digit = -1; // EOF
        return true;
    }

    char c = str->buf[0];
    str->buf++;
    str->len--;

    if (c >= '0' && c <= '9') {
        *digit = c - '0';
    } else if (c >= 'a' && c <= 'f') {
        *digit = c - 'a' + 10;
    } else if (c >= 'A' && c <= 'F') {
        *digit = c - 'A' + 10;
    } else {
        return false; // invalid char
    }

    return true;
}

bool pl_str_decode_hex(void *alloc, pl_str hex, pl_str *out)
{
    if (!out)
        return false;

    char *buf = pl_alloc(alloc, hex.len / 2);
    int len = 0;

    while (hex.len) {
        int a, b;
        if (!get_hexdigit(&hex, &a) || !get_hexdigit(&hex, &b))
            goto error; // invalid char
        if (a < 0) // EOF
            break;
        if (b < 0) // only one digit
            goto error;

        buf[len++] = (a << 4) | b;
    }

    *out = (pl_str) { buf, len };
    return true;

error:
    pl_free(buf);
    return false;
}
