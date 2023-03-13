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
    // Also append an extra \0 for convenience, since a lot of the time
    // this function will be used to generate a string buffer
    grow_str(alloc, str, str->len + append.len + 1);
    if (append.len) {
        memcpy(str->buf + str->len, append.buf, append.len);
        str->len += append.len;
    }
    str->buf[str->len] = '\0';
}

void pl_str_append_raw(void *alloc, pl_str *str, const void *ptr, size_t size)
{
    if (!size)
        return;
    grow_str(alloc, str, str->len + size);
    memcpy(str->buf + str->len, ptr, size);
    str->len += size;
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
    str->len += vsnprintf((char *) (str->buf + str->len), size + 1, fmt, ap);
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

    uint8_t *buf = pl_alloc(alloc, hex.len / 2);
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

struct pl_str_builder_t {
    PL_ARRAY(pl_str_template) templates;
    pl_str args;
    pl_str output;
};

pl_str_builder pl_str_builder_alloc(void *alloc)
{
    pl_str_builder b = pl_zalloc_ptr(alloc, b);
    return b;
}

void pl_str_builder_free(pl_str_builder *b)
{
    if (*b)
        pl_free_ptr(b);
}

void pl_str_builder_reset(pl_str_builder b)
{
    *b = (struct pl_str_builder_t) {
        .templates.elem = b->templates.elem,
        .args.buf       = b->args.buf,
        .output.buf     = b->output.buf,
    };
}

uint64_t pl_str_builder_hash(const pl_str_builder b)
{
    size_t size = b->templates.num * sizeof(b->templates.elem[0]);
    uint64_t hash = pl_mem_hash(b->templates.elem, size);
    pl_hash_merge(&hash, pl_str_hash(b->args));
    return hash;
}

pl_str pl_str_builder_exec(pl_str_builder b)
{
    pl_str args = b->args;

    b->output.len = 0;
    for (int i = 0; i < b->templates.num; i++) {
        size_t consumed = b->templates.elem[i](b, &b->output, args.buf);
        pl_assert(consumed <= args.len);
        args = pl_str_drop(args, consumed);
    }

    // Terminate with an extra \0 byte for convenience
    grow_str(b, &b->output, b->output.len + 1);
    b->output.buf[b->output.len] = '\0';
    return b->output;
}

void pl_str_builder_append(pl_str_builder b, pl_str_template tmpl,
                           const void *args, size_t size)
{
    PL_ARRAY_APPEND(b, b->templates, tmpl);
    pl_str_append_raw(b, &b->args, args, size);
}

void pl_str_builder_concat(pl_str_builder b, const pl_str_builder append)
{
    PL_ARRAY_CONCAT(b, b->templates, append->templates);
    pl_str_append_raw(b, &b->args, append->args.buf, append->args.len);
}

static size_t template_str_ptr(void *alloc, pl_str *buf, const uint8_t *args)
{
    const char *str;
    memcpy(&str, args, sizeof(str));
    pl_str_append_raw(alloc, buf, str, strlen(str));
    return sizeof(str);
}

void pl_str_builder_const_str(pl_str_builder b, const char *str)
{
    pl_str_builder_append(b, template_str_ptr, &str, sizeof(str));
}

static size_t template_str(void *alloc, pl_str *buf, const uint8_t *args)
{
    pl_str str;
    memcpy(&str.len, args, sizeof(str.len));
    pl_str_append_raw(alloc, buf, args + sizeof(str.len), str.len);
    return sizeof(str.len) + str.len;
}

void pl_str_builder_str(pl_str_builder b, const pl_str str)
{
    pl_str_builder_append(b, template_str, &str.len, sizeof(str.len));
    pl_str_append_raw(b, &b->args, str.buf, str.len);
}

void pl_str_builder_printf_c(pl_str_builder b, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    pl_str_builder_vprintf_c(b, fmt, ap);
    va_end(ap);
}

static size_t template_printf(void *alloc, pl_str *str, const uint8_t *args)
{
    const char *fmt;
    memcpy(&fmt, args, sizeof(fmt));
    args += sizeof(fmt);

    return sizeof(fmt) + pl_str_append_memprintf_c(alloc, str, fmt, args);
}

void pl_str_builder_vprintf_c(pl_str_builder b, const char *fmt, va_list ap)
{
    pl_str_builder_append(b, template_printf, &fmt, sizeof(fmt));

    // Push all of the variadic arguments directly onto `b->args`
    for (const char *c; (c = strchr(fmt, '%')) != NULL; fmt = c + 1) {
        c++;
        switch (c[0]) {
#define WRITE(T, x) pl_str_append_raw(b, &b->args, &(T) {x}, sizeof(T))
        case '%': continue;
        case 'c': WRITE(char,       va_arg(ap, int)); break;
        case 'd': WRITE(int,        va_arg(ap, int)); break;
        case 'u': WRITE(unsigned,   va_arg(ap, unsigned)); break;
        case 'f': WRITE(double,     va_arg(ap, double)); break;
        case 'h':
            assert(c[1] == 'x');
            WRITE(unsigned short, va_arg(ap, unsigned));
            c++;
            break;
        case 'l':
            assert(c[1] == 'l');
            switch (c[2]) {
            case 'u': WRITE(long long unsigned, va_arg(ap, long long unsigned)); break;
            case 'd': WRITE(long long int,      va_arg(ap, long long int)); break;
            default: abort();
            }
            c += 2;
            break;
        case 'z':
            assert(c[1] == 'u');
            WRITE(size_t, va_arg(ap, size_t));
            c++;
            break;
        case 's': {
            pl_str str = pl_str0(va_arg(ap, const char *));
            pl_str_append(b, &b->args, str);
            b->args.len++; // expand to include \0 byte (from pl_str_append)
            break;
        }
        case '.': {
            assert(c[1] == '*');
            assert(c[2] == 's');
            int len = va_arg(ap, int);
            const char *str = va_arg(ap, const char *);
            WRITE(int, len);
            pl_str_append_raw(b, &b->args, str, len);
            c += 2;
            break;
        }
        default:
            fprintf(stderr, "Invalid conversion character: '%c'!\n", c[0]);
            abort();
        }
#undef WRITE
    }
}
