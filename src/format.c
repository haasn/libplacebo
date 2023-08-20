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
 * License along with libplacebo.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <math.h>

#include "common.h"

void pl_str_append_asprintf_c(void *alloc, pl_str *str, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    pl_str_append_vasprintf_c(alloc, str, fmt, ap);
    va_end(ap);
}

void pl_str_append_vasprintf_c(void *alloc, pl_str *str, const char *fmt,
                               va_list ap)
{
    for (const char *c; (c = strchr(fmt, '%')) != NULL; fmt = c + 1) {
        // Append the preceding string literal
        pl_str_append_raw(alloc, str, fmt, c - fmt);
        c++; // skip '%'

        char buf[32];
        int len;

        // The format character follows the % sign
        switch (c[0]) {
        case '%':
            pl_str_append_raw(alloc, str, c, 1);
            continue;
        case 's': {
            const char *arg = va_arg(ap, const char *);
            pl_str_append_raw(alloc, str, arg, strlen(arg));
            continue;
        }
        case '.': { // only used for %.*s
            assert(c[1] == '*');
            assert(c[2] == 's');
            len = va_arg(ap, int);
            pl_str_append_raw(alloc, str, va_arg(ap, char *), len);
            c += 2; // skip '*s'
            continue;
        }
        case 'c':
            buf[0] = (char) va_arg(ap, int);
            len = 1;
            break;
        case 'd':
            len = pl_str_print_int(buf, sizeof(buf), va_arg(ap, int));
            break;
        case 'h': ; // only used for %hx
            assert(c[1] == 'x');
            len = pl_str_print_hex(buf, sizeof(buf), (unsigned short) va_arg(ap, unsigned int));
            c++;
            break;
        case 'u':
            len = pl_str_print_uint(buf, sizeof(buf), va_arg(ap, unsigned int));
            break;
        case 'l':
            assert(c[1] == 'l');
            switch (c[2]) {
            case 'u':
                len = pl_str_print_uint64(buf, sizeof(buf), va_arg(ap, unsigned long long));
                break;
            case 'd':
                len = pl_str_print_int64(buf, sizeof(buf), va_arg(ap, long long));
                break;
            default: pl_unreachable();
            }
            c += 2;
            break;
        case 'z':
            assert(c[1] == 'u');
            len = pl_str_print_uint64(buf, sizeof(buf), va_arg(ap, size_t));
            c++;
            break;
        case 'f':
            len = pl_str_print_double(buf, sizeof(buf), va_arg(ap, double));
            break;
        default:
            fprintf(stderr, "Invalid conversion character: '%c'!\n", c[0]);
            abort();
        }

        pl_str_append_raw(alloc, str, buf, len);
    }

    // Append the remaining string literal
    pl_str_append(alloc, str, pl_str0(fmt));
}

size_t pl_str_append_memprintf_c(void *alloc, pl_str *str, const char *fmt,
                                 const void *args)
{
    const uint8_t *ptr = args;

    for (const char *c; (c = strchr(fmt, '%')) != NULL; fmt = c + 1) {
        pl_str_append_raw(alloc, str, fmt, c - fmt);
        c++;

        char buf[32];
        int len;

#define LOAD(var)                           \
  do {                                      \
      memcpy(&(var), ptr, sizeof(var));     \
      ptr += sizeof(var);                   \
  } while (0)

        switch (c[0]) {
        case '%':
            pl_str_append_raw(alloc, str, c, 1);
            continue;
        case 's': {
            len = strlen((const char *) ptr);
            pl_str_append_raw(alloc, str, ptr, len);
            ptr += len + 1; // also skip \0
            continue;
        }
        case '.': {
            assert(c[1] == '*');
            assert(c[2] == 's');
            LOAD(len);
            pl_str_append_raw(alloc, str, ptr, len);
            ptr += len; // no trailing \0
            c += 2;
            continue;
        }
        case 'c':
            LOAD(buf[0]);
            len = 1;
            break;
        case 'd': ;
            int d;
            LOAD(d);
            len = pl_str_print_int(buf, sizeof(buf), d);
            break;
        case 'h': ;
            assert(c[1] == 'x');
            unsigned short hx;
            LOAD(hx);
            len = pl_str_print_hex(buf, sizeof(buf), hx);
            c++;
            break;
        case 'u': ;
            unsigned u;
            LOAD(u);
            len = pl_str_print_uint(buf, sizeof(buf), u);
            break;
        case 'l':
            assert(c[1] == 'l');
            switch (c[2]) {
            case 'u': ;
                long long unsigned llu;
                LOAD(llu);
                len = pl_str_print_uint64(buf, sizeof(buf), llu);
                break;
            case 'd': ;
                long long int lld;
                LOAD(lld);
                len = pl_str_print_int64(buf, sizeof(buf), lld);
                break;
            default: pl_unreachable();
            }
            c += 2;
            break;
        case 'z': ;
            assert(c[1] == 'u');
            size_t zu;
            LOAD(zu);
            len = pl_str_print_uint64(buf, sizeof(buf), zu);
            c++;
            break;
        case 'f': ;
            double f;
            LOAD(f);
            len = pl_str_print_double(buf, sizeof(buf), f);
            break;
        default:
            fprintf(stderr, "Invalid conversion character: '%c'!\n", c[0]);
            abort();
        }

        pl_str_append_raw(alloc, str, buf, len);
    }
#undef LOAD

    pl_str_append(alloc, str, pl_str0(fmt));
    return (uintptr_t) ptr - (uintptr_t) args;
}
