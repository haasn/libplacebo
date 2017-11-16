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

#define _XOPEN_SOURCE 700

#include <stdlib.h>
#include <locale.h>

#ifdef __APPLE__
# include <string.h>
# include <xlocale.h>
#endif

#include "osdep/printf.h"

static locale_t cloc;

void printf_c_init()
{
    cloc = newlocale(LC_NUMERIC_MASK, "C", (locale_t) 0);
    if (!cloc)
        abort();
}

void printf_c_uninit()
{
    freelocale(cloc);
    cloc = (locale_t) 0;
}

#define WRAP_VA(fn, ...)                            \
    ({                                              \
        locale_t oldloc = uselocale((locale_t) 0);  \
        uselocale(cloc);                            \
        int ret_va = fn(__VA_ARGS__);               \
        uselocale(oldloc);                          \
        ret_va;                                     \
    })

#define WRAP(fn, ...)                               \
    ({                                              \
        va_list ap;                                 \
        va_start(ap, format);                       \
        int ret = WRAP_VA(v##fn, __VA_ARGS__, ap);  \
        va_end(ap);                                 \
        ret;                                        \
    })

int printf_c(const char *format, ...)
{
    return WRAP(printf, format);
}

int fprintf_c(FILE *stream, const char *format, ...)
{
    return WRAP(fprintf, stream, format);
}

int sprintf_c(char *str, const char *format, ...)
{
    return WRAP(sprintf, str, format);
}

int snprintf_c(char *str, size_t size, const char *format, ...)
{
    return WRAP(snprintf, str, size, format);
}

int vprintf_c(const char *format, va_list ap)
{
    return WRAP_VA(vprintf, format, ap);
}

int vfprintf_c(FILE *stream, const char *format, va_list ap)
{
    return WRAP_VA(vfprintf, stream, format, ap);
}

int vsprintf_c(char *str, const char *format, va_list ap)
{
    return WRAP_VA(vsprintf, str, format, ap);
}

int vsnprintf_c(char *str, size_t size, const char *format, va_list ap)
{
    return WRAP_VA(vsnprintf, str, size, format, ap);
}
