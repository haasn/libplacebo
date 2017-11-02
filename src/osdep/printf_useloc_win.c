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

#include <stdlib.h>
#include <locale.h>

#include "osdep/printf.h"

static _locale_t cloc;

void printf_c_init()
{
    cloc = _create_locale(LC_ALL, "C");
    if (!cloc)
        abort();
}

void printf_c_uninit()
{
    _free_locale(cloc);
    cloc = (_locale_t) 0;
}

int vprintf_c(const char *format, va_list ap)
{
    return _vprintf_l(format, cloc, ap);
}

int vfprintf_c(FILE *stream, const char *format, va_list ap)
{
    return _vfprintf_l(stream, format, cloc, ap);
}

int vsprintf_c(char *str, const char *format, va_list ap)
{
    return _vsprintf_l(str, format, cloc, ap);
}

int vsnprintf_c(char *str, size_t size, const char *format, va_list ap)
{
    return _vsnprintf_l(str, size, format, cloc, ap);
}

#define WRAP(fn, ...)                               \
    ({                                              \
        va_list ap;                                 \
        va_start(ap, format);                       \
        int ret = fn(__VA_ARGS__, ap);              \
        va_end(ap);                                 \
        ret;                                        \
    })

int printf_c(const char *format, ...)
{
    return WRAP(vprintf_c, format);
}

int fprintf_c(FILE *stream, const char *format, ...)
{
    return WRAP(vfprintf_c, stream, format);
}

int sprintf_c(char *str, const char *format, ...)
{
    return WRAP(vsprintf_c, str, format);
}

int snprintf_c(char *str, size_t size, const char *format, ...)
{
    return WRAP(vsnprintf_c, str, size, format);
}

