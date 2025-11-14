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

#include <charconv>
#include <limits>
#include <system_error>

#if __has_include(<fast_float/fast_float.h>)
# include <fast_float/fast_float.h>
#endif

#include "pl_string.h"

[[maybe_unused]]
static int ccStrPrintDouble( char *str, int bufsize, int decimals, double value );

namespace {

template <typename T>
struct has_std_to_chars_impl {
    template <typename CT>
    static auto _(CT s) -> decltype(std::to_chars(s, s, std::declval<T>()), std::true_type{});
    static auto _(...) -> std::false_type;
    static constexpr bool value = decltype(_((char *){}))::value;
};

template <typename T>
constexpr bool has_std_to_chars = has_std_to_chars_impl<T>::value;

#if defined(HAS_STD_TO_CHARS_FP) && HAS_STD_TO_CHARS_FP == 0
template <>
constexpr bool has_std_to_chars<float>  = false;
template <>
constexpr bool has_std_to_chars<double> = false;
#endif

template <typename T, typename... Args>
static inline int to_chars(char *buf, size_t len, T n, Args ...args)
{
    if constexpr (has_std_to_chars<T>) {
        auto [ptr, ec] = std::to_chars(buf, buf + len, n, args...);
        return ec == std::errc() ? ptr - buf : 0;
    } else {
        static_assert(std::is_same_v<float, T> || std::is_same_v<double, T>,
                      "Not implemented!");
        // FIXME: Fallback for GCC <= 10 currently required for MinGW-w64 on
        // Ubuntu 22.04. Remove this when Ubuntu 24.04 is released, as it will
        // provide newer MinGW-w64 GCC and it will be safe to require it.
        return ccStrPrintDouble(buf, len, std::numeric_limits<T>::max_digits10, n);
    }
}

template <typename T>
struct has_std_from_chars_impl {
    template <typename CT>
    static auto _(CT s) -> decltype(std::from_chars(s, s, std::declval<T&>()), std::true_type{});
    static auto _(...) -> std::false_type;
    static constexpr bool value = decltype(_((const char *){}))::value;
};

template <typename T>
constexpr bool has_std_from_chars = has_std_from_chars_impl<T>::value;

template <typename T, typename... Args>
static inline bool from_chars(pl_str str, T &n, Args ...args)
{
    if constexpr (has_std_from_chars<T>) {
        auto [ptr, ec] = std::from_chars((const char *) str.buf,
                                         (const char *) str.buf + str.len,
                                         n, args...);
        return ec == std::errc();
    } else {
        constexpr bool is_fp = std::is_same_v<float, T> || std::is_same_v<double, T>;
        static_assert(is_fp, "Not implemented!");
#if !__has_include(<fast_float/fast_float.h>)
        static_assert(!is_fp, "<fast_float/fast_float.h> is required, but not " \
                              "found. Please run `git submodule update --init`" \
                              " or provide <fast_float/fast_float.h>");
#else
        // FIXME: Fallback for libc++, as it does not implement floating-point
        // variant of std::from_chars. Remove this when appropriate.
        auto [ptr, ec] = fast_float::from_chars((const char *) str.buf,
                                                (const char *) str.buf + str.len,
                                                n, args...);
        return ec == std::errc();
#endif
    }
}

}

#define CHAR_CONVERT(name, type, ...)                           \
    int pl_str_print_##name(char *buf, size_t len, type n)      \
    {                                                           \
        return to_chars(buf, len, n __VA_OPT__(,) __VA_ARGS__); \
    }                                                           \
    bool pl_str_parse_##name(pl_str str, type *n)               \
    {                                                           \
        return from_chars(str, *n __VA_OPT__(,) __VA_ARGS__);   \
    }

CHAR_CONVERT(hex, unsigned short, 16)
CHAR_CONVERT(int, int)
CHAR_CONVERT(uint, unsigned int)
CHAR_CONVERT(int64, int64_t)
CHAR_CONVERT(uint64, uint64_t)
CHAR_CONVERT(float, float)
CHAR_CONVERT(double, double)

/* *****************************************************************************
 *
 * Copyright (c) 2007-2016 Alexis Naveros.
 * Modified for use with libplacebo by Niklas Haas
 * Changes include:
 *  - Removed a CC_MIN macro dependency by equivalent logic
 *  - Removed CC_ALWAYSINLINE
 *  - Fixed (!seq) check to (!seqlength)
 *  - Added support for scientific notation (e.g. 1.0e10) in ccSeqParseDouble
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 * claim that you wrote the original software. If you use this software
 * in a product, an acknowledgment in the product documentation would be
 * appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 * misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 * -----------------------------------------------------------------------------
 */

static int ccStrPrintDouble( char *str, int bufsize, int decimals, double value )
{
    int size, offset, index;
    int32_t frac, accumsub;
    double muldec;
    uint32_t u32;
    uint64_t u64;

    size = 0;
    if( value < 0.0 )
    {
        size = 1;
        *str++ = '-';
        bufsize--;
        value = -value;
    }

    if( value < 4294967296.0 )
    {
        u32 = (uint32_t)value;
        offset = pl_str_print_uint( str, bufsize, u32 );
        if (!offset)
            goto error;
        size += offset;
        bufsize -= size;
        value -= (double)u32;
    }
    else if( value < 18446744073709551616.0 )
    {
        u64 = (uint64_t)value;
        offset = pl_str_print_uint64( str, bufsize, u64 );
        if (!offset)
            goto error;
        size += offset;
        bufsize -= size;
        value -= (double)u64;
    }
    else
        goto error;

    if (decimals > bufsize - 2)
        decimals = bufsize - 2;
    if( decimals <= 0 )
        return size;

    muldec = 10.0;
    accumsub = 0;
    str += offset;

    for( index = 0 ; index < decimals ; index++ )
    {
        // Skip printing insignificant decimal digits
        if (value * muldec - accumsub <= std::numeric_limits<double>::epsilon())
            break;
        if (index == 0) {
            size += 1;
            *str++ = '.';
        }
        frac = (int32_t)( value * muldec ) - accumsub;
        frac = PL_CLAMP(frac, 0, 9); // FIXME: why is this needed?
        str[index] = '0' + (char)frac;
        accumsub += frac;
        accumsub = ( accumsub << 3 ) + ( accumsub << 1 );
        if( muldec < 10000000 )
            muldec *= 10.0;
        else
        {
            value *= 10000000.0;
            value -= (int32_t)value;
            muldec = 10.0;
            accumsub = 0;
        }
    }
    // Round up the last decimal digit
    if ( str[ index - 1 ] < '9' && (int32_t)( value * muldec ) - accumsub >= 5 )
        str[ index - 1 ]++;
    str[ index ] = 0;
    size += index;
    return size;

error:
    if( bufsize < 4 )
        *str = 0;
    else
    {
        str[0] = 'E';
        str[1] = 'R';
        str[2] = 'R';
        str[3] = 0;
    }
    return 0;
}
