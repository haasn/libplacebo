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
#include <cmath>
#include <limits>
#include <system_error>

#include <fast_float/fast_float.h>

#include "pl_string.h"

extern "C" int ccStrPrintDouble( char *str, int bufsize, int decimals, double value );

namespace {

template <typename T>
concept has_std_to_chars = requires(char *begin, char *end, T &n)
{
    std::to_chars(begin, end, n);
};

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
concept has_std_from_chars = requires(const char *begin, const char *end, T &n)
{
    std::from_chars(begin, end, n);
};

template <typename T, typename... Args>
static inline bool from_chars(pl_str str, T &n, Args ...args)
{
    if constexpr (has_std_from_chars<T>) {
        auto [ptr, ec] = std::from_chars((const char *) str.buf,
                                         (const char *) str.buf + str.len,
                                         n, args...);
        return ec == std::errc();
    } else {
        static_assert(std::is_same_v<float, T> || std::is_same_v<double, T>,
                      "Not implemented!");
        // FIXME: Fallback for libc++, as it does not implement floating-point
        // variant of std::from_chars. Remove this when appropriate.
        auto [ptr, ec] = fast_float::from_chars((const char *) str.buf,
                                                (const char *) str.buf + str.len,
                                                n, args...);
        return ec == std::errc();
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

// CHAR_CONVERT(hex, unsigned short, 16)
// CHAR_CONVERT(int, int)
// CHAR_CONVERT(uint, unsigned int)
// CHAR_CONVERT(int64, int64_t)
// CHAR_CONVERT(uint64, uint64_t)
// CHAR_CONVERT(float, float)
// CHAR_CONVERT(double, double)

extern "C" int print_hex(char *buf, unsigned int x)
{
    static const char hexdigits[] = "0123456789abcdef";
    const int nibbles0 = __builtin_clz(x | 1) >> 2;
    buf -= nibbles0;

    switch (nibbles0) {
    pl_static_assert(sizeof(unsigned int) == sizeof(uint32_t));
    case 0: buf[0] = hexdigits[(x >> 28) & 0xF]; // fall through
    case 1: buf[1] = hexdigits[(x >> 24) & 0xF]; // fall through
    case 2: buf[2] = hexdigits[(x >> 20) & 0xF]; // fall through
    case 3: buf[3] = hexdigits[(x >> 16) & 0xF]; // fall through
    case 4: buf[4] = hexdigits[(x >> 12) & 0xF]; // fall through
    case 5: buf[5] = hexdigits[(x >>  8) & 0xF]; // fall through
    case 6: buf[6] = hexdigits[(x >>  4) & 0xF]; // fall through
    case 7: buf[7] = hexdigits[(x >>  0) & 0xF];
            return 8 - nibbles0;
    }

    pl_unreachable();
}

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

static const char ccStrPrintDecimalTable[201] =
{
  "00010203040506070809"
  "10111213141516171819"
  "20212223242526272829"
  "30313233343536373839"
  "40414243444546474849"
  "50515253545556575859"
  "60616263646566676869"
  "70717273747576777879"
  "80818283848586878889"
  "90919293949596979899"
};

static inline int ccStrPrintLength32( uint32_t n )
{
    int size;
    if( n >= 10000 )
    {
        if( n >= 10000000 )
        {
            if( n >= 1000000000 )
                size = 10;
            else if( n >= 100000000 )
                size = 9;
            else
                size = 8;
        }
        else if( n >= 1000000 )
            size = 7;
        else if( n >= 100000 )
            size = 6;
        else
            size = 5;
    }
    else
    {
        if( n >= 100 )
        {
            if( n >= 1000 )
                size = 4;
            else
                size = 3;
        }
        else if( n >= 10 )
            size = 2;
        else
            size = 1;
    }
    return size;
}

static inline int ccStrPrintLength64( uint64_t n )
{
    int size;
    if( n >= 10000 )
    {
        if( n >= 10000000 )
        {
            if( n >= 10000000000LL )
            {
                if( n >= 10000000000000LL )
                {
                    if( n >= 10000000000000000LL )
                    {
                        if( n >= 10000000000000000000ULL )
                            size = 20;
                        else if( n >= 1000000000000000000LL )
                            size = 19;
                        else if( n >= 100000000000000000LL )
                            size = 18;
                        else
                            size = 17;
                    }
                    else if( n >= 1000000000000000LL )
                        size = 16;
                    else if( n >= 100000000000000LL )
                        size = 15;
                    else
                        size = 14;
                }
                else if( n >= 1000000000000LL )
                    size = 13;
                else if( n >= 100000000000LL )
                    size = 12;
                else
                    size = 11;
            }
            else if( n >= 1000000000 )
                size = 10;
            else if( n >= 100000000 )
                size = 9;
            else
                size = 8;
        }
        else
        {
            if( n >= 1000000 )
                size = 7;
            else if( n >= 100000 )
                size = 6;
            else
                size = 5;
        }
    }
    else if( n >= 100 )
    {
        if( n >= 1000 )
            size = 4;
        else
            size = 3;
    }
    else if( n >= 10 )
        size = 2;
    else
        size = 1;
    return size;
}

extern "C" int ccStrPrintInt32( char *str, int32_t n )
{
    int sign, size, retsize, pos;
    uint32_t val32;
    const char *src;

    if( n == 0 )
    {
        str[0] = '0';
        str[1] = 0;
        return 1;
    }

    sign = -( n < 0 );
    val32 = ( n ^ sign ) - sign;
    size = ccStrPrintLength32( val32 );

    if( sign )
    {
        size++;
        str[0] = '-';
    }
    retsize = size;
    str[size] = 0;
    str += size - 1;

    while( val32 >= 100 )
    {
        pos = val32 % 100;
        val32 /= 100;
        src = &ccStrPrintDecimalTable[ pos << 1 ];
        str[-1] = src[0];
        str[0] = src[1];
        str -= 2;
    }
    while( val32 > 0 )
    {
        *str-- = '0' + ( val32 % 10 );
        val32 /= 10;
    }

    return retsize;
}

extern "C" int ccStrPrintUint32( char *str, uint32_t n )
{
    int size, retsize, pos;
    uint32_t val32;
    const char *src;

    if( n == 0 )
    {
        str[0] = '0';
        str[1] = 0;
        return 1;
    }

    val32 = n;
    size = ccStrPrintLength32( val32 );
    retsize = size;
    str[size] = 0;
    str += size - 1;

    while( val32 >= 100 )
    {
        pos = val32 % 100;
        val32 /= 100;
        src = &ccStrPrintDecimalTable[ pos << 1 ];
        str[-1] = src[0];
        str[0] = src[1];
        str -= 2;
    }
    while( val32 > 0 )
    {
        *str-- = '0' + ( val32 % 10 );
        val32 /= 10;
    }

    return retsize;
}

extern "C" int ccStrPrintInt64( char *str, int64_t n )
{
    int sign, size, retsize, pos;
    uint64_t val64;
    const char *src;

    if( n == 0 )
    {
        str[0] = '0';
        str[1] = 0;
        return 1;
    }

    sign = -( n < 0 );
    val64 = ( n ^ sign ) - sign;
    size = ccStrPrintLength64( val64 );

    if( sign )
    {
        size++;
        str[0] = '-';
    }
    retsize = size;
    str[size] = 0;
    str += size - 1;

    while( val64 >= 100 )
    {
        pos = val64 % 100;
        val64 /= 100;
        src = &ccStrPrintDecimalTable[ pos << 1 ];
        str[-1] = src[0];
        str[0] = src[1];
        str -= 2;
    }
    while( val64 > 0 )
    {
        *str-- = '0' + ( val64 % 10 );
        val64 /= 10;
    }

    return retsize;
}

extern "C" int ccStrPrintUint64( char *str, uint64_t n )
{
    int size, retsize, pos;
    uint64_t val64;
    const char *src;

    if( n == 0 )
    {
        str[0] = '0';
        str[1] = 0;
        return 1;
    }

    val64 = n;
    size = ccStrPrintLength64( val64 );

    retsize = size;
    str[size] = 0;
    str += size - 1;

    while( val64 >= 100 )
    {
        pos = val64 % 100;
        val64 /= 100;
        src = &ccStrPrintDecimalTable[ pos << 1 ];
        str[-1] = src[0];
        str[0] = src[1];
        str -= 2;
    }
    while( val64 > 0 )
    {
        *str-- = '0' + ( val64 % 10 );
        val64 /= 10;
    }

    return retsize;
}

#define CC_STR_PRINT_BUFSIZE_INT32 (12)
#define CC_STR_PRINT_BUFSIZE_UINT32 (11)
#define CC_STR_PRINT_BUFSIZE_INT64 (21)
#define CC_STR_PRINT_BUFSIZE_UINT64 (20)
#define CC_STR_PRINT_DOUBLE_MAX_DECIMAL (24)

static const double ccStrPrintBiasTable[CC_STR_PRINT_DOUBLE_MAX_DECIMAL+1] =
{ 0.5, 0.05, 0.005, 0.0005, 0.00005, 0.000005, 0.0000005, 0.00000005, 0.000000005, 0.0000000005, 0.00000000005, 0.000000000005, 0.0000000000005, 0.00000000000005, 0.000000000000005, 0.0000000000000005, 0.00000000000000005, 0.000000000000000005, 0.0000000000000000005, 0.00000000000000000005, 0.000000000000000000005, 0.0000000000000000000005, 0.00000000000000000000005, 0.000000000000000000000005, 0.0000000000000000000000005 };

extern "C" int ccStrPrintDouble( char *str, int bufsize, int decimals, double value )
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

    /* Add bias matching the count of desired decimals in order to round the right way */
    if( decimals > CC_STR_PRINT_DOUBLE_MAX_DECIMAL )
        decimals = CC_STR_PRINT_DOUBLE_MAX_DECIMAL;
    value += ccStrPrintBiasTable[decimals];

    if( value < 4294967296.0 )
    {
        if( bufsize < CC_STR_PRINT_BUFSIZE_UINT32 )
            goto error;
        u32 = (int32_t)value;
        offset = ccStrPrintUint32( str, u32 );
        size += offset;
        bufsize -= size;
        value -= (double)u32;
    }
    else if( value < 18446744073709551616.0 )
    {
        if( bufsize < CC_STR_PRINT_BUFSIZE_UINT64 )
            goto error;
        u64 = (int64_t)value;
        offset = ccStrPrintUint64( str, u64 );
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

    str[offset] = '.';
    muldec = 10.0;
    accumsub = 0;
    str += offset + 1;

    for( index = 0 ; index < decimals ; index++ )
    {
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
    str[ index ] = 0;
    size += index + 1;
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

#define CC_CHAR_IS_DELIMITER(c) ((c)<=' ')

extern "C" int ccSeqParseInt64( char *seq, int seqlength, int64_t *retint )
{
  int i, negflag;
  char c;
  int64_t workint;

  *retint = 0;
  if( !( seqlength ) )
    return 0;
  negflag = 0;
  i = 0;
  if( *seq == '-' )
  {
    negflag = 1;
    i = 1;
  } else if( *seq == '+' )
    i = 1;

  workint = 0;
  for( ; i < seqlength ; i++ )
  {
    c = seq[i];
    if( ( c >= '0' ) && ( c <= '9' ) )
    {
      if( workint >= (int64_t)0xcccccccccccccccLL )
        return 0;
      workint = ( workint * 10 ) + ( c - '0' );
    }
    else if( CC_CHAR_IS_DELIMITER( c ) )
      break;
    else
      return 0;
  }

  if( negflag )
    workint = -workint;
  *retint = workint;
  return 1;
}

extern "C" int ccSeqParseUint64( char *seq, int seqlength, uint64_t *retint )
{
  int i;
  char c;
  uint64_t workint;

  *retint = 0;
  if( !( seqlength ) )
    return 0;
  i = 0;
  if( *seq == '+' )
    i = 1;

  workint = 0;
  for( ; i < seqlength ; i++ )
  {
    c = seq[i];
    if( ( c >= '0' ) && ( c <= '9' ) )
    {
      if( workint >= (uint64_t)0x1999999999999999LL )
        return 0;
      workint = ( workint * 10 ) + ( c - '0' );
    }
    else if( CC_CHAR_IS_DELIMITER( c ) )
      break;
    else
      return 0;
  }

  *retint = workint;
  return 1;
}

// Function copied from musl libc exp10(), to avoid portability issues
// Copyright (c) 2005-2020 Rich Felker, et al.
// Available under the terms of the MIT license
static inline double ccExp10(double x)
{
    static const double p10[] = {
        1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10,
        1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
        1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
        1e10, 1e11, 1e12, 1e13, 1e14, 1e15
    };

    double n, y = modf(x, &n);
    union {double f; uint64_t i;} u = {n};
    /* fabs(n) < 16 without raising invalid on nan */
    if ((u.i>>52 & 0x7ff) < 0x3ff+4) {
        if (!y) return p10[(int)n+15];
        y = exp2(3.32192809488736234787031942948939 * y);
        return y * p10[(int)n+15];
    }
    return pow(10.0, x);
}

extern "C" int ccSeqParseDouble( char *seq, int seqlength, double *retdouble )
{
  int i, negflag;
  char c;
  double accum;
  double decfactor;
  int64_t exponent;

  *retdouble = 0.0;
  i = 0;
  if( !( seqlength ) )
    return 0;
  negflag = ( seq[i] == '-' );
  i += negflag;

  accum = 0.0;
  for( ; i < seqlength ; i++ )
  {
    c = seq[i];
    if( ( c >= '0' ) && ( c <= '9' ) )
      accum = ( accum * 10.0 ) + (double)( c - '0' );
    else if( CC_CHAR_IS_DELIMITER( c ) )
      goto done;
    else if( c == 'e' || c == 'E' )
      goto sci;
    else if( c == '.' )
      break;
    else
      return 0;
  }

  i++;
  decfactor = 0.1;
  for( ; i < seqlength ; i++ )
  {
    c = seq[i];
    if( ( c >= '0' ) && ( c <= '9' ) )
    {
      accum += (double)( c - '0' ) * decfactor;
      decfactor *= 0.1;
    }
    else if( CC_CHAR_IS_DELIMITER( c ) )
      goto done;
    else if( c == 'e' || c == 'E' )
      goto sci;
    else
      return 0;
  }

done:
  if( negflag )
    accum = -accum;
  *retdouble = (double)accum;
  return 1;

sci:
  i++;
  if( !ccSeqParseInt64( seq + i, seqlength - i, &exponent ) )
    return 0;
  accum *= ccExp10 ( exponent );
  goto done;
}
