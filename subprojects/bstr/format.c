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

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#include <bstr.h>

int ccStrPrintInt32( char *str, int32_t n );
int ccStrPrintUint32( char *str, uint32_t n );
int ccStrPrintInt64( char *str, int64_t n );
int ccStrPrintUint64( char *str, uint64_t n );
int ccStrPrintDouble( char *str, int bufsize, int decimals, double value );

#define CC_STR_PRINT_BUFSIZE_INT32 (12)
#define CC_STR_PRINT_BUFSIZE_UINT32 (11)
#define CC_STR_PRINT_BUFSIZE_INT64 (21)
#define CC_STR_PRINT_BUFSIZE_UINT64 (20)

void bstr_xappend_asprintf_c(void *tactx, bstr *s, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    bstr_xappend_vasprintf_c(tactx, s, fmt, ap);
    va_end(ap);
}

void bstr_xappend_vasprintf_c(void *tactx, bstr *s, const char *fmt,
                              va_list ap)
{
    for (const char *c; (c = strchr(fmt, '%')) != NULL; fmt = c + 1) {
        // Append the preceding string literal
        bstr_xappend(tactx, s, (struct bstr) { (char *) fmt, c - fmt });
        c++; // skip '%'

        char buf[32];
        int len;

        // The format character follows the % sign
        switch (c[0]) {
        case '%':
            bstr_xappend(tactx, s, bstr0("%"));
            continue;
        case 'c':
            buf[0] = (char) va_arg(ap, int);
            bstr_xappend(tactx, s, (struct bstr) { buf, 1 });
            continue;
        case 's': {
            const char *arg = va_arg(ap, const char *);
            bstr_xappend(tactx, s, bstr0(arg));
            continue;
        }
        case '.': { // only used for %.*s
            assert(c[1] == '*');
            assert(c[2] == 's');
            struct bstr str;
            str.len = va_arg(ap, int);
            str.start = va_arg(ap, char *);
            bstr_xappend(tactx, s, str);
            c += 2; // skip '*s'
            continue;
        }
        case 'd':
            len = ccStrPrintInt32(buf, va_arg(ap, int));
            bstr_xappend(tactx, s, (struct bstr) { buf, len });
            continue;
        case 'u':
            len = ccStrPrintUint32(buf, va_arg(ap, unsigned int));
            bstr_xappend(tactx, s, (struct bstr) { buf, len });
            continue;
        case 'z':
            assert(c[1] == 'u');
            len = ccStrPrintUint64(buf, va_arg(ap, size_t));
            bstr_xappend(tactx, s, (struct bstr) { buf, len });
            c++;
            continue;
        case 'f':
            len = ccStrPrintDouble(buf, sizeof(buf), 6, va_arg(ap, double));
            bstr_xappend(tactx, s, (struct bstr) { buf, len });
            continue;
        default:
            fprintf(stderr, "Invalid conversion character: '%c'!\n", c[0]);
            abort();
        }
    }

    // Append the remaining string literal
    bstr_xappend(tactx, s, bstr0(fmt));
}

/* *****************************************************************************
 *
 * Copyright (c) 2007-2016 Alexis Naveros.
 * Modified for use with libplacebo by Niklas Haas
 * Changes include:
 *  - Removed a CC_MIN macro dependency by equivalent logic
 *  - Removed CC_ALWAYSINLINE
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

int ccStrPrintInt32( char *str, int32_t n )
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

int ccStrPrintUint32( char *str, uint32_t n )
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

int ccStrPrintInt64( char *str, int64_t n )
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

int ccStrPrintUint64( char *str, uint64_t n )
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

#define CC_STR_PRINT_DOUBLE_MAX_DECIMAL (24)
 
static const double ccStrPrintBiasTable[CC_STR_PRINT_DOUBLE_MAX_DECIMAL+1] =
{ 0.5, 0.05, 0.005, 0.0005, 0.00005, 0.000005, 0.0000005, 0.00000005, 0.000000005, 0.0000000005, 0.00000000005, 0.000000000005, 0.0000000000005, 0.00000000000005, 0.000000000000005, 0.0000000000000005, 0.00000000000000005, 0.000000000000000005, 0.0000000000000000005, 0.00000000000000000005, 0.000000000000000000005, 0.0000000000000000000005, 0.00000000000000000000005, 0.000000000000000000000005, 0.0000000000000000000000005 };
 
int ccStrPrintDouble( char *str, int bufsize, int decimals, double value )
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
