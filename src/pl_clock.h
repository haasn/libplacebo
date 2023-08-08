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

#include <time.h>
#include <stdint.h>

#include "os.h"

#ifdef PL_HAVE_WIN32
# include <windows.h>
# define PL_CLOCK_QPC
#elif defined(PL_HAVE_APPLE)
# include <Availability.h>
# if (defined(__MAC_OS_X_VERSION_MIN_REQUIRED)  && __MAC_OS_X_VERSION_MIN_REQUIRED  < 101200) || \
     (defined(__IPHONE_OS_VERSION_MIN_REQUIRED) && __IPHONE_OS_VERSION_MIN_REQUIRED < 100000) || \
     (defined(__TV_OS_VERSION_MIN_REQUIRED)     && __TV_OS_VERSION_MIN_REQUIRED     < 100000) || \
     (defined(__WATCH_OS_VERSION_MIN_REQUIRED)  && __WATCH_OS_VERSION_MIN_REQUIRED  < 30000)  || \
     !defined(CLOCK_MONOTONIC_RAW)
#  include <mach/mach_time.h>
#  define PL_CLOCK_MACH
# else
#  define PL_CLOCK_MONOTONIC_RAW
# endif
#elif defined(CLOCK_MONOTONIC_RAW)
# define PL_CLOCK_MONOTONIC_RAW
#elif defined(TIME_UTC)
# define PL_CLOCK_TIMESPEC_GET
#else
# warning "pl_clock not implemented for this platform!"
#endif

typedef uint64_t pl_clock_t;

static inline pl_clock_t pl_clock_now(void)
{
#if defined(PL_CLOCK_QPC)

    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return counter.QuadPart;

#elif defined(PL_CLOCK_MACH)

    return mach_absolute_time();

#else

    struct timespec tp = { .tv_sec = 0, .tv_nsec = 0 };
#if defined(PL_CLOCK_MONOTONIC_RAW)
    clock_gettime(CLOCK_MONOTONIC_RAW, &tp);
#elif defined(PL_CLOCK_TIMESPEC_GET)
    timespec_get(&tp, TIME_UTC);
#endif
    return tp.tv_sec * UINT64_C(1000000000) + tp.tv_nsec;

#endif
}

static inline double pl_clock_diff(pl_clock_t a, pl_clock_t b)
{
    double frequency = 1e9;

#if defined(PL_CLOCK_QPC)

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    frequency = freq.QuadPart;

#elif defined(PL_CLOCK_MACH)

    mach_timebase_info_data_t time_base;
    if (mach_timebase_info(&time_base) != KERN_SUCCESS)
        return 0;
    frequency = (time_base.denom * 1e9) / time_base.numer;

#endif

    if (b > a)
        return (b - a) / -frequency;
    else
        return (a - b) / frequency;
}
