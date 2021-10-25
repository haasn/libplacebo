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

#ifdef PL_HAVE_WIN32
#include "pl_thread_win32.h"
#elif defined(PL_HAVE_PTHREAD)
#include "pl_thread_pthread.h"
#else
#error No threading implementation available!
#endif

#define pl_mutex_init(mutex) \
    pl_mutex_init_type(mutex, PL_MUTEX_NORMAL)
