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

#include "os.h"

enum pl_mutex_type {
    PL_MUTEX_NORMAL = 0,
    PL_MUTEX_RECURSIVE,
};

#define pl_mutex_init(mutex) \
    pl_mutex_init_type(mutex, PL_MUTEX_NORMAL)

// Note: This is never compiled, and only documents the API. The actual
// implementations of these prototypes may be macros.
#ifdef PL_API_REFERENCE

typedef void pl_mutex;
void pl_mutex_init_type(pl_mutex *mutex, enum pl_mutex_type mtype);
int pl_mutex_destroy(pl_mutex *mutex);
int pl_mutex_lock(pl_mutex *mutex);
int pl_mutex_unlock(pl_mutex *mutex);

typedef void pl_cond;
int pl_cond_init(pl_cond *cond);
int pl_cond_destroy(pl_cond *cond);
int pl_cond_broadcast(pl_cond *cond);
int pl_cond_signal(pl_cond *cond);

// `timeout` is in nanoseconds, or UINT64_MAX to block forever
int pl_cond_timedwait(pl_cond *cond, pl_mutex *mutex, uint64_t timeout);
int pl_cond_wait(pl_cond *cond, pl_mutex *mutex);

typedef void pl_static_mutex;
#define PL_STATIC_MUTEX_INITIALIZER
int pl_static_mutex_lock(pl_static_mutex *mutex);
int pl_static_mutex_unlock(pl_static_mutex *mutex);

typedef void pl_thread;
#define PL_THREAD_VOID void
#define PL_THREAD_RETURN() return
int pl_thread_create(pl_thread *thread, PL_THREAD_VOID (*fun)(void *), void *arg);
int pl_thread_join(pl_thread thread);

// Returns true if slept the full time, false otherwise
bool pl_thread_sleep(double t);

#endif

// Actual platform-specific implementation
#ifdef PL_HAVE_WIN32
#include "pl_thread_win32.h"
#elif defined(PL_HAVE_PTHREAD)
#include "pl_thread_pthread.h"
#else
#error No threading implementation available!
#endif
