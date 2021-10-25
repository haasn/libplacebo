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

#include <pthread.h>
#include <time.h>

typedef pthread_mutex_t pl_mutex;
typedef pthread_cond_t  pl_cond;

enum pl_mutex_type {
    PL_MUTEX_NORMAL = 0,
    PL_MUTEX_RECURSIVE,
};

static inline int pl_mutex_init_type_internal(pl_mutex *mutex, enum pl_mutex_type mtype)
{
    int mutex_type;
    switch (mtype) {
        case PL_MUTEX_RECURSIVE:
            mutex_type = PTHREAD_MUTEX_RECURSIVE;
            break;
        case PL_MUTEX_NORMAL:
        default:
        #ifndef NDEBUG
            mutex_type = PTHREAD_MUTEX_ERRORCHECK;
        #else
            mutex_type = PTHREAD_MUTEX_DEFAULT;
        #endif
            break;
    }

    int ret = 0;
    pthread_mutexattr_t attr;
    ret = pthread_mutexattr_init(&attr);
    if (ret != 0)
        return ret;

    pthread_mutexattr_settype(&attr, mutex_type);
    ret = pthread_mutex_init(mutex, &attr);
    pthread_mutexattr_destroy(&attr);
    return ret;
}

#define pl_mutex_init_type(mutex, mtype) \
    PL_CHECK_ERR(pl_mutex_init_type_internal(mutex, mtype))

#define pl_mutex_destroy    pthread_mutex_destroy
#define pl_mutex_lock       pthread_mutex_lock
#define pl_mutex_unlock     pthread_mutex_unlock

#define pl_cond_init(cond)  pthread_cond_init(cond, NULL)
#define pl_cond_destroy     pthread_cond_destroy
#define pl_cond_broadcast   pthread_cond_broadcast
#define pl_cond_signal      pthread_cond_signal
#define pl_cond_wait        pthread_cond_wait

static inline int pl_cond_timedwait(pl_cond *cond, pl_mutex *mutex, uint64_t timeout)
{
    if (timeout == UINT64_MAX)
        return pthread_cond_wait(cond, mutex);

    return pthread_cond_timedwait(cond, mutex, &(struct timespec) {
            .tv_sec  = (timeout) / 1000000000LLU,
            .tv_nsec = (timeout) % 1000000000LLU,
        });
}
