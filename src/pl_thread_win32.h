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

#include <windows.h>
#include <errno.h>

typedef CRITICAL_SECTION   pl_mutex;
typedef CONDITION_VARIABLE pl_cond;

static inline int pl_mutex_init_type_internal(pl_mutex *mutex, enum pl_mutex_type mtype)
{
    return !InitializeCriticalSectionEx(mutex, 0, 0);
}

#define pl_mutex_init_type(mutex, mtype) \
    PL_CHECK_ERR(pl_mutex_init_type_internal(mutex, mtype))

#define pl_mutex_init(mutex) \
    pl_mutex_init_type(mutex, PL_MUTEX_NORMAL)

static inline int pl_mutex_destroy(pl_mutex *mutex)
{
    DeleteCriticalSection(mutex);
    return 0;
}

static inline int pl_mutex_lock(pl_mutex *mutex)
{
    EnterCriticalSection(mutex);
    return 0;
}

static inline int pl_mutex_unlock(pl_mutex *mutex)
{
    LeaveCriticalSection(mutex);
    return 0;
}

static inline int pl_cond_init(pl_cond *cond)
{
    InitializeConditionVariable(cond);
    return 0;
}

static inline int pl_cond_destroy(pl_cond *cond)
{
    // condition variables are not destroyed
    return 0;
}

static inline int pl_cond_broadcast(pl_cond *cond)
{
    WakeAllConditionVariable(cond);
    return 0;
}

static inline int pl_cond_signal(pl_cond *cond)
{
    WakeConditionVariable(cond);
    return 0;
}

static inline int pl_cond_wait(pl_cond *cond, pl_mutex *mutex)
{
    return !SleepConditionVariableCS(cond, mutex, INFINITE);
}

static inline int pl_cond_timedwait(pl_cond *cond, pl_mutex *mutex, uint64_t timeout)
{
    if (timeout == UINT64_MAX)
        return pl_cond_wait(cond, mutex);

    BOOL bRet = SleepConditionVariableCS(cond, mutex, PL_MIN(timeout / 1000000LLU, INFINITE - 1));
    if (bRet == FALSE)
    {
        if (GetLastError() == ERROR_TIMEOUT)
            return ETIMEDOUT;
        else
            return EINVAL;
    }
    return 0;
}
