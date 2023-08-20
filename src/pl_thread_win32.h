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
#include <process.h>
#include <stdint.h>
#include <errno.h>

#include <pl_assert.h>

typedef CRITICAL_SECTION   pl_mutex;
typedef CONDITION_VARIABLE pl_cond;

static inline int pl_mutex_init_type_internal(pl_mutex *mutex, enum pl_mutex_type mtype)
{
    (void) mtype;
    return !InitializeCriticalSectionEx(mutex, 0, 0);
}

#define pl_mutex_init_type(mutex, mtype) \
    pl_assert(!pl_mutex_init_type_internal(mutex, mtype))

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
    (void) cond;
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

    timeout /= UINT64_C(1000000);
    if (timeout > INFINITE - 1)
        timeout = INFINITE - 1;

    BOOL bRet = SleepConditionVariableCS(cond, mutex, timeout);
    if (bRet == FALSE)
    {
        if (GetLastError() == ERROR_TIMEOUT)
            return ETIMEDOUT;
        else
            return EINVAL;
    }
    return 0;
}

typedef SRWLOCK pl_static_mutex;
#define PL_STATIC_MUTEX_INITIALIZER SRWLOCK_INIT

static inline int pl_static_mutex_lock(pl_static_mutex *mutex)
{
    AcquireSRWLockExclusive(mutex);
    return 0;
}

static inline int pl_static_mutex_unlock(pl_static_mutex *mutex)
{
    ReleaseSRWLockExclusive(mutex);
    return 0;
}

typedef HANDLE pl_thread;
#define PL_THREAD_VOID unsigned __stdcall
#define PL_THREAD_RETURN() return 0

static inline int pl_thread_create(pl_thread *thread,
                                   PL_THREAD_VOID (*fun)(void *),
                                   void *__restrict arg)
{
    *thread = (HANDLE) _beginthreadex(NULL, 0, fun, arg, 0, NULL);
    return *thread ? 0 : -1;
}

static inline int pl_thread_join(pl_thread thread)
{
    DWORD ret = WaitForSingleObject(thread, INFINITE);
    if (ret != WAIT_OBJECT_0)
        return ret == WAIT_ABANDONED ? EINVAL : EDEADLK;
    CloseHandle(thread);
    return 0;
}

static inline bool pl_thread_sleep(double t)
{
    // Time is expected in 100 nanosecond intervals.
    // Negative values indicate relative time.
    LARGE_INTEGER time = { .QuadPart = -(LONGLONG) (t * 1e7) };

    if (time.QuadPart >= 0)
        return true;

    bool ret = false;

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
# define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x2
#endif

    HANDLE timer = CreateWaitableTimerEx(NULL, NULL,
                                         CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                         TIMER_ALL_ACCESS);

    // CREATE_WAITABLE_TIMER_HIGH_RESOLUTION is supported in Windows 10 1803+,
    // retry without it.
    if (!timer)
        timer = CreateWaitableTimerEx(NULL, NULL, 0, TIMER_ALL_ACCESS);

    if (!timer)
        goto end;

    if (!SetWaitableTimer(timer, &time, 0, NULL, NULL, 0))
        goto end;

    if (WaitForSingleObject(timer, INFINITE) != WAIT_OBJECT_0)
        goto end;

    ret = true;

end:
    if (timer)
        CloseHandle(timer);
    return ret;
}
