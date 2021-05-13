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

#include <stdarg.h>
#include <time.h>
#include <pthread.h>

#include "common.h"

// Internal logging-related functions

// Warning: Not entirely thread-safe. Exercise caution when using. May result
// in either false positives or false negatives. Make sure to re-run this
// function while `lock` is held, to ensure no race conditions on the check.
static inline bool pl_msg_test(pl_log log, enum pl_log_level lev)
{
    return log && log->params.log_cb && log->params.log_level >= lev;
}

void pl_msg(pl_log log, enum pl_log_level lev, const char *fmt, ...)
    PL_PRINTF(3, 4);

// Convenience macros
#define pl_fatal(log, ...)      pl_msg(log, PL_LOG_FATAL, __VA_ARGS__)
#define pl_err(log, ...)        pl_msg(log, PL_LOG_ERR, __VA_ARGS__)
#define pl_warn(log, ...)       pl_msg(log, PL_LOG_WARN, __VA_ARGS__)
#define pl_info(log, ...)       pl_msg(log, PL_LOG_INFO, __VA_ARGS__)
#define pl_debug(log, ...)      pl_msg(log, PL_LOG_DEBUG, __VA_ARGS__)
#define pl_trace(log, ...)      pl_msg(log, PL_LOG_TRACE, __VA_ARGS__)

#define PL_MSG(obj, lev, ...)   pl_msg((obj)->log, lev, __VA_ARGS__)

#define PL_FATAL(obj, ...)      PL_MSG(obj, PL_LOG_FATAL, __VA_ARGS__)
#define PL_ERR(obj, ...)        PL_MSG(obj, PL_LOG_ERR, __VA_ARGS__)
#define PL_WARN(obj, ...)       PL_MSG(obj, PL_LOG_WARN, __VA_ARGS__)
#define PL_INFO(obj, ...)       PL_MSG(obj, PL_LOG_INFO, __VA_ARGS__)
#define PL_DEBUG(obj, ...)      PL_MSG(obj, PL_LOG_DEBUG, __VA_ARGS__)
#define PL_TRACE(obj, ...)      PL_MSG(obj, PL_LOG_TRACE, __VA_ARGS__)

// Log something with line numbers included
void pl_msg_source(pl_log log, enum pl_log_level lev, const char *src);

// Temporarily cap the log level to a certain verbosity. This is intended for
// things like probing formats, attempting to create buffers that may fail, and
// other types of operations in which we want to suppress errors. Call with
// PL_LOG_NONE to disable this cap.
//
// Warning: This is generally not thread-safe, and only provided as a temporary
// hack until a better solution can be thought of.
void pl_log_level_cap(pl_log log, enum pl_log_level cap);

// CPU execution time reporting helper
static inline void pl_log_cpu_time(pl_log log, time_t start, time_t stop,
                                   const char *operation)
{
    double ms = (stop - start) * 1e3 / CLOCKS_PER_SEC;
    enum pl_log_level lev = PL_LOG_DEBUG;
    if (ms > 10)
        lev = PL_LOG_INFO;
    if (ms > 1000)
        lev = PL_LOG_WARN;

    pl_msg(log, lev, "Spent %.3f ms %s%s", ms, operation,
           ms > 100 ? " (slow!)" : "");
}
