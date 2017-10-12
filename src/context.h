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
#include "bstr/bstr.h"

#include "common.h"

struct pl_context {
    struct pl_context_params params;
    struct bstr logbuffer;
};

// Logging-related functions
static inline bool pl_msg_test(struct pl_context *ctx, enum pl_log_level lev)
{
    return ctx->params.log_cb && ctx->params.log_level >= lev;
}

void pl_msg(struct pl_context *ctx, enum pl_log_level lev, const char *fmt, ...)
    PRINTF_ATTRIBUTE(3, 4);
void pl_msg_va(struct pl_context *ctx, enum pl_log_level lev, const char *fmt,
               va_list va);

// Convenience macros
#define pl_fatal(log, ...)      pl_msg(ctx, PL_LOG_FATAL, __VA_ARGS__)
#define pl_err(log, ...)        pl_msg(ctx, PL_LOG_ERR, __VA_ARGS__)
#define pl_warn(log, ...)       pl_msg(ctx, PL_LOG_WARN, __VA_ARGS__)
#define pl_info(log, ...)       pl_msg(ctx, PL_LOG_INFO, __VA_ARGS__)
#define pl_debug(log, ...)      pl_msg(ctx, PL_LOG_DEBUG, __VA_ARGS__)
#define pl_trace(log, ...)      pl_msg(ctx, PL_LOG_TRACE, __VA_ARGS__)

#define PL_MSG(obj, lev, ...)   pl_msg((obj)->ctx, lev, __VA_ARGS__)

#define PL_FATAL(obj, ...)      PL_MSG(obj, PL_LOG_FATAL, __VA_ARGS__)
#define PL_ERR(obj, ...)        PL_MSG(obj, PL_LOG_ERR, __VA_ARGS__)
#define PL_WARN(obj, ...)       PL_MSG(obj, PL_LOG_WARN, __VA_ARGS__)
#define PL_INFO(obj, ...)       PL_MSG(obj, PL_LOG_INFO, __VA_ARGS__)
#define PL_DEBUG(obj, ...)      PL_MSG(obj, PL_LOG_DEBUG, __VA_ARGS__)
#define PL_TRACE(obj, ...)      PL_MSG(obj, PL_LOG_TRACE, __VA_ARGS__)

// Log something with line numbers included
void pl_msg_source(struct pl_context *ctx, enum pl_log_level lev, const char *src);
