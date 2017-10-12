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

#include <stdio.h>

#include "common.h"
#include "context.h"

struct pl_context *pl_context_create(int api_ver,
                                     const struct pl_context_params *params)
{
    if (api_ver != PL_API_VER) {
        fprintf(stderr,
               "*************************************************************\n"
               "libplacebo: ABI mismatch detected!\n\n"
               "This is usually indicative of a linking mismatch, and will\n"
               "result in serious issues including stack corruption, random\n"
               "crashes and arbitrary code exection. Aborting as a safety\n"
               "precaution!\n");
        abort();
    }

    struct pl_context *ctx = talloc_zero(NULL, struct pl_context);
    ctx->params = *params;
    return ctx;
}

void pl_context_destroy(struct pl_context **ctx)
{
    TA_FREEP(ctx);
}

static FILE *default_stream(void *stream, enum pl_log_level level)
{
    return stream ? stream : level < PL_LOG_WARN ? stderr : stdout;
}

void pl_log_simple(void *stream, enum pl_log_level level, const char *msg)
{
    static const char *prefix[] = {
        [PL_LOG_FATAL] = "fatal",
        [PL_LOG_ERR]   = "error",
        [PL_LOG_WARN]  = "warn",
        [PL_LOG_INFO]  = "info",
        [PL_LOG_DEBUG] = "debug",
        [PL_LOG_TRACE] = "trace",
    };

    FILE *h = default_stream(stream, level);
    fprintf(h, "%5s: %s\n", prefix[level], msg);
}

void pl_log_color(void *stream, enum pl_log_level level, const char *msg)
{
    static const char *color[] = {
        [PL_LOG_FATAL] = "31;1", // bright red
        [PL_LOG_ERR]   = "31",   // red
        [PL_LOG_WARN]  = "33",   // yellow/orange
        [PL_LOG_INFO]  = "32",   // green
        [PL_LOG_DEBUG] = "34",   // blue
        [PL_LOG_TRACE] = "30;1", // bright black
    };

    FILE *h = default_stream(stream, level);
    fprintf(h, "\033[%sm%s\033[0m\n", color[level], msg);
}

void pl_msg(struct pl_context *ctx, enum pl_log_level lev, const char *fmt, ...)
{
    va_list va;
    va_start(va, fmt);
    pl_msg_va(ctx, lev, fmt, va);
    va_end(va);
}

void pl_msg_va(struct pl_context *ctx, enum pl_log_level lev, const char *fmt,
               va_list va)
{
    if (!pl_msg_test(ctx, lev))
        return;

    ctx->logbuffer.len = 0;
    bstr_xappend_vasprintf(ctx, &ctx->logbuffer, fmt, va);
    ctx->params.log_cb(ctx->params.log_priv, lev, ctx->logbuffer.start);
}

void pl_msg_source(struct pl_context *ctx, enum pl_log_level lev, const char *src)
{
    if (!pl_msg_test(ctx, lev) || !src)
        return;

    int line = 1;
    while (*src) {
        const char *end = strchr(src, '\n');
        const char *next = end + 1;
        if (!end)
            next = end = src + strlen(src);
        pl_msg(ctx, lev, "[%3d] %.*s", line, (int)(end - src), src);
        line++;
        src = next;
    }
}
