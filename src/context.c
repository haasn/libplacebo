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
#include <locale.h>
#include <pthread.h>

#include "common.h"
#include "context.h"

int pl_fix_ver()
{
    return BUILD_FIX_VER;
}

const char *pl_version()
{
    return BUILD_VERSION;
}

static pthread_mutex_t pl_ctx_mutex = PTHREAD_MUTEX_INITIALIZER;
static int pl_ctx_refcount;

static void global_init(void)
{
#ifndef NDEBUG
    const char *enable_leak = getenv("LIBPLACEBO_LEAK_REPORT");
    if (enable_leak && strcmp(enable_leak, "1") == 0)
        talloc_enable_leak_report();
#endif
}

static void global_uninit(void)
{
#ifndef NDEBUG
    talloc_print_leak_report();
#endif
}

struct pl_context *pl_context_create(int api_ver,
                                     const struct pl_context_params *params)
{
    if (api_ver != PL_API_VER) {
        fprintf(stderr,
               "*************************************************************\n"
               "libplacebo: ABI mismatch detected! (requested: %d, compiled: %d)\n"
               "\n"
               "This is usually indicative of a linking mismatch, and will\n"
               "result in serious issues including stack corruption, random\n"
               "crashes and arbitrary code execution. Aborting as a safety\n"
               "precaution. Fix your system!\n",
               api_ver, PL_API_VER);
        abort();
    }

    // Do global initialization only when refcount is 0
    pthread_mutex_lock(&pl_ctx_mutex);
    if (pl_ctx_refcount++ == 0)
        global_init();

    struct pl_context *ctx = talloc_zero(NULL, struct pl_context);
    ctx->params = *PL_DEF(params, &pl_context_default_params);
    int err = pthread_mutex_init(&ctx->lock, NULL);
    if (err != 0) {
        fprintf(stderr, "Failed initializing pthread mutex: %s\n", strerror(err));
        pl_ctx_refcount--;
        talloc_free(ctx);
        ctx = NULL;
    }

    pthread_mutex_unlock(&pl_ctx_mutex);
    pl_info(ctx, "Initialized libplacebo %s (API v%d)", PL_VERSION, PL_API_VER);

    return ctx;
}

const struct pl_context_params pl_context_default_params = {0};

void pl_context_destroy(struct pl_context **pctx)
{
    struct pl_context *ctx = *pctx;
    if (!ctx)
        return;

    pthread_mutex_lock(&ctx->lock);
    pthread_mutex_destroy(&ctx->lock);
    talloc_free(ctx);
    *pctx = NULL;

    // Do global uninitialization only when refcount reaches 0
    pthread_mutex_lock(&pl_ctx_mutex);
    if (--pl_ctx_refcount == 0)
        global_uninit();
    pthread_mutex_unlock(&pl_ctx_mutex);
}

void pl_context_update(struct pl_context *ctx,
                       const struct pl_context_params *params)
{
    pthread_mutex_lock(&ctx->lock);
    ctx->params = *PL_DEF(params, &pl_context_default_params);
    pthread_mutex_unlock(&ctx->lock);
}

void pl_log_level_cap(struct pl_context *ctx, enum pl_log_level cap)
{
    pthread_mutex_lock(&ctx->lock);
    ctx->log_level_cap = cap;
    pthread_mutex_unlock(&ctx->lock);
}

static FILE *default_stream(void *stream, enum pl_log_level level)
{
    return PL_DEF(stream, level <= PL_LOG_WARN ? stderr : stdout);
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
    if (level <= PL_LOG_WARN)
        fflush(h);
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
    if (level <= PL_LOG_WARN)
        fflush(h);
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
    // Test log message without taking the lock, to avoid thrashing the
    // lock for thousands of trace messages unless those are actually
    // enabled. This may be a false negative, in which case log messages may
    // be lost as a result. But this shouldn't be a big deal, since any
    // situation leading to lost log messages would itself be a race condition.
    if (!pl_msg_test(ctx, lev))
        return;

    // Re-test the log message level with held lock to avoid false positives,
    // which would be a considerably bigger deal than false negatives
    pthread_mutex_lock(&ctx->lock);

    // Apply this cap before re-testing the log level, to avoid giving users
    // messages that should have been dropped by the log level.
    lev = PL_MAX(lev, ctx->log_level_cap);

    if (!pl_msg_test(ctx, lev))
        goto done;

    ctx->logbuffer.len = 0;
    pl_str_xappend_vasprintf(ctx, &ctx->logbuffer, fmt, va);
    ctx->params.log_cb(ctx->params.log_priv, lev, ctx->logbuffer.buf);

done:
    pthread_mutex_unlock(&ctx->lock);
}

void pl_msg_source(struct pl_context *ctx, enum pl_log_level lev, const char *src)
{
    if (!pl_msg_test(ctx, lev) || !src)
        return;

    int line = 1;
    while (*src) {
        const char *end = strchr(src, '\n');
        if (!end) {
            pl_msg(ctx, lev, "[%3d] %s", line, src);
            break;
        }

        pl_msg(ctx, lev, "[%3d] %.*s", line, (int)(end - src), src);
        src = end + 1;
        line++;
    }
}
