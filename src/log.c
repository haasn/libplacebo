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

#include "common.h"
#include "log.h"
#include "pl_thread.h"

struct priv {
    pl_mutex lock;
    enum pl_log_level log_level_cap;
    pl_str logbuffer;
};

pl_log pl_log_create(int api_ver, const struct pl_log_params *params)
{
    (void) api_ver;
    struct pl_log *log = pl_zalloc_obj(NULL, log, struct priv);
    struct priv *p = PL_PRIV(log);
    log->params = *PL_DEF(params, &pl_log_default_params);
    pl_mutex_init(&p->lock);
    pl_info(log, "Initialized libplacebo %s (API v%d)", PL_VERSION, PL_API_VER);
    return log;
}

const struct pl_log_params pl_log_default_params = {0};

void pl_log_destroy(pl_log *plog)
{
    pl_log log = *plog;
    if (!log)
        return;

    struct priv *p = PL_PRIV(log);
    pl_mutex_destroy(&p->lock);
    pl_free((void *) log);
    *plog = NULL;
}

struct pl_log_params pl_log_update(pl_log ptr, const struct pl_log_params *params)
{
    struct pl_log *log = (struct pl_log *) ptr;
    if (!log)
        return pl_log_default_params;

    struct priv *p = PL_PRIV(log);
    pl_mutex_lock(&p->lock);
    struct pl_log_params prev_params = log->params;
    log->params = *PL_DEF(params, &pl_log_default_params);
    pl_mutex_unlock(&p->lock);

    return prev_params;
}

enum pl_log_level pl_log_level_update(pl_log ptr, enum pl_log_level level)
{
    struct pl_log *log = (struct pl_log *) ptr;
    if (!log)
        return PL_LOG_NONE;

    struct priv *p = PL_PRIV(log);
    pl_mutex_lock(&p->lock);
    enum pl_log_level prev_level = log->params.log_level;
    log->params.log_level = level;
    pl_mutex_unlock(&p->lock);

    return prev_level;
}

void pl_log_level_cap(pl_log log, enum pl_log_level cap)
{
    if (!log)
        return;

    struct priv *p = PL_PRIV(log);
    pl_mutex_lock(&p->lock);
    p->log_level_cap = cap;
    pl_mutex_unlock(&p->lock);
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

static void pl_msg_va(pl_log log, enum pl_log_level lev,
                      const char *fmt, va_list va)
{
    // Test log message without taking the lock, to avoid thrashing the
    // lock for thousands of trace messages unless those are actually
    // enabled. This may be a false negative, in which case log messages may
    // be lost as a result. But this shouldn't be a big deal, since any
    // situation leading to lost log messages would itself be a race condition.
    if (!pl_msg_test(log, lev))
        return;

    // Re-test the log message level with held lock to avoid false positives,
    // which would be a considerably bigger deal than false negatives
    struct priv *p = PL_PRIV(log);
    pl_mutex_lock(&p->lock);

    // Apply this cap before re-testing the log level, to avoid giving users
    // messages that should have been dropped by the log level.
    lev = PL_MAX(lev, p->log_level_cap);
    if (!pl_msg_test(log, lev))
        goto done;

    p->logbuffer.len = 0;
    pl_str_append_vasprintf((void *) log, &p->logbuffer, fmt, va);
    log->params.log_cb(log->params.log_priv, lev, (char *) p->logbuffer.buf);

done:
    pl_mutex_unlock(&p->lock);
}

void pl_msg(pl_log log, enum pl_log_level lev, const char *fmt, ...)
{
    va_list va;
    va_start(va, fmt);
    pl_msg_va(log, lev, fmt, va);
    va_end(va);
}

void pl_msg_source(pl_log log, enum pl_log_level lev, const char *src)
{
    if (!pl_msg_test(log, lev) || !src)
        return;

    int line = 1;
    while (*src) {
        const char *end = strchr(src, '\n');
        if (!end) {
            pl_msg(log, lev, "[%3d] %s", line, src);
            break;
        }

        pl_msg(log, lev, "[%3d] %.*s", line, (int)(end - src), src);
        src = end + 1;
        line++;
    }
}

#ifdef PL_HAVE_UNWIND
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#include <dlfcn.h>

void pl_log_stack_trace(pl_log log, enum pl_log_level lev)
{
    if (!pl_msg_test(log, lev))
        return;

    unw_cursor_t cursor;
    unw_context_t uc;
    unw_word_t ip, off;
    unw_getcontext(&uc);
    unw_init_local(&cursor, &uc);

    int depth = 0;
    pl_msg(log, lev, "  Backtrace:");
    while (unw_step(&cursor) > 0) {
        char symbol[256] = "<unknown>";
        Dl_info info = {
            .dli_fname = "<unknown>",
        };

        unw_get_reg(&cursor, UNW_REG_IP, &ip);
        unw_get_proc_name(&cursor, symbol, sizeof(symbol), &off);
        dladdr((void *) (uintptr_t) ip, &info);
        pl_msg(log, lev, "    #%-2d 0x%016" PRIxPTR " in %s+0x%" PRIxPTR" at %s+0x%" PRIxPTR,
               depth++, ip, symbol, off, info.dli_fname, ip - (uintptr_t) info.dli_fbase);
    }
}

#elif defined(PL_HAVE_EXECINFO) && !defined(MSAN)
#include <execinfo.h>

void pl_log_stack_trace(pl_log log, enum pl_log_level lev)
{
    if (!pl_msg_test(log, lev))
        return;

    PL_ARRAY(void *) buf = {0};
    size_t buf_avail = 16;
    do {
        buf_avail *= 2;
        PL_ARRAY_RESIZE(NULL, buf, buf_avail);
        buf.num = backtrace(buf.elem, buf_avail);
    } while (buf.num == buf_avail);

    pl_msg(log, lev, "  Backtrace:");
    char **strings = backtrace_symbols(buf.elem, buf.num);
    for (int i = 0; i < buf.num; i++)
        pl_msg(log, lev, "    #%-2d %s", i, strings[i]);

    free(strings);
    pl_free(buf.elem);
}

#else
void pl_log_stack_trace(pl_log log, enum pl_log_level lev) { }
#endif
