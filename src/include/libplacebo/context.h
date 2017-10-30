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
 * License along with libplacebo.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef LIBPLACEBO_CONTEXT_H_
#define LIBPLACEBO_CONTEXT_H_

#include "config.h"

// Meta-object to serve as a global entrypoint for the purposes of resource
// allocation, logging, etc.. Note on thread safety: the pl_context and
// everything allocated from it are *not* thread-safe except where otherwise
// noted. That is, multiple pl_context objects are safe to use from multiple
// threads, but a single pl_context and all of its derived resources and
// contexts must be used from a single thread at all times.
struct pl_context;

// The log level associated with a given log message.
enum pl_log_level {
    PL_LOG_NONE = 0,
    PL_LOG_FATAL,   // results in total loss of function of a major component
    PL_LOG_ERR,     // serious error; may result in degraded function
    PL_LOG_WARN,    // warning; potentially bad, probably user-relevant
    PL_LOG_INFO,    // informational message, also potentially harmless errors
    PL_LOG_DEBUG,   // verbose debug message, informational
    PL_LOG_TRACE,   // very noisy trace of activity,, usually benign
    PL_LOG_ALL = PL_LOG_TRACE,
};

// Global options for a pl_context.
struct pl_context_params {
    // Logging callback. All messages, informational or otherwise, will get
    // redirected to this callback. The logged messages do not include trailing
    // newlines. Optional.
    void (*log_cb)(void *log_priv, enum pl_log_level level, const char *msg);
    void *log_priv;

    // The current log level. Controls the level of message that will be
    // redirected ot the log callback. Setting this to PL_LOG_ALL means all
    // messages will be forwarded, but doing so indiscriminately can result
    // in increased CPU usage as it may enable extra debug paths based on the
    // configured log level.
    enum pl_log_level log_level;
};

// Creates a new, blank pl_context. The argument `api_ver` must be given as
// PL_API_VER (this is used to detect ABI mismatch due to broken linking).
// `params` defaults to pl_context_default_params if left as NULL.
// Returns NULL on failure.
struct pl_context *pl_context_create(int api_ver,
                                     const struct pl_context_params *params);

// Equal to (struct pl_context_params) {0}
extern const struct pl_context_params pl_context_default_params;

// Except where otherwise noted, all objects allocated from this pl_context
// must be destroyed by the user before the pl_context is destroyed.
//
// Note: As a rule of thumb, all _destroy functions take the pointer to the
// object to free as their parameter. This pointer is overwritten by NULL
// afterwards. Calling a _destroy function on &{NULL} is valid, but calling it
// on NULL itself is invalid.
void pl_context_destroy(struct pl_context **ctx);

// Two simple, stream-based loggers. You can use these as the log_cb. If you
// also set log_priv to a FILE* (e.g. stdout or stderr) it will be printed
// there; otherwise, it will be printed to stdout or stderr depending on the
// log level.
//
// The version with colors will use ANSI escape sequences to indicate the log
// level. The version without will use explicit prefixes.
void pl_log_simple(void *stream, enum pl_log_level level, const char *msg);
void pl_log_color(void *stream, enum pl_log_level level, const char *msg);

#endif // LIBPLACEBO_CONTEXT_H_
