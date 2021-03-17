#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Needed to include the generated in-tree config.h
#include "../src/config.h"

#include <libplacebo/context.h>
#include <libplacebo/renderer.h>

static inline struct pl_context *demo_context() {
    return pl_context_create(PL_API_VER, &(struct pl_context_params) {
        .log_cb = isatty(fileno(stdout)) ? pl_log_color : pl_log_simple,
#ifdef NDEBUG
        .log_level = PL_LOG_INFO,
#else
        .log_level = PL_LOG_DEBUG,
#endif
    });
}
