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

#pragma once

#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

static inline struct pl_context *pl_test_context()
{
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    return pl_context_create(PL_API_VER, &(struct pl_context_params) {
        .log_cb    = isatty(fileno(stdout)) ? pl_log_color : pl_log_simple,
        .log_level = PL_LOG_ALL,
    });
}

static inline void require(bool b, const char *msg)
{
    if (!b) {
        fprintf(stderr, "%s", msg);
        exit(1);
    }
}

static inline bool feq(float a, float b)
{
    return fabs(a - b) < 1e-6 * fmax(1.0, fabs(a));
}

#define REQUIRE(cond) require((cond), #cond)
#define RANDOM (random() / (float) RAND_MAX)
#define SKIP 77
