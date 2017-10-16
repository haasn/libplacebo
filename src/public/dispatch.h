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

#ifndef LIBPLACEBO_DISPATCH_H_
#define LIBPLACEBO_DISPATCH_H_

#include "shaders.h"
#include "ra.h"

struct pl_dispatch;

// Creates a new shader dispatch object. This object provides a translation
// layer between generated shaders (pl_shader) and the ra context such that it
// can be used to execute shaders. This dispatch object will also provide
// shader caching (for efficient re-use).
struct pl_dispatch *pl_dispatch_create(struct pl_context *ctx, const struct ra *ra);
void pl_dispatch_destroy(struct pl_dispatch **dp);

// Returns a blank pl_shader object, suitable for recording rendering commands.
// For more information, see the header documentation in `shaders/*.h`.
struct pl_shader *pl_dispatch_begin(struct pl_dispatch *dp);

// Dispatch a generated shader (via the pl_shader mechanism). The results of
// shader execution will be rendered to `target`. Returns whether or not the
// dispatch was successful. This operation will take over ownership of the
// pl_shader passed to it, and return it back to the internal pool.
bool pl_dispatch_finish(struct pl_dispatch *dp, struct pl_shader *sh,
                        const struct ra_tex *target);

#endif // LIBPLACEBO_DISPATCH_H
