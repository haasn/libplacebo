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

#include "common.h"

// One texture per level of an image pyramid, coarsest level last.
typedef PL_ARRAY(pl_tex) pl_tex_pyramid;

// Destroys every level of `pyramid`, along with the array holding them.
void pl_pyramid_destroy(pl_gpu gpu, pl_tex_pyramid *pyramid);

// Number of levels before degenerating into a single pixel.
int pl_pyramid_max_levels(int w, int h);

// Resize `pyramid` to hold exactly `levels` textures, destroying the levels
// beyond that and leaving new levels to be allocated by `pl_tex_recreate`.
void pl_pyramid_resize(pl_gpu gpu, void *alloc, pl_tex_pyramid *pyramid,
                       int levels);

// Allocate or resize `pyramid` to hold exactly `levels` textures created with
// `params`, the dimensions of level `i` being those of `params` halved `i`
// times (rounding up). Levels that already fit are kept, contents included.
bool pl_pyramid_recreate(pl_gpu gpu, void *alloc, pl_tex_pyramid *pyramid,
                         int levels, const struct pl_tex_params *params);

// Fills every level of `pyramid` but the first by reducing the level before it.
bool pl_pyramid_build(pl_gpu gpu, pl_dispatch dispatch,
                      const pl_tex_pyramid *pyramid);
