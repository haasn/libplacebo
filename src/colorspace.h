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

#include <libplacebo/colorspace.h>

// Returns true if two primaries are 'compatible', which is the case if
// they preserve the relationship between primaries (red=red, green=green,
// blue=blue). In other words, this is false for synthetic primaries that have
// channels misordered from the convention (e.g. for some test ICC profiles).
PL_API bool pl_primaries_compatible(const struct pl_raw_primaries *a,
                                    const struct pl_raw_primaries *b);

// Clip points in the first gamut (src) to be fully contained inside the second
// gamut (dst). Only works on compatible primaries (pl_primaries_compatible).
PL_API struct pl_raw_primaries
pl_primaries_clip(const struct pl_raw_primaries *src,
                  const struct pl_raw_primaries *dst);
