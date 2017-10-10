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

#include "common.h"

struct pl_rect2d pl_rect2d_normalize(struct pl_rect2d rc)
{
    return (struct pl_rect2d) {
        .x0 = PL_MIN(rc.x0, rc.x1),
        .x1 = PL_MAX(rc.x0, rc.x1),
        .y0 = PL_MIN(rc.y0, rc.y1),
        .y1 = PL_MAX(rc.y0, rc.y1),
    };
}

struct pl_rect3d pl_rect3d_normalize(struct pl_rect3d rc)
{
    return (struct pl_rect3d) {
        .x0 = PL_MIN(rc.x0, rc.x1),
        .x1 = PL_MAX(rc.x0, rc.x1),
        .y0 = PL_MIN(rc.y0, rc.y1),
        .y1 = PL_MAX(rc.y0, rc.y1),
        .z0 = PL_MIN(rc.z0, rc.z1),
        .z1 = PL_MAX(rc.z0, rc.z1),
    };
}
