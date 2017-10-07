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

#ifndef LIBPLACEBO_COMMON_H_
#define LIBPLACEBO_COMMON_H_

#include "context.h"

// Some common utility types
struct pl_rect2d {
    int x0, y0;
    int x1, y1;
};

struct pl_rect3d {
    int x0, y0, z0;
    int x1, y1, z1;
};

#define pl_rect_w(r) ((r).x1 - (r).x0)
#define pl_rect_h(r) ((r).y1 - (r).y0)
#define pl_rect_d(r) ((r).z1 - (r).z0)

#endif // LIBPLACEBO_COMMON_H_
