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

bool pl_rect2d_eq(struct pl_rect2d a, struct pl_rect2d b);
bool pl_rect3d_eq(struct pl_rect3d a, struct pl_rect3d b);

// "Normalize" a rectangle: This returns an equivalent rectangle with the
// property that d1 >= d0 for all dimensions.
struct pl_rect2d pl_rect2d_normalize(struct pl_rect2d rc);
struct pl_rect3d pl_rect3d_normalize(struct pl_rect3d rc);

// Represents a row-major matrix, i.e. the following matrix
//     [ a11 a12 a13 ]
//     [ a21 a22 a23 ]
//     [ a31 a32 a33 ]
// is represented in C like this:
//   { { a11, a12, a13 },
//     { a21, a22, a23 },
//     { a31, a32, a33 } };
struct pl_matrix3x3 {
    float m[3][3];
};

extern const struct pl_matrix3x3 pl_matrix3x3_identity;

// Applies a matrix to a float vector in-place.
void pl_matrix3x3_apply(struct pl_matrix3x3 mat, float vec[3]);

// Scales a color matrix by a linear factor.
struct pl_matrix3x3 pl_matrix3x3_scale(struct pl_matrix3x3 mat, float scale);

// Inverts a matrix. Only use where precision is not that important.
struct pl_matrix3x3 pl_matrix3x3_invert(struct pl_matrix3x3 mat);

// Composes/multiplies two matrices. Returns A * B
struct pl_matrix3x3 pl_matrix3x3_mul(struct pl_matrix3x3 a, struct pl_matrix3x3 b);

// Represents an affine transformation, which is basically a 3x3 matrix
// together with a column vector to add onto the output.
struct pl_transform3x3 {
    struct pl_matrix3x3 mat;
    float c[3];
};

extern const struct pl_transform3x3 pl_transform3x3_identity;

// Applies a transform to a float vector in-place.
void pl_transform3x3_apply(struct pl_transform3x3 t, float vec[3]);

// Scales the output of a transform by a linear factor. Since an affine
// transformation is non-linear, this does not commute. If you want to scale
// the *input* of a transform, use pl_matrix3x3_scale on `t.mat`.
struct pl_transform3x3 pl_transform3x3_scale(struct pl_transform3x3 t, float scale);

// Inverts a transform. Only use where precision is not that important.
struct pl_transform3x3 pl_transform3x3_invert(struct pl_transform3x3 t);

#endif // LIBPLACEBO_COMMON_H_
