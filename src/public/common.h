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
#include <stdbool.h>

// Some common utility types. These are overloaded to support 2D, 3D and
// integer/float variants.
struct pl_rect2d {
    int x0, y0;
    int x1, y1;
};

struct pl_rect3d {
    int x0, y0, z0;
    int x1, y1, z1;
};

struct pl_rect2df {
    float x0, y0;
    float x1, y1;
};

struct pl_rect3df {
    float x0, y0;
    float x1, y1;
    float z0, z1;
};

// These macros will work for any of the above pl_rect variants (with enough
// dimensions). Careful: double-evaluation hazard
#define pl_rect_w(r) ((r).x1 - (r).x0)
#define pl_rect_h(r) ((r).y1 - (r).y0)
#define pl_rect_d(r) ((r).z1 - (r).z0)

#define pl_rect2d_eq(a, b) \
    ((a).x0 == (b).x0 && (a).x1 == (b).x1 && \
     (a).y0 == (b).y0 && (a).y1 == (b).y1)

#define pl_rect3d_eq(a, b) \
    ((a).x0 == (b).x0 && (a).x1 == (b).x1 && \
     (a).y0 == (b).y0 && (a).y1 == (b).y1 && \
     (a).z0 == (b).z0 && (a).z1 == (b).z1)

// "Normalize" a rectangle: This ensures d1 >= d0 for all dimensions.
void pl_rect2d_normalize(struct pl_rect2d *rc);
void pl_rect3d_normalize(struct pl_rect3d *rc);

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
void pl_matrix3x3_apply(const struct pl_matrix3x3 *mat, float vec[3]);

// Scales a color matrix by a linear factor.
void pl_matrix3x3_scale(struct pl_matrix3x3 *mat, float scale);

// Inverts a matrix. Only use where precision is not that important.
void pl_matrix3x3_invert(struct pl_matrix3x3 *mat);

// Composes/multiplies two matrices. Multiples B into A, i.e.
// A := A * B
void pl_matrix3x3_mul(struct pl_matrix3x3 *a, const struct pl_matrix3x3 *b);

// Represents an affine transformation, which is basically a 3x3 matrix
// together with a column vector to add onto the output.
struct pl_transform3x3 {
    struct pl_matrix3x3 mat;
    float c[3];
};

extern const struct pl_transform3x3 pl_transform3x3_identity;

// Applies a transform to a float vector in-place.
void pl_transform3x3_apply(const struct pl_transform3x3 *t, float vec[3]);

// Scales the output of a transform by a linear factor. Since an affine
// transformation is non-linear, this does not commute. If you want to scale
// the *input* of a transform, use pl_matrix3x3_scale on `t.mat`.
void pl_transform3x3_scale(struct pl_transform3x3 *t, float scale);

// Inverts a transform. Only use where precision is not that important.
void pl_transform3x3_invert(struct pl_transform3x3 *t);

// 2D analog of the above structs. Since these are featured less prominently,
// we omit some of the other helper functions.
struct pl_matrix2x2 {
    float m[2][2];
};

extern const struct pl_matrix2x2 pl_matrix2x2_identity;

void pl_matrix2x2_apply(const struct pl_matrix2x2 *mat, float vec[2]);

struct pl_transform2x2 {
    struct pl_matrix2x2 mat;
    float c[2];
};

extern const struct pl_transform2x2 pl_transform2x2_identity;

void pl_transform2x2_apply(const struct pl_transform2x2 *t, float vec[2]);

#endif // LIBPLACEBO_COMMON_H_
