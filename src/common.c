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

void pl_rect2d_normalize(struct pl_rect2d *rc)
{
    *rc = (struct pl_rect2d) {
        .x0 = PL_MIN(rc->x0, rc->x1),
        .x1 = PL_MAX(rc->x0, rc->x1),
        .y0 = PL_MIN(rc->y0, rc->y1),
        .y1 = PL_MAX(rc->y0, rc->y1),
    };
}

void pl_rect3d_normalize(struct pl_rect3d *rc)
{
    *rc = (struct pl_rect3d) {
        .x0 = PL_MIN(rc->x0, rc->x1),
        .x1 = PL_MAX(rc->x0, rc->x1),
        .y0 = PL_MIN(rc->y0, rc->y1),
        .y1 = PL_MAX(rc->y0, rc->y1),
        .z0 = PL_MIN(rc->z0, rc->z1),
        .z1 = PL_MAX(rc->z0, rc->z1),
    };
}

const struct pl_matrix3x3 pl_matrix3x3_identity = {{
    { 1, 0, 0 },
    { 0, 1, 0 },
    { 0, 0, 1 },
}};

void pl_matrix3x3_apply(const struct pl_matrix3x3 *mat, float vec[3])
{
    float x = vec[0], y = vec[1], z = vec[2];

    for (int i = 0; i < 3; i++)
        vec[i] = mat->m[i][0] * x + mat->m[i][1] * y + mat->m[i][2] * z;
}

void pl_matrix3x3_scale(struct pl_matrix3x3 *mat, float scale)
{
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            mat->m[i][j] *= scale;
    }
}

void pl_matrix3x3_invert(struct pl_matrix3x3 *mat)
{
    float m00 = mat->m[0][0], m01 = mat->m[0][1], m02 = mat->m[0][2],
          m10 = mat->m[1][0], m11 = mat->m[1][1], m12 = mat->m[1][2],
          m20 = mat->m[2][0], m21 = mat->m[2][1], m22 = mat->m[2][2];

    // calculate the adjoint
    mat->m[0][0] =  (m11 * m22 - m21 * m12);
    mat->m[0][1] = -(m01 * m22 - m21 * m02);
    mat->m[0][2] =  (m01 * m12 - m11 * m02);
    mat->m[1][0] = -(m10 * m22 - m20 * m12);
    mat->m[1][1] =  (m00 * m22 - m20 * m02);
    mat->m[1][2] = -(m00 * m12 - m10 * m02);
    mat->m[2][0] =  (m10 * m21 - m20 * m11);
    mat->m[2][1] = -(m00 * m21 - m20 * m01);
    mat->m[2][2] =  (m00 * m11 - m10 * m01);

    // calculate the determinant (as inverse == 1/det * adjoint,
    // adjoint * m == identity * det, so this calculates the det)
    float det = m00 * mat->m[0][0] + m10 * mat->m[0][1] + m20 * mat->m[0][2];
    det = 1.0f / det;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            mat->m[i][j] *= det;
    }
}

void pl_matrix3x3_mul(struct pl_matrix3x3 *a, const struct pl_matrix3x3 *b)
{
    float a00 = a->m[0][0], a01 = a->m[0][1], a02 = a->m[0][2],
          a10 = a->m[1][0], a11 = a->m[1][1], a12 = a->m[1][2],
          a20 = a->m[2][0], a21 = a->m[2][1], a22 = a->m[2][2];

    for (int i = 0; i < 3; i++) {
        a->m[0][i] = a00 * b->m[0][i] + a01 * b->m[1][i] + a02 * b->m[2][i];
        a->m[1][i] = a10 * b->m[0][i] + a11 * b->m[1][i] + a12 * b->m[2][i];
        a->m[2][i] = a20 * b->m[0][i] + a21 * b->m[1][i] + a22 * b->m[2][i];
    }
}

const struct pl_transform3x3 pl_transform3x3_identity = {
    .mat = {{
        { 1, 0, 0 },
        { 0, 1, 0 },
        { 0, 0, 1 },
    }},
};

void pl_transform3x3_apply(const struct pl_transform3x3 *t, float vec[3])
{
    pl_matrix3x3_apply(&t->mat, vec);

    for (int i = 0; i < 3; i++)
        vec[i] += t->c[i];
}

void pl_transform3x3_scale(struct pl_transform3x3 *t, float scale)
{
    pl_matrix3x3_scale(&t->mat, scale);

    for (int i = 0; i < 3; i++)
        t->c[i] *= scale;
}

// based on DarkPlaces engine (relicensed from GPL to LGPL)
void pl_transform3x3_invert(struct pl_transform3x3 *t)
{
    pl_matrix3x3_invert(&t->mat);

    float m00 = t->mat.m[0][0], m01 = t->mat.m[0][1], m02 = t->mat.m[0][2],
          m10 = t->mat.m[1][0], m11 = t->mat.m[1][1], m12 = t->mat.m[1][2],
          m20 = t->mat.m[2][0], m21 = t->mat.m[2][1], m22 = t->mat.m[2][2];

    // fix the constant coefficient
    // rgb = M * yuv + C
    // M^-1 * rgb = yuv + M^-1 * C
    // yuv = M^-1 * rgb - M^-1 * C
    //                  ^^^^^^^^^^
    float c0 = t->c[0], c1 = t->c[1], c2 = t->c[2];
    t->c[0] = -(m00 * c0 + m01 * c1 + m02 * c2);
    t->c[1] = -(m10 * c0 + m11 * c1 + m12 * c2);
    t->c[2] = -(m20 * c0 + m21 * c1 + m22 * c2);
}

const struct pl_matrix2x2 pl_matrix2x2_identity = {{
    { 1, 0 },
    { 0, 1 },
}};

void pl_matrix2x2_apply(const struct pl_matrix2x2 *mat, float vec[2])
{
    float x = vec[0], y = vec[1];

    for (int i = 0; i < 2; i++)
        vec[i] = mat->m[i][0] * x + mat->m[i][1] * y;
}

const struct pl_transform2x2 pl_transform2x2_identity = {
    .mat = {{
        { 1, 0 },
        { 0, 1 },
    }},
};

void pl_transform2x2_apply(const struct pl_transform2x2 *t, float vec[2])
{
    pl_matrix2x2_apply(&t->mat, vec);

    for (int i = 0; i < 2; i++)
        vec[i] += t->c[i];
}
