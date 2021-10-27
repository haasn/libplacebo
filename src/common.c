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

#include <math.h>

#include "common.h"

int pl_fix_ver(void)
{
    return BUILD_FIX_VER;
}

const char *pl_version(void)
{
    return BUILD_VERSION;
}

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

void pl_rect2df_normalize(struct pl_rect2df *rc)
{
    *rc = (struct pl_rect2df) {
        .x0 = PL_MIN(rc->x0, rc->x1),
        .x1 = PL_MAX(rc->x0, rc->x1),
        .y0 = PL_MIN(rc->y0, rc->y1),
        .y1 = PL_MAX(rc->y0, rc->y1),
    };
}

void pl_rect3df_normalize(struct pl_rect3df *rc)
{
    *rc = (struct pl_rect3df) {
        .x0 = PL_MIN(rc->x0, rc->x1),
        .x1 = PL_MAX(rc->x0, rc->x1),
        .y0 = PL_MIN(rc->y0, rc->y1),
        .y1 = PL_MAX(rc->y0, rc->y1),
        .z0 = PL_MIN(rc->z0, rc->z1),
        .z1 = PL_MAX(rc->z0, rc->z1),
    };
}

struct pl_rect2d pl_rect2df_round(const struct pl_rect2df *rc)
{
    return (struct pl_rect2d) {
        .x0 = roundf(rc->x0),
        .x1 = roundf(rc->x1),
        .y0 = roundf(rc->y0),
        .y1 = roundf(rc->y1),
    };
}

struct pl_rect3d pl_rect3df_round(const struct pl_rect3df *rc)
{
    return (struct pl_rect3d) {
        .x0 = roundf(rc->x0),
        .x1 = roundf(rc->x1),
        .y0 = roundf(rc->y0),
        .y1 = roundf(rc->y1),
        .z0 = roundf(rc->z0),
        .z1 = roundf(rc->z1),
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

void pl_matrix3x3_apply_rc(const struct pl_matrix3x3 *mat, struct pl_rect3df *rc)
{
    float x0 = rc->x0, x1 = rc->x1,
          y0 = rc->y0, y1 = rc->y1,
          z0 = rc->z0, z1 = rc->z1;

    rc->x0 = mat->m[0][0] * x0 + mat->m[0][1] * y0 + mat->m[0][2] * z0;
    rc->y0 = mat->m[1][0] * x0 + mat->m[1][1] * y0 + mat->m[1][2] * z0;
    rc->z0 = mat->m[2][0] * x0 + mat->m[2][1] * y0 + mat->m[2][2] * z0;

    rc->x1 = mat->m[0][0] * x1 + mat->m[0][1] * y1 + mat->m[0][2] * z1;
    rc->y1 = mat->m[1][0] * x1 + mat->m[1][1] * y1 + mat->m[1][2] * z1;
    rc->z1 = mat->m[2][0] * x1 + mat->m[2][1] * y1 + mat->m[2][2] * z1;
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

void pl_matrix3x3_rmul(const struct pl_matrix3x3 *a, struct pl_matrix3x3 *b)
{
    struct pl_matrix3x3 m = *a;
    pl_matrix3x3_mul(&m, b);
    *b = m;
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

void pl_transform3x3_apply_rc(const struct pl_transform3x3 *t, struct pl_rect3df *rc)
{
    pl_matrix3x3_apply_rc(&t->mat, rc);

    rc->x0 += t->c[0];
    rc->x1 += t->c[0];
    rc->y0 += t->c[1];
    rc->y1 += t->c[1];
    rc->z0 += t->c[2];
    rc->z1 += t->c[2];
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

void pl_matrix2x2_apply_rc(const struct pl_matrix2x2 *mat, struct pl_rect2df *rc)
{
    float x0 = rc->x0, x1 = rc->x1,
          y0 = rc->y0, y1 = rc->y1;

    rc->x0 = mat->m[0][0] * x0 + mat->m[0][1] * y0;
    rc->y0 = mat->m[1][0] * x0 + mat->m[1][1] * y0;

    rc->x1 = mat->m[0][0] * x1 + mat->m[0][1] * y1;
    rc->y1 = mat->m[1][0] * x1 + mat->m[1][1] * y1;
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

void pl_transform2x2_apply_rc(const struct pl_transform2x2 *t, struct pl_rect2df *rc)
{
    pl_matrix2x2_apply_rc(&t->mat, rc);

    rc->x0 += t->c[0];
    rc->x1 += t->c[0];
    rc->y0 += t->c[1];
    rc->y1 += t->c[1];
}

float pl_rect2df_aspect(const struct pl_rect2df *rc)
{
    float w = fabs(pl_rect_w(*rc)), h = fabs(pl_rect_h(*rc));
    return h ? (w / h) : 0.0;
}

void pl_rect2df_aspect_set(struct pl_rect2df *rc, float aspect, float panscan)
{
    pl_assert(aspect >= 0);
    float orig_aspect = pl_rect2df_aspect(rc);
    if (!aspect || !orig_aspect)
        return;

    float scale_x, scale_y;
    if (aspect > orig_aspect) {
        // New aspect is wider than the original, so we need to either grow in
        // scale_x (panscan=1) or shrink in scale_y (panscan=0)
        scale_x = powf(aspect / orig_aspect, panscan);
        scale_y = powf(aspect / orig_aspect, panscan - 1.0);
    } else if (aspect < orig_aspect) {
        // New aspect is taller, so either grow in scale_y (panscan=1) or
        // shrink in scale_x (panscan=0)
        scale_x = powf(orig_aspect / aspect, panscan - 1.0);
        scale_y = powf(orig_aspect / aspect, panscan);
    } else {
        return; // No change in aspect
    }

    pl_rect2df_stretch(rc, scale_x, scale_y);
}

void pl_rect2df_aspect_fit(struct pl_rect2df *rc, const struct pl_rect2df *src,
                           float panscan)
{
    float orig_w = fabs(pl_rect_w(*rc)),
          orig_h = fabs(pl_rect_h(*rc));
    if (!orig_w || !orig_h)
        return;

    // If either one of these is larger than 1, then we need to shrink to fit,
    // otherwise we can just directly stretch the rect.
    float scale_x = fabs(pl_rect_w(*src)) / orig_w,
          scale_y = fabs(pl_rect_h(*src)) / orig_h;

    if (scale_x > 1.0 || scale_y > 1.0) {
        pl_rect2df_aspect_copy(rc, src, panscan);
    } else {
        pl_rect2df_stretch(rc, scale_x, scale_y);
    }
}

void pl_rect2df_stretch(struct pl_rect2df *rc, float stretch_x, float stretch_y)
{
    float midx = (rc->x0 + rc->x1) / 2.0,
          midy = (rc->y0 + rc->y1) / 2.0;

    rc->x0 = rc->x0 * stretch_x + midx * (1.0 - stretch_x);
    rc->x1 = rc->x1 * stretch_x + midx * (1.0 - stretch_x);
    rc->y0 = rc->y0 * stretch_y + midy * (1.0 - stretch_y);
    rc->y1 = rc->y1 * stretch_y + midy * (1.0 - stretch_y);
}

void pl_rect2df_offset(struct pl_rect2df *rc, float offset_x, float offset_y)
{
    if (rc->x1 < rc->x0)
        offset_x = -offset_x;
    if (rc->y1 < rc->y0)
        offset_y = -offset_y;

    rc->x0 += offset_x;
    rc->x1 += offset_x;
    rc->y0 += offset_y;
    rc->y1 += offset_y;
}

void pl_rect2df_rotate(struct pl_rect2df *rc, pl_rotation rot)
{
    if (!(rot = pl_rotation_normalize(rot)))
        return;

    float x0 = rc->x0, y0 = rc->y0, x1 = rc->x1, y1 = rc->y1;
    if (rot >= PL_ROTATION_180) {
        rot -= PL_ROTATION_180;
        PL_SWAP(x0, x1);
        PL_SWAP(y0, y1);
    }

    switch (rot) {
    case PL_ROTATION_0:
        *rc = (struct pl_rect2df) {
            .x0 = x0,
            .y0 = y0,
            .x1 = x1,
            .y1 = y1,
        };
        return;
    case PL_ROTATION_90:
        *rc = (struct pl_rect2df) {
            .x0 = y1,
            .y0 = x0,
            .x1 = y0,
            .y1 = x1,
        };
        return;
    default: pl_unreachable();
    }
}
