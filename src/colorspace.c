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

bool pl_color_system_is_ycbcr_like(enum pl_color_system sys)
{
    switch (sys) {
    case PL_COLOR_SYSTEM_RGB:
    case PL_COLOR_SYSTEM_XYZ:
        return false;
    case PL_COLOR_SYSTEM_UNKNOWN:
    case PL_COLOR_SYSTEM_BT_601:
    case PL_COLOR_SYSTEM_BT_709:
    case PL_COLOR_SYSTEM_SMPTE_240M:
    case PL_COLOR_SYSTEM_BT_2020_NC:
    case PL_COLOR_SYSTEM_BT_2020_C:
    case PL_COLOR_SYSTEM_YCGCO:
        return true;
    default: abort();
    };
}

enum pl_color_system pl_color_system_guess_ycbcr(int width, int height)
{
    if (width >= 1280 || height > 576) {
        // Typical HD content
        return PL_COLOR_SYSTEM_BT_709;
    } else {
        // Typical SD content
        return PL_COLOR_SYSTEM_BT_601;
    }
}

const struct pl_color_repr pl_color_repr_unknown = {0};

void pl_color_repr_merge(struct pl_color_repr *orig,
                         const struct pl_color_repr *new)
{
    if (!orig->sys)
        orig->sys = new->sys;
    if (!orig->levels)
        orig->levels = new->levels;
    if (!orig->bit_depth)
        orig->bit_depth = new->bit_depth;
}

bool pl_color_repr_equal(struct pl_color_repr c1, struct pl_color_repr c2)
{
    return c1.sys == c2.sys &&
           c1.levels == c2.levels &&
           c1.bit_depth == c2.bit_depth;
}

float pl_color_repr_texture_mul(struct pl_color_repr repr, int new_bits)
{
    int old_bits = repr.bit_depth;
    int hi_bits = old_bits > new_bits ? old_bits : new_bits;
    int lo_bits = old_bits < new_bits ? old_bits : new_bits;
    assert(hi_bits >= lo_bits);

    float mult = 1.0;
    if (!hi_bits || !lo_bits)
        return mult;

    if (pl_color_system_is_ycbcr_like(repr.sys)) {
        // High bit depth YUV uses a range shifted from 8-bit
        mult = (1LL << lo_bits) / ((1LL << hi_bits) - 1.0) * 255.0 / 256;
    } else {
        // Non-YUV always uses the full range available
        mult = ((1LL << lo_bits) - 1.) / ((1LL << hi_bits) - 1.);
    }

    return new_bits >= old_bits ? mult : 1.0 / mult;
}

bool pl_color_primaries_is_wide_gamut(enum pl_color_primaries prim)
{
    switch (prim) {
    case PL_COLOR_PRIM_UNKNOWN:
    case PL_COLOR_PRIM_BT_601_525:
    case PL_COLOR_PRIM_BT_601_625:
    case PL_COLOR_PRIM_BT_709:
    case PL_COLOR_PRIM_BT_470M:
        return false;
    case PL_COLOR_PRIM_BT_2020:
    case PL_COLOR_PRIM_APPLE:
    case PL_COLOR_PRIM_ADOBE:
    case PL_COLOR_PRIM_PRO_PHOTO:
    case PL_COLOR_PRIM_CIE_1931:
    case PL_COLOR_PRIM_DCI_P3:
    case PL_COLOR_PRIM_V_GAMUT:
    case PL_COLOR_PRIM_S_GAMUT:
        return true;
    default: abort();
    }
}

enum pl_color_primaries pl_color_primaries_guess(int width, int height)
{
    // HD content
    if (width >= 1280 || height > 576)
        return PL_COLOR_PRIM_BT_709;

    switch (height) {
    case 576: // Typical PAL content, including anamorphic/squared
        return PL_COLOR_PRIM_BT_601_625;

    case 480: // Typical NTSC content, including squared
    case 486: // NTSC Pro or anamorphic NTSC
        return PL_COLOR_PRIM_BT_601_525;

    default: // No good metric, just pick BT.709 to minimize damage
        return PL_COLOR_PRIM_BT_709;
    }
}

float pl_color_transfer_nominal_peak(enum pl_color_transfer trc)
{
    switch (trc) {
    case PL_COLOR_TRC_UNKNOWN:
    case PL_COLOR_TRC_BT_1886:
    case PL_COLOR_TRC_SRGB:
    case PL_COLOR_TRC_LINEAR:
    case PL_COLOR_TRC_GAMMA18:
    case PL_COLOR_TRC_GAMMA22:
    case PL_COLOR_TRC_GAMMA28:
    case PL_COLOR_TRC_PRO_PHOTO:
        return 1.0;
    case PL_COLOR_TRC_PQ:       return 10000.0 / PL_COLOR_REF_WHITE;
    case PL_COLOR_TRC_HLG:      return 12.0;
    case PL_COLOR_TRC_V_LOG:    return 46.0855;
    case PL_COLOR_TRC_S_LOG1:   return 6.52;
    case PL_COLOR_TRC_S_LOG2:   return 9.212;
    default: abort();
    }
}

bool pl_color_light_is_scene_referred(enum pl_color_light light)
{
    switch (light) {
    case PL_COLOR_LIGHT_UNKNOWN:
    case PL_COLOR_LIGHT_DISPLAY:
        return false;
    case PL_COLOR_LIGHT_SCENE_HLG:
    case PL_COLOR_LIGHT_SCENE_709_1886:
    case PL_COLOR_LIGHT_SCENE_1_2:
        return true;
    default: abort();
    }
}

const struct pl_color_space pl_color_space_unknown = {0};

const struct pl_color_space pl_color_space_srgb = {
    .primaries = PL_COLOR_PRIM_BT_709,
    .transfer  = PL_COLOR_TRC_SRGB,
    .light     = PL_COLOR_LIGHT_DISPLAY,
};

const struct pl_color_space pl_color_space_bt709 = {
    .primaries = PL_COLOR_PRIM_BT_709,
    .transfer  = PL_COLOR_TRC_BT_1886,
    .light     = PL_COLOR_LIGHT_DISPLAY,
};

const struct pl_color_space pl_color_space_hdr10 = {
    .primaries = PL_COLOR_PRIM_BT_2020,
    .transfer  = PL_COLOR_TRC_PQ,
    .light     = PL_COLOR_LIGHT_DISPLAY,
};

const struct pl_color_space pl_color_space_bt2020_hlg = {
    .primaries = PL_COLOR_PRIM_BT_2020,
    .transfer  = PL_COLOR_TRC_HLG,
    .light     = PL_COLOR_LIGHT_SCENE_HLG,
};

void pl_color_space_merge(struct pl_color_space *orig,
                          const struct pl_color_space *new)
{
    if (!orig->primaries)
        orig->primaries = new->primaries;
    if (!orig->transfer)
        orig->transfer = new->transfer;
    if (!orig->light)
        orig->light = new->light;
    if (!orig->sig_peak)
        orig->sig_peak = new->sig_peak;
}

bool pl_color_space_equal(struct pl_color_space c1, struct pl_color_space c2)
{
    return c1.primaries == c2.primaries &&
           c1.transfer  == c2.transfer &&
           c1.light     == c2.light &&
           c1.sig_peak  == c2.sig_peak;
}

const struct pl_color_adjustment pl_color_adjustment_neutral = {
    .brightness = 0.0,
    .contrast   = 1.0,
    .saturation = 1.0,
    .hue        = 0.0,
    .gamma      = 1.0,
};

void pl_chroma_location_offset(enum pl_chroma_location loc, int *x, int *y)
{
    switch (loc) {
    case PL_CHROMA_UNKNOWN:
    case PL_CHROMA_CENTER:
        *x = 0;
        *y = 0;
        return;
    case PL_CHROMA_LEFT:
        *x = -1;
        *y = 0;
        return;
    default: abort();
    }
}

struct pl_raw_primaries pl_raw_primaries_get(enum pl_color_primaries prim)
{
    /*
    Values from: ITU-R Recommendations BT.470-6, BT.601-7, BT.709-5, BT.2020-0

    https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-6-199811-S!!PDF-E.pdf
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.601-7-201103-I!!PDF-E.pdf
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.709-5-200204-I!!PDF-E.pdf
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2020-0-201208-I!!PDF-E.pdf

    Other colorspaces from https://en.wikipedia.org/wiki/RGB_color_space#Specifications
    */

    // CIE standard illuminant series
    static const struct pl_cie_xy
        d50 = {0.34577, 0.35850},
        d65 = {0.31271, 0.32902},
        c   = {0.31006, 0.31616},
        e   = {1.0/3.0, 1.0/3.0};

    switch (prim) {
    case PL_COLOR_PRIM_BT_470M:
        return (struct pl_raw_primaries) {
            .red   = {0.670, 0.330},
            .green = {0.210, 0.710},
            .blue  = {0.140, 0.080},
            .white = c
        };
    case PL_COLOR_PRIM_BT_601_525:
        return (struct pl_raw_primaries) {
            .red   = {0.630, 0.340},
            .green = {0.310, 0.595},
            .blue  = {0.155, 0.070},
            .white = d65
        };
    case PL_COLOR_PRIM_BT_601_625:
        return (struct pl_raw_primaries) {
            .red   = {0.640, 0.330},
            .green = {0.290, 0.600},
            .blue  = {0.150, 0.060},
            .white = d65
        };
    // This is the default assumption if no colorspace information could
    // be determined, eg. for files which have no video channel.
    case PL_COLOR_PRIM_UNKNOWN:
    case PL_COLOR_PRIM_BT_709:
        return (struct pl_raw_primaries) {
            .red   = {0.640, 0.330},
            .green = {0.300, 0.600},
            .blue  = {0.150, 0.060},
            .white = d65
        };
    case PL_COLOR_PRIM_BT_2020:
        return (struct pl_raw_primaries) {
            .red   = {0.708, 0.292},
            .green = {0.170, 0.797},
            .blue  = {0.131, 0.046},
            .white = d65
        };
    case PL_COLOR_PRIM_APPLE:
        return (struct pl_raw_primaries) {
            .red   = {0.625, 0.340},
            .green = {0.280, 0.595},
            .blue  = {0.115, 0.070},
            .white = d65
        };
    case PL_COLOR_PRIM_ADOBE:
        return (struct pl_raw_primaries) {
            .red   = {0.640, 0.330},
            .green = {0.210, 0.710},
            .blue  = {0.150, 0.060},
            .white = d65
        };
    case PL_COLOR_PRIM_PRO_PHOTO:
        return (struct pl_raw_primaries) {
            .red   = {0.7347, 0.2653},
            .green = {0.1596, 0.8404},
            .blue  = {0.0366, 0.0001},
            .white = d50
        };
    case PL_COLOR_PRIM_CIE_1931:
        return (struct pl_raw_primaries) {
            .red   = {0.7347, 0.2653},
            .green = {0.2738, 0.7174},
            .blue  = {0.1666, 0.0089},
            .white = e
        };
    // From SMPTE RP 431-2
    case PL_COLOR_PRIM_DCI_P3:
        return (struct pl_raw_primaries) {
            .red   = {0.680, 0.320},
            .green = {0.265, 0.690},
            .blue  = {0.150, 0.060},
            .white = d65
        };
    // From Panasonic VARICAM reference manual
    case PL_COLOR_PRIM_V_GAMUT:
        return (struct pl_raw_primaries) {
            .red   = {0.730, 0.280},
            .green = {0.165, 0.840},
            .blue  = {0.100, -0.03},
            .white = d65
        };
    // From Sony S-Log reference manual
    case PL_COLOR_PRIM_S_GAMUT:
        return (struct pl_raw_primaries) {
            .red   = {0.730, 0.280},
            .green = {0.140, 0.855},
            .blue  = {0.100, -0.05},
            .white = d65
        };
    default: abort();
    }
}

static void invert_matrix3x3(float m[3][3])
{
    float m00 = m[0][0], m01 = m[0][1], m02 = m[0][2],
          m10 = m[1][0], m11 = m[1][1], m12 = m[1][2],
          m20 = m[2][0], m21 = m[2][1], m22 = m[2][2];

    // calculate the adjoint
    m[0][0] =  (m11 * m22 - m21 * m12);
    m[0][1] = -(m01 * m22 - m21 * m02);
    m[0][2] =  (m01 * m12 - m11 * m02);
    m[1][0] = -(m10 * m22 - m20 * m12);
    m[1][1] =  (m00 * m22 - m20 * m02);
    m[1][2] = -(m00 * m12 - m10 * m02);
    m[2][0] =  (m10 * m21 - m20 * m11);
    m[2][1] = -(m00 * m21 - m20 * m01);
    m[2][2] =  (m00 * m11 - m10 * m01);

    // calculate the determinant (as inverse == 1/det * adjoint,
    // adjoint * m == identity * det, so this calculates the det)
    float det = m00 * m[0][0] + m10 * m[0][1] + m20 * m[0][2];
    det = 1.0f / det;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            m[i][j] *= det;
    }
}

// A := A * B
static void mul_matrix3x3(float a[3][3], float b[3][3])
{
    float a00 = a[0][0], a01 = a[0][1], a02 = a[0][2],
          a10 = a[1][0], a11 = a[1][1], a12 = a[1][2],
          a20 = a[2][0], a21 = a[2][1], a22 = a[2][2];

    for (int i = 0; i < 3; i++) {
        a[0][i] = a00 * b[0][i] + a01 * b[1][i] + a02 * b[2][i];
        a[1][i] = a10 * b[0][i] + a11 * b[1][i] + a12 * b[2][i];
        a[2][i] = a20 * b[0][i] + a21 * b[1][i] + a22 * b[2][i];
    }
}

struct pl_color_matrix pl_color_matrix_invert(struct pl_color_matrix out)
{
    invert_matrix3x3(out.m);
    return out;
}

// based on DarkPlaces engine (relicensed from GPL to LGPL)
struct pl_color_transform pl_color_transform_invert(struct pl_color_transform in)
{
    struct pl_color_transform out = { .mat = pl_color_matrix_invert(in.mat) };
    float m00 = out.mat.m[0][0], m01 = out.mat.m[0][1], m02 = out.mat.m[0][2],
          m10 = out.mat.m[1][0], m11 = out.mat.m[1][1], m12 = out.mat.m[1][2],
          m20 = out.mat.m[2][0], m21 = out.mat.m[2][1], m22 = out.mat.m[2][2];

    // fix the constant coefficient
    // rgb = M * yuv + C
    // M^-1 * rgb = yuv + M^-1 * C
    // yuv = M^-1 * rgb - M^-1 * C
    //                  ^^^^^^^^^^
    out.c[0] = -(m00 * in.c[0] + m01 * in.c[1] + m02 * in.c[2]);
    out.c[1] = -(m10 * in.c[0] + m11 * in.c[1] + m12 * in.c[2]);
    out.c[2] = -(m20 * in.c[0] + m21 * in.c[1] + m22 * in.c[2]);
    return out;
}

// Compute the RGB/XYZ matrix as described here:
// http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
struct pl_color_matrix pl_get_rgb2xyz_matrix(struct pl_raw_primaries prim)
{
    struct pl_color_matrix out = {{{0}}};
    float S[3], X[4], Z[4];

    // Convert from CIE xyY to XYZ. Note that Y=1 holds true for all primaries
    X[0] = prim.red.x   / prim.red.y;
    X[1] = prim.green.x / prim.green.y;
    X[2] = prim.blue.x  / prim.blue.y;
    X[3] = prim.white.x / prim.white.y;

    Z[0] = (1 - prim.red.x   - prim.red.y)   / prim.red.y;
    Z[1] = (1 - prim.green.x - prim.green.y) / prim.green.y;
    Z[2] = (1 - prim.blue.x  - prim.blue.y)  / prim.blue.y;
    Z[3] = (1 - prim.white.x - prim.white.y) / prim.white.y;

    // S = XYZ^-1 * W
    for (int i = 0; i < 3; i++) {
        out.m[0][i] = X[i];
        out.m[1][i] = 1;
        out.m[2][i] = Z[i];
    }

    invert_matrix3x3(out.m);

    for (int i = 0; i < 3; i++)
        S[i] = out.m[i][0] * X[3] + out.m[i][1] * 1 + out.m[i][2] * Z[3];

    // M = [Sc * XYZc]
    for (int i = 0; i < 3; i++) {
        out.m[0][i] = S[i] * X[i];
        out.m[1][i] = S[i] * 1;
        out.m[2][i] = S[i] * Z[i];
    }

    return out;
}

struct pl_color_matrix pl_get_xyz2rgb_matrix(struct pl_raw_primaries prim)
{
    // For simplicity, just invert the rgb2xyz matrix
    struct pl_color_matrix out = pl_get_rgb2xyz_matrix(prim);
    invert_matrix3x3(out.m);
    return out;
}

// M := M * XYZd<-XYZs
static void apply_chromatic_adaptation(struct pl_cie_xy src,
                                       struct pl_cie_xy dest, float m[3][3])
{
    // If the white points are nearly identical, this is a wasteful identity
    // operation.
    if (fabs(src.x - dest.x) < 1e-6 && fabs(src.y - dest.y) < 1e-6)
        return;

    // XYZd<-XYZs = Ma^-1 * (I*[Cd/Cs]) * Ma
    // http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html
    float C[3][2], tmp[3][3] = {{0}};

    // Ma = Bradford matrix, arguably most popular method in use today.
    // This is derived experimentally and thus hard-coded.
    float bradford[3][3] = {
        {  0.8951,  0.2664, -0.1614 },
        { -0.7502,  1.7135,  0.0367 },
        {  0.0389, -0.0685,  1.0296 },
    };

    for (int i = 0; i < 3; i++) {
        // source cone
        C[i][0] = bradford[i][0] * pl_cie_X(src)
                + bradford[i][1] * 1
                + bradford[i][2] * pl_cie_Z(src);

        // dest cone
        C[i][1] = bradford[i][0] * pl_cie_X(dest)
                + bradford[i][1] * 1
                + bradford[i][2] * pl_cie_Z(dest);
    }

    // tmp := I * [Cd/Cs] * Ma
    for (int i = 0; i < 3; i++)
        tmp[i][i] = C[i][1] / C[i][0];

    mul_matrix3x3(tmp, bradford);

    // M := M * Ma^-1 * tmp
    invert_matrix3x3(bradford);
    mul_matrix3x3(m, bradford);
    mul_matrix3x3(m, tmp);
}

struct pl_color_matrix pl_get_color_mapping_matrix(struct pl_raw_primaries src,
                                                   struct pl_raw_primaries dst,
                                                   enum pl_rendering_intent intent)
{
    // In saturation mapping, we don't care about accuracy and just want
    // primaries to map to primaries, making this an identity transformation.
    if (intent == PL_INTENT_SATURATION) {
        return (struct pl_color_matrix) {{
            { 1, 0, 0 },
            { 0, 1, 0 },
            { 0, 0, 1 }
        }};
    }

    // RGBd<-RGBs = RGBd<-XYZd * XYZd<-XYZs * XYZs<-RGBs
    // Equations from: http://www.brucelindbloom.com/index.html?Math.html
    // Note: Perceptual is treated like relative colorimetric. There's no
    // definition for perceptual other than "make it look good".

    // RGBd<-XYZd matrix
    struct pl_color_matrix out = pl_get_xyz2rgb_matrix(dst);

    // Chromatic adaptation, except in absolute colorimetric intent
    if (intent != PL_INTENT_ABSOLUTE_COLORIMETRIC)
        apply_chromatic_adaptation(src.white, dst.white, out.m);

    // XYZs<-RGBs
    struct pl_color_matrix tmp = pl_get_rgb2xyz_matrix(src);
    mul_matrix3x3(out.m, tmp.m);
    return out;
}

/* Fill in the Y, U, V vectors of a yuv-to-rgb conversion matrix
 * based on the given luma weights of the R, G and B components (lr, lg, lb).
 * lr+lg+lb is assumed to equal 1.
 * This function is meant for colorspaces satisfying the following
 * conditions (which are true for common YUV colorspaces):
 * - The mapping from input [Y, U, V] to output [R, G, B] is linear.
 * - Y is the vector [1, 1, 1].  (meaning input Y component maps to 1R+1G+1B)
 * - U maps to a value with zero R and positive B ([0, x, y], y > 0;
 *   i.e. blue and green only).
 * - V maps to a value with zero B and positive R ([x, y, 0], x > 0;
 *   i.e. red and green only).
 * - U and V are orthogonal to the luma vector [lr, lg, lb].
 * - The magnitudes of the vectors U and V are the minimal ones for which
 *   the image of the set Y=[0...1],U=[-0.5...0.5],V=[-0.5...0.5] under the
 *   conversion function will cover the set R=[0...1],G=[0...1],B=[0...1]
 *   (the resulting matrix can be converted for other input/output ranges
 *   outside this function).
 * Under these conditions the given parameters lr, lg, lb uniquely
 * determine the mapping of Y, U, V to R, G, B.
 */
static struct pl_color_matrix luma_coeffs(float lr, float lg, float lb)
{
    assert(fabs(lr+lg+lb - 1) < 1e-6);
    return (struct pl_color_matrix) {{
        {1, 0,                    2 * (1-lr)          },
        {1, -2 * (1-lb) * lb/lg, -2 * (1-lr) * lr/lg  },
        {1,  2 * (1-lb),          0                   },
    }};
}

struct pl_color_transform pl_get_decoding_matrix(struct pl_color_repr repr,
                                                 struct pl_color_adjustment params,
                                                 enum pl_color_levels out_levels,
                                                 int out_bits)
{
    struct pl_color_matrix m;
    switch (repr.sys) {
    case PL_COLOR_SYSTEM_UNKNOWN: // fall through
    case PL_COLOR_SYSTEM_BT_709:     m = luma_coeffs(0.2126, 0.7152, 0.0722); break;
    case PL_COLOR_SYSTEM_BT_601:     m = luma_coeffs(0.2990, 0.5870, 0.1140); break;
    case PL_COLOR_SYSTEM_SMPTE_240M: m = luma_coeffs(0.2122, 0.7013, 0.0865); break;
    case PL_COLOR_SYSTEM_BT_2020_NC: m = luma_coeffs(0.2627, 0.6780, 0.0593); break;
    case PL_COLOR_SYSTEM_BT_2020_C:
        // Note: This outputs into the [-0.5,0.5] range for chroma information.
        m = (struct pl_color_matrix) {{
            {0, 0, 1},
            {1, 0, 0},
            {0, 1, 0}
        }};
        break;
    case PL_COLOR_SYSTEM_YCGCO:
        m = (struct pl_color_matrix) {{
            {1,  -1,  1},
            {1,   1,  0},
            {1,  -1, -1},
        }};
        break;
    case PL_COLOR_SYSTEM_RGB:
        m = (struct pl_color_matrix) {{
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1}
        }};
        break;
    case PL_COLOR_SYSTEM_XYZ:
        // For lack of anything saner to do, just assume the caller wants
        // BT.709 primaries, which is a reasonable assumption.
        m = pl_get_xyz2rgb_matrix(pl_raw_primaries_get(PL_COLOR_PRIM_BT_709));
        break;
    default: abort();
    }

    struct pl_color_transform out = { .mat = m };

    // Apply hue and saturation in the correct way depending on the colorspace.
    if (pl_color_system_is_ycbcr_like(repr.sys)) {
        // Hue is equivalent to rotating input [U, V] subvector around the origin.
        // Saturation scales [U, V].
        float huecos = params.saturation * cos(params.hue);
        float huesin = params.saturation * sin(params.hue);
        for (int i = 0; i < 3; i++) {
            float u = out.mat.m[i][1], v = out.mat.m[i][2];
            out.mat.m[i][1] = huecos * u - huesin * v;
            out.mat.m[i][2] = huesin * u + huecos * v;
        }
    }
    // FIXME: apply saturation for RGB

    float s = 1.0;
    if (repr.bit_depth && out_bits)
        s = pl_color_repr_texture_mul(repr, out_bits);

    // As a convenience, we use the 255-scale values in the code below
    s /= 255.0;

    // NOTE: The yuvfull ranges as presented here are arguably ambiguous,
    // and conflict with at least the full-range YCbCr/ICtCp values as defined
    // by ITU-R BT.2100. If somebody ever complains about full-range YUV looking
    // different from their reference display, this comment is probably why.
    struct yuvlevels { double ymin, ymax, cmax, cmid; }
        yuvlim  = { 16*s, 235*s, 240*s, 128*s },
        yuvfull = {  0*s, 255*s, 255*s, 128*s },
        anyfull = {  0*s, 255*s, 255*s/2, 0 }, // cmax picked to make cmul=ymul
        yuvlev;

    if (pl_color_system_is_ycbcr_like(repr.sys)) {
        switch (repr.levels) {
        case PL_COLOR_LEVELS_UNKNOWN: // fall through
        case PL_COLOR_LEVELS_TV: yuvlev = yuvlim; break;
        case PL_COLOR_LEVELS_PC: yuvlev = yuvfull; break;
        default: abort();
        }
    } else {
        yuvlev = anyfull;
    }

    struct rgblevels { double min, max; }
        rgblim =  { 16/255., 235/255. },
        rgbfull = {      0,        1  },
        rgblev;

    switch (out_levels) {
    case PL_COLOR_LEVELS_UNKNOWN: // fall through
    case PL_COLOR_LEVELS_PC: rgblev = rgbfull; break;
    case PL_COLOR_LEVELS_TV: rgblev = rgblim; break;
    default: abort();
    }

    double ymul = (rgblev.max - rgblev.min) / (yuvlev.ymax - yuvlev.ymin);
    double cmul = (rgblev.max - rgblev.min) / (yuvlev.cmax - yuvlev.cmid) / 2;

    // Contrast scales the output value range (gain)
    ymul *= params.contrast;
    cmul *= params.contrast;

    for (int i = 0; i < 3; i++) {
        out.mat.m[i][0] *= ymul;
        out.mat.m[i][1] *= cmul;
        out.mat.m[i][2] *= cmul;
        // Set c so that Y=umin,UV=cmid maps to RGB=min (black to black),
        // also add brightness offset (black lift)
        out.c[i] = rgblev.min - out.mat.m[i][0] * yuvlev.ymin
                 - (out.mat.m[i][1] + out.mat.m[i][2]) * yuvlev.cmid
                 + params.brightness;
    }

    return out;
}
