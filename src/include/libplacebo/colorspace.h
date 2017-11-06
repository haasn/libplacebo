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

#ifndef LIBPLACEBO_COLORSPACE_H_
#define LIBPLACEBO_COLORSPACE_H_

#include <stdbool.h>
#include "common.h"

// The underlying color representation (e.g. RGB, XYZ or YCbCr)
enum pl_color_system {
    PL_COLOR_SYSTEM_UNKNOWN = 0,
    // YCbCr-like color systems:
    PL_COLOR_SYSTEM_BT_601,      // ITU-R Rec. BT.601 (SD)
    PL_COLOR_SYSTEM_BT_709,      // ITU-R Rec. BT.709 (HD)
    PL_COLOR_SYSTEM_SMPTE_240M,  // SMPTE-240M
    PL_COLOR_SYSTEM_BT_2020_NC,  // ITU-R Rec. BT.2020 (non-constant luminance)
    PL_COLOR_SYSTEM_BT_2020_C,   // ITU-R Rec. BT.2020 (constant luminance)
    PL_COLOR_SYSTEM_YCGCO,       // YCgCo (derived from RGB)
    // Other color systems:
    PL_COLOR_SYSTEM_RGB,         // Red, Green and Blue
    PL_COLOR_SYSTEM_XYZ,         // CIE 1931 XYZ, pre-encoded with gamma 2.6
    PL_COLOR_SYSTEM_COUNT
};

bool pl_color_system_is_ycbcr_like(enum pl_color_system sys);

// Returns true for color systems that are linear transformations of the RGB
// equivalent, i.e. are simple matrix multiplications. For color systems with
// this property, pl_get_decoding_matrix is sufficient for conversion to RGB.
bool pl_color_system_is_linear(enum pl_color_system sys);

// Guesses the best YCbCr-like colorspace based on a image given resolution.
// This only picks conservative values. (In particular, BT.2020 is never
// auto-guessed, even for 4K resolution content)
enum pl_color_system pl_color_system_guess_ycbcr(int width, int height);

// The numerical range of the representation (where applicable).
enum pl_color_levels {
    PL_COLOR_LEVELS_UNKNOWN = 0,
    PL_COLOR_LEVELS_TV,         // TV range, e.g. 16-235
    PL_COLOR_LEVELS_PC,         // PC range, e.g. 0-255
    PL_COLOR_LEVELS_COUNT,
};

// The alpha representation mode.
enum pl_alpha_mode {
    PL_ALPHA_UNKNOWN = 0,   // or no alpha channel present
    PL_ALPHA_INDEPENDENT ,  // alpha channel is separate from the video
    PL_ALPHA_PREMULTIPLIED, // alpha channel is multiplied into the colors
};

// The underlying bit-wise representation of a color sample. For example,
// a 10-bit TV-range YCbCr value uploaded to a 16 bit texture would have
// sample_depth=16 color_depth=10 bit_shift=0.
//
// For another example, a 12-bit XYZ full range sample shifted to 16-bits with
// the lower 4 bits all set to 0 would have sample_depth=16 color_depth=12
// bit_shift=4. (libavcodec likes outputting this type of `xyz12`)
//
// To explain the meaning of `sample_depth` further; the consideration factor
// here is the fact that GPU sampling will normalized the sampled color to the
// range 0.0 - 1.0 in a manner dependent on the number of bits in the texture
// format. So if you upload a 10-bit YCbCr value unpadded as 16-bit color
// samples, all of the sampled values will be extremely close to 0.0. In such a
// case, `pl_color_repr_normalize` would return a high scaling factor, which
// would pull the color up to their 16-bit range.
struct pl_bit_encoding {
    int sample_depth; // the number of bits the color is stored/sampled as
    int color_depth;  // the effect number of bits of the color information
    int bit_shift;    // a representational bit shift applied to the color
};

// Returns whether two bit encodings are exactly identical.
bool pl_bit_encoding_equal(const struct pl_bit_encoding *b1,
                           const struct pl_bit_encoding *b2);

// Struct describing the underlying color system and representation. This
// information is needed to convert an encoded color to a normalized RGB triple
// in the range 0-1.
struct pl_color_repr {
    enum pl_color_system sys;
    enum pl_color_levels levels;
    enum pl_alpha_mode alpha;
    struct pl_bit_encoding bits; // or {0} if unknown
};

extern const struct pl_color_repr pl_color_repr_unknown;

// Returns whether two colorspace representations are exactly identical.
bool pl_color_repr_equal(const struct pl_color_repr *c1,
                         const struct pl_color_repr *c2);

// Replaces unknown values in the first struct by those of the second struct.
void pl_color_repr_merge(struct pl_color_repr *orig,
                         const struct pl_color_repr *new);

// This function normalizes the color representation such that
// color_depth=sample_depth and bit_shift=0; and returns the scaling factor
// that must be multiplied into the color value to accomplish this, assuming
// it has already been sampled by the GPU. If unknown, the color and sample
// depth will both be inferred as 8 bits for the purposes of this conversion.
float pl_color_repr_normalize(struct pl_color_repr *repr);

// The colorspace's primaries (gamut)
enum pl_color_primaries {
    PL_COLOR_PRIM_UNKNOWN = 0,
    // Standard gamut:
    PL_COLOR_PRIM_BT_601_525,   // ITU-R Rec. BT.601 (525-line = NTSC, SMPTE-C)
    PL_COLOR_PRIM_BT_601_625,   // ITU-R Rec. BT.601 (625-line = PAL, SECAM)
    PL_COLOR_PRIM_BT_709,       // ITU-R Rec. BT.709 (HD), also sRGB
    PL_COLOR_PRIM_BT_470M,      // ITU-R Rec. BT.470 M
    // Wide gamut:
    PL_COLOR_PRIM_BT_2020,      // ITU-R Rec. BT.2020 (UltraHD)
    PL_COLOR_PRIM_APPLE,        // Apple RGB
    PL_COLOR_PRIM_ADOBE,        // Adobe RGB (1998)
    PL_COLOR_PRIM_PRO_PHOTO,    // ProPhoto RGB (ROMM)
    PL_COLOR_PRIM_CIE_1931,     // CIE 1931 RGB primaries
    PL_COLOR_PRIM_DCI_P3,       // DCI-P3 (Digital Cinema)
    PL_COLOR_PRIM_V_GAMUT,      // Panasonic V-Gamut (VARICAM)
    PL_COLOR_PRIM_S_GAMUT,      // Sony S-Gamut
    PL_COLOR_PRIM_COUNT
};

bool pl_color_primaries_is_wide_gamut(enum pl_color_primaries prim);

// Guesses the best primaries based on a resolution. This always guesses
// conservatively, i.e. it will never return a wide gamut color space even if
// the resolution is 4K.
enum pl_color_primaries pl_color_primaries_guess(int width, int height);

// The colorspace's transfer function (gamma / EOTF)
enum pl_color_transfer {
    PL_COLOR_TRC_UNKNOWN = 0,
    // Standard dynamic range:
    PL_COLOR_TRC_BT_1886,       // ITU-R Rec. BT.1886 (CRT emulation + OOTF)
    PL_COLOR_TRC_SRGB,          // IEC 61966-2-4 sRGB (CRT emulation)
    PL_COLOR_TRC_LINEAR,        // Linear light content
    PL_COLOR_TRC_GAMMA18,       // Pure power gamma 1.8
    PL_COLOR_TRC_GAMMA22,       // Pure power gamma 2.2
    PL_COLOR_TRC_GAMMA28,       // Pure power gamma 2.8
    PL_COLOR_TRC_PRO_PHOTO,     // ProPhoto RGB (ROMM)
    // High dynamic range:
    PL_COLOR_TRC_PQ,            // ITU-R BT.2100 PQ (perceptual quantizer), aka SMPTE ST2048
    PL_COLOR_TRC_HLG,           // ITU-R BT.2100 HLG (hybrid log-gamma), aka ARIB STD-B67
    PL_COLOR_TRC_V_LOG,         // Panasonic V-Log (VARICAM)
    PL_COLOR_TRC_S_LOG1,        // Sony S-Log1
    PL_COLOR_TRC_S_LOG2,        // Sony S-Log2
    PL_COLOR_TRC_COUNT
};

// Returns the nominal peak of a given transfer function, relative to the
// reference white. This refers to the highest encodable signal level.
// Always equal to 1.0 for SDR curves.
float pl_color_transfer_nominal_peak(enum pl_color_transfer trc);

static inline bool pl_color_transfer_is_hdr(enum pl_color_transfer trc)
{
    return pl_color_transfer_nominal_peak(trc) > 1.0;
}

// This defines the standard reference white level (in cd/m^2) that is assumed
// throughout standards such as those from by ITU-R, EBU, etc.
// This is particularly relevant for HDR conversions, as this value is used
// as a reference for conversions between absolute transfer curves (e.g. PQ)
// and relative transfer curves (e.g. SDR, HLG).
#define PL_COLOR_REF_WHITE 100.0

// The semantic interpretation of the decoded image, how is it mastered?
enum pl_color_light {
    PL_COLOR_LIGHT_UNKNOWN = 0,
    PL_COLOR_LIGHT_DISPLAY,     // Display-referred, output as-is
    PL_COLOR_LIGHT_SCENE_HLG,   // Scene-referred, HLG OOTF
    PL_COLOR_LIGHT_SCENE_709_1886, // Scene-referred, OOTF = 709/1886 interaction
    PL_COLOR_LIGHT_SCENE_1_2,   // Scene-referred, OOTF = gamma 1.2
    PL_COLOR_LIGHT_COUNT
};

bool pl_color_light_is_scene_referred(enum pl_color_light light);

// Rendering intent for colorspace transformations. These constants match the
// ICC specification (Table 23)
enum pl_rendering_intent {
    PL_INTENT_PERCEPTUAL = 0,
    PL_INTENT_RELATIVE_COLORIMETRIC = 1,
    PL_INTENT_SATURATION = 2,
    PL_INTENT_ABSOLUTE_COLORIMETRIC = 3
};

// Struct describing a physical color space. This information is needed to
// turn a normalized RGB triple into its physical meaning, as well as to convert
// between color spaces.
struct pl_color_space {
    enum pl_color_primaries primaries;
    enum pl_color_transfer transfer;
    enum pl_color_light light;

    // The highest value that occurs in the signal, relative to the reference
    // white. (0 = unknown)
    float sig_peak;

    // The average light level that occurs in the signal, relative to the
    // reference white. (0 = unknown)
    float sig_avg;
};

// Replaces unknown values in the first struct by those of the second struct.
void pl_color_space_merge(struct pl_color_space *orig,
                          const struct pl_color_space *new);

// Returns whether two colorspaces are exactly identical.
bool pl_color_space_equal(struct pl_color_space c1, struct pl_color_space c2);

// Some common color spaces
extern const struct pl_color_space pl_color_space_unknown;
extern const struct pl_color_space pl_color_space_srgb;
extern const struct pl_color_space pl_color_space_bt709;
extern const struct pl_color_space pl_color_space_hdr10;
extern const struct pl_color_space pl_color_space_bt2020_hlg;
extern const struct pl_color_space pl_color_space_monitor; // typical display

// This represents metadata about extra operations to perform during colorspace
// conversion, which correspond to artistic adjustments of the color.
struct pl_color_adjustment {
    // Brightness boost. 0.0 = neutral, 1.0 = solid white, -1.0 = solid black
    float brightness;
    // Contrast boost. 1.0 = neutral, 0.0 = solid black
    float contrast;
    // Saturation gain. 1.0 = neutral, 0.0 = grayscale
    float saturation;
    // Hue shift, corresponding to a rotation around the [U, V] subvector.
    // Only meaningful for YCbCr-like colorspaces. 0.0 = neutral
    float hue;
    // Gamma adjustment. 1.0 = neutral, 0.0 = solid black
    float gamma;
};

// A struct pre-filled with all-neutral values.
extern const struct pl_color_adjustment pl_color_adjustment_neutral;

// Represents the chroma placement with respect to the luma samples. This is
// only relevant for YCbCr-like colorspaces with chroma subsampling.
enum pl_chroma_location {
    PL_CHROMA_UNKNOWN = 0,
    PL_CHROMA_LEFT,             // MPEG2/4, H.264
    PL_CHROMA_CENTER,           // MPEG1, JPEG
    PL_CHROMA_COUNT,
};

// Fills *x and *y with the offset in luma pixels corresponding to a given
// chroma location.
void pl_chroma_location_offset(enum pl_chroma_location loc, float *x, float *y);

// Represents a single CIE xy coordinate (e.g. CIE Yxy with Y = 1.0)
struct pl_cie_xy {
    float x, y;
};

// Recovers (X / Y) from a CIE xy value.
static inline float pl_cie_X(struct pl_cie_xy xy) {
    return xy.x / xy.y;
}

// Recovers (Z / Y) from a CIE xy value.
static inline float pl_cie_Z(struct pl_cie_xy xy) {
    return (1 - xy.x - xy.y) / xy.y;
}

// Represents the raw physical primaries corresponding to a color space.
struct pl_raw_primaries {
    struct pl_cie_xy red, green, blue, white;
};

// Returns the raw primaries for a given color space.
const struct pl_raw_primaries *pl_raw_primaries_get(enum pl_color_primaries prim);

// Returns an RGB->XYZ conversion matrix for a given set of primaries.
// Multiplying this into the RGB color transforms it to CIE XYZ, centered
// around the color space's white point.
struct pl_matrix3x3 pl_get_rgb2xyz_matrix(const struct pl_raw_primaries *prim);

// Similar to pl_get_rgb2xyz_matrix, but gives the inverse transformation.
struct pl_matrix3x3 pl_get_xyz2rgb_matrix(const struct pl_raw_primaries *prim);

// Returns a primary adaptation matrix, which converts from one set of
// primaries to another. This is an RGB->RGB transformation. For rendering
// intents other than PL_INTENT_ABSOLUTE_COLORIMETRIC, the white point is
// adapted using the Bradford matrix.
struct pl_matrix3x3 pl_get_color_mapping_matrix(const struct pl_raw_primaries *src,
                                                const struct pl_raw_primaries *dst,
                                                enum pl_rendering_intent intent);

// Returns a color decoding matrix for a given combination of source color
// representation and adjustment parameters. This mutates the color_repr to
// reflect the change. If `params` is left as NULL, it defaults to
// &pl_color_adjustment_neutral.
//
// This function always performs a conversion to RGB; conversions from
// arbitrary color representations to other arbitrary other color
// representations are currently not supported. Not all color systems support
// all of the color adjustment parameters. (In particular, hue/sat adjustments
// are currently only supported for YCbCr-like color systems)
//
// Note: For BT.2020 constant-luminance, this outputs chroma information in the
// range [-0.5, 0.5]. Since the CL system conversion is non-linear, further
// processing must be done by the caller. The channel order is CrYCb.
//
// Note: For XYZ system, the input/encoding gamma must be pre-applied by the
// user, typically this has a value of 2.6.
struct pl_transform3x3 pl_color_repr_decode(struct pl_color_repr *repr,
                                    const struct pl_color_adjustment *params);

#endif // LIBPLACEBO_COLORSPACE_H_
