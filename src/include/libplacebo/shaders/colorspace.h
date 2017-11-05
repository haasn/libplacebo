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

#ifndef LIBPLACEBO_SHADERS_COLORSPACE_H_
#define LIBPLACEBO_SHADERS_COLORSPACE_H_

// Color space transformation shaders. These all input and output a color
// value (PL_SHADER_SIG_COLOR).

#include <stdint.h>

#include "../colorspace.h"
#include "../shaders.h"

// Decode the color into normalized RGB, given a specified color_repr. This
// also takes care of additional pre- and post-conversions requires for the
// "special" color systems (XYZ, BT.2020-C, etc.). The int `texture_bits`, if
// present, indicate the depth of the texture we've sampled the color from -
// similar to the semantics on `pl_get_scaled_decoding_matrix`. If `params`
// is left as NULL, it defaults to &pl_color_adjustment_neutral.
//
// Note: This function always returns PC-range RGB with pre-multiplied alpha.
// It mutates the pl_color_repr to reflect the change.
void pl_shader_decode_color(struct pl_shader *sh, struct pl_color_repr *repr,
                            const struct pl_color_adjustment *params,
                            int texture_bits);

// Linearize (expand) `vec4 color`, given a specified color_transfer. In
// essence, this is the ITU-R EOTF, calculated on an idealized (reference)
// monitor with a white point of PL_COLOR_REF_WHITE and infinite contrast.
void pl_shader_linearize(struct pl_shader *sh, enum pl_color_transfer trc);

// Delinearize (compress), given a TRC as output. This corresponds to the
// inverse EOTF (not the OETF) in ITU-R terminology, again assuming a
// reference monitor.
void pl_shader_delinearize(struct pl_shader *sh, enum pl_color_transfer trc);

// A collection of various tone mapping algorithms supported by libplacebo.
enum pl_tone_mapping_algorithm {
    // Performs no tone-mapping, just clips out-of-gamut colors. Retains perfect
    // color accuracy for in-gamut colors but completely destroys out-of-gamut
    // information.
    PL_TONE_MAPPING_CLIP,

    // Generalization of the reinhard tone mapping algorithm to support an
    // additional linear slope near black. The tone mapping parameter indicates
    // the trade-off between the linear section and the non-linear section.
    // Essentially, for param=0.5, every color value below 0.5 will be mapped
    // linearly, with the higher values being non-linearly tone mapped. Values
    // near 1.0 make this curve behave like CLIP, and values near 0.0 make this
    // curve behave like REINHARD. The default value is 0.3, which provides a
    // good balance between colorimetric accuracy and preserving out-of-gamut
    // details. The name is derived from its function shape (ax+b)/(cx+d), which
    // is known as a Möbius transformation in mathematics.
    PL_TONE_MAPPING_MOBIUS,

    // Simple non-linear, global tone mapping algorithm. Named after Erik
    // Reinhard. The parameter specifies the local contrast coefficient at the
    // display peak. Essentially, a value of param=0.5 implies that the
    // reference white will be about half as bright as when clipping. Defaults
    // to 0.5, which results in the simplest formulation of this function.
    PL_TONE_MAPPING_REINHARD,

    // Piece-wise, filmic tone-mapping algorithm developed by John Hable for
    // use in Uncharted 2, inspired by a similar tone-mapping algorithm used by
    // Kodak. Popularized by its use in video games with HDR rendering.
    // Preserves both dark and bright details very well, but comes with the
    // drawback of darkening the overall image quite significantly. Users are
    // recommended to use HDR peak detection to compensate for the missing
    // brightness. This is sort of similar to REINHARD tone-mapping + parameter
    // 0.24.
    PL_TONE_MAPPING_HABLE,

    // Fits a gamma (power) function to transfer between the source and target
    // color spaces. This preserves details at all scales fairly accurately,
    // but can result in an image with a muted or dull appearance. Best when
    // combined with peak detection. The parameter is used as the exponent of
    // the gamma function, defaulting to 1.8.
    PL_TONE_MAPPING_GAMMA,

    // Linearly stretches the source gamut to the destination gamut. This will
    // preserve all details accurately, but results in a significantly darker
    // image. Best when combined with peak detection. The parameter can be used
    // as an aditional scaling coefficient to make the image (linearly)
    // brighter or darker. Defaults to 1.0.
    PL_TONE_MAPPING_LINEAR,
};

struct pl_color_map_params {
    // The rendering intent to use for RGB->RGB primary conversions.
    // Defaults to PL_INTENT_RELATIVE_COLORIMETRIC.
    enum pl_rendering_intent intent;

    // Algorithm and configuration used for tone-mapping. For non-tunable
    // algorithms, the `param` is ignored. If the tone mapping parameter is
    // left as 0.0, the tone-mapping curve's preferred default parameter will
    // be used. The default algorithm is PL_TONE_MAPPING_HABLE.
    enum pl_tone_mapping_algorithm tone_mapping_algo;
    float tone_mapping_param;

    // Desaturation coefficient. This essentially desaturates very bright
    // spectral colors towards white, resulting in a more natural-looking
    // depiction of very bright sunlit regions or images of the sunlit sky. The
    // coefficient indicates the strength of the desaturation - higher values
    // desaturate more strongly. The default value is 0.5, which is fairly
    // conservative - due in part to the excessive use of extremely bright
    // scenes in badly mastered HDR content. Using a value of 1.0 makes it
    // approximately match the desaturation strength used by the ACES ODT. A
    // setting of 0.0 disables this.
    float tone_mapping_desaturate;

    // If true, enables the gamut warning feature. This will visibly highlight
    // all out-of-gamut colors (by inverting them), if they would have been
    // clipped as a result of gamut/tone mapping. (Obviously, this feature only
    // really makes sense with TONE_MAPPING_CLIP)
    bool gamut_warning;

    // If peak_detect_state is set to a valid pointer, this enables the peak
    // detection feature. The resource object will be implicitly created and
    // updated by pl_shader_color_map, but must be destroyed by the caller when
    // no longer needed. Subsequent calls to pl_color_map for subsequent frames
    // should re-use the same peak_detect_state. `peak_detect_frames` defaults
    // to 10 and specifies how many frames to smooth (average) over, and must
    // be at least 1.
    struct pl_shader_obj **peak_detect_state;
    int peak_detect_frames;
};

extern const struct pl_color_map_params pl_color_map_default_params;

// Maps `vec4 color` from one color space to another color space according
// to the parameters (described in greater depth above). If `params` is left
// as NULL, it defaults to &pl_color_map_default_params. If `prelinearized`
// is true, the logic will assume the input has already been linearized by the
// caller (e.g. as part of a previous linear light scaling operation).
void pl_shader_color_map(struct pl_shader *sh,
                         const struct pl_color_map_params *params,
                         struct pl_color_space src, struct pl_color_space dst,
                         bool prelinearized);

enum pl_dither_method {
    // Ordered dither. Low quality, but cheap. Requires GLSL 130+.
    PL_DITHER_ORDERED,

    // Dither with white noise. Also fairly cheap, and unlike the other modes
    // it doesn't show any repeating patterns either spatially or temporally,
    // but the downside is that this is visually fairly jarring due to the
    // presence of low frequencies.
    PL_DITHER_WHITE_NOISE,
};

struct pl_dither_params {
    // The source of the dither noise to use.
    enum pl_dither_method method;

    // Enables temporal dithering. This reduces the persistence of dithering
    // artifacts by perturbing the dithering matrix per frame. When enabled,
    // `temporal_dither_index` should be incremented by 1 for successive frames.
    //
    // Warning: This can cause nasty aliasing artifacts on some LCD screens.
    bool temporal;
    uint8_t temporal_index;
};

// Dither the colors to a lower depth, given in bits. This can be used on input
// colors of any precision. Basically, this rounds the colors to only linear
// multiples of the stated bit depth. The average intensity of the result
// will not change (i.e., the dither noise is balanced in both directions)
//
// Warning: This dithering algorithm is not gamma-invariant; so using it for
// very low bit depths (below 4 or so) will noticeably increase the brightness
// of the resulting image. When doing low bit depth dithering for aesthetic
// purposes, it's recommended that the user explicitly (de)linearize the colors
// before and after this algorithm.
void pl_shader_dither(struct pl_shader *sh, int new_depth,
                      const struct pl_dither_params *params);

#endif // LIBPLACEBO_SHADERS_COLORSPACE_H_
