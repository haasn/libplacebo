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

#include <libplacebo/colorspace.h>
#include <libplacebo/shaders.h>

// Decode the color into normalized RGB, given a specified color_repr. This
// also takes care of additional pre- and post-conversions requires for the
// "special" color systems (XYZ, BT.2020-C, etc.). If `params` is left as NULL,
// it defaults to &pl_color_adjustment_neutral.
//
// Note: This function always returns PC-range RGB with pre-multiplied alpha.
// It mutates the pl_color_repr to reflect the change.
void pl_shader_decode_color(struct pl_shader *sh, struct pl_color_repr *repr,
                            const struct pl_color_adjustment *params);

// Encodes a color from normalized, PC-range, pre-multiplied RGB into a given
// representation. That is, this performs the inverse operation of
// `pl_shader_decode_color` (sans color adjustments).
void pl_shader_encode_color(struct pl_shader *sh,
                            const struct pl_color_repr *repr);

// Linearize (expand) `vec4 color`, given a specified color_transfer. In
// essence, this loosely corresponds to the ITU-R EOTF, calculated on an
// idealized (reference) monitor with a white point of PL_COLOR_REF_WHITE and
// infinite contrast.
//
// Note: Unlike the ITU-R EOTF, it never includes the OOTF - even for systems
// where the EOTF includes the OOTF (such as HLG).
void pl_shader_linearize(struct pl_shader *sh, enum pl_color_transfer trc);

// Delinearize (compress), given a TRC as output. This loosely corresponds to
// the inverse EOTF (not the OETF) in ITU-R terminology, again assuming a
// reference monitor.
void pl_shader_delinearize(struct pl_shader *sh, enum pl_color_transfer trc);

struct pl_sigmoid_params {
    // The center (bias) of the sigmoid curve. Must be between 0.0 and 1.0.
    // If left as NULL, defaults to 0.75
    float center;

    // The slope (steepness) of the sigmoid curve. Must be between 1.0 and 20.0.
    // If left as NULL, defaults to 6.5.
    float slope;
};

extern const struct pl_sigmoid_params pl_sigmoid_default_params;

// Applies a sigmoidal color transform to all channels. This helps avoid
// ringing artifacts during upscaling by bringing the color information closer
// to neutral and away from the extremes. If `params` is NULL, it defaults to
// &pl_sigmoid_default_params.
//
// Warning: This function clamps the input to the interval [0,1]; and as such
// it should *NOT* be used on already-decoded high-dynamic range content.
void pl_shader_sigmoidize(struct pl_shader *sh,
                          const struct pl_sigmoid_params *params);

// This performs the inverse operation to `pl_shader_sigmoidize`.
void pl_shader_unsigmoidize(struct pl_shader *sh,
                            const struct pl_sigmoid_params *params);

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
    //
    // This is the recommended tone mapping function to use when stretching an
    // SDR curve over an HDR display (i.e. `dst.sig_scale > 1.0`), which can
    // come in handy when calibrating a true HDR display to an SDR curve
    // for compatibility with legacy display stacks.
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

    // The tone mapping algorithm can operate in two modes: The first is known
    // as "desaturating" (per-channel) mode, aka "hollywood/TV" style tone
    // mapping; and the second is called "saturating" (linear) mode, aka
    // "chromatic/colorimetric" tone mapping. The saturating tone mapping
    // algorithm preserves colors from the source faithfully, but can suffer
    // from weird-looking, blown out highlights in very bright regions. To
    // provide a trade-off between these two approaches, we mix the result
    // between the two approaches based on the overall brightness of the pixel.
    //
    // These settings control the parameter of this mixing. The `strength`
    // controls how much of the desaturating result is mixed into the pixel,
    // with values ranging from 0.0 to 1.0 - while the `base` and `exponent`
    // controls the placement and steepness of the mixing curve.
    //
    // If you want to always use the saturating/colorimetric tone mapping, set
    // the strength to 0.0. If you want to always use the desaturating/hollywood
    // tone mapping, set the strength to 1.0 and the exponent to 0.0. The
    // default settings are strength 0.75, exponent 1.5 and base 0.18, which
    // provides a reasonable balance.
    float desaturation_strength;
    float desaturation_exponent;
    float desaturation_base;

    // When tone mapping, this represents the upper limit of how much the
    // scene may be over-exposed in order to hit the `dst.sig_avg` target.
    // If left unset, defaults to 1.0, which corresponds to no boost.
    float max_boost;

    // If true, enables the gamut warning feature. This will visibly highlight
    // all out-of-gamut colors (by inverting them), if they would have been
    // clipped as a result of gamut/tone mapping. (Obviously, this feature only
    // really makes sense with TONE_MAPPING_CLIP)
    bool gamut_warning;

    // If set to something nonzero, this enables the peak detection feature.
    // Controls how many frames to smooth (average) the results over, in order
    // to prevent jitter due to sparkling highlights. Defaults to 63.
    int peak_detect_frames;

    // When using peak detection, setting this to a nonzero value enables
    // scene change detection. If the current frame's average brightness
    // differs from the averaged frame brightness of the previous frames by
    // this much or more, the averaged value will be discarded and the state
    // reset. Doing so helps prevent annoying "eye adaptation"-like effects
    // when transitioning between dark and bright scenes. Defaults to 0.2.
    float scene_threshold;
};

extern const struct pl_color_map_params pl_color_map_default_params;

// Maps `vec4 color` from one color space to another color space according
// to the parameters (described in greater depth above). If `params` is left
// as NULL, it defaults to &pl_color_map_default_params. If `prelinearized`
// is true, the logic will assume the input has already been linearized by the
// caller (e.g. as part of a previous linear light scaling operation).
//
// When the user wishes to use peak detection, `peak_detect_state` should be
// set to the pointer of an object that will hold the state for the frame
// averaging, which must be destroyed by the user when no longer required.
// Successive calls to the same shader should re-use the same object. May
// be safely left as NULL, which will disable the peak detection feature.
//
// Note: Due to the nature of the peak detection implementation, the detected
// metadata is delayed by one frame. This may cause a single frame of wrong
// metadata on rapid scene transitions, or following the start of playback.
void pl_shader_color_map(struct pl_shader *sh,
                         const struct pl_color_map_params *params,
                         struct pl_color_space src, struct pl_color_space dst,
                         struct pl_shader_obj **peak_detect_state,
                         bool prelinearized);

// Applies a set of cone distortion parameters to `vec4 color` in a given color
// space. This can be used to simulate color blindness. See `pl_cone_params`
// for more information.
void pl_shader_cone_distort(struct pl_shader *sh, struct pl_color_space csp,
                            const struct pl_cone_params *params);

enum pl_dither_method {
    // Dither with blue noise. Very high quality, but requires the use of a
    // LUT. Warning: Computing a blue noise texture with a large size can be
    // very slow, however this only needs to be performed once. Even so, using
    // this with a `lut_size` greater than 6 is generally ill-advised. This is
    // the preferred/default dither method.
    PL_DITHER_BLUE_NOISE,

    // Dither with an ordered (bayer) dither matrix, using a LUT. Low quality,
    // and since this also uses a LUT, there's generally no advantage to picking
    // this instead of `PL_DITHER_BLUE_NOISE`. It's mainly there for testing.
    PL_DITHER_ORDERED_LUT,

    // The same as `PL_DITHER_ORDERED_LUT`, but uses fixed function math instead
    // of a LUT. This is faster, but only supports a fixed dither matrix size
    // of 16x16 (equal to a `lut_size` of 4). Requires GLSL 130+.
    PL_DITHER_ORDERED_FIXED,

    // Dither with white noise. This does not require a LUT and is fairly cheap
    // to compute. Unlike the other modes it doesn't show any repeating
    // patterns either spatially or temporally, but the downside is that this
    // is visually fairly jarring due to the presence of low frequencies in the
    // noise spectrum. Used as a fallback when the above methods are not
    // available.
    PL_DITHER_WHITE_NOISE,
};

struct pl_dither_params {
    // The source of the dither noise to use.
    enum pl_dither_method method;

    // For the dither methods which require the use of a LUT, this controls
    // the size of the LUT (base 2). If left as NULL, this defaults to 6, which
    // is equivalent to a 64x64 dither matrix. Must not be larger than 8.
    int lut_size;

    // Enables temporal dithering. This reduces the persistence of dithering
    // artifacts by perturbing the dithering matrix per frame.
    // Warning: This can cause nasty aliasing artifacts on some LCD screens.
    bool temporal;
};

extern const struct pl_dither_params pl_dither_default_params;

// Dither the colors to a lower depth, given in bits. This can be used on input
// colors of any precision. Basically, this rounds the colors to only linear
// multiples of the stated bit depth. The average intensity of the result
// will not change (i.e., the dither noise is balanced in both directions).
// If `params` is NULL, it defaults to &pl_dither_default_params.
//
// For the dither methods which require the use of a LUT, `dither_state` must
// be set to a valid pointer. To avoid thrashing the resource, users should
// avoid trying to re-use the same LUT for different dither configurations. If
// passed as NULL, libplacebo will automatically fall back to dither algorithms
// that don't require the use of a LUT.
//
// Warning: This dithering algorithm is not gamma-invariant; so using it for
// very low bit depths (below 4 or so) will noticeably increase the brightness
// of the resulting image. When doing low bit depth dithering for aesthetic
// purposes, it's recommended that the user explicitly (de)linearize the colors
// before and after this algorithm.
void pl_shader_dither(struct pl_shader *sh, int new_depth,
                      struct pl_shader_obj **dither_state,
                      const struct pl_dither_params *params);

struct pl_3dlut_params {
    // The rendering intent to use when computing the color transformation. A
    // recommended value is PL_INTENT_RELATIVE_COLORIMETRIC for color-accurate
    // video reproduction, or PL_INTENT_PERCEPTUAL for profiles containing
    // meaningful perceptual mapping tables.
    enum pl_rendering_intent intent;

    // The size of the 3DLUT to generate. If left as NULL, these individually
    // default to 64, which is the recommended default for all three.
    size_t size_r, size_g, size_b;
};

extern const struct pl_3dlut_params pl_3dlut_default_params;

struct pl_3dlut_profile {
    // The nominal, closest approximation representation of the color profile,
    // as permitted by `pl_color_space` enums. This will be used as a fallback
    // in the event that an ICC profile is absent, or that parsing the ICC
    // profile fails. This is also that will be returned for the corresponding
    // field in `pl_3dlut_result` when the ICC profile is in use.
    struct pl_color_space color;

    // The ICC profile itself. (Optional)
    struct pl_icc_profile profile;
};

struct pl_3dlut_result {
    // The source color space. This is the color space that the colors should
    // actually be in at the point in time that they're ingested by the 3DLUT.
    // This may differ from the `pl_color_space color` specified in the
    // `pl_color_profile`. Users should make sure to apply
    // `pl_shader_color_map` in order to get the colors into this format before
    // applying `pl_shader_3dlut`.
    //
    // Note: `pl_shader_color_map` is a no-op when the source and destination
    // color spaces are the same, so this can safely be used without disturbing
    // the colors in the event that an ICC profile is actually in use.
    struct pl_color_space src_color;

    // The destination color space. This is the color space that the colors
    // will (nominally) be in at the time they exit the 3DLUT.
    struct pl_color_space dst_color;
};

#if PL_HAVE_LCMS

// Updates/generates a 3DLUT. Returns success. If true, `out` will be updated
// to a struct describing the color space chosen for the input and output of
// the 3DLUT. (See `pl_color_profile`)
// If `params` is NULL, it defaults to &pl_3dlut_default_params.
//
// Note: This function must always be called before `pl_shader_3dlut`, on the
// same `pl_shader` object, The only reason it's separate from `pl_shader_3dlut`
// is to give users a chance to adapt the input colors to the color space
// chosen by the 3DLUT before applying it.
bool pl_3dlut_update(struct pl_shader *sh,
                     const struct pl_3dlut_profile *src,
                     const struct pl_3dlut_profile *dst,
                     struct pl_shader_obj **lut3d,
                     struct pl_3dlut_result *out,
                     const struct pl_3dlut_params *params);

// Actually applies a 3DLUT as generated by `pl_3dlut_update`. The reason this
// is separated from `pl_3dlut_update` is so that the user has the chance to
// correctly map the colors into the specified `src_color` space. This should
// be called only on the `pl_shader_obj` previously updated by
// `pl_3dlut_update`, and only when that function returned true.
void pl_3dlut_apply(struct pl_shader *sh, struct pl_shader_obj **lut3d);

#endif // PL_HAVE_LCMS

#endif // LIBPLACEBO_SHADERS_COLORSPACE_H_
