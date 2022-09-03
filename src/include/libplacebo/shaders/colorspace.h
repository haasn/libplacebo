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

#include <libplacebo/colorspace.h>
#include <libplacebo/tone_mapping.h>
#include <libplacebo/shaders.h>

// For backwards compatibility
#include <libplacebo/shaders/dithering.h>

PL_API_BEGIN

// Transform the input color, in its given representation, to ensure
// compatibility with the indicated alpha mode. Mutates `repr` to reflect the
// change. Note that this is a no-op if the input is PL_ALPHA_UNKNOWN.
void pl_shader_set_alpha(pl_shader sh, struct pl_color_repr *repr,
                         enum pl_alpha_mode mode);

// Colorspace reshaping for PL_COLOR_SYSTEM_DOLBYVISION. Note that this is done
// automatically by `pl_shader_decode_color` for PL_COLOR_SYSTEM_DOLBYVISION.
void pl_shader_dovi_reshape(pl_shader sh, const struct pl_dovi_metadata *data);

// Decode the color into normalized RGB, given a specified color_repr. This
// also takes care of additional pre- and post-conversions requires for the
// "special" color systems (XYZ, BT.2020-C, etc.). If `params` is left as NULL,
// it defaults to &pl_color_adjustment_neutral.
//
// Note: This function always returns PC-range RGB with independent alpha.
// It mutates the pl_color_repr to reflect the change.
void pl_shader_decode_color(pl_shader sh, struct pl_color_repr *repr,
                            const struct pl_color_adjustment *params);

// Encodes a color from normalized, PC-range, independent alpha RGB into a
// given representation. That is, this performs the inverse operation of
// `pl_shader_decode_color` (sans color adjustments).
void pl_shader_encode_color(pl_shader sh, const struct pl_color_repr *repr);

// Linearize (expand) `vec4 color`, given a specified color space. In essence,
// this corresponds to the ITU-R EOTF.
//
// Note: Unlike the ITU-R EOTF, it never includes the OOTF - even for systems
// where the EOTF includes the OOTF (such as HLG).
void pl_shader_linearize(pl_shader sh, const struct pl_color_space *csp);

// Delinearize (compress), given a color space as output. This loosely
// corresponds to the inverse EOTF (not the OETF) in ITU-R terminology, again
// assuming a reference monitor.
void pl_shader_delinearize(pl_shader sh, const struct pl_color_space *csp);

struct pl_sigmoid_params {
    // The center (bias) of the sigmoid curve. Must be between 0.0 and 1.0.
    // If left as NULL, defaults to 0.75
    float center;

    // The slope (steepness) of the sigmoid curve. Must be between 1.0 and 20.0.
    // If left as NULL, defaults to 6.5.
    float slope;
};

#define PL_SIGMOID_DEFAULTS \
    .center = 0.75,         \
    .slope  = 6.50,

#define pl_sigmoid_params(...) (&(struct pl_sigmoid_params) { PL_SIGMOID_DEFAULTS __VA_ARGS__ })
extern const struct pl_sigmoid_params pl_sigmoid_default_params;

// Applies a sigmoidal color transform to all channels. This helps avoid
// ringing artifacts during upscaling by bringing the color information closer
// to neutral and away from the extremes. If `params` is NULL, it defaults to
// &pl_sigmoid_default_params.
//
// Warning: This function clamps the input to the interval [0,1]; and as such
// it should *NOT* be used on already-decoded high-dynamic range content.
void pl_shader_sigmoidize(pl_shader sh, const struct pl_sigmoid_params *params);

// This performs the inverse operation to `pl_shader_sigmoidize`.
void pl_shader_unsigmoidize(pl_shader sh, const struct pl_sigmoid_params *params);

struct pl_peak_detect_params {
    // Smoothing coefficient for the detected values. This controls the time
    // parameter (tau) of an IIR low pass filter. In other words, it represent
    // the cutoff period (= 1 / cutoff frequency) in frames. Frequencies below
    // this length will be suppressed. This helps block out annoying
    // "sparkling" or "flickering" due to small variations in frame-to-frame
    // brightness.
    //
    // If left unset, this defaults to 100.0.
    float smoothing_period;

    // In order to avoid reacting sluggishly on scene changes as a result of
    // the low-pass filter, we disable it when the difference between the
    // current frame brightness and the average frame brightness exceeds a
    // given threshold difference. But rather than a single hard cutoff, which
    // would lead to weird discontinuities on fades, we gradually disable it
    // over a small window of brightness ranges. These parameters control the
    // lower and upper bounds of this window, in dB.
    //
    // The default values are 5.5 and 10.0, respectively. To disable this logic
    // entirely, set either one to a negative value.
    float scene_threshold_low;
    float scene_threshold_high;

    // In order to avoid clipping on fade-ins or other sudden brightness
    // increases, we always over-estimate the peak brightness (in percent)
    // by this amount, as a percentage of the actual measured peak. If left
    // as 0.0, this logic is disabled. The default value is 0.05.
    float overshoot_margin;

    // To avoid over-tone-mapping very dark scenes (or black frames), this
    // imposes a hard lower bound on the detected peak. If left as 0.0, it
    // instead defaults to a value of 1.0.
    float minimum_peak;
};

#define PL_PEAK_DETECT_DEFAULTS         \
    .smoothing_period       = 100.0,    \
    .scene_threshold_low    = 5.5,      \
    .scene_threshold_high   = 10.0,     \
    .overshoot_margin       = 0.05,     \
    .minimum_peak           = 1.0,

#define pl_peak_detect_params(...) (&(struct pl_peak_detect_params) { PL_PEAK_DETECT_DEFAULTS __VA_ARGS__ })
extern const struct pl_peak_detect_params pl_peak_detect_default_params;

// This function can be used to measure the CLL and FALL of a video
// source automatically, using a compute shader. The measured values are
// smoothed automatically (depending on the parameters), so to keep track of
// the measured results over time, a tone mapping shader state object is used
// to hold the state. Returns false on failure initializing the tone mapping
// object, or if compute shaders are not supported.
//
// It's important that the same shader object is used for successive frames
// belonging to the same source. If the source changes (e.g. due to a file
// change or seek), the user should reset it with `pl_reset_detected_peak` (or
// destroy it and use a new state object).
//
// The parameter `csp` holds the representation of the color values that are
// the input to this function. (They must already be in decoded RGB form, i.e.
// alternate color representations are not supported)
bool pl_shader_detect_peak(pl_shader sh, struct pl_color_space csp,
                           pl_shader_obj *state,
                           const struct pl_peak_detect_params *params);

// After dispatching the above shader, this function *may* be used to read out
// the detected CLL and FALL directly (in PL_HDR_NORM units). If the shader
// has never been dispatched yet, i.e. no information is available, this will
// return false.
//
// Note: This function will block until the shader object is no longer in use
// by the GPU, so its use should be avoided due to performance reasons. This
// function is *not* needed when the user only wants to use `pl_shader_color_map`,
// since that can ingest the results from the state object directly. It only
// serves as a utility/debugging function.
bool pl_get_detected_peak(const pl_shader_obj state,
                          float *out_cll, float *out_fall);

// Resets the peak detection state in a given tone mapping state object. This
// is not equal to `pl_shader_obj_destroy`, because it does not destroy any
// state used by `pl_shader_tone_map`.
void pl_reset_detected_peak(pl_shader_obj state);

// Deprecated. See <libplacebo/tone_mapping.h> for replacements.
enum pl_tone_mapping_algorithm {
    PL_TONE_MAPPING_CLIP,
    PL_TONE_MAPPING_MOBIUS,
    PL_TONE_MAPPING_REINHARD,
    PL_TONE_MAPPING_HABLE,
    PL_TONE_MAPPING_GAMMA,
    PL_TONE_MAPPING_LINEAR,
    PL_TONE_MAPPING_BT_2390,
    PL_TONE_MAPPING_ALGORITHM_COUNT,
};

enum pl_tone_map_mode {
    // Picks the best tone-mapping mode based on internal heuristics.
    PL_TONE_MAP_AUTO,

    // Per-channel tone-mapping in RGB. Guarantees no clipping and heavily
    // desaturates the output, but distorts the colors quite significantly.
    PL_TONE_MAP_RGB,

    // Tone-mapping is performed on the brightest component found in the
    // signal. Good at preserving details in highlights, but has a tendency to
    // crush blacks.
    PL_TONE_MAP_MAX,

    // Tone-map per-channel for highlights and linearly (luma-based) for
    // midtones/shadows, based on a fixed gamma 2.4 coefficient curve.
    PL_TONE_MAP_HYBRID,

    // Tone-map linearly on the luma component, and adjust (desaturate) the
    // chromaticities to compensate using a simple constant factor. This is
    // essentially the mode used in ITU-R BT.2446 method A.
    PL_TONE_MAP_LUMA,

    PL_TONE_MAP_MODE_COUNT,
};

enum pl_gamut_mode {
    // Do nothing, simply clip out-of-range colors to the RGB volume.
    PL_GAMUT_CLIP,

    // Equal to PL_GAMUT_CLIP but also highlights out-of-gamut colors (by
    // coloring them pink).
    PL_GAMUT_WARN,

    // Linearly reduces content brightness to preserves saturated details,
    // followed by clipping the remaining out-of-gamut colors. As the name
    // implies, this makes everything darker, but provides a good balance
    // between preserving details and colors.
    PL_GAMUT_DARKEN,

    // Hard-desaturates out-of-gamut colors towards white, while preserving the
    // luminance. Has a tendency to shift colors.
    PL_GAMUT_DESATURATE,

    PL_GAMUT_MODE_COUNT,
};

struct pl_color_map_params {
    // The rendering intent to use for gamut mapping. Note that this does not
    // affect tone mapping, which is always applied independently (to get the
    // equivalent of colorimetric intent for tone mapping, set the function to
    // NULL).
    //
    // Defaults to PL_INTENT_RELATIVE_COLORIMETRIC
    enum pl_rendering_intent intent;

    // How to handle out-of-gamut colors when changing the content primaries.
    enum pl_gamut_mode gamut_mode;

    // Function and configuration used for tone-mapping. For non-tunable
    // functions, the `param` is ignored. If the tone mapping parameter is
    // left as 0.0, the tone-mapping curve's preferred default parameter will
    // be used. The default function is pl_tone_map_auto.
    //
    // Note: This pointer changing invalidates the LUT, so make sure to only
    // use stable (or static) storage for the pl_tone_map_function.
    const struct pl_tone_map_function *tone_mapping_function;
    enum pl_tone_map_mode tone_mapping_mode;
    float tone_mapping_param;

    // If true, and supported by the given tone mapping function, libplacebo
    // will perform inverse tone mapping to expand the dynamic range of a
    // signal. libplacebo is not liable for any HDR-induced eye damage.
    bool inverse_tone_mapping;

    // Extra crosstalk factor to apply before tone-mapping. Optional. May help
    // to improve the appearance of very bright, monochromatic highlights.
    float tone_mapping_crosstalk;

    // Tone mapping LUT size. Defaults to 256. Note that when combining
    // this with peak detection, the resulting LUT is actually squared, so
    // avoid setting it too high.
    int lut_size;

    // --- Debugging options

    // Force the use of a full tone-mapping LUT even for functions that have
    // faster pure GLSL replacements (e.g. clip).
    bool force_tone_mapping_lut;

    // --- Deprecated fields
    enum pl_tone_mapping_algorithm tone_mapping_algo PL_DEPRECATED;
    float desaturation_strength PL_DEPRECATED;
    float desaturation_exponent PL_DEPRECATED;
    float desaturation_base PL_DEPRECATED;
    float max_boost PL_DEPRECATED;
    bool gamut_warning PL_DEPRECATED;   // replaced by PL_GAMUT_WARN
    bool gamut_clipping PL_DEPRECATED;  // replaced by PL_GAMUT_DESATURATE
};

#define PL_COLOR_MAP_DEFAULTS                                   \
    .intent                 = PL_INTENT_RELATIVE_COLORIMETRIC,  \
    .gamut_mode             = PL_GAMUT_CLIP,                    \
    .tone_mapping_function  = &pl_tone_map_auto,                \
    .tone_mapping_mode      = PL_TONE_MAP_AUTO,                 \
    .tone_mapping_crosstalk = 0.04,                             \
    .lut_size               = 256,

#define pl_color_map_params(...) (&(struct pl_color_map_params) { PL_COLOR_MAP_DEFAULTS __VA_ARGS__ })
extern const struct pl_color_map_params pl_color_map_default_params;

// Maps `vec4 color` from one color space to another color space according
// to the parameters (described in greater depth above). If `params` is left
// as NULL, it defaults to `&pl_color_map_default_params`. If `prelinearized`
// is true, the logic will assume the input has already been linearized by the
// caller (e.g. as part of a previous linear light scaling operation).
//
// `tone_mapping_state` is required if tone mapping is desired, and will be
// used to store state related to tone mapping. Note that this is the same
// state object used by the peak detection shader (`pl_shader_detect_peak`). If
// that function has been called on the same state object before this one, the
// detected values may be used to guide the tone mapping algorithm.
//
// Note: The peak detection state object is only updated after the shader is
// dispatched, so if `pl_shader_detect_peak` is called as part of the same
// shader as `pl_shader_color_map`, the results will end up delayed by one
// frame. If frame-level accuracy is desired, then users should call
// `pl_shader_detect_peak` separately and dispatch the resulting shader
// *before* dispatching this one.
void pl_shader_color_map(pl_shader sh,
                         const struct pl_color_map_params *params,
                         struct pl_color_space src, struct pl_color_space dst,
                         pl_shader_obj *tone_mapping_state,
                         bool prelinearized);

// Applies a set of cone distortion parameters to `vec4 color` in a given color
// space. This can be used to simulate color blindness. See `pl_cone_params`
// for more information.
void pl_shader_cone_distort(pl_shader sh, struct pl_color_space csp,
                            const struct pl_cone_params *params);

PL_API_END

#endif // LIBPLACEBO_SHADERS_COLORSPACE_H_
