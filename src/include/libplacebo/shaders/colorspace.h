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
#include <libplacebo/gamut_mapping.h>
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
//
// Note: For DCDM XYZ decoding output is linear
void pl_shader_decode_color(pl_shader sh, struct pl_color_repr *repr,
                            const struct pl_color_adjustment *params);

// Encodes a color from normalized, PC-range, independent alpha RGB into a
// given representation. That is, this performs the inverse operation of
// `pl_shader_decode_color` (sans color adjustments).
//
// Note: For DCDM XYZ encoding input is expected to be linear
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
    // lower and upper bounds of this window, in units of 1% PQ.
    //
    // The default values are 5.5 and 10.0, respectively. To disable this logic
    // entirely, set either one to a negative value.
    float scene_threshold_low;
    float scene_threshold_high;

    // To avoid over-tone-mapping very dark scenes (or black frames), this
    // imposes a hard lower bound on the detected peak. If left as 0.0, it
    // instead defaults to a value of 1.0.
    float minimum_peak;

    // Which percentile of the input image brightness histogram to consider as
    // the true peak of the scene. If this is set to 100 (or 0), the brightest
    // pixel is measured. Otherwise, the top of the frequency distribution is
    // progressively cut off. Setting this too low will cause clipping of very
    // bright details, but can improve the dynamic brightness range of scenes
    // with very bright isolated highlights.
    //
    // The default of 99.995% is very conservative and should cause no major
    // issues in typical content.
    float percentile;

    // Allows the peak detection result to be delayed by up to a single frame,
    // which can sometimes improve thoughpout. Disabled by default.
    bool allow_delayed;

    // --- Deprecated fields
    float overshoot_margin PL_DEPRECATED;
};

#define PL_PEAK_DETECT_DEFAULTS         \
    .smoothing_period       = 100.0f,   \
    .scene_threshold_low    = 5.5f,     \
    .scene_threshold_high   = 10.0f,    \
    .minimum_peak           = 1.0f,     \
    .percentile             = 99.995f,

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

// After dispatching the above shader, this function can be used to retrieve
// the detected dynamic HDR10+ metadata parameters. The other fields of
// `metadata` are not written to. Returns whether or not any values were
// written. If not, the values are left untouched, so this can be used to
// safely update `pl_hdr_metadata` values in-place. This function may or may
// not block, depending on the previous setting of `allow_delayed`.
bool pl_get_detected_hdr_metadata(const pl_shader_obj state,
                                  struct pl_hdr_metadata *metadata);

// After dispatching the above shader, this function *may* be used to read out
// the detected CLL and FALL directly (in PL_HDR_NORM units). If the shader
// has never been dispatched yet, i.e. no information is available, this will
// return false.
//
// Deprecated in favor of `pl_get_detected_hdr_metadata`
PL_DEPRECATED bool pl_get_detected_peak(const pl_shader_obj state,
                                        float *out_cll, float *out_fall);

// Resets the peak detection state in a given tone mapping state object. This
// is not equal to `pl_shader_obj_destroy`, because it does not destroy any
// state used by `pl_shader_tone_map`.
void pl_reset_detected_peak(pl_shader_obj state);

// Deprecated and unused. Libplacebo now always performs a variant of the old
// hybrid tone-mapping, mixing together the intensity (I) and per-channel (LMS)
// results.
enum pl_tone_map_mode {
    PL_TONE_MAP_AUTO    PL_DEPRECATED_ENUMERATOR,
    PL_TONE_MAP_RGB     PL_DEPRECATED_ENUMERATOR,
    PL_TONE_MAP_MAX     PL_DEPRECATED_ENUMERATOR,
    PL_TONE_MAP_HYBRID  PL_DEPRECATED_ENUMERATOR,
    PL_TONE_MAP_LUMA    PL_DEPRECATED_ENUMERATOR,
    PL_TONE_MAP_MODE_COUNT,
};

// Deprecated by <libplacebo/gamut_mapping.h>
enum pl_gamut_mode {
    PL_GAMUT_CLIP       PL_DEPRECATED_ENUMERATOR, // pl_gamut_map_clip
    PL_GAMUT_WARN       PL_DEPRECATED_ENUMERATOR, // pl_gamut_map_highlight
    PL_GAMUT_DARKEN     PL_DEPRECATED_ENUMERATOR, // pl_gamut_map_darken
    PL_GAMUT_DESATURATE PL_DEPRECATED_ENUMERATOR, // pl_gamut_map_desaturate
    PL_GAMUT_MODE_COUNT,
};

struct pl_color_map_params {
    // --- Gamut mapping options

    // Gamut mapping function to use to handle out-of-gamut colors, including
    // colors which are out-of-gamut as a consequence of tone mapping.
    const struct pl_gamut_map_function *gamut_mapping;

    // Gamut mapping 3DLUT size, for channels ICh. Defaults to {33, 25, 45}
    int lut3d_size[3];

    // --- Tone mapping options

    // Function and configuration used for tone-mapping. For non-tunable
    // functions, the `param` is ignored. If the tone mapping parameter is
    // left as 0.0, the tone-mapping curve's preferred default parameter will
    // be used. The default function is pl_tone_map_auto.
    //
    // Note: This pointer changing invalidates the LUT, so make sure to only
    // use stable (or static) storage for the pl_tone_map_function.
    const struct pl_tone_map_function *tone_mapping_function;
    float tone_mapping_param;

    // If true, and supported by the given tone mapping function, libplacebo
    // will perform inverse tone mapping to expand the dynamic range of a
    // signal. libplacebo is not liable for any HDR-induced eye damage.
    bool inverse_tone_mapping;

    // If nonzero, this much of the upper range of the tone-mapped result is
    // smoothly mixed with a per-channel (LMS) tone-mapped version. Helps avoid
    // unnatural blown-out highlights when tone-mapping very bright, strongly
    // saturated colors.
    float hybrid_mix;

    // Data source to use when tone-mapping. Setting this to a specific
    // value allows overriding the default metadata preference logic.
    enum pl_hdr_metadata_type metadata;

    // Tone mapping LUT size. Defaults to 256.
    int lut_size;

    // --- Debugging options

    // Force the use of a full tone-mapping LUT even for functions that have
    // faster pure GLSL replacements (e.g. clip).
    bool force_tone_mapping_lut;

    // Visualize the tone-mapping LUT and gamut mapping 3DLUT, in IPT space.
    bool visualize_lut;

    // Controls where to draw the visualization, relative to the rendered
    // video (dimensions 0-1). Optional, defaults to the full picture.
    pl_rect2df visualize_rect;

    // Controls the rotation of the 3DLUT visualization.
    float visualize_hue;    // useful range [-pi, pi]
    float visualize_theta;  // useful range [0, pi/2]

    // Graphically highlight hard-clipped pixels during tone-mapping (i.e.
    // pixels that exceed the claimed source luminance range).
    bool show_clipping;

    // --- Deprecated fields
    enum pl_tone_map_mode tone_mapping_mode PL_DEPRECATED; // see `hybrid_mix`
    float tone_mapping_crosstalk PL_DEPRECATED;     // now hard-coded as 0.04
    enum pl_rendering_intent intent PL_DEPRECATED;  // see `gamut_mapping`
    enum pl_gamut_mode gamut_mode PL_DEPRECATED;    // see `gamut_mapping`
};

#define PL_COLOR_MAP_DEFAULTS                                   \
    .gamut_mapping          = &pl_gamut_map_perceptual,         \
    .tone_mapping_function  = &pl_tone_map_auto,                \
    .hybrid_mix             = 0.20f,                            \
    .metadata               = PL_HDR_METADATA_ANY,              \
    .lut3d_size             = {33, 25, 45},                     \
    .lut_size               = 256,

#define pl_color_map_params(...) (&(struct pl_color_map_params) { PL_COLOR_MAP_DEFAULTS __VA_ARGS__ })
extern const struct pl_color_map_params pl_color_map_default_params;

// Maps `vec4 color` from one color space to another color space according
// to the parameters (described in greater depth above). If `params` is left
// as NULL, it defaults to `&pl_color_map_default_params`. If `prelinearized`
// is true, the logic will assume the input has already been linearized by the
// caller (e.g. as part of a previous linear light scaling operation).
//
// `state` will be used to store generated LUTs. Note that this is the same
// state object used by `pl_shader_detect_peak`, and if that function has been
// called on `state` prior to `pl_shader_color_map`, the detected values will
// be used to guide the tone mapping algorithm. If this is not provided,
// tone/gamut mapping are disabled.
void pl_shader_color_map(pl_shader sh,
                         const struct pl_color_map_params *params,
                         struct pl_color_space src, struct pl_color_space dst,
                         pl_shader_obj *state, bool prelinearized);

// Applies a set of cone distortion parameters to `vec4 color` in a given color
// space. This can be used to simulate color blindness. See `pl_cone_params`
// for more information.
void pl_shader_cone_distort(pl_shader sh, struct pl_color_space csp,
                            const struct pl_cone_params *params);

PL_API_END

#endif // LIBPLACEBO_SHADERS_COLORSPACE_H_
