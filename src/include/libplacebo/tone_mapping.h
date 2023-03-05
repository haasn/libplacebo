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

#ifndef LIBPLACEBO_TONE_MAPPING_H_
#define LIBPLACEBO_TONE_MAPPING_H_

#include <stddef.h>
#include <stdbool.h>

#include <libplacebo/common.h>
#include <libplacebo/colorspace.h>

PL_API_BEGIN

struct pl_tone_map_params;
struct pl_tone_map_function {
    const char *name;        // Identifier
    const char *description; // Friendly / longer name

    // If set, `pl_tone_map_params.param` can be adjusted to alter the
    // characteristics of the tone mapping function in some way. (Optional)
    const char *param_desc; // Name of parameter
    float param_min;
    float param_def;
    float param_max;

    // This controls the type of values input/output to/from `map`
    enum pl_hdr_scaling scaling;

    // The tone-mapping function itself. Iterates over all values in `lut`, and
    // adapts them as needed.
    //
    // Note that the `params` struct fed into this function is guaranteed to
    // satisfy `params->input_scaling == params->output_scaling == scaling`,
    // and also obeys `params->input_max >= params->output_max`.
    void (*map)(float *lut, const struct pl_tone_map_params *params);

    // Inverse tone mapping function. Optional. If absent, this tone mapping
    // curve only works in the forwards direction.
    //
    // For this function, `params->input_max <= params->output_max`.
    void (*map_inverse)(float *lut, const struct pl_tone_map_params *params);

    // Private data. Unused by libplacebo, but may be accessed by `map`.
    void *priv;
};

struct pl_tone_map_params {
    // If `function` is NULL, defaults to `pl_tone_map_clip`.
    const struct pl_tone_map_function *function;
    float param; // or 0.0 for default

    // The desired input/output scaling of the tone map. If this differs from
    // `function->scaling`, any required conversion will be performed.
    //
    // Note that to maximize LUT efficiency, it's *highly* recommended to use
    // either PL_HDR_PQ or PL_HDR_SQRT as the input scaling, except when
    // using `pl_tone_map_sample`.
    enum pl_hdr_scaling input_scaling;
    enum pl_hdr_scaling output_scaling;

    // The size of the resulting LUT. (For `pl_tone_map_generate` only)
    size_t lut_size;

    // The characteristics of the input, in `input_scaling` units.
    float input_min;
    float input_max;
    float input_avg; // or 0 if unknown

    // The desired characteristics of the output, in `output_scaling` units.
    float output_min;
    float output_max;

    // The input HDR metadata. Only used by a select few tone-mapping
    // functions, currently only SMPTE ST2094. (Optional)
    struct pl_hdr_metadata hdr;
};

#define pl_tone_map_params(...) (&(struct pl_tone_map_params) { __VA_ARGS__ });

// Note: Only does pointer equality testing on `function`
bool pl_tone_map_params_equal(const struct pl_tone_map_params *a,
                              const struct pl_tone_map_params *b);

// Resolves `pl_tone_map_auto` to a specific tone-mapping function, based on
// the tone mapping parameters, and also clamps/defaults the parameter.
void pl_tone_map_params_infer(struct pl_tone_map_params *params);

// Returns true if the given tone mapping configuration effectively represents
// a no-op configuration. Tone mapping can be skipped in this case (although
// strictly speaking, the LUT would still clip illegal input values)
bool pl_tone_map_params_noop(const struct pl_tone_map_params *params);

// Generate a tone-mapping LUT for a given configuration. This will always
// span the entire input range, as given by `input_min` and `input_max`.
void pl_tone_map_generate(float *out, const struct pl_tone_map_params *params);

// Samples a tone mapping function at a single position. Note that this is less
// efficient than `pl_tone_map_generate` for generating multiple values.
//
// Ignores `params->lut_size`.
float pl_tone_map_sample(float x, const struct pl_tone_map_params *params);

// Special tone mapping function that means "automatically pick a good function
// based on the HDR levels". This is an opaque tone map function with no
// meaningful internal representation. (Besides `name` and `description`)
extern const struct pl_tone_map_function pl_tone_map_auto;

// Performs no tone-mapping, just clips out-of-range colors. Retains perfect
// color accuracy for in-range colors but completely destroys out-of-range
// information. Does not perform any black point adaptation.
extern const struct pl_tone_map_function pl_tone_map_clip;

// EETF from SMPTE ST 2094-40 Annex B, which uses the provided OOTF based on
// Bezier curves to perform tone-mapping. The OOTF used is adjusted based on
// the ratio between the targeted and actual display peak luminances.
//
// In the absence of HDR10+ metadata, falls back to a simple constant bezier
// curve with tunable knee point. The parameter gives the target brightness
// adaptation strength for the knee point, defaulting to 0.7.
extern const struct pl_tone_map_function pl_tone_map_st2094_40;

// EETF from SMPTE ST 2094-10 Annex B.2, which takes into account the input
// signal average luminance in addition to the maximum/minimum. The parameter
// gives the target brightness adaptation strength for the knee point,
// defaulting to 0.5.
//
// Note: This does *not* currently include the subjective gain/offset/gamma
// controls defined in Annex B.3. (Open an issue with a valid sample file if
// you want such parameters to be respected.)
extern const struct pl_tone_map_function pl_tone_map_st2094_10;

// EETF from the ITU-R Report BT.2390, a hermite spline roll-off with linear
// segment. The knee point offset is configurable. Note that this defaults to
// 1.0, rather than the value of 0.5 from the ITU-R spec.
extern const struct pl_tone_map_function pl_tone_map_bt2390;

// EETF from ITU-R Report BT.2446, method A. Can be used for both forward
// and inverse tone mapping. Not configurable.
extern const struct pl_tone_map_function pl_tone_map_bt2446a;

// Simple spline consisting of two polynomials, joined by a single pivot point,
// which is tuned based on the source scene average brightness (taking into
// account HDR10+ metadata if available). The parameter can be used to tune the
// desired subjective contrast characteristics. Higher values make the curve
// steeper (closer to `clip`), preserving midtones at the cost of losing
// shadow/highlight details, while lower values make the curve shallower
// (closer to `linear`), preserving highlights at the cost of losing midtone
// contrast. Values above 1.0 are possible, resulting in an output with more
// contrast than the input. The default value is 0.5. This function can be used
// for both forward and inverse tone mapping.
extern const struct pl_tone_map_function pl_tone_map_spline;

// Simple non-linear, global tone mapping algorithm. Named after Erik Reinhard.
// The parameter specifies the local contrast coefficient at the display peak.
// Essentially, a value of param=0.5 implies that the reference white will be
// about half as bright as when clipping. Defaults to 0.5, which results in the
// simplest formulation of this function.
extern const struct pl_tone_map_function pl_tone_map_reinhard;

// Generalization of the reinhard tone mapping algorithm to support an
// additional linear slope near black. The tone mapping parameter indicates the
// trade-off between the linear section and the non-linear section.
// Essentially, for param=0.5, every color value below 0.5 will be mapped
// linearly, with the higher values being non-linearly tone mapped. Values near
// 1.0 make this curve behave like pl_tone_map_clip, and values near 0.0 make
// this curve behave like pl_tone_map_reinhard. The default value is 0.3, which
// provides a good balance between colorimetric accuracy and preserving
// out-of-gamut details. The name is derived from its function shape
// (ax+b)/(cx+d), which is known as a Möbius transformation in mathematics.
extern const struct pl_tone_map_function pl_tone_map_mobius;

// Piece-wise, filmic tone-mapping algorithm developed by John Hable for use in
// Uncharted 2, inspired by a similar tone-mapping algorithm used by Kodak.
// Popularized by its use in video games with HDR rendering. Preserves both
// dark and bright details very well, but comes with the drawback of changing
// the average brightness quite significantly. This is sort of similar to
// pl_tone_map_reinhard with parameter 0.24.
extern const struct pl_tone_map_function pl_tone_map_hable;

// Fits a gamma (power) function to transfer between the source and target
// color spaces, effectively resulting in a perceptual hard-knee joining two
// roughly linear sections. This preserves details at all scales fairly
// accurately, but can result in an image with a muted or dull appearance. The
// parameter is used as the cutoff point, defaulting to 0.5.
extern const struct pl_tone_map_function pl_tone_map_gamma;

// Linearly stretches the input range to the output range, in PQ space. This
// will preserve all details accurately, but results in a significantly
// different average brightness. Can be used for inverse tone-mapping in
// addition to regular tone-mapping. The parameter can be used as an additional
// linear gain coefficient (defaulting to 1.0).
extern const struct pl_tone_map_function pl_tone_map_linear;

// A list of built-in tone mapping functions, terminated by NULL
extern const struct pl_tone_map_function * const pl_tone_map_functions[];
extern const int pl_num_tone_map_functions; // excluding trailing NULL

// Find the tone mapping function with the given name, or NULL on failure.
const struct pl_tone_map_function *pl_find_tone_map_function(const char *name);

PL_API_END

#endif // LIBPLACEBO_TONE_MAPPING_H_
