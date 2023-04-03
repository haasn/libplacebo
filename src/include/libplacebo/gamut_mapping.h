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

#ifndef LIBPLACEBO_GAMUT_MAPPING_H_
#define LIBPLACEBO_GAMUT_MAPPING_H_

#include <libplacebo/common.h>
#include <libplacebo/colorspace.h>

PL_API_BEGIN

struct pl_gamut_map_params;
struct pl_gamut_map_function {
    const char *name;        // Identifier
    const char *description; // Friendly / longer name

    // The gamut-mapping function itself. Iterates over all values in `lut`,
    // and adapts them as needed.
    void (*map)(float *lut, const struct pl_gamut_map_params *params);

    // Returns true if `map` supports both stretching and contracting the
    // gamut. In this case, `map` is always executed, even if the output gamut
    // is larger than the input gamut.
    bool bidirectional;

    // Private data. Unused by libplacebo, but may be accessed by `map`.
    void *priv;
};

struct pl_gamut_map_params {
    // If `function` is NULL, defaults to `pl_gamut_map_clip`.
    const struct pl_gamut_map_function *function;

    // The desired input/output primaries. This affects the subjective color
    // volume in which the desired mapping shall take place.
    struct pl_raw_primaries input_gamut;
    struct pl_raw_primaries output_gamut;

    // Minimum/maximum luminance (PQ) of the target display. Note that the same
    // value applies to both the input and output, since it's assumed that tone
    // mapping has already happened by this stage. This effectively defines the
    // legal gamut boundary in RGB space.
    //
    // This also defines the I channel value range, for `pl_gamut_map_generate`
    float min_luma;
    float max_luma;

    // Extra margin of maximum input chrominance, to add headroom when reducing
    // source peak intensity (tone mapping).
    float chroma_margin;

    // -- LUT generation options (for `pl_gamut_map_generate` only)

    // The size of the resulting LUT, per channel.
    //
    // Note: For quality, it's generally best to increase h > I > C
    int lut_size_I;
    int lut_size_C;
    int lut_size_h;

    // The stride (in number of floats) between elements in the resulting LUT.
    int lut_stride;
};

#define pl_gamut_map_params(...) (&(struct pl_gamut_map_params) { __VA_ARGS__ })

// Note: Only does pointer equality testing on `function`
bool pl_gamut_map_params_equal(const struct pl_gamut_map_params *a,
                               const struct pl_gamut_map_params *b);

// Returns true if the given gamut mapping configuration effectively represents
// a no-op configuration. Gamut mapping can be skipped in this case.
bool pl_gamut_map_params_noop(const struct pl_gamut_map_params *params);

// Generate a gamut-mapping LUT for a given configuration. LUT samples are
// stored as IPTPQc4 values, but the LUT itself is indexed by IChPQc4,spanning
// the effective range [min_luma, max_luma] × [0, 0.5] × [-pi,pi].
//
// This ordering is designed to keep frequently co-occurring values close in
// memory, while permitting simple wrapping of the 'h' component.
void pl_gamut_map_generate(float *out, const struct pl_gamut_map_params *params);

// Samples a gamut mapping function for a single IPTPQc4 value. The input
// values are updated in-place.
void pl_gamut_map_sample(float x[3], const struct pl_gamut_map_params *params);

// Performs no gamut-mapping, just hard clips out-of-range colors per-channel.
extern const struct pl_gamut_map_function pl_gamut_map_clip;

// Performs a perceptually balanced, colorimetric gamut mapping using a soft
// knee function to roll-off clipped regions.
extern const struct pl_gamut_map_function pl_gamut_map_perceptual;

// Performs relative colorimetric clipping, while maintaining an exponential
// relationship between brightness and chromaticity.
extern const struct pl_gamut_map_function pl_gamut_map_relative;

// Performs simple RGB->RGB saturation mapping. The input R/G/B channels are
// mapped directly onto the output R/G/B channels. Will never clip, but will
// distort all hues and/or result in a faded look.
extern const struct pl_gamut_map_function pl_gamut_map_saturation;

// Performs absolute colorimetric clipping. Like pl_gamut_map_relative, but
// does not adapt the white point.
extern const struct pl_gamut_map_function pl_gamut_map_absolute;

// Performs constant-luminance colorimetric clipping, desaturing colors
// towards white until they're in-range.
extern const struct pl_gamut_map_function pl_gamut_map_desaturate;

// Uniformly darkens the input slightly to prevent clipping on blown-out
// highlights, then clamps colorimetrically to the input gamut boundary,
// biased slightly to preserve chromaticity over luminance.
extern const struct pl_gamut_map_function pl_gamut_map_darken;

// Performs no gamut mapping, but simply highlights out-of-gamut pixels.
extern const struct pl_gamut_map_function pl_gamut_map_highlight;

// Linearly/uniformly desaturates the image in order to bring the entire
// image into the target gamut.
extern const struct pl_gamut_map_function pl_gamut_map_linear;

// A list of built-in gamut mapping functions, terminated by NULL
extern const struct pl_gamut_map_function * const pl_gamut_map_functions[];
extern const int pl_num_gamut_map_functions; // excluding trailing NULL
                                             //
// Find the gamut mapping function with the given name, or NULL on failure.
const struct pl_gamut_map_function *pl_find_gamut_map_function(const char *name);

PL_API_END

#endif // LIBPLACEBO_GAMUT_MAPPING_H_
