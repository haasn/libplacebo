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

#ifndef LIBPLACEBO_FILTER_KERNELS_H_
#define LIBPLACEBO_FILTER_KERNELS_H_

#include <stdbool.h>
#include <libplacebo/log.h>

PL_API_BEGIN

#define PL_FILTER_MAX_PARAMS 2

// Represents a single filter function, i.e. kernel or windowing function.
// To invoke a filter with a different configuration than the default, you can
// make a copy of this struct and modify the non-const fields before passing it
// to pl_filter_initialize.
struct pl_filter_function {
    // These bools indicate whether or not `radius` and `params` may be
    // modified by the user.
    bool resizable;
    bool tunable[PL_FILTER_MAX_PARAMS];

    // The underlying filter function itself: Computes the weight as a function
    // of the offset. All filter functions must be normalized such that x=0 is
    // the center point, and in particular weight(0) = 1.0. The functions may
    // be undefined for values of x outside [0, radius].
    double (*weight)(const struct pl_filter_function *k, double x);

    // This field may be used to adjust the function's radius. Defaults to the
    // the radius needed to represent a single filter lobe (tap). If the
    // function is not resizable, this field must not be modified - otherwise
    // the result of filter evaluation is undefined.
    float radius;

    // These fields may be used to adjust the function. Defaults to the
    // function's preferred defaults. if the relevant setting is not tunable,
    // they are ignored entirely.
    float params[PL_FILTER_MAX_PARAMS];

    // The cosmetic name associated with this filter function. Optional.
    const char *name;
};

bool pl_filter_function_eq(const struct pl_filter_function *a,
                           const struct pl_filter_function *b);

// Box filter: Entirely 1.0 within the radius, entirely 0.0 outside of it.
// This is also sometimes called a Dirichlet window
extern const struct pl_filter_function pl_filter_function_box;

// Triangle filter: Linear transitions from 1.0 at x=0 to 0.0 at x=radius.
// This is also sometimes called a Bartlett window.
extern const struct pl_filter_function pl_filter_function_triangle;

// Cosine filter: Ordinary cosine function, single lobe.
extern const struct pl_filter_function pl_filter_function_cosine;

// Hann function: Cosine filter named after Julius von Hann. Also commonly
// mislabeled as a "Hanning" function, due to its similarly to the Hamming
// function.
extern const struct pl_filter_function pl_filter_function_hann;

// Hamming function: Cosine filter named after Richard Hamming.
extern const struct pl_filter_function pl_filter_function_hamming;

// Welch filter: Polynomial function consisting of a single parabolic section.
extern const struct pl_filter_function pl_filter_function_welch;

// Kaiser filter: Approximation of the DPSS window using Bessel functions.
// Also sometimes called a Kaiser-Bessel window.
// Parameter [0]: Shape (alpha). Determines the trade-off between the main lobe
//                and the side lobes.
extern const struct pl_filter_function pl_filter_function_kaiser;

// Blackman filter: Cosine filter named after Ralph Beebe Blackman.
// Parameter [0]: Scale (alpha). Influences the shape. The defaults result in
//                zeros at the third and fourth sidelobes.
extern const struct pl_filter_function pl_filter_function_blackman;

// Bohman filter: 2nd order Cosine filter.
extern const struct pl_filter_function pl_filter_function_bohman;

// Gaussian function: Similar to the Gaussian distribution, this defines a
// bell curve function.
// Parameter [0]: Scale (t), increasing makes the result blurrier.
extern const struct pl_filter_function pl_filter_function_gaussian;

// Quadratic function: 2nd order approximation of the gaussian function. Also
// sometimes called a "quadric" window.
extern const struct pl_filter_function pl_filter_function_quadratic;

// Sinc function: Widely used for both kernels and windows, sinc(x) = sin(x)/x.
extern const struct pl_filter_function pl_filter_function_sinc;

// Jinc function: Similar to sinc, but extended to the 2D domain. Widely
// used as the kernel of polar (EWA) filters. Also sometimes called a Sombrero
// function.
extern const struct pl_filter_function pl_filter_function_jinc;

// Sphinx function: Similar to sinc and jinx, but extended to the 3D domain.
// The name is derived from "spherical" sinc. Can be used to filter 3D signals
// in theory.
extern const struct pl_filter_function pl_filter_function_sphinx;

// B/C-tunable Spline function: This is a family of commonly used spline
// functions with two tunable parameters. Does not need to be windowed.
// Parameter [0]: "B"
// Parameter [1]: "C"
// Due to its populariy, this function is available in several variants.
// B = 0.0,  C = 0.0:  "base" bcspline, AKA Hermite spline (blocky)
// B = 0.0,  C = 0.5:  Catmull-Rom filter (sharp)
// B = 1/3,  C = 1/3:  Mitchell-Netravali filter (soft, doesn't ring)
// B ≈ 0.37, C ≈ 0.31: Robidoux filter (used by ImageMagick)
// B ≈ 0.26, C ≈ 0.37: RobidouxSharp filter. (sharper variant of Robidoux)
extern const struct pl_filter_function pl_filter_function_bcspline;
extern const struct pl_filter_function pl_filter_function_catmull_rom;
extern const struct pl_filter_function pl_filter_function_mitchell;
extern const struct pl_filter_function pl_filter_function_robidoux;
extern const struct pl_filter_function pl_filter_function_robidouxsharp;

// Bicubic function: Very smooth and free of ringing, but very blurry. Does not
// need to be windowed.
extern const struct pl_filter_function pl_filter_function_bicubic;

// Piecewise approximations of the Lanczos filter function (sinc-windowed
// sinc). Referred to as "spline16", "spline36" and "spline64" mainly for
// historical reasons, based on their fixed radii of 2, 3 and 4 (respectively).
// These do not need to be windowed.
extern const struct pl_filter_function pl_filter_function_spline16;
extern const struct pl_filter_function pl_filter_function_spline36;
extern const struct pl_filter_function pl_filter_function_spline64;

struct pl_filter_function_preset {
    const char *name;
    const struct pl_filter_function *function;
};

// A list of built-in filter function presets, terminated by {0}
extern const struct pl_filter_function_preset pl_filter_function_presets[];
extern const int pl_num_filter_function_presets; // excluding trailing {0}

// Find the filter function preset with the given name, or NULL on failure.
const struct pl_filter_function_preset *pl_find_filter_function_preset(const char *name);

// Backwards compatibility
#define pl_named_filter_function        pl_filter_function_preset
#define pl_named_filter_functions       pl_filter_function_presets
#define pl_find_named_filter_function   pl_find_filter_function_preset

// Represents a particular configuration/combination of filter functions to
// form a filter.
struct pl_filter_config {
    const struct pl_filter_function *kernel; // The kernel function
    const struct pl_filter_function *window; // The windowing function. Optional

    // Represents a clamping coefficient for negative weights. A value of 0.0
    // (the default) represents no clamping. A value of 1.0 represents full
    // clamping, i.e. all negative weights will be clamped to 0. Values in
    // between will be linearly scaled.
    float clamp;

    // Additional blur coefficient. This effectively stretches the kernel,
    // without changing the effective radius of the filter radius. Setting this
    // to a value of 0.0 is equivalent to disabling it. Values significantly
    // below 1.0 may seriously degrade the visual output, and should be used
    // with care.
    float blur;

    // Additional taper coefficient. This essentially flattens the function's
    // center. The values within [-taper, taper] will return 1.0, with the
    // actual function being squished into the remainder of [taper, radius].
    // Defaults to 0.0.
    float taper;

    // If true, this filter is intended to be used as a polar/2D filter (EWA)
    // instead of a separable/1D filter. Does not affect the actual sampling,
    // but provides information about how the results are to be interpreted.
    bool polar;

    // The cosmetic name associated with this filter config. Optional.
    const char *name;
};

bool pl_filter_config_eq(const struct pl_filter_config *a,
                         const struct pl_filter_config *b);

// Samples a given filter configuration at a given x coordinate, while
// respecting all parameters of the configuration.
double pl_filter_sample(const struct pl_filter_config *c, double x);

// A list of built-in filter configurations. Since they are just combinations
// of the above filter functions, they are not described in much further
// detail.
extern const struct pl_filter_config pl_filter_spline16;    // 2 taps
extern const struct pl_filter_config pl_filter_spline36;    // 3 taps
extern const struct pl_filter_config pl_filter_spline64;    // 4 taps
extern const struct pl_filter_config pl_filter_nearest;     // AKA box
extern const struct pl_filter_config pl_filter_bilinear;    // AKA triangle
extern const struct pl_filter_config pl_filter_gaussian;
// Sinc family (all configured to 3 taps):
extern const struct pl_filter_config pl_filter_sinc;        // unwindowed,
extern const struct pl_filter_config pl_filter_lanczos;     // sinc-sinc
extern const struct pl_filter_config pl_filter_ginseng;     // sinc-jinc
extern const struct pl_filter_config pl_filter_ewa_jinc;    // unwindowed
extern const struct pl_filter_config pl_filter_ewa_lanczos; // jinc-jinc
extern const struct pl_filter_config pl_filter_ewa_ginseng; // jinc-sinc
extern const struct pl_filter_config pl_filter_ewa_hann;    // jinc-hann
// Spline family
extern const struct pl_filter_config pl_filter_bicubic;
extern const struct pl_filter_config pl_filter_catmull_rom;
extern const struct pl_filter_config pl_filter_mitchell;
extern const struct pl_filter_config pl_filter_mitchell_clamp; // clamp = 1.0
extern const struct pl_filter_config pl_filter_robidoux;
extern const struct pl_filter_config pl_filter_robidouxsharp;
extern const struct pl_filter_config pl_filter_ewa_robidoux;
extern const struct pl_filter_config pl_filter_ewa_robidouxsharp;

// Backwards compatibility
#define pl_filter_box       pl_filter_nearest
#define pl_filter_triangle  pl_filter_bilinear

struct pl_filter_preset {
    const char *name;
    const struct pl_filter_config *filter;

    // Longer / friendly name, or NULL for aliases
    const char *description;
};

// A list of built-in filter presets, terminated by {0}
extern const struct pl_filter_preset pl_filter_presets[];
extern const int pl_num_filter_presets; // excluding trailing {0}

// Find the filter preset with the given name, or NULL on failure.
const struct pl_filter_preset *pl_find_filter_preset(const char *name);

// Backwards compatibility
#define pl_named_filter_config  pl_filter_preset
#define pl_named_filters        pl_filter_presets
#define pl_find_named_filter    pl_find_filter_preset

// Parameters for filter generation.
struct pl_filter_params {
    // The particular filter configuration to be sampled. config.kernel must
    // be set to a valid pl_filter_function.
    struct pl_filter_config config;

    // The precision of the resulting LUT. A value of 64 should be fine for
    // most practical purposes, but higher or lower values may be justified
    // depending on the use case. This value must be set to something > 0.
    int lut_entries;

    // When set to values above 1.0, the filter will be computed at a size
    // larger than the radius would otherwise require, in order to prevent
    // aliasing when downscaling. In practice, this should be set to the
    // inverse of the scaling ratio, i.e. src_size / dst_size.
    float filter_scale;

    // --- polar filers only (config.polar)

    // As a micro-optimization, all samples below this cutoff value will be
    // ignored when updating the cutoff radius. Setting it to a value of 0.0
    // disables this optimization.
    float cutoff;

    // --- separable filters only (!config.polar)

    // Indicates the maximum row size that is supported by the calling code, or
    // 0 for no limit.
    int max_row_size;

    // Indicates the row stride alignment. For some use cases (e.g. uploading
    // the weights as a texture), there are certain alignment requirements for
    // each row. The chosen row_size will always be a multiple of this value.
    // Specifying 0 indicates no alignment requirements.
    int row_stride_align;
};

#define pl_filter_params(...) (&(struct pl_filter_params) { __VA_ARGS__ })

// Represents an initialized instance of a particular filter, with a
// precomputed LUT. The interpretation of the LUT depends on the type of the
// filter (polar or separable).
typedef const struct pl_filter_t {
    // Deep copy of the parameters, for convenience.
    struct pl_filter_params params;

    // Contains the true radius of the computed filter. This may be
    // larger than `config.kernel->radius` depending on the `scale` passed to
    // pl_filter_generate. This is only relevant for polar filters, where it
    // affects the value range of *weights.
    float radius;

    // The computed look-up table (LUT). For polar filters, this is interpreted
    // as a 1D array with dimensions [lut_entries] containing the raw filter
    // samples on the scale [0, radius]. For separable (non-polar) filters,
    // this is interpreted as a 2D array with dimensions
    // [lut_entries][row_stride]. The inner rows contain the `row_size` samples
    // to convolve with the corresponding input pixels. The outer coordinate is
    // used to very the fractional offset (phase). So for example, if the
    // sample position to reconstruct is directly aligned with the source
    // texels, you would use the values from weights[0]. If the sample position
    // to reconstruct is exactly half-way between two source texels (180° out
    // of phase), you would use the values from weights[lut_entries/2].
    const float *weights;

    // --- polar filters only (params.config.polar)

    // Contains the effective cut-off radius for this filter. Samples outside
    // of this cutoff radius may be discarded. Computed based on the `cutoff`
    // value specified at filter generation. Only relevant for polar filters
    // since skipping samples outside of the radius can be a significant
    // performance gain for EWA sampling.
    float radius_cutoff;

    // --- separable filters only (!params.config.polar)

    // The number of source texels to convolve over for each row. This value
    // will never exceed the given `max_row_size`. If the filter ends up
    // cut off because of this, the bool `insufficient` will be set to true.
    int row_size;
    bool insufficient;

    // The separation (in *weights) between each row of the filter. Always
    // a multiple of params.row_stride_align.
    int row_stride;
} *pl_filter;

// Generate (compute) a filter instance based on a given filter configuration.
// The resulting pl_filter must be freed with `pl_filter_free` when no longer
// needed. Returns NULL if filter generation fails due to invalid parameters
// (i.e. missing a required parameter).
pl_filter pl_filter_generate(pl_log log, const struct pl_filter_params *params);
void pl_filter_free(pl_filter *filter);

PL_API_END

#endif // LIBPLACEBO_FILTER_KERNELS_H_
