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

#ifndef LIBPLACEBO_SHADERS_SAMPLING_H_
#define LIBPLACEBO_SHADERS_SAMPLING_H_

// Sampling operations. These shaders perform some form of sampling operation
// from a given pl_tex. In order to use these, the pl_shader *must* have been
// created using the same `gpu` as the originating `pl_tex`. Otherwise, this
// is undefined behavior. They require nothing (PL_SHADER_SIG_NONE) and return
// a color (PL_SHADER_SIG_COLOR).

#include <libplacebo/filters.h>
#include <libplacebo/shaders.h>

PL_API_BEGIN

// Common parameters for sampling operations
struct pl_sample_src {
    // There are two mutually exclusive ways of providing the source to sample
    // from:
    //
    // 1. Provide the texture and sampled region directly. This generates
    // a shader with input signature `PL_SHADER_SIG_NONE`, which binds the
    // texture as a descriptor (and the coordinates as a vertex attribute)
    pl_tex tex;             // texture to sample
    pl_rect2df rect;        // sub-rect to sample from (optional)
    enum pl_tex_address_mode address_mode; // preferred texture address mode

    // 2. Have the shader take it as an argument. Doing this requires
    // specifying the missing metadata of the texture backing the sampler, so
    // that the shader generation can generate the correct code.
    int tex_w, tex_h;             // dimensions of the actual texture
    enum pl_fmt_type format;      // format of the sampler being accepted
    enum pl_sampler_type sampler; // type of the sampler being accepted
    enum pl_tex_sample_mode mode; // sample mode of the sampler being accepted
    float sampled_w, sampled_h;   // dimensions of the sampled region (optional)

    // Common metadata for both sampler input types:
    int components;   // number of components to sample (optional)
    uint8_t component_mask; // bitmask of components to sample (optional)
    int new_w, new_h; // dimensions of the resulting output (optional)
    float scale;      // factor to multiply into sampled signal (optional)

    // Note: `component_mask` and `components` are mutually exclusive, the
    // former is preferred if both are specified.
};

#define pl_sample_src(...) (&(struct pl_sample_src) { __VA_ARGS__ })

struct pl_deband_params {
    // The number of debanding steps to perform per sample. Each step reduces a
    // bit more banding, but takes time to compute. Note that the strength of
    // each step falls off very quickly, so high numbers (>4) are practically
    // useless. Defaults to 1.
    int iterations;

    // The debanding filter's cut-off threshold. Higher numbers increase the
    // debanding strength dramatically, but progressively diminish image
    // details. Defaults to 4.0.
    float threshold;

    // The debanding filter's initial radius. The radius increases linearly
    // for each iteration. A higher radius will find more gradients, but a
    // lower radius will smooth more aggressively. Defaults to 16.0.
    float radius;

    // Add some extra noise to the image. This significantly helps cover up
    // remaining quantization artifacts. Higher numbers add more noise.
    // Note: When debanding HDR sources, even a small amount of grain can
    // result in a very big change to the brightness level. It's recommended to
    // either scale this value down or disable it entirely for HDR.
    //
    // Defaults to 6.0, which is very mild.
    float grain;

    // 'Neutral' grain value for each channel being debanded (sorted in order
    // from low to high index). Grain application will be modulated to avoid
    // disturbing colors close to this value. Set this to a value corresponding
    // to black in the relevant colorspace.
    float grain_neutral[3];
};

#define PL_DEBAND_DEFAULTS  \
    .iterations = 1,        \
    .threshold  = 4.0,      \
    .radius     = 16.0,     \
    .grain      = 6.0,

#define pl_deband_params(...) (&(struct pl_deband_params) {PL_DEBAND_DEFAULTS __VA_ARGS__ })
extern const struct pl_deband_params pl_deband_default_params;

// Debands a given texture and returns the sampled color in `vec4 color`. If
// `params` is left as NULL, it defaults to &pl_deband_default_params. Note
// that `tex->params.format` must have PL_FMT_CAP_LINEAR. When the given
// `pl_sample_src` implies scaling, this effectively performs bilinear
// sampling on the input (but not the output).
//
// Note: This can also be used as a pure grain function, by setting the number
// of iterations to 0.
void pl_shader_deband(pl_shader sh, const struct pl_sample_src *src,
                      const struct pl_deband_params *params);

// Performs direct / native texture sampling, using whatever texture filter is
// available (linear for linearly sampleable sources, nearest otherwise).
//
// Note: This is generally very low quality and should be avoided if possible,
// for both upscaling and downscaling.
bool pl_shader_sample_direct(pl_shader sh, const struct pl_sample_src *src);

// Performs hardware-accelerated nearest neighbour sampling. This is similar to
// `pl_shader_sample_direct`, but forces nearest neighbour interpolation.
bool pl_shader_sample_nearest(pl_shader sh, const struct pl_sample_src *src);

// Performs hardware-accelerated bilinear sampling. This is similar to
// `pl_shader_sample_direct`, but forces bilinear interpolation.
bool pl_shader_sample_bilinear(pl_shader sh, const struct pl_sample_src *src);

// Performs hardware-accelerated / efficient bicubic sampling. This is more
// efficient than using the generalized sampling routines and
// pl_filter_function_bicubic. Only works well when upscaling - avoid for
// downscaling.
bool pl_shader_sample_bicubic(pl_shader sh, const struct pl_sample_src *src);

// A sampler that is similar to nearest neighbour sampling, but tries to
// preserve pixel aspect ratios. This is mathematically equivalent to taking an
// idealized image with square pixels, sampling it at an infinite resolution,
// and then downscaling that to the desired resolution. (Hence it being called
// "oversample"). Good for pixel art.
//
// The threshold provides a cutoff threshold below which the contribution of
// pixels should be ignored, trading some amount of aspect ratio distortion for
// a slightly crisper image. A value of `threshold == 0.5` makes this filter
// equivalent to regular nearest neighbour sampling.
bool pl_shader_sample_oversample(pl_shader sh, const struct pl_sample_src *src,
                                 float threshold);

struct pl_sample_filter_params {
    // The filter to use for sampling.
    struct pl_filter_config filter;
    // The precision of the LUT. Defaults to 64 if unspecified.
    int lut_entries;
    // See `pl_filter_params.cutoff`. Defaults to 0.001 if unspecified. Only
    // relevant for polar filters.
    float cutoff;
    // Antiringing strength. A value of 0.0 disables antiringing, and a value
    // of 1.0 enables full-strength antiringing. Defaults to 0.0 if
    // unspecified. Only relevant for separated/orthogonal filters.
    float antiring;
    // Disable the use of compute shaders (e.g. if rendering to non-storable tex)
    bool no_compute;
    // Disable the use of filter widening / anti-aliasing (for downscaling)
    bool no_widening;

    // This shader object is used to store the LUT, and will be recreated
    // if necessary. To avoid thrashing the resource, users should avoid trying
    // to re-use the same LUT for different filter configurations or scaling
    // ratios. Must be set to a valid pointer, and the target NULL-initialized.
    pl_shader_obj *lut;
};

#define pl_sample_filter_params(...) (&(struct pl_sample_filter_params) { __VA_ARGS__ })

// Performs polar sampling. This internally chooses between an optimized compute
// shader, and various fragment shaders, depending on the supported GLSL version
// and GPU features. Returns whether or not it was successful.
//
// Note: `params->filter.polar` must be true to use this function.
bool pl_shader_sample_polar(pl_shader sh, const struct pl_sample_src *src,
                            const struct pl_sample_filter_params *params);

// Performs orthogonal (1D) sampling. Using this twice in a row (once vertical
// and once horizontal) effectively performs a 2D upscale. This is lower
// quality than polar sampling, but significantly faster, and therefore the
// recommended default. Returns whether or not it was successful.
//
// `src` must represent a scaling operation that only scales in one direction,
// i.e. either only X or only Y. The other direction must be left unscaled.
//
// Note: Due to internal limitations, this may currently only be used on 2D
// textures - even though the basic principle would work for 1D and 3D textures
// as well.
bool pl_shader_sample_ortho2(pl_shader sh, const struct pl_sample_src *src,
                             const struct pl_sample_filter_params *params);

enum PL_DEPRECATED { // for `int pass`
    PL_SEP_VERT = 0,
    PL_SEP_HORIZ,
    PL_SEP_PASSES
};

PL_API_END

#endif // LIBPLACEBO_SHADERS_SAMPLING_H_
