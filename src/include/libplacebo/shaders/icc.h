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

#ifndef LIBPLACEBO_SHADERS_ICC_H_
#define LIBPLACEBO_SHADERS_ICC_H_

// Functions for generating and applying ICC-derived (3D)LUTs

#include <libplacebo/colorspace.h>
#include <libplacebo/shaders.h>

PL_API_BEGIN

struct pl_icc_params {
    // The rendering intent to use, for profiles with multiple intents. A
    // recommended value is PL_INTENT_RELATIVE_COLORIMETRIC for color-accurate
    // video reproduction, or PL_INTENT_PERCEPTUAL for profiles containing
    // meaningful perceptual mapping tables for some more suitable color space
    // like BT.709.
    //
    // If this is set to the special value PL_INTENT_AUTO, will use the
    // preferred intent provided by the profile header.
    enum pl_rendering_intent intent;

    // The size of the 3DLUT to generate. If left as NULL, these individually
    // default to values appropriate for the profile. (Based on internal
    // precision heuristics)
    size_t size_r, size_g, size_b;

    // This field can be used to override the detected brightness level of the
    // ICC profile. If you set this to the special value 0 (or a negative
    // number), libplacebo will attempt reading the brightness value from the
    // ICC profile's tagging (if available), falling back to PL_COLOR_SDR_WHITE
    // if unavailable.
    float max_luma;

    // Force black point compensation. May help avoid crushed or raised black
    // points on "improper" profiles containing e.g. colorimetric tables that
    // do not round-trip. Should not be required on well-behaved profiles,
    // or when using PL_INTENT_PERCEPTUAL, but YMMV.
    bool force_bpc;

    // 3DLUT caching API. Providing these functions can help speed up ICC LUT
    // generation by saving/loading profiles to/from disk. Both of these
    // callbacks are optional.
    void *cache_priv;
    //
    // This is called to inform users of new cache entries. The user may store
    // this cache to disk or some other internal caching mechanism.
    void (*cache_save)(void *priv, uint64_t sig, const uint8_t *cache, size_t size);
    //
    // This is called to query for existing cache entries. The user should look
    // up this cache entry and write its contents to `cache`, ensuring that no
    // more than `size` bytes are written, and return `true` on success.
    bool (*cache_load)(void *priv, uint64_t sig, uint8_t *cache, size_t size);
    //
    // Note: The `signature` of a cache entry is NOT equal to the `signature`
    // of the underlying `pl_icc_object` - it is split up into separate entries
    // for `pl_icc_decode` and `pl_icc_encode`, and also includes a hashed
    // representation of the encoded parameters.
    //
    // Note: These callbacks will only be called from within `pl_icc_decode` /
    // `pl_icc_encode`, so `cache_priv` should exceed this lifetime.
};

#define PL_ICC_DEFAULTS                         \
    .intent = PL_INTENT_RELATIVE_COLORIMETRIC,  \
    .max_luma = PL_COLOR_SDR_WHITE,

#define pl_icc_params(...) (&(struct pl_icc_params) { PL_ICC_DEFAULTS __VA_ARGS__ })
extern const struct pl_icc_params pl_icc_default_params;

// This object represents a "parsed" ICC profile.
typedef const struct pl_icc_object_t {
    // Provided params, with the `intent` and `size` fields set (as described)
    struct pl_icc_params params;

    // Signature of the corresponding ICC profile.
    uint64_t signature;

    // Detected color space (or UNKNOWN for profiles which don't contain an
    // exact match), with HDR metedata set to the detected gamut and
    // white/black value ranges.
    struct pl_color_space csp;

    // Best estimate of profile gamma. This only serves as a rough guideline.
    float gamma;

    // Smallest containing primary set, always set.
    enum pl_color_primaries containing_primaries;
} *pl_icc_object;

// Attempts opening/parsing the contents of an ICC profile. The resulting
// object is memory managed and may outlive the original profile - access
// to the underlying profile is no longer needed once this returns.
pl_icc_object pl_icc_open(pl_log log, const struct pl_icc_profile *profile,
                          const struct pl_icc_params *params);
void pl_icc_close(pl_icc_object *icc);

// Decode the input from the colorspace determined by the attached ICC profile
// to linear light RGB (in the profile's containing primary set). `lut` must be
// set to a shader object that will store the GPU resources associated with the
// generated LUT. The resulting color space will be written to `out_csp`.
void pl_icc_decode(pl_shader sh, pl_icc_object profile, pl_shader_obj *lut,
                   struct pl_color_space *out_csp);

// Encode the input from linear light RGB (in the profile's containing primary
// set) into the colorspace determined by the attached ICC profile. `lut` must
// be set to a shader object that will store the GPU resources associated with
// the generated LUT.
void pl_icc_encode(pl_shader sh, pl_icc_object profile, pl_shader_obj *lut);

PL_API_END

#endif // LIBPLACEBO_SHADERS_ICC_H_
