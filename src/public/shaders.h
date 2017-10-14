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

#ifndef LIBPLACEBO_SHADERS_H_
#define LIBPLACEBO_SHADERS_H_

#include "colorspace.h"
#include "ra.h"

// Represents a vertex attribute. The four values will be bound to the four
// corner vertices respectively, in clockwise order starting from the top left.
struct pl_shader_va {
    struct ra_vertex_attrib attr;
    const void *data[4];
};

// Represents a bound shared variable / descriptor
struct pl_shader_var {
    struct ra_var var;  // the underlying variable description
    const void *data;   // the raw data (interpretation as with ra_var_update)
    bool dynamic;       // if true, the value is expected to change frequently
};

struct pl_shader_desc {
    struct ra_desc desc; // the underlying descriptor description
    const void *binding; // the object being bound (as for ra_desc_binding)
};

// Represents a shader fragment. This is not a complete shader, but a
// collection of shader text together with description of the input required to
// satisfy it.
struct pl_shader {
    // These fields are read-only.
    struct pl_context *ctx;
    const struct ra *ra;
    void *priv;

    // The shader text, as literal GLSL. The `header` is assumed to be outside
    // of any function definition, and will be used to define new helper
    // functions if required. The `body` is assumed to be inside a function
    // (typically `main`), and defines the requested transformation logic.
    const char *glsl_header;
    const char *glsl_body;

    // The required work group size, if this is a compute shader. If any of
    // these integers is 0, then the shader is not considered a compute shader
    // and this field can safely be ignored.
    int compute_work_groups[3];

    // If this pass is a compute shader, this field indicates the shared memory
    // size requirements for this shader pass.
    size_t compute_shmem;

    // A set of input vertex attributes needed by this shader fragment.
    struct pl_shader_va *vertex_attribs;
    int num_vertex_attribs;

    // A set of input variables needed by this shader fragment.
    struct pl_shader_var *variables;
    int num_variables;

    // A list of input descriptors needed by this shader fragment,
    struct pl_shader_desc *descriptors;
    int num_descriptors;
};

// Creates a new, blank, mutable pl_shader object. The pl_shader_* family of
// functions will mutate this, which may update any of the pointers referenced
// by this struct. As such, it should be assumed that every pl_shader_*
// operation totally invalidates the contents of the struct pl_shader and
// everything pointed at by it. The resulting struct pl_shader is implicitly
// destroyed when the pl_context is destroyed.
//
// If `ra` is non-NULL, then this `ra` will be used to create objects such as
// textures and buffers, or check for required capabilities, for operations
// which depend on either of those. This is fully optional, i.e. these GLSL
// primitives are designed to be used without a dependency on `ra` wherever
// possible.
struct pl_shader *pl_shader_alloc(struct pl_context *ctx,
                                  const struct ra *ra);

// Frees a pl_shader and all resources associated with it.
void pl_shader_free(struct pl_shader **sh);

// Resets a pl_shader to a blank slate, without releasing internal memory.
// If you're going to be re-generating shaders often, this function will let
// you skip the re-allocation overhead.
void pl_shader_reset(struct pl_shader *sh);

// Returns whether or not a pl_shader needs to be run as a compute shader.
bool pl_shader_is_compute(const struct pl_shader *sh);

//-----------------------------------------------------------------------------
// Color space transformation shaders. As a convention, all of these operations
// are assumed to operate in-place a pre-defined `vec4 color`, the
// interpretation of which depends on the operation performed but which is
// always normalized to the range 0-1 such that a value of 1.0 represents the
// color space's nominal peak.

// Decode the color into normalized RGB, given a specified color_repr. This
// also takes care of additional pre- and post-conversions requires for the
// "special" color systems (XYZ, BT.2020-C, etc.). The int `texture_bits`, if
// present, indicate the depth of the texture we've sampled the color from -
// similar to the semantics on `pl_get_scaled_decoding_matrix`.
//
// Note: This function always returns PC-range RGB with pre-multiplied alpha.
// It mutates the pl_color_repr to reflect the change.
void pl_shader_decode_color(struct pl_shader *sh, struct pl_color_repr *repr,
                            struct pl_color_adjustment params, int texture_bits);

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
    // drawback of darkening the overall image quite significantly. This is
    // sort of similar to REINHARD tone-mapping + parameter 0.24.
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
    // be used. The default algorithm is PL_TONE_MAPPING_MOBIUS.
    enum pl_tone_mapping_algorithm tone_mapping_algo;
    float tone_mapping_param;

    // Desaturation coefficient. This essentially desaturates very bright
    // spectral colors towards white, resulting in a more natural-looking
    // depiction of very bright sunlit regions or images of the sunlit sky. The
    // interpretation of the coefficient is the brightness level at which
    // desaturation starts. For example, if this is set to a value of 1.2,
    // colors within 1.2 times the reference white level are preserved, and
    // colors exceeding it are gradually desaturated towards white. Values
    // below 1.0 would start to desaturate even in-gamut colors, and values
    // tending towards infinitey would turn this operation into a no-op. A
    // value of 0.0 completely disables this behavior. The default value is
    // 2.0, which provides a good balance between realistic-looking highlights
    // and preserving saturation.
    float tone_mapping_desaturate;

    // If true, enables the gamut warning feature. This will visibly highlight
    // all out-of-gamut colors (by inverting them), if they would have been
    // clipped as a result of gamut/tone mapping. (Obviously, this feature only
    // makes sense with TONE_MAPPING_CLIP)
    bool gamut_warning;
};

extern const struct pl_color_map_params pl_color_map_default_params;

// Maps `vec4 color` from one color space to another color space according
// to the parameters (described in greater depth above). If `prelinearized`
// is true, the logic will assume the input has already been linearized by the
// caller (e.g. as part of a previous linear light scaling operation).
void pl_shader_color_map(struct pl_shader *sh,
                         const struct pl_color_map_params *params,
                         struct pl_color_space src, struct pl_color_space dst,
                         bool prelinearized);

//-----------------------------------------------------------------------------
// Sampling operations. These shaders perform some form of sampling operation
// from a given ra_tex. In order to use these, the pl_shader *must* have been
// created using the same `ra` as the originating `ra_tex`. Otherwise, this
// is undefined behavior. They output their results in `vec4 color`, which
// they introduce into the scope.

struct pl_deband_params {
    // This is used as a seed for the (frame-local) PRNG. No state is preserved
    // across invocations, so the user must manually vary this across frames
    // to achieve temporal randomness.
    float seed;

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
};

extern const struct pl_deband_params pl_deband_default_params;

// Debands a given texture and returns the sampled color in `vec4 color`.
// Note: This can also be used as a pure grain function, by setting the number
// of iterations to 0.
void pl_shader_deband(struct pl_shader *sh, const struct ra_tex *tex,
                      const struct pl_deband_params *params);

#endif // LIBPLACEBO_SHADERS_H_
