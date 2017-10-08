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

// Represents a shader fragment. This is not a complete shader, but a
// collection of shader text together with description of the input required to
// satisfy it.
struct pl_shader {
    // These fields are read-only.
    struct pl_context *ctx;
    const struct ra *ra;
    void *priv;

    // The shader body, as literal GLSL. The exact interpretation of this GLSL
    // depends on the function generating the pl_shader fragment.
    const char *glsl;

    // The required work group size, if this is a compute shader. If any of
    // these integers is 0, then the shader is not considered a compute shader
    // and this field can safely be ignored.
    int compute_work_groups[3];

    // If this pass is a compute shader, this field indicates the shared memory
    // size requirements for this shader pass.
    size_t compute_shmem;

    // A set of input variables needed to satisfy this shader fragment, together
    // with their underlying raw data (as required for ra_var_update).
    struct ra_var *variables;
    const void **variable_data;
    int num_variables;

    // A list of input descriptors needed to satisfy this shader fragment,
    // together with the underlying objects bound to the corresponding
    // descriptor (as required for ra_desc_update).
    struct ra_desc *descriptors;
    const void **descriptor_bindings;
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
void pl_shader_free(struct pl_shader **shader);

// Returns whether or not a pl_shader needs to be run as a compute shader.
bool pl_shader_is_compute(const struct pl_shader *shader);

// Built-in shader fragments that represent colorspace transformations. As a
// convention, all of these operations are assumed to operate in-place a
// pre-defined `vec4 color`, the interpretation of which depends on the
// operation performed but which is always normalized to the range 0-1 such
// that a value of 1.0 represents the color space's nominal peak.

// Linearize (expand) `vec4 color`, given a specified color_transfer. In
// essence, this is the ITU-R EOTF, calculated on an idealized (reference)
// monitor with a white point of PL_COLOR_REF_WHITE and infinite contrast.
void pl_shader_linearize(struct pl_shader *s, enum pl_color_transfer trc);

// Delinearize (compress), given a TRC as output. This corresponds to the
// inverse EOTF (not the OETF) in ITU-R terminology, again assuming a
// reference monitor.
void pl_shader_delinearize(struct pl_shader *s, enum pl_color_transfer trc);

// Applies the OOTF / inverse OOTF described by a given pl_color_light. That
// is, the OOTF will always take the `vec4 color` from the specified `light`
// to display-referred space, and the inverse OOTF will always take the color
// from display-referred space to the specified `light`.
// The value of `peak` should be set to the encoded color's nominal peak
// (which can be obtained from pl_color_transfer_nominal_peak).
void pl_shader_ootf(struct pl_shader *s, enum pl_color_light light, float peak);
void pl_shader_inverse_ootf(struct pl_shader *s, enum pl_color_light light, float peak);

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
    enum pl_rendering_intent intent;

    // Algorithm and configuration used for tone-mapping. For non-tunable
    // algorithms, the `param` is ignored. If the tone mapping parameter is
    // left as 0.0, the tone-mapping curve's preferred default parameter will
    // be used
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
    // value of 0.0 completely disables this behavior. A recommended value is
    // 2.0, which provides a good balance between realistic-looking highlights
    // and preserving saturation.
    float tone_mapping_desaturate;

    // If true, enables the gamut warning feature. This will visibly highlight
    // all out-of-gamut colors (by inverting them), if they would have been
    // clipped as a result of gamut/tone mapping. (Obviously, this feature only
    // makes sense with TONE_MAPPING_CLIP)
    bool gamut_warning;

    // If peak_detect_ssbo is set to a valid pointer, this enables the peak
    // detection feature. The buffer will be implicitly created and updated by
    // pl_shader_color_map, but must be destroyed by the caller when no longer
    // needed. Subsequent calls to pl_color_map for subsequent frames should
    // re-use the same peak_detect_ssbo. `peak_detect_frames` specifies how
    // many frames to smooth (average) over, and must be at least 1. A
    // recommend value range is 50-100, which smooths the peak over a time
    // period of typically 1-2 seconds and results in a fairly jitter-free
    // result while still reacting relatively quickly to scene transitions.
    const struct ra_buf **peak_detect_ssbo;
    int peak_detect_frames;
};

// Contains a built-in definition of pl_color_map_params initialized to the
// recommended default values.
extern const struct pl_color_map_params pl_color_map_recommended_params;

// Maps `vec4 color` from one color space to another color space according
// to the parameters (described in greater depth above). If `prelinearized`
// is true, the logic will assume the input has already been linearized by the
// caller (e.g. as part of a previous linear light scaling operation).
void pl_shader_color_map(struct pl_shader *s,
                         const struct pl_color_map_params *params,
                         struct pl_color_space src, struct pl_color_space dst,
                         bool prelinearized);

#endif // LIBPLACEBO_SHADERS_H_
