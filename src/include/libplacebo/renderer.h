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

#ifndef LIBPLACEBO_RENDERER_H_
#define LIBPLACEBO_RENDERER_H_

#include <libplacebo/colorspace.h>
#include <libplacebo/filters.h>
#include <libplacebo/gpu.h>
#include <libplacebo/shaders/colorspace.h>
#include <libplacebo/shaders/sampling.h>
#include <libplacebo/swapchain.h>

struct pl_renderer;

// Creates a new renderer object, which is backed by a GPU context. This is a
// high-level object that takes care of the rendering chain as a whole, from
// the source textures to the finished frame.
struct pl_renderer *pl_renderer_create(struct pl_context *ctx,
                                       const struct pl_gpu *gpu);
void pl_renderer_destroy(struct pl_renderer **rr);

// Flushes the internal redraw cache of this renderer. This is normally not
// needed, even if the image parameters, colorspace or target configuration
// change, since libplacebo will internally detect such circumstances and
// invalidate stale caches. Doing this explicitly *may* be useful to ensure
// that memory resources associated with old objects are freed; or in case
// the user wants to switch to a new file with a different interpretation of
// `pl_image.signature`.
void pl_renderer_flush_cache(struct pl_renderer *rr);

// Represents the options used for rendering. These affect the quality of
// the result.
struct pl_render_params {
    // Configures the algorithms used for upscaling and downscaling,
    // respectively. If left as NULL, then libplacebo will only use inexpensive
    // sampling (bilinear or neareast neighbour depending on the capabilities
    // of the hardware / texture).
    //
    // Note: Setting `downscaler` to NULL also implies `skip_anti_aliasing`,
    // since the built-in GPU sampling algorithms can't anti-alias.
    //
    // Note: If set to the same address as the built-in `pl_filter_bicubic`,
    // `pl_filter_box` etc.; libplacebo will also use the more efficient
    // direct sampling algorithm where possible without quality loss.
    const struct pl_filter_config *upscaler;
    const struct pl_filter_config *downscaler;

    // The number of entries for the scaler LUTs. Defaults to 64 if left unset.
    int lut_entries;

    // The anti-ringing strength to apply to non-polar filters. See the
    // equivalent option in `pl_sample_filter_params` for more information.
    float antiringing_strength;

    // Configures the algorithm used for frame mixing (when using
    // `pl_render_image_mix`). Ignored otherwise. As a special requirement,
    // this must be a filter config with `polar` set to false, since it's only
    // used for 1D mixing and thus only 1D filters are compatible. If left as
    // NULL, then libplacebo will use a built-in, inexpensive frame mixing
    // algorithm.
    //
    // It's worth pointing out that this built-in frame mixing can often be
    // better than any of the available filter configurations. So it's not a
    // bad idea to leave this as NULL. In fact, that's the recommended default.
    const struct pl_filter_config *frame_mixer;

    // Configures the settings used to deband source textures. Leaving this as
    // NULL disables debanding.
    //
    // Note: The `deband_params.grain` setting is automatically adjusted to
    // prevent blowing up on HDR sources. The user need not account for this.
    const struct pl_deband_params *deband_params;

    // Configures the settings used to sigmoidize the image before upscaling.
    // This is not always used. If NULL, disables sigmoidization.
    const struct pl_sigmoid_params *sigmoid_params;

    // Configures the color adjustment parameters used to decode the color.
    // This can be used to apply additional artistic settings such as
    // desaturation, etc. If NULL, defaults to &pl_color_adjustment_neutral.
    const struct pl_color_adjustment *color_adjustment;

    // Configures the settings used to detect the peak of the source content,
    // for HDR sources. Has no effect on SDR content. If NULL, peak detection
    // is disabled.
    const struct pl_peak_detect_params *peak_detect_params;

    // Configures the settings used to tone map from HDR to SDR, or from higher
    // gamut to standard gamut content. If NULL, defaults to
    // `&pl_color_map_default_params`.
    const struct pl_color_map_params *color_map_params;

    // Configures the settings used to dither to the output depth. Leaving this
    // as NULL disables dithering.
    const struct pl_dither_params *dither_params;

    // Configures the settings used to generate a 3DLUT, if required. If NULL,
    // defaults to `&pl_3dlut_default_params`.
    const struct pl_3dlut_params *lut3d_params;

    // Configures the settings used to simulate color blindness, if desired.
    // If NULL, this feature is disabled.
    const struct pl_cone_params *cone_params;

    // --- Performance / quality trade-off options:
    // These should generally be left off where quality is desired, as they can
    // degrade the result quite noticeably; but may be useful for older or
    // slower hardware. Note that libplacebo will automatically disable
    // advanced features on hardware where they are unsupported, regardless of
    // these settings. So only enable them if you need a performance bump.

    // Disables anti-aliasing on downscaling. This will result in moiré
    // artifacts and nasty, jagged pixels when downscaling, except for some
    // very limited special cases (e.g. bilinear downsampling to exactly 0.5x).
    //
    // Significantly speeds up downscaling with high downscaling ratios.
    bool skip_anti_aliasing;

    // Cutoff value for polar sampling. See the equivalent option in
    // `pl_sample_filter_params` for more information.
    float polar_cutoff;

    // Skips dispatching the high-quality scalers for overlay textures, and
    // always falls back to built-in GPU samplers. Note: The scalers are
    // already disabled if the overlay texture does not need to be scaled.
    bool disable_overlay_sampling;

    // Allows the peak detection result to be delayed by up to a single frame,
    // which can sometimes (not always) allow skipping some otherwise redundant
    // sampling work. Only relevant when peak detection is active (i.e.
    // params->peak_detect_params is set and the source is HDR).
    bool allow_delayed_peak_detect;

    // --- Performance tuning / debugging options
    // These may affect performance or may make debugging problems easier,
    // but shouldn't have any effect on the quality.

    // Disables the use of a redraw cache. Normally, when rendering the same
    // frame multiple times (as identified via pl_image.signature), libplacebo
    // will try to skip redraws by using a cache of results. However, in some
    // circumstances, such as when the user knows that there will be no or
    // infrequent redraws, or when the user can't come up with meaningful
    // `signature` values, this field will allow disabling the use of a cache.
    //
    // It's worth pointing out that the user can toggle this field on and off
    // at any point in time, even on subsequent frames. The meaning of the
    // field simply means that libplacebo will act as if the cache didn't
    // exist; it will not be read from, written to, or updated.
    //
    // It's also worth pointing out that this option being `false` does not
    // guarantee the use of a redraw cache. It will be implicitly disabled, for
    // example, if the hardware does not support the required features
    // (typically the presence of blittable texture formats).
    bool skip_redraw_caching;

    // Disables linearization / sigmoidization before scaling. This might be
    // useful when tracking down unexpected image artifacts or excessing
    // ringing, but it shouldn't normally be necessary.
    bool disable_linear_scaling;

    // Forces the use of the "general" scaling algorithms even when using the
    // special-cased built-in presets like `pl_filter_bicubic`. Basically, this
    // disables the more efficient implementations in favor of the slower,
    // general-purpose ones.
    bool disable_builtin_scalers;

    // Forces the use of a 3DLUT, even in cases where the use of one is
    // unnecessary. This is slower, but may improve the quality of the gamut
    // reduction step, if one is performed.
    bool force_3dlut;
};

// This contains the default/recommended options for reasonable image quality,
// while also not being too terribly slow. All of the *_params structs
// are defaulted to the corresponding *_default_params.
extern const struct pl_render_params pl_render_default_params;

#define PL_MAX_PLANES 4

// High level description of a single slice of an image. This basically
// represents a single 2D plane, with any number of components
struct pl_plane {
    // The texture underlying this plane. The texture must be 2D, and
    // `texture->params.sampleable` must be true.
    const struct pl_tex *texture;

    // Describes the number and interpretation of the components in this plane.
    // This defines the mapping from component index to the canonical component
    // order (RGBA, YCbCrA or XYZA). It's worth pointing out that this is
    // completely separate from `texture->format.sample_order`. The latter is
    // essentially irrelevant/transparent for the API user, since it just
    // determines which order the texture data shows up as inside the GLSL
    // shader; whereas this field controls the actual meaning of the component.
    //
    // Example; if the user has a plane with just {Y} and a plane with just
    // {Cb Cr}, and a GPU that only supports bgra formats, you would still
    // specify the component mapping as {0} and {1 2} respectively, even though
    // the GPU is sampling the data in the order BGRA. Use -1 for "ignored"
    // components.
    int components;           // number of relevant components
    int component_mapping[4]; // semantic index of each component

    // Controls the sample offset, relative to the "reference" dimensions. For
    // an example of what to set here, see `pl_chroma_location_offset`. Note
    // that this is given in unit of reference pixels. For a graphical example,
    // imagine you have a 2x2 image with a 1x1 (subsampled) plane. Without any
    // shift (0.0), the situation looks like this:
    //
    // X-------X  X = reference pixel
    // |       |  P = plane pixel
    // |   P   |
    // |       |
    // X-------X
    //
    // For 4:2:0 subsampling, this corresponds to PL_CHROMA_CENTER. If the
    // shift_x was instead set to -0.5, the `P` pixel would be offset to the
    // left by half the separation between the reference (`X` pixels), resulting
    // in the following:
    //
    // X-------X  X = reference pixel
    // |       |  P = plane pixel
    // P       |
    // |       |
    // X-------X
    //
    // For 4:2:0 subsampling, this corresponds to PL_CHROMA_LEFT.
    //
    // Note: It's recommended to fill this using `pl_chroma_location_offset` on
    // the chroma planes.
    float shift_x, shift_y;
};

enum pl_overlay_mode {
    PL_OVERLAY_NORMAL = 0, // treat the texture as a normal, full-color texture
    PL_OVERLAY_MONOCHROME, // treat the texture as a single-component alpha map
};

// A struct representing an image overlay (e.g. for subtitles or on-screen
// status messages, controls, ...)
struct pl_overlay {
    // The plane to overlay. Multi-plane overlays are not supported. If
    // necessary, multiple planes can be combined by treating them as separate
    // overlays with different base colors.
    //
    // Note: shift_x/y are simply treated as a uniform sampling offset.
    struct pl_plane plane;
    // The (absolute) coordinates at which to render this overlay texture. May
    // be flipped, and partially or wholly outside the image. If the size does
    // not exactly match the texture, it will be scaled/stretched to fit.
    struct pl_rect2d rect;

    // This controls the coloring mode of this overlay.
    enum pl_overlay_mode mode;
    // If `mode` is PL_OVERLAY_MONOCHROME, then the texture is treated as an
    // alpha map and multiplied by this base color. Ignored for the other modes.
    float base_color[3];

    // This controls the colorspace information for this overlay. The contents
    // of the texture / the value of `color` are interpreted according to this.
    struct pl_color_repr repr;
    struct pl_color_space color;
};

// High-level description of a source image to render
struct pl_image {
    // A generic signature uniquely identifying this image. The contents don't
    // matter, as long as they're unique for "identical" frames. This signature
    // is used to cache intermediate results, thus speeding up redraws.
    // In practice, the user might set this to e.g. an incrementing counter.
    //
    // If the user can't ensure the uniqueness of this signature for whatever
    // reason, they must set `pl_render_params.skip_redraw_caching`, in which
    // case the contents of this field are ignored.
    //
    // NOTE: Re-using the same `signature` also requires that the contents of
    // the planes (plane[i].texture) as well as any overlays has not changed
    // since the previous usage. In other words, touching the texture in any
    // way using the pl_tex_* APIs and then trying to re-use them for the same
    // signature, or trying to re-use the same signature with different
    // textures, is undefined behavior. (It's the *contents* that matter here,
    // the actual texture object can be a different one, as long as the
    // contents and parameters are the same)
    uint64_t signature;

    // Each frame is split up into some number of planes, each of which may
    // carry several components and be of any size / offset.
    int num_planes;
    struct pl_plane planes[PL_MAX_PLANES];

    // Color representation / encoding / semantics associated with this image.
    struct pl_color_repr repr;
    struct pl_color_space color;

    // Optional ICC profile associated with this image.
    struct pl_icc_profile profile;

    // The reference dimensions of this image. For typical content, this is the
    // dimensions of the largest (non-subsampled) plane, e.g. luma. Note that
    // for anamorphic content, this is the size of the texture itself, not the
    // "nominal" size of the video. (Anamorphic pixel ratio conversions are
    // done implicitly by differing the aspect ratio between `src_rect` and
    // `dst_rect`)
    int width;
    int height;

    // The source rectangle which we want to render from, relative to the
    // reference dimensions. Pixels outside of this rectangle will ostensibly
    // be ignored, but note that they may still contribute to the output data
    // due to the effects of texture filtering. `src_rect` may be flipped, and
    // may be partially or wholly outside the bounds of the texture. (Optional)
    struct pl_rect2df src_rect;

    // A list of additional overlays to render directly on top of this image.
    // These overlays will be treated as though they were part of the image,
    // which means they will be affected by the main scaler as well as by
    // frame mixing algorithms. See also `pl_target.overlays`
    const struct pl_overlay *overlays;
    int num_overlays;

    // Note on subsampling and plane correspondence: All planes belonging to
    // the same image will only be streched by an integer multiple (or inverse
    // thereof) in order to match the reference dimensions of this image. For
    // example, suppose you have an 8x4 image. A valid plane scaling would be
    // 4x2 -> 8x4 or 4x4 -> 4x4, but not 6x4 -> 8x4. So if a 6x4 plane is
    // given, then it would be treated like a cropped 8x4 plane (since 1.0 is
    // the closest scaling ratio to the actual ratio of 1.3).
    //
    // For an explanation of why this makes sense, consider the relatively
    // common example of a subsampled, oddly sized (e.g. jpeg) image. In such
    // cases, for example a 35x23 image, the 4:2:0 subsampled chroma plane
    // would have to end up as 17.5x11.5, which gets rounded up to 18x12 by
    // implementations. So in this example, the 18x12 chroma plane would get
    // treated by libplacebo as an oversized chroma plane - i.e. the plane
    // would get sampled as if it was 17.5 pixels wide and 11.5 pixels large.
};

// Represents the target of a rendering operation
struct pl_render_target {
    // The framebuffer (or texture) we want to render to. Must have `renderable`
    // set. The other capabilities are optional, but in particular `storable`
    // and `blittable` can help boost performance if available.
    const struct pl_tex *fbo;

    // The destination rectangle which we want to render into. If this is
    // larger or smaller than the src_rect, or if the aspect ratio is
    // different, scaling will occur. `dst_rect` may be flipped, and may be
    // partially or wholly outside the bounds of the fbo. (Optional)
    struct pl_rect2d dst_rect;

    // The color representation and space of the output. If this does not match
    // the color space of the source, libplacebo will convert the colors
    // automatically.
    struct pl_color_repr repr;
    struct pl_color_space color;

    // Optional ICC profile associated with this render target.
    struct pl_icc_profile profile;

    // A list of additional overlays to render directly onto the output. These
    // overlays will be rendered after the image itself has been fully scaled
    // and output, and will not be affected by e.g. frame mixing. See also
    // `pl_image.overlays`
    const struct pl_overlay *overlays;
    int num_overlays;
};

// Fills in a pl_render_target based on a swapchain frame's FBO and metadata.
void pl_render_target_from_swapchain(struct pl_render_target *out_target,
                                     const struct pl_swapchain_frame *frame);

// Render a single image to a target using the given parameters. This is
// fully dynamic, i.e. the params can change at any time. libplacebo will
// internally detect and flush whatever caches are invalidated as a result of
// changing colorspace, size etc.
//
// Note on lifetime: Once this call returns, the passed structures may be
// freely overwritten or discarded by the caller, even the referenced
// `pl_tex` objects may be freely reused.
bool pl_render_image(struct pl_renderer *rr, const struct pl_image *image,
                     const struct pl_render_target *target,
                     const struct pl_render_params *params);

/* TODO

// Represents a mixture of input images, distributed temporally.
//
// NOTE: Images must be sorted by timestamp, i.e. `distances` must be
// monotonically increasing.
struct pl_image_mix {
    // The number of images in this mixture. The number of images should be
    // sufficient to meet the needs of the configured frame mixer. See the
    // section below for more information.
    int num_images;

    // A list of the images themselves. The images can have different
    // colorspaces, configurations of planes, or even sizes. Note: when using
    // frame mixing, it's absolutely critical that all of the images have
    // a unique value of `pl_image.signature`.
    const struct pl_image *images;

    // A list of relative distance vectors for each image, respectively.
    // Basically, the "current" instant is always assigned a position of 0.0;
    // and this distances array will give the relative offset (either negative
    // or positive) of the images in the mixture. The values are expected to be
    // normalized such that a separation of 1.0 corresponds to roughly one
    // nominal source frame duration. So a constant framerate video file will
    // always have distances like e.g. {-2.3, -1.3, -0.3, 0.7, 1.7, 2.7}, using
    // an example radius of 3.
    //
    // The interpretation of timestamps is that frames are momentary samples
    // centered on this timestamp. In other words, in the example above, the
    // frame with timestamp 0.7 should (theoretically) be visible in the
    // interval [0.2, 1.2] on a zero-order-hold (nearest neighbour) display.
    //
    // In cases where the framerate is variable (e.g. VFR video), the choice of
    // what to scale to use can be difficult to answer. A typical choice would
    // be either to use the canonical (container-tagged) framerate, or the
    // highest momentary framerate, as a reference. If all else fails, you
    // could also use the display's framerate.
    const float *distances;

    // The duration for which the resulting image will be held, using the same
    // scale as the `distance`. This duration is centered around the instant
    // 0.0. Basically, the image being rendered is assumed to be displayed from
    // the time -vsync_duration/2 up to the time vsync_duration/2. If the
    // display has a variable frame-rate (e.g. Adaptive Sync), then you're
    // better off not using this function and instead just painting the frames
    // directly using `pl_render_image` as they come in.
    float vsync_duration;

    // Explanation of the frame mixing radius: The algorithm chosen in
    // `pl_render_params.frame_mixing` has a canonical radius equal to
    // `pl_filter_config.kernel->radius`. This means that the frame mixing
    // algorithm will (only) need to consult all of the frames that have a
    // distance within the interval [-radius, radius]. As such, the user should
    // include all such frames in `images`, but may prune or omit frames that
    // lie outside it.
    //
    // The built-in frame mixing (`pl_render_params.frame_mixing == NULL`) has
    // a canonical radius equal to `vsync_duration/2`.
};

// Render a mixture of images to the target using the given parameters. This
// functions much like a generalization of `pl_render_image`, for when the API
// user has more control over the frame queue / vsync timings and can present a
// complete picture of the current instant's neighbourhood. This allows
// libplacebo to use frame blending in order to eliminate judder artifacts
// typically associated with source/display frame rate mismatch.
//
// In particular, pl_render_image can be semantically viewed as a special case
// of pl_render_image_mix, where num_images = 1, that frame's distance is 0.0,
// and the vsync_duration is 0.0. (But using `pl_render_image` instead of
// `pl_render_image_mix` in such an example can still be more efficient)
bool pl_render_image_mix(struct pl_renderer *rr, const struct pl_image_mix *mix,
                         const struct pl_render_target *target,
                         const struct pl_render_params *params);
*/

#endif // LIBPLACEBO_RENDERER_H_
