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

#include "ra.h"

struct pl_renderer;

// Creates a new renderer object, which is backed by a RA context. This is a
// high-level object that takes care of the rendering chain as a whole, from
// the source textures to the finished frame.
struct pl_renderer *pl_renderer_create(struct pl_context *ctx,
                                       const struct ra *ra);
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
    // sampling (bicubic, bilinear or neareast neighbour depending on the
    // capabilities of the hardware).
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
    const struct pl_deband_params *deband_params;

    // Configures the color adjustment parameters used to decode the color.
    // This can be used to apply additional artistic settings such as
    // desaturation, etc. If NULL, defaults to &pl_color_adjustment_neutral.
    const struct pl_color_adjustment *color_adjustment;

    // Configures the settings used to tone map from HDR to SDR, or from higher
    // gamut to standard gamut content. If NULL, defaults to
    // `&pl_color_map_default_params`.
    const struct pl_color_map_params *color_map_params;

    // Configures the settings used to dither to the output depth. Leaving this
    // as NULL disables dithering.
    const struct pl_dither_params *dither_params;

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

    // Forces the use of the "general" scaling algorithms even when using the
    // special-cased built-in presets like `pl_filter_bicubic`. Basically, this
    // disables the more efficient implementations in favor of the slower,
    // general-purpose ones.
    bool disable_builtin_scalers;
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
    const struct ra_tex *texture;

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
    // the GPU is sampling the data in the order BGRA.
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
    float shift_x, shift_y;
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
    uint64_t signature;

    // Each frame is split up into some number of planes, each of which may
    // carry several components and be of any size / offset.
    int num_planes;
    struct pl_plane planes[PL_MAX_PLANES];

    // Color representation / encoding / semantics associated with this image
    struct pl_color_repr repr;
    struct pl_color_space color;

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
    // may be partially or wholly outside the bounds of the texture.
    struct pl_rect2df src_rect;
};

// Represents the target of a rendering operation
struct pl_render_target {
    // The framebuffer (or texture) we want to render to. Must have `renderable`
    // set. The other capabilities are optional, but in particular `storable`
    // and `blittable` can help boost performance if available.
    const struct ra_tex *fbo;

    // The destination rectangle which we want to render into. If this is
    // larger or smaller than the src_rect, or if the aspect ratio is
    // different, scaling will occur. `dst_rect` may be flipped, and may be
    // partially or wholly outside the bounds of the fbo.
    struct pl_rect2d dst_rect;

    // The color representation and space of the output. If this does not match
    // the color space of the source, libplacebo will convert the colors
    // automatically.
    struct pl_color_repr repr;
    struct pl_color_space color;
};

// Render a single image to a target using the given parameters. This is
// fully dynamic, i.e. the params can change at any time. libplacebo will
// internally detect and flush whatever caches are invalidated as a result of
// changing colorspace, size etc.
bool pl_render_image(struct pl_renderer *rr, const struct pl_image *image,
                     const struct pl_render_target *target,
                     const struct pl_render_params *params);

/* TODO

// Represents a mixture of input images, distributed temporally
struct pl_image_mix {
    // The number of images in this mixture. The number of images should be
    // sufficient to meet the needs of the configured frame mixer. See the
    // section below for more information.
    int num_images;

    // A list of the images themselves. The images can have different
    // colorspaces, configurations of planes, or even sizes. Note: when using
    // frame mixing, it's absolutely critical that all of the images have
    // a unique value of `pl_image.signature`.
    struct pl_image *images;

    // A list of relative distance vectors for each image, respectively.
    // Basically, the "current" instant is always assigned a position of 0.0;
    // and this distances array will give the relative offset (either negative
    // or positive) of the images in the mixture. The values are expected to be
    // normalized such that a separation of 1.0 corresponds to roughly one
    // nominal source frame duration. So a constant framerate video file will
    // always have distances like e.g. {-2.3, -1.3, -0.3, 0.7, 1.7, 2.7}, using
    // an example radius of 3.
    //
    // In cases where the framerate is variable (e.g. VFR video), the choice of
    // what to scale to use can be difficult to answer. A typical choice would
    // be either to use the canonical (container-tagged) framerate, or the
    // highest momentary framerate, as a reference.
    float *distances;

    // The duration for which the resulting image will be held, using the same
    // scale as the `distance`. This duration is centered around the instant
    // 0.0. Basically, the image is assumed to be displayed from the time
    // -vsync_duration/2 up to the time vsync_duration/2.
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
    // a canonical radius equal to vsync_duration/2.
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
