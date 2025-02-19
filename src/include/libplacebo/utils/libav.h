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

#ifndef LIBPLACEBO_LIBAV_H_
#define LIBPLACEBO_LIBAV_H_

#include <libplacebo/config.h>
#include <libplacebo/gpu.h>
#include <libplacebo/shaders/deinterlacing.h>
#include <libplacebo/utils/upload.h>

#if defined(__cplusplus) && !defined(PL_LIBAV_IMPLEMENTATION)
# define PL_LIBAV_API
# define PL_LIBAV_IMPLEMENTATION 0
# warning Remember to include this file with a PL_LIBAV_IMPLEMENTATION set to 1 in \
          C translation unit to provide implementation. Suppress this warning by \
          defining PL_LIBAV_IMPLEMENTATION to 0 in C++ files.
#elif !defined(PL_LIBAV_IMPLEMENTATION)
# define PL_LIBAV_API static inline
# define PL_LIBAV_IMPLEMENTATION 1
#else
# define PL_LIBAV_API
#endif

PL_API_BEGIN

#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/version.h>
#include <libavcodec/avcodec.h>

#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 16, 100) && defined(PL_HAVE_DOVI)
# define PL_HAVE_LAV_DOLBY_VISION
# include <libavutil/dovi_meta.h>
#endif

#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(56, 61, 100)
# define PL_HAVE_LAV_FILM_GRAIN
# include <libavutil/film_grain_params.h>
#endif

#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(56, 25, 100)
# define PL_HAVE_LAV_HDR
# include <libavutil/hdr_dynamic_metadata.h>
# include <libavutil/mastering_display_metadata.h>
#endif

//------------------------------------------------------------------------
// Important note: For support for AVVkFrame, which depends on <vulkan.h>,
// users *SHOULD* include <vulkan/vulkan.h> manually before this header.
//------------------------------------------------------------------------


// Fill in the details of a `pl_frame` from an AVFrame. This function will
// explicitly clear `out_frame`, setting all extra fields to 0. After this
// function returns, the only missing data is information related to the plane
// texture itself (`planes[N].texture`), as well as any overlays (e.g.
// subtitles).
//
// Note: If the AVFrame contains an embedded ICC profile or H.274 film grain
// metadata, the resulting `out_image->profile` will reference this pointer,
// meaning that in general, the `pl_frame` is only guaranteed to be valid as
// long as the AVFrame is not freed.
//
// Note: This will ignore Dolby Vision metadata by default (to avoid leaking
// memory), either switch to pl_map_avframe_ex or do it manually using
// pl_map_dovi_metadata.
PL_LIBAV_API void pl_frame_from_avframe(struct pl_frame *out_frame, const AVFrame *frame);

// Deprecated aliases for backwards compatibility
#define pl_image_from_avframe pl_frame_from_avframe
#define pl_target_from_avframe pl_frame_from_avframe

// Copy extra metadata from an AVStream to a pl_frame. This should be called
// after `pl_frame_from_avframe` or `pl_map_avframe` (respectively), and sets
// metadata associated with stream-level side data. This is needed because
// FFmpeg rather annoyingly does not propagate stream-level metadata to frames.
PL_LIBAV_API void pl_frame_copy_stream_props(struct pl_frame *out_frame,
                                             const AVStream *stream);

#ifdef PL_HAVE_LAV_HDR
struct pl_av_hdr_metadata {
    // All fields are optional and may be passed as `NULL`.
    const AVMasteringDisplayMetadata *mdm;
    const AVContentLightMetadata *clm;
    const AVDynamicHDRPlus *dhp;
};

// Helper function to update a `pl_hdr_metadata` struct from HDR10/HDR10+
// metadata in the FFmpeg format. Unspecified/invalid elements will be left
// uninitialized in `out`.
PL_LIBAV_API void pl_map_hdr_metadata(struct pl_hdr_metadata *out,
                                const struct pl_av_hdr_metadata *metadata);
#endif

#ifdef PL_HAVE_LAV_DOLBY_VISION
// Helper function to map Dolby Vision metadata from the FFmpeg format.
PL_LIBAV_API void pl_map_dovi_metadata(struct pl_dovi_metadata *out,
                                       const AVDOVIMetadata *metadata);

// Helper function to map Dolby Vision metadata from the FFmpeg format
// to `pl_dovi_metadata`, and adds it to the `pl_frame`.
// The `pl_frame` colorspace fields and HDR struct are also updated with
// values from the `AVDOVIMetadata`.
//
// Note: The `pl_dovi_metadata` must be allocated externally.
// Also, currently the metadata is only used if the `AVDOVIRpuDataHeader`
// `disable_residual_flag` field is not zero and can be checked before allocating.
PL_DEPRECATED_IN(v7.343) PL_LIBAV_API void pl_frame_map_avdovi_metadata(
                                               struct pl_frame *out_frame,
                                               struct pl_dovi_metadata *dovi,
                                               const AVDOVIMetadata *metadata);

// Helper function to map Dolby Vision metadata from the FFmpeg format
// to `pl_dovi_metadata`, and adds it to the `pl_color_repr`.
// The `pl_color_space` fields and HDR struct are also updated with
// values from the `AVDOVIMetadata`.
//
// Note: The `pl_dovi_metadata` must be allocated externally.
// Also, currently the metadata is only used if the `AVDOVIRpuDataHeader`
// `disable_residual_flag` field is not zero and can be checked before allocating.
PL_LIBAV_API void pl_map_avdovi_metadata(struct pl_color_space *color,
                                         struct pl_color_repr *repr,
                                         struct pl_dovi_metadata *dovi,
                                         const AVDOVIMetadata *metadata);
#endif

// Helper function to test if a pixfmt would be supported by the GPU.
// Essentially, this can be used to check if `pl_map_avframe` would work for a
// given AVPixelFormat, without actually uploading or allocating anything.
PL_LIBAV_API bool pl_test_pixfmt(pl_gpu gpu, enum AVPixelFormat pixfmt);

// Variant of `pl_test_pixfmt` that also tests for the given capabilities
// being present. Note that in the presence of hardware accelerated frames,
// this cannot be tested without frame-specific information (i.e. swformat),
// but in practice this should be a non-issue as GPU-native hwformats will
// probably be fully supported.
PL_LIBAV_API bool pl_test_pixfmt_caps(pl_gpu gpu, enum AVPixelFormat pixfmt,
                                      enum pl_fmt_caps caps);

// Like `pl_frame_from_avframe`, but the texture pointers are also initialized
// to ensure they have the correct size and format to match the AVframe.
// Similar in spirit to `pl_recreate_plane`, and the same notes apply. `tex`
// must be an array of 4 pointers of type `pl_tex`, each either
// pointing to a valid texture, or NULL. Returns whether successful.
PL_LIBAV_API bool pl_frame_recreate_from_avframe(pl_gpu gpu, struct pl_frame *out_frame,
                                                 pl_tex tex[4], const AVFrame *frame);

struct pl_avframe_params {
    // The AVFrame to map. Required.
    const AVFrame *frame;

    // Backing textures for frame data. Required for all non-hwdec formats.
    // This must point to an array of four valid textures (or NULL entries).
    //
    // Note: Not cleaned up by `pl_unmap_avframe`. The intent is for users to
    // re-use this texture array for subsequent frames, to avoid texture
    // creation/destruction overhead.
    pl_tex *tex;

    // Also map Dolby Vision metadata (if supported). Note that this also
    // overrides the colorimetry metadata (forces BT.2020+PQ).
    bool map_dovi;
};

#define PL_AVFRAME_DEFAULTS \
    .map_dovi = true,

#define pl_avframe_params(...) (&(struct pl_avframe_params) { PL_AVFRAME_DEFAULTS __VA_ARGS__ })

// Very high level helper function to take an `AVFrame` and map it to the GPU.
// The resulting `pl_frame` remains valid until `pl_unmap_avframe` is called,
// which must be called at some point to clean up state. The `AVFrame` is
// automatically ref'd and unref'd if needed. Returns whether successful.
//
// Note: `out_frame->user_data` points to a privately managed opaque struct
// and must not be touched by the user.
PL_LIBAV_API bool pl_map_avframe_ex(pl_gpu gpu, struct pl_frame *out_frame,
                                    const struct pl_avframe_params *params);
PL_LIBAV_API void pl_unmap_avframe(pl_gpu gpu, struct pl_frame *frame);

// Backwards compatibility with previous versions of this API.
PL_LIBAV_API bool pl_map_avframe(pl_gpu gpu, struct pl_frame *out_frame,
                                 pl_tex tex[4], const AVFrame *avframe);

// Return the AVFrame* that a pl_frame was mapped from (via pl_map_avframe_ex)
// Note: This reference is attached to the `pl_frame` and will get freed by
// pl_unmap_avframe.
PL_LIBAV_API AVFrame *pl_get_mapped_avframe(const struct pl_frame *frame);

// Download the texture contents of a `pl_frame` back to a corresponding
// AVFrame. Blocks until completion.
//
// Note: This function performs minimal verification, so incorrect usage will
// likely result in broken frames. Use `pl_frame_recreate_from_avframe` to
// ensure matching formats.
PL_LIBAV_API bool pl_download_avframe(pl_gpu gpu,
                                      const struct pl_frame *frame,
                                      AVFrame *out_frame);

// Helper functions to update the colorimetry data in an AVFrame based on
// the values specified in the given color space / color repr / profile.
//
// Note: These functions can and will allocate AVFrame side data if needed,
// in particular to encode HDR metadata in `space.hdr`.
PL_LIBAV_API void pl_avframe_set_color(AVFrame *frame, struct pl_color_space space);
PL_LIBAV_API void pl_avframe_set_repr(AVFrame *frame, struct pl_color_repr repr);
PL_LIBAV_API void pl_avframe_set_profile(AVFrame *frame, struct pl_icc_profile profile);

// Map an AVPixelFormat to an array of pl_plane_data structs. The array must
// have at least `av_pix_fmt_count_planes(fmt)` elements, but never more than
// 4. This function leaves `width`, `height` and `row_stride`, as well as the
// data pointers, uninitialized.
//
// If `bits` is non-NULL, this function will attempt aligning the resulting
// `pl_plane_data` struct for optimal compatibility, placing the resulting
// `pl_bit_depth` metadata into `bits`.
//
// Returns the number of plane structs written to, or 0 on error.
//
// Note: This function is usually clumsier to use than the higher-level
// functions above, but it might have some fringe use cases, for example if
// the user wants to replace the data buffers by `pl_buf` references in the
// `pl_plane_data` before uploading it to the GPU.
PL_LIBAV_API int pl_plane_data_from_pixfmt(struct pl_plane_data data[4],
                                           struct pl_bit_encoding *bits,
                                           enum AVPixelFormat pix_fmt);

// Callback for AVCodecContext.get_buffer2 that allocates memory from
// persistently mapped buffers. This can be more efficient than regular
// system memory, especially on platforms that don't support importing
// PL_HANDLE_HOST_PTR as buffers.
//
// Note: `avctx->opaque` must be a pointer that *points* to the GPU instance.
// That is, it should have type `pl_gpu *`.
PL_LIBAV_API int pl_get_buffer2(AVCodecContext *avctx, AVFrame *pic, int flags);

// Mapping functions for the various libavutil enums. Note that these are not
// quite 1:1, and even for values that exist in both, the semantics sometimes
// differ. Some special cases (e.g. ICtCp, or XYZ) are handled differently in
// libplacebo and libavutil, respectively.
//
// Because of this, it's generally recommended to avoid these and instead use
// helpers like `pl_frame_from_avframe`, which contain extra logic to patch
// through all of the special cases.
PL_LIBAV_API enum pl_color_system pl_system_from_av(enum AVColorSpace spc);
PL_LIBAV_API enum AVColorSpace pl_system_to_av(enum pl_color_system sys);
PL_LIBAV_API enum pl_color_levels pl_levels_from_av(enum AVColorRange range);
PL_LIBAV_API enum AVColorRange pl_levels_to_av(enum pl_color_levels levels);
PL_LIBAV_API enum pl_color_primaries pl_primaries_from_av(enum AVColorPrimaries prim);
PL_LIBAV_API enum AVColorPrimaries pl_primaries_to_av(enum pl_color_primaries prim);
PL_LIBAV_API enum pl_color_transfer pl_transfer_from_av(enum AVColorTransferCharacteristic trc);
PL_LIBAV_API enum AVColorTransferCharacteristic pl_transfer_to_av(enum pl_color_transfer trc);
PL_LIBAV_API enum pl_chroma_location pl_chroma_from_av(enum AVChromaLocation loc);
PL_LIBAV_API enum AVChromaLocation pl_chroma_to_av(enum pl_chroma_location loc);
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(60, 11, 100)
PL_LIBAV_API enum pl_alpha_mode pl_alpha_from_av(enum AVAlphaMode mode);
PL_LIBAV_API enum AVAlphaMode pl_alpha_to_av(enum pl_alpha_mode mode);
#endif

// Helper function to generate a `pl_color_space` struct from an AVFrame.
PL_LIBAV_API void pl_color_space_from_avframe(struct pl_color_space *out_csp,
                                              const AVFrame *frame);

// Helper function to pick the right `pl_field` value for an AVFrame.
PL_LIBAV_API enum pl_field pl_field_from_avframe(const AVFrame *frame);

#ifdef PL_HAVE_LAV_FILM_GRAIN
// Fill in film grain parameters from an AVFilmGrainParams.
//
// Note: The resulting struct will only remain valid as long as the
// `AVFilmGrainParams` remains valid.
PL_LIBAV_API void pl_film_grain_from_av(struct pl_film_grain_data *out_data,
                                        const AVFilmGrainParams *fgp);
#endif

// Deprecated alias for backwards compatibility
#define pl_swapchain_colors_from_avframe pl_color_space_from_avframe

// Actual implementation, included as part of this header to avoid having
// a compile-time dependency on libavutil.
#if PL_LIBAV_IMPLEMENTATION
# include <libplacebo/utils/libav_internal.h>
#endif

PL_API_END

#endif // LIBPLACEBO_LIBAV_H_
