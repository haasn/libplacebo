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

// Note: Support for various hardware decoding formats depends on support
// being available in FFmpeg. To ensure compatibility with this header, please
// make sure to include, if possible, the following headers, *before* including
// this one:
//
// - <libavutil/hwcontext_vulkan.h> (for AV_PIX_FMT_VULKAN)

#include <libplacebo/gpu.h>
#include <libplacebo/utils/upload.h>

PL_API_BEGIN

#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavcodec/avcodec.h>

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
static void pl_frame_from_avframe(struct pl_frame *out_frame, const AVFrame *frame);

// Deprecated aliases for backwards compatibility
#define pl_image_from_avframe pl_frame_from_avframe
#define pl_target_from_avframe pl_frame_from_avframe

// Copy extra metadata from an AVStream to a pl_frame. This should be called
// after `pl_frame_from_avframe` or `pl_upload_avframe` (respectively), and
// sets metadata associated with stream-level side data. This is needed because
// FFmpeg rather annoyingly does not propagate stream-level metadata to frames.
static void pl_frame_copy_stream_props(struct pl_frame *out_frame,
                                       const AVStream *stream);

// Helper function to generate a `pl_swapchain_colors` struct from an AVFrame.
// Useful to update the swapchain colorspace mode dynamically (e.g. for HDR).
static void pl_swapchain_colors_from_avframe(struct pl_swapchain_colors *out_colors,
                                             const AVFrame *frame);

// Helper function to test if a pixfmt would be supported by the GPU.
// Essentially, this can be used to check if `pl_upload_avframe` would work for
// a given AVPixelFormat, without actually uploading or allocating anything.
static bool pl_test_pixfmt(pl_gpu gpu, enum AVPixelFormat pixfmt);

// Like `pl_frame_from_avframe`, but the texture pointers are also initialized
// to ensure they have the correct size and format to match the AVframe.
// Similar in spirit to `pl_recreate_plane`, and the same notes apply. `tex`
// must be an array of 4 pointers of type `pl_tex`, each either
// pointing to a valid texture, or NULL. Returns whether successful.
static bool pl_frame_recreate_from_avframe(pl_gpu gpu, struct pl_frame *out_frame,
                                           pl_tex tex[4], const AVFrame *frame);

// Very high level helper function to take an `AVFrame` and map it to the GPU.
// The resulting `pl_frame` remains valid until `pl_unmap_avframe` is called,
// which must be called at some point to clean up state. The `AVFrame` is
// automatically ref'd and unref'd if needed. Returns whether successful.
//
// `tex` must point to an array of four textures (or NULL), which will be used
// as backing storage for frames if possible. (Note that hwaccel-mapped frames
// will never use this backing storage)
//
// Note: `out_frame->user_data` will hold a reference to the AVFrame
// corresponding to the `pl_frame`. It will automatically be unref'd by
// `pl_unmap_avframe`.
//
// Note: `pl_unmap_avframe` will not clean up any textures in `tex`.
static bool pl_map_avframe(pl_gpu gpu, struct pl_frame *out_frame,
                           pl_tex tex[4], const AVFrame *avframe);
static void pl_unmap_avframe(pl_gpu gpu, struct pl_frame *frame);

// Deprecated variant of `pl_map_frame`, with the following differences:
// - Does not support hardware-accelerated frames
// - Does not require manual unmapping
// - Does not touch `frame->user_data`.
// - `frame` must not be freed by the user before `frame` is done being used
static PL_DEPRECATED bool pl_upload_avframe(pl_gpu gpu, struct pl_frame *out_frame,
                                            pl_tex tex[4], const AVFrame *frame);

// Download the texture contents of a `pl_frame` back to a corresponding
// AVFrame. Blocks until completion.
//
// Note: This function performs minimal verification, so incorrect usage will
// likely result in broken frames. Use `pl_frame_recreate_from_avframe` to
// ensure matching formats.
static bool pl_download_avframe(pl_gpu gpu,
                                const struct pl_frame *frame,
                                AVFrame *out_frame);

// Helper functions to update the colorimetry data in an AVFrame based on
// the values specified in the given color space / color repr / profile.
//
// Note: These functions can and will allocate AVFrame side data if needed,
// in particular to encode `space.sig_peak` etc.
static void pl_avframe_set_color(AVFrame *frame, struct pl_color_space space);
static void pl_avframe_set_repr(AVFrame *frame, struct pl_color_repr repr);
static void pl_avframe_set_profile(AVFrame *frame, struct pl_icc_profile profile);

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
static int pl_plane_data_from_pixfmt(struct pl_plane_data data[4],
                                     struct pl_bit_encoding *bits,
                                     enum AVPixelFormat pix_fmt);

// Callback for AVCodecContext.get_buffer2 that allocates memory from
// persistently mapped buffers. This can be more efficient than regular
// system memory, especially on platforms that don't support importing
// PL_HANDLE_HOST_PTR as buffers.
//
// Note: `avctx->opaque` must be a pointer that *points* to the GPU instance.
// That is, it should have type `pl_gpu *`.
static int pl_get_buffer2(AVCodecContext *avctx, AVFrame *pic, int flags);

// Mapping functions for the various libavutil enums. Note that these are not
// quite 1:1, and even for values that exist in both, the semantics sometimes
// differ. Some special cases (e.g. ICtCp, or XYZ) are handled differently in
// libplacebo and libavutil, respectively.
//
// Because of this, it's generally recommended to avoid these and instead use
// helpers like `pl_frame_from_avframe`, which contain extra logic to patch
// through all of the special cases.
static enum pl_color_system pl_system_from_av(enum AVColorSpace spc);
static enum AVColorSpace pl_system_to_av(enum pl_color_system sys);
static enum pl_color_levels pl_levels_from_av(enum AVColorRange range);
static enum AVColorRange pl_levels_to_av(enum pl_color_levels levels);
static enum pl_color_primaries pl_primaries_from_av(enum AVColorPrimaries prim);
static enum AVColorPrimaries pl_primaries_to_av(enum pl_color_primaries prim);
static enum pl_color_transfer pl_transfer_from_av(enum AVColorTransferCharacteristic trc);
static enum AVColorTransferCharacteristic pl_transfer_to_av(enum pl_color_transfer trc);
static enum pl_chroma_location pl_chroma_from_av(enum AVChromaLocation loc);
static enum AVChromaLocation pl_chroma_to_av(enum pl_chroma_location loc);

// Actual implementation, included as part of this header to avoid having
// a compile-time dependency on libavutil.
#include <libplacebo/utils/libav_internal.h>

PL_API_END

#endif // LIBPLACEBO_LIBAV_H_
