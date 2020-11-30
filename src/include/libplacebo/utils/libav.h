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

#include <libplacebo/gpu.h>
#include <libplacebo/utils/upload.h>
#include <libavutil/frame.h>

// Fill in the details of a `pl_frame` from an AVFrame. This function will
// explicitly clear `out_frame`, setting all extra fields to 0. After this
// function returns, the only missing data is information related to the plane
// texture itself (`planes[N].texture`), as well as any overlays (e.g.
// subtitles).
//
// Note: If the AVFrame contains an embedded ICC profile, the resulting
// `out_image->profile` will reference this pointer, meaning that in general,
// the `pl_frame` is only guaranteed to be valid as long as the AVFrame is not
// freed.
static void pl_frame_from_avframe(struct pl_frame *out_frame, const AVFrame *frame);

// Deprecated aliases for backwards compatibility
#define pl_image_from_avframe pl_frame_from_avframe
#define pl_target_from_avframe pl_frame_from_avframe

// Helper function to test if a pixfmt would be supported by the GPU.
// Essentially, this can be used to check if `pl_upload_avframe` would work for
// a given AVPixelFormat, without actually uploading or allocating anything.
static bool pl_test_pixfmt(const struct pl_gpu *gpu, enum AVPixelFormat pixfmt);

// Very high level helper function to take an `AVFrame` and upload it to the
// GPU. Similar in spirit to `pl_upload_plane`, and the same notes apply. `tex`
// must be an array of 4 pointers of type (const struct pl_tex *), each either
// pointing to a valid texture, or NULL. Returns whether successful.
//
// Note that this function will currently fail on HW accelerated AVFrame
// formats. For those, users must still use the specific interop functions from
// e.g. <libplacebo/vulkan.h>, depending on the HWAccel type.
static bool pl_upload_avframe(const struct pl_gpu *gpu,
                              struct pl_frame *out_frame,
                              const struct pl_tex *tex[4],
                              const AVFrame *frame);

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

#endif // LIBPLACEBO_LIBAV_H_
