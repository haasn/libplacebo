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

#ifndef LIBPLACEBO_DAV1D_H_
#define LIBPLACEBO_DAV1D_H_

#include <libplacebo/gpu.h>
#include <libplacebo/utils/upload.h>
#include <dav1d/picture.h>

// Fill in the details of a `pl_frame` from a Dav1dPicture. This function will
// explicitly clear `out_frame`, setting all extra fields to 0. After this
// function returns, the only missing data is information related to the plane
// texture itself (`planes[N].texture`).
static void pl_frame_from_dav1dpicture(struct pl_frame *out_frame,
                                       const Dav1dPicture *picture);

// Very high level helper function to take a `Dav1dPicture` and upload it to
// the GPU. Similar in spirit to `pl_upload_plane`, and the same notes apply.
// `tex` must be an array of 3 pointers of type (const struct pl_tex *), each
// either pointing to a valid texture, or NULL. Returns whether successful.
//
// Note: This is less efficient than `pl_upload_and_unref_dav1dpicture`!
static bool pl_upload_dav1dpicture(const struct pl_gpu *gpu,
                                   struct pl_frame *out_frame,
                                   const struct pl_tex *tex[3],
                                   const Dav1dPicture *picture);

// Variant of `pl_upload_dav1dpicture` that also unrefs the picture after
// uploading. This is needed to take advantage of internal asynchronous
// operations. `picture` will be cleared to {0}, and the internally referenced
// buffers unref'd at some arbitrary time in the future.
//
// Note: If this fails (returns false), the Dav1dPicture does not get unref'd.
static bool pl_upload_and_unref_dav1dpicture(const struct pl_gpu *gpu,
                                             struct pl_frame *out_frame,
                                             const struct pl_tex *tex[3],
                                             Dav1dPicture *picture);

// Allocate a Dav1dPicture from persistently mapped buffers. This can be more
// efficient than regular Dav1dPictures, especially when using the synchronous
// `pl_upload_dav1dpicture`, or on platforms that don't support importing
// PL_HANDLE_HOST_PTR as buffers. Returns 0 or a negative DAV1D_ERR value.
//
// Note: These are *not* thread-safe, and should not be used directly as a
// Dav1dPicAllocator unless wrapped by a thread-safe layer.
static int pl_allocate_dav1dpicture(Dav1dPicture *picture, const struct pl_gpu *gpu);
static void pl_release_dav1dpicture(Dav1dPicture *picture, const struct pl_gpu *gpu);

// Mapping functions for the various Dav1dColor* enums. Note that these are not
// quite 1:1, and even for values that exist in both, the semantics sometimes
// differ. Some special cases (e.g. ICtCp, or XYZ) are handled differently in
// libplacebo and libdav1d, respectively.
static enum pl_color_system pl_system_from_dav1d(enum Dav1dMatrixCoefficients mc);
static enum Dav1dMatrixCoefficients pl_system_to_dav1d(enum pl_color_system sys);
static enum pl_color_levels pl_levels_from_dav1d(int color_range);
static int pl_levels_to_dav1d(enum pl_color_levels levels);
static enum pl_color_primaries pl_primaries_from_dav1d(enum Dav1dColorPrimaries prim);
static enum Dav1dColorPrimaries pl_primaries_to_dav1d(enum pl_color_primaries prim);
static enum pl_color_transfer pl_transfer_from_dav1d(enum Dav1dTransferCharacteristics trc);
static enum Dav1dTransferCharacteristics pl_transfer_to_dav1d(enum pl_color_transfer trc);
static enum pl_chroma_location pl_chroma_from_dav1d(enum Dav1dChromaSamplePosition loc);
static enum Dav1dChromaSamplePosition pl_chroma_to_dav1d(enum pl_chroma_location loc);

// Actual implementation, included as part of this header to avoid having
// a compile-time dependency on libdav1d.
#include <libplacebo/utils/dav1d_internal.h>

#endif // LIBPLACEBO_DAV1D_H_
