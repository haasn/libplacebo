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
#include <dav1d/dav1d.h>

#if defined(__cplusplus) && !defined(PL_DAV1D_IMPLEMENTATION)
# define PL_DAV1D_API
# define PL_DAV1D_IMPLEMENTATION 0
# warning Remember to include this file with a PL_DAV1D_IMPLEMENTATION set to 1 in \
          C translation unit to provide implementation. Suppress this warning by \
          defining PL_DAV1D_IMPLEMENTATION to 0 in C++ files.
#elif !defined(PL_DAV1D_IMPLEMENTATION)
# define PL_DAV1D_API static inline
# define PL_DAV1D_IMPLEMENTATION 1
#else
# define PL_DAV1D_API
#endif

PL_API_BEGIN

// Fill in the details of a `pl_frame` from a Dav1dPicture. This function will
// explicitly clear `out_frame`, setting all extra fields to 0. After this
// function returns, the only missing data is information related to the plane
// texture itself (`planes[N].texture`).
//
// Note: This will include all possible metadata, including HDR metadata and
// AV1 film grain data. Users should explicitly clear this out if undesired.
PL_DAV1D_API void pl_frame_from_dav1dpicture(struct pl_frame *out_frame,
                                             const Dav1dPicture *picture);

// Helper function to generate a `pl_color_space` struct from a Dav1dPicture.
// Useful to update the swapchain colorspace mode dynamically (e.g. for HDR).
PL_DAV1D_API void pl_swapchain_colors_from_dav1dpicture(struct pl_color_space *out_colors,
                                                        const Dav1dPicture *picture);

struct pl_dav1d_upload_params {
    // The picture to upload. Not modified unless `asynchronous` is true.
    Dav1dPicture *picture;

    // If true, film grain present in `picture` will be exported to the
    // `pl_frame` as well. This should be set to false unless the user has
    // disabled `Dav1dSettings.apply_grain`.
    bool film_grain;

    // If true, libplacebo will probe for the allocation metadata set by
    // `pl_allocate_dav1dpicture`, and directly import the attached buffers
    // (saving a memcpy in some cases). Has no effect if the Dav1dPicture was
    // not allocated using `pl_allocate_dav1dpicture`.
    //
    // Note: When this is the case, `asynchronous` has no further effect -
    // uploads from attached buffers are already asynchronous.
    bool gpu_allocated;

    // If true, `picture` will be asynchronously uploaded and unref'd
    // internally by libplacebo, and the struct passed by the user cleared to
    // {0}. This is needed to avoid `memcpy` in some cases, so setting it to
    // true is highly recommended wherever possible.
    //
    // Note: If `pl_upload_dav1dpicture` returns false, `picture` does not get
    // unref'd.
    bool asynchronous;
};

#define pl_dav1d_upload_params(...) (&(struct pl_dav1d_upload_params) { __VA_ARGS__ })

// Very high level helper function to take a `Dav1dPicture` and upload it to
// the GPU. Similar in spirit to `pl_upload_plane`, and the same notes apply.
// `tex` must be an array of 3 pointers of type `pl_tex`, each
// either pointing to a valid texture, or NULL. Returns whether successful.
PL_DAV1D_API bool pl_upload_dav1dpicture(pl_gpu gpu,
                                         struct pl_frame *out_frame, pl_tex tex[3],
                                         const struct pl_dav1d_upload_params *params);

// Allocate a Dav1dPicture from persistently mapped buffers. This can be more
// efficient than regular Dav1dPictures, especially when using the synchronous
// `pl_upload_dav1dpicture`, or on platforms that don't support importing
// PL_HANDLE_HOST_PTR as buffers. Returns 0 or a negative DAV1D_ERR value.
//
// Note: These may only be used directly as a Dav1dPicAllocator if the `gpu`
// passed as the value of `cookie` is `pl_gpu.limits.thread_safe`. Otherwise,
// the user must manually synchronize this to ensure it runs on the correct
// thread.
PL_DAV1D_API int pl_allocate_dav1dpicture(Dav1dPicture *picture, void *gpu);
PL_DAV1D_API void pl_release_dav1dpicture(Dav1dPicture *picture, void *gpu);

// Mapping functions for the various Dav1dColor* enums. Note that these are not
// quite 1:1, and even for values that exist in both, the semantics sometimes
// differ. Some special cases (e.g. ICtCp, or XYZ) are handled differently in
// libplacebo and libdav1d, respectively.
PL_DAV1D_API enum pl_color_system pl_system_from_dav1d(enum Dav1dMatrixCoefficients mc);
PL_DAV1D_API enum Dav1dMatrixCoefficients pl_system_to_dav1d(enum pl_color_system sys);
PL_DAV1D_API enum pl_color_levels pl_levels_from_dav1d(int color_range);
PL_DAV1D_API int pl_levels_to_dav1d(enum pl_color_levels levels);
PL_DAV1D_API enum pl_color_primaries pl_primaries_from_dav1d(enum Dav1dColorPrimaries prim);
PL_DAV1D_API enum Dav1dColorPrimaries pl_primaries_to_dav1d(enum pl_color_primaries prim);
PL_DAV1D_API enum pl_color_transfer pl_transfer_from_dav1d(enum Dav1dTransferCharacteristics trc);
PL_DAV1D_API enum Dav1dTransferCharacteristics pl_transfer_to_dav1d(enum pl_color_transfer trc);
PL_DAV1D_API enum pl_chroma_location pl_chroma_from_dav1d(enum Dav1dChromaSamplePosition loc);
PL_DAV1D_API enum Dav1dChromaSamplePosition pl_chroma_to_dav1d(enum pl_chroma_location loc);


// Actual implementation, included as part of this header to avoid having
// a compile-time dependency on libdav1d.
#if PL_DAV1D_IMPLEMENTATION
# include <libplacebo/utils/dav1d_internal.h>
#endif

PL_API_END

#endif // LIBPLACEBO_DAV1D_H_
