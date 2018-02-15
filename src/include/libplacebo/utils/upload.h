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

#include <stdint.h>

#include <libplacebo/gpu.h>
#include <libplacebo/renderer.h>

#ifndef LIBPLACEBO_UPLOAD_H_
#define LIBPLACEBO_UPLOAD_H_

// This file contains a utility function to assist in uploading data from host
// memory to a texture. In particular, the texture will be suitable for use as
// a `pl_plane`.

// Description of the host representation of an image plane
struct pl_plane_data {
    enum pl_fmt_type type; // meaning of the data (must not be UINT or SINT)
    int width, height;     // dimensions of the plane
    int component_size[4]; // size in bits of each coordinate
    int component_pad[4];  // ignored bits preceding each component
    int component_map[4];  // semantic meaning of each component (pixel order)
    size_t pixel_stride;   // offset in bytes between pixels (required)
    size_t row_stride;     // offset in bytes between rows (optional)
    const void *pixels;    // the actual data underlying this plane

    // Note: When using this together with `pl_image`, there is some amount of
    // overlap between `component_pad` and `pl_color_repr.bits`. Some key
    // differences between the two:
    //
    // - the bits from `component_pad` are ignored; whereas the superfluous bits
    //   in a `pl_color_repr` must be 0.
    // - the `component_pad` exists to align the component size and placement
    //   with the capabilities of GPUs; the `pl_color_repr` exists to control
    //   the semantics of the color samples on a finer granularity.
    // - the `pl_color_repr` applies to the color sample as a whole, and
    //   therefore applies to all planes; the `component_pad` can be different
    //   for each plane.
    // - `component_pad` interacts with float textures by moving the actual
    //   float in memory. `pl_color_repr` interacts with float data as if
    //   the float was converted from an integer under full range semantics.
    //
    // To help establish the motivating difference, a typical example of a use
    // case would be yuv420p10. Since 10-bit GPU texture support is limited,
    // and working with non-byte-aligned pixels is awkward in general, the
    // convention is to represent yuv420p10 as 16-bit samples with either the
    // high or low bits set to 0. In this scenario, the `component_size` of the
    // `pl_plane_repr` and `pl_bit_encoding.sample_depth` would be 16, while
    // the `pl_bit_encoding.color_depth` would be 10 (and additionally, the
    // `pl_bit_encoding.bit_shift` would be either 0 or  6, depending on
    // whether the low or the high bits are used).
    //
    // On the contrary, something like a packed, 8-bit XBGR format (where the
    // X bits are ignored and may contain garbage) would set `component_pad[0]`
    // to 8, and the component_size[0:2] (respectively) to 8 as well.
    //
    // As a general rule of thumb, for maximum compatibility, you should try
    // and align component_size/component_pad to multiples of 8 and explicitly
    // clear any remaining superfluous bits (+ use `pl_color_repr.bits` to
    // ensure they're decoded correctly). You should also try to align the
    // `pixel_stride` to a power of two.
};

// Fills in the `component_size`, `component_pad` and `component_map` fields
// based on the supplied mask for each component (in semantic order, i.e.
// RGBA). If `mask` does not have a contiguous range of set bits, then the
// result is undefined and probably not useful.
void pl_plane_data_from_mask(struct pl_plane_data *data, uint64_t mask[4]);

// Helper function to find a suitable `pl_fmt` based on a pl_plane_data's
// requirements. This is called internally by `pl_upload_plane`, but it's
// exposed to users both as a convenience and so they may pre-emptively check
// if a format would be supported without actually having to attempt the upload.
const struct pl_fmt *pl_plane_find_fmt(const struct pl_gpu *gpu, int out_map[4],
                                       const struct pl_plane_data *data);

// Upload an image plane to a texture, and output the resulting `pl_plane`
// struct. `tex` must be a valid pointer to a texture (or NULL), which will be
// destroyed and reinitialized if it does not already exist or is incompatible.
// Returns whether successful.
//
// The resulting texture is guaranteed to be `sampleable`, and it will also try
// and maximize compatibility with the other `pl_renderer` requirements
// (blittable, linear filterable, etc.).
bool pl_upload_plane(const struct pl_gpu *gpu, struct pl_plane *out_plane,
                     const struct pl_tex **tex, const struct pl_plane_data *data);

#endif // LIBPLACEBO_UPLOAD_H_
