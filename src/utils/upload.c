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
 * License along with libplacebo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "log.h"
#include "common.h"
#include "gpu.h"

#include <libplacebo/utils/upload.h>

#define MAX_COMPS 4

struct comp {
    int order; // e.g. 0, 1, 2, 3 for RGBA
    int size;  // size in bits
    int shift; // bit-shift / offset in bits
};

static int compare_comp(const void *pa, const void *pb)
{
    const struct comp *a = pa, *b = pb;

    // Move all of the components with a size of 0 to the end, so they can
    // be ignored outright
    if (a->size && !b->size)
        return -1;
    if (b->size && !a->size)
        return 1;

    // Otherwise, just compare based on the shift
    return PL_CMP(a->shift, b->shift);
}

void pl_plane_data_from_comps(struct pl_plane_data *data, int size[4],
                              int shift[4])
{
    struct comp comps[MAX_COMPS];
    for (int i = 0; i < PL_ARRAY_SIZE(comps); i++) {
        comps[i].order = i;
        comps[i].size = size[i];
        comps[i].shift = shift[i];
    }

    // Sort the components by shift
    qsort(comps, MAX_COMPS, sizeof(struct comp), compare_comp);

    // Generate the resulting component size/pad/map
    int offset = 0;
    for (int i = 0; i < MAX_COMPS; i++)  {
        if (comps[i].size) {
            assert(comps[i].shift >= offset);
            data->component_size[i] = comps[i].size;
            data->component_pad[i] = comps[i].shift - offset;
            data->component_map[i] = comps[i].order;
            offset += data->component_size[i] + data->component_pad[i];
        } else {
            // Clear the superfluous entries for sanity
            data->component_size[i] = 0;
            data->component_pad[i] = 0;
            data->component_map[i] = 0;
        }
    }
}

void pl_plane_data_from_mask(struct pl_plane_data *data, uint64_t mask[4])
{
    int size[4];
    int shift[4];

    for (int i = 0; i < PL_ARRAY_SIZE(size); i++) {
        size[i] = __builtin_popcountll(mask[i]);
        shift[i] = PL_MAX(0, __builtin_ffsll(mask[i]) - 1);

        // Sanity checking
        uint64_t mask_reconstructed = (1LLU << size[i]) - 1;
        mask_reconstructed <<= shift[i];
        pl_assert(mask_reconstructed == mask[i]);
    }

    pl_plane_data_from_comps(data, size, shift);
}

bool pl_plane_data_align(struct pl_plane_data *data, struct pl_bit_encoding *out_bits)
{
    struct pl_plane_data aligned = *data;
    struct pl_bit_encoding bits = {0};

    int offset = 0;

#define SET_TEST(var, value)                \
    do {                                    \
        if (offset == 0) {                  \
            (var) = (value);                \
        } else if ((var) != (value)) {      \
            goto misaligned;                \
        }                                   \
    } while (0)

    for (int i = 0; i < MAX_COMPS; i++) {
        if (!aligned.component_size[i])
            break;

        // Can't meaningfully align alpha channel, so just skip it. This is a
        // limitation of the fact that `pl_bit_encoding` only applies to the
        // main color channels, and changing this would be very nontrivial.
        if (aligned.component_map[i] == PL_CHANNEL_A)
            continue;

        // Color depth is the original component size, before alignment
        SET_TEST(bits.color_depth, aligned.component_size[i]);

        // Try consuming padding of the current component to align down. This
        // corresponds to an extra bit shift to the left.
        int comp_start = offset + aligned.component_pad[i];
        int left_delta = comp_start - PL_ALIGN2(comp_start - 7, 8);
        left_delta = PL_MIN(left_delta, aligned.component_pad[i]);
        aligned.component_pad[i] -= left_delta;
        aligned.component_size[i] += left_delta;
        SET_TEST(bits.bit_shift, left_delta);

        // Try consuming padding of the next component to align up. This
        // corresponds to simply ignoring some extra 0s on the end.
        int comp_end = comp_start + aligned.component_size[i] - left_delta;
        int right_delta = PL_ALIGN2(comp_end, 8) - comp_end;
        if (i+1 == MAX_COMPS || !aligned.component_size[i+1]) {
            // This is the last component, so we can be greedy
            aligned.component_size[i] += right_delta;
        } else {
            right_delta = PL_MIN(right_delta, aligned.component_pad[i+1]);
            aligned.component_pad[i+1] -= right_delta;
            aligned.component_size[i] += right_delta;
        }

        // Sample depth is the new total component size, including padding
        SET_TEST(bits.sample_depth, aligned.component_size[i]);

        offset += aligned.component_pad[i] + aligned.component_size[i];
    }

    // Easy sanity check, to make sure that we don't exceed the known stride
    if (aligned.pixel_stride && offset > aligned.pixel_stride * 8)
        goto misaligned;

    *data = aligned;
    if (out_bits)
        *out_bits = bits;
    return true;

misaligned:
    // Can't properly align anything, so just do a no-op
    if (out_bits)
        *out_bits = (struct pl_bit_encoding) {0};
    return false;
}

pl_fmt pl_plane_find_fmt(pl_gpu gpu, int out_map[4], const struct pl_plane_data *data)
{
    int dummy[4] = {0};
    out_map = PL_DEF(out_map, dummy);

    // Endian swapping requires compute shaders (currently)
    if (data->swapped && !gpu->limits.max_ssbo_size)
        return NULL;

    // Count the number of components and initialize out_map
    int num = 0;
    for (int i = 0; i < PL_ARRAY_SIZE(data->component_size); i++) {
        out_map[i] = -1;
        if (data->component_size[i])
            num = i+1;
    }

    for (int n = 0; n < gpu->num_formats; n++) {
        pl_fmt fmt = gpu->formats[n];
        if (fmt->opaque || fmt->num_components < num)
            continue;
        if (fmt->type != data->type || fmt->texel_size != data->pixel_stride)
            continue;
        if (!(fmt->caps & PL_FMT_CAP_SAMPLEABLE))
            continue;

        int idx = 0;

        // Try mapping all pl_plane_data components to texture components
        for (int i = 0; i < num; i++) {
            // If there's padding we have to map it to an unused physical
            // component first
            int pad = data->component_pad[i];
            if (pad && (idx >= 4 || fmt->host_bits[idx++] != pad))
                goto next_fmt;

            // Otherwise, try and match this component
            int size = data->component_size[i];
            if (size && (idx >= 4 || fmt->host_bits[idx] != size))
                goto next_fmt;
            out_map[idx++] = data->component_map[i];
        }

        // Reject misaligned formats, check this last to only log such errors
        // if this is the only thing preventing a format from being used, as
        // this is likely an issue in the API usage.
        if (data->row_stride % fmt->texel_align) {
            PL_WARN(gpu, "Rejecting texture format '%s' due to misalignment: "
                    "Row stride %zu is not a clean multiple of texel size %zu! "
                    "This is likely an API usage bug.",
                    fmt->name, data->row_stride, fmt->texel_align);
            continue;
        }

        return fmt;

next_fmt: ; // acts as `continue`
    }

    return NULL;
}

bool pl_upload_plane(pl_gpu gpu, struct pl_plane *out_plane,
                     pl_tex *tex, const struct pl_plane_data *data)
{
    pl_assert(!data->buf ^ !data->pixels); // exactly one

    int out_map[4];
    pl_fmt fmt = pl_plane_find_fmt(gpu, out_map, data);
    if (!fmt) {
        PL_ERR(gpu, "Failed picking any compatible texture format for a plane!");
        return false;

        // TODO: try soft-converting to a supported format using e.g zimg?
    }

    bool ok = pl_tex_recreate(gpu, tex, pl_tex_params(
        .w = data->width,
        .h = data->height,
        .format = fmt,
        .sampleable = true,
        .host_writable = true,
        .blit_src = fmt->caps & PL_FMT_CAP_BLITTABLE,
    ));

    if (!ok) {
        PL_ERR(gpu, "Failed initializing plane texture!");
        return false;
    }

    if (out_plane) {
        out_plane->texture = *tex;
        out_plane->components = 0;
        for (int i = 0; i < PL_ARRAY_SIZE(out_map); i++) {
            out_plane->component_mapping[i] = out_map[i];
            if (out_map[i] >= 0)
                out_plane->components = i+1;
        }
    }

    struct pl_tex_transfer_params params = {
        .tex        = *tex,
        .rc.x1      = data->width, // set these for `pl_tex_transfer_size`
        .rc.y1      = data->height,
        .rc.z1      = 1,
        .row_pitch  = PL_DEF(data->row_stride, data->width * fmt->texel_size),
        .ptr        = (void *) data->pixels,
        .buf        = data->buf,
        .buf_offset = data->buf_offset,
        .callback   = data->callback,
        .priv       = data->priv,
    };

    pl_buf swapbuf = NULL;
    if (data->swapped) {
        const size_t aligned = PL_ALIGN2(pl_tex_transfer_size(&params), 4);
        swapbuf = pl_buf_create(gpu, pl_buf_params(
            .size           = aligned,
            .storable       = true,
            .initial_data   = params.ptr,

            // Note: This may over-read from `ptr` if `ptr` is not aligned to a
            // word boundary, but the extra texels will be ignored by
            // `pl_tex_upload` so this UB should be a non-issue in practice.
        ));
        if (!swapbuf) {
            PL_ERR(gpu, "Failed creating endian swapping buffer!");
            return false;
        }

        struct pl_buf_copy_swap_params swap_params = {
            .src        = swapbuf,
            .dst        = swapbuf,
            .size       = aligned,
            .wordsize   = fmt->texel_size / fmt->num_components,
        };

        bool can_reuse = params.buf && params.buf->params.storable &&
                         params.buf_offset % 4 == 0 &&
                         params.buf_offset + aligned <= params.buf->params.size;

        if (params.ptr) {
            // Data is already uploaded (no-op), can swap in-place
        } else if (can_reuse) {
            // We can sample directly from the source buffer
            swap_params.src = params.buf;
            swap_params.src_offset = params.buf_offset;
        } else {
            // We sadly need to do a second memcpy
            assert(params.buf);
            PL_TRACE(gpu, "Double-slow path! pl_buf_copy -> pl_buf_copy_swap...");
            pl_buf_copy(gpu, swapbuf, 0, params.buf, params.buf_offset,
                        PL_MIN(aligned, params.buf->params.size - params.buf_offset));
        }

        if (!pl_buf_copy_swap(gpu, &swap_params)) {
            PL_ERR(gpu, "Failed swapping endianness!");
            pl_buf_destroy(gpu, &swapbuf);
            return false;
        }

        params.ptr = NULL;
        params.buf = swapbuf;
        params.buf_offset = 0;
    }

    ok = pl_tex_upload(gpu, &params);
    pl_buf_destroy(gpu, &swapbuf);
    return ok;
}

bool pl_recreate_plane(pl_gpu gpu, struct pl_plane *out_plane,
                       pl_tex *tex, const struct pl_plane_data *data)
{
    if (data->swapped) {
        PL_ERR(gpu, "Cannot call pl_recreate_plane on non-native endian plane "
               "data, this is only supported for `pl_upload_plane`!");
        return false;
    }

    int out_map[4];
    pl_fmt fmt = pl_plane_find_fmt(gpu, out_map, data);
    if (!fmt) {
        PL_ERR(gpu, "Failed picking any compatible texture format for a plane!");
        return false;
    }

    bool ok = pl_tex_recreate(gpu, tex, pl_tex_params(
        .w = data->width,
        .h = data->height,
        .format = fmt,
        .renderable = true,
        .host_readable = fmt->caps & PL_FMT_CAP_HOST_READABLE,
        .blit_dst = fmt->caps & PL_FMT_CAP_BLITTABLE,
        .storable = fmt->caps & PL_FMT_CAP_STORABLE,
    ));

    if (!ok) {
        PL_ERR(gpu, "Failed initializing plane texture!");
        return false;
    }

    if (out_plane) {
        out_plane->texture = *tex;
        out_plane->components = 0;
        for (int i = 0; i < PL_ARRAY_SIZE(out_map); i++) {
            out_plane->component_mapping[i] = out_map[i];
            if (out_map[i] >= 0)
                out_plane->components = i+1;
        }
    }

    return true;
}
