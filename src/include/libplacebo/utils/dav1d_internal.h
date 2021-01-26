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

#ifndef LIBPLACEBO_DAV1D_H_
#error This header should be included as part of <libplacebo/utils/dav1d.h>
#else

#include <stdlib.h>

static inline enum pl_color_system pl_system_from_dav1d(enum Dav1dMatrixCoefficients mc)
{
    switch (mc) {
        case DAV1D_MC_IDENTITY:     return PL_COLOR_SYSTEM_RGB; // or XYZ (unlikely)
        case DAV1D_MC_BT709:        return PL_COLOR_SYSTEM_BT_709;
        case DAV1D_MC_UNKNOWN:      return PL_COLOR_SYSTEM_UNKNOWN;
        case DAV1D_MC_FCC:          return PL_COLOR_SYSTEM_UNKNOWN; // missing
        case DAV1D_MC_BT470BG:      return PL_COLOR_SYSTEM_BT_601;
        case DAV1D_MC_BT601:        return PL_COLOR_SYSTEM_BT_601;
        case DAV1D_MC_SMPTE240:     return PL_COLOR_SYSTEM_SMPTE_240M;
        case DAV1D_MC_SMPTE_YCGCO:  return PL_COLOR_SYSTEM_YCGCO;
        case DAV1D_MC_BT2020_NCL:   return PL_COLOR_SYSTEM_BT_2020_NC;
        case DAV1D_MC_BT2020_CL:    return PL_COLOR_SYSTEM_BT_2020_C;
        case DAV1D_MC_SMPTE2085:    return PL_COLOR_SYSTEM_UNKNOWN; // missing
        case DAV1D_MC_CHROMAT_NCL:  return PL_COLOR_SYSTEM_UNKNOWN; // missing
        case DAV1D_MC_CHROMAT_CL:   return PL_COLOR_SYSTEM_UNKNOWN; // missing
        // Note: this colorspace is confused between PQ and HLG, which dav1d
        // requires inferring from other sources, but libplacebo makes
        // explicit. Default to PQ as it's the more common scenario.
        case DAV1D_MC_ICTCP:        return PL_COLOR_SYSTEM_BT_2100_PQ;
        case DAV1D_MC_RESERVED: abort();
    }

    return PL_COLOR_SYSTEM_UNKNOWN;
}

static inline enum Dav1dMatrixCoefficients pl_system_to_dav1d(enum pl_color_system sys)
{
    switch (sys) {
    case PL_COLOR_SYSTEM_UNKNOWN:       return DAV1D_MC_UNKNOWN;
    case PL_COLOR_SYSTEM_BT_601:        return DAV1D_MC_BT601;
    case PL_COLOR_SYSTEM_BT_709:        return DAV1D_MC_BT709;
    case PL_COLOR_SYSTEM_SMPTE_240M:    return DAV1D_MC_SMPTE240;
    case PL_COLOR_SYSTEM_BT_2020_NC:    return DAV1D_MC_BT2020_NCL;
    case PL_COLOR_SYSTEM_BT_2020_C:     return DAV1D_MC_BT2020_CL;
    case PL_COLOR_SYSTEM_BT_2100_PQ:    return DAV1D_MC_ICTCP;
    case PL_COLOR_SYSTEM_BT_2100_HLG:   return DAV1D_MC_ICTCP;
    case PL_COLOR_SYSTEM_YCGCO:         return DAV1D_MC_SMPTE_YCGCO;
    case PL_COLOR_SYSTEM_RGB:           return DAV1D_MC_IDENTITY;
    case PL_COLOR_SYSTEM_XYZ:           return DAV1D_MC_IDENTITY;
    case PL_COLOR_SYSTEM_COUNT: abort();
    }

    return DAV1D_MC_UNKNOWN;
}

static inline enum pl_color_levels pl_levels_from_dav1d(int color_range)
{
    return color_range ? PL_COLOR_LEVELS_FULL : PL_COLOR_LEVELS_LIMITED;
}

static inline int pl_levels_to_dav1d(enum pl_color_levels levels)
{
    return levels == PL_COLOR_LEVELS_FULL;
}

static inline enum pl_color_primaries pl_primaries_from_dav1d(enum Dav1dColorPrimaries prim)
{
    switch (prim) {
    case DAV1D_COLOR_PRI_BT709:         return PL_COLOR_PRIM_BT_709;
    case DAV1D_COLOR_PRI_UNKNOWN:       return PL_COLOR_PRIM_UNKNOWN;
    case DAV1D_COLOR_PRI_RESERVED:      return PL_COLOR_PRIM_UNKNOWN;
    case DAV1D_COLOR_PRI_BT470M:        return PL_COLOR_PRIM_BT_470M;
    case DAV1D_COLOR_PRI_BT470BG:       return PL_COLOR_PRIM_BT_601_625;
    case DAV1D_COLOR_PRI_BT601:         return PL_COLOR_PRIM_BT_601_525;
    case DAV1D_COLOR_PRI_SMPTE240:      return PL_COLOR_PRIM_BT_601_525;
    case DAV1D_COLOR_PRI_FILM:          return PL_COLOR_PRIM_FILM_C;
    case DAV1D_COLOR_PRI_BT2020:        return PL_COLOR_PRIM_BT_2020;
    case DAV1D_COLOR_PRI_XYZ:           return PL_COLOR_PRIM_CIE_1931;
    case DAV1D_COLOR_PRI_SMPTE431:      return PL_COLOR_PRIM_DCI_P3;
    case DAV1D_COLOR_PRI_SMPTE432:      return PL_COLOR_PRIM_DISPLAY_P3;
    case DAV1D_COLOR_PRI_EBU3213:       return PL_COLOR_PRIM_EBU_3213;
    }

    return PL_COLOR_PRIM_UNKNOWN;
}

static inline enum Dav1dColorPrimaries pl_primaries_to_dav1d(enum pl_color_primaries prim)
{
    switch (prim) {
    case PL_COLOR_PRIM_UNKNOWN:     return DAV1D_COLOR_PRI_UNKNOWN;
    case PL_COLOR_PRIM_BT_601_525:  return DAV1D_COLOR_PRI_BT601;
    case PL_COLOR_PRIM_BT_601_625:  return DAV1D_COLOR_PRI_BT470BG;
    case PL_COLOR_PRIM_BT_709:      return DAV1D_COLOR_PRI_BT709;
    case PL_COLOR_PRIM_BT_470M:     return DAV1D_COLOR_PRI_BT470M;
    case PL_COLOR_PRIM_EBU_3213:    return DAV1D_COLOR_PRI_EBU3213;
    case PL_COLOR_PRIM_BT_2020:     return DAV1D_COLOR_PRI_BT2020;
    case PL_COLOR_PRIM_APPLE:       return DAV1D_COLOR_PRI_UNKNOWN; // missing
    case PL_COLOR_PRIM_ADOBE:       return DAV1D_COLOR_PRI_UNKNOWN; // missing
    case PL_COLOR_PRIM_PRO_PHOTO:   return DAV1D_COLOR_PRI_UNKNOWN; // missing
    case PL_COLOR_PRIM_CIE_1931:    return DAV1D_COLOR_PRI_XYZ;
    case PL_COLOR_PRIM_DCI_P3:      return DAV1D_COLOR_PRI_SMPTE431;
    case PL_COLOR_PRIM_DISPLAY_P3:  return DAV1D_COLOR_PRI_SMPTE432;
    case PL_COLOR_PRIM_V_GAMUT:     return DAV1D_COLOR_PRI_UNKNOWN; // missing
    case PL_COLOR_PRIM_S_GAMUT:     return DAV1D_COLOR_PRI_UNKNOWN; // missing
    case PL_COLOR_PRIM_FILM_C:      return DAV1D_COLOR_PRI_FILM;
    case PL_COLOR_PRIM_COUNT: abort();
    }

    return DAV1D_COLOR_PRI_UNKNOWN;
}

static inline enum pl_color_transfer pl_transfer_from_dav1d(enum Dav1dTransferCharacteristics trc)
{
    switch (trc) {
    case DAV1D_TRC_BT709:           return PL_COLOR_TRC_BT_1886; // EOTF != OETF
    case DAV1D_TRC_UNKNOWN:         return PL_COLOR_TRC_UNKNOWN;
    case DAV1D_TRC_BT470M:          return PL_COLOR_TRC_GAMMA22;
    case DAV1D_TRC_BT470BG:         return PL_COLOR_TRC_GAMMA28;
    case DAV1D_TRC_BT601:           return PL_COLOR_TRC_BT_1886; // EOTF != OETF
    case DAV1D_TRC_SMPTE240:        return PL_COLOR_TRC_BT_1886; // EOTF != OETF
    case DAV1D_TRC_LINEAR:          return PL_COLOR_TRC_LINEAR;
    case DAV1D_TRC_LOG100:          return PL_COLOR_TRC_UNKNOWN; // missing
    case DAV1D_TRC_LOG100_SQRT10:   return PL_COLOR_TRC_UNKNOWN; // missing
    case DAV1D_TRC_IEC61966:        return PL_COLOR_TRC_BT_1886; // EOTF != OETF
    case DAV1D_TRC_BT1361:          return PL_COLOR_TRC_BT_1886; // ETOF != OETF
    case DAV1D_TRC_SRGB:            return PL_COLOR_TRC_SRGB;
    case DAV1D_TRC_BT2020_10BIT:    return PL_COLOR_TRC_BT_1886; // EOTF != OETF
    case DAV1D_TRC_BT2020_12BIT:    return PL_COLOR_TRC_BT_1886; // EOTF != OETF
    case DAV1D_TRC_SMPTE2084:       return PL_COLOR_TRC_PQ;
    case DAV1D_TRC_SMPTE428:        return PL_COLOR_TRC_UNKNOWN; // missing
    case DAV1D_TRC_HLG:             return PL_COLOR_TRC_HLG;
    case DAV1D_TRC_RESERVED: abort();
    }

    return PL_COLOR_TRC_UNKNOWN;
}

static inline enum Dav1dTransferCharacteristics pl_transfer_to_dav1d(enum pl_color_transfer trc)
{
    switch (trc) {
    case PL_COLOR_TRC_UNKNOWN:      return DAV1D_TRC_UNKNOWN;
    case PL_COLOR_TRC_BT_1886:      return DAV1D_TRC_BT709;       // EOTF != OETF
    case PL_COLOR_TRC_SRGB:         return DAV1D_TRC_SRGB;
    case PL_COLOR_TRC_LINEAR:       return DAV1D_TRC_LINEAR;
    case PL_COLOR_TRC_GAMMA18:      return DAV1D_TRC_UNKNOWN; // missing
    case PL_COLOR_TRC_GAMMA22:      return DAV1D_TRC_BT470M;
    case PL_COLOR_TRC_GAMMA28:      return DAV1D_TRC_BT470BG;
    case PL_COLOR_TRC_PRO_PHOTO:    return DAV1D_TRC_UNKNOWN; // missing
    case PL_COLOR_TRC_PQ:           return DAV1D_TRC_SMPTE2084;
    case PL_COLOR_TRC_HLG:          return DAV1D_TRC_HLG;
    case PL_COLOR_TRC_V_LOG:        return DAV1D_TRC_UNKNOWN; // missing
    case PL_COLOR_TRC_S_LOG1:       return DAV1D_TRC_UNKNOWN; // missing
    case PL_COLOR_TRC_S_LOG2:       return DAV1D_TRC_UNKNOWN; // missing
    case PL_COLOR_TRC_COUNT: abort();
    }

    return DAV1D_TRC_UNKNOWN;
}

static inline enum pl_chroma_location pl_chroma_from_dav1d(enum Dav1dChromaSamplePosition loc)
{
    switch (loc) {
    case DAV1D_CHR_UNKNOWN:     return PL_CHROMA_UNKNOWN;
    case DAV1D_CHR_VERTICAL:    return PL_CHROMA_LEFT;
    case DAV1D_CHR_COLOCATED:   return PL_CHROMA_TOP_LEFT;
    }

    return PL_CHROMA_UNKNOWN;
}

static inline enum Dav1dChromaSamplePosition pl_chroma_to_dav1d(enum pl_chroma_location loc)
{
    switch (loc) {
    case PL_CHROMA_UNKNOWN:         return DAV1D_CHR_UNKNOWN;
    case PL_CHROMA_LEFT:            return DAV1D_CHR_VERTICAL;
    case PL_CHROMA_CENTER:          return DAV1D_CHR_UNKNOWN; // missing
    case PL_CHROMA_TOP_LEFT:        return DAV1D_CHR_COLOCATED;
    case PL_CHROMA_TOP_CENTER:      return DAV1D_CHR_UNKNOWN; // missing
    case PL_CHROMA_BOTTOM_LEFT:     return DAV1D_CHR_UNKNOWN; // missing
    case PL_CHROMA_BOTTOM_CENTER:   return DAV1D_CHR_UNKNOWN; // missing
    case PL_CHROMA_COUNT: abort();
    }

    return DAV1D_CHR_UNKNOWN;
}

static inline float pl_fixed24_8(uint32_t n)
{
    return (float) n / (1 << 8);
}

static inline void pl_frame_from_dav1dpicture(struct pl_frame *out,
                                              const Dav1dPicture *picture)
{
    const Dav1dSequenceHeader *seq_hdr = picture->seq_hdr;
    int num_planes;
    switch (picture->p.layout) {
    case DAV1D_PIXEL_LAYOUT_I400:
        num_planes = 1;
        break;
    case DAV1D_PIXEL_LAYOUT_I420:
    case DAV1D_PIXEL_LAYOUT_I422:
    case DAV1D_PIXEL_LAYOUT_I444:
        num_planes = 3;
        break;
    default: abort();
    }

    *out = (struct pl_frame) {
        .num_planes = num_planes,
        .planes = {
            // Components are always in order, which makes things easy
            {
                .components = 1,
                .component_mapping = {0},
            }, {
                .components = 1,
                .component_mapping = {1},
            }, {
                .components = 1,
                .component_mapping = {2},
            },
        },
        .crop = {
            0, 0, picture->p.w, picture->p.h,
        },
        .color = {
            .primaries = pl_primaries_from_dav1d(seq_hdr->pri),
            .transfer = pl_transfer_from_dav1d(seq_hdr->trc),
            .light = PL_COLOR_LIGHT_UNKNOWN,
        },
        .repr = {
            .sys = pl_system_from_dav1d(seq_hdr->mtrx),
            .levels = pl_levels_from_dav1d(seq_hdr->color_range),
            .bits.color_depth = picture->p.bpc,
        },
    };

    if (seq_hdr->mtrx == DAV1D_MC_ICTCP && seq_hdr->trc == DAV1D_TRC_HLG) {

        // dav1d makes no distinction between PQ and HLG ICtCp, so we need
        // to manually fix it in the case that we have HLG ICtCp data.
        out->repr.sys = PL_COLOR_SYSTEM_BT_2100_HLG;

    } else if (seq_hdr->mtrx == DAV1D_MC_IDENTITY &&
               seq_hdr->pri == DAV1D_COLOR_PRI_XYZ)
    {

        // dav1d handles this as a special case, but doesn't provide an
        // explicit flag for it either, so we have to resort to this ugly hack,
        // even though CIE 1931 RGB *is* a valid thing in principle!
        out->repr.sys= PL_COLOR_SYSTEM_XYZ;

    } else if (!out->repr.sys) {

        // PL_COLOR_SYSTEM_UNKNOWN maps to RGB, so hard-code this one
        out->repr.sys = pl_color_system_guess_ycbcr(picture->p.w, picture->p.h);
    }

    const Dav1dContentLightLevel *cll = picture->content_light;
    if (cll) {
        out->color.sig_peak = cll->max_content_light_level / PL_COLOR_SDR_WHITE;
        out->color.sig_avg = cll->max_frame_average_light_level / PL_COLOR_SDR_WHITE;
    }

    // This overrides the CLL values above, if both are present
    const Dav1dMasteringDisplay *md = picture->mastering_display;
    if (md)
        out->color.sig_peak = pl_fixed24_8(md->max_luminance) / PL_COLOR_SDR_WHITE;

    // Make sure this value is more or less legal
    if (out->color.sig_peak < 1.0 || out->color.sig_peak > 50.0)
        out->color.sig_peak = 0.0;

    if (picture->frame_hdr->film_grain.present) {
        const Dav1dFilmGrainData *fg = &picture->frame_hdr->film_grain.data;
        struct pl_av1_grain_data *av1 = &out->av1_grain;
        *av1 = (struct pl_av1_grain_data) {
            .grain_seed = fg->seed,
            .num_points_y = fg->num_y_points,
            .chroma_scaling_from_luma = fg->chroma_scaling_from_luma,
            .num_points_uv = { fg->num_uv_points[0], fg->num_uv_points[1] },
            .scaling_shift = fg->scaling_shift,
            .ar_coeff_lag = fg->ar_coeff_lag,
            .ar_coeff_shift = fg->ar_coeff_shift,
            .grain_scale_shift = fg->grain_scale_shift,
            .uv_mult = { fg->uv_mult[0], fg->uv_mult[1] },
            .uv_mult_luma = { fg->uv_luma_mult[0], fg->uv_luma_mult[1] },
            .uv_offset = { fg->uv_offset[0], fg->uv_offset[1] },
            .overlap = fg->overlap_flag,
        };

        memcpy(av1->points_y, fg->y_points, sizeof(av1->points_y));
        memcpy(av1->points_uv, fg->uv_points, sizeof(av1->points_uv));
        memcpy(av1->ar_coeffs_y, fg->ar_coeffs_y, sizeof(av1->ar_coeffs_y));
        memcpy(av1->ar_coeffs_uv[0], fg->ar_coeffs_uv[0], sizeof(av1->ar_coeffs_uv[0]));
        memcpy(av1->ar_coeffs_uv[1], fg->ar_coeffs_uv[1], sizeof(av1->ar_coeffs_uv[1]));
    }

    switch (picture->p.layout) {
    case DAV1D_PIXEL_LAYOUT_I400:
    case DAV1D_PIXEL_LAYOUT_I444:
        break;
    case DAV1D_PIXEL_LAYOUT_I420:
    case DAV1D_PIXEL_LAYOUT_I422:
        // Only set the chroma location for definitely subsampled images
        pl_frame_set_chroma_location(out, pl_chroma_from_dav1d(seq_hdr->chr));
        break;
    }
}

#define PL_MAGIC0 0x2c2a1269
#define PL_MAGIC1 0xc6d02577

struct pl_dav1dalloc {
    uint32_t magic[2];
    const struct pl_gpu *gpu;
    const struct pl_buf *buf;
};

struct pl_dav1dref {
    Dav1dPicture pic;
    uint8_t count;
};

static void pl_dav1dpicture_unref(void *priv)
{
    struct pl_dav1dref *ref = priv;
    if (--ref->count == 0) {
        dav1d_picture_unref(&ref->pic);
        free(ref);
    }
}

static bool pl_upload_dav1dpicture_internal(const struct pl_gpu *gpu,
                                            struct pl_frame *out,
                                            const struct pl_tex *tex[3],
                                            Dav1dPicture *pic,
                                            bool unref)
{
    pl_frame_from_dav1dpicture(out, pic);

    const int bytes = (pic->p.bpc + 7) / 8; // rounded up
    int sub_x = 0, sub_y = 0;
    switch (pic->p.layout) {
    case DAV1D_PIXEL_LAYOUT_I400:
    case DAV1D_PIXEL_LAYOUT_I444:
        break;
    case DAV1D_PIXEL_LAYOUT_I420:
        sub_x = sub_y = 1;
        break;
    case DAV1D_PIXEL_LAYOUT_I422:
        sub_x = 1;
        break;
    }

    struct pl_plane_data data[3] = {
        {
            // Y plane
            .type           = PL_FMT_UNORM,
            .width          = pic->p.w,
            .height         = pic->p.h,
            .pixel_stride   = bytes,
            .row_stride     = pic->stride[0],
            .component_size = {bytes * 8},
            .component_map  = {0},
        }, {
            // U plane
            .type           = PL_FMT_UNORM,
            .width          = pic->p.w >> sub_x,
            .height         = pic->p.h >> sub_y,
            .pixel_stride   = bytes,
            .row_stride     = pic->stride[1],
            .component_size = {bytes * 8},
            .component_map  = {1},
        }, {
            // V plane
            .type           = PL_FMT_UNORM,
            .width          = pic->p.w >> sub_x,
            .height         = pic->p.h >> sub_y,
            .pixel_stride   = bytes,
            .row_stride     = pic->stride[1],
            .component_size = {bytes * 8},
            .component_map  = {2},
        },
    };

    const struct pl_buf *buf = NULL;
    struct pl_dav1dalloc *alloc = pic->allocator_data;
    struct pl_dav1dref *ref = NULL;

    if (alloc && alloc->magic[0] == PL_MAGIC0 && alloc->magic[1] == PL_MAGIC1) {
        // Re-use pre-allocated buffers directly
        assert(alloc->gpu == gpu);
        buf = alloc->buf;
    } else if (unref) {
        ref = malloc(sizeof(*ref));
        if (!ref)
            return false;
        memcpy(&ref->pic, pic, sizeof(Dav1dPicture));
        ref->count = out->num_planes;
    }

    for (int p = 0; p < out->num_planes; p++) {
        if (buf) {
            data[p].buf = buf;
            data[p].buf_offset = (uintptr_t) pic->data[p] - (uintptr_t) buf->data;
        } else {
            data[p].pixels = pic->data[p];
            if (ref) {
                data[p].priv = ref;
                data[p].callback = pl_dav1dpicture_unref;
            }
        }

        if (!pl_upload_plane(gpu, &out->planes[p], &tex[p], &data[p])) {
            free(ref);
            return false;
        }
    }

    if (unref) {
        if (ref) {
            *pic = (Dav1dPicture) {0};
        } else {
            dav1d_picture_unref(pic);
        }
    }

    return true;
}

static inline bool pl_upload_dav1dpicture(const struct pl_gpu *gpu,
                                          struct pl_frame *out_frame,
                                          const struct pl_tex *tex[3],
                                          const Dav1dPicture *picture)
{
    return pl_upload_dav1dpicture_internal(gpu, out_frame, tex,
                                           (Dav1dPicture *) picture, false);
}

static inline bool pl_upload_and_unref_dav1dpicture(const struct pl_gpu *gpu,
                                                    struct pl_frame *out_frame,
                                                    const struct pl_tex *tex[3],
                                                    Dav1dPicture *picture)
{
    return pl_upload_dav1dpicture_internal(gpu, out_frame, tex, picture, true);
}

// Align to a power of 2
#define PL_ALIGN2(x, align) (((x) + (align) - 1) & ~((align) - 1))

static inline int pl_allocate_dav1dpicture(Dav1dPicture *p, const struct pl_gpu *gpu)
{
    if (!gpu->limits.max_buf_size)
        return DAV1D_ERR(ENOTSUP);

    // Copied from dav1d_default_picture_alloc
    const int hbd = p->p.bpc > 8;
    const int aligned_w = PL_ALIGN2(p->p.w, 128);
    const int aligned_h = PL_ALIGN2(p->p.h, 128);
    const int has_chroma = p->p.layout != DAV1D_PIXEL_LAYOUT_I400;
    const int ss_ver = p->p.layout == DAV1D_PIXEL_LAYOUT_I420;
    const int ss_hor = p->p.layout != DAV1D_PIXEL_LAYOUT_I444;
    p->stride[0] = aligned_w << hbd;
    p->stride[1] = has_chroma ? (aligned_w >> ss_hor) << hbd : 0;

    // Align strides up to multiples of the GPU performance hints
    p->stride[0] = PL_ALIGN2(p->stride[0], gpu->limits.align_tex_xfer_stride);
    p->stride[1] = PL_ALIGN2(p->stride[1], gpu->limits.align_tex_xfer_stride);

    // Aligning offsets to 4 also implicity aligns to the texel size (1 or 2)
    size_t off_align = PL_ALIGN2(gpu->limits.align_tex_xfer_offset, 4);
    const size_t y_sz = PL_ALIGN2(p->stride[0] * aligned_h, off_align);
    const size_t uv_sz = PL_ALIGN2(p->stride[1] * (aligned_h >> ss_ver), off_align);

    // The extra DAV1D_PICTURE_ALIGNMENTs are to brute force plane alignment,
    // even in the case that the driver gives us insane alignments
    const size_t pic_size = y_sz + 2 * uv_sz;
    const size_t total_size = pic_size + DAV1D_PICTURE_ALIGNMENT * 4;

    // Validate size limitations
    if (total_size > gpu->limits.max_buf_size)
        return DAV1D_ERR(ENOMEM);

    const struct pl_buf *buf = pl_buf_create(gpu, &(struct pl_buf_params) {
        .size = total_size,
        .host_mapped = true,
        .memory_type = PL_BUF_MEM_HOST,
    });

    if (!buf)
        return DAV1D_ERR(ENOMEM);

    struct pl_dav1dalloc *alloc = malloc(sizeof(struct pl_dav1dalloc));
    if (!alloc) {
        pl_buf_destroy(gpu, &buf);
        return DAV1D_ERR(ENOMEM);
    }

    *alloc = (struct pl_dav1dalloc) {
        .magic = { PL_MAGIC0, PL_MAGIC1 },
        .gpu = gpu,
        .buf = buf,
    };

    assert(buf->data);
    uintptr_t base = (uintptr_t) buf->data, data[3];
    data[0] = PL_ALIGN2(base, DAV1D_PICTURE_ALIGNMENT);
    data[1] = PL_ALIGN2(data[0] + y_sz, DAV1D_PICTURE_ALIGNMENT);
    data[2] = PL_ALIGN2(data[1] + uv_sz, DAV1D_PICTURE_ALIGNMENT);

    p->allocator_data = alloc;
    p->data[0] = (void *) data[0];
    p->data[1] = (void *) data[1];
    p->data[2] = (void *) data[2];
    return 0;
}

static inline void pl_release_dav1dpicture(Dav1dPicture *p, const struct pl_gpu *gpu)
{
    struct pl_dav1dalloc *alloc = p->allocator_data;
    if (!alloc)
        return;

    assert(alloc->magic[0] == PL_MAGIC0);
    assert(alloc->magic[1] == PL_MAGIC1);
    assert(alloc->gpu == gpu);
    pl_buf_destroy(alloc->gpu, &alloc->buf);
    free(alloc);

    p->data[0] = p->data[1] = p->data[2] = p->allocator_data = NULL;
}

#undef PL_ALIGN2
#undef PL_MAGIC0
#undef PL_MAGIC1

#endif // LIBPLACEBO_DAV1D_H_
