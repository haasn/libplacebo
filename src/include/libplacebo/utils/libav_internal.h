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

#ifndef LIBPLACEBO_LIBAV_H_
# error This header should be included as part of <libplacebo/utils/libav.h>
#else

#include <assert.h>

#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/imgutils.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/pixdesc.h>
#include <libavutil/display.h>

#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(56, 61, 100)
# define HAVE_LAV_FILM_GRAIN
# include <libavutil/film_grain_params.h>
#endif

#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 8, 100)
# define HAVE_LAV_VULKAN
# include <libavutil/hwcontext_vulkan.h>
# include <libplacebo/vulkan.h>
#endif

static inline enum pl_color_system pl_system_from_av(enum AVColorSpace spc)
{
    switch (spc) {
    case AVCOL_SPC_RGB:                 return PL_COLOR_SYSTEM_RGB;
    case AVCOL_SPC_BT709:               return PL_COLOR_SYSTEM_BT_709;
    case AVCOL_SPC_UNSPECIFIED:         return PL_COLOR_SYSTEM_UNKNOWN;
    case AVCOL_SPC_RESERVED:            return PL_COLOR_SYSTEM_UNKNOWN;
    case AVCOL_SPC_FCC:                 return PL_COLOR_SYSTEM_UNKNOWN; // missing
    case AVCOL_SPC_BT470BG:             return PL_COLOR_SYSTEM_BT_601;
    case AVCOL_SPC_SMPTE170M:           return PL_COLOR_SYSTEM_BT_601;
    case AVCOL_SPC_SMPTE240M:           return PL_COLOR_SYSTEM_SMPTE_240M;
    case AVCOL_SPC_YCGCO:               return PL_COLOR_SYSTEM_YCGCO;
    case AVCOL_SPC_BT2020_NCL:          return PL_COLOR_SYSTEM_BT_2020_NC;
    case AVCOL_SPC_BT2020_CL:           return PL_COLOR_SYSTEM_BT_2020_C;
    case AVCOL_SPC_SMPTE2085:           return PL_COLOR_SYSTEM_UNKNOWN; // missing
    case AVCOL_SPC_CHROMA_DERIVED_NCL:  return PL_COLOR_SYSTEM_UNKNOWN; // missing
    case AVCOL_SPC_CHROMA_DERIVED_CL:   return PL_COLOR_SYSTEM_UNKNOWN; // missing
    // Note: this colorspace is confused between PQ and HLG, which libav*
    // requires inferring from other sources, but libplacebo makes explicit.
    // Default to PQ as it's the more common scenario.
    case AVCOL_SPC_ICTCP:               return PL_COLOR_SYSTEM_BT_2100_PQ;
    case AVCOL_SPC_NB:                  return PL_COLOR_SYSTEM_COUNT;
    }

    return PL_COLOR_SYSTEM_UNKNOWN;
}

static inline enum AVColorSpace pl_system_to_av(enum pl_color_system sys)
{
    switch (sys) {
    case PL_COLOR_SYSTEM_UNKNOWN:       return AVCOL_SPC_UNSPECIFIED;
    case PL_COLOR_SYSTEM_BT_601:        return AVCOL_SPC_SMPTE170M;
    case PL_COLOR_SYSTEM_BT_709:        return AVCOL_SPC_BT709;
    case PL_COLOR_SYSTEM_SMPTE_240M:    return AVCOL_SPC_SMPTE240M;
    case PL_COLOR_SYSTEM_BT_2020_NC:    return AVCOL_SPC_BT2020_NCL;
    case PL_COLOR_SYSTEM_BT_2020_C:     return AVCOL_SPC_BT2020_CL;
    case PL_COLOR_SYSTEM_BT_2100_PQ:    return AVCOL_SPC_ICTCP;
    case PL_COLOR_SYSTEM_BT_2100_HLG:   return AVCOL_SPC_ICTCP;
    case PL_COLOR_SYSTEM_DOLBYVISION:   return AVCOL_SPC_UNSPECIFIED; // missing
    case PL_COLOR_SYSTEM_YCGCO:         return AVCOL_SPC_YCGCO;
    case PL_COLOR_SYSTEM_RGB:           return AVCOL_SPC_RGB;
    case PL_COLOR_SYSTEM_XYZ:           return AVCOL_SPC_UNSPECIFIED; // handled differently
    case PL_COLOR_SYSTEM_COUNT:         return AVCOL_SPC_NB;
    }

    return AVCOL_SPC_UNSPECIFIED;
}

static inline enum pl_color_levels pl_levels_from_av(enum AVColorRange range)
{
    switch (range) {
    case AVCOL_RANGE_UNSPECIFIED:       return PL_COLOR_LEVELS_UNKNOWN;
    case AVCOL_RANGE_MPEG:              return PL_COLOR_LEVELS_LIMITED;
    case AVCOL_RANGE_JPEG:              return PL_COLOR_LEVELS_FULL;
    case AVCOL_RANGE_NB:                return PL_COLOR_LEVELS_COUNT;
    }

    return PL_COLOR_LEVELS_UNKNOWN;
}

static inline enum AVColorRange pl_levels_to_av(enum pl_color_levels levels)
{
    switch (levels) {
    case PL_COLOR_LEVELS_UNKNOWN:       return AVCOL_RANGE_UNSPECIFIED;
    case PL_COLOR_LEVELS_LIMITED:       return AVCOL_RANGE_MPEG;
    case PL_COLOR_LEVELS_FULL:          return AVCOL_RANGE_JPEG;
    case PL_COLOR_LEVELS_COUNT:         return AVCOL_RANGE_NB;
    }

    return AVCOL_RANGE_UNSPECIFIED;
}

static inline enum pl_color_primaries pl_primaries_from_av(enum AVColorPrimaries prim)
{
    switch (prim) {
    case AVCOL_PRI_RESERVED0:       return PL_COLOR_PRIM_UNKNOWN;
    case AVCOL_PRI_BT709:           return PL_COLOR_PRIM_BT_709;
    case AVCOL_PRI_UNSPECIFIED:     return PL_COLOR_PRIM_UNKNOWN;
    case AVCOL_PRI_RESERVED:        return PL_COLOR_PRIM_UNKNOWN;
    case AVCOL_PRI_BT470M:          return PL_COLOR_PRIM_BT_470M;
    case AVCOL_PRI_BT470BG:         return PL_COLOR_PRIM_BT_601_625;
    case AVCOL_PRI_SMPTE170M:       return PL_COLOR_PRIM_BT_601_525;
    case AVCOL_PRI_SMPTE240M:       return PL_COLOR_PRIM_BT_601_525;
    case AVCOL_PRI_FILM:            return PL_COLOR_PRIM_FILM_C;
    case AVCOL_PRI_BT2020:          return PL_COLOR_PRIM_BT_2020;
    case AVCOL_PRI_SMPTE428:        return PL_COLOR_PRIM_CIE_1931;
    case AVCOL_PRI_SMPTE431:        return PL_COLOR_PRIM_DCI_P3;
    case AVCOL_PRI_SMPTE432:        return PL_COLOR_PRIM_DISPLAY_P3;
    case AVCOL_PRI_JEDEC_P22:       return PL_COLOR_PRIM_EBU_3213;
    case AVCOL_PRI_NB:              return PL_COLOR_PRIM_COUNT;
    }

    return PL_COLOR_PRIM_UNKNOWN;
}

static inline enum AVColorPrimaries pl_primaries_to_av(enum pl_color_primaries prim)
{
    switch (prim) {
    case PL_COLOR_PRIM_UNKNOWN:     return AVCOL_PRI_UNSPECIFIED;
    case PL_COLOR_PRIM_BT_601_525:  return AVCOL_PRI_SMPTE170M;
    case PL_COLOR_PRIM_BT_601_625:  return AVCOL_PRI_BT470BG;
    case PL_COLOR_PRIM_BT_709:      return AVCOL_PRI_BT709;
    case PL_COLOR_PRIM_BT_470M:     return AVCOL_PRI_BT470M;
    case PL_COLOR_PRIM_EBU_3213:    return AVCOL_PRI_JEDEC_P22;
    case PL_COLOR_PRIM_BT_2020:     return AVCOL_PRI_BT2020;
    case PL_COLOR_PRIM_APPLE:       return AVCOL_PRI_UNSPECIFIED; // missing
    case PL_COLOR_PRIM_ADOBE:       return AVCOL_PRI_UNSPECIFIED; // missing
    case PL_COLOR_PRIM_PRO_PHOTO:   return AVCOL_PRI_UNSPECIFIED; // missing
    case PL_COLOR_PRIM_CIE_1931:    return AVCOL_PRI_SMPTE428;
    case PL_COLOR_PRIM_DCI_P3:      return AVCOL_PRI_SMPTE431;
    case PL_COLOR_PRIM_DISPLAY_P3:  return AVCOL_PRI_SMPTE432;
    case PL_COLOR_PRIM_V_GAMUT:     return AVCOL_PRI_UNSPECIFIED; // missing
    case PL_COLOR_PRIM_S_GAMUT:     return AVCOL_PRI_UNSPECIFIED; // missing
    case PL_COLOR_PRIM_FILM_C:      return AVCOL_PRI_FILM;
    case PL_COLOR_PRIM_COUNT:       return AVCOL_PRI_NB;
    }

    return AVCOL_PRI_UNSPECIFIED;
}

static inline enum pl_color_transfer pl_transfer_from_av(enum AVColorTransferCharacteristic trc)
{
    switch (trc) {
    case AVCOL_TRC_RESERVED0:       return PL_COLOR_TRC_UNKNOWN;
    case AVCOL_TRC_BT709:           return PL_COLOR_TRC_BT_1886; // EOTF != OETF
    case AVCOL_TRC_UNSPECIFIED:     return PL_COLOR_TRC_UNKNOWN;
    case AVCOL_TRC_RESERVED:        return PL_COLOR_TRC_UNKNOWN;
    case AVCOL_TRC_GAMMA22:         return PL_COLOR_TRC_GAMMA22;
    case AVCOL_TRC_GAMMA28:         return PL_COLOR_TRC_GAMMA28;
    case AVCOL_TRC_SMPTE170M:       return PL_COLOR_TRC_BT_1886; // EOTF != OETF
    case AVCOL_TRC_SMPTE240M:       return PL_COLOR_TRC_BT_1886; // EOTF != OETF
    case AVCOL_TRC_LINEAR:          return PL_COLOR_TRC_LINEAR;
    case AVCOL_TRC_LOG:             return PL_COLOR_TRC_UNKNOWN; // missing
    case AVCOL_TRC_LOG_SQRT:        return PL_COLOR_TRC_UNKNOWN; // missing
    case AVCOL_TRC_IEC61966_2_4:    return PL_COLOR_TRC_BT_1886; // EOTF != OETF
    case AVCOL_TRC_BT1361_ECG:      return PL_COLOR_TRC_BT_1886; // ETOF != OETF
    case AVCOL_TRC_IEC61966_2_1:    return PL_COLOR_TRC_SRGB;
    case AVCOL_TRC_BT2020_10:       return PL_COLOR_TRC_BT_1886; // EOTF != OETF
    case AVCOL_TRC_BT2020_12:       return PL_COLOR_TRC_BT_1886; // EOTF != OETF
    case AVCOL_TRC_SMPTE2084:       return PL_COLOR_TRC_PQ;
    case AVCOL_TRC_SMPTE428:        return PL_COLOR_TRC_UNKNOWN; // missing
    case AVCOL_TRC_ARIB_STD_B67:    return PL_COLOR_TRC_HLG;
    case AVCOL_TRC_NB:              return PL_COLOR_TRC_COUNT;
    }

    return PL_COLOR_TRC_UNKNOWN;
}

static inline enum AVColorTransferCharacteristic pl_transfer_to_av(enum pl_color_transfer trc)
{
    switch (trc) {
    case PL_COLOR_TRC_UNKNOWN:      return AVCOL_TRC_UNSPECIFIED;
    case PL_COLOR_TRC_BT_1886:      return AVCOL_TRC_BT709;       // EOTF != OETF
    case PL_COLOR_TRC_SRGB:         return AVCOL_TRC_IEC61966_2_1;
    case PL_COLOR_TRC_LINEAR:       return AVCOL_TRC_LINEAR;
    case PL_COLOR_TRC_GAMMA18:      return AVCOL_TRC_UNSPECIFIED; // missing
    case PL_COLOR_TRC_GAMMA20:      return AVCOL_TRC_UNSPECIFIED; // missing
    case PL_COLOR_TRC_GAMMA22:      return AVCOL_TRC_GAMMA22;
    case PL_COLOR_TRC_GAMMA24:      return AVCOL_TRC_UNSPECIFIED; // missing
    case PL_COLOR_TRC_GAMMA26:      return AVCOL_TRC_UNSPECIFIED; // missing
    case PL_COLOR_TRC_GAMMA28:      return AVCOL_TRC_GAMMA28;
    case PL_COLOR_TRC_PRO_PHOTO:    return AVCOL_TRC_UNSPECIFIED; // missing
    case PL_COLOR_TRC_PQ:           return AVCOL_TRC_SMPTE2084;
    case PL_COLOR_TRC_HLG:          return AVCOL_TRC_ARIB_STD_B67;
    case PL_COLOR_TRC_V_LOG:        return AVCOL_TRC_UNSPECIFIED; // missing
    case PL_COLOR_TRC_S_LOG1:       return AVCOL_TRC_UNSPECIFIED; // missing
    case PL_COLOR_TRC_S_LOG2:       return AVCOL_TRC_UNSPECIFIED; // missing
    case PL_COLOR_TRC_COUNT:        return AVCOL_TRC_NB;
    }

    return AVCOL_TRC_UNSPECIFIED;
}

static inline enum pl_chroma_location pl_chroma_from_av(enum AVChromaLocation loc)
{
    switch (loc) {
    case AVCHROMA_LOC_UNSPECIFIED:  return PL_CHROMA_UNKNOWN;
    case AVCHROMA_LOC_LEFT:         return PL_CHROMA_LEFT;
    case AVCHROMA_LOC_CENTER:       return PL_CHROMA_CENTER;
    case AVCHROMA_LOC_TOPLEFT:      return PL_CHROMA_TOP_LEFT;
    case AVCHROMA_LOC_TOP:          return PL_CHROMA_TOP_CENTER;
    case AVCHROMA_LOC_BOTTOMLEFT:   return PL_CHROMA_BOTTOM_LEFT;
    case AVCHROMA_LOC_BOTTOM:       return PL_CHROMA_BOTTOM_CENTER;
    case AVCHROMA_LOC_NB:           return PL_CHROMA_COUNT;
    }

    return PL_CHROMA_UNKNOWN;
}

static inline enum AVChromaLocation pl_chroma_to_av(enum pl_chroma_location loc)
{
    switch (loc) {
    case PL_CHROMA_UNKNOWN:         return AVCHROMA_LOC_UNSPECIFIED;
    case PL_CHROMA_LEFT:            return AVCHROMA_LOC_LEFT;
    case PL_CHROMA_CENTER:          return AVCHROMA_LOC_CENTER;
    case PL_CHROMA_TOP_LEFT:        return AVCHROMA_LOC_TOPLEFT;
    case PL_CHROMA_TOP_CENTER:      return AVCHROMA_LOC_TOP;
    case PL_CHROMA_BOTTOM_LEFT:     return AVCHROMA_LOC_BOTTOMLEFT;
    case PL_CHROMA_BOTTOM_CENTER:   return AVCHROMA_LOC_BOTTOM;
    case PL_CHROMA_COUNT:           return AVCHROMA_LOC_NB;
    }

    return AVCHROMA_LOC_UNSPECIFIED;
}

static inline int pl_plane_data_num_comps(const struct pl_plane_data *data)
{
    for (int i = 0; i < 4; i++) {
        if (data->component_size[i] == 0)
            return i;
    }

    return 4;
}

static inline int pl_plane_data_from_pixfmt(struct pl_plane_data out_data[4],
                                            struct pl_bit_encoding *out_bits,
                                            enum AVPixelFormat pix_fmt)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(pix_fmt);
    int planes = av_pix_fmt_count_planes(pix_fmt);
    struct pl_plane_data aligned_data[4];
    struct pl_bit_encoding bits;
    bool first;
    if (!desc || planes < 0) // e.g. AV_PIX_FMT_NONE
        return 0;

    if (desc->flags & AV_PIX_FMT_FLAG_BE) {
        // Big endian formats are almost definitely not supported in any
        // reasonable manner, erroring as a safety precation
        return 0;
    }

    if (desc->flags & AV_PIX_FMT_FLAG_BITSTREAM) {
        // Bitstream formats will most likely never be supported
        return 0;
    }

    if (desc->flags & AV_PIX_FMT_FLAG_PAL) {
        // Palette formats are (currently) not supported
        return 0;
    }

    if (desc->flags & AV_PIX_FMT_FLAG_BAYER) {
        // Bayer format don't have valid `desc->offset` values, so we can't
        // use `pl_plane_data_from_mask` on them.
        return 0;
    }

    if (desc->nb_components == 0 || desc->nb_components > 4) {
        // Bogus components, possibly fake/virtual/hwaccel format?
        return 0;
    }

    if (planes > 4)
        return 0; // This shouldn't ever happen

    // Fill in the details for each plane
    for (int p = 0; p < planes; p++) {
        struct pl_plane_data *data = &out_data[p];
        uint64_t masks[4] = {0};
        data->type = (desc->flags & AV_PIX_FMT_FLAG_FLOAT)
                        ? PL_FMT_FLOAT
                        : PL_FMT_UNORM;

        data->pixel_stride = 0;

        for (int c = 0; c < desc->nb_components; c++) {
            const AVComponentDescriptor *comp = &desc->comp[c];
            if (comp->plane != p)
                continue;

            masks[c] = (1LLU << comp->depth) - 1; // e.g. 0xFF for depth=8
            masks[c] <<= comp->shift;
            masks[c] <<= comp->offset * 8;

            if (data->pixel_stride && (int) data->pixel_stride != comp->step) {
                // Pixel format contains components with different pixel stride
                // (e.g. packed YUYV), this is currently not supported
                return 0;
            }
            data->pixel_stride = comp->step;
        }

        pl_plane_data_from_mask(data, masks);
    }

    if (!out_bits)
        return planes;

    // Attempt aligning all of the planes for optimum compatibility
    first = true;
    for (int p = 0; p < planes; p++) {
        aligned_data[p] = out_data[p];

        // Planes with only an alpha component should be ignored
        if (pl_plane_data_num_comps(&aligned_data[p]) == 1 &&
            aligned_data[p].component_map[0] == PL_CHANNEL_A)
        {
            continue;
        }

        if (!pl_plane_data_align(&aligned_data[p], &bits))
            goto misaligned;

        if (first) {
            *out_bits = bits;
            first = false;
        } else {
            if (!pl_bit_encoding_equal(&bits, out_bits))
                goto misaligned;
        }
    }

    // Overwrite the planes by their aligned versions
    for (int p = 0; p < planes; p++)
        out_data[p] = aligned_data[p];

    return planes;

misaligned:
    *out_bits = (struct pl_bit_encoding) {0};
    return planes;
}

static inline bool pl_test_pixfmt(pl_gpu gpu,
                                  enum AVPixelFormat pixfmt)
{
    struct pl_bit_encoding bits;
    struct pl_plane_data data[4];
    int planes;

    switch (pixfmt) {
    case AV_PIX_FMT_DRM_PRIME:
    case AV_PIX_FMT_VAAPI:
        return gpu->import_caps.tex & PL_HANDLE_DMA_BUF;

#ifdef HAVE_LAV_VULKAN
    case AV_PIX_FMT_VULKAN:
        return pl_vulkan_get(gpu);
#endif

    default: break;
    }

    planes = pl_plane_data_from_pixfmt(data, &bits, pixfmt);
    if (!planes)
        return false;

    for (int i = 0; i < planes; i++) {
        data[i].row_stride = 0;
        if (!pl_plane_find_fmt(gpu, NULL, &data[i]))
            return false;
    }

    return true;
}

static inline void pl_avframe_set_color(AVFrame *frame, struct pl_color_space space)
{
    const AVFrameSideData *sd;

    frame->color_primaries = pl_primaries_to_av(space.primaries);
    frame->color_trc = pl_transfer_to_av(space.transfer);
    // No way to map space.light, so just ignore it

    if (space.sig_peak > 1 && space.sig_peak < pl_color_transfer_nominal_peak(space.transfer)) {
        sd = av_frame_get_side_data(frame, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
        if (!sd) {
            sd = av_frame_new_side_data(frame, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL,
                                        sizeof(AVContentLightMetadata));
        }

        if (sd) {
            AVContentLightMetadata *clm = (AVContentLightMetadata *) sd->data;
            *clm = (AVContentLightMetadata) {
                .MaxCLL = space.sig_peak * PL_COLOR_SDR_WHITE,
                .MaxFALL = space.sig_avg * PL_COLOR_SDR_WHITE,
            };

            if (!clm->MaxFALL)
                clm->MaxFALL = clm->MaxCLL;
        }
    }

    if (space.sig_floor > 0.0) {
        sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
        if (!sd) {
            sd = av_frame_new_side_data(frame, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA,
                                        sizeof(AVMasteringDisplayMetadata));
        }

        if (sd) {
            AVMasteringDisplayMetadata *mdm = (AVMasteringDisplayMetadata *) sd->data;
            *mdm = (AVMasteringDisplayMetadata) {
                .max_luminance = av_d2q(space.sig_peak * PL_COLOR_SDR_WHITE, 100000),
                .min_luminance = av_d2q(space.sig_floor * PL_COLOR_SDR_WHITE, 100000),
                .has_luminance = 1,
            };
        }
    }

    // No way to map space.sig_scale, so just ignore it
}

static inline void pl_avframe_set_repr(AVFrame *frame, struct pl_color_repr repr)
{
    frame->colorspace = pl_system_to_av(repr.sys);
    frame->color_range = pl_levels_to_av(repr.levels);

    // No real way to map repr.bits, the image format already has to match
}

static inline void pl_avframe_set_profile(AVFrame *frame, struct pl_icc_profile profile)
{
    const AVFrameSideData *sd;
    av_frame_remove_side_data(frame, AV_FRAME_DATA_ICC_PROFILE);

    if (!profile.len)
        return;

    sd = av_frame_new_side_data(frame, AV_FRAME_DATA_ICC_PROFILE, profile.len);
    memcpy(sd->data, profile.data, profile.len);
}

static inline void pl_frame_from_avframe(struct pl_frame *out,
                                         const AVFrame *frame)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(frame->format);
    int planes = av_pix_fmt_count_planes(frame->format);
    const AVFrameSideData *sd;
    assert(desc);

    if (desc->flags & AV_PIX_FMT_FLAG_HWACCEL) {
        const AVHWFramesContext *hwfc = (AVHWFramesContext *) frame->hw_frames_ctx->data;
        desc = av_pix_fmt_desc_get(hwfc->sw_format);
        planes = av_pix_fmt_count_planes(hwfc->sw_format);
    }

    // This should never fail, and there's nothing really useful we can do in
    // this failure case anyway, since this is a `void` function.
    assert(planes <= 4);

    *out = (struct pl_frame) {
        .num_planes = planes,
        .crop = {
            .x0 = frame->crop_left,
            .y0 = frame->crop_top,
            .x1 = frame->width - frame->crop_right,
            .y1 = frame->height - frame->crop_bottom,
        },
        .color = {
            .primaries = pl_primaries_from_av(frame->color_primaries),
            .transfer = pl_transfer_from_av(frame->color_trc),
            .light = PL_COLOR_LIGHT_UNKNOWN,
        },
        .repr = {
            .sys = pl_system_from_av(frame->colorspace),
            .levels = pl_levels_from_av(frame->color_range),
            .alpha = (desc->flags & AV_PIX_FMT_FLAG_ALPHA)
                        ? PL_ALPHA_INDEPENDENT
                        : PL_ALPHA_UNKNOWN,

            // For sake of simplicity, just use the first component's depth as
            // the authoritative color depth for the whole image. Usually, this
            // will be overwritten by more specific information when using e.g.
            // `pl_map_avframe`, but for the sake of e.g. users wishing to map
            // hwaccel frames manually, this is a good default.
            .bits.color_depth = desc->comp[0].depth,
        },
    };

    if (frame->colorspace == AVCOL_SPC_ICTCP &&
        frame->color_trc == AVCOL_TRC_ARIB_STD_B67)
    {
        // libav* makes no distinction between PQ and HLG ICtCp, so we need
        // to manually fix it in the case that we have HLG ICtCp data.
        out->repr.sys = PL_COLOR_SYSTEM_BT_2100_HLG;

    } else if (strncmp(desc->name, "xyz", 3) == 0) {

        // libav* handles this as a special case, but doesn't provide an
        // explicit flag for it either, so we have to resort to this ugly
        // hack...
        out->repr.sys= PL_COLOR_SYSTEM_XYZ;

    } else if (desc->flags & AV_PIX_FMT_FLAG_RGB) {

        out->repr.sys = PL_COLOR_SYSTEM_RGB;
        out->repr.levels = PL_COLOR_LEVELS_FULL; // libav* ignores levels for RGB

    } else if (!out->repr.sys) {

        // libav* likes leaving this as UNKNOWN for YCbCr frames, which
        // confuses libplacebo since we infer UNKNOWN as RGB. To get around
        // this, explicitly infer a suitable colorspace for non-RGB formats.
        out->repr.sys = pl_color_system_guess_ycbcr(frame->width, frame->height);
    }

    if ((sd = av_frame_get_side_data(frame, AV_FRAME_DATA_ICC_PROFILE))) {
        out->profile = (struct pl_icc_profile) {
            .data = sd->data,
            .len = sd->size,
        };

        // Needed to ensure profile uniqueness
        pl_icc_profile_compute_signature(&out->profile);
    }

    if ((sd = av_frame_get_side_data(frame, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL))) {
        const AVContentLightMetadata *clm = (AVContentLightMetadata *) sd->data;
        out->color.sig_peak = clm->MaxCLL / PL_COLOR_SDR_WHITE;
        out->color.sig_avg = clm->MaxFALL / PL_COLOR_SDR_WHITE;
    }

    // This overrides the CLL values above, if both are present
    if ((sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA))) {
        const AVMasteringDisplayMetadata *mdm = (AVMasteringDisplayMetadata *) sd->data;
        if (mdm->has_luminance) {
            out->color.sig_peak = av_q2d(mdm->max_luminance) / PL_COLOR_SDR_WHITE;
            out->color.sig_floor = av_q2d(mdm->min_luminance) / PL_COLOR_SDR_WHITE;
        }
    }

    if ((sd = av_frame_get_side_data(frame, AV_FRAME_DATA_DISPLAYMATRIX))) {
        double rot = av_display_rotation_get((const int32_t *) sd->data);
        out->rotation = pl_rotation_normalize(4.5 - rot / 90.0);
    }

    // Make sure this value is more or less legal
    if (out->color.sig_peak < 1.0 || out->color.sig_peak > 50.0)
        out->color.sig_peak = 0.0;

#ifdef HAVE_LAV_FILM_GRAIN
    if ((sd = av_frame_get_side_data(frame, AV_FRAME_DATA_FILM_GRAIN_PARAMS))) {
        const AVFilmGrainParams *fgp = (AVFilmGrainParams *) sd->data;
        out->film_grain.seed = fgp->seed;

        switch (fgp->type) {
        case AV_FILM_GRAIN_PARAMS_NONE: break;
        case AV_FILM_GRAIN_PARAMS_AV1: {
            const AVFilmGrainAOMParams *src = &fgp->codec.aom;
            struct pl_av1_grain_data *dst = &out->film_grain.params.av1;
            out->film_grain.type = PL_FILM_GRAIN_AV1;
            *dst = (struct pl_av1_grain_data) {
                .num_points_y = src->num_y_points,
                .chroma_scaling_from_luma = src->chroma_scaling_from_luma,
                .num_points_uv = { src->num_uv_points[0], src->num_uv_points[1] },
                .scaling_shift = src->scaling_shift,
                .ar_coeff_lag = src->ar_coeff_lag,
                .ar_coeff_shift = src->ar_coeff_shift,
                .grain_scale_shift = src->grain_scale_shift,
                .uv_mult = { src->uv_mult[0], src->uv_mult[1] },
                .uv_mult_luma = { src->uv_mult_luma[0], src->uv_mult_luma[1] },
                .uv_offset = { src->uv_offset[0], src->uv_offset[1] },
                .overlap = src->overlap_flag,
            };

            assert(sizeof(dst->ar_coeffs_uv) == sizeof(src->ar_coeffs_uv));
            memcpy(dst->points_y, src->y_points, sizeof(dst->points_y));
            memcpy(dst->points_uv, src->uv_points, sizeof(dst->points_uv));
            memcpy(dst->ar_coeffs_y, src->ar_coeffs_y, sizeof(dst->ar_coeffs_y));
            memcpy(dst->ar_coeffs_uv, src->ar_coeffs_uv, sizeof(dst->ar_coeffs_uv));
            break;
        }
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 2, 100)
        case AV_FILM_GRAIN_PARAMS_H274: {
            const AVFilmGrainH274Params *src = &fgp->codec.h274;
            struct pl_h274_grain_data *dst = &out->film_grain.params.h274;
            out->film_grain.type = PL_FILM_GRAIN_H274;
            *dst = (struct pl_h274_grain_data) {
                .model_id = src->model_id,
                .blending_mode_id = src->blending_mode_id,
                .log2_scale_factor = src->log2_scale_factor,
                .component_model_present = {
                    src->component_model_present[0],
                    src->component_model_present[1],
                    src->component_model_present[2],
                },
                .intensity_interval_lower_bound = {
                    src->intensity_interval_lower_bound[0],
                    src->intensity_interval_lower_bound[1],
                    src->intensity_interval_lower_bound[2],
                },
                .intensity_interval_upper_bound = {
                    src->intensity_interval_upper_bound[0],
                    src->intensity_interval_upper_bound[1],
                    src->intensity_interval_upper_bound[2],
                },
                .comp_model_value = {
                    src->comp_model_value[0],
                    src->comp_model_value[1],
                    src->comp_model_value[2],
                },
            };
            memcpy(dst->num_intensity_intervals, src->num_intensity_intervals,
                   sizeof(dst->num_intensity_intervals));
            memcpy(dst->num_model_values, src->num_model_values,
                   sizeof(dst->num_model_values));
            break;
        }
#endif
        }
    }
#endif // HAVE_LAV_FILM_GRAIN

    for (int p = 0; p < out->num_planes; p++) {
        struct pl_plane *plane = &out->planes[p];

        // Fill in the component mapping array
        for (int c = 0; c < desc->nb_components; c++) {
            if (desc->comp[c].plane == p)
                plane->component_mapping[plane->components++] = c;
        }

        // Clear the superfluous components
        for (int c = plane->components; c < 4; c++)
            plane->component_mapping[c] = PL_CHANNEL_NONE;
    }

    // Only set the chroma location for definitely subsampled images, makes no
    // sense otherwise
    if (desc->log2_chroma_w || desc->log2_chroma_h) {
        enum pl_chroma_location loc = pl_chroma_from_av(frame->chroma_location);
        pl_frame_set_chroma_location(out, loc);
    }
}

static inline void pl_frame_copy_stream_props(struct pl_frame *out,
                                              const AVStream *stream)
{
    const uint8_t *sd;
    if ((sd = av_stream_get_side_data(stream, AV_PKT_DATA_CONTENT_LIGHT_LEVEL, NULL))) {
        const AVContentLightMetadata *clm = (AVContentLightMetadata *) sd;
        out->color.sig_peak = clm->MaxCLL / PL_COLOR_SDR_WHITE;
        out->color.sig_avg = clm->MaxFALL / PL_COLOR_SDR_WHITE;
    }

    if ((sd = av_stream_get_side_data(stream, AV_PKT_DATA_MASTERING_DISPLAY_METADATA, NULL))) {
        const AVMasteringDisplayMetadata *mdm = (AVMasteringDisplayMetadata *) sd;
        if (mdm->has_luminance) {
            out->color.sig_peak = av_q2d(mdm->max_luminance) / PL_COLOR_SDR_WHITE;
            out->color.sig_floor = av_q2d(mdm->min_luminance) / PL_COLOR_SDR_WHITE;
        }
    }

    if ((sd = av_stream_get_side_data(stream, AV_PKT_DATA_DISPLAYMATRIX, NULL))) {
        double rot = av_display_rotation_get((const int32_t *) sd);
        out->rotation = pl_rotation_normalize(4.5 - rot / 90.0);
    }
}

static inline void pl_swapchain_colors_from_avframe(struct pl_swapchain_colors *out_colors,
                                                    const AVFrame *avframe)
{
    const AVFrameSideData *sd;
    struct pl_frame frame;
    pl_frame_from_avframe(&frame, avframe);

    *out_colors = (struct pl_swapchain_colors) {
        .primaries = frame.color.primaries,
        .transfer = frame.color.transfer,
    };

    if ((sd = av_frame_get_side_data(avframe, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL))) {
        const AVContentLightMetadata *clm = (AVContentLightMetadata *) sd->data;
        out_colors->hdr.max_cll = clm->MaxCLL;
        out_colors->hdr.max_fall = clm->MaxFALL;
    }

    if ((sd = av_frame_get_side_data(avframe, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA))) {
        const AVMasteringDisplayMetadata *mdm = (AVMasteringDisplayMetadata *) sd->data;
        if (mdm->has_luminance) {
            out_colors->hdr.min_luma = av_q2d(mdm->min_luminance);
            out_colors->hdr.max_luma = av_q2d(mdm->max_luminance);
        }

        if (mdm->has_primaries) {
            out_colors->hdr.prim.red.x   = av_q2d(mdm->display_primaries[0][0]);
            out_colors->hdr.prim.red.y   = av_q2d(mdm->display_primaries[0][1]);
            out_colors->hdr.prim.green.x = av_q2d(mdm->display_primaries[1][0]);
            out_colors->hdr.prim.green.y = av_q2d(mdm->display_primaries[1][1]);
            out_colors->hdr.prim.blue.x  = av_q2d(mdm->display_primaries[2][0]);
            out_colors->hdr.prim.blue.y  = av_q2d(mdm->display_primaries[2][1]);
            out_colors->hdr.prim.white.x = av_q2d(mdm->white_point[0]);
            out_colors->hdr.prim.white.y = av_q2d(mdm->white_point[1]);
        }
    }
}

static inline bool pl_frame_recreate_from_avframe(pl_gpu gpu,
                                                  struct pl_frame *out,
                                                  pl_tex tex[4],
                                                  const AVFrame *frame)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(frame->format);
    struct pl_plane_data data[4] = {0};
    int planes;

    pl_frame_from_avframe(out, frame);
    planes = pl_plane_data_from_pixfmt(data, &out->repr.bits, frame->format);
    if (!planes)
        return false;

    for (int p = 0; p < planes; p++) {
        bool is_chroma = p == 1 || p == 2; // matches lavu logic
        data[p].width = AV_CEIL_RSHIFT(frame->width, is_chroma ? desc->log2_chroma_w : 0);
        data[p].height = AV_CEIL_RSHIFT(frame->height, is_chroma ? desc->log2_chroma_h : 0);

        if (!pl_recreate_plane(gpu, &out->planes[p], &tex[p], &data[p]))
            return false;
    }

    return true;
}

static void pl_avframe_free_cb(void *priv)
{
    AVFrame *frame = priv;
    av_frame_free(&frame);
}

#define PL_MAGIC0 0xfb5b3b8b
#define PL_MAGIC1 0xee659f6d

struct pl_avalloc {
    uint32_t magic[2];
    pl_gpu gpu;
    pl_buf buf;
};

static void pl_fix_hwframe_sample_depth(struct pl_frame *out, const AVFrame *frame)
{
    const AVHWFramesContext *hwfc = (AVHWFramesContext *) frame->hw_frames_ctx->data;
    pl_fmt fmt = out->planes[0].texture->params.format;
    struct pl_bit_encoding *bits = &out->repr.bits;

    bits->sample_depth = fmt->component_depth[0];

    switch (hwfc->sw_format) {
    case AV_PIX_FMT_P010: bits->bit_shift = 6; break;
    default: break;
    }
}

static bool pl_map_avframe_drm(pl_gpu gpu, struct pl_frame *out,
                               const AVFrame *frame)
{
    const AVHWFramesContext *hwfc = (AVHWFramesContext *) frame->hw_frames_ctx->data;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(hwfc->sw_format);
    const AVDRMFrameDescriptor *drm = (AVDRMFrameDescriptor *) frame->data[0];
    assert(frame->format == AV_PIX_FMT_DRM_PRIME);
    if (!(gpu->import_caps.tex & PL_HANDLE_DMA_BUF))
        return false;

    assert(drm->nb_layers >= out->num_planes);
    for (int n = 0; n < out->num_planes; n++) {
        const AVDRMLayerDescriptor *layer = &drm->layers[n];
        const AVDRMPlaneDescriptor *plane = &layer->planes[0];
        const AVDRMObjectDescriptor *object = &drm->objects[plane->object_index];
        pl_fmt fmt = pl_find_fourcc(gpu, layer->format);
        bool is_chroma = n == 1 || n == 2;
        if (!fmt || !pl_fmt_has_modifier(fmt, object->format_modifier))
            return false;

        assert(layer->nb_planes == 1); // we only support planar formats
        out->planes[n].texture = pl_tex_create(gpu, pl_tex_params(
            .w = AV_CEIL_RSHIFT(frame->width, is_chroma ? desc->log2_chroma_w : 0),
            .h = AV_CEIL_RSHIFT(frame->height, is_chroma ? desc->log2_chroma_h : 0),
            .format = fmt,
            .sampleable = true,
            .blit_src = fmt->caps & PL_FMT_CAP_BLITTABLE,
            .import_handle = PL_HANDLE_DMA_BUF,
            .shared_mem = {
                .handle.fd = object->fd,
                .size = object->size,
                .offset = plane->offset,
                .drm_format_mod = object->format_modifier,
                .stride_w = plane->pitch,
            },
        ));
        if (!out->planes[n].texture)
            return false;
    }

    pl_fix_hwframe_sample_depth(out, frame);
    return true;
}

// Derive a DMABUF from any other hwaccel format, and map that instead
static bool pl_map_avframe_derived(pl_gpu gpu, struct pl_frame *out,
                                   const AVFrame *frame)
{
    const int flags = AV_HWFRAME_MAP_READ | AV_HWFRAME_MAP_DIRECT;
    AVFrame *derived = av_frame_alloc();
    derived->width = frame->width;
    derived->height = frame->height;
    derived->format = AV_PIX_FMT_DRM_PRIME;
    derived->hw_frames_ctx = av_buffer_ref(frame->hw_frames_ctx);
    if (av_hwframe_map(derived, frame, flags) < 0)
        goto error;
    if (av_frame_copy_props(derived, frame) < 0)
        goto error;
    if (!pl_map_avframe_drm(gpu, out, derived))
        goto error;

    av_frame_free((AVFrame **) &out->user_data);
    out->user_data = derived;
    return true;

error:
    av_frame_free(&derived);
    return false;
}

#ifdef HAVE_LAV_VULKAN
static bool pl_map_avframe_vulkan(pl_gpu gpu, struct pl_frame *out,
                                  const AVFrame *frame)
{
    const AVHWFramesContext *hwfc = (AVHWFramesContext *) frame->hw_frames_ctx->data;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(hwfc->sw_format);
    const AVVulkanFramesContext *vkfc = hwfc->hwctx;
    const VkFormat *vk_fmt = av_vkfmt_from_pixfmt(hwfc->sw_format);
    AVVkFrame *vkf = (AVVkFrame *) frame->data[0];
    pl_vulkan vk = pl_vulkan_get(gpu);
    assert(frame->format == AV_PIX_FMT_VULKAN);
    if (!vk)
        return false;

    for (int n = 0; n < out->num_planes; n++) {
        pl_tex *tex = &out->planes[n].texture;
        bool chroma = n == 1 || n == 2;
        *tex = pl_vulkan_wrap(gpu, pl_vulkan_wrap_params(
            .image = vkf->img[n],
            .width = AV_CEIL_RSHIFT(frame->width, chroma ? desc->log2_chroma_w : 0),
            .height = AV_CEIL_RSHIFT(frame->height, chroma ? desc->log2_chroma_h : 0),
            .format = vk_fmt[n],
            .usage = vkfc->usage,
        ));
        if (!*tex)
            return false;

        pl_vulkan_release(gpu, *tex, vkf->layout[n], (pl_vulkan_sem) {
            .sem = vkf->sem[n],
            .value = vkf->sem_value[n],
        });
    }

    pl_fix_hwframe_sample_depth(out, frame);
    return true;
}

static void pl_unmap_avframe_vulkan(pl_gpu gpu, struct pl_frame *frame)
{
    const AVFrame *avframe = frame->user_data;
    AVVkFrame *vkf = (AVVkFrame *) avframe->data[0];
    int ok;

    for (int n = 0; n < frame->num_planes; n++) {
        pl_tex *tex = &frame->planes[n].texture;
        if (!*tex)
            continue;
        ok = pl_vulkan_hold_raw(gpu, *tex, &vkf->layout[n], (pl_vulkan_sem) {
            .sem = vkf->sem[n],
            .value = vkf->sem_value[n] + 1,
        });

        vkf->access[n] = 0;
        vkf->sem_value[n] += !!ok;
        pl_tex_destroy(gpu, tex);
    }
}
#endif

static inline bool pl_map_avframe(pl_gpu gpu, struct pl_frame *out,
                                  pl_tex tex[4], const AVFrame *frame)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(frame->format);
    struct pl_plane_data data[4] = {0};
    struct pl_avalloc *alloc;
    int planes;

    pl_frame_from_avframe(out, frame);
    out->user_data = av_frame_clone(frame);

    switch (frame->format) {
    case AV_PIX_FMT_DRM_PRIME:
        if (!pl_map_avframe_drm(gpu, out, frame))
            goto error;
        return true;

    case AV_PIX_FMT_VAAPI:
        if (!pl_map_avframe_derived(gpu, out, frame))
            goto error;
        return true;

#ifdef HAVE_LAV_VULKAN
    case AV_PIX_FMT_VULKAN:
        if (!pl_map_avframe_vulkan(gpu, out, frame))
            goto error;
        return true;
#endif

    default: break;
    }

    planes = pl_plane_data_from_pixfmt(data, &out->repr.bits, frame->format);
    if (!planes)
        goto error;

    for (int p = 0; p < planes; p++) {
        bool is_chroma = p == 1 || p == 2; // matches lavu logic
        data[p].width = AV_CEIL_RSHIFT(frame->width, is_chroma ? desc->log2_chroma_w : 0);
        data[p].height = AV_CEIL_RSHIFT(frame->height, is_chroma ? desc->log2_chroma_h : 0);
        data[p].row_stride = frame->linesize[p];
        data[p].pixels = frame->data[p];

        // Probe for frames allocated by pl_get_buffer2
        alloc = frame->buf[p] ? av_buffer_get_opaque(frame->buf[p]) : NULL;
        if (alloc && alloc->magic[0] == PL_MAGIC0 && alloc->magic[1] == PL_MAGIC1) {
            data[p].pixels = NULL;
            data[p].buf = alloc->buf;
            data[p].buf_offset = (uintptr_t) frame->data[p] - (uintptr_t) alloc->buf->data;
        } else if (gpu->limits.callbacks) {
            // Use asynchronous upload if possible
            data[p].callback = pl_avframe_free_cb;
            data[p].priv = av_frame_clone(frame);
        }

        if (!pl_upload_plane(gpu, &out->planes[p], &tex[p], &data[p])) {
            av_frame_free((AVFrame **) &data[p].priv);
            goto error;
        }

        out->planes[p].texture = tex[p];
    }

    return true;

error:
    pl_unmap_avframe(gpu, out);
    return false;
}

static inline void pl_unmap_avframe(pl_gpu gpu, struct pl_frame *frame)
{
    AVFrame *avframe = frame->user_data;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(avframe->format);

#ifdef HAVE_LAV_VULKAN
    if (avframe->format == AV_PIX_FMT_VULKAN)
        pl_unmap_avframe_vulkan(gpu, frame);
#endif

    if (desc->flags & AV_PIX_FMT_FLAG_HWACCEL) {
        for (int i = 0; i < 4; i++)
            pl_tex_destroy(gpu, &frame->planes[i].texture);
    }

    av_frame_free(&avframe);
    memset(frame, 0, sizeof(*frame)); // sanity
}

static inline bool pl_upload_avframe(pl_gpu gpu, struct pl_frame *out_frame,
                                     pl_tex tex[4], const AVFrame *frame)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(frame->format);
    if (desc->flags & AV_PIX_FMT_FLAG_HWACCEL)
        return false;

    if (!pl_map_avframe(gpu, out_frame, tex, frame))
        return false;

    av_frame_free((AVFrame **) &out_frame->user_data);
    return true;
}

static void pl_done_cb(void *priv)
{
    bool *status = priv;
    *status = true;
}

static inline bool pl_download_avframe(pl_gpu gpu,
                                       const struct pl_frame *frame,
                                       AVFrame *out_frame)
{
    bool done[4] = {0};
    if (frame->num_planes != av_pix_fmt_count_planes(out_frame->format))
        return false;

    for (int p = 0; p < frame->num_planes; p++) {
        bool ok = pl_tex_download(gpu, pl_tex_transfer_params(
            .tex = frame->planes[p].texture,
            .row_pitch = out_frame->linesize[p],
            .ptr = out_frame->data[p],
            // Use synchronous transfer for the last plane
            .callback = (p+1) < frame->num_planes ? pl_done_cb : NULL,
            .priv = &done[p],
        ));

        if (!ok)
            return false;
    }

    for (int p = 0; p < frame->num_planes - 1; p++) {
        while (!done[p])
            pl_tex_poll(gpu, frame->planes[p].texture, UINT64_MAX);
    }

    return true;
}

#define PL_ALIGN2(x, align) (((x) + (align) - 1) & ~((align) - 1))
#define PL_MAX(x, y) ((x) > (y) ? (x) : (y))

static inline void pl_avalloc_free(void *opaque, uint8_t *data)
{
    struct pl_avalloc *alloc = opaque;
    assert(alloc->magic[0] == PL_MAGIC0);
    assert(alloc->magic[1] == PL_MAGIC1);
    assert(alloc->buf->data == data);
    pl_buf_destroy(alloc->gpu, &alloc->buf);
    free(alloc);
}

static inline int pl_get_buffer2(AVCodecContext *avctx, AVFrame *pic, int flags)
{
    int alignment[AV_NUM_DATA_POINTERS];
    int width = pic->width;
    int height = pic->height;
    size_t planesize[4];
    int ret = 0;

    pl_gpu *pgpu = avctx->opaque;
    pl_gpu gpu = pgpu ? *pgpu : NULL;
    struct pl_plane_data data[4];
    struct pl_avalloc *alloc;
    int planes = pl_plane_data_from_pixfmt(data, NULL, pic->format);

    // Sanitize frame structs
    memset(pic->data, 0, sizeof(pic->data));
    memset(pic->linesize, 0, sizeof(pic->linesize));
    memset(pic->buf, 0, sizeof(pic->buf));
    pic->extended_data = pic->data;
    pic->extended_buf = NULL;

    if (!(avctx->codec->capabilities & AV_CODEC_CAP_DR1) || !planes)
        goto fallback;
    if (!gpu || !gpu->limits.thread_safe || !gpu->limits.max_mapped_size)
        goto fallback;

    avcodec_align_dimensions2(avctx, &width, &height, alignment);
    if ((ret = av_image_fill_linesizes(pic->linesize, pic->format, width)))
        return ret;

    for (int p = 0; p < 4; p++) {
        alignment[p] = PL_ALIGN2(alignment[p], gpu->limits.align_tex_xfer_pitch);
        alignment[p] = PL_ALIGN2(alignment[p], gpu->limits.align_tex_xfer_offset);
        pic->linesize[p] = PL_ALIGN2(pic->linesize[p], alignment[p]);
    }

#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(56, 56, 100)
    ret = av_image_fill_plane_sizes(planesize, pic->format, height, (ptrdiff_t[4]) {
        pic->linesize[0], pic->linesize[1], pic->linesize[2], pic->linesize[3],
    });
    if (ret < 0)
        return ret;
#else
    uint8_t *ptrs[4], * const base = (uint8_t *) 0x10000;
    ret = av_image_fill_pointers(ptrs, pic->format, height, base, pic->linesize);
    if (ret < 0)
        return ret;
    for (int p = 0; p < 4; p++)
        planesize[p] = (uintptr_t) ptrs[p] - (uintptr_t) base;
#endif

    for (int p = 0; p < planes; p++) {
        const size_t buf_size = planesize[p] + alignment[p];
        if (buf_size > gpu->limits.max_mapped_size) {
            av_frame_unref(pic);
            goto fallback;
        }

        alloc = malloc(sizeof(*alloc));
        if (!alloc) {
            av_frame_unref(pic);
            return AVERROR(ENOMEM);
        }

        *alloc = (struct pl_avalloc) {
            .magic = { PL_MAGIC0, PL_MAGIC1 },
            .gpu = gpu,
            .buf = pl_buf_create(gpu, pl_buf_params(
                .size = buf_size,
                .memory_type = PL_BUF_MEM_HOST,
                .host_mapped = true,
            )),
        };

        if (!alloc->buf) {
            free(alloc);
            av_frame_unref(pic);
            return AVERROR(ENOMEM);
        }

        pic->data[p] = (uint8_t *) PL_ALIGN2((uintptr_t) alloc->buf->data, alignment[p]);
        pic->buf[p] = av_buffer_create(alloc->buf->data, buf_size, pl_avalloc_free, alloc, 0);
        if (!pic->buf[p]) {
            free(alloc);
            pl_buf_destroy(gpu, &alloc->buf);
            av_frame_unref(pic);
            return AVERROR(ENOMEM);
        }
    }

    return 0;

fallback:
    return avcodec_default_get_buffer2(avctx, pic, flags);
}

#undef PL_MAGIC0
#undef PL_MAGIC1
#undef PL_ALIGN
#undef PL_MAX

#endif // LIBPLACEBO_LIBAV_H_
