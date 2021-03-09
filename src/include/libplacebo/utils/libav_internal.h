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
#error This header should be included as part of <libplacebo/utils/libav.h>
#else

#include <assert.h>

#include <libavutil/hwcontext.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/pixdesc.h>

#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(56, 61, 100)
#define HAVE_LAV_FILM_GRAIN
#include <libavutil/film_grain_params.h>
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
    case PL_COLOR_TRC_GAMMA22:      return AVCOL_TRC_GAMMA22;
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
    assert(desc);

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

    int planes = av_pix_fmt_count_planes(pix_fmt);
    if (planes > 4)
        return 0; // This shouldn't ever happen

    // Fill in the details for each plane
    for (int p = 0; p < planes; p++) {
        struct pl_plane_data *data = &out_data[p];
        data->type = (desc->flags & AV_PIX_FMT_FLAG_FLOAT)
                        ? PL_FMT_FLOAT
                        : PL_FMT_UNORM;

        uint64_t masks[4] = {0};
        data->pixel_stride = 0;

        for (int c = 0; c < desc->nb_components; c++) {
            const AVComponentDescriptor *comp = &desc->comp[c];
            if (comp->plane != p)
                continue;

            masks[c] = (1 << comp->depth) - 1; // e.g. 0xFF for depth=8
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
    struct pl_plane_data data[4];
    struct pl_bit_encoding bits;
    bool first = true;

    for (int p = 0; p < planes; p++) {
        data[p] = out_data[p];

        // Planes with only an alpha component should be ignored
        bool is_alpha = pl_plane_data_num_comps(&data[p]) == 1 &&
                        data[p].component_map[0] == PL_CHANNEL_A;

        if (is_alpha)
            continue;

        if (!pl_plane_data_align(&data[p], &bits))
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
        out_data[p] = data[p];

    return planes;

misaligned:
    *out_bits = (struct pl_bit_encoding) {0};
    return planes;
}

static inline bool pl_test_pixfmt(const struct pl_gpu *gpu,
                                  enum AVPixelFormat pixfmt)
{
    struct pl_bit_encoding bits;
    struct pl_plane_data data[4];
    int planes = pl_plane_data_from_pixfmt(data, &bits, pixfmt);
    if (!planes)
        return false;

    for (int i = 0; i < planes; i++) {
        if (!pl_plane_find_fmt(gpu, NULL, &data[i]))
            return false;
    }

    return true;
}

static inline void pl_avframe_set_color(AVFrame *frame, struct pl_color_space space)
{
    frame->color_primaries = pl_primaries_to_av(space.primaries);
    frame->color_trc = pl_transfer_to_av(space.transfer);
    // No way to map space.light, so just ignore it

    const AVFrameSideData *sd;
    if (space.sig_peak > 1 && space.sig_peak < pl_color_transfer_nominal_peak(space.transfer)) {
        sd = av_frame_get_side_data(frame, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
        if (!sd) {
            sd = av_frame_new_side_data(frame, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL,
                                        sizeof(AVContentLightMetadata));
        }

        if (sd) {
            AVContentLightMetadata *clm = (AVContentLightMetadata *) sd->data;
            *clm = (struct AVContentLightMetadata) {
                .MaxCLL = space.sig_peak * PL_COLOR_SDR_WHITE,
                .MaxFALL = space.sig_avg * PL_COLOR_SDR_WHITE,
            };

            if (!clm->MaxFALL)
                clm->MaxFALL = clm->MaxCLL;
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
    av_frame_remove_side_data(frame, AV_FRAME_DATA_ICC_PROFILE);

    if (!profile.len)
        return;

    const AVFrameSideData *sd;
    sd = av_frame_new_side_data(frame, AV_FRAME_DATA_ICC_PROFILE, profile.len);
    memcpy(sd->data, profile.data, profile.len);
}

static inline void pl_frame_from_avframe(struct pl_frame *out,
                                         const AVFrame *frame)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(frame->format);
    int planes = av_pix_fmt_count_planes(frame->format);
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
            // `pl_upload_avframe`, but for the sake of e.g. users only wishing
            // to map hwaccel frames, this is a good default.
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

    const AVFrameSideData *sd;
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
        if (mdm->has_luminance)
            out->color.sig_peak = av_q2d(mdm->max_luminance) / PL_COLOR_SDR_WHITE;
    }

    // Make sure this value is more or less legal
    if (out->color.sig_peak < 1.0 || out->color.sig_peak > 50.0)
        out->color.sig_peak = 0.0;

#ifdef HAVE_LAV_FILM_GRAIN
    if ((sd = av_frame_get_side_data(frame, AV_FRAME_DATA_FILM_GRAIN_PARAMS))) {
        const AVFilmGrainParams *fgp = (AVFilmGrainParams *) sd->data;
        switch (fgp->type) {
        case AV_FILM_GRAIN_PARAMS_NONE:
            break;
        case AV_FILM_GRAIN_PARAMS_AV1: {
            const AVFilmGrainAOMParams *aom = &fgp->codec.aom;
            struct pl_av1_grain_data *av1 = &out->av1_grain;
            *av1 = (struct pl_av1_grain_data) {
                .grain_seed = fgp->seed,
                .num_points_y = aom->num_y_points,
                .chroma_scaling_from_luma = aom->chroma_scaling_from_luma,
                .num_points_uv = { aom->num_uv_points[0], aom->num_uv_points[1] },
                .scaling_shift = aom->scaling_shift,
                .ar_coeff_lag = aom->ar_coeff_lag,
                .ar_coeff_shift = aom->ar_coeff_shift,
                .grain_scale_shift = aom->grain_scale_shift,
                .uv_mult = { aom->uv_mult[0], aom->uv_mult[1] },
                .uv_mult_luma = { aom->uv_mult_luma[0], aom->uv_mult_luma[1] },
                .uv_offset = { aom->uv_offset[0], aom->uv_offset[1] },
                .overlap = aom->overlap_flag,
            };

            assert(sizeof(av1->ar_coeffs_uv) == sizeof(aom->ar_coeffs_uv));
            memcpy(av1->points_y, aom->y_points, sizeof(av1->points_y));
            memcpy(av1->points_uv, aom->uv_points, sizeof(av1->points_uv));
            memcpy(av1->ar_coeffs_y, aom->ar_coeffs_y, sizeof(av1->ar_coeffs_y));
            memcpy(av1->ar_coeffs_uv, aom->ar_coeffs_uv, sizeof(av1->ar_coeffs_uv));
            break;
        }
        }
    }
#endif // HAVE_LAV_AV1_GRAIN

    for (int p = 0; p < out->num_planes; p++) {
        struct pl_plane *plane = &out->planes[p];

        // Fill in the component mapping array
        for (int c = 0; c < desc->nb_components; c++) {
            if (desc->comp[c].plane != p)
                continue;

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

static inline bool pl_frame_recreate_from_avframe(const struct pl_gpu *gpu,
                                                  struct pl_frame *out,
                                                  const struct pl_tex *tex[4],
                                                  const AVFrame *frame)
{
    pl_frame_from_avframe(out, frame);

    struct pl_plane_data data[4] = {0};
    int planes = pl_plane_data_from_pixfmt(data, &out->repr.bits, frame->format);
    if (!planes)
        return false;

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(frame->format);
    for (int p = 0; p < planes; p++) {
        bool is_chroma = p == 1 || p == 2; // matches lavu logic
        data[p].width = frame->width >> (is_chroma ? desc->log2_chroma_w : 0);
        data[p].height = frame->height >> (is_chroma ? desc->log2_chroma_h : 0);

        if (!pl_recreate_plane(gpu, &out->planes[p], &tex[p], &data[p]))
            return false;
    }

    return true;
}

static void pl_avbuffer_free(void *priv)
{
    AVBufferRef *buf = priv;
    av_buffer_unref(&buf);
}

static inline bool pl_upload_avframe(const struct pl_gpu *gpu,
                                     struct pl_frame *out,
                                     const struct pl_tex *tex[4],
                                     const AVFrame *frame)
{
    pl_frame_from_avframe(out, frame);

    // TODO: support HW-accelerated formats (e.g. wrapping vulkan frames,
    // importing DRM buffers, and so on)

    struct pl_plane_data data[4] = {0};
    int planes = pl_plane_data_from_pixfmt(data, &out->repr.bits, frame->format);
    if (!planes)
        return false;

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(frame->format);
    for (int p = 0; p < planes; p++) {
        bool is_chroma = p == 1 || p == 2; // matches lavu logic
        data[p].width = frame->width >> (is_chroma ? desc->log2_chroma_w : 0);
        data[p].height = frame->height >> (is_chroma ? desc->log2_chroma_h : 0);
        data[p].row_stride = frame->linesize[p];
        data[p].pixels = frame->data[p];

        // Try using an asynchronous upload if possible, by taking a new ref to
        // the plane data
        if (frame->buf[p]) {
            data[p].callback = pl_avbuffer_free;
            data[p].priv = av_buffer_ref(frame->buf[p]);
        }

        if (!pl_upload_plane(gpu, &out->planes[p], &tex[p], &data[p]))
            return false;
    }

    return true;
}

static void pl_done_cb(void *priv)
{
    bool *status = priv;
    *status = true;
}

static inline bool pl_download_avframe(const struct pl_gpu *gpu,
                                       const struct pl_frame *frame,
                                       AVFrame *out_frame)
{
    bool done[4] = {0};
    if (frame->num_planes != av_pix_fmt_count_planes(out_frame->format))
        return false;

    for (int p = 0; p < frame->num_planes; p++) {
        size_t texel_size = frame->planes[p].texture->params.format->texel_size;
        bool ok = pl_tex_download(gpu, &(struct pl_tex_transfer_params) {
            .tex = frame->planes[p].texture,
            .stride_w = out_frame->linesize[p] / texel_size,
            .ptr = out_frame->data[p],
            // Use synchronous transfer for the last plane
            .callback = (p+1) < frame->num_planes ? pl_done_cb : NULL,
            .priv = &done[p],
        });

        if (!ok)
            return false;
    }

    for (int p = 0; p < frame->num_planes - 1; p++) {
        while (!done[p])
            pl_tex_poll(gpu, frame->planes[p].texture, UINT64_MAX);
    }

    return true;
}

#endif // LIBPLACEBO_LIBAV_H_
