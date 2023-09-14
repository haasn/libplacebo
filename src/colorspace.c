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

#include <math.h>

#include "common.h"
#include "hash.h"

#include <libplacebo/colorspace.h>
#include <libplacebo/tone_mapping.h>

bool pl_color_system_is_ycbcr_like(enum pl_color_system sys)
{
    switch (sys) {
    case PL_COLOR_SYSTEM_UNKNOWN:
    case PL_COLOR_SYSTEM_RGB:
    case PL_COLOR_SYSTEM_XYZ:
        return false;
    case PL_COLOR_SYSTEM_BT_601:
    case PL_COLOR_SYSTEM_BT_709:
    case PL_COLOR_SYSTEM_SMPTE_240M:
    case PL_COLOR_SYSTEM_BT_2020_NC:
    case PL_COLOR_SYSTEM_BT_2020_C:
    case PL_COLOR_SYSTEM_BT_2100_PQ:
    case PL_COLOR_SYSTEM_BT_2100_HLG:
    case PL_COLOR_SYSTEM_DOLBYVISION:
    case PL_COLOR_SYSTEM_YCGCO:
        return true;
    case PL_COLOR_SYSTEM_COUNT: break;
    };

    pl_unreachable();
}

bool pl_color_system_is_linear(enum pl_color_system sys)
{
    switch (sys) {
    case PL_COLOR_SYSTEM_UNKNOWN:
    case PL_COLOR_SYSTEM_RGB:
    case PL_COLOR_SYSTEM_BT_601:
    case PL_COLOR_SYSTEM_BT_709:
    case PL_COLOR_SYSTEM_SMPTE_240M:
    case PL_COLOR_SYSTEM_BT_2020_NC:
    case PL_COLOR_SYSTEM_YCGCO:
        return true;
    case PL_COLOR_SYSTEM_BT_2020_C:
    case PL_COLOR_SYSTEM_BT_2100_PQ:
    case PL_COLOR_SYSTEM_BT_2100_HLG:
    case PL_COLOR_SYSTEM_DOLBYVISION:
    case PL_COLOR_SYSTEM_XYZ:
        return false;
    case PL_COLOR_SYSTEM_COUNT: break;
    };

    pl_unreachable();
}

enum pl_color_system pl_color_system_guess_ycbcr(int width, int height)
{
    if (width >= 1280 || height > 576) {
        // Typical HD content
        return PL_COLOR_SYSTEM_BT_709;
    } else {
        // Typical SD content
        return PL_COLOR_SYSTEM_BT_601;
    }
}

bool pl_bit_encoding_equal(const struct pl_bit_encoding *b1,
                           const struct pl_bit_encoding *b2)
{
    return b1->sample_depth == b2->sample_depth &&
           b1->color_depth  == b2->color_depth &&
           b1->bit_shift    == b2->bit_shift;
}

const struct pl_color_repr pl_color_repr_unknown = {0};

const struct pl_color_repr pl_color_repr_rgb = {
    .sys    = PL_COLOR_SYSTEM_RGB,
    .levels = PL_COLOR_LEVELS_FULL,
};

const struct pl_color_repr pl_color_repr_sdtv = {
    .sys    = PL_COLOR_SYSTEM_BT_601,
    .levels = PL_COLOR_LEVELS_LIMITED,
};

const struct pl_color_repr pl_color_repr_hdtv = {
    .sys    = PL_COLOR_SYSTEM_BT_709,
    .levels = PL_COLOR_LEVELS_LIMITED,
};

const struct pl_color_repr pl_color_repr_uhdtv = {
    .sys    = PL_COLOR_SYSTEM_BT_2020_NC,
    .levels = PL_COLOR_LEVELS_LIMITED,
};

const struct pl_color_repr pl_color_repr_jpeg = {
    .sys    = PL_COLOR_SYSTEM_BT_601,
    .levels = PL_COLOR_LEVELS_FULL,
};

bool pl_color_repr_equal(const struct pl_color_repr *c1,
                         const struct pl_color_repr *c2)
{
    return c1->sys    == c2->sys &&
           c1->levels == c2->levels &&
           c1->alpha  == c2->alpha &&
           c1->dovi   == c2->dovi &&
           pl_bit_encoding_equal(&c1->bits, &c2->bits);
}

static struct pl_bit_encoding pl_bit_encoding_merge(const struct pl_bit_encoding *orig,
                                                    const struct pl_bit_encoding *new)
{
    return (struct pl_bit_encoding) {
        .sample_depth = PL_DEF(orig->sample_depth, new->sample_depth),
        .color_depth  = PL_DEF(orig->color_depth,  new->color_depth),
        .bit_shift    = PL_DEF(orig->bit_shift,    new->bit_shift),
    };
}

void pl_color_repr_merge(struct pl_color_repr *orig, const struct pl_color_repr *new)
{
    *orig = (struct pl_color_repr) {
        .sys    = PL_DEF(orig->sys,    new->sys),
        .levels = PL_DEF(orig->levels, new->levels),
        .alpha  = PL_DEF(orig->alpha,  new->alpha),
        .dovi   = PL_DEF(orig->dovi,   new->dovi),
        .bits   = pl_bit_encoding_merge(&orig->bits, &new->bits),
    };
}

enum pl_color_levels pl_color_levels_guess(const struct pl_color_repr *repr)
{
    if (repr->sys == PL_COLOR_SYSTEM_DOLBYVISION)
        return PL_COLOR_LEVELS_FULL;

    if (repr->levels)
        return repr->levels;

    return pl_color_system_is_ycbcr_like(repr->sys)
                ? PL_COLOR_LEVELS_LIMITED
                : PL_COLOR_LEVELS_FULL;
}

float pl_color_repr_normalize(struct pl_color_repr *repr)
{
    float scale = 1.0;
    struct pl_bit_encoding *bits = &repr->bits;

    if (bits->bit_shift) {
        scale /= (1LL << bits->bit_shift);
        bits->bit_shift = 0;
    }

    // If one of these is set but not the other, use the set one
    int tex_bits = PL_DEF(bits->sample_depth, 8);
    int col_bits = PL_DEF(bits->color_depth, tex_bits);
    tex_bits = PL_DEF(tex_bits, col_bits);

    if (pl_color_levels_guess(repr) == PL_COLOR_LEVELS_LIMITED) {
        // Limit range is always shifted directly
        scale *= (float) (1LL << tex_bits) / (1LL << col_bits);
    } else {
        // Full range always uses the full range available
        scale *= ((1LL << tex_bits) - 1.) / ((1LL << col_bits) - 1.);
    }

    bits->color_depth = bits->sample_depth;
    return scale;
}

bool pl_color_primaries_is_wide_gamut(enum pl_color_primaries prim)
{
    switch (prim) {
    case PL_COLOR_PRIM_UNKNOWN:
    case PL_COLOR_PRIM_BT_601_525:
    case PL_COLOR_PRIM_BT_601_625:
    case PL_COLOR_PRIM_BT_709:
    case PL_COLOR_PRIM_BT_470M:
    case PL_COLOR_PRIM_EBU_3213:
        return false;
    case PL_COLOR_PRIM_BT_2020:
    case PL_COLOR_PRIM_APPLE:
    case PL_COLOR_PRIM_ADOBE:
    case PL_COLOR_PRIM_PRO_PHOTO:
    case PL_COLOR_PRIM_CIE_1931:
    case PL_COLOR_PRIM_DCI_P3:
    case PL_COLOR_PRIM_DISPLAY_P3:
    case PL_COLOR_PRIM_V_GAMUT:
    case PL_COLOR_PRIM_S_GAMUT:
    case PL_COLOR_PRIM_FILM_C:
    case PL_COLOR_PRIM_ACES_AP0:
    case PL_COLOR_PRIM_ACES_AP1:
        return true;
    case PL_COLOR_PRIM_COUNT: break;
    }

    pl_unreachable();
}

enum pl_color_primaries pl_color_primaries_guess(int width, int height)
{
    // HD content
    if (width >= 1280 || height > 576)
        return PL_COLOR_PRIM_BT_709;

    switch (height) {
    case 576: // Typical PAL content, including anamorphic/squared
        return PL_COLOR_PRIM_BT_601_625;

    case 480: // Typical NTSC content, including squared
    case 486: // NTSC Pro or anamorphic NTSC
        return PL_COLOR_PRIM_BT_601_525;

    default: // No good metric, just pick BT.709 to minimize damage
        return PL_COLOR_PRIM_BT_709;
    }
}

// HLG 75% value (scene-referred)
#define HLG_75 3.17955

float pl_color_transfer_nominal_peak(enum pl_color_transfer trc)
{
    switch (trc) {
    case PL_COLOR_TRC_UNKNOWN:
    case PL_COLOR_TRC_BT_1886:
    case PL_COLOR_TRC_SRGB:
    case PL_COLOR_TRC_LINEAR:
    case PL_COLOR_TRC_GAMMA18:
    case PL_COLOR_TRC_GAMMA20:
    case PL_COLOR_TRC_GAMMA22:
    case PL_COLOR_TRC_GAMMA24:
    case PL_COLOR_TRC_GAMMA26:
    case PL_COLOR_TRC_GAMMA28:
    case PL_COLOR_TRC_PRO_PHOTO:
    case PL_COLOR_TRC_ST428:
        return 1.0;
    case PL_COLOR_TRC_PQ:       return 10000.0 / PL_COLOR_SDR_WHITE;
    case PL_COLOR_TRC_HLG:      return 12.0 / HLG_75;
    case PL_COLOR_TRC_V_LOG:    return 46.0855;
    case PL_COLOR_TRC_S_LOG1:   return 6.52;
    case PL_COLOR_TRC_S_LOG2:   return 9.212;
    case PL_COLOR_TRC_COUNT: break;
    }

    pl_unreachable();
}

const struct pl_hdr_metadata pl_hdr_metadata_empty = {0};
const struct pl_hdr_metadata pl_hdr_metadata_hdr10 ={
    .prim = {
        .red   = {0.708,    0.292},
        .green = {0.170,    0.797},
        .blue  = {0.131,    0.046},
        .white = {0.31271,  0.32902},
    },
    .min_luma = 0,
    .max_luma = 10000,
    .max_cll  = 10000,
    .max_fall = 0, // unknown
};

static const float PQ_M1 = 2610./4096 * 1./4,
                   PQ_M2 = 2523./4096 * 128,
                   PQ_C1 = 3424./4096,
                   PQ_C2 = 2413./4096 * 32,
                   PQ_C3 = 2392./4096 * 32;

float pl_hdr_rescale(enum pl_hdr_scaling from, enum pl_hdr_scaling to, float x)
{
    if (from == to)
        return x;
    if (!x) // micro-optimization for common value
        return x;

    x = fmaxf(x, 0.0f);

    // Convert input to PL_SCALE_RELATIVE
    switch (from) {
    case PL_HDR_PQ:
        x = powf(x, 1.0f / PQ_M2);
        x = fmaxf(x - PQ_C1, 0.0f) / (PQ_C2 - PQ_C3 * x);
        x = powf(x, 1.0f / PQ_M1);
        x *= 10000.0f;
        // fall through
    case PL_HDR_NITS:
        x /= PL_COLOR_SDR_WHITE;
        // fall through
    case PL_HDR_NORM:
        goto output;
    case PL_HDR_SQRT:
        x *= x;
        goto output;
    case PL_HDR_SCALING_COUNT:
        break;
    }

    pl_unreachable();

output:
    // Convert PL_SCALE_RELATIVE to output
    switch (to) {
    case PL_HDR_NORM:
        return x;
    case PL_HDR_SQRT:
        return sqrtf(x);
    case PL_HDR_NITS:
        return x * PL_COLOR_SDR_WHITE;
    case PL_HDR_PQ:
        x *= PL_COLOR_SDR_WHITE / 10000.0f;
        x = powf(x, PQ_M1);
        x = (PQ_C1 + PQ_C2 * x) / (1.0f + PQ_C3 * x);
        x = powf(x, PQ_M2);
        return x;
    case PL_HDR_SCALING_COUNT:
        break;
    }

    pl_unreachable();
}

static inline bool pl_hdr_bezier_equal(const struct pl_hdr_bezier *a,
                                       const struct pl_hdr_bezier *b)
{
    return a->target_luma == b->target_luma &&
           a->knee_x      == b->knee_x &&
           a->knee_y      == b->knee_y &&
           a->num_anchors == b->num_anchors &&
           !memcmp(a->anchors, b->anchors, sizeof(a->anchors[0]) * a->num_anchors);
}

bool pl_hdr_metadata_equal(const struct pl_hdr_metadata *a,
                           const struct pl_hdr_metadata *b)
{
    return pl_raw_primaries_equal(&a->prim, &b->prim) &&
           a->min_luma == b->min_luma &&
           a->max_luma == b->max_luma &&
           a->max_cll  == b->max_cll  &&
           a->max_fall == b->max_fall &&
           a->scene_max[0] == b->scene_max[0] &&
           a->scene_max[1] == b->scene_max[1] &&
           a->scene_max[2] == b->scene_max[2] &&
           a->scene_avg == b->scene_avg &&
           pl_hdr_bezier_equal(&a->ootf, &b->ootf) &&
           a->max_pq_y == b->max_pq_y &&
           a->avg_pq_y == b->avg_pq_y;
}

void pl_hdr_metadata_merge(struct pl_hdr_metadata *orig,
                           const struct pl_hdr_metadata *update)
{
    pl_raw_primaries_merge(&orig->prim, &update->prim);
    if (!orig->min_luma)
        orig->min_luma = update->min_luma;
    if (!orig->max_luma)
        orig->max_luma = update->max_luma;
    if (!orig->max_cll)
        orig->max_cll = update->max_cll;
    if (!orig->max_fall)
        orig->max_fall = update->max_fall;
    if (!orig->scene_max[1])
        memcpy(orig->scene_max, update->scene_max, sizeof(orig->scene_max));
    if (!orig->scene_avg)
        orig->scene_avg = update->scene_avg;
    if (!orig->ootf.target_luma)
        orig->ootf = update->ootf;
    if (!orig->max_pq_y)
        orig->max_pq_y = update->max_pq_y;
    if (!orig->avg_pq_y)
        orig->avg_pq_y = update->avg_pq_y;
}

bool pl_hdr_metadata_contains(const struct pl_hdr_metadata *data,
                              enum pl_hdr_metadata_type type)
{
    bool has_hdr10 = data->max_luma;
    bool has_hdr10plus = data->scene_avg && (data->scene_max[0] ||
                                             data->scene_max[1] ||
                                             data->scene_max[2]);
    bool has_cie_y = data->max_pq_y && data->avg_pq_y;

    switch (type) {
    case PL_HDR_METADATA_NONE:          return true;
    case PL_HDR_METADATA_ANY:           return has_hdr10 || has_hdr10plus || has_cie_y;
    case PL_HDR_METADATA_HDR10:         return has_hdr10;
    case PL_HDR_METADATA_HDR10PLUS:     return has_hdr10plus;
    case PL_HDR_METADATA_CIE_Y:         return has_cie_y;
    case PL_HDR_METADATA_TYPE_COUNT:    break;
    }

    pl_unreachable();
}

const struct pl_color_space pl_color_space_unknown = {0};

const struct pl_color_space pl_color_space_srgb = {
    .primaries = PL_COLOR_PRIM_BT_709,
    .transfer  = PL_COLOR_TRC_SRGB,
};

const struct pl_color_space pl_color_space_bt709 = {
    .primaries = PL_COLOR_PRIM_BT_709,
    .transfer  = PL_COLOR_TRC_BT_1886,
};

const struct pl_color_space pl_color_space_hdr10 = {
    .primaries = PL_COLOR_PRIM_BT_2020,
    .transfer  = PL_COLOR_TRC_PQ,
};

const struct pl_color_space pl_color_space_bt2020_hlg = {
    .primaries = PL_COLOR_PRIM_BT_2020,
    .transfer  = PL_COLOR_TRC_HLG,
};

const struct pl_color_space pl_color_space_monitor = {
    .primaries = PL_COLOR_PRIM_BT_709, // sRGB primaries
    .transfer  = PL_COLOR_TRC_UNKNOWN, // unknown SDR response
};

bool pl_color_space_is_hdr(const struct pl_color_space *csp)
{
    return csp->hdr.max_luma > PL_COLOR_SDR_WHITE ||
           pl_color_transfer_is_hdr(csp->transfer);
}

bool pl_color_space_is_black_scaled(const struct pl_color_space *csp)
{
    switch (csp->transfer) {
    case PL_COLOR_TRC_UNKNOWN:
    case PL_COLOR_TRC_SRGB:
    case PL_COLOR_TRC_LINEAR:
    case PL_COLOR_TRC_GAMMA18:
    case PL_COLOR_TRC_GAMMA20:
    case PL_COLOR_TRC_GAMMA22:
    case PL_COLOR_TRC_GAMMA24:
    case PL_COLOR_TRC_GAMMA26:
    case PL_COLOR_TRC_GAMMA28:
    case PL_COLOR_TRC_PRO_PHOTO:
    case PL_COLOR_TRC_ST428:
    case PL_COLOR_TRC_HLG:
        return true;

    case PL_COLOR_TRC_BT_1886:
    case PL_COLOR_TRC_PQ:
    case PL_COLOR_TRC_V_LOG:
    case PL_COLOR_TRC_S_LOG1:
    case PL_COLOR_TRC_S_LOG2:
        return false;

    case PL_COLOR_TRC_COUNT: break;
    }

    pl_unreachable();
}

void pl_color_space_merge(struct pl_color_space *orig,
                          const struct pl_color_space *new)
{
    if (!orig->primaries)
        orig->primaries = new->primaries;
    if (!orig->transfer)
        orig->transfer = new->transfer;
    pl_hdr_metadata_merge(&orig->hdr, &new->hdr);
}

bool pl_color_space_equal(const struct pl_color_space *c1,
                          const struct pl_color_space *c2)
{
    return c1->primaries == c2->primaries &&
           c1->transfer  == c2->transfer &&
           pl_hdr_metadata_equal(&c1->hdr, &c2->hdr);
}

// Estimates luminance from maxRGB by looking at how monochromatic MaxSCL is
static void luma_from_maxrgb(const struct pl_color_space *csp,
                             enum pl_hdr_scaling scaling,
                             float *out_max, float *out_avg)
{
    const float maxscl = PL_MAX3(csp->hdr.scene_max[0],
                                 csp->hdr.scene_max[1],
                                 csp->hdr.scene_max[2]);
    if (!maxscl)
        return;

    struct pl_raw_primaries prim = csp->hdr.prim;
    pl_raw_primaries_merge(&prim, pl_raw_primaries_get(csp->primaries));
    const pl_matrix3x3 rgb2xyz = pl_get_rgb2xyz_matrix(&prim);

    const float max_luma = rgb2xyz.m[1][0] * csp->hdr.scene_max[0] +
                           rgb2xyz.m[1][1] * csp->hdr.scene_max[1] +
                           rgb2xyz.m[1][2] * csp->hdr.scene_max[2];

    const float coef = max_luma / maxscl;
    *out_max = pl_hdr_rescale(PL_HDR_NITS, scaling, max_luma);
    *out_avg = pl_hdr_rescale(PL_HDR_NITS, scaling, coef * csp->hdr.scene_avg);
}

static inline bool metadata_compat(enum pl_hdr_metadata_type metadata,
                                   enum pl_hdr_metadata_type compat)
{
    return metadata == PL_HDR_METADATA_ANY || metadata == compat;
}

void pl_color_space_nominal_luma_ex(const struct pl_nominal_luma_params *params)
{
    if (!params || (!params->out_min && !params->out_max && !params->out_avg))
        return;

    const struct pl_color_space *csp = params->color;
    const enum pl_hdr_scaling scaling = params->scaling;

    float min_luma = 0, max_luma = 0, avg_luma = 0;
    if (params->metadata != PL_HDR_METADATA_NONE) {
        // Initialize from static HDR10 metadata, in all cases
        min_luma = pl_hdr_rescale(PL_HDR_NITS, scaling, csp->hdr.min_luma);
        max_luma = pl_hdr_rescale(PL_HDR_NITS, scaling, csp->hdr.max_luma);
    }

    if (metadata_compat(params->metadata, PL_HDR_METADATA_HDR10PLUS) &&
        pl_hdr_metadata_contains(&csp->hdr, PL_HDR_METADATA_HDR10PLUS))
    {
        luma_from_maxrgb(csp, scaling, &max_luma, &avg_luma);
    }

    if (metadata_compat(params->metadata, PL_HDR_METADATA_CIE_Y) &&
        pl_hdr_metadata_contains(&csp->hdr, PL_HDR_METADATA_CIE_Y))
    {
        max_luma = pl_hdr_rescale(PL_HDR_PQ, scaling, csp->hdr.max_pq_y);
        avg_luma = pl_hdr_rescale(PL_HDR_PQ, scaling, csp->hdr.avg_pq_y);
    }

    // Clamp to sane value range
    const float hdr_min = pl_hdr_rescale(PL_HDR_NITS, scaling, PL_COLOR_HDR_BLACK);
    const float hdr_max = pl_hdr_rescale(PL_HDR_PQ,   scaling, 1.0f);
    max_luma = max_luma ? PL_CLAMP(max_luma, hdr_min, hdr_max) : 0;
    min_luma = min_luma ? PL_CLAMP(min_luma, hdr_min, hdr_max) : 0;
    if ((max_luma && min_luma >= max_luma) || min_luma >= hdr_max)
        min_luma = max_luma = 0; // sanity

    // PQ is always scaled down to absolute black, ignoring HDR metadata
    if (csp->transfer == PL_COLOR_TRC_PQ)
        min_luma = hdr_min;

    // Baseline/fallback metadata, inferred entirely from the colorspace
    // description and built-in default assumptions
    if (!max_luma) {
        if (csp->transfer == PL_COLOR_TRC_HLG) {
            max_luma = pl_hdr_rescale(PL_HDR_NITS, scaling, PL_COLOR_HLG_PEAK);
        } else {
            const float peak = pl_color_transfer_nominal_peak(csp->transfer);
            max_luma = pl_hdr_rescale(PL_HDR_NORM, scaling, peak);
        }
    }

    if (!min_luma) {
        if (pl_color_transfer_is_hdr(csp->transfer)) {
            min_luma = hdr_min;
        } else {
            const float peak = pl_hdr_rescale(scaling, PL_HDR_NITS, max_luma);
            min_luma = pl_hdr_rescale(PL_HDR_NITS, scaling,
                                      peak / PL_COLOR_SDR_CONTRAST);
        }
    }

    if (avg_luma)
        avg_luma = PL_CLAMP(avg_luma, min_luma, max_luma); // sanity

    if (params->out_min)
        *params->out_min = min_luma;
    if (params->out_max)
        *params->out_max = max_luma;
    if (params->out_avg)
        *params->out_avg = avg_luma;
}

void pl_color_space_nominal_luma(const struct pl_color_space *csp,
                                 float *out_min, float *out_max)
{
    pl_color_space_nominal_luma_ex(pl_nominal_luma_params(
        .color      = csp,
        .metadata   = PL_HDR_METADATA_ANY,
        .scaling    = PL_HDR_NORM,
        .out_min    = out_min,
        .out_max    = out_max,
    ));
}

void pl_color_space_infer(struct pl_color_space *space)
{
    if (!space->primaries)
        space->primaries = PL_COLOR_PRIM_BT_709;
    if (!space->transfer)
        space->transfer = PL_COLOR_TRC_BT_1886;

    // Sanitize the static HDR metadata
    pl_color_space_nominal_luma_ex(pl_nominal_luma_params(
        .color      = space,
        .metadata   = PL_HDR_METADATA_HDR10,
        .scaling    = PL_HDR_NITS,
        .out_max    = &space->hdr.max_luma,
        // Preserve tagged minimum
        .out_min    = space->hdr.min_luma ? NULL : &space->hdr.min_luma,
    ));

    // Default the signal color space based on the nominal raw primaries
    if (!pl_primaries_valid(&space->hdr.prim))
        space->hdr.prim = *pl_raw_primaries_get(space->primaries);
}

static void infer_both_ref(struct pl_color_space *space,
                           struct pl_color_space *ref)
{
    pl_color_space_infer(ref);

    if (!space->primaries) {
        if (pl_color_primaries_is_wide_gamut(ref->primaries)) {
            space->primaries = PL_COLOR_PRIM_BT_709;
        } else {
            space->primaries = ref->primaries;
        }
    }

    if (!space->transfer) {
        switch (ref->transfer) {
        case PL_COLOR_TRC_UNKNOWN:
        case PL_COLOR_TRC_COUNT:
            pl_unreachable();
        case PL_COLOR_TRC_BT_1886:
        case PL_COLOR_TRC_SRGB:
        case PL_COLOR_TRC_GAMMA22:
            // Re-use input transfer curve to avoid small adaptations
            space->transfer = ref->transfer;
            break;
        case PL_COLOR_TRC_PQ:
        case PL_COLOR_TRC_HLG:
        case PL_COLOR_TRC_V_LOG:
        case PL_COLOR_TRC_S_LOG1:
        case PL_COLOR_TRC_S_LOG2:
            // Pick BT.1886 model because it models SDR contrast accurately,
            // and we need contrast information for tone mapping
            space->transfer = PL_COLOR_TRC_BT_1886;
            break;
        case PL_COLOR_TRC_PRO_PHOTO:
            // ProPhotoRGB and sRGB are both piecewise with linear slope
            space->transfer = PL_COLOR_TRC_SRGB;
            break;
        case PL_COLOR_TRC_LINEAR:
        case PL_COLOR_TRC_GAMMA18:
        case PL_COLOR_TRC_GAMMA20:
        case PL_COLOR_TRC_GAMMA24:
        case PL_COLOR_TRC_GAMMA26:
        case PL_COLOR_TRC_GAMMA28:
        case PL_COLOR_TRC_ST428:
            // Pick pure power output curve to avoid introducing black crush
            space->transfer = PL_COLOR_TRC_GAMMA22;
            break;
        }
    }

    // Infer the remaining fields after making the above choices
    pl_color_space_infer(space);
}

void pl_color_space_infer_ref(struct pl_color_space *space,
                              const struct pl_color_space *refp)
{
    // Make a copy of `refp` to infer missing values first
    struct pl_color_space ref = *refp;
    infer_both_ref(space, &ref);
}

void pl_color_space_infer_map(struct pl_color_space *src,
                              struct pl_color_space *dst)
{
    bool unknown_src_contrast = !src->hdr.min_luma;
    bool unknown_dst_contrast = !dst->hdr.min_luma;

    infer_both_ref(dst, src);

    // If the src has an unspecified gamma curve with dynamic black scaling,
    // default it to match the dst colorspace contrast. This does not matter in
    // most cases, but ensures that BT.1886 is tuned to the appropriate black
    // point by default.
    bool dynamic_src_contrast = pl_color_space_is_black_scaled(src) ||
                                src->transfer == PL_COLOR_TRC_BT_1886;
    if (unknown_src_contrast && dynamic_src_contrast)
        src->hdr.min_luma = dst->hdr.min_luma;

    // Do the same in reverse if both src and dst are SDR curves
    bool src_is_sdr = !pl_color_space_is_hdr(src);
    bool dst_is_sdr = !pl_color_space_is_hdr(dst);
    if (unknown_dst_contrast && src_is_sdr && dst_is_sdr)
        dst->hdr.min_luma = src->hdr.min_luma;

    // If the src is HLG and the output is HDR, tune the HLG peak to the output
    if (src->transfer == PL_COLOR_TRC_HLG && pl_color_space_is_hdr(dst))
        src->hdr.max_luma = dst->hdr.max_luma;
}

const struct pl_color_adjustment pl_color_adjustment_neutral = {
    PL_COLOR_ADJUSTMENT_NEUTRAL
};

void pl_chroma_location_offset(enum pl_chroma_location loc, float *x, float *y)
{
    *x = *y = 0;

    // This is the majority of subsampled chroma content out there
    loc = PL_DEF(loc, PL_CHROMA_LEFT);

    switch (loc) {
    case PL_CHROMA_LEFT:
    case PL_CHROMA_TOP_LEFT:
    case PL_CHROMA_BOTTOM_LEFT:
        *x = -0.5;
        break;
    default: break;
    }

    switch (loc) {
    case PL_CHROMA_TOP_LEFT:
    case PL_CHROMA_TOP_CENTER:
        *y = -0.5;
        break;
    default: break;
    }

    switch (loc) {
    case PL_CHROMA_BOTTOM_LEFT:
    case PL_CHROMA_BOTTOM_CENTER:
        *y = 0.5;
        break;
    default: break;
    }
}

struct pl_cie_xy pl_white_from_temp(float temp)
{
    temp = PL_CLAMP(temp, 2500, 25000);

    double ti = 1000.0 / temp, ti2 = ti * ti, ti3 = ti2 * ti, x;
    if (temp <= 7000) {
        x = -4.6070 * ti3 + 2.9678 * ti2 + 0.09911 * ti + 0.244063;
    } else {
        x = -2.0064 * ti3 + 1.9018 * ti2 + 0.24748 * ti + 0.237040;
    }

    return (struct pl_cie_xy) {
        .x = x,
        .y = -3 * (x*x) + 2.87 * x - 0.275,
    };
}

bool pl_raw_primaries_equal(const struct pl_raw_primaries *a,
                            const struct pl_raw_primaries *b)
{
    return pl_cie_xy_equal(&a->red,   &b->red)   &&
           pl_cie_xy_equal(&a->green, &b->green) &&
           pl_cie_xy_equal(&a->blue,  &b->blue)  &&
           pl_cie_xy_equal(&a->white, &b->white);
}

bool pl_raw_primaries_similar(const struct pl_raw_primaries *a,
                              const struct pl_raw_primaries *b)
{
    float delta = fabsf(a->red.x   - b->red.x)   +
                  fabsf(a->red.y   - b->red.y)   +
                  fabsf(a->green.x - b->green.x) +
                  fabsf(a->green.y - b->green.y) +
                  fabsf(a->blue.x  - b->blue.x)  +
                  fabsf(a->blue.y  - b->blue.y)  +
                  fabsf(a->white.x - b->white.x) +
                  fabsf(a->white.y - b->white.y);

    return delta < 0.001;
}

void pl_raw_primaries_merge(struct pl_raw_primaries *orig,
                            const struct pl_raw_primaries *update)
{
    union {
        struct pl_raw_primaries prim;
        float raw[8];
    } *pa = (void *) orig,
      *pb = (void *) update;

    pl_static_assert(sizeof(*pa) == sizeof(*orig));
    for (int i = 0; i < PL_ARRAY_SIZE(pa->raw); i++)
        pa->raw[i] = PL_DEF(pa->raw[i], pb->raw[i]);
}

const struct pl_raw_primaries *pl_raw_primaries_get(enum pl_color_primaries prim)
{
    /*
    Values from: ITU-R Recommendations BT.470-6, BT.601-7, BT.709-5, BT.2020-0

    https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-6-199811-S!!PDF-E.pdf
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.601-7-201103-I!!PDF-E.pdf
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.709-5-200204-I!!PDF-E.pdf
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2020-0-201208-I!!PDF-E.pdf

    Other colorspaces from https://en.wikipedia.org/wiki/RGB_color_space#Specifications
    */

    // CIE standard illuminant series
#define CIE_D50 {0.3457, 0.3585}
#define CIE_D65 {0.3127, 0.3290}
#define CIE_C   {0.3100, 0.3160}
#define CIE_E   {1.0/3.0, 1.0/3.0}
#define DCI     {0.3140, 0.3510}

    static const struct pl_raw_primaries primaries[] = {
        [PL_COLOR_PRIM_BT_470M] = {
            .red   = {0.670, 0.330},
            .green = {0.210, 0.710},
            .blue  = {0.140, 0.080},
            .white = CIE_C,
        },

        [PL_COLOR_PRIM_BT_601_525] = {
            .red   = {0.630, 0.340},
            .green = {0.310, 0.595},
            .blue  = {0.155, 0.070},
            .white = CIE_D65,
        },
        [PL_COLOR_PRIM_BT_601_625] = {
            .red   = {0.640, 0.330},
            .green = {0.290, 0.600},
            .blue  = {0.150, 0.060},
            .white = CIE_D65,
        },
        [PL_COLOR_PRIM_BT_709] = {
            .red   = {0.640, 0.330},
            .green = {0.300, 0.600},
            .blue  = {0.150, 0.060},
            .white = CIE_D65,
        },
        [PL_COLOR_PRIM_BT_2020] = {
            .red   = {0.708, 0.292},
            .green = {0.170, 0.797},
            .blue  = {0.131, 0.046},
            .white = CIE_D65,
        },
        [PL_COLOR_PRIM_APPLE] = {
            .red   = {0.625, 0.340},
            .green = {0.280, 0.595},
            .blue  = {0.115, 0.070},
            .white = CIE_D65,
        },
        [PL_COLOR_PRIM_ADOBE] = {
            .red   = {0.640, 0.330},
            .green = {0.210, 0.710},
            .blue  = {0.150, 0.060},
            .white = CIE_D65,
        },
        [PL_COLOR_PRIM_PRO_PHOTO] = {
            .red   = {0.7347, 0.2653},
            .green = {0.1596, 0.8404},
            .blue  = {0.0366, 0.0001},
            .white = CIE_D50,
        },
        [PL_COLOR_PRIM_CIE_1931] = {
            .red   = {0.7347, 0.2653},
            .green = {0.2738, 0.7174},
            .blue  = {0.1666, 0.0089},
            .white = CIE_E,
        },
    // From SMPTE RP 431-2
        [PL_COLOR_PRIM_DCI_P3] = {
            .red   = {0.680, 0.320},
            .green = {0.265, 0.690},
            .blue  = {0.150, 0.060},
            .white = DCI,
        },
        [PL_COLOR_PRIM_DISPLAY_P3] = {
            .red   = {0.680, 0.320},
            .green = {0.265, 0.690},
            .blue  = {0.150, 0.060},
            .white = CIE_D65,
        },
    // From Panasonic VARICAM reference manual
        [PL_COLOR_PRIM_V_GAMUT] = {
            .red   = {0.730, 0.280},
            .green = {0.165, 0.840},
            .blue  = {0.100, -0.03},
            .white = CIE_D65,
        },
    // From Sony S-Log reference manual
        [PL_COLOR_PRIM_S_GAMUT] = {
            .red   = {0.730, 0.280},
            .green = {0.140, 0.855},
            .blue  = {0.100, -0.05},
            .white = CIE_D65,
        },
    // From FFmpeg source code
        [PL_COLOR_PRIM_FILM_C] = {
            .red   = {0.681, 0.319},
            .green = {0.243, 0.692},
            .blue  = {0.145, 0.049},
            .white = CIE_C,
        },
        [PL_COLOR_PRIM_EBU_3213] = {
            .red   = {0.630, 0.340},
            .green = {0.295, 0.605},
            .blue  = {0.155, 0.077},
            .white = CIE_D65,
        },
    // From Wikipedia
        [PL_COLOR_PRIM_ACES_AP0] = {
            .red   = {0.7347, 0.2653},
            .green = {0.0000, 1.0000},
            .blue  = {0.0001, -0.0770},
            .white = {0.32168, 0.33767},
        },
        [PL_COLOR_PRIM_ACES_AP1] = {
            .red   = {0.713, 0.293},
            .green = {0.165, 0.830},
            .blue  = {0.128, 0.044},
            .white = {0.32168, 0.33767},
        },
    };

    // This is the default assumption if no colorspace information could
    // be determined, eg. for files which have no video channel.
    if (!prim)
        prim = PL_COLOR_PRIM_BT_709;

    pl_assert(prim < PL_ARRAY_SIZE(primaries));
    return &primaries[prim];
}

// Compute the RGB/XYZ matrix as described here:
// http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
pl_matrix3x3 pl_get_rgb2xyz_matrix(const struct pl_raw_primaries *prim)
{
    pl_matrix3x3 out = {{{0}}};
    float S[3], X[4], Z[4];

    // Convert from CIE xyY to XYZ. Note that Y=1 holds true for all primaries
    X[0] = prim->red.x   / prim->red.y;
    X[1] = prim->green.x / prim->green.y;
    X[2] = prim->blue.x  / prim->blue.y;
    X[3] = prim->white.x / prim->white.y;

    Z[0] = (1 - prim->red.x   - prim->red.y)   / prim->red.y;
    Z[1] = (1 - prim->green.x - prim->green.y) / prim->green.y;
    Z[2] = (1 - prim->blue.x  - prim->blue.y)  / prim->blue.y;
    Z[3] = (1 - prim->white.x - prim->white.y) / prim->white.y;

    // S = XYZ^-1 * W
    for (int i = 0; i < 3; i++) {
        out.m[0][i] = X[i];
        out.m[1][i] = 1;
        out.m[2][i] = Z[i];
    }

    pl_matrix3x3_invert(&out);

    for (int i = 0; i < 3; i++)
        S[i] = out.m[i][0] * X[3] + out.m[i][1] * 1 + out.m[i][2] * Z[3];

    // M = [Sc * XYZc]
    for (int i = 0; i < 3; i++) {
        out.m[0][i] = S[i] * X[i];
        out.m[1][i] = S[i] * 1;
        out.m[2][i] = S[i] * Z[i];
    }

    return out;
}

pl_matrix3x3 pl_get_xyz2rgb_matrix(const struct pl_raw_primaries *prim)
{
    // For simplicity, just invert the rgb2xyz matrix
    pl_matrix3x3 out = pl_get_rgb2xyz_matrix(prim);
    pl_matrix3x3_invert(&out);
    return out;
}

// LMS<-XYZ revised matrix from CIECAM97, based on a linear transform and
// normalized for equal energy on monochrome inputs
static const pl_matrix3x3 m_cat97 = {{
    {  0.8562,  0.3372, -0.1934 },
    { -0.8360,  1.8327,  0.0033 },
    {  0.0357, -0.0469,  1.0112 },
}};

// M := M * XYZd<-XYZs
static void apply_chromatic_adaptation(struct pl_cie_xy src,
                                       struct pl_cie_xy dest,
                                       pl_matrix3x3 *mat)
{
    // If the white points are nearly identical, this is a wasteful identity
    // operation.
    if (fabs(src.x - dest.x) < 1e-6 && fabs(src.y - dest.y) < 1e-6)
        return;

    // XYZd<-XYZs = Ma^-1 * (I*[Cd/Cs]) * Ma
    // http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html
    // For Ma, we use the CIECAM97 revised (linear) matrix
    float C[3][2];

    for (int i = 0; i < 3; i++) {
        // source cone
        C[i][0] = m_cat97.m[i][0] * pl_cie_X(src)
                + m_cat97.m[i][1] * 1
                + m_cat97.m[i][2] * pl_cie_Z(src);

        // dest cone
        C[i][1] = m_cat97.m[i][0] * pl_cie_X(dest)
                + m_cat97.m[i][1] * 1
                + m_cat97.m[i][2] * pl_cie_Z(dest);
    }

    // tmp := I * [Cd/Cs] * Ma
    pl_matrix3x3 tmp = {0};
    for (int i = 0; i < 3; i++)
        tmp.m[i][i] = C[i][1] / C[i][0];

    pl_matrix3x3_mul(&tmp, &m_cat97);

    // M := M * Ma^-1 * tmp
    pl_matrix3x3 ma_inv = m_cat97;
    pl_matrix3x3_invert(&ma_inv);
    pl_matrix3x3_mul(mat, &ma_inv);
    pl_matrix3x3_mul(mat, &tmp);
}

pl_matrix3x3 pl_get_adaptation_matrix(struct pl_cie_xy src, struct pl_cie_xy dst)
{
    // Use BT.709 primaries (with chosen white point) as an XYZ reference
    struct pl_raw_primaries csp = *pl_raw_primaries_get(PL_COLOR_PRIM_BT_709);
    csp.white = src;

    pl_matrix3x3 rgb2xyz = pl_get_rgb2xyz_matrix(&csp);
    pl_matrix3x3 xyz2rgb = rgb2xyz;
    pl_matrix3x3_invert(&xyz2rgb);

    apply_chromatic_adaptation(src, dst, &xyz2rgb);
    pl_matrix3x3_mul(&xyz2rgb, &rgb2xyz);
    return xyz2rgb;
}

pl_matrix3x3 pl_ipt_rgb2lms(const struct pl_raw_primaries *prim)
{
    static const pl_matrix3x3 hpe = {{ // HPE XYZ->LMS (D65) method
        {  0.40024f, 0.70760f, -0.08081f },
        { -0.22630f, 1.16532f,  0.04570f },
        {  0.00000f, 0.00000f,  0.91822f },
    }};

    const float c = 0.04; // 4% crosstalk
    pl_matrix3x3 m = {{
        { 1 - 2*c,       c,       c },
        {       c, 1 - 2*c,       c },
        {       c,       c, 1 - 2*c },
    }};

    pl_matrix3x3_mul(&m, &hpe);

    // Apply chromatic adaptation to D65 if the input white point differs
    static const struct pl_cie_xy d65 = CIE_D65;
    apply_chromatic_adaptation(prim->white, d65, &m);

    const pl_matrix3x3 rgb2xyz = pl_get_rgb2xyz_matrix(prim);
    pl_matrix3x3_mul(&m, &rgb2xyz);
    return m;
}

pl_matrix3x3 pl_ipt_lms2rgb(const struct pl_raw_primaries *prim)
{
    pl_matrix3x3 m = pl_ipt_rgb2lms(prim);
    pl_matrix3x3_invert(&m);
    return m;
}

// As standardized in Ebner & Fairchild IPT (1998)
const pl_matrix3x3 pl_ipt_lms2ipt = {{
    { 0.4000,  0.4000,  0.2000 },
    { 4.4550, -4.8510,  0.3960 },
    { 0.8056,  0.3572, -1.1628 },
}};

// Numerically inverted from the matrix above
const pl_matrix3x3 pl_ipt_ipt2lms = {{
    { 1.0,  0.0975689,  0.205226 },
    { 1.0, -0.1138760,  0.133217 },
    { 1.0,  0.0326151, -0.676887 },
}};

const struct pl_cone_params pl_vision_normal        = {PL_CONE_NONE, 1.0};
const struct pl_cone_params pl_vision_protanomaly   = {PL_CONE_L,    0.5};
const struct pl_cone_params pl_vision_protanopia    = {PL_CONE_L,    0.0};
const struct pl_cone_params pl_vision_deuteranomaly = {PL_CONE_M,    0.5};
const struct pl_cone_params pl_vision_deuteranopia  = {PL_CONE_M,    0.0};
const struct pl_cone_params pl_vision_tritanomaly   = {PL_CONE_S,    0.5};
const struct pl_cone_params pl_vision_tritanopia    = {PL_CONE_S,    0.0};
const struct pl_cone_params pl_vision_monochromacy  = {PL_CONE_LM,   0.0};
const struct pl_cone_params pl_vision_achromatopsia = {PL_CONE_LMS,  0.0};

pl_matrix3x3 pl_get_cone_matrix(const struct pl_cone_params *params,
                                const struct pl_raw_primaries *prim)
{
    // LMS<-RGB := LMS<-XYZ * XYZ<-RGB
    pl_matrix3x3 rgb2lms = m_cat97;
    pl_matrix3x3 rgb2xyz = pl_get_rgb2xyz_matrix(prim);
    pl_matrix3x3_mul(&rgb2lms, &rgb2xyz);

    // LMS versions of the two opposing primaries, plus neutral
    float lms_r[3] = {1.0, 0.0, 0.0},
          lms_b[3] = {0.0, 0.0, 1.0},
          lms_w[3] = {1.0, 1.0, 1.0};

    pl_matrix3x3_apply(&rgb2lms, lms_r);
    pl_matrix3x3_apply(&rgb2lms, lms_b);
    pl_matrix3x3_apply(&rgb2lms, lms_w);

    float a, b, c = params->strength;
    pl_matrix3x3 distort;

    switch (params->cones) {
    case PL_CONE_NONE:
        return pl_matrix3x3_identity;

    case PL_CONE_L:
        // Solve to preserve neutral and blue
        a = (lms_b[0] - lms_b[2] * lms_w[0] / lms_w[2]) /
            (lms_b[1] - lms_b[2] * lms_w[1] / lms_w[2]);
        b = (lms_b[0] - lms_b[1] * lms_w[0] / lms_w[1]) /
            (lms_b[2] - lms_b[1] * lms_w[2] / lms_w[1]);
        assert(fabs(a * lms_w[1] + b * lms_w[2] - lms_w[0]) < 1e-6);

        distort = (pl_matrix3x3) {{
            {            c, (1.0 - c) * a, (1.0 - c) * b},
            {          0.0,           1.0,           0.0},
            {          0.0,           0.0,           1.0},
        }};
        break;

    case PL_CONE_M:
        // Solve to preserve neutral and blue
        a = (lms_b[1] - lms_b[2] * lms_w[1] / lms_w[2]) /
            (lms_b[0] - lms_b[2] * lms_w[0] / lms_w[2]);
        b = (lms_b[1] - lms_b[0] * lms_w[1] / lms_w[0]) /
            (lms_b[2] - lms_b[0] * lms_w[2] / lms_w[0]);
        assert(fabs(a * lms_w[0] + b * lms_w[2] - lms_w[1]) < 1e-6);

        distort = (pl_matrix3x3) {{
            {          1.0,           0.0,           0.0},
            {(1.0 - c) * a,             c, (1.0 - c) * b},
            {          0.0,           0.0,           1.0},
        }};
        break;

    case PL_CONE_S:
        // Solve to preserve neutral and red
        a = (lms_r[2] - lms_r[1] * lms_w[2] / lms_w[1]) /
            (lms_r[0] - lms_r[1] * lms_w[0] / lms_w[1]);
        b = (lms_r[2] - lms_r[0] * lms_w[2] / lms_w[0]) /
            (lms_r[1] - lms_r[0] * lms_w[1] / lms_w[0]);
        assert(fabs(a * lms_w[0] + b * lms_w[1] - lms_w[2]) < 1e-6);

        distort = (pl_matrix3x3) {{
            {          1.0,           0.0,           0.0},
            {          0.0,           1.0,           0.0},
            {(1.0 - c) * a, (1.0 - c) * b,             c},
        }};
        break;

    case PL_CONE_LM:
        // Solve to preserve neutral
        a = lms_w[0] / lms_w[2];
        b = lms_w[1] / lms_w[2];

        distort = (pl_matrix3x3) {{
            {            c,           0.0, (1.0 - c) * a},
            {          0.0,             c, (1.0 - c) * b},
            {          0.0,           0.0,           1.0},
        }};
        break;

    case PL_CONE_MS:
        // Solve to preserve neutral
        a = lms_w[1] / lms_w[0];
        b = lms_w[2] / lms_w[0];

        distort = (pl_matrix3x3) {{
            {          1.0,           0.0,           0.0},
            {(1.0 - c) * a,             c,           0.0},
            {(1.0 - c) * b,           0.0,             c},
        }};
        break;

    case PL_CONE_LS:
        // Solve to preserve neutral
        a = lms_w[0] / lms_w[1];
        b = lms_w[2] / lms_w[1];

        distort = (pl_matrix3x3) {{
            {            c, (1.0 - c) * a,           0.0},
            {          0.0,           1.0,           0.0},
            {          0.0, (1.0 - c) * b,             c},
        }};
        break;

    case PL_CONE_LMS: {
        // Rod cells only, which can be modelled somewhat as a combination of
        // L and M cones. Either way, this is pushing the limits of the our
        // color model, so this is only a rough approximation.
        const float w[3] = {0.3605, 0.6415, -0.002};
        assert(fabs(w[0] + w[1] + w[2] - 1.0) < 1e-6);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                distort.m[i][j] = (1.0 - c) * w[j] * lms_w[i] / lms_w[j];
                if (i == j)
                    distort.m[i][j] += c;
            }
        }
        break;
    }

    default:
        pl_unreachable();
    }

    // out := RGB<-LMS * distort * LMS<-RGB
    pl_matrix3x3 out = rgb2lms;
    pl_matrix3x3_invert(&out);
    pl_matrix3x3_mul(&out, &distort);
    pl_matrix3x3_mul(&out, &rgb2lms);

    return out;
}

pl_matrix3x3 pl_get_color_mapping_matrix(const struct pl_raw_primaries *src,
                                         const struct pl_raw_primaries *dst,
                                         enum pl_rendering_intent intent)
{
    // In saturation mapping, we don't care about accuracy and just want
    // primaries to map to primaries, making this an identity transformation.
    if (intent == PL_INTENT_SATURATION)
        return pl_matrix3x3_identity;

    // RGBd<-RGBs = RGBd<-XYZd * XYZd<-XYZs * XYZs<-RGBs
    // Equations from: http://www.brucelindbloom.com/index.html?Math.html
    // Note: Perceptual is treated like relative colorimetric. There's no
    // definition for perceptual other than "make it look good".

    // RGBd<-XYZd matrix
    pl_matrix3x3 xyz2rgb_d = pl_get_xyz2rgb_matrix(dst);

    // Chromatic adaptation, except in absolute colorimetric intent
    if (intent != PL_INTENT_ABSOLUTE_COLORIMETRIC)
        apply_chromatic_adaptation(src->white, dst->white, &xyz2rgb_d);

    // XYZs<-RGBs
    pl_matrix3x3 rgb2xyz_s = pl_get_rgb2xyz_matrix(src);
    pl_matrix3x3_mul(&xyz2rgb_d, &rgb2xyz_s);
    return xyz2rgb_d;
}

// Test the sign of 'p' relative to the line 'ab' (barycentric coordinates)
static float test_point_line(const struct pl_cie_xy p,
                             const struct pl_cie_xy a,
                             const struct pl_cie_xy b)
{
    return (p.x - b.x) * (a.y - b.y) - (a.x - b.x) * (p.y - b.y);
}

// Test if a point is entirely inside a gamut
static float test_point_gamut(struct pl_cie_xy point,
                              const struct pl_raw_primaries *prim)
{
    float d1 = test_point_line(point, prim->red, prim->green),
          d2 = test_point_line(point, prim->green, prim->blue),
          d3 = test_point_line(point, prim->blue, prim->red);

    bool has_neg = d1 < -1e-6f || d2 < -1e-6f || d3 < -1e-6f,
         has_pos = d1 >  1e-6f || d2 >  1e-6f || d3 >  1e-6f;

    return !(has_neg && has_pos);
}

bool pl_primaries_superset(const struct pl_raw_primaries *a,
                           const struct pl_raw_primaries *b)
{
    return test_point_gamut(b->red, a) &&
           test_point_gamut(b->green, a) &&
           test_point_gamut(b->blue, a);
}

bool pl_primaries_valid(const struct pl_raw_primaries *prim)
{
    // Test to see if the primaries form a valid triangle (nonzero area)
    float area = (prim->blue.x - prim->green.x) * (prim->red.y  - prim->green.y)
               - (prim->red.x  - prim->green.x) * (prim->blue.y - prim->green.y);

    return fabs(area) > 1e-6 && test_point_gamut(prim->white, prim);
}

static inline float xy_dist2(struct pl_cie_xy a, struct pl_cie_xy b)
{
    const float dx = a.x - b.x, dy = a.y - b.y;
    return dx * dx + dy * dy;
}

bool pl_primaries_compatible(const struct pl_raw_primaries *a,
                             const struct pl_raw_primaries *b)
{
    float RR = xy_dist2(a->red, b->red),    RG = xy_dist2(a->red, b->green),
          RB = xy_dist2(a->red, b->blue),   GG = xy_dist2(a->green, b->green),
          GB = xy_dist2(a->green, b->blue), BB = xy_dist2(a->blue, b->blue);
    return RR < RG && RR < RB && GG < RG && GG < GB && BB < RB && BB < GB;
}

// returns the intersection of the two lines defined by ab and cd
static struct pl_cie_xy intersection(struct pl_cie_xy a, struct pl_cie_xy b,
                                     struct pl_cie_xy c, struct pl_cie_xy d)
{
    float det = (a.x - b.x) * (c.y - d.y) - (a.y - b.y) * (c.x - d.x);
    float t = ((a.x - c.x) * (c.y - d.y) - (a.y - c.y) * (c.x - d.x)) / det;
    return (struct pl_cie_xy) {
        .x = t ? a.x + t * (b.x - a.x) : 0.0f,
        .y = t ? a.y + t * (b.y - a.y) : 0.0f,
    };
}

// x, y, z specified in clockwise order, with a, b, c being the enclosing gamut
static struct pl_cie_xy
clip_point(struct pl_cie_xy x, struct pl_cie_xy y, struct pl_cie_xy z,
           struct pl_cie_xy a, struct pl_cie_xy b, struct pl_cie_xy c)
{
    const float d1 = test_point_line(y, a, b);
    const float d2 = test_point_line(y, b, c);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        return y; // already inside triangle
    } else if (d1 > 0.0f && d2 > 0.0f) {
        return b; // target vertex fully enclosed
    } else if (d1 > 0.0f) {
        return intersection(a, b, y, z);
    } else {
        return intersection(x, y, b, c);
    }
}

struct pl_raw_primaries pl_primaries_clip(const struct pl_raw_primaries *src,
                                          const struct pl_raw_primaries *dst)
{
    return (struct pl_raw_primaries) {
        .red   = clip_point(src->green, src->red, src->blue,
                            dst->green, dst->red, dst->blue),
        .green = clip_point(src->blue, src->green, src->red,
                            dst->blue, dst->green, dst->red),
        .blue  = clip_point(src->red, src->blue, src->green,
                            dst->red, dst->blue, dst->green),
        .white = src->white,
    };
}

/* Fill in the Y, U, V vectors of a yuv-to-rgb conversion matrix
 * based on the given luma weights of the R, G and B components (lr, lg, lb).
 * lr+lg+lb is assumed to equal 1.
 * This function is meant for colorspaces satisfying the following
 * conditions (which are true for common YUV colorspaces):
 * - The mapping from input [Y, U, V] to output [R, G, B] is linear.
 * - Y is the vector [1, 1, 1].  (meaning input Y component maps to 1R+1G+1B)
 * - U maps to a value with zero R and positive B ([0, x, y], y > 0;
 *   i.e. blue and green only).
 * - V maps to a value with zero B and positive R ([x, y, 0], x > 0;
 *   i.e. red and green only).
 * - U and V are orthogonal to the luma vector [lr, lg, lb].
 * - The magnitudes of the vectors U and V are the minimal ones for which
 *   the image of the set Y=[0...1],U=[-0.5...0.5],V=[-0.5...0.5] under the
 *   conversion function will cover the set R=[0...1],G=[0...1],B=[0...1]
 *   (the resulting matrix can be converted for other input/output ranges
 *   outside this function).
 * Under these conditions the given parameters lr, lg, lb uniquely
 * determine the mapping of Y, U, V to R, G, B.
 */
static pl_matrix3x3 luma_coeffs(float lr, float lg, float lb)
{
    pl_assert(fabs(lr+lg+lb - 1) < 1e-6);
    return (pl_matrix3x3) {{
        {1, 0,                    2 * (1-lr)          },
        {1, -2 * (1-lb) * lb/lg, -2 * (1-lr) * lr/lg  },
        {1,  2 * (1-lb),          0                   },
    }};
}

// Applies hue and saturation controls to a YCbCr->RGB matrix
static inline void apply_hue_sat(pl_matrix3x3 *m,
                                 const struct pl_color_adjustment *params)
{
    // Hue is equivalent to rotating input [U, V] subvector around the origin.
    // Saturation scales [U, V].
    float huecos = params->saturation * cos(params->hue);
    float huesin = params->saturation * sin(params->hue);
    for (int i = 0; i < 3; i++) {
        float u = m->m[i][1], v = m->m[i][2];
        m->m[i][1] = huecos * u - huesin * v;
        m->m[i][2] = huesin * u + huecos * v;
    }
}

pl_transform3x3 pl_color_repr_decode(struct pl_color_repr *repr,
                                     const struct pl_color_adjustment *params)
{
    params = PL_DEF(params, &pl_color_adjustment_neutral);

    pl_matrix3x3 m;
    switch (repr->sys) {
    case PL_COLOR_SYSTEM_BT_709:     m = luma_coeffs(0.2126, 0.7152, 0.0722); break;
    case PL_COLOR_SYSTEM_BT_601:     m = luma_coeffs(0.2990, 0.5870, 0.1140); break;
    case PL_COLOR_SYSTEM_SMPTE_240M: m = luma_coeffs(0.2122, 0.7013, 0.0865); break;
    case PL_COLOR_SYSTEM_BT_2020_NC: m = luma_coeffs(0.2627, 0.6780, 0.0593); break;
    case PL_COLOR_SYSTEM_BT_2020_C:
        // Note: This outputs into the [-0.5,0.5] range for chroma information.
        m = (pl_matrix3x3) {{
            {0, 0, 1},
            {1, 0, 0},
            {0, 1, 0},
        }};
        break;
    case PL_COLOR_SYSTEM_BT_2100_PQ: {
        // Reversed from the matrix in the spec, hard-coded for efficiency
        // and precision reasons. Exact values truncated from ITU-T H-series
        // Supplement 18.
        static const float lm_t = 0.008609, lm_p = 0.111029625;
        m = (pl_matrix3x3) {{
            {1.0,  lm_t,  lm_p},
            {1.0, -lm_t, -lm_p},
            {1.0, 0.560031, -0.320627},
        }};
        break;
    }
    case PL_COLOR_SYSTEM_BT_2100_HLG: {
        // Similar to BT.2100 PQ, exact values truncated from WolframAlpha
        static const float lm_t = 0.01571858011, lm_p = 0.2095810681;
        m = (pl_matrix3x3) {{
            {1.0,  lm_t,  lm_p},
            {1.0, -lm_t, -lm_p},
            {1.0, 1.02127108, -0.605274491},
        }};
        break;
    }
    case PL_COLOR_SYSTEM_DOLBYVISION:
        m = repr->dovi->nonlinear;
        break;
    case PL_COLOR_SYSTEM_YCGCO:
        m = (pl_matrix3x3) {{
            {1,  -1,  1},
            {1,   1,  0},
            {1,  -1, -1},
        }};
        break;
    case PL_COLOR_SYSTEM_UNKNOWN: // fall through
    case PL_COLOR_SYSTEM_RGB:
        m = pl_matrix3x3_identity;
        break;
    case PL_COLOR_SYSTEM_XYZ: {
        // For lack of anything saner to do, just assume the caller wants
        // DCI-P3 primaries, which is a reasonable assumption.
        const struct pl_raw_primaries *dst = pl_raw_primaries_get(PL_COLOR_PRIM_DCI_P3);
        m = pl_get_xyz2rgb_matrix(dst);
        // DCDM X'Y'Z' is expected to have equal energy white point (EG 432-1 Annex H)
        apply_chromatic_adaptation((struct pl_cie_xy)CIE_E, dst->white, &m);
        break;
    }
    case PL_COLOR_SYSTEM_COUNT:
        pl_unreachable();
    }

    // Apply hue and saturation in the correct way depending on the colorspace.
    if (pl_color_system_is_ycbcr_like(repr->sys)) {
        apply_hue_sat(&m, params);
    } else if (params->saturation != 1.0 || params->hue != 0.0) {
        // Arbitrarily simulate hue shifts using the BT.709 YCbCr model
        pl_matrix3x3 yuv2rgb = luma_coeffs(0.2126, 0.7152, 0.0722);
        pl_matrix3x3 rgb2yuv = yuv2rgb;
        pl_matrix3x3_invert(&rgb2yuv);
        apply_hue_sat(&yuv2rgb, params);
        // M := RGB<-YUV * YUV<-RGB * M
        pl_matrix3x3_rmul(&rgb2yuv, &m);
        pl_matrix3x3_rmul(&yuv2rgb, &m);
    }

    // Apply color temperature adaptation, relative to BT.709 primaries
    if (params->temperature) {
        struct pl_cie_xy src = pl_white_from_temp(6500);
        struct pl_cie_xy dst = pl_white_from_temp(6500 + 3500 * params->temperature);
        pl_matrix3x3 adapt = pl_get_adaptation_matrix(src, dst);
        pl_matrix3x3_rmul(&adapt, &m);
    }

    pl_transform3x3 out = { .mat = m };
    int bit_depth = PL_DEF(repr->bits.sample_depth,
                    PL_DEF(repr->bits.color_depth, 8));

    double ymax, ymin, cmax, cmid;
    double scale = (1LL << bit_depth) / ((1LL << bit_depth) - 1.0);

    switch (pl_color_levels_guess(repr)) {
    case PL_COLOR_LEVELS_LIMITED: {
        ymax = 235 / 256. * scale;
        ymin =  16 / 256. * scale;
        cmax = 240 / 256. * scale;
        cmid = 128 / 256. * scale;
        break;
    }
    case PL_COLOR_LEVELS_FULL:
        // Note: For full-range YUV, there are multiple, subtly inconsistent
        // standards. So just pick the sanest implementation, which is to
        // assume MAX_INT == 1.0.
        ymax = 1.0;
        ymin = 0.0;
        cmax = 1.0;
        cmid = 128 / 256. * scale; // *not* exactly 0.5
        break;
    default:
        pl_unreachable();
    }

    double ymul = 1.0 / (ymax - ymin);
    double cmul = 0.5 / (cmax - cmid);

    double mul[3]   = { ymul, ymul, ymul };
    double black[3] = { ymin, ymin, ymin };

#ifdef PL_HAVE_DOVI
    if (repr->sys == PL_COLOR_SYSTEM_DOLBYVISION) {
        // The RPU matrix already includes levels normalization, but in this
        // case we also have to respect the signalled color offsets
        for (int i = 0; i < 3; i++) {
            mul[i] = 1.0;
            black[i] = repr->dovi->nonlinear_offset[i] * scale;
        }
    } else
#endif
    if (pl_color_system_is_ycbcr_like(repr->sys)) {
        mul[1]   = mul[2]   = cmul;
        black[1] = black[2] = cmid;
    }

    // Contrast scales the output value range (gain)
    // Brightness scales the constant output bias (black lift/boost)
    for (int i = 0; i < 3; i++) {
        mul[i]   *= params->contrast;
        out.c[i] += params->brightness;
    }

    // Multiply in the texture multiplier and adjust `c` so that black[j] keeps
    // on mapping to RGB=0 (black to black)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            out.mat.m[i][j] *= mul[j];
            out.c[i] -= out.mat.m[i][j] * black[j];
        }
    }

    // Finally, multiply in the scaling factor required to get the color up to
    // the correct representation.
    pl_matrix3x3_scale(&out.mat, pl_color_repr_normalize(repr));

    // Update the metadata to reflect the change.
    repr->sys    = PL_COLOR_SYSTEM_RGB;
    repr->levels = PL_COLOR_LEVELS_FULL;

    return out;
}

bool pl_icc_profile_equal(const struct pl_icc_profile *p1,
                          const struct pl_icc_profile *p2)
{
    if (p1->len != p2->len)
        return false;

    // Ignore signatures on length-0 profiles, as a special case
    return !p1->len || p1->signature == p2->signature;
}

void pl_icc_profile_compute_signature(struct pl_icc_profile *profile)
{
    if (!profile->len)
        profile->signature = 0;

    // In theory, we could get this value from the profile header itself if
    // lcms is available, but I'm not sure if it's even worth the trouble. Just
    // hard-code this to a pl_mem_hash(), which is decently fast anyway.
    profile->signature = pl_mem_hash(profile->data, profile->len);
}
