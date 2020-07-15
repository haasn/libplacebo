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

#include <lcms2.h>
#include <math.h>

#include "context.h"
#include "lcms.h"

static cmsHPROFILE get_profile(struct pl_context *ctx, cmsContext cms,
                               struct pl_3dlut_profile prof, cmsHPROFILE dstp,
                               struct pl_color_space *csp)
{
    *csp = prof.color;

    if (prof.profile.data) {
        pl_info(ctx, "Opening ICC profile..");
        cmsHPROFILE ret = cmsOpenProfileFromMemTHR(cms, prof.profile.data,
                                                   prof.profile.len);
        if (ret)
            return ret;
        pl_err(ctx, "Failed opening ICC profile, falling back to color struct");
    }

    // The input profile for the transformation is dependent on the video
    // primaries and transfer characteristics
    const struct pl_raw_primaries *prim = pl_raw_primaries_get(csp->primaries);
    csp->light = PL_COLOR_LIGHT_DISPLAY;

    cmsCIExyY wp_xyY = {prim->white.x, prim->white.y, 1.0};
    cmsCIExyYTRIPLE prim_xyY = {
        .Red   = {prim->red.x,   prim->red.y,   1.0},
        .Green = {prim->green.x, prim->green.y, 1.0},
        .Blue  = {prim->blue.x,  prim->blue.y,  1.0},
    };

    cmsToneCurve *tonecurve[3] = {0};
    switch (csp->transfer) {
    case PL_COLOR_TRC_LINEAR:  tonecurve[0] = cmsBuildGamma(cms, 1.0); break;
    case PL_COLOR_TRC_GAMMA18: tonecurve[0] = cmsBuildGamma(cms, 1.8); break;
    case PL_COLOR_TRC_GAMMA28: tonecurve[0] = cmsBuildGamma(cms, 2.8); break;

    // Catch-all bucket for unimplemented TRCs
    case PL_COLOR_TRC_UNKNOWN:
    case PL_COLOR_TRC_PQ:
    case PL_COLOR_TRC_HLG:
    case PL_COLOR_TRC_S_LOG1:
    case PL_COLOR_TRC_S_LOG2:
    case PL_COLOR_TRC_V_LOG:
    case PL_COLOR_TRC_GAMMA22:
        tonecurve[0] = cmsBuildGamma(cms, 2.2);
        csp->transfer = PL_COLOR_TRC_GAMMA22;
        break;

    case PL_COLOR_TRC_SRGB:
        // Values copied from Little-CMS
        tonecurve[0] = cmsBuildParametricToneCurve(cms, 4,
                (double[5]) {2.40, 1/1.055, 0.055/1.055, 1/12.92, 0.04045});
        break;

    case PL_COLOR_TRC_PRO_PHOTO:
        tonecurve[0] = cmsBuildParametricToneCurve(cms, 4,
                (double[5]){1.8, 1.0, 0.0, 1/16.0, 0.03125});
        break;

    case PL_COLOR_TRC_BT_1886: {
        if (!dstp) {
            pl_info(ctx, "No destination profile data available for accurate "
                    "BT.1886 emulation, falling back to gamma 2.2");
            tonecurve[0] = cmsBuildGamma(cms, 2.2);
            csp->transfer = PL_COLOR_TRC_GAMMA22;
            break;
        }

        // To build an appropriate BT.1886 transformation we need access to
        // the display's black point, so we LittleCMS' detection function.
        // Relative colorimetric is used since we want to approximate the
        // BT.1886 to the target device's actual black point even in e.g.
        // perceptual mode
        const int intent = PL_INTENT_RELATIVE_COLORIMETRIC;
        cmsCIEXYZ bp_XYZ;
        if (!cmsDetectBlackPoint(&bp_XYZ, dstp, intent, 0))
            return false;

        // Map this XYZ value back into the (linear) source space
        cmsToneCurve *linear = cmsBuildGamma(cms, 1.0);
        cmsHPROFILE rev_profile = cmsCreateRGBProfileTHR(cms, &wp_xyY, &prim_xyY,
                (cmsToneCurve*[3]){linear, linear, linear});
        cmsHPROFILE xyz_profile = cmsCreateXYZProfile();
        cmsHTRANSFORM xyz2src = cmsCreateTransformTHR(cms,
                xyz_profile, TYPE_XYZ_DBL, rev_profile, TYPE_RGB_DBL,
                intent, 0);
        cmsFreeToneCurve(linear);
        cmsCloseProfile(rev_profile);
        cmsCloseProfile(xyz_profile);
        if (!xyz2src)
            return false;

        double src_black[3] = {0};
        cmsDoTransform(xyz2src, &bp_XYZ, src_black, 1);
        cmsDeleteTransform(xyz2src);

        // Built-in contrast failsafe
        double contrast = 3.0 / (src_black[0] + src_black[1] + src_black[2]);
        if (contrast > 100000) {
            pl_warn(ctx, "ICC profile detected contrast very high (>100000),"
                    " falling back to contrast 1000 for sanity");
            src_black[0] = src_black[1] = src_black[2] = 1.0 / 1000;
        }

        // Build the parametric BT.1886 transfer curve, one per channel
        for (int i = 0; i < 3; i++) {
            const double gamma = 2.40;
            double binv = pow(src_black[i], 1.0/gamma);
            tonecurve[i] = cmsBuildParametricToneCurve(cms, 6,
                    (double[4]){gamma, 1.0 - binv, binv, 0.0});
        }
        break;
    }

    case PL_COLOR_TRC_COUNT:
    default: abort();
    }

    if (!tonecurve[0])
        return NULL;

    tonecurve[1] = PL_DEF(tonecurve[1], tonecurve[0]);
    tonecurve[2] = PL_DEF(tonecurve[2], tonecurve[0]);

    cmsHPROFILE ret = cmsCreateRGBProfileTHR(cms, &wp_xyY, &prim_xyY, tonecurve);

    cmsFreeToneCurve(tonecurve[0]);
    if (tonecurve[1] != tonecurve[0])
        cmsFreeToneCurve(tonecurve[1]);
    if (tonecurve[2] != tonecurve[0])
        cmsFreeToneCurve(tonecurve[2]);

    return ret;
}

static void error_callback(cmsContext cms, cmsUInt32Number code,
                           const char *msg)
{
    struct pl_context *ctx = cmsGetContextUserData(cms);
    pl_err(ctx, "lcms2: [%d] %s", (int) code, msg);
}

bool pl_lcms_compute_lut(struct pl_context *ctx, enum pl_rendering_intent intent,
                         struct pl_3dlut_profile src, struct pl_3dlut_profile dst,
                         float *out_data, int s_r, int s_g, int s_b,
                         struct pl_3dlut_result *out)
{
    bool ret = false;
    cmsHPROFILE srcp = NULL, dstp = NULL;
    cmsHTRANSFORM trafo = NULL;
    uint16_t *tmp = NULL;

    cmsContext cms = cmsCreateContext(NULL, ctx);
    if (!cms)
        goto error;

    cmsSetLogErrorHandlerTHR(cms, error_callback);
    dstp = get_profile(ctx, cms, dst, NULL, &out->dst_color);
    srcp = get_profile(ctx, cms, src, dstp, &out->src_color);
    if (!srcp || !dstp)
        goto error;

    uint32_t flags = cmsFLAGS_HIGHRESPRECALC | cmsFLAGS_BLACKPOINTCOMPENSATION |
                     cmsFLAGS_NOCACHE;

    trafo = cmsCreateTransformTHR(cms, srcp, TYPE_RGB_16, dstp, TYPE_RGBA_FLT,
                                  intent, flags);
    if (!trafo)
        goto error;

    pl_assert(s_r > 1 && s_g > 1 && s_b > 1);
    tmp = talloc_array(NULL, uint16_t, s_r * 3);

    for (int b = 0; b < s_b; b++) {
        for (int g = 0; g < s_g; g++) {
            // Fill in a single line of the temporary buffer
            for (int r = 0; r < s_r; r++) {
                tmp[r * 3 + 0] = r * 65535 / (s_r - 1);
                tmp[r * 3 + 1] = g * 65535 / (s_g - 1);
                tmp[r * 3 + 2] = b * 65535 / (s_b - 1);
            }

            // Transform this line into the right output position
            size_t offset = (b * s_g + g) * s_r * 4;
            cmsDoTransform(trafo, tmp, out_data + offset, s_r);
        }
    }

    ret = true;
    // fall through

error:
    if (trafo)
        cmsDeleteTransform(trafo);
    if (srcp)
        cmsCloseProfile(srcp);
    if (dstp)
        cmsCloseProfile(dstp);
    if (cms)
        cmsDeleteContext(cms);

    TA_FREEP(&tmp);
    return ret;
}
