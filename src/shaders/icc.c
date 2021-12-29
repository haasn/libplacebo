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

#include "shaders.h"

static cmsHPROFILE get_profile(pl_log log, cmsContext cms,
                               struct pl_icc_color_space iccsp,
                               struct pl_color_space *csp)
{
    *csp = iccsp.color;

    // The input profile for the transformation is dependent on the video
    // primaries, transfer characteristics, and brightness levels
    const float lb = csp->hdr.min_luma / PL_COLOR_SDR_WHITE;
    const float lw = csp->hdr.max_luma / PL_COLOR_SDR_WHITE;

    const struct pl_raw_primaries *prim = pl_raw_primaries_get(csp->primaries);
    cmsCIExyY wp_xyY = { prim->white.x, prim->white.y, 1.0 };
    cmsCIExyYTRIPLE prim_xyY = {
        .Red   = { prim->red.x,   prim->red.y,   1.0 },
        .Green = { prim->green.x, prim->green.y, 1.0 },
        .Blue  = { prim->blue.x,  prim->blue.y,  1.0 },
    };

    if (iccsp.profile.data) {
        pl_info(log, "Opening ICC profile..");
        cmsHPROFILE prof = cmsOpenProfileFromMemTHR(cms, iccsp.profile.data,
                                                   iccsp.profile.len);
        if (!prof) {
            pl_err(log, "Failed opening ICC profile, falling back to color struct");
            goto fallback;
        }

        // Update contrast information with detected black point
        const int intent = PL_INTENT_RELATIVE_COLORIMETRIC;
        cmsCIEXYZ bp_XYZ;
        if (!cmsDetectBlackPoint(&bp_XYZ, prof, intent, 0))
            return prof;

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
            return prof;

        double src_black[3] = {0};
        cmsDoTransform(xyz2src, &bp_XYZ, src_black, 1);
        cmsDeleteTransform(xyz2src);

        // Convert to (relative) output luminance using RGB->XYZ matrix
        struct pl_matrix3x3 rgb2xyz = pl_get_rgb2xyz_matrix(&csp->hdr.prim);
        float min_luma = 0.0f;
        for (int i = 0; i < 3; i++)
            min_luma += rgb2xyz.m[1][i] * src_black[i];

        csp->hdr.min_luma = min_luma * csp->hdr.max_luma;
        return prof;
    }

    // fall through
fallback:;

    cmsToneCurve *tonecurve = NULL;
    switch (csp->transfer) {
    case PL_COLOR_TRC_LINEAR:
        tonecurve = cmsBuildGamma(cms, 1.0);
        break;
    case PL_COLOR_TRC_GAMMA18:
        tonecurve = cmsBuildParametricToneCurve(cms, 6,
                (double[4]) { 1.8f, powf(lw - lb, 1/1.8f), 0, lb });
        break;
    case PL_COLOR_TRC_GAMMA20:
        tonecurve = cmsBuildParametricToneCurve(cms, 6,
                (double[4]) { 2.0f, powf(lw - lb, 1/2.0f), 0, lb });
        break;
    case PL_COLOR_TRC_GAMMA24:
        tonecurve = cmsBuildParametricToneCurve(cms, 6,
                (double[4]) { 2.4f, powf(lw - lb, 1/2.4f), 0, lb });
        break;
    case PL_COLOR_TRC_GAMMA26:
        tonecurve = cmsBuildParametricToneCurve(cms, 6,
                (double[4]) { 2.6f, powf(lw - lb, 1/2.6f), 0, lb });
        break;
    case PL_COLOR_TRC_GAMMA28:
        tonecurve = cmsBuildParametricToneCurve(cms, 6,
                (double[4]) { 2.6f, powf(lw - lb, 1/2.8f), 0, lb });
        break;
    case PL_COLOR_TRC_UNKNOWN:
    case PL_COLOR_TRC_PQ:
    case PL_COLOR_TRC_HLG:
    case PL_COLOR_TRC_S_LOG1:
    case PL_COLOR_TRC_S_LOG2:
    case PL_COLOR_TRC_V_LOG:
    case PL_COLOR_TRC_GAMMA22:
        // Catch-all bucket for unimplemented/unknown TRCs
        csp->transfer = PL_COLOR_TRC_GAMMA22;
        tonecurve = cmsBuildParametricToneCurve(cms, 6,
                (double[4]) { 2.2f, powf(lw - lb, 1/2.2f), 0, lb });
        break;
    case PL_COLOR_TRC_SRGB: {
        // Curve definition:
        //   (aX + b)^y + e  | X >= d
        //   cX + f          | X < d
        const float y = 2.4f;
        const float s = powf(lw - lb, 1/y);
        const float a = s / 1.055f;
        const float b = a * 0.055f;
        const float c = (lw - lb) / 12.92f;
        const float d = 0.04045f;
        tonecurve = cmsBuildParametricToneCurve(cms, 5,
                (double[7]) { y, a, b, c, d, lb, lb });
        break;
    }
    case PL_COLOR_TRC_PRO_PHOTO: {
        // Curve definition:
        //   (aX + b)^y + e  | X >= d
        //   cX + f          | X < d
        const float y = 1.8f;
        const float s = powf(lw - lb, 1/y);
        const float c = (lw - lb) / 16;
        const float d = 0.03125f;
        tonecurve = cmsBuildParametricToneCurve(cms, 5,
                (double[7]){ y, s, 0, c, d, lb, lb });
        break;
    }
    case PL_COLOR_TRC_BT_1886: {
        // Curve definition:
        //   (aX + b)^y + c
        const float y = 2.4f;
        const float lby = powf(lb, 1/y);
        const float lwy = powf(lw, 1/y);
        tonecurve = cmsBuildParametricToneCurve(cms, 6,
                (double[4]){ y, lwy - lby, lby, 0 });
        break;
    }
    case PL_COLOR_TRC_COUNT:
        pl_unreachable();
    }

    if (!tonecurve)
        return NULL;

    cmsToneCurve *curves[3] = { tonecurve, tonecurve, tonecurve };
    cmsHPROFILE ret = cmsCreateRGBProfileTHR(cms, &wp_xyY, &prim_xyY, curves);
    cmsFreeToneCurve(tonecurve);
    return ret;
}

static void error_callback(cmsContext cms, cmsUInt32Number code,
                           const char *msg)
{
    pl_log log = cmsGetContextUserData(cms);
    pl_err(log, "lcms2: [%d] %s", (int) code, msg);
}

struct sh_icc_obj {
    pl_log log;
    struct pl_icc_params params;
    struct pl_icc_color_space src, dst;
    struct pl_icc_result result;
    pl_shader_obj lut_obj;
    bool updated; // to detect misuse of the API
    bool ok;
    ident_t lut;
};

static void fill_icc(void *datap, const struct sh_lut_params *params)
{
    struct sh_icc_obj *obj = params->priv;
    pl_assert(params->comps == 4);

    struct pl_icc_color_space src = obj->src;
    cmsHPROFILE srcp = NULL, dstp = NULL;
    cmsHTRANSFORM trafo = NULL;
    uint16_t *tmp = NULL;
    obj->ok = false;

    cmsContext cms = cmsCreateContext(NULL, (void *) obj->log);
    if (!cms) {
        PL_ERR(obj, "Failed creating LittleCMS context!");
        goto error;
    }

    cmsSetLogErrorHandlerTHR(cms, error_callback);
    clock_t start = clock();
    dstp = get_profile(obj->log, cms, obj->dst, &obj->result.dst_color);
    if (obj->params.use_display_contrast) {
        src.color.hdr.max_luma = obj->result.dst_color.hdr.max_luma;
        src.color.hdr.min_luma = obj->result.dst_color.hdr.min_luma;
    }
    srcp = get_profile(obj->log, cms, src, &obj->result.src_color);
    clock_t after_profiles = clock();
    pl_log_cpu_time(obj->log, start, after_profiles, "opening ICC profiles");
    if (!srcp || !dstp)
        goto error;

    uint32_t flags = cmsFLAGS_HIGHRESPRECALC | cmsFLAGS_BLACKPOINTCOMPENSATION |
                     cmsFLAGS_NOCACHE;

    trafo = cmsCreateTransformTHR(cms, srcp, TYPE_RGB_16, dstp, TYPE_RGB_16,
                                  obj->params.intent, flags);
    clock_t after_transform = clock();
    pl_log_cpu_time(obj->log, after_profiles, after_transform, "creating ICC transform");
    if (!trafo) {
        PL_ERR(obj, "Failed creating CMS transform!");
        goto error;
    }

    int s_r = params->width, s_g = params->height, s_b = params->depth;
    pl_assert(s_r > 1 && s_g > 1 && s_b > 1);
    tmp = pl_alloc(NULL, 2 * s_r * 3 * sizeof(tmp[0]));

    uint16_t *out = tmp + s_r * 3;
    for (int b = 0; b < s_b; b++) {
        for (int g = 0; g < s_g; g++) {
            // Transform a single line of the output buffer
            for (int r = 0; r < s_r; r++) {
                tmp[r * 3 + 0] = r * 65535 / (s_r - 1);
                tmp[r * 3 + 1] = g * 65535 / (s_g - 1);
                tmp[r * 3 + 2] = b * 65535 / (s_b - 1);
            }
            cmsDoTransform(trafo, tmp, out, s_r);

            // Write this line into the right output position
            size_t offset = (b * s_g + g) * s_r * 4;
            float *data = ((float *) datap) + offset;
            for (int r = 0; r < s_r; r++) {
                data[r * 4 + 0] = out[r * 3 + 0] / 65535.0;
                data[r * 4 + 1] = out[r * 3 + 1] / 65535.0;
                data[r * 4 + 2] = out[r * 3 + 2] / 65535.0;
                data[r * 4 + 3] = 1.0;
            }
        }
    }

    pl_log_cpu_time(obj->log, after_transform, clock(), "generating ICC 3DLUT");
    obj->ok = true;
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

    pl_free_ptr(&tmp);
}

static void sh_icc_uninit(pl_gpu gpu, void *ptr)
{
    struct sh_icc_obj *obj = ptr;
    pl_shader_obj_destroy(&obj->lut_obj);
    *obj = (struct sh_icc_obj) {0};
}

static bool icc_csp_eq(const struct pl_icc_color_space *a,
                       const struct pl_icc_color_space *b)
{
    return pl_icc_profile_equal(&a->profile, &b->profile) &&
           pl_color_space_equal(&a->color, &b->color);
}

bool pl_icc_update(pl_shader sh,
                   const struct pl_icc_color_space *srcp,
                   const struct pl_icc_color_space *dstp,
                   pl_shader_obj *icc,
                   struct pl_icc_result *out,
                   const struct pl_icc_params *params)
{
    params = PL_DEF(params, &pl_icc_default_params);
    size_t s_r = PL_DEF(params->size_r, 64),
           s_g = PL_DEF(params->size_g, 64),
           s_b = PL_DEF(params->size_b, 64);

    struct sh_icc_obj *obj;
    obj = SH_OBJ(sh, icc, PL_SHADER_OBJ_ICC,
                 struct sh_icc_obj, sh_icc_uninit);
    if (!obj)
        return false;

    struct pl_icc_color_space src = *srcp, dst = *dstp;
    pl_color_space_infer(&src.color);
    pl_color_space_infer_ref(&dst.color, &src.color);

    bool changed = !icc_csp_eq(&obj->src, &src) ||
                   !icc_csp_eq(&obj->dst, &dst) ||
                   memcmp(&obj->params, params, sizeof(*params));

    // Update the object, since we need this information from `fill_icc`
    obj->log = sh->log;
    obj->params = *params;
    obj->src = src;
    obj->dst = dst;
    obj->lut = sh_lut(sh, sh_lut_params(
        .object = &obj->lut_obj,
        .type = PL_VAR_FLOAT,
        .width = s_r,
        .height = s_g,
        .depth = s_b,
        .comps = 4,
        .linear = true,
        .update = changed,
        .fill = fill_icc,
        .priv = obj,
    ));
    if (!obj->lut || !obj->ok)
        return false;

    obj->updated = true;
    *out = obj->result;
    return true;
}

void pl_icc_apply(pl_shader sh, pl_shader_obj *icc)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    struct sh_icc_obj *obj;
    obj = SH_OBJ(sh, icc, PL_SHADER_OBJ_ICC,
                 struct sh_icc_obj, sh_icc_uninit);
    if (!obj || !obj->lut || !obj->updated || !obj->ok) {
        SH_FAIL(sh, "pl_icc_apply called without prior pl_icc_update?");
        return;
    }

    sh_describe(sh, "ICC 3DLUT");
    GLSL("// pl_icc_apply \n"
         "color.rgb = %s(color.rgb).rgb; \n",
         obj->lut);

    obj->updated = false;
}

const struct pl_icc_params pl_icc_default_params = { PL_ICC_DEFAULTS };
