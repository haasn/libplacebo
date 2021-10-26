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
                               struct pl_icc_color_space iccsp, cmsHPROFILE dstp,
                               struct pl_color_space *csp)
{
    *csp = iccsp.color;

    if (iccsp.profile.data) {
        pl_info(log, "Opening ICC profile..");
        cmsHPROFILE ret = cmsOpenProfileFromMemTHR(cms, iccsp.profile.data,
                                                   iccsp.profile.len);
        if (ret)
            return ret;
        pl_err(log, "Failed opening ICC profile, falling back to color struct");
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
    case PL_COLOR_TRC_GAMMA20: tonecurve[0] = cmsBuildGamma(cms, 2.0); break;
    case PL_COLOR_TRC_GAMMA24: tonecurve[0] = cmsBuildGamma(cms, 2.4); break;
    case PL_COLOR_TRC_GAMMA26: tonecurve[0] = cmsBuildGamma(cms, 2.6); break;
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
            pl_info(log, "No destination profile data available for accurate "
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
        pl_unreachable();
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
    pl_log log = cmsGetContextUserData(cms);
    pl_err(log, "lcms2: [%d] %s", (int) code, msg);
}

struct sh_icc_obj {
    pl_log log;
    enum pl_rendering_intent intent;
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
    dstp = get_profile(obj->log, cms, obj->dst, NULL, &obj->result.dst_color);
    srcp = get_profile(obj->log, cms, obj->src, dstp, &obj->result.src_color);
    clock_t after_profiles = clock();
    pl_log_cpu_time(obj->log, start, after_profiles, "opening ICC profiles");
    if (!srcp || !dstp)
        goto error;

    uint32_t flags = cmsFLAGS_HIGHRESPRECALC | cmsFLAGS_BLACKPOINTCOMPENSATION |
                     cmsFLAGS_NOCACHE;

    trafo = cmsCreateTransformTHR(cms, srcp, TYPE_RGB_16, dstp, TYPE_RGB_16,
                                  obj->intent, flags);
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
                   const struct pl_icc_color_space *src,
                   const struct pl_icc_color_space *dst,
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

    bool changed = !icc_csp_eq(&obj->src, src) ||
                   !icc_csp_eq(&obj->dst, dst) ||
                   obj->intent != params->intent;

    // Update the object, since we need this information from `fill_icc`
    obj->log = sh->log;
    obj->intent = params->intent;
    obj->src = *src;
    obj->dst = *dst;
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
