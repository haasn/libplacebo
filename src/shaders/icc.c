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
#include "shaders.h"

const struct pl_icc_params pl_icc_default_params = { PL_ICC_DEFAULTS };

#ifdef PL_HAVE_LCMS

#include <lcms2.h>
#include <lcms2_plugin.h>

struct icc_priv {
    cmsContext cms;
    cmsHPROFILE profile;
    cmsHPROFILE approx; // approximation profile
    cmsCIEXYZ black;
};

static void error_callback(cmsContext cms, cmsUInt32Number code,
                           const char *msg)
{
    pl_log log = cmsGetContextUserData(cms);
    pl_err(log, "lcms2: [%d] %s", (int) code, msg);
}

void pl_icc_close(pl_icc_object *picc)
{
    pl_icc_object icc = *picc;
    if (!icc)
        return;

    struct icc_priv *p = PL_PRIV(icc);
    cmsCloseProfile(p->approx);
    cmsCloseProfile(p->profile);
    cmsDeleteContext(p->cms);
    pl_free_ptr((void **) picc);
}

static bool read_primaries(pl_icc_object icc, struct pl_raw_primaries *prim)
{
    struct icc_priv *p = PL_PRIV(icc);
    cmsHTRANSFORM tf;
    cmsHPROFILE xyz = cmsCreateXYZProfileTHR(p->cms);
    if (!xyz)
        return false;

    // We need to use an unadapted observer to get the raw values
    cmsFloat64Number prev_adapt = cmsSetAdaptationStateTHR(p->cms, 0.0);
    tf = cmsCreateTransformTHR(p->cms, p->profile, TYPE_RGB_8, xyz, TYPE_XYZ_DBL,
                               INTENT_ABSOLUTE_COLORIMETRIC,
                               /* Note: These flags mostly don't do anything
                                * anyway, but specify them regardless */
                               cmsFLAGS_NOCACHE |
                               cmsFLAGS_NOOPTIMIZE |
                               cmsFLAGS_LOWRESPRECALC |
                               cmsFLAGS_GRIDPOINTS(2));
    cmsSetAdaptationStateTHR(p->cms, prev_adapt);
    cmsCloseProfile(xyz);
    if (!tf)
        return false;

    static const uint8_t testprimaries[4][3] = {
        { 0xFF,    0,    0 }, /* red */
        {    0, 0xFF,    0 }, /* green */
        {    0,    0, 0xFF }, /* blue */
        { 0xFF, 0xFF, 0xFF }, /* white */
    };

    cmsCIEXYZ dst[4] = {0};
    cmsDoTransform(tf, testprimaries, dst, 4);
    cmsDeleteTransform(tf);

    prim->red   = pl_cie_from_XYZ(dst[0].X, dst[0].Y, dst[0].Z);
    prim->green = pl_cie_from_XYZ(dst[1].X, dst[1].Y, dst[1].Z);
    prim->blue  = pl_cie_from_XYZ(dst[2].X, dst[2].Y, dst[2].Z);
    prim->white = pl_cie_from_XYZ(dst[3].X, dst[3].Y, dst[3].Z);
    return true;
}

static bool detect_contrast(pl_icc_object icc, struct pl_hdr_metadata *hdr,
                            float max_luma)
{
    struct icc_priv *p = PL_PRIV(icc);
    cmsCIEXYZ *white = cmsReadTag(p->profile, cmsSigLuminanceTag);
    if (!cmsDetectDestinationBlackPoint(&p->black, p->profile, icc->params.intent, 0))
        return false;

    if (max_luma <= 0)
        max_luma = white ? white->Y : PL_COLOR_SDR_WHITE;

    hdr->max_luma = max_luma;
    hdr->min_luma = p->black.Y * max_luma;
    hdr->min_luma = PL_MAX(hdr->min_luma, 1e-6); // prevent true 0
    return true;
}

static void infer_clut_size(pl_log log, struct pl_icc_object *icc)
{
    struct icc_priv *p = PL_PRIV(icc);
    struct pl_icc_params *params = &icc->params;
    if (params->size_r && params->size_g && params->size_b) {
        pl_debug(log, "Using fixed 3DLUT size: %dx%dx%d",
                 (int) params->size_r, (int) params->size_g, (int) params->size_b);
        return;
    }

#define REQUIRE_SIZE(N) \
    params->size_r = PL_MAX(params->size_r, N); \
    params->size_g = PL_MAX(params->size_g, N); \
    params->size_b = PL_MAX(params->size_b, N)

    // Default size for sanity
    REQUIRE_SIZE(9);

    // Ensure enough precision to track the (absolute) black point
    if (p->black.Y > 1e-4) {
        float black_rel = powf(p->black.Y, 1.0f / icc->gamma);
        int min_size = 2 * (int) ceilf(1.0f / black_rel);
        REQUIRE_SIZE(min_size);
    }

    // Ensure enough precision to track the gamma curve
    if (cmsDetectRGBProfileGamma(p->profile, 1e-2) < 0) {
        REQUIRE_SIZE(65);
    } else if (cmsDetectRGBProfileGamma(p->profile, 1e-3) < 0) {
        REQUIRE_SIZE(33);
    } else if (cmsDetectRGBProfileGamma(p->profile, 1e-4) < 0) {
        REQUIRE_SIZE(17);
    }

    // Ensure enough precision to track any internal CLUTs
    cmsPipeline *pipe = NULL;
    switch (icc->params.intent) {
    case PL_INTENT_SATURATION:
        pipe = cmsReadTag(p->profile, cmsSigBToA2Tag);
        if (pipe)
            break;
        // fall through
    case PL_INTENT_RELATIVE_COLORIMETRIC:
    case PL_INTENT_ABSOLUTE_COLORIMETRIC:
    default:
        pipe = cmsReadTag(p->profile, cmsSigBToA1Tag);
        if (pipe)
            break;
        // fall through
    case PL_INTENT_PERCEPTUAL:
        pipe = cmsReadTag(p->profile, cmsSigBToA0Tag);
        break;
    }

    if (!pipe) {
        switch (icc->params.intent) {
        case PL_INTENT_SATURATION:
            pipe = cmsReadTag(p->profile, cmsSigAToB2Tag);
            if (pipe)
                break;
            // fall through
        case PL_INTENT_RELATIVE_COLORIMETRIC:
        case PL_INTENT_ABSOLUTE_COLORIMETRIC:
        default:
            pipe = cmsReadTag(p->profile, cmsSigAToB1Tag);
            if (pipe)
                break;
            // fall through
        case PL_INTENT_PERCEPTUAL:
            pipe = cmsReadTag(p->profile, cmsSigAToB0Tag);
            break;
        }
    }

    if (pipe) {
        for (cmsStage *stage = cmsPipelineGetPtrToFirstStage(pipe);
             stage; stage = cmsStageNext(stage))
        {
            switch (cmsStageType(stage)) {
            case cmsSigCLutElemType: ;
                _cmsStageCLutData *data = cmsStageData(stage);
                if (data->Params->nInputs != 3)
                    continue;
                params->size_r = PL_MAX(params->size_r, data->Params->nSamples[0]);
                params->size_g = PL_MAX(params->size_g, data->Params->nSamples[1]);
                params->size_b = PL_MAX(params->size_b, data->Params->nSamples[2]);
                break;

            default:
                continue;
            }
        }
    }

    // Clamp the output size to make sure profiles are not too large
    params->size_r = PL_MIN(params->size_r, 129);
    params->size_g = PL_MIN(params->size_g, 129);
    params->size_b = PL_MIN(params->size_b, 129);

    // Constrain the total LUT size to roughly 1M entries
    const size_t max_size = 1000000;
    size_t total_size = params->size_r * params->size_g * params->size_b;
    if (total_size > max_size) {
        float factor = powf((float) max_size / total_size, 1/3.0f);
        params->size_r = ceilf(factor * params->size_r);
        params->size_g = ceilf(factor * params->size_g);
        params->size_b = ceilf(factor * params->size_b);
    }

    pl_info(log, "Chosen 3DLUT size: %dx%dx%d",
            (int) params->size_r, (int) params->size_g, (int) params->size_b);
}

pl_icc_object pl_icc_open(pl_log log, const struct pl_icc_profile *profile,
                          const struct pl_icc_params *pparams)
{
    if (!profile->data)
        return NULL;

    struct pl_icc_object *icc = pl_zalloc_obj(NULL, icc, struct icc_priv);
    struct icc_priv *p = PL_PRIV(icc);
    struct pl_icc_params *params = &icc->params;
    *params = pparams ? *pparams : pl_icc_default_params;
    icc->signature = profile->signature;
    p->cms = cmsCreateContext(NULL, (void *) log);
    if (!p->cms) {
        pl_err(log, "Failed creating LittleCMS context!");
        goto error;
    }

    cmsSetLogErrorHandlerTHR(p->cms, error_callback);
    pl_info(log, "Opening ICC profile..");
    p->profile = cmsOpenProfileFromMemTHR(p->cms, profile->data, profile->len);
    if (!p->profile) {
        pl_err(log, "Failed opening ICC profile");
        goto error;
    }

    if (cmsGetColorSpace(p->profile) != cmsSigRgbData) {
        pl_err(log, "Invalid ICC profile: not RGB");
        goto error;
    }

    if (params->intent < 0 || params->intent > PL_INTENT_ABSOLUTE_COLORIMETRIC)
        params->intent = cmsGetHeaderRenderingIntent(p->profile);

    struct pl_raw_primaries *out_prim = &icc->csp.hdr.prim;
    if (!read_primaries(icc, out_prim))
        goto error;
    if (!detect_contrast(icc, &icc->csp.hdr, params->max_luma))
        goto error;
    if ((icc->gamma = cmsDetectRGBProfileGamma(p->profile, 10)) < 0)
        goto error;
    infer_clut_size(log, icc);

    const struct pl_raw_primaries *best = NULL;
    for (enum pl_color_primaries prim = 1; prim < PL_COLOR_PRIM_COUNT; prim++) {
        const struct pl_raw_primaries *raw = pl_raw_primaries_get(prim);
        if (!icc->csp.primaries && pl_raw_primaries_similar(raw, out_prim)) {
            icc->containing_primaries = prim;
            icc->csp.primaries = prim;
            best = raw;
            break;
        }

        if (pl_primaries_superset(raw, out_prim) &&
            (!best || pl_primaries_superset(best, raw)))
        {
            icc->containing_primaries = prim;
            best = raw;
        }
    }

    if (!best) {
        pl_err(log, "ICC profile too wide to handle!");
        goto error;
    }

    // Create approximation profile
    cmsToneCurve *curve = cmsBuildGamma(p->cms, icc->gamma);
    if (!curve)
        goto error;

    cmsCIExyY wp_xyY = { best->white.x, best->white.y, 1.0 };
    cmsCIExyYTRIPLE prim_xyY = {
        .Red   = { best->red.x,   best->red.y,   1.0 },
        .Green = { best->green.x, best->green.y, 1.0 },
        .Blue  = { best->blue.x,  best->blue.y,  1.0 },
    };

    p->approx = cmsCreateRGBProfileTHR(p->cms, &wp_xyY, &prim_xyY,
                        (cmsToneCurve *[3]){ curve, curve, curve });
    cmsFreeToneCurve(curve);
    if (!p->approx)
        goto error;

    return icc;

error:
    pl_icc_close((pl_icc_object *) &icc);
    return NULL;
}

static void fill_lut(void *datap, const struct sh_lut_params *params, bool decode)
{
    pl_icc_object icc = params->priv;
    struct icc_priv *p = PL_PRIV(icc);
    pl_log log = cmsGetContextUserData(p->cms);
    cmsHPROFILE srcp = decode ? p->profile : p->approx;
    cmsHPROFILE dstp = decode ? p->approx  : p->profile;

    clock_t start = clock();
    cmsHTRANSFORM tf = cmsCreateTransformTHR(p->cms, srcp, TYPE_RGB_16,
                                             dstp, TYPE_RGBA_FLT,
                                             icc->params.intent,
                                             cmsFLAGS_NOCACHE | cmsFLAGS_NOOPTIMIZE);
    if (!tf)
        return;

    clock_t after_transform = clock();
    pl_log_cpu_time(log, start, after_transform, "creating ICC transform");

    int s_r = params->width, s_g = params->height, s_b = params->depth;
    uint16_t *tmp = pl_alloc(NULL, s_r * 3 * sizeof(tmp[0]));
    for (int b = 0; b < s_b; b++) {
        for (int g = 0; g < s_g; g++) {
            // Transform a single line of the output buffer
            for (int r = 0; r < s_r; r++) {
                tmp[r * 3 + 0] = r * 65535 / (s_r - 1);
                tmp[r * 3 + 1] = g * 65535 / (s_g - 1);
                tmp[r * 3 + 2] = b * 65535 / (s_b - 1);
            }

            size_t offset = (b * s_g + g) * s_r * 4;
            float *data = ((float *) datap) + offset;
            cmsDoTransform(tf, tmp, data, s_r);
        }
    }

    pl_log_cpu_time(log, after_transform, clock(), "generating ICC 3DLUT");
    cmsDeleteTransform(tf);
    pl_free(tmp);
}

static void fill_decode(void *datap, const struct sh_lut_params *params)
{
    fill_lut(datap, params, true);
}

static void fill_encode(void *datap, const struct sh_lut_params *params)
{
    fill_lut(datap, params, false);
}

void pl_icc_decode(pl_shader sh, pl_icc_object icc, pl_shader_obj *lut_obj,
                   struct pl_color_space *out_csp)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    uint64_t sig = icc->signature;
    pl_hash_merge(&sig, icc->params.intent);

    ident_t lut = sh_lut(sh, sh_lut_params(
        .object     = lut_obj,
        .type       = PL_VAR_FLOAT,
        .width      = icc->params.size_r,
        .height     = icc->params.size_g,
        .depth      = icc->params.size_b,
        .comps      = 4,
        .linear     = true,
        .signature  = sig,
        .fill       = fill_decode,
        .priv       = (void *) icc,
    ));

    sh_describe(sh, "ICC 3DLUT");
    GLSL("// pl_icc_decode                      \n"
         "{                                     \n"
         "color.rgb = %s(color.rgb).rgb;        \n"
         "color.rgb = max(color.rgb, 0.0);      \n"
         "color.rgb = pow(color.rgb, vec3(%s)); \n"
         "color.rgb = %s * color.rgb;           \n"  // expand HDR levels
         "}                                     \n",
         lut, SH_FLOAT(icc->gamma),
         SH_FLOAT(pl_hdr_rescale(PL_HDR_NITS, PL_HDR_NORM, icc->csp.hdr.max_luma)));

    if (out_csp) {
        *out_csp = (struct pl_color_space) {
            .primaries  = icc->containing_primaries,
            .transfer   = PL_COLOR_TRC_LINEAR,
            .hdr        = icc->csp.hdr,
        };
    }
}

void pl_icc_encode(pl_shader sh, pl_icc_object icc, pl_shader_obj *lut_obj)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    ident_t lut = sh_lut(sh, sh_lut_params(
        .object     = lut_obj,
        .type       = PL_VAR_FLOAT,
        .width      = icc->params.size_r,
        .height     = icc->params.size_g,
        .depth      = icc->params.size_b,
        .comps      = 4,
        .linear     = true,
        .signature  = ~icc->signature, // avoid confusion with decoding LUTs
        .fill       = fill_encode,
        .priv       = (void *) icc,
    ));

    sh_describe(sh, "ICC 3DLUT");
    GLSL("// pl_icc_encode                      \n"
         "{                                     \n"
         "color.rgb = 1.0/%s * color.rgb;       \n"
         "color.rgb = max(color.rgb, 0.0);      \n"
         "color.rgb = pow(color.rgb, vec3(%s)); \n"
         "color.rgb = %s(color.rgb).rgb;        \n"
         "}                                     \n",
         SH_FLOAT(pl_hdr_rescale(PL_HDR_NITS, PL_HDR_NORM, icc->csp.hdr.max_luma)),
         SH_FLOAT(1.0f / icc->gamma), lut);
}

#else // !PL_HAVE_LCMS

void pl_icc_close(pl_icc_object *picc) {};
pl_icc_object pl_icc_open(pl_log log, const struct pl_icc_profile *profile,
                          const struct pl_icc_params *pparams)
{
    pl_err(log, "libplacebo compiled without LittleCMS 2 support!");
    return NULL;
}

void pl_icc_decode(pl_shader sh, pl_icc_object icc, pl_shader_obj *lut_obj,
                   struct pl_color_space *out_csp)
{
    pl_unreachable(); // can't get a pl_icc_object
}

void pl_icc_encode(pl_shader sh, pl_icc_object icc, pl_shader_obj *lut_obj)
{
    pl_unreachable();
}

#endif
