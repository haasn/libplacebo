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
    float a, b, scale; // approxmation tone curve parameters and scaling
    cmsCIEXYZ black;
    float gamma_stddev;
    uint64_t lut_sig;
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

static bool detect_csp(pl_icc_object icc, struct pl_raw_primaries *prim,
                       float *gamma)
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
                               cmsFLAGS_NOOPTIMIZE);
    cmsSetAdaptationStateTHR(p->cms, prev_adapt);
    cmsCloseProfile(xyz);
    if (!tf)
        return false;

    enum {
        RED,
        GREEN,
        BLUE,
        WHITE,
        BLACK,
        GRAY,
        RAMP,
    };

    static const uint8_t test[][3] = {
        [RED]   = { 0xFF,    0,    0 },
        [GREEN] = {    0, 0xFF,    0 },
        [BLUE]  = {    0,    0, 0xFF },
        [WHITE] = { 0xFF, 0xFF, 0xFF },
        [BLACK] = { 0x00, 0x00, 0x00 },
        [GRAY]  = { 0x80, 0x80, 0x80 },

        // Grayscale ramp (excluding endpoints)
#define V(d) { d, d, d }
                 V(0x01), V(0x02), V(0x03), V(0x04), V(0x05), V(0x06), V(0x07),
        V(0x08), V(0x09), V(0x0A), V(0x0B), V(0x0C), V(0x0D), V(0x0E), V(0x0F),
        V(0x10), V(0x11), V(0x12), V(0x13), V(0x14), V(0x15), V(0x16), V(0x17),
        V(0x18), V(0x19), V(0x1A), V(0x1B), V(0x1C), V(0x1D), V(0x1E), V(0x1F),
        V(0x20), V(0x21), V(0x22), V(0x23), V(0x24), V(0x25), V(0x26), V(0x27),
        V(0x28), V(0x29), V(0x2A), V(0x2B), V(0x2C), V(0x2D), V(0x2E), V(0x2F),
        V(0x30), V(0x31), V(0x32), V(0x33), V(0x34), V(0x35), V(0x36), V(0x37),
        V(0x38), V(0x39), V(0x3A), V(0x3B), V(0x3C), V(0x3D), V(0x3E), V(0x3F),
        V(0x40), V(0x41), V(0x42), V(0x43), V(0x44), V(0x45), V(0x46), V(0x47),
        V(0x48), V(0x49), V(0x4A), V(0x4B), V(0x4C), V(0x4D), V(0x4E), V(0x4F),
        V(0x50), V(0x51), V(0x52), V(0x53), V(0x54), V(0x55), V(0x56), V(0x57),
        V(0x58), V(0x59), V(0x5A), V(0x5B), V(0x5C), V(0x5D), V(0x5E), V(0x5F),
        V(0x60), V(0x61), V(0x62), V(0x63), V(0x64), V(0x65), V(0x66), V(0x67),
        V(0x68), V(0x69), V(0x6A), V(0x6B), V(0x6C), V(0x6D), V(0x6E), V(0x6F),
        V(0x70), V(0x71), V(0x72), V(0x73), V(0x74), V(0x75), V(0x76), V(0x77),
        V(0x78), V(0x79), V(0x7A), V(0x7B), V(0x7C), V(0x7D), V(0x7E), V(0x7F),
        V(0x80), V(0x81), V(0x82), V(0x83), V(0x84), V(0x85), V(0x86), V(0x87),
        V(0x88), V(0x89), V(0x8A), V(0x8B), V(0x8C), V(0x8D), V(0x8E), V(0x8F),
        V(0x90), V(0x91), V(0x92), V(0x93), V(0x94), V(0x95), V(0x96), V(0x97),
        V(0x98), V(0x99), V(0x9A), V(0x9B), V(0x9C), V(0x9D), V(0x9E), V(0x9F),
        V(0xA0), V(0xA1), V(0xA2), V(0xA3), V(0xA4), V(0xA5), V(0xA6), V(0xA7),
        V(0xA8), V(0xA9), V(0xAA), V(0xAB), V(0xAC), V(0xAD), V(0xAE), V(0xAF),
        V(0xB0), V(0xB1), V(0xB2), V(0xB3), V(0xB4), V(0xB5), V(0xB6), V(0xB7),
        V(0xB8), V(0xB9), V(0xBA), V(0xBB), V(0xBC), V(0xBD), V(0xBE), V(0xBF),
        V(0xC0), V(0xC1), V(0xC2), V(0xC3), V(0xC4), V(0xC5), V(0xC6), V(0xC7),
        V(0xC8), V(0xC9), V(0xCA), V(0xCB), V(0xCC), V(0xCD), V(0xCE), V(0xCF),
        V(0xD0), V(0xD1), V(0xD2), V(0xD3), V(0xD4), V(0xD5), V(0xD6), V(0xD7),
        V(0xD8), V(0xD9), V(0xDA), V(0xDB), V(0xDC), V(0xDD), V(0xDE), V(0xDF),
        V(0xE0), V(0xE1), V(0xE2), V(0xE3), V(0xE4), V(0xE5), V(0xE6), V(0xE7),
        V(0xE8), V(0xE9), V(0xEA), V(0xEB), V(0xEC), V(0xED), V(0xEE), V(0xEF),
        V(0xF0), V(0xF1), V(0xF2), V(0xF3), V(0xF4), V(0xF5), V(0xF6), V(0xF7),
        V(0xF8), V(0xF9), V(0xFA), V(0xFB), V(0xFC), V(0xFD), V(0xFE),
#undef V
    };

    cmsCIEXYZ dst[PL_ARRAY_SIZE(test)] = {0};
    cmsDoTransform(tf, test, dst, PL_ARRAY_SIZE(dst));
    cmsDeleteTransform(tf);

    // Read primaries from transformed RGBW values
    prim->red   = pl_cie_from_XYZ(dst[RED].X, dst[RED].Y, dst[RED].Z);
    prim->green = pl_cie_from_XYZ(dst[GREEN].X, dst[GREEN].Y, dst[GREEN].Z);
    prim->blue  = pl_cie_from_XYZ(dst[BLUE].X, dst[BLUE].Y, dst[BLUE].Z);
    prim->white = pl_cie_from_XYZ(dst[WHITE].X, dst[WHITE].Y, dst[WHITE].Z);

    // Rough estimate of overall gamma and starting point for curve black point
    const float y_approx = log(dst[GRAY].Y) / log(0.5);
    float b = powf(dst[BLACK].Y, 1 / y_approx);

    // Estimate mean and stddev of gamma (Welford's method)
    float M = 0.0, S = 0.0;
    int k = 1;
    for (int i = RAMP; i < PL_ARRAY_SIZE(dst); i++) { // exclude primaries
        if (dst[i].Y <= 0 || dst[i].Y >= 1)
            continue;
        float src = (1 - b) * (test[i][0] / 255.0) + b;
        float y = log(dst[i].Y) / log(src);
        float tmpM = M;
        M += (y - tmpM) / k;
        S += (y - tmpM) * (y - M);
        k++;

        // Update estimate of black point according to current gamma estimate
        b = powf(dst[BLACK].Y, 1 / M);
    }
    S = sqrt(S / (k - 1));
    if (S > 0.5)
        return false; // too big deviation, probably something broken??

    *gamma = M;
    p->gamma_stddev = S;
    return true;
}

static bool detect_contrast(pl_icc_object icc, struct pl_hdr_metadata *hdr,
                            float max_luma)
{
    struct icc_priv *p = PL_PRIV(icc);
    cmsCIEXYZ *white = cmsReadTag(p->profile, cmsSigLuminanceTag);
    enum pl_rendering_intent intent = icc->params.intent;
    /* LittleCMS refuses to detect an intent in absolute colorimetric intent,
     * so fall back to relative colorimetric since we only care about the
     * brightness value here */
    if (intent == PL_INTENT_ABSOLUTE_COLORIMETRIC)
        intent = PL_INTENT_RELATIVE_COLORIMETRIC;
    if (!cmsDetectDestinationBlackPoint(&p->black, p->profile, intent, 0))
        return false;

    if (max_luma <= 0)
        max_luma = white ? white->Y : PL_COLOR_SDR_WHITE;

    hdr->max_luma = max_luma;
    hdr->min_luma = p->black.Y * max_luma;
    hdr->min_luma = PL_MAX(hdr->min_luma, 1e-6); // prevent true 0
    return true;
}

static void infer_clut_size(pl_log log, struct pl_icc_object_t *icc)
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
    if (p->gamma_stddev > 1e-2) {
        REQUIRE_SIZE(65);
    } else if (p->gamma_stddev > 1e-3) {
        REQUIRE_SIZE(33);
    } else if (p->gamma_stddev > 1e-4) {
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

    struct pl_icc_object_t *icc = pl_zalloc_obj(NULL, icc, struct icc_priv);
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
    if (!detect_csp(icc, out_prim, &icc->gamma))
        goto error;
    if (!detect_contrast(icc, &icc->csp.hdr, params->max_luma))
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

    // Create approximation profile. Use a tone-curve based on a BT.1886-style
    // pure power curve, with an approximation gamma matched to the ICC
    // profile. We stretch the luminance range *before* the input to the gamma
    // function, to avoid numerical issues near the black point. (This removes
    // the need for a separate linear section)
    //
    // Y = scale * (aX + b)^y, where Y = PCS luma and X = encoded value ([0-1])
    p->scale = pl_hdr_rescale(PL_HDR_NITS, PL_HDR_NORM, icc->csp.hdr.max_luma);
    p->b = powf(icc->csp.hdr.min_luma / icc->csp.hdr.max_luma, 1.0f / icc->gamma);
    p->a = (1 - p->b);
    cmsToneCurve *curve = cmsBuildParametricToneCurve(p->cms, 2,
            (double[3]) { icc->gamma, p->a, p->b });
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

    // We need to create an ICC V2 profile because ICC V4 perceptual profiles
    // have normalized semantics, but we want colorimetric mapping with BPC
    cmsSetHeaderRenderingIntent(p->approx, icc->params.intent);
    cmsSetProfileVersion(p->approx, 2.2);

    // Hash all parameters affecting the generated 3DLUT
    p->lut_sig = icc->signature;
    pl_hash_merge(&p->lut_sig, params->intent);
    pl_hash_merge(&p->lut_sig, params->size_r);
    pl_hash_merge(&p->lut_sig, params->size_g);
    pl_hash_merge(&p->lut_sig, params->size_b);
    union { double d; uint64_t u; } v = { .d = icc->csp.hdr.max_luma };
    pl_hash_merge(&p->lut_sig, v.u);
    // min luma depends only on the max luma and profile

    return icc;

error:
    pl_icc_close((pl_icc_object *) &icc);
    return NULL;
}

static inline bool cache_load(pl_icc_object icc, uint64_t sig,
                              uint8_t *cache, size_t size)
{
    if (icc->params.cache_load)
        return icc->params.cache_load(icc->params.cache_priv, sig, cache, size);

    return false;
}

static void fill_lut(void *datap, const struct sh_lut_params *params, bool decode)
{
    pl_icc_object icc = params->priv;
    struct icc_priv *p = PL_PRIV(icc);
    pl_log log = cmsGetContextUserData(p->cms);
    cmsHPROFILE srcp = decode ? p->profile : p->approx;
    cmsHPROFILE dstp = decode ? p->approx  : p->profile;

    int s_r = params->width, s_g = params->height, s_b = params->depth;
    size_t data_size = s_r * s_g * s_b * sizeof(uint16_t[4]);
    if (cache_load(icc, params->signature, datap, data_size)) {
        pl_info(log, "Using cached 3DLUT (0x%"PRIX64")", params->signature);
        return;
    }

    clock_t start = clock();
    cmsHTRANSFORM tf = cmsCreateTransformTHR(p->cms, srcp, TYPE_RGB_16,
                                             dstp, TYPE_RGBA_16,
                                             icc->params.intent,
                                             cmsFLAGS_BLACKPOINTCOMPENSATION |
                                             cmsFLAGS_NOCACHE | cmsFLAGS_NOOPTIMIZE);
    if (!tf)
        return;

    clock_t after_transform = clock();
    pl_log_cpu_time(log, start, after_transform, "creating ICC transform");

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
            uint16_t *data = ((uint16_t *) datap) + offset;
            cmsDoTransform(tf, tmp, data, s_r);

            if (!icc->params.force_bpc)
                continue;

            // Fix the black point manually. Work-around for "improper"
            // profiles, as black point compensation should already have
            // taken care of this normally.
            const uint16_t knee = 16u << 8;
            if (tmp[0] >= knee || tmp[1] >= knee)
                continue;
            for (int r = 0; r < s_r; r++) {
                uint16_t s = (2 * tmp[1] + tmp[2] + tmp[r * 3]) >> 2;
                if (s >= knee)
                    break;
                for (int c = 0; c < 3; c++)
                    data[r * 3 + c] = (s * data[r * 3 + c] + (knee - s) * s) >> 12;
            }
        }
    }

    pl_log_cpu_time(log, after_transform, clock(), "generating ICC 3DLUT");
    cmsDeleteTransform(tf);
    pl_free(tmp);

    if (icc->params.cache_save) {
        icc->params.cache_save(icc->params.cache_priv, params->signature,
                               datap, data_size);
    }
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
    struct icc_priv *p = PL_PRIV(icc);
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    ident_t lut = sh_lut(sh, sh_lut_params(
        .object     = lut_obj,
        .var_type   = PL_VAR_FLOAT,
        .method     = SH_LUT_TETRAHEDRAL,
        .fmt        = pl_find_fmt(SH_GPU(sh), PL_FMT_UNORM, 4,
                                  16, 16, PL_FMT_CAP_LINEAR),
        .width      = icc->params.size_r,
        .height     = icc->params.size_g,
        .depth      = icc->params.size_b,
        .comps      = 4,
        .signature  = p->lut_sig,
        .fill       = fill_decode,
        .priv       = (void *) icc,
    ));

    // Y = scale * (aX + b)^y
    sh_describe(sh, "ICC 3DLUT");
    GLSL("// pl_icc_decode                      \n"
         "{                                     \n"
         "color.rgb = %s(color.rgb).rgb;        \n"
         "color.rgb = %s * color.rgb + vec3(%s);\n"
         "color.rgb = pow(color.rgb, vec3(%s)); \n"
         "color.rgb = %s * color.rgb;           \n"
         "}                                     \n",
         lut,
         SH_FLOAT(p->a), SH_FLOAT(p->b),
         SH_FLOAT(icc->gamma),
         SH_FLOAT(p->scale));

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
    struct icc_priv *p = PL_PRIV(icc);
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    ident_t lut = sh_lut(sh, sh_lut_params(
        .object     = lut_obj,
        .var_type   = PL_VAR_FLOAT,
        .method     = SH_LUT_TETRAHEDRAL,
        .fmt        = pl_find_fmt(SH_GPU(sh), PL_FMT_UNORM, 4,
                                  16, 16, PL_FMT_CAP_LINEAR),
        .width      = icc->params.size_r,
        .height     = icc->params.size_g,
        .depth      = icc->params.size_b,
        .comps      = 4,
        .signature  = ~p->lut_sig, // avoid confusion with decoding LUTs
        .fill       = fill_encode,
        .priv       = (void *) icc,
    ));

    // X = 1/a * (Y/scale)^(1/y) - b/a
    sh_describe(sh, "ICC 3DLUT");
    GLSL("// pl_icc_encode                      \n"
         "{                                     \n"
         "if (any(greaterThan(color.rgb, vec3(%s)))) \n"
         "    color.rgb = vec3(1,0,0); \n"
         "color.rgb = max(color.rgb, 0.0);      \n"
         "color.rgb = 1.0/%s * color.rgb;       \n"
         "color.rgb = pow(color.rgb, vec3(%s)); \n"
         "color.rgb = 1.0/%s * color.rgb - %s;  \n"
         "color.rgb = %s(color.rgb).rgb;        \n"
         "}                                     \n",
         SH_FLOAT(icc->csp.hdr.min_luma + 1e4),
         SH_FLOAT(p->scale),
         SH_FLOAT(1.0f / icc->gamma),
         SH_FLOAT(p->a), SH_FLOAT(p->b / p->a),
         lut);
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
