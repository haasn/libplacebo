#include "tests.h"
#include "log.h"

#include <libplacebo/gamut_mapping.h>
#include <libplacebo/tone_mapping.h>

//#define PRINT_LUTS

int main()
{
    pl_log log = pl_test_logger();

    // PQ unit tests
    REQUIRE_FEQ(pl_hdr_rescale(PL_HDR_PQ, PL_HDR_NITS, 0.0), 0.0,     1e-2);
    REQUIRE_FEQ(pl_hdr_rescale(PL_HDR_PQ, PL_HDR_NITS, 1.0), 10000.0, 1e-2);
    REQUIRE_FEQ(pl_hdr_rescale(PL_HDR_PQ, PL_HDR_NITS, 0.58), 203.0,  1e-2);

    // Test round-trip
    for (float x = 0.0f; x < 1.0f; x += 0.01f) {
        REQUIRE_FEQ(x, pl_hdr_rescale(PL_HDR_NORM, PL_HDR_PQ,
                       pl_hdr_rescale(PL_HDR_PQ, PL_HDR_NORM, x)),
                    1e-5);
    }

    static float lut[128];
    struct pl_tone_map_params params = {
        .input_scaling = PL_HDR_PQ,
        .output_scaling = PL_HDR_PQ,
        .lut_size = PL_ARRAY_SIZE(lut),
    };

    // Test regular tone-mapping
    params.input_min = pl_hdr_rescale(PL_HDR_NITS, params.input_scaling, 0.005);
    params.input_max = pl_hdr_rescale(PL_HDR_NITS, params.input_scaling, 1000.0);
    params.output_min = pl_hdr_rescale(PL_HDR_NORM, params.output_scaling, 0.001);
    params.output_max = pl_hdr_rescale(PL_HDR_NORM, params.output_scaling, 1.0);

    struct pl_tone_map_params params_inv = params;
    PL_SWAP(params_inv.input_min, params_inv.output_min);
    PL_SWAP(params_inv.input_max, params_inv.output_max);

    int tested_pure_bpc = 0;

    // Generate example tone mapping curves, forward and inverse
    for (int i = 0; i < pl_num_tone_map_functions; i++) {
        const struct pl_tone_map_function *fun = pl_tone_map_functions[i];
        if (fun == &pl_tone_map_auto)
            continue;

        printf("Testing tone-mapping function %s\n", fun->name);
        params.function = params_inv.function = fun;
        clock_t start = clock();
        pl_tone_map_generate(lut, &params);
        pl_log_cpu_time(log, start, clock(), "generating LUT");
        for (int j = 0; j < PL_ARRAY_SIZE(lut); j++) {
            REQUIRE(isfinite(lut[j]) && !isnan(lut[j]));
            if (j > 0)
                REQUIRE_CMP(lut[j], >=, lut[j - 1], "f");
#ifdef PRINT_LUTS
            printf("%f, %f\n", j / (PL_ARRAY_SIZE(lut) - 1.0f), lut[j]);
#endif
        }

        if (fun->map_inverse || !tested_pure_bpc++) {
            start = clock();
            pl_tone_map_generate(lut, &params_inv);
            pl_log_cpu_time(log, start, clock(), "generating inverse LUT");
            for (int j = 0; j < PL_ARRAY_SIZE(lut); j++) {
                REQUIRE(isfinite(lut[j]) && !isnan(lut[j]));
                if (j > 0)
                    REQUIRE_CMP(lut[j], >=, lut[j - 1], "f");
#ifdef PRINT_LUTS
                printf("%f, %f\n", j / (PL_ARRAY_SIZE(lut) - 1.0f), lut[j]);
#endif
            }
        }
    }

    // Test that `auto` is a no-op for 1:1 tone mapping
    params.output_min = params.input_min;
    params.output_max = params.input_max;
    params.function = &pl_tone_map_auto;
    pl_tone_map_generate(lut, &params);
    for (int j = 0; j < PL_ARRAY_SIZE(lut); j++) {
        float x = j / (PL_ARRAY_SIZE(lut) - 1.0f);
        x = PL_MIX(params.input_min, params.input_max, x);
        REQUIRE_FEQ(x, lut[j], 1e-5);
    }

    // Test some gamut mapping methods
    for (int i = 0; i < pl_num_gamut_map_functions; i++) {
        static const float min_rgb = 0.1f, max_rgb = PL_COLOR_SDR_WHITE;
        struct pl_gamut_map_params gamut = {
            .function     = pl_gamut_map_functions[i],
            .input_gamut  = *pl_raw_primaries_get(PL_COLOR_PRIM_BT_2020),
            .output_gamut = *pl_raw_primaries_get(PL_COLOR_PRIM_BT_709),
            .min_luma     = pl_hdr_rescale(PL_HDR_NITS, PL_HDR_PQ, min_rgb),
            .max_luma     = pl_hdr_rescale(PL_HDR_NITS, PL_HDR_PQ, max_rgb),
        };

        printf("Testing gamut-mapping function %s\n", gamut.function->name);

        // Require that black maps to black and white maps to white
        float black[3] = { gamut.min_luma, 0.0f, 0.0f };
        float white[3] = { gamut.max_luma, 0.0f, 0.0f };
        pl_gamut_map_sample(black, &gamut);
        pl_gamut_map_sample(white, &gamut);
        REQUIRE_FEQ(black[0], gamut.min_luma, 1e-4);
        REQUIRE_FEQ(black[1], 0.0f, 1e-4);
        REQUIRE_FEQ(black[2], 0.0f, 1e-4);
        if (gamut.function != &pl_gamut_map_darken)
            REQUIRE_FEQ(white[0], gamut.max_luma, 1e-4);
        REQUIRE_FEQ(white[1], 0.0f, 1e-4);
        REQUIRE_FEQ(white[2], 0.0f, 1e-4);
    }

    // Test that primaries round-trip for perceptual gamut mapping
    struct pl_gamut_map_params perceptual = {
        .function     = &pl_gamut_map_perceptual,
        .input_gamut  = *pl_raw_primaries_get(PL_COLOR_PRIM_BT_2020),
        .output_gamut = *pl_raw_primaries_get(PL_COLOR_PRIM_BT_709),
        .max_luma     = pl_hdr_rescale(PL_HDR_NORM, PL_HDR_PQ, 1.0f),
    };

    const pl_matrix3x3 rgb2lms_src = pl_ipt_rgb2lms(&perceptual.input_gamut);
    const pl_matrix3x3 rgb2lms_dst = pl_ipt_rgb2lms(&perceptual.output_gamut);
    static const float refpoints[][3] = {
        {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
        {0, 1, 1}, {1, 0, 1}, {1, 1, 0},
    };

    for (int i = 0; i < PL_ARRAY_SIZE(refpoints); i++) {
        float c[3]   = { refpoints[i][0], refpoints[i][1], refpoints[i][2] };
        float ref[3] = { refpoints[i][0], refpoints[i][1], refpoints[i][2] };
        pl_matrix3x3_apply(&rgb2lms_src, c);
        c[0] = pl_hdr_rescale(PL_HDR_NORM, PL_HDR_PQ, c[0]);
        c[1] = pl_hdr_rescale(PL_HDR_NORM, PL_HDR_PQ, c[1]);
        c[2] = pl_hdr_rescale(PL_HDR_NORM, PL_HDR_PQ, c[2]);
        pl_matrix3x3_apply(&pl_ipt_lms2ipt, c);
        pl_gamut_map_sample(c, &perceptual);

        pl_matrix3x3_apply(&rgb2lms_dst, ref);
        ref[0] = pl_hdr_rescale(PL_HDR_NORM, PL_HDR_PQ, ref[0]);
        ref[1] = pl_hdr_rescale(PL_HDR_NORM, PL_HDR_PQ, ref[1]);
        ref[2] = pl_hdr_rescale(PL_HDR_NORM, PL_HDR_PQ, ref[2]);
        pl_matrix3x3_apply(&pl_ipt_lms2ipt, ref);

        float hue_mapped = atan2f(c[2], c[1]);
        float hue_ref = atan2f(ref[2], ref[1]);
        REQUIRE_FEQ(hue_mapped, hue_ref, 1e-3);
    }

    pl_log_destroy(&log);
}
