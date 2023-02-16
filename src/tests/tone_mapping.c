#include "tests.h"
#include "log.h"

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

    pl_log_destroy(&log);
}
