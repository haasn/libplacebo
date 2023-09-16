#include "tests.h"

#include <libplacebo/filters.h>

int main()
{
    pl_log log = pl_test_logger();

    for (int i = 0; i < pl_num_filter_functions; i++) {
        const struct pl_filter_function *fun = pl_filter_functions[i];
        if (fun->opaque)
            continue;

        printf("Testing filter function '%s'\n", fun->name);

        struct pl_filter_ctx ctx = { .radius = fun->radius };
        memcpy(ctx.params, fun->params, sizeof(ctx.params));

        // Ensure the kernel is correctly scaled
        REQUIRE_FEQ(fun->weight(&ctx, 0.0), 1.0, 1e-7);

        // Only box filters are radius 1, these are unwindowed by design.
        // Gaussian technically never reaches 0 even at its preconfigured radius.
        if (fun->radius > 1.0 && fun != &pl_filter_function_gaussian)
            REQUIRE_FEQ(fun->weight(&ctx, fun->radius), 0.0, 1e-7);
    }

    for (int c = 0; c < pl_num_filter_configs; c++) {
        const struct pl_filter_config *conf = pl_filter_configs[c];
        if (conf->kernel->opaque || conf->polar)
            continue;

        printf("Testing filter config '%s'\n", conf->name);
        pl_filter flt = pl_filter_generate(log, pl_filter_params(
            .config      = *conf,
            .lut_entries = 128,
        ));

        // Ensure the weights for each row add up to unity
        REQUIRE(flt);
        for (int i = 0; i < flt->params.lut_entries; i++) {
            float sum = 0.0;
            REQUIRE(flt->row_size);
            REQUIRE_CMP(flt->row_stride, >=, flt->row_size, "d");
            for (int n = 0; n < flt->row_size; n++) {
                float w = flt->weights[i * flt->row_stride + n];
                sum += w;
            }
            REQUIRE_FEQ(sum, 1.0, 1e-6);
        }

        pl_filter_free(&flt);
    }

    pl_log_destroy(&log);
}
