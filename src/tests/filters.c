#include "tests.h"

#include <libplacebo/filters.h>

int main()
{
    pl_log log = pl_test_logger();
    for (const struct pl_filter_preset *conf = pl_filter_presets; conf->name; conf++) {
        if (!conf->filter)
            continue;

        struct pl_filter_params params = {
            .config      = *conf->filter,
            .lut_entries = 128,
        };

        printf("Testing filter '%s'\n", conf->name);
        pl_filter flt = pl_filter_generate(log, &params);
        REQUIRE(flt);

        if (params.config.polar) {
            // Ensure the kernel seems sanely scaled
            REQUIRE_FEQ(flt->weights[0], 1.0, 1e-7);
            REQUIRE_FEQ(flt->weights[params.lut_entries - 1], 0.0, 1e-7);
        } else {
            // Ensure the weights for each row add up to unity
            for (int i = 0; i < params.lut_entries; i++) {
                float sum = 0.0;
                REQUIRE(flt->row_size);
                REQUIRE_CMP(flt->row_stride, >=, flt->row_size, "d");
                for (int n = 0; n < flt->row_size; n++) {
                    float w = flt->weights[i * flt->row_stride + n];
                    sum += w;
                }
                REQUIRE_FEQ(sum, 1.0, 1e-6);
            }
        }

        pl_filter_free(&flt);
    }
    pl_log_destroy(&log);
}
