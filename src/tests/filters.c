#include "tests.h"

int main()
{
    struct pl_context *ctx = pl_test_context();
    for (const struct pl_named_filter_config *conf = pl_named_filters;
         conf->filter; conf++)
    {
        struct pl_filter_params params = {
            .config      = *conf->filter,
            .lut_entries = 128,
        };

        printf("== filter '%s' ==\n", conf->name);
        const struct pl_filter *flt = pl_filter_generate(ctx, &params);
        REQUIRE(flt);

        if (params.config.polar) {
            printf("lut:");
            for (int i = 0; i < params.lut_entries; i++)
                printf(" %f", flt->weights[i]);
            printf("\n");

            // Ensure the kernel seems sanely scaled
            REQUIRE(feq(flt->weights[0], 1.0));
            REQUIRE(feq(flt->weights[params.lut_entries - 1], 0.0));
        } else {
            // Ensure the weights for each row add up to unity
            for (int i = 0; i < params.lut_entries; i++) {
                printf("row %d:", i);
                float sum = 0.0;
                REQUIRE(flt->row_size);
                REQUIRE(flt->row_stride >= flt->row_size);
                for (int n = 0; n < flt->row_size; n++) {
                    float w = flt->weights[i * flt->row_stride + n];
                    printf(" %f", w);
                    sum += w;
                }
                printf(" = %f\n", sum);
                REQUIRE(feq(sum, 1.0));
            }
        }

        pl_filter_free(&flt);
    }
    pl_context_destroy(&ctx);
}
