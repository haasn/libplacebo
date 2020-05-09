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

        printf("Testing filter '%s'\n", conf->name);
        const struct pl_filter *flt = pl_filter_generate(ctx, &params);
        REQUIRE(flt);

        if (params.config.polar) {
            // Ensure the kernel seems sanely scaled
            REQUIRE(feq(flt->weights[0], 1.0, 1e-7));
            REQUIRE(feq(flt->weights[params.lut_entries - 1], 0.0, 1e-7));
        } else {
            // Ensure the weights for each row add up to unity
            for (int i = 0; i < params.lut_entries; i++) {
                float sum = 0.0;
                REQUIRE(flt->row_size);
                REQUIRE(flt->row_stride >= flt->row_size);
                for (int n = 0; n < flt->row_size; n++) {
                    float w = flt->weights[i * flt->row_stride + n];
                    sum += w;
                }
                REQUIRE(feq(sum, 1.0, 1e-6));
            }
        }

        pl_filter_free(&flt);
    }

    // Test nearest neighbour at a variety of scales
    int size = 1280;
    for (int offset = 0; offset < 128; offset++) {
        float scale = (float) size / (size - offset);
        printf("Testing nearest filter at scale %f\n", scale);

        const struct pl_filter *flt = pl_filter_generate(ctx, &(struct pl_filter_params) {
            .config = pl_filter_nearest,
            .lut_entries = 64,
            .filter_scale = scale,
        });

        REQUIRE(flt);
        for (int i = 0; i < 64; i++) {
            REQUIRE(flt->row_size == 2);
            const float *w = &flt->weights[i * flt->row_stride];

            if (i < 64 / 2) {
                REQUIRE(feq(w[0], 1.0, 1e-7));
                REQUIRE(feq(w[1], 0.0, 1e-7));
            } else {
                REQUIRE(feq(w[0], 0.0, 1e-7));
                REQUIRE(feq(w[1], 1.0, 1e-7));
            }
        }

        pl_filter_free(&flt);
    }

    pl_context_destroy(&ctx);
}
