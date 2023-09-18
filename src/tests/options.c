#include "tests.h"

#include <libplacebo/options.h>

static void count_cb(void *priv, pl_opt_data data)
{
    int *num = priv;
    printf("Iterating over option: %s = %s\n", data->opt->key, data->text);
    (*num)++;
}

static void set_cb(void *priv, pl_opt_data data)
{
    pl_options dst = priv;
    REQUIRE(pl_options_set_str(dst, data->opt->key, data->text));
}

int main()
{
    pl_log log = pl_test_logger();
    pl_options test = pl_options_alloc(log);

    REQUIRE_STREQ(pl_options_save(test), "");
    REQUIRE(pl_options_load(test, ""));
    REQUIRE_STREQ(pl_options_save(test), "");

    pl_options_reset(test, &pl_render_fast_params);
    REQUIRE_STREQ(pl_options_save(test), "");
    REQUIRE(pl_options_load(test, "preset=fast"));
    REQUIRE_STREQ(pl_options_save(test), "");

    const char *def_opts = "upscaler=spline36,downscaler=mitchell,frame_mixer=oversample,sigmoid=yes,peak_detect=yes,dither=yes";
    pl_options_reset(test, &pl_render_default_params);
    REQUIRE_STREQ(pl_options_save(test), def_opts);
    struct pl_options_t def_pre = *test;
    pl_options_reset(test, NULL);
    REQUIRE_STREQ(pl_options_save(test), "");
    REQUIRE(pl_options_load(test, def_opts));
    REQUIRE_STREQ(pl_options_save(test), def_opts);
    REQUIRE_MEMEQ(test, &def_pre, sizeof(*test));
    pl_options_reset(test, NULL);
    REQUIRE(pl_options_load(test, "preset=default"));
    REQUIRE_STREQ(pl_options_save(test), def_opts);
    REQUIRE_MEMEQ(test, &def_pre, sizeof(*test));

    int num = 0;
    pl_options_iterate(test, count_cb, &num);
    REQUIRE_CMP(num, ==, 6, "d");

    pl_opt_data data;
    REQUIRE((data = pl_options_get(test, "lut_entries")));
    REQUIRE_STREQ(data->opt->key, "lut_entries");
    REQUIRE_CMP(*(int *) data->value, =, pl_render_default_params.lut_entries, "d");
    REQUIRE_STREQ(data->text, "64");

    const char *hq_opts = "upscaler=ewa_lanczossharp,downscaler=mitchell,frame_mixer=hermite,deband=yes,sigmoid=yes,peak_detect=yes,peak_percentile=99.99500274658203,contrast_recovery=0.30000001192092896,dither=yes";
    // fallback can produce different precision
    const char *hq_opts2 = "upscaler=ewa_lanczossharp,downscaler=mitchell,frame_mixer=hermite,deband=yes,sigmoid=yes,peak_detect=yes,peak_percentile=99.99500274658203125,contrast_recovery=0.30000001192092896,dither=yes";

    pl_options_reset(test, &pl_render_high_quality_params);
    const char *opts = pl_options_save(test);
    if (!strcmp(opts, hq_opts2))
        hq_opts = hq_opts2;
    REQUIRE_STREQ(opts, hq_opts);
    struct pl_options_t hq_pre = *test;
    pl_options_reset(test, NULL);
    REQUIRE_STREQ(pl_options_save(test), "");
    REQUIRE(pl_options_load(test, hq_opts));
    REQUIRE_STREQ(pl_options_save(test), hq_opts);
    REQUIRE_MEMEQ(test, &hq_pre, sizeof(*test));
    REQUIRE(pl_options_load(test, "preset=high_quality"));
    REQUIRE_STREQ(pl_options_save(test), hq_opts);
    REQUIRE_MEMEQ(test, &hq_pre, sizeof(*test));

    pl_options test2 = pl_options_alloc(log);
    pl_options_iterate(test, set_cb, test2);
    REQUIRE_STREQ(pl_options_save(test), pl_options_save(test2));
    pl_options_free(&test2);

    // Test custom scalers
    pl_options_reset(test, pl_render_params(
        .upscaler = &(struct pl_filter_config) {
            .kernel = &pl_filter_function_jinc,
            .window = &pl_filter_function_jinc,
            .radius = 4.0,
            .polar  = true,
        },
    ));
    const char *jinc4_opts = "upscaler=custom,upscaler_kernel=jinc,upscaler_window=jinc,upscaler_radius=4,upscaler_polar=yes";
    REQUIRE_STREQ(pl_options_save(test), jinc4_opts);
    struct pl_options_t jinc4_pre = *test;
    pl_options_reset(test, NULL);
    REQUIRE(pl_options_load(test, "upscaler=custom,upscaler_preset=ewa_lanczos,upscaler_radius=4.0,upscaler_clamp=0.0"));
    REQUIRE_STREQ(pl_options_save(test), jinc4_opts);
    REQUIRE_MEMEQ(test, &jinc4_pre, sizeof(*test));

    // Test params presets
    pl_options_reset(test, NULL);
    REQUIRE(pl_options_load(test, "cone=yes,cone_preset=deuteranomaly"));
    REQUIRE_STREQ(pl_options_save(test), "cone=yes,cones=m,cone_strength=0.5");

    // Test error paths
    pl_options bad = pl_options_alloc(NULL);
    REQUIRE(!pl_options_load(bad, "scale_preset=help"));
    REQUIRE(!pl_options_load(bad, "dither_method=invalid"));
    REQUIRE(!pl_options_load(bad, "lut_entries=-1"));
    REQUIRE(!pl_options_load(bad, "deband_iterations=100"));
    REQUIRE(!pl_options_load(bad, "tone_lut_size=abc"));
    REQUIRE(!pl_options_load(bad, "show_clipping=hello"));
    REQUIRE(!pl_options_load(bad, "brightness=2.0"));
    REQUIRE(!pl_options_load(bad, "gamma=oops"));
    REQUIRE(!pl_options_load(bad, "invalid"));
    REQUIRE(!pl_options_load(bad, "="));
    REQUIRE(!pl_options_load(bad, "preset==bar"));
    REQUIRE(!pl_options_load(bad, "peak_percentile=E8203125"));
    REQUIRE(!pl_options_get(bad, "invalid"));
    REQUIRE_STREQ(pl_options_save(bad), "");
    pl_options_free(&bad);

    pl_options_free(&test);
    pl_log_destroy(&log);
    return 0;
}
