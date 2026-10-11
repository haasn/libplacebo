#include "utils.h"
#include "options.h"

#include <libplacebo/compositor.h>
#include <libplacebo/options.h>

// Set an option on an arbitrary struct, by path
#define SET(log, base, list, key, value) \
    pl_options_set_str_ex(log, base, list, pl_str0(key), pl_str0(value))

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

    const char *def_opts = "upscaler=lanczos,downscaler=hermite,frame_mixer=oversample,sigmoid=yes,peak_detect=yes,dither=yes";
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
    REQUIRE((data = pl_options_get(test, "tile_size")));
    REQUIRE_STREQ(data->opt->key, "tile_size");
    REQUIRE_CMP(*(int *) data->value, =, pl_render_default_params.tile_size, "d");
    REQUIRE_STREQ(data->text, "32");

    const char *hq_opts = "upscaler=ewa_lanczossharp,downscaler=hermite,frame_mixer=oversample,deband=yes,sigmoid=yes,peak_detect=yes,peak_percentile=99.99500274658203,contrast_recovery=0.30000001192092896,dither=yes";
    // fallback can produce different precision
    const char *hq_opts2 = "upscaler=ewa_lanczossharp,downscaler=hermite,frame_mixer=oversample,deband=yes,sigmoid=yes,peak_detect=yes,peak_percentile=99.99500274658203125,contrast_recovery=0.30000001192092896,dither=yes";

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

    // Test the option lists for structs outside of `pl_options`
    struct pl_color_space csp = {0};
    REQUIRE(SET(log, &csp, pl_color_space_opts, "primaries", "bt.2020"));
    REQUIRE_CMP(csp.primaries, ==, PL_COLOR_PRIM_BT_2020, "d");
    REQUIRE(SET(log, &csp, pl_color_space_opts, "transfer", "pq"));
    REQUIRE_CMP(csp.transfer, ==, PL_COLOR_TRC_PQ, "d");
    REQUIRE(SET(log, &csp, pl_color_space_opts, "hdr.max_luma", "1000"));
    REQUIRE_CMP(csp.hdr.max_luma, ==, 1000.0f, "f");

    struct pl_color_repr repr = {0};
    REQUIRE(SET(log, &repr, pl_color_repr_opts, "sys", "bt.2020-ncl"));
    REQUIRE_CMP(repr.sys, ==, PL_COLOR_SYSTEM_BT_2020_NC, "d");
    REQUIRE(SET(log, &repr, pl_color_repr_opts, "levels", "full"));
    REQUIRE_CMP(repr.levels, ==, PL_COLOR_LEVELS_FULL, "d");
    REQUIRE(SET(log, &repr, pl_color_repr_opts, "alpha", "premultiplied"));
    REQUIRE_CMP(repr.alpha, ==, PL_ALPHA_PREMULTIPLIED, "d");
    REQUIRE(SET(log, &repr, pl_color_repr_opts, "bits.sample_depth", "10"));
    REQUIRE_CMP(repr.bits.sample_depth, ==, 10, "d");
    REQUIRE_CMP(repr.bits.color_depth, ==, 0, "d"); // neighbour untouched

    struct pl_input_t input = {0};
    REQUIRE(SET(log, &input, pl_input_opts, "z_index", "-3"));
    REQUIRE_CMP(input.z_index, ==, -3, "d");
    REQUIRE(SET(log, &input, pl_input_opts, "rotation", "1"));
    REQUIRE_CMP(input.rotation, ==, PL_ROTATION_90, "d");
    REQUIRE(SET(log, &input, pl_input_opts, "fit", "scale_down"));
    REQUIRE_CMP(input.fit, ==, PL_FIT_SCALE_DOWN, "d");
    REQUIRE(SET(log, &input, pl_input_opts, "stop_mode", "down"));
    REQUIRE_CMP(input.stop_mode, ==, PL_PTS_ROUND_DOWN, "d");
    REQUIRE(SET(log, &input, pl_input_opts, "dst_rect.x1", "0.5"));
    REQUIRE_CMP(input.dst_rect.x1, ==, 0.5f, "f");

    // Three levels deep, through two different sub-structs
    REQUIRE(SET(log, &input, pl_input_opts, "color.hdr.min_luma", "0.005"));
    REQUIRE_CMP(input.color.hdr.min_luma, ==, 0.005f, "f");
    REQUIRE(SET(log, &input, pl_input_opts, "repr.bits.color_depth", "10"));
    REQUIRE_CMP(input.repr.bits.color_depth, ==, 10, "d");

    // PTS fields are `double`, and must not be truncated to `float`
    const double pts = 1234567.890123456;
    REQUIRE(SET(log, &input, pl_input_opts, "seek_pts", "1234567.890123456"));
    REQUIRE_CMP(input.seek_pts, ==, pts, "f");
    REQUIRE_CMP(input.seek_pts, !=, (double) (float) pts, "f");

    // Test error paths
    REQUIRE(!SET(NULL, &csp, pl_color_space_opts, "hdr", "1000"));
    REQUIRE(!SET(NULL, &csp, pl_color_space_opts, "hdr.nope", "1"));
    REQUIRE(!SET(NULL, &csp, pl_color_space_opts, "primaries.x", "bt.709"));
    REQUIRE(!SET(NULL, &csp, pl_color_space_opts, "nope.max_luma", "1"));
    REQUIRE(!SET(NULL, &csp, pl_color_space_opts, "primaries", "bt2020"));
    REQUIRE(!SET(NULL, &input, pl_input_opts, "seek_pts", "abc"));
    REQUIRE(!SET(NULL, &input, pl_input_opts, "fit", "squish"));
    REQUIRE(!SET(NULL, &input, pl_input_opts, "params", "yes")); // not exposed
    REQUIRE(!SET(NULL, &input, pl_input_opts, "", "1"));

    pl_options_free(&test);
    pl_log_destroy(&log);
    return 0;
}
