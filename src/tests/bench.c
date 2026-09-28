#include "utils.h"

#include <libplacebo/dispatch.h>
#include <libplacebo/vulkan.h>
#include <libplacebo/shaders/colorspace.h>
#include <libplacebo/shaders/deinterlacing.h>
#include <libplacebo/shaders/motion.h>
#include <libplacebo/shaders/sampling.h>

enum {
    // Image configuration
    NUM_TEX     = 16,
    WIDTH       = 1920,
    HEIGHT      = 1080,
    DEPTH       = 16,
    COMPS       = 4,

    // Queue configuration
    NUM_QUEUES  = NUM_TEX,
    ASYNC_TX    = 1,
    ASYNC_COMP  = 1,

    // Test configuration
    TEST_MS     = 1000,
    WARMUP_MS   = 500,

    MAX_SRC     = 16,

    // Displacement between neighboring frames of that sequence
    FRAME_DX    = 6,
    FRAME_DY    = 3,
};

// `dx`/`dy` translate the generated image for motion estimation.
static pl_tex create_test_img(pl_gpu gpu, float dx, float dy)
{
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, COMPS, DEPTH, 32, PL_FMT_CAP_LINEAR);
    REQUIRE(fmt);

    const float xc = (WIDTH  - 1) / 2.0f + dx;
    const float yc = (HEIGHT - 1) / 2.0f + dy;
    const float kf = 0.5f / sqrtf(((WIDTH  - 1) / 2.0f) * ((WIDTH  - 1) / 2.0f) +
                                  ((HEIGHT - 1) / 2.0f) * ((HEIGHT - 1) / 2.0f));
    const float invphi = 0.61803398874989;
    const float freqR = kf * M_PI * 0.2f;
    const float freqG = freqR * invphi;
    const float freqB = freqG * invphi;
    float *data = malloc(WIDTH * HEIGHT * COMPS * sizeof(float));
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            float *color = &data[(y * WIDTH + x) * COMPS];
            float xx = x - xc, yy = y - yc;
            float r2 = xx * xx + yy * yy;
            switch ((int) COMPS) {
            case 4: color[3] = 1.0; /* fall through */
            case 3: color[2] = 0.5f * sinf(freqB * r2) + 0.5f; /* fall through */
            case 2: color[1] = 0.5f * sinf(freqG * r2) + 0.5f; /* fall through */
            case 1: color[0] = 0.5f * sinf(freqR * r2) + 0.5f; break;
            }
        }
    }

    pl_tex tex = pl_tex_create(gpu, pl_tex_params(
        .format         = fmt,
        .w              = WIDTH,
        .h              = HEIGHT,
        .sampleable     = true,
        .initial_data   = data,
    ));

    free(data);
    REQUIRE(tex);
    return tex;
}

struct bench {
    void (*run_sh)(pl_shader sh, pl_shader_obj *state,
                   pl_tex src);

    void (*run_tex)(pl_gpu gpu, pl_tex tex);

    // For algorithms that dispatch multiple separate shaders, or perform
    // a mix of texture allocations and shader dispatches.
    void (*run_multi)(pl_gpu gpu, pl_dispatch dp, pl_shader_obj *state,
                      pl_tex *srcs, int num_src, unsigned long iter);

    // Length of the source sequence handed to `run_multi`. Ignored by the
    // other benchmarking functions, which only see srcs[0].
    int num_src;
};

static void run_bench(pl_gpu gpu, pl_dispatch dp,
                      pl_shader_obj *state, pl_tex *srcs, int num_src,
                      unsigned long iter, pl_tex fbo, pl_timer timer,
                      const struct bench *bench)
{
    REQUIRE(bench);
    REQUIRE(bench->run_sh || bench->run_tex || bench->run_multi);
    if (bench->run_sh) {
        pl_shader sh = pl_dispatch_begin(dp);
        bench->run_sh(sh, state, srcs[0]);

        pl_dispatch_finish(dp, pl_dispatch_params(
            .shader = &sh,
            .target = fbo,
            .timer = timer,
        ));
    } else if (bench->run_multi) {
        bench->run_multi(gpu, dp, state, srcs, num_src, iter);
    } else {
        bench->run_tex(gpu, fbo);
    }
}

static void benchmark(pl_gpu gpu, const char *name,
                      const struct bench *bench)
{
    pl_dispatch dp = pl_dispatch_create(gpu->log, gpu);
    REQUIRE(dp);
    pl_shader_obj state = NULL;

    const int num_src = bench->run_multi ? bench->num_src : 1;
    // A multi-pass benchmark requires at least one pair of frames.
    REQUIRE(num_src >= (bench->run_multi ? 2 : 1) && num_src <= MAX_SRC);

    pl_tex srcs[MAX_SRC] = {0};
    for (int i = 0; i < num_src; i++)
        srcs[i] = create_test_img(gpu, i * FRAME_DX, i * FRAME_DY);

    // Create the FBOs
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, COMPS, DEPTH, 32,
                             PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_BLITTABLE);
    REQUIRE(fmt);

    pl_tex fbos[NUM_TEX] = {0};
    for (int i = 0; i < NUM_TEX; i++) {
        fbos[i] = pl_tex_create(gpu, pl_tex_params(
            .format         = fmt,
            .w              = WIDTH,
            .h              = HEIGHT,
            .renderable     = true,
            .blit_dst       = true,
            .host_writable  = true,
            .host_readable  = true,
            .storable       = !!(fmt->caps & PL_FMT_CAP_STORABLE),
        ));
        REQUIRE(fbos[i]);

        pl_tex_clear(gpu, fbos[i], (float[4]){ 0.0 });
    }

    // Run the benchmark and flush+block once to force shader compilation etc.
    run_bench(gpu, dp, &state, srcs, num_src, 0, fbos[0], NULL, bench);
    pl_gpu_finish(gpu);

    // Perform the actual benchmark
    pl_clock_t start_warmup = 0, start_test = 0;
    unsigned long frames = 0, frames_warmup = 0;

    pl_timer timer = pl_timer_create(gpu);
    uint64_t gputime_total = 0;
    unsigned long gputime_count = 0;
    uint64_t gputime;

    start_warmup = pl_clock_now();
    do {
        const int idx = frames % NUM_TEX;
        while (pl_tex_poll(gpu, fbos[idx], UINT64_MAX))
            ; // do nothing
        run_bench(gpu, dp, &state, srcs, num_src, frames, fbos[idx],
                  start_test ? timer : NULL, bench);
        pl_gpu_flush(gpu);
        frames++;

        if (start_test) {
            while ((gputime = pl_timer_query(gpu, timer))) {
                gputime_total += gputime;
                gputime_count++;
            }
        }

        pl_clock_t now = pl_clock_now();
        if (start_test) {
            if (pl_clock_diff(now, start_test) > TEST_MS * 1e-3)
                break;
        } else if (pl_clock_diff(now, start_warmup) > WARMUP_MS * 1e-3 &&
                   frames >= num_src)
        {
            start_test = now;
            frames_warmup = frames;
        }
    } while (true);

    // Force the GPU to finish execution and re-measure the final stop time
    pl_gpu_finish(gpu);

    pl_clock_t stop = pl_clock_now();
    while ((gputime = pl_timer_query(gpu, timer))) {
        gputime_total += gputime;
        gputime_count++;
    }

    frames -= frames_warmup;
    double secs = pl_clock_diff(stop, start_test);
    printf("'%s':\t%4lu frames in %1.6f seconds => %2.6f ms/frame (%5.2f FPS)",
          name, frames, secs, 1000 * secs / frames, frames / secs);
    if (gputime_count)
        printf(", gpu time: %2.6f ms", 1e-6 * gputime_total / gputime_count);
    printf("\n");

    pl_timer_destroy(gpu, &timer);
    pl_shader_obj_destroy(&state);
    pl_dispatch_destroy(&dp);
    for (int i = 0; i < num_src; i++)
        pl_tex_destroy(gpu, &srcs[i]);
    for (int i = 0; i < NUM_TEX; i++)
        pl_tex_destroy(gpu, &fbos[i]);
}

// List of benchmarks
static void bench_deband(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    pl_shader_deband(sh, pl_sample_src( .tex = src ), NULL);
}

static void bench_deband_heavy(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    pl_shader_deband(sh, pl_sample_src( .tex = src ), pl_deband_params(
        .iterations = 4,
        .threshold  = 4.0,
        .radius     = 4.0,
        .grain      = 16.0,
    ));
}

static void bench_bilinear(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    REQUIRE(pl_shader_sample_bilinear(sh, pl_sample_src( .tex = src )));
}

static void bench_bicubic(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    REQUIRE(pl_shader_sample_bicubic(sh, pl_sample_src( .tex = src )));
}

static void bench_hermite(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    REQUIRE(pl_shader_sample_hermite(sh, pl_sample_src( .tex = src )));
}

static void bench_gaussian(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    REQUIRE(pl_shader_sample_gaussian(sh, pl_sample_src( .tex = src )));
}

static void bench_dither_blue(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    REQUIRE(pl_shader_sample_direct(sh, pl_sample_src( .tex = src )));
    pl_shader_dither(sh, 8, state, pl_dither_params(
        .method = PL_DITHER_BLUE_NOISE,
    ));
}

static void bench_dither_white(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    REQUIRE(pl_shader_sample_direct(sh, pl_sample_src( .tex = src )));
    pl_shader_dither(sh, 8, state, pl_dither_params(
        .method = PL_DITHER_WHITE_NOISE,
    ));
}

static void bench_dither_ordered_fix(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    REQUIRE(pl_shader_sample_direct(sh, pl_sample_src( .tex = src )));
    pl_shader_dither(sh, 8, state, pl_dither_params(
        .method = PL_DITHER_ORDERED_FIXED,
    ));
}

static void bench_polar(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    struct pl_sample_filter_params params = {
        .filter = pl_filter_ewa_lanczos,
        .lut = state,
    };

    REQUIRE(pl_shader_sample_polar(sh, pl_sample_src( .tex = src ), &params));
}

static void bench_polar_nocompute(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    struct pl_sample_filter_params params = {
        .filter = pl_filter_ewa_lanczos,
        .no_compute = true,
        .lut = state,
    };

    REQUIRE(pl_shader_sample_polar(sh, pl_sample_src( .tex = src ), &params));
}

static void bench_hdr_peak(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    REQUIRE(pl_shader_sample_direct(sh, pl_sample_src( .tex = src )));
    REQUIRE(pl_shader_detect_peak(sh, pl_color_space_hdr10, state, &pl_peak_detect_default_params));
}

static void bench_hdr_peak_hq(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    REQUIRE(pl_shader_sample_direct(sh, pl_sample_src( .tex = src )));
    REQUIRE(pl_shader_detect_peak(sh, pl_color_space_hdr10, state, &pl_peak_detect_high_quality_params));
}

static void bench_hdr_lut(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    struct pl_color_map_params params = {
        PL_COLOR_MAP_DEFAULTS
        .tone_mapping_function  = &pl_tone_map_bt2390,
        .tone_mapping_mode      = PL_TONE_MAP_RGB,
    };

    REQUIRE(pl_shader_sample_direct(sh, pl_sample_src( .tex = src )));
    pl_shader_color_map_ex(sh, &params, pl_color_map_args(
        .src = pl_color_space_hdr10,
        .dst = pl_color_space_monitor,
        .state = state,
    ));
}

static void bench_hdr_clip(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    struct pl_color_map_params params = {
        PL_COLOR_MAP_DEFAULTS
        .tone_mapping_function  = &pl_tone_map_clip,
        .tone_mapping_mode      = PL_TONE_MAP_RGB,
    };

    REQUIRE(pl_shader_sample_direct(sh, pl_sample_src( .tex = src )));
    pl_shader_color_map_ex(sh, &params, pl_color_map_args(
        .src = pl_color_space_hdr10,
        .dst = pl_color_space_monitor,
        .state = state,
    ));
}

static void bench_weave(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    struct pl_deinterlace_source dsrc = {
        .cur = pl_field_pair(src),
        .field = PL_FIELD_TOP,
    };

    pl_shader_deinterlace(sh, &dsrc, pl_deinterlace_params(
        .algo = PL_DEINTERLACE_WEAVE,
    ));
}

static void bench_bob(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    struct pl_deinterlace_source dsrc = {
        .cur = pl_field_pair(src),
        .field = PL_FIELD_TOP,
    };

    pl_shader_deinterlace(sh, &dsrc, pl_deinterlace_params(
        .algo = PL_DEINTERLACE_BOB,
    ));
}

static void bench_yadif(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    struct pl_deinterlace_source dsrc = {
        .prev = pl_field_pair(src),
        .cur = pl_field_pair(src),
        .next = pl_field_pair(src),
        .field = PL_FIELD_TOP,
    };

    pl_shader_deinterlace(sh, &dsrc, pl_deinterlace_params(
        .algo = PL_DEINTERLACE_YADIF,
    ));
}

static void bench_bwdif(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    struct pl_deinterlace_source dsrc = {
        .prev = pl_field_pair(src),
        .cur = pl_field_pair(src),
        .next = pl_field_pair(src),
        .field = PL_FIELD_TOP,
    };

    pl_shader_deinterlace(sh, &dsrc, pl_deinterlace_params(
        .algo = PL_DEINTERLACE_BWDIF,
    ));
}

// Maintain a frame signature for each frame. Motion estimation invalidates
// it's cached data upon non-sequential frame signatures between calls.
static uint64_t motion_sig = 1;

static void run_motion(pl_gpu gpu, pl_dispatch dp, pl_shader_obj *state,
                       pl_tex src_a, pl_tex src_b,
                       enum pl_motion_direction dir, bool global,
                       const struct pl_motion_estimate_params *params)
{
    const uint64_t sig = motion_sig++;

    REQUIRE(pl_motion_estimate(pl_motion_estimate_input(
        .gpu        = gpu,
        .dispatch   = dp,
        .a          = {
            .tex    = src_a,
            .sig    = sig,
            .repr   = pl_color_repr_rgb,
            .color  = pl_color_space_srgb,
        },
        .b          = {
            .tex    = src_b,
            .sig    = sig + 1,
            .repr   = pl_color_repr_rgb,
            .color  = pl_color_space_srgb,
        },
        .direction  = dir,
        .estimate_global = global,
        .state      = state,
    ), params));
}

// Estimate motion between sequential pairs of frames, resetting the temporal
// state upon looping, or on every call if `cold`.
static void run_motion_seq(pl_gpu gpu, pl_dispatch dp, pl_shader_obj *state,
                           pl_tex *srcs, int num_src, unsigned long iter,
                           bool cold, enum pl_motion_direction dir, bool global,
                           const struct pl_motion_estimate_params *params)
{
    const int i = iter % (num_src - 1);
    if (!i || cold)
        pl_motion_reset(*state); // no-op while the state is still NULL

    // Nothing is rendered into an FBO, so throttle on the source texture
    // instead, which was last used `num_src - 1` iterations ago
    while (pl_tex_poll(gpu, srcs[i + 1], UINT64_MAX))
        ; // do nothing

    run_motion(gpu, dp, state, srcs[i], srcs[i + 1], dir, global, params);
}

static void bench_motion(pl_gpu gpu, pl_dispatch dp, pl_shader_obj *state,
                         pl_tex *srcs, int num_src, unsigned long iter)
{
    run_motion_seq(gpu, dp, state, srcs, num_src, iter, false,
                   PL_MOTION_FORWARD, false, NULL);
}

static void bench_motion_cold(pl_gpu gpu, pl_dispatch dp, pl_shader_obj *state,
                              pl_tex *srcs, int num_src, unsigned long iter)
{
    run_motion_seq(gpu, dp, state, srcs, num_src, iter, true,
                   PL_MOTION_FORWARD, false, NULL);
}

static void bench_motion_bidir(pl_gpu gpu, pl_dispatch dp, pl_shader_obj *state,
                               pl_tex *srcs, int num_src, unsigned long iter)
{
    run_motion_seq(gpu, dp, state, srcs, num_src, iter, false,
                   PL_MOTION_BOTH, false, NULL);
}

static void bench_motion_block4(pl_gpu gpu, pl_dispatch dp, pl_shader_obj *state,
                                pl_tex *srcs, int num_src, unsigned long iter)
{
    run_motion_seq(gpu, dp, state, srcs, num_src, iter, false,
                   PL_MOTION_FORWARD, false,
                   pl_motion_estimate_params( .block_size = 4 ));
}

static void bench_motion_global(pl_gpu gpu, pl_dispatch dp,
                                pl_shader_obj *state, pl_tex *srcs, int num_src,
                                unsigned long iter)
{
    run_motion_seq(gpu, dp, state, srcs, num_src, iter, false,
                   PL_MOTION_FORWARD, true, NULL);
}

static void bench_av1_grain(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    struct pl_film_grain_params params = {
        .data = {
            .type = PL_FILM_GRAIN_AV1,
            .params.av1 = av1_grain_data,
            .seed = rand(),
        },
        .tex = src,
        .components = 3,
        .component_mapping = {0, 1, 2},
        .repr = &(struct pl_color_repr) {0},
    };

    REQUIRE(pl_shader_film_grain(sh, state, &params));
}

static void bench_av1_grain_lap(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    struct pl_film_grain_params params = {
        .data = {
            .type = PL_FILM_GRAIN_AV1,
            .params.av1 = av1_grain_data,
            .seed = rand(),
        },
        .tex = src,
        .components = 3,
        .component_mapping = {0, 1, 2},
        .repr = &(struct pl_color_repr) {0},
    };

    params.data.params.av1.overlap = true;
    REQUIRE(pl_shader_film_grain(sh, state, &params));
}

static void bench_h274_grain(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    struct pl_film_grain_params params = {
        .data = {
            .type = PL_FILM_GRAIN_H274,
            .params.h274 = h274_grain_data,
            .seed = rand(),
        },
        .tex = src,
        .components = 3,
        .component_mapping = {0, 1, 2},
        .repr = &(struct pl_color_repr) {0},
    };

    REQUIRE(pl_shader_film_grain(sh, state, &params));
}

static void bench_reshape_poly(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    REQUIRE(pl_shader_sample_direct(sh, pl_sample_src( .tex = src )));
    pl_shader_dovi_reshape(sh, &(struct pl_dovi_metadata) { .comp = {
        {
            .num_pivots = 8,
            .pivots = {0.0, 0.0048828125, 0.0419921875, 0.177734375,
                       0.427734375, 0.677734375, 0.927734375, 0.9990234375},
            .method = {0, 0, 0, 0, 0, 0, 0},
            .poly_coeffs = {
                {0.00290930271, 2.30019712, 50.1446037},
                {0.00725257397, 1.88119054, -4.49443769},
                {0.0150123835, 1.61106598, -1.64833081},
                {0.0498571396, 1.2059114, -0.430627108},
                {0.0878019333, 1.01845241, -0.19669354},
                {0.120447636, 0.920134187, -0.122338772},
                {2.12430835, -3.30913281, 2.10893941},
            },
        }, {
            .num_pivots = 2,
            .pivots = {0.0, 0.9990234375},
            .method = {0},
            .poly_coeffs = {{-0.397901177, 1.85908031, 0}},
        }, {
            .num_pivots = 2,
            .pivots = {0.0, 0.9990234375},
            .method = {0},
            .poly_coeffs = {{-0.399355531, 1.85591626, 0}},
        },
    }});
}

static void bench_reshape_mmr(pl_shader sh, pl_shader_obj *state, pl_tex src)
{
    REQUIRE(pl_shader_sample_direct(sh, pl_sample_src( .tex = src )));
    pl_shader_dovi_reshape(sh, &dovi_meta); // this includes MMR
}

static float data[WIDTH * HEIGHT * COMPS + 8192];

static void bench_download(pl_gpu gpu, pl_tex tex)
{
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(
        .tex = tex,
        .ptr = (uint8_t *) PL_ALIGN((uintptr_t) data, 4096),
    )));
}

static void bench_upload(pl_gpu gpu, pl_tex tex)
{
    REQUIRE(pl_tex_upload(gpu, pl_tex_transfer_params(
        .tex = tex,
        .ptr = (uint8_t *) PL_ALIGN((uintptr_t) data, 4096),
    )));
}

static void dummy_cb(void *arg) {}

static void bench_download_async(pl_gpu gpu, pl_tex tex)
{
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(
        .tex = tex,
        .ptr = (uint8_t *) PL_ALIGN((uintptr_t) data, 4096),
        .callback = dummy_cb,
    )));
}

static void bench_upload_async(pl_gpu gpu, pl_tex tex)
{
    REQUIRE(pl_tex_upload(gpu, pl_tex_transfer_params(
        .tex = tex,
        .ptr = (uint8_t *) PL_ALIGN((uintptr_t) data, 4096),
        .callback = dummy_cb,
    )));
}

int main()
{
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    pl_log log = pl_log_create(PL_API_VER, pl_log_params(
        .log_cb     = isatty(fileno(stdout)) ? pl_log_color : pl_log_simple,
        .log_level  = PL_LOG_WARN,
    ));

    pl_vulkan vk = pl_vulkan_create(log, pl_vulkan_params(
        .allow_software = true,
        .async_transfer = ASYNC_TX,
        .async_compute  = ASYNC_COMP,
        .queue_count    = NUM_QUEUES,
    ));

    if (!vk)
        return SKIP;

#define BENCH_SH(fn)    &(struct bench) { .run_sh = fn }
#define BENCH_TEX(fn)   &(struct bench) { .run_tex = fn }
#define BENCH_MULTI(fn) BENCH_MULTI_N(fn, MAX_SRC)
#define BENCH_MULTI_N(fn, n) &(struct bench) { .run_multi = fn, .num_src = (n) }

    printf("= Running benchmarks =\n");
    benchmark(vk->gpu, "tex_download ptr", BENCH_TEX(bench_download));
    benchmark(vk->gpu, "tex_download ptr async", BENCH_TEX(bench_download_async));
    benchmark(vk->gpu, "tex_upload ptr", BENCH_TEX(bench_upload));
    benchmark(vk->gpu, "tex_upload ptr async", BENCH_TEX(bench_upload_async));
    benchmark(vk->gpu, "bilinear", BENCH_SH(bench_bilinear));
    benchmark(vk->gpu, "bicubic", BENCH_SH(bench_bicubic));
    benchmark(vk->gpu, "hermite", BENCH_SH(bench_hermite));
    benchmark(vk->gpu, "gaussian", BENCH_SH(bench_gaussian));
    benchmark(vk->gpu, "deband", BENCH_SH(bench_deband));
    benchmark(vk->gpu, "deband_heavy", BENCH_SH(bench_deband_heavy));

    // Deinterlacing
    benchmark(vk->gpu, "weave", BENCH_SH(bench_weave));
    benchmark(vk->gpu, "bob", BENCH_SH(bench_bob));
    benchmark(vk->gpu, "yadif", BENCH_SH(bench_yadif));
    benchmark(vk->gpu, "bwdif", BENCH_SH(bench_bwdif));

    // Polar sampling
    benchmark(vk->gpu, "polar", BENCH_SH(bench_polar));
    if (vk->gpu->glsl.compute)
        benchmark(vk->gpu, "polar_nocompute", BENCH_SH(bench_polar_nocompute));

    // Dithering algorithms
    benchmark(vk->gpu, "dither_blue", BENCH_SH(bench_dither_blue));
    benchmark(vk->gpu, "dither_white", BENCH_SH(bench_dither_white));
    benchmark(vk->gpu, "dither_ordered_fixed", BENCH_SH(bench_dither_ordered_fix));

    // HDR peak detection
    if (vk->gpu->glsl.compute) {
        benchmark(vk->gpu, "hdr_peakdetect",    BENCH_SH(bench_hdr_peak));
        benchmark(vk->gpu, "hdr_peakdetect_hq", BENCH_SH(bench_hdr_peak_hq));
    }

    // Tone mapping
    benchmark(vk->gpu, "hdr_lut", BENCH_SH(bench_hdr_lut));
    benchmark(vk->gpu, "hdr_clip", BENCH_SH(bench_hdr_clip));

    // Misc stuff
    benchmark(vk->gpu, "av1_grain", BENCH_SH(bench_av1_grain));
    benchmark(vk->gpu, "av1_grain_lap", BENCH_SH(bench_av1_grain_lap));
    benchmark(vk->gpu, "h274_grain", BENCH_SH(bench_h274_grain));
    benchmark(vk->gpu, "reshape_poly", BENCH_SH(bench_reshape_poly));
    benchmark(vk->gpu, "reshape_mmr", BENCH_SH(bench_reshape_mmr));

    if (vk->gpu->glsl.compute) {
        benchmark(vk->gpu, "motion", BENCH_MULTI(bench_motion));
        benchmark(vk->gpu, "motion_cold", BENCH_MULTI(bench_motion_cold));
        benchmark(vk->gpu, "motion_bidir", BENCH_MULTI(bench_motion_bidir));
        benchmark(vk->gpu, "motion_block4", BENCH_MULTI(bench_motion_block4));
        benchmark(vk->gpu, "motion_global", BENCH_MULTI(bench_motion_global));
    }

    pl_vulkan_destroy(&vk);
    pl_log_destroy(&log);
    return 0;
}
