#include "tests.h"
#include <sys/time.h>

#define TEX_SIZE 2048
#define CUBE_SIZE 64
#define NUM_FBOS 16
#define BENCH_DUR 3

static const struct pl_tex *create_test_img(const struct pl_gpu *gpu)
{
    const struct pl_fmt *fmt;
    fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 16, 32, PL_FMT_CAP_LINEAR);
    REQUIRE(fmt);

    int cube_stride = TEX_SIZE / CUBE_SIZE;
    int cube_count  = cube_stride * cube_stride;

    assert(cube_count * CUBE_SIZE * CUBE_SIZE == TEX_SIZE * TEX_SIZE);
    float *data = malloc(TEX_SIZE * TEX_SIZE * sizeof(float[4]));
    for (int n = 0; n < cube_count; n++) {
        int xbase = (n % cube_stride) * CUBE_SIZE;
        int ybase = (n / cube_stride) * CUBE_SIZE;
        for (int g = 0; g < CUBE_SIZE; g++) {
            for (int r = 0; r < CUBE_SIZE; r++) {
                int xpos = xbase + r;
                int ypos = ybase + g;
                assert(xpos < TEX_SIZE && ypos < TEX_SIZE);

                float *color = &data[(ypos * TEX_SIZE + xpos) * 4];
                color[0] = (float) r / CUBE_SIZE;
                color[1] = (float) g / CUBE_SIZE;
                color[2] = (float) n / cube_count;
                color[3] = 1.0;
            }
        }
    }

    const struct pl_tex *tex = pl_tex_create(gpu, &(struct pl_tex_params) {
        .format         = fmt,
        .w              = TEX_SIZE,
        .h              = TEX_SIZE,
        .sampleable     = true,
        .sample_mode    = PL_TEX_SAMPLE_LINEAR,
        .initial_data   = data,
    });

    free(data);
    REQUIRE(tex);
    return tex;
}

struct bench {
    void (*run_sh)(struct pl_shader *sh, struct pl_shader_obj **state,
                   const struct pl_tex *src);

    void (*run_tex)(const struct pl_gpu *gpu, const struct pl_tex *tex);
};

static void run_bench(const struct pl_gpu *gpu, struct pl_dispatch *dp,
                      struct pl_shader_obj **state, const struct pl_tex *src,
                      const struct pl_tex *fbo, struct pl_timer *timer,
                      const struct bench *bench)
{
    if (bench->run_sh) {
        struct pl_shader *sh = pl_dispatch_begin(dp);
        bench->run_sh(sh, state, src);

        pl_dispatch_finish(dp, &(struct pl_dispatch_params) {
            .shader = &sh,
            .target = fbo,
            .timer = timer,
        });
    } else {
        bench->run_tex(gpu, fbo);
    }
}

static void benchmark(const struct pl_gpu *gpu, const char *name,
                      const struct bench *bench)
{
    struct pl_dispatch *dp = pl_dispatch_create(gpu->ctx, gpu);
    struct pl_shader_obj *state = NULL;
    const struct pl_tex *src = create_test_img(gpu);

    // Create the FBOs
    const struct pl_fmt *fmt;
    fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 16, 32,
                      PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_BLITTABLE);
    REQUIRE(fmt);

    const struct pl_tex *fbos[NUM_FBOS] = {0};
    for (int i = 0; i < NUM_FBOS; i++) {
        fbos[i] = pl_tex_create(gpu, &(struct pl_tex_params) {
            .format         = fmt,
            .w              = TEX_SIZE,
            .h              = TEX_SIZE,
            .renderable     = true,
            .blit_dst       = true,
            .host_writable  = true,
            .host_readable  = true,
            .storable       = !!(fmt->caps & PL_FMT_CAP_STORABLE),
        });
        REQUIRE(fbos[i]);

        pl_tex_clear(gpu, fbos[i], (float[4]){ 0.0 });
    }

    // Run the benchmark and flush+block once to force shader compilation etc.
    run_bench(gpu, dp, &state, src, fbos[0], NULL, bench);
    pl_gpu_finish(gpu);

    // Perform the actual benchmark
    struct timeval start = {0}, stop = {0};
    unsigned long frames = 0;
    int index = 0;

    struct pl_timer *timer = pl_timer_create(gpu);
    uint64_t gputime_total = 0;
    unsigned long gputime_count = 0;
    uint64_t gputime;

    gettimeofday(&start, NULL);
    do {
        frames++;
        run_bench(gpu, dp, &state, src, fbos[index++], timer, bench);
        index %= NUM_FBOS;
        if (index == 0) {
            pl_gpu_flush(gpu);
            gettimeofday(&stop, NULL);
        }
        while ((gputime = pl_timer_query(gpu, timer))) {
            gputime_total += gputime;
            gputime_count++;
        }
    } while (stop.tv_sec - start.tv_sec < BENCH_DUR);

    // Force the GPU to finish execution and re-measure the final stop time
    pl_gpu_finish(gpu);

    gettimeofday(&stop, NULL);
    while ((gputime = pl_timer_query(gpu, timer))) {
        gputime_total += gputime;
        gputime_count++;
    }

    float secs = (float) (stop.tv_sec - start.tv_sec) +
                 1e-6 * (stop.tv_usec - start.tv_usec);
    printf("'%s':\t%4lu frames in %1.6f seconds => %2.6f ms/frame (%5.2f FPS)",
          name, frames, secs, 1000 * secs / frames, frames / secs);
    if (gputime_count)
        printf(", gpu time: %2.6f ms", 1e-6 * (gputime_total / gputime_count));
    printf("\n");

    pl_timer_destroy(gpu, &timer);
    pl_shader_obj_destroy(&state);
    pl_dispatch_destroy(&dp);
    pl_tex_destroy(gpu, &src);
    for (int i = 0; i < NUM_FBOS; i++)
        pl_tex_destroy(gpu, &fbos[i]);
}

// List of benchmarks
static void bench_deband(struct pl_shader *sh, struct pl_shader_obj **state,
                         const struct pl_tex *src)
{
    pl_shader_deband(sh, &(struct pl_sample_src) { .tex = src }, NULL);
}

static void bench_deband_heavy(struct pl_shader *sh, struct pl_shader_obj **state,
                               const struct pl_tex *src)
{
    pl_shader_deband(sh, &(struct pl_sample_src) { .tex = src },
        &(struct pl_deband_params) {
        .iterations = 4,
        .threshold  = 4.0,
        .radius     = 4.0,
        .grain      = 16.0,
    });
}

static void bench_bilinear(struct pl_shader *sh, struct pl_shader_obj **state,
                          const struct pl_tex *src)
{
    pl_shader_sample_direct(sh, &(struct pl_sample_src) { .tex = src });
}

static void bench_bicubic(struct pl_shader *sh, struct pl_shader_obj **state,
                          const struct pl_tex *src)
{
    pl_shader_sample_bicubic(sh, &(struct pl_sample_src) { .tex = src });
}

static void bench_dither_blue(struct pl_shader *sh, struct pl_shader_obj **state,
                              const struct pl_tex *src)
{
    struct pl_dither_params params = pl_dither_default_params;
    params.method = PL_DITHER_BLUE_NOISE;

    pl_shader_sample_direct(sh, &(struct pl_sample_src) { .tex = src });
    pl_shader_dither(sh, 8, state, &params);
}

static void bench_dither_white(struct pl_shader *sh, struct pl_shader_obj **state,
                               const struct pl_tex *src)
{
    struct pl_dither_params params = pl_dither_default_params;
    params.method = PL_DITHER_WHITE_NOISE;

    pl_shader_sample_direct(sh, &(struct pl_sample_src) { .tex = src });
    pl_shader_dither(sh, 8, state, &params);
}

static void bench_dither_ordered_fix(struct pl_shader *sh,
                                     struct pl_shader_obj **state,
                                     const struct pl_tex *src)
{
    struct pl_dither_params params = pl_dither_default_params;
    params.method = PL_DITHER_ORDERED_FIXED;

    pl_shader_sample_direct(sh, &(struct pl_sample_src) { .tex = src });
    pl_shader_dither(sh, 8, state, &params);
}

static void bench_polar(struct pl_shader *sh, struct pl_shader_obj **state,
                        const struct pl_tex *src)
{
    struct pl_sample_filter_params params = {
        .filter = pl_filter_ewa_lanczos,
        .lut = state,
    };

    pl_shader_sample_polar(sh, &(struct pl_sample_src) { .tex = src }, &params);
}

static void bench_polar_nocompute(struct pl_shader *sh,
                                  struct pl_shader_obj **state,
                                  const struct pl_tex *src)
{
    struct pl_sample_filter_params params = {
        .filter = pl_filter_ewa_lanczos,
        .no_compute = true,
        .lut = state,
    };

    pl_shader_sample_polar(sh, &(struct pl_sample_src) { .tex = src }, &params);
}


static void bench_hdr_peak(struct pl_shader *sh, struct pl_shader_obj **state,
                            const struct pl_tex *src)
{
    pl_shader_sample_direct(sh, &(struct pl_sample_src) { .tex = src });
    pl_shader_detect_peak(sh, pl_color_space_hdr10, state, NULL);
}

static void bench_av1_grain(struct pl_shader *sh, struct pl_shader_obj **state,
                            const struct pl_tex *src)
{
    struct pl_av1_grain_params params = {
        .data = av1_grain_data,
        .tex = src,
        .components = 3,
        .component_mapping = {0, 1, 2},
        .repr = &(struct pl_color_repr) {0},
    };

    params.data.grain_seed = rand();
    pl_shader_av1_grain(sh, state, &params);
}

static void bench_av1_grain_lap(struct pl_shader *sh, struct pl_shader_obj **state,
                                const struct pl_tex *src)
{
    struct pl_av1_grain_params params = {
        .data = av1_grain_data,
        .tex = src,
        .components = 3,
        .component_mapping = {0, 1, 2},
        .repr = &(struct pl_color_repr) {0},
    };

    params.data.overlap = true;
    params.data.grain_seed = rand();
    pl_shader_av1_grain(sh, state, &params);
}

static float data[TEX_SIZE * TEX_SIZE * 4 + 8192];

static void bench_download(const struct pl_gpu *gpu, const struct pl_tex *tex)
{
    pl_tex_download(gpu, &(struct pl_tex_transfer_params) {
        .tex = tex,
        .ptr = (uint8_t *) PL_ALIGN((uintptr_t) data, 4096),
    });
}

static void bench_upload(const struct pl_gpu *gpu, const struct pl_tex *tex)
{
    pl_tex_upload(gpu, &(struct pl_tex_transfer_params) {
        .tex = tex,
        .ptr = (uint8_t *) PL_ALIGN((uintptr_t) data, 4096),
    });
}

int main()
{
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    struct pl_context *ctx;
    ctx = pl_context_create(PL_API_VER, &(struct pl_context_params) {
        .log_cb     = isatty(fileno(stdout)) ? pl_log_color : pl_log_simple,
        .log_level  = PL_LOG_WARN,
    });

    const struct pl_vulkan *vk = pl_vulkan_create(ctx, &(struct pl_vulkan_params) {
        .allow_software = true,
        .async_compute = true,
        .queue_count = NUM_FBOS,
    });

    if (!vk)
        return SKIP;

#define BENCH_SH(fn) &(struct bench) { .run_sh = fn }
#define BENCH_TEX(fn) &(struct bench) { .run_tex = fn }

    printf("= Running benchmarks =\n");
    benchmark(vk->gpu, "tex_download ptr", BENCH_TEX(bench_download));
    benchmark(vk->gpu, "tex_upload ptr", BENCH_TEX(bench_upload));
    benchmark(vk->gpu, "bilinear", BENCH_SH(bench_bilinear));
    benchmark(vk->gpu, "bicubic", BENCH_SH(bench_bicubic));
    benchmark(vk->gpu, "deband", BENCH_SH(bench_deband));
    benchmark(vk->gpu, "deband_heavy", BENCH_SH(bench_deband_heavy));

    // Polar sampling
    benchmark(vk->gpu, "polar", BENCH_SH(bench_polar));
    if (vk->gpu->caps & PL_GPU_CAP_COMPUTE)
        benchmark(vk->gpu, "polar_nocompute", BENCH_SH(bench_polar_nocompute));

    // Dithering algorithms
    benchmark(vk->gpu, "dither_blue", BENCH_SH(bench_dither_blue));
    benchmark(vk->gpu, "dither_white", BENCH_SH(bench_dither_white));
    benchmark(vk->gpu, "dither_ordered_fixed", BENCH_SH(bench_dither_ordered_fix));

    // HDR peak detection
    if (vk->gpu->caps & PL_GPU_CAP_COMPUTE)
        benchmark(vk->gpu, "hdr_peakdetect", BENCH_SH(bench_hdr_peak));

    // Misc stuff
    benchmark(vk->gpu, "av1_grain", BENCH_SH(bench_av1_grain));
    benchmark(vk->gpu, "av1_grain_lap", BENCH_SH(bench_av1_grain_lap));

    pl_vulkan_destroy(&vk);
    pl_context_destroy(&ctx);
    return 0;
}
