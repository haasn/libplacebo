#include "tests.h"
#include "time.h"

#define TEX_SIZE 2048
#define CUBE_SIZE 64
#define NUM_FBOS 10
#define BENCH_DUR 1

static const struct ra_tex *create_test_img(const struct ra *ra)
{
    const struct ra_fmt *fmt;
    fmt = ra_find_fmt(ra, RA_FMT_FLOAT, 4, 16, 32, RA_FMT_CAP_LINEAR);
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

    const struct ra_tex *tex = ra_tex_create(ra, &(struct ra_tex_params) {
        .format         = fmt,
        .w              = TEX_SIZE,
        .h              = TEX_SIZE,
        .sampleable     = true,
        .sample_mode    = RA_TEX_SAMPLE_LINEAR,
        .initial_data   = data,
    });

    free(data);
    REQUIRE(tex);
    return tex;
}

struct fbo {
    const struct ra_buf *buf;
    const struct ra_tex *tex;
};

static void create_fbos(const struct ra *ra, struct fbo fbos[NUM_FBOS])
{
    const struct ra_fmt *fmt;
    fmt = ra_find_fmt(ra, RA_FMT_FLOAT, 4, 16, 0, RA_FMT_CAP_RENDERABLE);
    REQUIRE(fmt);

    for (int i = 0; i < NUM_FBOS; i++) {
        fbos[i].tex = ra_tex_create(ra, &(struct ra_tex_params) {
            .format         = fmt,
            .w              = TEX_SIZE,
            .h              = TEX_SIZE,
            .renderable     = true,
            .host_readable  = true,
            .storable       = !!(fmt->caps & RA_FMT_CAP_STORABLE),
        });
        REQUIRE(fbos[i].tex);

        fbos[i].buf = ra_buf_create(ra, &(struct ra_buf_params) {
            .type           = RA_BUF_TEX_TRANSFER,
            .size           = fmt->texel_size,
            .host_readable  = true,
        });
        REQUIRE(fbos[i].buf);
    }
}

typedef void (*bench_fn)(struct pl_shader *sh, struct pl_shader_obj **state,
                         const struct ra_tex *src);

static void run_bench(const struct ra *ra, struct pl_dispatch *dp,
                      struct pl_shader_obj **state, const struct ra_tex *src,
                      struct fbo fbo, bench_fn bench)
{
    // Hard block until the FBO is free
    while (ra_buf_poll(ra, fbo.buf, 1000000)); // 1 ms

    pl_dispatch_reset_frame(dp);
    struct pl_shader *sh = pl_dispatch_begin(dp);
    bench(sh, state, src);
    pl_dispatch_finish(dp, sh, fbo.tex, NULL);

    bool ok = ra_tex_download(ra, &(struct ra_tex_transfer_params) {
        .tex        = fbo.tex,
        .buf        = fbo.buf,
        // Transfer a single pixel:
        .stride_w   = 1,
        .rc         = {
            0, 0, 0,
            1, 1, 1
        },
    });
    REQUIRE(ok);
}

static void benchmark(const struct ra *ra, const char *name, bench_fn bench)
{
    struct pl_dispatch *dp = pl_dispatch_create(ra->ctx, ra);
    struct pl_shader_obj *state = NULL;
    const struct ra_tex *src = create_test_img(ra);
    struct fbo fbos[NUM_FBOS] = {0};
    create_fbos(ra, fbos);

    // Run the benchmark and flush+block once to force shader compilation etc.
    run_bench(ra, dp, &state, src, fbos[0], bench);
    ra_flush(ra);
    while (ra_buf_poll(ra, fbos[0].buf, 1000000000)); // 1 s

    // Perform the actual benchmark
    clock_t start = clock(), stop = {0};
    unsigned long frames = 0;
    int index = 0;

    do {
        frames++;
        run_bench(ra, dp, &state, src, fbos[index++], bench);
        index %= NUM_FBOS;
        stop = clock();
    } while (stop - start < BENCH_DUR * CLOCKS_PER_SEC);

    float secs = (float) (stop - start) / CLOCKS_PER_SEC;
    printf("'%s':\t%4lu frames in %1.6f seconds => %2.6f ms/frame (%5.2f FPS)\n",
          name, frames, secs, 1000 * secs / frames, frames / secs);

    pl_shader_obj_destroy(&state);
    pl_dispatch_destroy(&dp);
    ra_tex_destroy(ra, &src);
    for (int i = 0; i < NUM_FBOS; i++) {
        ra_tex_destroy(ra, &fbos[i].tex);
        ra_buf_destroy(ra, &fbos[i].buf);
    }
}

// List of benchmarks
static void bench_bt2020c(struct pl_shader *sh, struct pl_shader_obj **state,
                          const struct ra_tex *src)
{
    struct pl_color_repr repr = {
        .sys    = PL_COLOR_SYSTEM_BT_2020_C,
        .levels = PL_COLOR_LEVELS_TV,
    };

    pl_shader_sample_direct(sh, &(struct pl_sample_src) { .tex = src });
    pl_shader_decode_color(sh, &repr, NULL);
}

static void bench_deband(struct pl_shader *sh, struct pl_shader_obj **state,
                         const struct ra_tex *src)
{
    pl_shader_deband(sh, src, NULL);
}

static void bench_deband_heavy(struct pl_shader *sh, struct pl_shader_obj **state,
                               const struct ra_tex *src)
{
    pl_shader_deband(sh, src, &(struct pl_deband_params) {
        .iterations = 4,
        .threshold  = 4.0,
        .radius     = 4.0,
        .grain      = 16.0,
    });
}

static void bench_bicubic(struct pl_shader *sh, struct pl_shader_obj **state,
                          const struct ra_tex *src)
{
    pl_shader_sample_bicubic(sh, &(struct pl_sample_src) { .tex = src });
}

static void bench_dither_blue(struct pl_shader *sh, struct pl_shader_obj **state,
                              const struct ra_tex *src)
{
    struct pl_dither_params params = pl_dither_default_params;
    params.method = PL_DITHER_BLUE_NOISE;

    pl_shader_sample_direct(sh, &(struct pl_sample_src) { .tex = src });
    pl_shader_dither(sh, 8, state, &params);
}

static void bench_dither_white(struct pl_shader *sh, struct pl_shader_obj **state,
                               const struct ra_tex *src)
{
    struct pl_dither_params params = pl_dither_default_params;
    params.method = PL_DITHER_WHITE_NOISE;

    pl_shader_sample_direct(sh, &(struct pl_sample_src) { .tex = src });
    pl_shader_dither(sh, 8, state, &params);
}

static void bench_dither_ordered_lut(struct pl_shader *sh,
                                     struct pl_shader_obj **state,
                                     const struct ra_tex *src)
{
    struct pl_dither_params params = pl_dither_default_params;
    params.method = PL_DITHER_ORDERED_LUT;

    pl_shader_sample_direct(sh, &(struct pl_sample_src) { .tex = src });
    pl_shader_dither(sh, 8, state, &params);
}

static void bench_dither_ordered_fix(struct pl_shader *sh,
                                     struct pl_shader_obj **state,
                                     const struct ra_tex *src)
{
    struct pl_dither_params params = pl_dither_default_params;
    params.method = PL_DITHER_ORDERED_FIXED;

    pl_shader_sample_direct(sh, &(struct pl_sample_src) { .tex = src });
    pl_shader_dither(sh, 8, state, &params);
}

static void bench_polar(struct pl_shader *sh, struct pl_shader_obj **state,
                        const struct ra_tex *src)
{
    struct pl_sample_polar_params params = {
        .filter = pl_filter_ewa_lanczos,
        .lut = state,
    };

    pl_shader_sample_polar(sh, &(struct pl_sample_src) { .tex = src }, &params);
}

static void bench_polar_nocompute(struct pl_shader *sh,
                                  struct pl_shader_obj **state,
                                  const struct ra_tex *src)
{
    struct pl_sample_polar_params params = {
        .filter = pl_filter_ewa_lanczos,
        .no_compute = true,
        .lut = state,
    };

    pl_shader_sample_polar(sh, &(struct pl_sample_src) { .tex = src }, &params);
}


static void bench_hdr_hable(struct pl_shader *sh, struct pl_shader_obj **state,
                            const struct ra_tex *src)
{
    struct pl_color_map_params params = {
        .tone_mapping_algo = PL_TONE_MAPPING_HABLE,
    };

    pl_shader_sample_direct(sh, &(struct pl_sample_src) { .tex = src });
    pl_shader_color_map(sh, &params, pl_color_space_hdr10, pl_color_space_monitor,
                        state, false);
}

static void bench_hdr_mobius(struct pl_shader *sh, struct pl_shader_obj **state,
                             const struct ra_tex *src)
{
    struct pl_color_map_params params = {
        .tone_mapping_algo = PL_TONE_MAPPING_MOBIUS,
    };

    pl_shader_sample_direct(sh, &(struct pl_sample_src) { .tex = src });
    pl_shader_color_map(sh, &params, pl_color_space_hdr10, pl_color_space_monitor,
                        state, false);
}

static void bench_hdr_peak(struct pl_shader *sh, struct pl_shader_obj **state,
                            const struct ra_tex *src)
{
    struct pl_color_map_params params = {
        .tone_mapping_algo = PL_TONE_MAPPING_CLIP,
        .peak_detect_frames = 10,
    };

    pl_shader_sample_direct(sh, &(struct pl_sample_src) { .tex = src });
    pl_shader_color_map(sh, &params, pl_color_space_hdr10, pl_color_space_monitor,
                        state, false);
}

static void bench_hdr_desat(struct pl_shader *sh, struct pl_shader_obj **state,
                            const struct ra_tex *src)
{
    struct pl_color_map_params params = {
        .tone_mapping_algo = PL_TONE_MAPPING_CLIP,
        .tone_mapping_desaturate = 1.0,
    };

    pl_shader_sample_direct(sh, &(struct pl_sample_src) { .tex = src });
    pl_shader_color_map(sh, &params, pl_color_space_hdr10, pl_color_space_monitor,
                        state, false);
}

int main()
{
    struct pl_context *ctx;
    ctx = pl_context_create(PL_API_VER, &(struct pl_context_params) {
        .log_cb     = isatty(fileno(stdout)) ? pl_log_color : pl_log_simple,
        .log_level  = PL_LOG_WARN,
    });

    const struct pl_vulkan *vk = pl_vulkan_create(ctx, NULL);
    if (!vk)
        return SKIP;

    printf("= Running benchmarks =\n");
    benchmark(vk->ra, "bt2020c", bench_bt2020c);
    benchmark(vk->ra, "bicubic", bench_bicubic);
    benchmark(vk->ra, "deband", bench_deband);
    benchmark(vk->ra, "deband_heavy", bench_deband_heavy);

    // Dithering algorithms
    benchmark(vk->ra, "dither_blue", bench_dither_blue);
    benchmark(vk->ra, "dither_white", bench_dither_white);
    benchmark(vk->ra, "dither_ordered_lut", bench_dither_ordered_lut);
    benchmark(vk->ra, "dither_ordered_fixed", bench_dither_ordered_fix);

    // Polar sampling
    benchmark(vk->ra, "polar", bench_polar);
    if (vk->ra->caps & RA_CAP_COMPUTE)
        benchmark(vk->ra, "polar_nocompute", bench_polar_nocompute);

    // HDR tone mapping
    benchmark(vk->ra, "hdr_hable", bench_hdr_hable);
    benchmark(vk->ra, "hdr_mobius", bench_hdr_mobius);
    benchmark(vk->ra, "hdr_desaturate", bench_hdr_desat);
    if (vk->ra->caps & RA_CAP_COMPUTE)
        benchmark(vk->ra, "hdr_peakdetect", bench_hdr_peak);

    return 0;
}
