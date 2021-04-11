#include "tests.h"
#include "shaders.h"

static void pl_buffer_tests(const struct pl_gpu *gpu)
{
    const size_t buf_size = 1024;
    if (buf_size > gpu->limits.max_buf_size)
        return;

    uint8_t *test_src = malloc(buf_size * 2);
    uint8_t *test_dst = test_src + buf_size;
    assert(test_src && test_dst);
    memset(test_dst, 0, buf_size);
    for (int i = 0; i < buf_size; i++)
        test_src[i] = (RANDOM * 256);

    const struct pl_buf *buf = NULL, *tbuf = NULL;

    printf("test buffer static creation and readback\n");
    buf = pl_buf_create(gpu, &(struct pl_buf_params) {
        .size = buf_size,
        .host_readable = true,
        .initial_data = test_src,
    });

    REQUIRE(buf);
    REQUIRE(pl_buf_read(gpu, buf, 0, test_dst, buf_size));
    REQUIRE(memcmp(test_src, test_dst, buf_size) == 0);
    pl_buf_destroy(gpu, &buf);

    printf("test buffer empty creation, update and readback\n");
    memset(test_dst, 0, buf_size);
    buf = pl_buf_create(gpu, &(struct pl_buf_params) {
        .size = buf_size,
        .host_writable = true,
        .host_readable = true,
    });

    REQUIRE(buf);
    pl_buf_write(gpu, buf, 0, test_src, buf_size);
    REQUIRE(pl_buf_read(gpu, buf, 0, test_dst, buf_size));
    REQUIRE(memcmp(test_src, test_dst, buf_size) == 0);
    pl_buf_destroy(gpu, &buf);

    printf("test buffer-buffer copy and readback\n");
    memset(test_dst, 0, buf_size);
    buf = pl_buf_create(gpu, &(struct pl_buf_params) {
        .size = buf_size,
        .initial_data = test_src,
    });

    tbuf = pl_buf_create(gpu, &(struct pl_buf_params) {
        .size = buf_size,
        .host_readable = true,
    });

    REQUIRE(buf && tbuf);
    pl_buf_copy(gpu, tbuf, 0, buf, 0, buf_size);
    REQUIRE(pl_buf_read(gpu, tbuf, 0, test_dst, buf_size));
    REQUIRE(memcmp(test_src, test_dst, buf_size) == 0);
    pl_buf_destroy(gpu, &buf);
    pl_buf_destroy(gpu, &tbuf);

    if (gpu->caps & PL_GPU_CAP_MAPPED_BUFFERS) {
        printf("test host mapped buffer readback\n");
        buf = pl_buf_create(gpu, &(struct pl_buf_params) {
            .size = buf_size,
            .host_mapped = true,
            .initial_data = test_src,
        });

        REQUIRE(buf);
        REQUIRE(!pl_buf_poll(gpu, buf, 0));
        REQUIRE(memcmp(test_src, buf->data, buf_size) == 0);
        pl_buf_destroy(gpu, &buf);
    }

    free(test_src);
}

static void test_cb(void *priv)
{
    bool *flag = priv;
    *flag = true;
}

static void pl_test_roundtrip(const struct pl_gpu *gpu, const struct pl_tex *tex[2],
                              uint8_t *src, uint8_t *dst)
{
    if (!tex[0] || !tex[1]) {
        printf("failed creating test textures... skipping this test\n");
        return;
    }

    int texels = tex[0]->params.w;
    texels *= tex[0]->params.h ? tex[0]->params.h : 1;
    texels *= tex[0]->params.d ? tex[0]->params.d : 1;

    const struct pl_fmt *fmt = tex[0]->params.format;
    size_t bytes = texels * fmt->texel_size;
    memset(src, 0, bytes);
    memset(dst, 0, bytes);

    for (size_t i = 0; i < bytes; i++)
        src[i] = (RANDOM * 256);

    struct pl_timer *ul, *dl;
    ul = pl_timer_create(gpu);
    dl = pl_timer_create(gpu);

    bool ran_ul = false, ran_dl = false;

    REQUIRE(pl_tex_upload(gpu, &(struct pl_tex_transfer_params){
        .tex = tex[0],
        .ptr = src,
        .timer = ul,
        .callback = (gpu->caps & PL_GPU_CAP_CALLBACKS) ? test_cb : NULL,
        .priv = &ran_ul,
    }));

    // Test blitting, if possible for this format
    const struct pl_tex *dst_tex = tex[0];
    if (tex[0]->params.blit_src && tex[1]->params.blit_dst) {
        pl_tex_clear(gpu, tex[1], (float[4]){0.0}); // for testing
        pl_tex_blit(gpu, &(struct pl_tex_blit_params) {
            .src = tex[0],
            .dst = tex[1],
        });
        dst_tex = tex[1];
    }

    REQUIRE(pl_tex_download(gpu, &(struct pl_tex_transfer_params){
        .tex = dst_tex,
        .ptr = dst,
        .timer = dl,
        .callback = (gpu->caps & PL_GPU_CAP_CALLBACKS) ? test_cb : NULL,
        .priv = &ran_dl,
    }));

    pl_gpu_finish(gpu);
    if (gpu->caps & PL_GPU_CAP_CALLBACKS)
        REQUIRE(ran_ul && ran_dl);

    if (fmt->emulated && fmt->type == PL_FMT_FLOAT) {
        // TODO: can't memcmp here because bits might be lost due to the
        // emulated 16/32 bit upload paths, figure out a better way to
        // generate data and verify the roundtrip!
    } else {
        REQUIRE(memcmp(src, dst, bytes) == 0);
    }

    // Report timer results
    printf("upload time: %"PRIu64", download time: %"PRIu64"\n",
           pl_timer_query(gpu, ul), pl_timer_query(gpu, dl));

    pl_timer_destroy(gpu, &ul);
    pl_timer_destroy(gpu, &dl);
}

static void pl_texture_tests(const struct pl_gpu *gpu)
{
    const size_t max_size = 16*16*16 * 4 *sizeof(double);
    uint8_t *test_src = malloc(max_size * 2);
    uint8_t *test_dst = test_src + max_size;

    for (int f = 0; f < gpu->num_formats; f++) {
        const struct pl_fmt *fmt = gpu->formats[f];
        if (fmt->opaque || !(fmt->caps & PL_FMT_CAP_HOST_READABLE))
            continue;

        printf("testing texture roundtrip for format %s\n", fmt->name);
        assert(fmt->texel_size <= 4 * sizeof(double));

        struct pl_tex_params ref_params = {
            .format        = fmt,
            .blit_src      = (fmt->caps & PL_FMT_CAP_BLITTABLE),
            .blit_dst      = (fmt->caps & PL_FMT_CAP_BLITTABLE),
            .host_writable = true,
            .host_readable = true,
        };

        const struct pl_tex *tex[2];

        if (gpu->limits.max_tex_1d_dim >= 16) {
            printf("... 1D\n");
            struct pl_tex_params params = ref_params;
            params.w = 16;
            if (!(gpu->caps & PL_GPU_CAP_BLITTABLE_1D_3D))
                params.blit_src = params.blit_dst = false;
            for (int i = 0; i < PL_ARRAY_SIZE(tex); i++)
                tex[i] = pl_tex_create(gpu, &params);
            pl_test_roundtrip(gpu, tex, test_src, test_dst);
            for (int i = 0; i < PL_ARRAY_SIZE(tex); i++)
                pl_tex_destroy(gpu, &tex[i]);
        }

        if (gpu->limits.max_tex_2d_dim >= 16) {
            printf("... 2D\n");
            struct pl_tex_params params = ref_params;
            params.w = params.h = 16;
            for (int i = 0; i < PL_ARRAY_SIZE(tex); i++)
                tex[i] = pl_tex_create(gpu, &params);
            pl_test_roundtrip(gpu, tex, test_src, test_dst);
            for (int i = 0; i < PL_ARRAY_SIZE(tex); i++)
                pl_tex_destroy(gpu, &tex[i]);
        }

        if (gpu->limits.max_tex_3d_dim >= 16) {
            printf("... 3D\n");
            struct pl_tex_params params = ref_params;
            params.w = params.h = params.d = 16;
            if (!(gpu->caps & PL_GPU_CAP_BLITTABLE_1D_3D))
                params.blit_src = params.blit_dst = false;
            for (int i = 0; i < PL_ARRAY_SIZE(tex); i++)
                tex[i] = pl_tex_create(gpu, &params);
            pl_test_roundtrip(gpu, tex, test_src, test_dst);
            for (int i = 0; i < PL_ARRAY_SIZE(tex); i++)
                pl_tex_destroy(gpu, &tex[i]);
        }
    }

    free(test_src);
}

static void pl_shader_tests(const struct pl_gpu *gpu)
{
    if (gpu->glsl.version < 410)
        return;

    const char *vert_shader =
        "#version 410                               \n"
        "layout(location=0) in vec2 vertex_pos;     \n"
        "layout(location=1) in vec3 vertex_color;   \n"
        "layout(location=0) out vec3 frag_color;    \n"
        "void main() {                              \n"
        "    gl_Position = vec4(vertex_pos, 0, 1);  \n"
        "    frag_color = vertex_color;             \n"
        "}";

    const char *frag_shader =
        "#version 410                               \n"
        "layout(location=0) in vec3 frag_color;     \n"
        "layout(location=0) out vec4 out_color;     \n"
        "void main() {                              \n"
        "    out_color = vec4(frag_color, 1.0);     \n"
        "}";

    const struct pl_fmt *fbo_fmt;
    enum pl_fmt_caps caps = PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_BLITTABLE |
                            PL_FMT_CAP_LINEAR;

    fbo_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 16, 32, caps);
    if (!fbo_fmt)
        return;

#define FBO_W 16
#define FBO_H 16

    const struct pl_tex *fbo;
    fbo = pl_tex_create(gpu, &(struct pl_tex_params) {
        .format         = fbo_fmt,
        .w              = FBO_W,
        .h              = FBO_H,
        .renderable     = true,
        .storable       = !!(fbo_fmt->caps & PL_FMT_CAP_STORABLE),
        .host_readable  = true,
        .blit_dst       = true,
    });
    REQUIRE(fbo);

    pl_tex_clear(gpu, fbo, (float[4]){0});

    const struct pl_fmt *vert_fmt;
    vert_fmt = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 3);
    REQUIRE(vert_fmt);

    static const struct vertex { float pos[2]; float color[3]; } vertices[] = {
        {{-1.0, -1.0}, {0, 0, 0}},
        {{ 1.0, -1.0}, {1, 0, 0}},
        {{-1.0,  1.0}, {0, 1, 0}},
        {{ 1.0,  1.0}, {1, 1, 0}},
    };

    const struct pl_pass *pass;
    pass = pl_pass_create(gpu, &(struct pl_pass_params) {
        .type           = PL_PASS_RASTER,
        .target_dummy   = *fbo,
        .vertex_shader  = vert_shader,
        .glsl_shader    = frag_shader,

        .vertex_type    = PL_PRIM_TRIANGLE_STRIP,
        .vertex_stride  = sizeof(struct vertex),
        .num_vertex_attribs = 2,
        .vertex_attribs = (struct pl_vertex_attrib[]) {{
            .name     = "vertex_pos",
            .fmt      = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 2),
            .location = 0,
            .offset   = offsetof(struct vertex, pos),
        }, {
            .name     = "vertex_color",
            .fmt      = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 3),
            .location = 1,
            .offset   = offsetof(struct vertex, color),
        }},
    });
    REQUIRE(pass);
    REQUIRE(pass->params.cached_program_len);

    struct pl_timer *timer = pl_timer_create(gpu);
    pl_pass_run(gpu, &(struct pl_pass_run_params) {
        .pass           = pass,
        .target         = fbo,
        .vertex_count   = PL_ARRAY_SIZE(vertices),
        .vertex_data    = vertices,
        .timer          = timer,
    });

    // Wait until this pass is complete and report the timer result
    pl_gpu_finish(gpu);
    printf("timer query result: %"PRIu64"\n", pl_timer_query(gpu, timer));
    pl_timer_destroy(gpu, &timer);

    static float data[FBO_H * FBO_W * 4] = {0};

    // Test against the known pattern of `src`, only useful for roundtrip tests
#define TEST_FBO_PATTERN(eps, fmt, ...)                                     \
    do {                                                                    \
        printf("testing pattern of " fmt "\n", __VA_ARGS__);                \
        REQUIRE(pl_tex_download(gpu, &(struct pl_tex_transfer_params) {     \
            .tex = fbo,                                                     \
            .ptr = data,                                                    \
        }));                                                                \
                                                                            \
        for (int y = 0; y < FBO_H; y++) {                                   \
            for (int x = 0; x < FBO_W; x++) {                               \
                float *color = &data[(y * FBO_W + x) * 4];                  \
                REQUIRE(feq(color[0], (x + 0.5) / FBO_W, eps));             \
                REQUIRE(feq(color[1], (y + 0.5) / FBO_H, eps));             \
                REQUIRE(feq(color[2], 0.0, eps));                           \
                REQUIRE(feq(color[3], 1.0, eps));                           \
            }                                                               \
        }                                                                   \
    } while (0)

    TEST_FBO_PATTERN(1e-6, "%s", "initial rendering");

    if (sizeof(vertices) <= gpu->limits.max_vbo_size) {
        // Test the use of an explicit vertex buffer
        const struct pl_buf *vert = pl_buf_create(gpu, &(struct pl_buf_params) {
            .size = sizeof(vertices),
            .initial_data = vertices,
            .drawable = true,
        });

        REQUIRE(vert);
        pl_pass_run(gpu, &(struct pl_pass_run_params) {
            .pass           = pass,
            .target         = fbo,
            .vertex_count   = sizeof(vertices) / sizeof(struct vertex),
            .vertex_buf     = vert,
            .buf_offset     = 0,
        });

        pl_buf_destroy(gpu, &vert);
        TEST_FBO_PATTERN(1e-6, "%s", "using vertex buffer");
    }

    // Test the use of index buffers
    static const uint16_t indices[] = { 3, 2, 1, 0 };
    pl_pass_run(gpu, &(struct pl_pass_run_params) {
        .pass           = pass,
        .target         = fbo,
        .vertex_count   = PL_ARRAY_SIZE(indices),
        .vertex_data    = vertices,
        .index_data     = indices,
    });

    pl_pass_destroy(gpu, &pass);
    TEST_FBO_PATTERN(1e-6, "%s", "using indexed rendering");

    // Test the use of pl_dispatch
    struct pl_dispatch *dp = pl_dispatch_create(gpu->ctx, gpu);
    struct pl_shader *sh = pl_dispatch_begin(dp);
    REQUIRE(pl_shader_custom(sh, &(struct pl_custom_shader) {
        .body       = "color = vec4(col, 1.0);",
        .input      = PL_SHADER_SIG_NONE,
        .output     = PL_SHADER_SIG_COLOR,
    }));

    REQUIRE(pl_dispatch_vertex(dp, &(struct pl_dispatch_vertex_params) {
        .shader         = &sh,
        .target         = fbo,
        .vertex_stride  = sizeof(struct vertex),
        .vertex_position_idx = 0,
        .num_vertex_attribs = 2,
        .vertex_attribs = (struct pl_vertex_attrib[]) {{
            .name   = "pos",
            .fmt    = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 2),
            .offset = offsetof(struct vertex, pos),
        }, {
            .name   = "col",
            .fmt    = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 3),
            .offset = offsetof(struct vertex, color),
        }},

        .vertex_type    = PL_PRIM_TRIANGLE_STRIP,
        .vertex_coords  = PL_COORDS_NORMALIZED,
        .vertex_count   = PL_ARRAY_SIZE(vertices),
        .vertex_data    = vertices,
    }));

    TEST_FBO_PATTERN(1e-6, "%s", "using custom vertices");

    const struct pl_tex *src;
    src = pl_tex_create(gpu, &(struct pl_tex_params) {
        .format         = fbo_fmt,
        .w              = FBO_W,
        .h              = FBO_H,
        .sampleable     = true,
        .initial_data   = data,
    });

    // Test encoding/decoding of all gamma functions, color spaces, etc.
    for (enum pl_color_transfer trc = 0; trc < PL_COLOR_TRC_COUNT; trc++) {
        sh = pl_dispatch_begin(dp);
        pl_shader_sample_nearest(sh, &(struct pl_sample_src) { .tex = src });
        pl_shader_delinearize(sh, trc);
        pl_shader_linearize(sh, trc);
        REQUIRE(pl_dispatch_finish(dp, &(struct pl_dispatch_params) {
            .shader = &sh,
            .target = fbo,
        }));

        float epsilon = pl_color_transfer_is_hdr(trc) ? 1e-4 : 1e-6;
        TEST_FBO_PATTERN(epsilon, "transfer function %d", (int) trc);
    }

    for (enum pl_color_system sys = 0; sys < PL_COLOR_SYSTEM_COUNT; sys++) {
        sh = pl_dispatch_begin(dp);
        pl_shader_sample_nearest(sh, &(struct pl_sample_src) { .tex = src });
        pl_shader_encode_color(sh, &(struct pl_color_repr) { .sys = sys });
        pl_shader_decode_color(sh, &(struct pl_color_repr) { .sys = sys }, NULL);
        REQUIRE(pl_dispatch_finish(dp, &(struct pl_dispatch_params) {
            .shader = &sh,
            .target = fbo,
        }));

        float epsilon;
        switch (sys) {
        case PL_COLOR_SYSTEM_BT_2020_C:
            epsilon = 1e-5;
            break;

        case PL_COLOR_SYSTEM_BT_2100_PQ:
        case PL_COLOR_SYSTEM_BT_2100_HLG:
            // These seem to be horrifically noisy and prone to breaking on
            // edge cases for some reason
            // TODO: figure out why!
            continue;
            break;

        default: epsilon = 1e-6; break;
        }

        TEST_FBO_PATTERN(epsilon, "color system %d", (int) sys);
    }

    for (enum pl_color_light light = 0; light < PL_COLOR_LIGHT_COUNT; light++) {
        sh = pl_dispatch_begin(dp);
        struct pl_color_space src_space = { .light = light };
        struct pl_color_space dst_space = { 0 };
        pl_shader_sample_nearest(sh, &(struct pl_sample_src) { .tex = src });
        pl_shader_color_map(sh, NULL, src_space, dst_space, NULL, false);
        pl_shader_color_map(sh, NULL, dst_space, src_space, NULL, false);
        REQUIRE(pl_dispatch_finish(dp, &(struct pl_dispatch_params) {
            .shader = &sh,
            .target = fbo,
        }));

        TEST_FBO_PATTERN(1e-6, "light %d", (int) light);
    }

    // Repeat this a few times to test the caching
    for (int i = 0; i < 10; i++) {
        if (i == 5) {
            printf("Recreating pl_dispatch to test the caching\n");
            size_t size = pl_dispatch_save(dp, NULL);
            REQUIRE(size > 0);
            uint8_t *cache = malloc(size);
            REQUIRE(cache);
            REQUIRE(pl_dispatch_save(dp, cache) == size);

            pl_dispatch_destroy(&dp);
            dp = pl_dispatch_create(gpu->ctx, gpu);
            pl_dispatch_load(dp, cache);

#ifndef MSAN
            // Test to make sure the pass regenerates the same cache, but skip
            // this on MSAN because it doesn't like it when we read from
            // program cache data generated by the non-instrumented GPU driver
            uint64_t hash = pl_str_hash((pl_str) { cache, size });
            REQUIRE(pl_dispatch_save(dp, NULL) == size);
            REQUIRE(pl_dispatch_save(dp, cache) == size);
            REQUIRE(pl_str_hash((pl_str) { cache, size }) == hash);
#endif
            free(cache);
        }

        sh = pl_dispatch_begin(dp);

        // For testing, force the use of CS if possible
        if (gpu->caps & PL_GPU_CAP_COMPUTE) {
            sh->is_compute = true;
            sh->res.compute_group_size[0] = 8;
            sh->res.compute_group_size[1] = 8;
        }

        pl_shader_deband(sh,
            &(struct pl_sample_src) {
                .tex            = src,
            },
            &(struct pl_deband_params) {
                .iterations     = 0,
                .grain          = 0.0,
        });

        REQUIRE(pl_dispatch_finish(dp, &(struct pl_dispatch_params) {
            .shader = &sh,
            .target = fbo,
        }));
        TEST_FBO_PATTERN(1e-6, "deband iter %d", i);
    }

    // Test peak detection and readback if possible
    sh = pl_dispatch_begin(dp);
    pl_shader_sample_nearest(sh, &(struct pl_sample_src) { .tex = src });

    struct pl_shader_obj *peak_state = NULL;
    if (pl_shader_detect_peak(sh, pl_color_space_monitor, &peak_state, NULL)) {
        REQUIRE(pl_dispatch_compute(dp, &(struct pl_dispatch_compute_params) {
            .shader = &sh,
            .width = fbo->params.w,
            .height = fbo->params.h,
        }));

        float peak, avg;
        REQUIRE(pl_get_detected_peak(peak_state, &peak, &avg));
        printf("detected peak: %f, average: %f\n", peak, avg);

        float real_peak = 0, real_avg = 0;
        for (int y = 0; y < FBO_H; y++) {
            for (int x = 0; x < FBO_W; x++) {
                float *color = &data[(y * FBO_W + x) * 4];
                float smax = powf(PL_MAX(color[0], color[1]), 2.2);
                float slog = logf(PL_MAX(smax, 0.001));
                real_peak = PL_MAX(smax, real_peak);
                real_avg += slog;
            }
        }

        real_peak *= 1.0 + pl_peak_detect_default_params.overshoot_margin;
        real_avg = expf(real_avg / (FBO_W * FBO_H));
        printf("real peak: %f, real average: %f\n", real_peak, real_avg);
        REQUIRE(feq(peak, real_peak, 1e-4));
        REQUIRE(feq(avg, real_avg, 1e-3));
    }

    pl_dispatch_abort(dp, &sh);
    pl_shader_obj_destroy(&peak_state);

#ifdef PL_HAVE_LCMS
    // Test the use of ICC profiles if available
    sh = pl_dispatch_begin(dp);
    pl_shader_sample_nearest(sh, &(struct pl_sample_src) { .tex = src });

    struct pl_shader_obj *icc = NULL;
    struct pl_icc_color_space src_color = { .color = pl_color_space_bt709 };
    struct pl_icc_color_space dst_color = { .color = pl_color_space_srgb };
    struct pl_icc_result out;

    if (pl_icc_update(sh, &src_color, &dst_color, &icc, &out, NULL)) {
        pl_icc_apply(sh, &icc);
        REQUIRE(pl_dispatch_finish(dp, &(struct pl_dispatch_params) {
            .shader = &sh,
            .target = fbo,
        }));
    }

    pl_dispatch_abort(dp, &sh);
    pl_shader_obj_destroy(&icc);
#endif

    // Test AV1 grain synthesis
    struct pl_shader_obj *grain = NULL;
    for (int i = 0; i < 2; i++) {
        struct pl_av1_grain_params grain_params = {
            .data = av1_grain_data,
            .tex = src,
            .components = 3,
            .component_mapping = { 0, 1, 2 },
            .repr = &(struct pl_color_repr) {
                .sys = PL_COLOR_SYSTEM_BT_709,
                .levels = PL_COLOR_LEVELS_LIMITED,
                .bits = { .color_depth = 10, .sample_depth = 10 },
            },
        };
        grain_params.data.grain_seed = rand();
        grain_params.data.overlap = !!i;

        sh = pl_dispatch_begin(dp);
        pl_shader_av1_grain(sh, &grain, &grain_params);
        REQUIRE(pl_dispatch_finish(dp, &(struct pl_dispatch_params) {
            .shader = &sh,
            .target = fbo,
        }));
    }
    pl_shader_obj_destroy(&grain);

    // Test custom shaders
    struct pl_custom_shader custom = {
        .header =
            "vec3 invert(vec3 color)            \n"
            "{                                  \n"
            "    return vec3(1.0) - color;      \n"
            "}                                  \n",

        .body =
            "color = vec4(gl_FragCoord.xy, 0.0, 1.0);   \n"
            "color.rgb = invert(color.rgb) + offset;    \n",

        .input = PL_SHADER_SIG_NONE,
        .output = PL_SHADER_SIG_COLOR,

        .num_variables = 1,
        .variables = &(struct pl_shader_var) {
            .var = pl_var_float("offset"),
            .data = &(float) { 0.1 },
        },
    };

    sh = pl_dispatch_begin(dp);
    REQUIRE(pl_shader_custom(sh, &custom));
    REQUIRE(pl_dispatch_finish(dp, &(struct pl_dispatch_params) {
        .shader = &sh,
        .target = fbo,
    }));

    pl_dispatch_destroy(&dp);
    pl_tex_destroy(gpu, &src);
    pl_tex_destroy(gpu, &fbo);
}

static void pl_scaler_tests(const struct pl_gpu *gpu)
{
    const struct pl_fmt *src_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 16, 32,
                                               PL_FMT_CAP_LINEAR);

    const struct pl_fmt *fbo_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 16, 32,
                                               PL_FMT_CAP_RENDERABLE);
    if (!src_fmt || !fbo_fmt)
        return;

    float *fbo_data = NULL;
    struct pl_shader_obj *lut = NULL;

    static float data_5x5[5][5] = {
        { 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0 },
        { 0, 0, 1, 0, 0 },
        { 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0 },
    };

    const struct pl_tex *dot5x5 = pl_tex_create(gpu, &(struct pl_tex_params) {
        .w              = 5,
        .h              = 5,
        .format         = src_fmt,
        .sampleable     = true,
        .initial_data   = &data_5x5[0][0],
    });

    struct pl_tex_params fbo_params = {
        .w              = 100,
        .h              = 100,
        .format         = fbo_fmt,
        .renderable     = true,
        .storable       = !!(fbo_fmt->caps & PL_FMT_CAP_STORABLE),
        .host_readable  = true,
    };

    const struct pl_tex *fbo = pl_tex_create(gpu, &fbo_params);
    if (!fbo) {
        printf("Failed creating readable FBO... falling back to non-readable\n");
        fbo_params.host_readable = false;
        fbo = pl_tex_create(gpu, &fbo_params);
    }

    struct pl_dispatch *dp = pl_dispatch_create(gpu->ctx, gpu);
    if (!dot5x5 || !fbo || !dp)
        goto error;

    struct pl_shader *sh = pl_dispatch_begin(dp);
    REQUIRE(pl_shader_sample_polar(sh,
        &(struct pl_sample_src) {
            .tex        = dot5x5,
            .new_w      = fbo->params.w,
            .new_h      = fbo->params.h,
        },
        &(struct pl_sample_filter_params) {
            .filter     = pl_filter_ewa_lanczos,
            .lut        = &lut,
            .no_compute = !fbo->params.storable,
        }
    ));
    REQUIRE(pl_dispatch_finish(dp, &(struct pl_dispatch_params) {
        .shader = &sh,
        .target = fbo,
    }));

    if (fbo->params.host_readable) {
        fbo_data = malloc(fbo->params.w * fbo->params.h * sizeof(float));
        REQUIRE(pl_tex_download(gpu, &(struct pl_tex_transfer_params) {
            .tex            = fbo,
            .ptr            = fbo_data,
        }));

        int max = 255;
        printf("P2\n%d %d\n%d\n", fbo->params.w, fbo->params.h, max);
        for (int y = 0; y < fbo->params.h; y++) {
            for (int x = 0; x < fbo->params.w; x++) {
                float v = fbo_data[y * fbo->params.h + x];
                printf("%d ", (int) round(fmin(fmax(v, 0.0), 1.0) * max));
            }
            printf("\n");
        }
    }

error:
    free(fbo_data);
    pl_shader_obj_destroy(&lut);
    pl_dispatch_destroy(&dp);
    pl_tex_destroy(gpu, &dot5x5);
    pl_tex_destroy(gpu, &fbo);
}

static const char *user_shader_tests[] = {

    // Test hooking, saving and loading
    "// Example of a comment at the beginning                               \n"
    "                                                                       \n"
    "//!HOOK NATIVE                                                         \n"
    "//!DESC upscale image                                                  \n"
    "//!BIND HOOKED                                                         \n"
    "//!WIDTH HOOKED.w 10 *                                                 \n"
    "//!HEIGHT HOOKED.h 10 *                                                \n"
    "//!SAVE NATIVEBIG                                                      \n"
    "//!WHEN NATIVE.w 500 <                                                 \n"
    "                                                                       \n"
    "vec4 hook()                                                            \n"
    "{                                                                      \n"
    "    return HOOKED_texOff(0);                                           \n"
    "}                                                                      \n"
    "                                                                       \n"
    "//!HOOK MAIN                                                           \n"
    "//!DESC downscale bigger image                                         \n"
    "//!WHEN NATIVE.w 500 <                                                 \n"
    "//!BIND NATIVEBIG                                                      \n"
    "                                                                       \n"
    "vec4 hook()                                                            \n"
    "{                                                                      \n"
    "    return NATIVEBIG_texOff(0);                                        \n"
    "}                                                                      \n",

    // Test use of textures
    "//!HOOK MAIN                                                           \n"
    "//!DESC turn everything into colorful pixels                           \n"
    "//!BIND HOOKED                                                         \n"
    "//!BIND DISCO                                                          \n"
    "//!COMPONENTS 3                                                        \n"
    "                                                                       \n"
    "vec4 hook()                                                            \n"
    "{                                                                      \n"
    "    return vec4(DISCO_tex(HOOKED_pos * 10.0).rgb, 1);                  \n"
    "}                                                                      \n"
    "                                                                       \n"
    "//!TEXTURE DISCO                                                       \n"
    "//!SIZE 3 3                                                            \n"
    "//!FORMAT rgba32f                                                      \n"
    "//!FILTER NEAREST                                                      \n"
    "//!BORDER REPEAT                                                       \n"
    "0000803f000000000000000000000000000000000000803f0000000000000000000000000"
    "00000000000803f00000000000000000000803f0000803f000000000000803f0000000000"
    "00803f000000000000803f0000803f00000000000000009a99993e9a99993e9a99993e000"
    "000009a99193F9A99193f9a99193f000000000000803f0000803f0000803f00000000  \n",

    // Test use of storage/buffer resources
    "//!HOOK MAIN                                                           \n"
    "//!DESC attach some storage objects                                    \n"
    "//!BIND tex_storage                                                    \n"
    "//!BIND buf_uniform                                                    \n"
    "//!BIND buf_storage                                                    \n"
    "//!COMPONENTS 4                                                        \n"
    "                                                                       \n"
    "vec4 hook()                                                            \n"
    "{                                                                      \n"
    "    return vec4(foo, bar, bat);                                        \n"
    "}                                                                      \n"
    "                                                                       \n"
    "//!TEXTURE tex_storage                                                 \n"
    "//!SIZE 100 100                                                        \n"
    "//!FORMAT r32f                                                         \n"
    "//!STORAGE                                                             \n"
    "                                                                       \n"
    "//!BUFFER buf_uniform                                                  \n"
    "//!VAR float foo                                                       \n"
    "//!VAR float bar                                                       \n"
    "0000000000000000                                                       \n"
    "                                                                       \n"
    "//!BUFFER buf_storage                                                  \n"
    "//!VAR vec2 bat                                                        \n"
    "//!VAR int big[32];                                                    \n"
    "//!STORAGE                                                             \n"

};

static const char *test_luts[] = {

    "TITLE \"1D identity\"  \n"
    "LUT_1D_SIZE 2          \n"
    "0.0 0.0 0.0            \n"
    "1.0 1.0 1.0            \n",

    "TITLE \"3D identity\"  \n"
    "LUT_3D_SIZE 2          \n"
    "0.0 0.0 0.0            \n"
    "1.0 0.0 0.0            \n"
    "0.0 1.0 0.0            \n"
    "1.0 1.0 0.0            \n"
    "0.0 0.0 1.0            \n"
    "1.0 0.0 1.0            \n"
    "0.0 1.0 1.0            \n"
    "1.0 1.0 1.0            \n"

};

static bool frame_passthrough(const struct pl_gpu *gpu, const struct pl_tex **tex,
                              const struct pl_source_frame *src, struct pl_frame *out_frame)
{
    const struct pl_frame *frame = src->frame_data;
    *out_frame = *frame;
    return true;
}

static enum pl_queue_status get_frame_ptr(struct pl_source_frame *out_frame,
                                          const struct pl_queue_params *qparams)
{
    const struct pl_source_frame **pframe = qparams->priv;
    if (!(*pframe)->frame_data)
        return PL_QUEUE_EOF;

    *out_frame = *(*pframe)++;
    return PL_QUEUE_OK;
}

static void pl_render_tests(const struct pl_gpu *gpu)
{
    const struct pl_tex *img5x5_tex = NULL, *fbo = NULL;
    struct pl_renderer *rr = NULL;

    float *fbo_data = NULL;
    static float data_5x5[5][5] = {
        { 0.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0 },
        { 1.0, 0.0, 0.5, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 1.0, 0.0 },
        { 0.0, 0.3, 0.0, 0.0, 0.0 },
    };

    const int width = 5, height = 5;
    struct pl_plane img5x5 = {0};
    struct pl_plane_data img5x5_data = {
        .type = PL_FMT_FLOAT,
        .width = width,
        .height = height,
        .component_size = { 8 * sizeof(float) },
        .component_map  = { 0 },
        .pixel_stride = sizeof(float),
        .pixels = &data_5x5,
    };

    if (!pl_recreate_plane(gpu, NULL, &fbo, &img5x5_data))
        return;

    if (!pl_upload_plane(gpu, &img5x5, &img5x5_tex, &img5x5_data))
        goto error;

    rr = pl_renderer_create(gpu->ctx, gpu);
    pl_tex_clear(gpu, fbo, (float[4]){0});

    struct pl_frame image = {
        .num_planes     = 1,
        .planes         = { img5x5 },
        .repr = {
            .sys        = PL_COLOR_SYSTEM_BT_709,
            .levels     = PL_COLOR_LEVELS_FULL,
        },
        .color          = pl_color_space_bt709,
        .crop           = {-1.0, 0.0, width - 1.0, height},
    };

    struct pl_frame target = {
        .num_planes     = 1,
        .planes         = {{
            .texture            = fbo,
            .components         = 3,
            .component_mapping  = {0, 1, 2},
        }},
        .crop           = {2, 2, fbo->params.w - 2, fbo->params.h - 2},
        .repr = {
            .sys        = PL_COLOR_SYSTEM_RGB,
            .levels     = PL_COLOR_LEVELS_FULL,
        },
        .color          = pl_color_space_srgb,
    };

    REQUIRE(pl_render_image(rr, &image, &target, NULL));

    fbo_data = malloc(fbo->params.w * fbo->params.h * sizeof(float[4]));
    REQUIRE(pl_tex_download(gpu, &(struct pl_tex_transfer_params) {
        .tex            = fbo,
        .ptr            = fbo_data,
    }));

    // TODO: embed a reference texture and ensure it matches

    // Test a bunch of different params
#define TEST(SNAME, STYPE, DEFAULT, FIELD, LIMIT)                       \
    do {                                                                \
        for (int i = 0; i <= LIMIT; i++) {                              \
            struct pl_render_params params = pl_render_default_params;  \
            params.force_dither = true;                                 \
            struct STYPE tmp = DEFAULT;                                 \
            tmp.FIELD = i;                                              \
            params.SNAME = &tmp;                                        \
            for (int p = 0; p < 5; p++) {                               \
                REQUIRE(pl_render_image(rr, &image, &target, &params)); \
                pl_gpu_flush(gpu);                                      \
            }                                                           \
        }                                                               \
    } while (0)

#define TEST_PARAMS(NAME, FIELD, LIMIT) \
    TEST(NAME##_params, pl_##NAME##_params, pl_##NAME##_default_params, FIELD, LIMIT)

    for (int i = 0; i < pl_num_filter_presets; i++) {
        struct pl_render_params params = pl_render_default_params;
        params.upscaler = pl_filter_presets[i].filter;
        REQUIRE(pl_render_image(rr, &image, &target, &params));
        pl_gpu_flush(gpu);
    }

    TEST_PARAMS(deband, iterations, 3);
    TEST_PARAMS(sigmoid, center, 1);
    TEST_PARAMS(color_map, intent, PL_INTENT_ABSOLUTE_COLORIMETRIC);
    TEST_PARAMS(color_map, gamut_warning, 1);
    TEST_PARAMS(dither, method, PL_DITHER_WHITE_NOISE);
    TEST_PARAMS(dither, temporal, true);
    TEST(cone_params, pl_cone_params, pl_vision_deuteranomaly, strength, 0);

    // Test HDR stuff
    image.color.sig_scale = 10.0;
    target.color.sig_scale = 2.0;
    TEST_PARAMS(color_map, tone_mapping_algo, PL_TONE_MAPPING_BT_2390);
    TEST_PARAMS(color_map, desaturation_strength, 1);
    image.color.sig_scale = target.color.sig_scale = 0.0;

    // Test some misc stuff
    struct pl_render_params params = pl_render_default_params;
    params.color_adjustment = &(struct pl_color_adjustment) {
        .brightness = 0.1,
        .contrast = 0.9,
        .saturation = 1.5,
        .gamma = 0.8,
        .temperature = 0.3,
    };
    REQUIRE(pl_render_image(rr, &image, &target, &params));
    params = pl_render_default_params;

    params.force_icc_lut = true;
    REQUIRE(pl_render_image(rr, &image, &target, &params));
    params = pl_render_default_params;

    image.av1_grain = av1_grain_data;
    REQUIRE(pl_render_image(rr, &image, &target, &params));
    image.av1_grain = (struct pl_av1_grain_data) {0};

    // Test mpv-style custom shaders
    for (int i = 0; i < PL_ARRAY_SIZE(user_shader_tests); i++) {
        printf("testing user shader:\n\n%s\n", user_shader_tests[i]);
        const struct pl_hook *hook;
        hook = pl_mpv_user_shader_parse(gpu, user_shader_tests[i],
                                        strlen(user_shader_tests[i]));

        if (gpu->caps & PL_GPU_CAP_COMPUTE) {
            REQUIRE(hook);
        } else {
            // Not all shaders compile without compute shader support
            if (!hook)
                continue;
        }

        params.hooks = &hook;
        params.num_hooks = 1;
        REQUIRE(pl_render_image(rr, &image, &target, &params));

        pl_mpv_user_shader_destroy(&hook);
    }
    params = pl_render_default_params;

    // Test custom LUTs
    for (int i = 0; i < PL_ARRAY_SIZE(test_luts); i++) {
        printf("testing custom lut %d\n", i);
        struct pl_custom_lut *lut;
        lut = pl_lut_parse_cube(gpu->ctx, test_luts[i], strlen(test_luts[i]));
        REQUIRE(lut);

        // Test all three at the same time to reduce the number of tests
        image.lut = target.lut = params.lut = lut;

        for (enum pl_lut_type t = PL_LUT_UNKNOWN; t <= PL_LUT_CONVERSION; t++) {
            printf("testing LUT method %d\n", t);
            image.lut_type = target.lut_type = params.lut_type = t;
            REQUIRE(pl_render_image(rr, &image, &target, &params));
        }

        image.lut = target.lut = params.lut = NULL;
        pl_lut_free(&lut);
    }

    // Test overlays
    image.num_overlays = 1;
    image.overlays = &(struct pl_overlay) {
        .tex = img5x5.texture,
        .mode = PL_OVERLAY_NORMAL,
        .num_parts = 2,
        .parts = (struct pl_overlay_part[]) {{
            .src = {0, 0, 2, 2},
            .dst = {30, 100, 40, 200},
        }, {
            .src = {2, 2, 5, 5},
            .dst = {1000, -1, 3, 5},
        }},
    };
    REQUIRE(pl_render_image(rr, &image, &target, &params));
    params.disable_fbos = true;
    REQUIRE(pl_render_image(rr, &image, &target, &params));
    image.num_overlays = 0;
    params = pl_render_default_params;

    target.num_overlays = 1;
    target.overlays = &(struct pl_overlay) {
        .tex = img5x5.texture,
        .mode = PL_OVERLAY_MONOCHROME,
        .num_parts = 1,
        .parts = &(struct pl_overlay_part) {
            .src = {5, 5, 15, 15},
            .dst = {5, 5, 15, 15},
            .color = {1.0, 0.5, 0.0},
        },
    };
    REQUIRE(pl_render_image(rr, &image, &target, &params));
    REQUIRE(pl_render_image(rr, NULL, &target, &params));
    target.num_overlays = 0;

    // Attempt frame mixing, using the mixer queue helper
    printf("testing frame mixing \n");
    struct pl_render_params mix_params = {
        .frame_mixer = &pl_filter_mitchell_clamp,
    };

    struct pl_queue_params qparams = {
        .radius = pl_frame_mix_radius(&mix_params),
        .vsync_duration = 1.0 / 60.0,
        .frame_duration = 1.0 / 24.0,
    };

#define NUM_MIX_FRAMES 20
    struct pl_source_frame srcframes[NUM_MIX_FRAMES+1];
    srcframes[NUM_MIX_FRAMES] = (struct pl_source_frame) {0};
    for (int i = 0; i < NUM_MIX_FRAMES; i++) {
        srcframes[i] = (struct pl_source_frame) {
            .pts = i * qparams.frame_duration,
            .map = frame_passthrough,
            .frame_data = &image,
        };
    }

    struct pl_queue *queue = pl_queue_create(gpu);
    enum pl_queue_status ret;

    // Test pre-pushing all frames, with delayed EOF.
    for (int i = 0; i < NUM_MIX_FRAMES; i++) {
        if (!pl_queue_push_block(queue, 1, &srcframes[i])) // mini-sleep
            pl_queue_push(queue, &srcframes[i]); // push it anyway, for testing
    }

    struct pl_frame_mix mix;
    while ((ret = pl_queue_update(queue, &mix, &qparams)) != PL_QUEUE_EOF) {
        if (ret == PL_QUEUE_MORE) {
            REQUIRE(qparams.pts > 0.0);
            pl_queue_push(queue, NULL); // push delayed EOF
            continue;
        }

        REQUIRE(ret == PL_QUEUE_OK);
        REQUIRE(pl_render_image_mix(rr, &mix, &target, &mix_params));

        // Simulate advancing vsync
        qparams.pts += qparams.vsync_duration;
    }

    // Test dynamically pulling all frames, with oversample mixer
    const struct pl_source_frame *frame_ptr = &srcframes[0];
    mix_params.frame_mixer = &pl_oversample_frame_mixer;

    qparams = (struct pl_queue_params) {
        .radius = pl_frame_mix_radius(&mix_params),
        .vsync_duration = qparams.vsync_duration,
        .frame_duration = qparams.frame_duration,
        .get_frame = get_frame_ptr,
        .priv = &frame_ptr,
    };

    pl_queue_reset(queue);
    while ((ret = pl_queue_update(queue, &mix, &qparams)) != PL_QUEUE_EOF) {
        REQUIRE(ret == PL_QUEUE_OK);
        REQUIRE(mix.num_frames <= 2);
        REQUIRE(pl_render_image_mix(rr, &mix, &target, &mix_params));
        qparams.pts += qparams.vsync_duration;
    }

    // Test large PTS jump
    pl_queue_reset(queue);
    REQUIRE(pl_queue_update(queue, &mix, &qparams) == PL_QUEUE_EOF);

    pl_queue_destroy(&queue);

error:
    free(fbo_data);
    pl_renderer_destroy(&rr);
    pl_tex_destroy(gpu, &img5x5_tex);
    pl_tex_destroy(gpu, &fbo);
}

static struct pl_hook_res noop_hook(void *priv, const struct pl_hook_params *params)
{
    return (struct pl_hook_res) {0};
}

static void pl_ycbcr_tests(const struct pl_gpu *gpu)
{
    struct pl_renderer *rr = pl_renderer_create(gpu->ctx, gpu);
    if (!rr)
        return;

    struct pl_plane_data data[3];
    for (int i = 0; i < 3; i++) {
        const int sub = i > 0 ? 1 : 0;
        const int width = (323 + sub) >> sub;
        const int height = (255 + sub) >> sub;

        data[i] = (struct pl_plane_data) {
            .type = PL_FMT_UNORM,
            .width = width,
            .height = height,
            .component_size = {16},
            .component_map = {i},
            .pixel_stride = sizeof(uint16_t),
            .row_stride = PL_ALIGN2(width * sizeof(uint16_t),
                                    gpu->limits.align_tex_xfer_stride),
        };
    }

    const struct pl_fmt *fmt = pl_plane_find_fmt(gpu, NULL, &data[0]);
    if (!fmt || !(fmt->caps & (PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE)))
        return;

    const struct pl_tex *src_tex[3] = {0};
    const struct pl_tex *dst_tex[3] = {0};
    struct pl_frame img = {
        .num_planes = 3,
        .repr = pl_color_repr_hdtv,
        .color = pl_color_space_bt709,
    };

    struct pl_frame target = {
        .num_planes = 3,
        .repr = pl_color_repr_hdtv,
        .color = pl_color_space_bt709,
    };

    uint8_t *src_buffer[3] = {0};
    uint8_t *dst_buffer = NULL;
    for (int i = 0; i < 3; i++) {
        // Generate some arbitrary data for the buffer
        src_buffer[i] = malloc(data[i].height * data[i].row_stride);
        if (!src_buffer[i])
            goto error;

        data[i].pixels = src_buffer[i];
        for (int y = 0; y < data[i].height; y++) {
            for (int x = 0; x < data[i].width; x++) {
                size_t off = y * data[i].row_stride + x * data[i].pixel_stride;
                uint16_t *pixel = (uint16_t *) &src_buffer[i][off];
                int gx = 200 + 100 * i, gy = 300 + 150 * i;
                *pixel = (gx * x) ^ (gy * y); // whatever
            }
        }

        REQUIRE(pl_upload_plane(gpu, &img.planes[i], &src_tex[i], &data[i]));
    }

    // This co-sites chroma pixels with pixels in the RGB image, meaning we
    // get an exact round-trip when sampling both ways. This makes it useful
    // as a test case, even though it's not common in the real world.
    pl_frame_set_chroma_location(&img, PL_CHROMA_TOP_LEFT);

    for (int i = 0; i < 3; i++) {
        dst_tex[i] = pl_tex_create(gpu, &(struct pl_tex_params) {
            .format = fmt,
            .w = data[i].width,
            .h = data[i].height,
            .renderable = true,
            .host_readable = true,
            .storable = fmt->caps & PL_FMT_CAP_STORABLE,
            .blit_dst = fmt->caps & PL_FMT_CAP_BLITTABLE,
        });

        if (!dst_tex[i])
            goto error;

        target.planes[i] = img.planes[i];
        target.planes[i].texture = dst_tex[i];
    }

    REQUIRE(pl_render_image(rr, &img, &target, &(struct pl_render_params) {
        .num_hooks = 1,
        .hooks = &(const struct pl_hook *){&(struct pl_hook) {
            // Forces chroma merging, to test the chroma merging code
            .stages = PL_HOOK_CHROMA_INPUT,
            .hook = noop_hook,
        }},
    }));

    size_t buf_size = data[0].height * data[0].row_stride;
    dst_buffer = malloc(buf_size);
    if (!dst_buffer)
        goto error;

    for (int i = 0; i < 3; i++) {
        memset(dst_buffer, 0xAA, buf_size);
        REQUIRE(pl_tex_download(gpu, &(struct pl_tex_transfer_params) {
            .tex = dst_tex[i],
            .ptr = dst_buffer,
            .stride_w = data[i].row_stride / data[i].pixel_stride,
        }));

        for (int y = 0; y < data[i].height; y++) {
            for (int x = 0; x < data[i].width; x++) {
                size_t off = y * data[i].row_stride + x * data[i].pixel_stride;
                uint16_t *src_pixel = (uint16_t *) &src_buffer[i][off];
                uint16_t *dst_pixel = (uint16_t *) &dst_buffer[off];
                int diff = abs((int) *src_pixel - (int) *dst_pixel);
                REQUIRE(diff <= 50); // a little under 0.1%
            }
        }
    }

error:
    pl_renderer_destroy(&rr);
    free(dst_buffer);
    for (int i = 0; i < 3; i++) {
        free(src_buffer[i]);
        pl_tex_destroy(gpu, &src_tex[i]);
        pl_tex_destroy(gpu, &dst_tex[i]);
    }
}

static void pl_test_export_import(const struct pl_gpu *gpu,
                                  enum pl_handle_type handle_type)
{
    // Test texture roundtrip

    if (!(gpu->export_caps.tex & handle_type) ||
        !(gpu->import_caps.tex & handle_type))
        goto skip_tex;

    const struct pl_fmt *fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 1, 0, 0,
                                           PL_FMT_CAP_BLITTABLE);
    if (!fmt)
        goto skip_tex;

    printf("testing texture import/export\n");

    const struct pl_tex *export = pl_tex_create(gpu, &(struct pl_tex_params) {
        .w = 32,
        .h = 32,
        .format = fmt,
        .export_handle = handle_type,
    });
    REQUIRE(export);
    REQUIRE(export->shared_mem.handle.fd > -1);

    const struct pl_tex *import = pl_tex_create(gpu, &(struct pl_tex_params) {
        .w = 32,
        .h = 32,
        .format = fmt,
        .import_handle = handle_type,
        .shared_mem = export->shared_mem,
    });
    REQUIRE(import);

    pl_tex_destroy(gpu, &import);
    pl_tex_destroy(gpu, &export);

skip_tex: ;

    // Test buffer roundtrip

    if (!(gpu->export_caps.buf & handle_type) ||
        !(gpu->import_caps.buf & handle_type))
        return;

    printf("testing buffer import/export\n");

    const struct pl_buf *exp_buf = pl_buf_create(gpu, &(struct pl_buf_params) {
        .size = 32,
        .export_handle = handle_type,
    });
    REQUIRE(exp_buf);
    REQUIRE(exp_buf->shared_mem.handle.fd > -1);

    const struct pl_buf *imp_buf = pl_buf_create(gpu, &(struct pl_buf_params) {
        .size = 32,
        .import_handle = handle_type,
        .shared_mem = exp_buf->shared_mem,
    });
    REQUIRE(imp_buf);

    pl_buf_destroy(gpu, &imp_buf);
    pl_buf_destroy(gpu, &exp_buf);
}

static void pl_test_host_ptr(const struct pl_gpu *gpu)
{
    if (!(gpu->import_caps.buf & PL_HANDLE_HOST_PTR))
        return;

#ifdef __unix__

    printf("testing host ptr\n");
    REQUIRE(gpu->caps & PL_GPU_CAP_MAPPED_BUFFERS);

    const size_t size = 2 << 20;
    const size_t offset = 2 << 10;
    const size_t slice = 2 << 16;

    uint8_t *data = aligned_alloc(0x1000, size);
    for (int i = 0; i < size; i++)
        data[i] = (uint8_t) i;

    const struct pl_buf *buf = pl_buf_create(gpu, &(struct pl_buf_params) {
        .type = PL_BUF_TEX_TRANSFER,
        .size = slice,
        .import_handle = PL_HANDLE_HOST_PTR,
        .shared_mem = {
            .handle.ptr = data,
            .size = size,
            .offset = offset,
        },
        .host_mapped = true,
    });

    REQUIRE(buf);
    REQUIRE(memcmp(data + offset, buf->data, slice) == 0);

    pl_buf_destroy(gpu, &buf);
    free(data);

#endif // unix
}

static void gpu_shader_tests(const struct pl_gpu *gpu)
{
    pl_buffer_tests(gpu);
    pl_texture_tests(gpu);
    pl_shader_tests(gpu);
    pl_scaler_tests(gpu);
    pl_render_tests(gpu);
    pl_ycbcr_tests(gpu);

    REQUIRE(!pl_gpu_is_failed(gpu));
}

static void gpu_interop_tests(const struct pl_gpu *gpu)
{
    pl_test_export_import(gpu, PL_HANDLE_DMA_BUF);
    pl_test_host_ptr(gpu);

    REQUIRE(!pl_gpu_is_failed(gpu));
}
