#include "tests.h"
#include "shaders.h"

static void pl_test_roundtrip(const struct pl_gpu *gpu, const struct pl_tex *tex,
                              float *src, float *dst)
{
    REQUIRE(tex);

    int texels = tex->params.w;
    texels *= tex->params.h ? tex->params.h : 1;
    texels *= tex->params.d ? tex->params.d : 1;

    for (int i = 0; i < texels; i++)
        src[i] = RANDOM;

    REQUIRE(pl_tex_upload(gpu, &(struct pl_tex_transfer_params){
        .tex = tex,
        .ptr = src,
    }));

    REQUIRE(pl_tex_download(gpu, &(struct pl_tex_transfer_params){
        .tex = tex,
        .ptr = dst,
    }));

    for (int i = 0; i < texels; i++)
        REQUIRE(src[i] == dst[i]);
}

static void pl_texture_tests(const struct pl_gpu *gpu)
{
    const struct pl_fmt *fmt;
    fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 16, 32, 0);
    if (!fmt)
        return;

    struct pl_tex_params params = {
        .format        = fmt,
        .host_writable = true,
        .host_readable = true,
    };

    static float src[16*16*16] = {0};
    static float dst[16*16*16] = {0};

    const struct pl_tex *tex = NULL;
    if (gpu->limits.max_tex_1d_dim >= 16) {
        params.w = 16;
        tex = pl_tex_create(gpu, &params);
        pl_test_roundtrip(gpu, tex, src, dst);
        pl_tex_destroy(gpu, &tex);
    }

    if (gpu->limits.max_tex_2d_dim >= 16) {
        params.w = params.h = 16;
        tex = pl_tex_create(gpu, &params);
        pl_test_roundtrip(gpu, tex, src, dst);
        pl_tex_destroy(gpu, &tex);
    }

    if (gpu->limits.max_tex_3d_dim >= 16) {
        params.w = params.h = params.d = 16;
        tex = pl_tex_create(gpu, &params);
        pl_test_roundtrip(gpu, tex, src, dst);
        pl_tex_destroy(gpu, &tex);
    }
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

    struct vertex { float pos[2]; float color[3]; } vertices[] = {
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

    pl_pass_run(gpu, &(struct pl_pass_run_params) {
        .pass           = pass,
        .target         = fbo,
        .vertex_data    = vertices,
        .vertex_count   = sizeof(vertices) / sizeof(struct vertex),
    });

    static float data[FBO_H * FBO_W * 4] = {0};
    pl_tex_download(gpu, &(struct pl_tex_transfer_params) {
        .tex = fbo,
        .ptr = data,
    });

    for (int y = 0; y < FBO_H; y++) {
        for (int x = 0; x < FBO_W; x++) {
            float *color = &data[(y * FBO_W + x) * 4];
            printf("color: %f %f %f %f\n", color[0], color[1], color[2], color[3]);
            REQUIRE(feq(color[0], (x + 0.5) / FBO_W));
            REQUIRE(feq(color[1], (y + 0.5) / FBO_H));
            REQUIRE(feq(color[2], 0.0));
            REQUIRE(feq(color[3], 1.0));
        }
    }

    pl_pass_destroy(gpu, &pass);
    pl_tex_clear(gpu, fbo, (float[4]){0});

    // Test the use of pl_dispatch
    struct pl_dispatch *dp = pl_dispatch_create(gpu->ctx, gpu);

    const struct pl_tex *src;
    src = pl_tex_create(gpu, &(struct pl_tex_params) {
        .format         = fbo_fmt,
        .w              = FBO_W,
        .h              = FBO_H,
        .sampleable     = true,
        .sample_mode    = PL_TEX_SAMPLE_LINEAR,
        .initial_data   = data,
    });

    // Repeat this a few times to test the caching
    for (int i = 0; i < 10; i++) {
        printf("iteration %d\n", i);
        struct pl_shader *sh = pl_dispatch_begin(dp);

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

        pl_shader_linearize(sh, PL_COLOR_TRC_GAMMA22);
        REQUIRE(pl_dispatch_finish(dp, &sh, fbo, NULL, NULL));
    }

    pl_tex_download(gpu, &(struct pl_tex_transfer_params) {
        .tex = fbo,
        .ptr = data,
    });

    for (int y = 0; y < FBO_H; y++) {
        for (int x = 0; x < FBO_W; x++) {
            float *color = &data[(y * FBO_W + x) * 4];
            printf("color: %f %f %f %f\n", color[0], color[1], color[2], color[3]);
            REQUIRE(feq(color[0], pow((x + 0.5) / FBO_W, 2.2)));
            REQUIRE(feq(color[1], pow((y + 0.5) / FBO_H, 2.2)));
            REQUIRE(feq(color[2], 0.0));
            REQUIRE(feq(color[3], 1.0));
        }
    }

    struct pl_shader *sh;

#if PL_HAVE_LCMS
    // Test the use of 3DLUTs if available
    sh = pl_dispatch_begin(dp);
    pl_shader_sample_direct(sh, &(struct pl_sample_src) { .tex = src });

    struct pl_shader_obj *lut3d = NULL;
    struct pl_3dlut_profile src_color = { .color = pl_color_space_bt709 };
    struct pl_3dlut_profile dst_color = { .color = pl_color_space_srgb };
    struct pl_3dlut_result out;

    if (pl_3dlut_update(sh, &src_color, &dst_color, &lut3d, &out, NULL)) {
        pl_3dlut_apply(sh, &lut3d);
        REQUIRE(pl_dispatch_finish(dp, &sh, fbo, NULL, NULL));
    }

    pl_dispatch_abort(dp, &sh);
    pl_shader_obj_destroy(&lut3d);
#endif

    // Test AV1 grain synthesis
    struct pl_shader_obj *grain = NULL;
    for (int i = 0; i < 2; i++) {
        struct pl_grain_params grain_params = av1_grain_params;
        grain_params.width = FBO_W;
        grain_params.height = FBO_H;
        grain_params.grain_seed = rand();
        grain_params.overlap = !!i;
        sh = pl_dispatch_begin(dp);
        pl_shader_sample_direct(sh, &(struct pl_sample_src) { .tex = src });
        pl_shader_av1_grain(sh, &grain, (enum pl_channel[]){0, 1, 2}, NULL,
                            &grain_params);
        REQUIRE(pl_dispatch_finish(dp, &sh, fbo, NULL, NULL));
    }
    pl_shader_obj_destroy(&grain);

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
        .sample_mode    = PL_TEX_SAMPLE_LINEAR,
        .address_mode   = PL_TEX_ADDRESS_CLAMP,
        .initial_data   = &data_5x5[0][0],
    });

    const struct pl_tex *fbo = pl_tex_create(gpu, &(struct pl_tex_params) {
        .w              = 100,
        .h              = 100,
        .format         = fbo_fmt,
        .renderable     = true,
        .storable       = !!(fbo_fmt->caps & PL_FMT_CAP_STORABLE),
        .host_readable  = true,
    });

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
    REQUIRE(pl_dispatch_finish(dp, &sh, fbo, NULL, NULL));

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

error:
    free(fbo_data);
    pl_shader_obj_destroy(&lut);
    pl_dispatch_destroy(&dp);
    pl_tex_destroy(gpu, &dot5x5);
    pl_tex_destroy(gpu, &fbo);
}

static void pl_render_tests(const struct pl_gpu *gpu)
{
    const struct pl_fmt *fbo_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 16, 32,
                                               PL_FMT_CAP_RENDERABLE |
                                               PL_FMT_CAP_BLITTABLE);
    if (!fbo_fmt)
        return;

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
    const struct pl_tex *img5x5_tex = NULL;
    bool ok = pl_upload_plane(gpu, &img5x5, &img5x5_tex, &(struct pl_plane_data) {
        .type = PL_FMT_FLOAT,
        .width = width,
        .height = height,
        .component_size = { 8 * sizeof(float) },
        .component_map  = { 0 },
        .pixel_stride = sizeof(float),
        .pixels = &data_5x5,
    });

    if (!ok) {
        pl_tex_destroy(gpu, &img5x5.texture);
        return;
    }

    const struct pl_tex *fbo = pl_tex_create(gpu, &(struct pl_tex_params) {
        .w              = 40,
        .h              = 40,
        .format         = fbo_fmt,
        .renderable     = true,
        .blit_dst       = true,
        .storable       = !!(fbo_fmt->caps & PL_FMT_CAP_STORABLE),
        .host_readable  = true,
    });

    struct pl_renderer *rr = pl_renderer_create(gpu->ctx, gpu);
    if (!fbo || !rr)
        goto error;

    pl_tex_clear(gpu, fbo, (float[4]){0});

    struct pl_image image = {
        .signature      = 0,
        .num_planes     = 1,
        .planes         = { img5x5 },
        .repr = {
            .sys        = PL_COLOR_SYSTEM_BT_709,
            .levels     = PL_COLOR_LEVELS_PC,
        },
        .color          = pl_color_space_bt709,
        .width          = width,
        .height         = height,
        .src_rect       = {-1.0, 0.0, width - 1.0, height},
    };

    struct pl_render_target target = {
        .fbo            = fbo,
        .dst_rect       = {2, 2, fbo->params.w - 2, fbo->params.h - 2},
        .repr = {
            .sys        = PL_COLOR_SYSTEM_RGB,
            .levels     = PL_COLOR_LEVELS_PC,
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

    for (const struct pl_named_filter_config *f = pl_named_filters; f->name; f++) {
        struct pl_render_params params = pl_render_default_params;
        params.upscaler = f->filter;
        REQUIRE(pl_render_image(rr, &image, &target, &params));
        pl_gpu_flush(gpu);
    }

    TEST_PARAMS(deband, iterations, 3);
    TEST_PARAMS(sigmoid, center, 1);
    TEST_PARAMS(color_map, intent, PL_INTENT_ABSOLUTE_COLORIMETRIC);
    TEST_PARAMS(color_map, gamut_warning, 1);
    TEST_PARAMS(dither, method, PL_DITHER_WHITE_NOISE);
    TEST(cone_params, pl_cone_params, pl_vision_deuteranomaly, strength, 0);

    // Test HDR stuff
    image.color.sig_scale = 10.0;
    target.color.sig_scale = 2.0;
    TEST_PARAMS(color_map, tone_mapping_algo, PL_TONE_MAPPING_LINEAR);
    TEST_PARAMS(color_map, desaturation_strength, 1);
    image.color.sig_scale = target.color.sig_scale = 0.0;

    // Test some misc stuff
    struct pl_render_params params = pl_render_default_params;
    params.color_adjustment = &(struct pl_color_adjustment) {
        .brightness = 0.1,
        .contrast = 0.9,
        .saturation = 1.5,
        .gamma = 0.8,
    };

    REQUIRE(pl_render_image(rr, &image, &target, &params));

error:
    free(fbo_data);
    pl_renderer_destroy(&rr);
    pl_tex_destroy(gpu, &img5x5_tex);
    pl_tex_destroy(gpu, &fbo);
}

static void gpu_tests(const struct pl_gpu *gpu)
{
    pl_texture_tests(gpu);
    pl_shader_tests(gpu);
    pl_scaler_tests(gpu);
    pl_render_tests(gpu);
}
