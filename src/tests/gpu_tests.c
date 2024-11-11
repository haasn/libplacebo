#include "gpu_tests.h"
#include "shaders.h"

#include <libplacebo/renderer.h>
#include <libplacebo/utils/frame_queue.h>
#include <libplacebo/utils/upload.h>

//#define PRINT_OUTPUT

void pl_buffer_tests(pl_gpu gpu)
{
    const size_t buf_size = 1024;
    if (buf_size > gpu->limits.max_buf_size)
        return;
    printf("pl_buffer_tests:\n");

    uint8_t *test_src = malloc(buf_size * 2);
    uint8_t *test_dst = test_src + buf_size;
    assert(test_src && test_dst);
    memset(test_dst, 0, buf_size);
    for (int i = 0; i < buf_size; i++)
        test_src[i] = RANDOM_U8;

    pl_buf buf = NULL, tbuf = NULL;

    printf("- test buffer static creation and readback\n");
    buf = pl_buf_create(gpu, pl_buf_params(
        .size = buf_size,
        .host_readable = true,
        .initial_data = test_src,
    ));

    REQUIRE(buf);
    REQUIRE(pl_buf_read(gpu, buf, 0, test_dst, buf_size));
    REQUIRE_MEMEQ(test_src, test_dst, buf_size);
    pl_buf_destroy(gpu, &buf);

    printf("- test buffer empty creation, update and readback\n");
    memset(test_dst, 0, buf_size);
    buf = pl_buf_create(gpu, pl_buf_params(
        .size = buf_size,
        .host_writable = true,
        .host_readable = true,
    ));

    REQUIRE(buf);
    pl_buf_write(gpu, buf, 0, test_src, buf_size);
    REQUIRE(pl_buf_read(gpu, buf, 0, test_dst, buf_size));
    REQUIRE_MEMEQ(test_src, test_dst, buf_size);
    pl_buf_destroy(gpu, &buf);

    printf("- test buffer-buffer copy and readback\n");
    memset(test_dst, 0, buf_size);
    buf = pl_buf_create(gpu, pl_buf_params(
        .size = buf_size,
        .initial_data = test_src,
    ));

    tbuf = pl_buf_create(gpu, pl_buf_params(
        .size = buf_size,
        .host_readable = true,
    ));

    REQUIRE(buf && tbuf);
    pl_buf_copy(gpu, tbuf, 0, buf, 0, buf_size);
    REQUIRE(pl_buf_read(gpu, tbuf, 0, test_dst, buf_size));
    REQUIRE_MEMEQ(test_src, test_dst, buf_size);
    pl_buf_destroy(gpu, &buf);
    pl_buf_destroy(gpu, &tbuf);

    if (buf_size <= gpu->limits.max_mapped_size) {
        printf("- test host mapped buffer readback\n");
        buf = pl_buf_create(gpu, pl_buf_params(
            .size = buf_size,
            .host_mapped = true,
            .initial_data = test_src,
        ));

        REQUIRE(buf);
        REQUIRE(!pl_buf_poll(gpu, buf, 0));
        REQUIRE_MEMEQ(test_src, buf->data, buf_size);
        pl_buf_destroy(gpu, &buf);
    }

    // `compute_queues` check is to exclude dummy GPUs here
    if (buf_size <= gpu->limits.max_ssbo_size && gpu->limits.compute_queues)
    {
        printf("- test endian swapping\n");
        buf = pl_buf_create(gpu, pl_buf_params(
            .size = buf_size,
            .storable = true,
            .initial_data = test_src,
        ));

        tbuf = pl_buf_create(gpu, pl_buf_params(
            .size = buf_size,
            .storable = true,
            .host_readable = true,
        ));

        REQUIRE(buf && tbuf);
        REQUIRE(pl_buf_copy_swap(gpu, &(struct pl_buf_copy_swap_params) {
            .src = buf,
            .dst = tbuf,
            .size = buf_size,
            .wordsize = 2,
        }));
        REQUIRE(pl_buf_read(gpu, tbuf, 0, test_dst, buf_size));
        for (int i = 0; i < buf_size / 2; i++) {
            REQUIRE_CMP(test_src[2 * i + 0], ==, test_dst[2 * i + 1], PRIu8);
            REQUIRE_CMP(test_src[2 * i + 1], ==, test_dst[2 * i + 0], PRIu8);
        }
        // test endian swap in-place
        REQUIRE(pl_buf_copy_swap(gpu, &(struct pl_buf_copy_swap_params) {
            .src = tbuf,
            .dst = tbuf,
            .size = buf_size,
            .wordsize = 4,
        }));
        REQUIRE(pl_buf_read(gpu, tbuf, 0, test_dst, buf_size));
        for (int i = 0; i < buf_size / 4; i++) {
            REQUIRE_CMP(test_src[4 * i + 0], ==, test_dst[4 * i + 2], PRIu8);
            REQUIRE_CMP(test_src[4 * i + 1], ==, test_dst[4 * i + 3], PRIu8);
            REQUIRE_CMP(test_src[4 * i + 2], ==, test_dst[4 * i + 0], PRIu8);
            REQUIRE_CMP(test_src[4 * i + 3], ==, test_dst[4 * i + 1], PRIu8);
        }
        pl_buf_destroy(gpu, &buf);
        pl_buf_destroy(gpu, &tbuf);
    }

    free(test_src);
}

static void test_cb(void *priv)
{
    bool *flag = priv;
    *flag = true;
}

static void pl_test_roundtrip(pl_gpu gpu, pl_tex tex[2],
                              uint8_t *src, uint8_t *dst)
{
    if (!tex[0] || !tex[1]) {
        printf("failed creating test textures... skipping this test\n");
        return;
    }

    int texels = tex[0]->params.w;
    texels *= tex[0]->params.h ? tex[0]->params.h : 1;
    texels *= tex[0]->params.d ? tex[0]->params.d : 1;

    pl_fmt fmt = tex[0]->params.format;
    size_t bytes = texels * fmt->texel_size;
    memset(src, 0, bytes);
    memset(dst, 0, bytes);

    for (int i = 0; i < texels; i++) {
        uint8_t *data = &src[i * fmt->texel_size];
        if (fmt->type == PL_FMT_FLOAT) {
            for (int n = 0; n < fmt->num_components; n++) {
                switch (fmt->component_depth[n]) {
                case 16: *(uint16_t *) data = RANDOM_F16; data += 2; break;
                case 32: *(float *)    data = RANDOM_F32; data += 4; break;
                case 64: *(double *)   data = RANDOM_F64; data += 8; break;
                }
            }
        } else {
            for (int n = 0; n < fmt->texel_size; n++)
                data[n] = RANDOM_U8;
        }
    }

    pl_timer ul, dl;
    ul = pl_timer_create(gpu);
    dl = pl_timer_create(gpu);

    bool ran_ul = false, ran_dl = false;

    REQUIRE(pl_tex_upload(gpu, &(struct pl_tex_transfer_params){
        .tex = tex[0],
        .ptr = src,
        .timer = ul,
        .callback = gpu->limits.callbacks ? test_cb : NULL,
        .priv = &ran_ul,
    }));

    // Test blitting, if possible for this format
    pl_tex dst_tex = tex[0];
    if (tex[0]->params.blit_src && tex[1]->params.blit_dst) {
        pl_tex_clear_ex(gpu, tex[1], (union pl_clear_color){0}); // for testing
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
        .callback = gpu->limits.callbacks ? test_cb : NULL,
        .priv = &ran_dl,
    }));

    pl_gpu_finish(gpu);
    if (gpu->limits.callbacks)
        REQUIRE(ran_ul && ran_dl);

    if (fmt->emulated && fmt->type == PL_FMT_FLOAT) {
        // TODO: can't memcmp here because bits might be lost due to the
        // emulated 16/32 bit upload paths, figure out a better way to
        // generate data and verify the roundtrip!
    } else {
        REQUIRE_MEMEQ(src, dst, bytes);
    }

    // Report timer results
    printf("upload time: %"PRIu64", download time: %"PRIu64"\n",
           pl_timer_query(gpu, ul), pl_timer_query(gpu, dl));

    pl_timer_destroy(gpu, &ul);
    pl_timer_destroy(gpu, &dl);
}

void pl_texture_tests(pl_gpu gpu)
{
    const size_t max_size = 16*16*16 * 4 *sizeof(double);
    uint8_t *test_src = malloc(max_size * 2);
    uint8_t *test_dst = test_src + max_size;
    printf("pl_texture_tests:\n");

    for (int f = 0; f < gpu->num_formats; f++) {
        pl_fmt fmt = gpu->formats[f];
        if (fmt->opaque || !(fmt->caps & PL_FMT_CAP_HOST_READABLE))
            continue;

        printf("- testing texture roundtrip for format %s\n", fmt->name);
        assert(fmt->texel_size <= 4 * sizeof(double));

        struct pl_tex_params ref_params = {
            .format        = fmt,
            .blit_src      = (fmt->caps & PL_FMT_CAP_BLITTABLE),
            .blit_dst      = (fmt->caps & PL_FMT_CAP_BLITTABLE),
            .host_writable = true,
            .host_readable = true,
            .debug_tag     = PL_DEBUG_TAG,
        };

        pl_tex tex[2];

        if (gpu->limits.max_tex_1d_dim >= 16) {
            printf("-- 1D\n");
            struct pl_tex_params params = ref_params;
            params.w = 16;
            if (!gpu->limits.blittable_1d_3d)
                params.blit_src = params.blit_dst = false;
            for (int i = 0; i < PL_ARRAY_SIZE(tex); i++)
                tex[i] = pl_tex_create(gpu, &params);
            pl_test_roundtrip(gpu, tex, test_src, test_dst);
            for (int i = 0; i < PL_ARRAY_SIZE(tex); i++)
                pl_tex_destroy(gpu, &tex[i]);
        }

        if (gpu->limits.max_tex_2d_dim >= 16) {
            printf("-- 2D\n");
            struct pl_tex_params params = ref_params;
            params.w = params.h = 16;
            for (int i = 0; i < PL_ARRAY_SIZE(tex); i++)
                tex[i] = pl_tex_create(gpu, &params);
            pl_test_roundtrip(gpu, tex, test_src, test_dst);
            for (int i = 0; i < PL_ARRAY_SIZE(tex); i++)
                pl_tex_destroy(gpu, &tex[i]);
        }

        if (gpu->limits.max_tex_3d_dim >= 16) {
            printf("-- 3D\n");
            struct pl_tex_params params = ref_params;
            params.w = params.h = params.d = 16;
            if (!gpu->limits.blittable_1d_3d)
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

static void pl_planar_tests(pl_gpu gpu)
{
    pl_fmt fmt = pl_find_named_fmt(gpu, "g8_b8_r8_420");
    if (!fmt)
        return;
    printf("pl_planar_tests:\n");
    REQUIRE_CMP(fmt->num_planes, ==, 3, "d");

    const int width = 64, height = 32;
    pl_tex tex = pl_tex_create(gpu, pl_tex_params(
        .w              = width,
        .h              = height,
        .format         = fmt,
        .blit_dst       = true,
        .host_readable  = true,
    ));
    if (!tex)
        return;
    for (int i = 0; i < fmt->num_planes; i++)
        REQUIRE(tex->planes[i]);

    pl_tex plane = tex->planes[1];
    uint8_t data[(width * height) >> 2];
    REQUIRE_CMP(plane->params.w * plane->params.h, ==, PL_ARRAY_SIZE(data), "d");

    pl_tex_clear(gpu, plane, (float[]){ (float) 0x80 / 0xFF, 0.0, 0.0, 1.0 });
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(
        .tex = plane,
        .ptr = data,
    )));

    uint8_t ref[PL_ARRAY_SIZE(data)];
    memset(ref, 0x80, sizeof(ref));
    REQUIRE_MEMEQ(data, ref, PL_ARRAY_SIZE(data));

    pl_tex_destroy(gpu, &tex);
}

static void pl_shader_tests(pl_gpu gpu)
{
    if (gpu->glsl.version < 410)
        return;
    printf("pl_shader_tests:\n");

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

    pl_fmt fbo_fmt;
    enum pl_fmt_caps caps = PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_BLITTABLE |
                            PL_FMT_CAP_LINEAR;

    fbo_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 16, 32, caps);
    if (!fbo_fmt)
        return;

#define FBO_W 16
#define FBO_H 16

    pl_tex fbo;
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

    pl_tex_clear_ex(gpu, fbo, (union pl_clear_color){0});

    pl_fmt vert_fmt;
    vert_fmt = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 3);
    REQUIRE(vert_fmt);

    static const struct vertex { float pos[2]; float color[3]; } vertices[] = {
        {{-1.0, -1.0}, {0, 0, 0}},
        {{ 1.0, -1.0}, {1, 0, 0}},
        {{-1.0,  1.0}, {0, 1, 0}},
        {{ 1.0,  1.0}, {1, 1, 0}},
    };

    pl_pass pass;
    pass = pl_pass_create(gpu, &(struct pl_pass_params) {
        .type           = PL_PASS_RASTER,
        .target_format  = fbo_fmt,
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
    if (pass->params.cached_program || pass->params.cached_program_len) {
        // Ensure both are set if either one is set
        REQUIRE(pass->params.cached_program);
        REQUIRE(pass->params.cached_program_len);
    }

    pl_timer timer = pl_timer_create(gpu);
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

    static float test_data[FBO_H * FBO_W * 4] = {0};

    // Test against the known pattern of `src`, only useful for roundtrip tests
#define TEST_FBO_PATTERN(eps, fmt, ...)                                     \
    do {                                                                    \
        printf("- testing pattern of " fmt "\n", __VA_ARGS__);                \
        REQUIRE(pl_tex_download(gpu, &(struct pl_tex_transfer_params) {     \
            .tex = fbo,                                                     \
            .ptr = test_data,                                               \
        }));                                                                \
                                                                            \
        for (int y = 0; y < FBO_H; y++) {                                   \
            for (int x = 0; x < FBO_W; x++) {                               \
                float *color = &test_data[(y * FBO_W + x) * 4];             \
                REQUIRE_FEQ(color[0], (x + 0.5) / FBO_W, eps);              \
                REQUIRE_FEQ(color[1], (y + 0.5) / FBO_H, eps);              \
                REQUIRE_FEQ(color[2], 0.0, eps);                            \
                REQUIRE_FEQ(color[3], 1.0, eps);                            \
            }                                                               \
        }                                                                   \
    } while (0)

    TEST_FBO_PATTERN(1e-6, "%s", "initial rendering");

    if (sizeof(vertices) <= gpu->limits.max_vbo_size) {
        // Test the use of an explicit vertex buffer
        pl_buf vert = pl_buf_create(gpu, &(struct pl_buf_params) {
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
    pl_dispatch dp = pl_dispatch_create(gpu->log, gpu);
    pl_shader sh = pl_dispatch_begin(dp);
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

    static float src_data[FBO_H * FBO_W * 4] = {0};
    memcpy(src_data, test_data, sizeof(src_data));

    pl_tex src;
    src = pl_tex_create(gpu, &(struct pl_tex_params) {
        .format         = fbo_fmt,
        .w              = FBO_W,
        .h              = FBO_H,
        .storable       = fbo->params.storable,
        .sampleable     = true,
        .initial_data   = src_data,
    });

    if (fbo->params.storable) {
        // Test 1x1 blit, to make sure the scaling code runs
        REQUIRE(pl_tex_blit_compute(gpu, &(struct pl_tex_blit_params) {
            .src = src,
            .dst = fbo,
            .src_rc = {0, 0, 0, 1, 1, 1},
            .dst_rc = {0, 0, 0, FBO_W, FBO_H, 1},
            .sample_mode = PL_TEX_SAMPLE_NEAREST,
        }));

        // Test non-resizing blit, which uses the efficient imageLoad path
        REQUIRE(pl_tex_blit_compute(gpu, &(struct pl_tex_blit_params) {
            .src = src,
            .dst = fbo,
            .src_rc = {0, 0, 0, FBO_W, FBO_H, 1},
            .dst_rc = {0, 0, 0, FBO_W, FBO_H, 1},
            .sample_mode = PL_TEX_SAMPLE_NEAREST,
        }));

        TEST_FBO_PATTERN(1e-6, "%s", "pl_tex_blit_compute");
    }

    // Test encoding/decoding of all gamma functions, color spaces, etc.
    for (enum pl_color_transfer trc = 0; trc < PL_COLOR_TRC_COUNT; trc++) {
        struct pl_color_space test_csp = {
            .transfer = trc,
            .hdr.min_luma = PL_COLOR_HDR_BLACK,
        };
        sh = pl_dispatch_begin(dp);
        pl_shader_sample_nearest(sh, pl_sample_src( .tex = src ));
        pl_shader_delinearize(sh, &test_csp);
        pl_shader_linearize(sh, &test_csp);
        REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(
            .shader = &sh,
            .target = fbo,
        )));

        float epsilon = pl_color_transfer_is_hdr(trc) ? 1e-4 : 1e-6;
        TEST_FBO_PATTERN(epsilon, "transfer function: %s", pl_color_transfer_name(trc));
    }

    for (enum pl_color_system sys = 0; sys < PL_COLOR_SYSTEM_COUNT; sys++) {
        if (sys == PL_COLOR_SYSTEM_DOLBYVISION)
            continue; // requires metadata
        sh = pl_dispatch_begin(dp);
        pl_shader_sample_nearest(sh, pl_sample_src( .tex = src ));
        pl_shader_encode_color(sh, &(struct pl_color_repr) { .sys = sys });
        pl_shader_decode_color(sh, &(struct pl_color_repr) { .sys = sys }, NULL);
        REQUIRE(pl_dispatch_finish(dp, &(struct pl_dispatch_params) {
            .shader = &sh,
            .target = fbo,
        }));

        float epsilon;
        switch (sys) {
        case PL_COLOR_SYSTEM_BT_2020_C:
        case PL_COLOR_SYSTEM_XYZ:
            epsilon = 1e-5;
            break;

        case PL_COLOR_SYSTEM_BT_2100_PQ:
        case PL_COLOR_SYSTEM_BT_2100_HLG:
            // These seem to be horrifically noisy and prone to breaking on
            // edge cases for some reason
            // TODO: figure out why!
            continue;

        default: epsilon = 1e-6; break;
        }

        TEST_FBO_PATTERN(epsilon, "color system: %s", pl_color_system_name(sys));
    }

    // Repeat this a few times to test the caching
    pl_cache cache = pl_cache_create(pl_cache_params( .log = gpu->log ));
    pl_gpu_set_cache(gpu, cache);
    for (int i = 0; i < 10; i++) {
        if (i == 5) {
            printf("Recreating pl_dispatch to test the caching\n");
            size_t size = pl_dispatch_save(dp, NULL);
            REQUIRE(size);
            uint8_t *cache_data = malloc(size);
            REQUIRE(cache_data);
            REQUIRE_CMP(pl_dispatch_save(dp, cache_data), ==, size, "zu");

            pl_dispatch_destroy(&dp);
            dp = pl_dispatch_create(gpu->log, gpu);
            pl_dispatch_load(dp, cache_data);

            // Test to make sure the pass regenerates the same cache
            uint64_t hash = pl_str_hash((pl_str) { cache_data, size });
            REQUIRE_CMP(pl_dispatch_save(dp, NULL), ==, size, "zu");
            REQUIRE_CMP(pl_dispatch_save(dp, cache_data), ==, size, "zu");
            REQUIRE_CMP(pl_str_hash((pl_str) { cache_data, size }), ==, hash, PRIu64);
            free(cache_data);
        }

        sh = pl_dispatch_begin(dp);

        // For testing, force the use of CS if possible
        if (gpu->glsl.compute) {
            sh->type = SH_COMPUTE;
            sh->group_size[0] = 8;
            sh->group_size[1] = 8;
        }

        pl_shader_deband(sh, pl_sample_src( .tex = src ), pl_deband_params(
            .iterations     = 0,
            .grain          = 0.0,
        ));

        REQUIRE(pl_dispatch_finish(dp, &(struct pl_dispatch_params) {
            .shader = &sh,
            .target = fbo,
        }));
        TEST_FBO_PATTERN(1e-6, "deband iter %d", i);
    }

    pl_gpu_set_cache(gpu, NULL);
    pl_cache_destroy(&cache);

    // Test peak detection and readback if possible
    sh = pl_dispatch_begin(dp);
    pl_shader_sample_nearest(sh, pl_sample_src( .tex = src ));

    pl_shader_obj peak_state = NULL;
    struct pl_color_space csp_gamma22 = { .transfer = PL_COLOR_TRC_GAMMA22 };
    struct pl_peak_detect_params peak_params = { .minimum_peak = 0.01 };
    if (pl_shader_detect_peak(sh, csp_gamma22, &peak_state, &peak_params)) {
        REQUIRE(pl_dispatch_compute(dp, &(struct pl_dispatch_compute_params) {
            .shader = &sh,
            .width = fbo->params.w,
            .height = fbo->params.h,
        }));

        struct pl_hdr_metadata hdr;
        REQUIRE(pl_get_detected_hdr_metadata(peak_state, &hdr));

        float real_peak = 0, real_avg = 0;
        for (int y = 0; y < FBO_H; y++) {
            for (int x = 0; x < FBO_W; x++) {
                float *color = &src_data[(y * FBO_W + x) * 4];
                float luma = 0.212639f * powf(color[0], 2.2f) +
                             0.715169f * powf(color[1], 2.2f) +
                             0.072192f * powf(color[2], 2.2f);
                luma = pl_hdr_rescale(PL_HDR_NORM, PL_HDR_PQ, luma);
                real_peak = PL_MAX(real_peak, luma);
                real_avg += luma;
            }
        }
        real_avg = real_avg / (FBO_W * FBO_H);
        REQUIRE_FEQ(hdr.max_pq_y, real_peak, 1e-4);
        REQUIRE_FEQ(hdr.avg_pq_y, real_avg,  1e-3);
    }

    pl_dispatch_abort(dp, &sh);
    pl_shader_obj_destroy(&peak_state);

    // Test film grain synthesis
    pl_shader_obj grain = NULL;
    struct pl_film_grain_params grain_params = {
        .tex = src,
        .components = 3,
        .component_mapping = { 0, 1, 2},
        .repr = &(struct pl_color_repr) {
            .sys = PL_COLOR_SYSTEM_BT_709,
            .levels = PL_COLOR_LEVELS_LIMITED,
            .bits = { .color_depth = 10, .sample_depth = 10 },
        },
    };

    for (int i = 0; i < 2; i++) {
        grain_params.data.type = PL_FILM_GRAIN_AV1;
        grain_params.data.params.av1 = av1_grain_data;
        grain_params.data.params.av1.overlap = !!i;
        grain_params.data.seed = rand();

        sh = pl_dispatch_begin(dp);
        pl_shader_film_grain(sh, &grain, &grain_params);
        REQUIRE(pl_dispatch_finish(dp, &(struct pl_dispatch_params) {
            .shader = &sh,
            .target = fbo,
        }));
    }

    if (gpu->glsl.compute) {
        grain_params.data.type = PL_FILM_GRAIN_H274;
        grain_params.data.params.h274 = h274_grain_data;
        grain_params.data.seed = rand();

        sh = pl_dispatch_begin(dp);
        pl_shader_film_grain(sh, &grain, &grain_params);
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

    // Test dolbyvision
    struct pl_color_repr repr = {
        .sys = PL_COLOR_SYSTEM_DOLBYVISION,
        .dovi = &dovi_meta,
    };

    sh = pl_dispatch_begin(dp);
    pl_shader_sample_direct(sh, pl_sample_src( .tex = src ));
    pl_shader_decode_color(sh, &repr, NULL);
    REQUIRE(pl_dispatch_finish(dp, &(struct pl_dispatch_params) {
        .shader = &sh,
        .target = fbo,
    }));

    // Test deinterlacing
    sh = pl_dispatch_begin(dp);
    pl_shader_deinterlace(sh, pl_deinterlace_source( .cur = pl_field_pair(src) ), NULL);
    REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(
        .shader = &sh,
        .target = fbo,
    )));

    // Test error diffusion
    if (fbo->params.storable) {
        for (int i = 0; i < pl_num_error_diffusion_kernels; i++) {
            const struct pl_error_diffusion_kernel *k = pl_error_diffusion_kernels[i];
            printf("- testing error diffusion kernel '%s'\n", k->name);
            sh = pl_dispatch_begin(dp);
            bool ok = pl_shader_error_diffusion(sh, pl_error_diffusion_params(
                .input_tex  = src,
                .output_tex = fbo,
                .new_depth  = 8,
                .kernel     = k,
            ));

            if (!ok) {
                fprintf(stderr, "kernel '%s' exceeds GPU limits, skipping...\n", k->name);
                continue;
            }

            REQUIRE(pl_dispatch_compute(dp, pl_dispatch_compute_params(
                .shader = &sh,
                .dispatch_size = {1, 1, 1},
            )));
        }
    }

    pl_dispatch_destroy(&dp);
    pl_tex_destroy(gpu, &src);
    pl_tex_destroy(gpu, &fbo);
}

static void pl_scaler_tests(pl_gpu gpu)
{
    pl_fmt src_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 16, 32, PL_FMT_CAP_LINEAR);
    pl_fmt fbo_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 16, 32, PL_FMT_CAP_RENDERABLE);
    if (!src_fmt || !fbo_fmt)
        return;
    printf("pl_scaler_tests:\n");

    float *fbo_data = NULL;
    pl_shader_obj lut = NULL;

    static float data_5x5[5][5] = {
        { 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0 },
        { 0, 0, 1, 0, 0 },
        { 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0 },
    };

    pl_tex dot5x5 = pl_tex_create(gpu, &(struct pl_tex_params) {
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
        .storable       = fbo_fmt->caps & PL_FMT_CAP_STORABLE,
        .host_readable  = fbo_fmt->caps & PL_FMT_CAP_HOST_READABLE,
    };

    pl_tex fbo = pl_tex_create(gpu, &fbo_params);
    pl_dispatch dp = pl_dispatch_create(gpu->log, gpu);
    if (!dot5x5 || !fbo || !dp)
        goto error;

    pl_shader sh = pl_dispatch_begin(dp);
    REQUIRE(pl_shader_sample_polar(sh,
        pl_sample_src(
            .tex        = dot5x5,
            .new_w      = fbo->params.w,
            .new_h      = fbo->params.h,
        ),
        pl_sample_filter_params(
            .filter     = pl_filter_ewa_lanczos,
            .lut        = &lut,
            .no_compute = !fbo->params.storable,
        )
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

#ifdef PRINT_OUTPUT
        int max = 255;
        printf("P2\n%d %d\n%d\n", fbo->params.w, fbo->params.h, max);
        for (int y = 0; y < fbo->params.h; y++) {
            for (int x = 0; x < fbo->params.w; x++) {
                float v = fbo_data[y * fbo->params.h + x];
                printf("%d ", (int) round(fmin(fmax(v, 0.0), 1.0) * max));
            }
            printf("\n");
        }
#endif
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
    "//!FORMAT rgba8                                                        \n"
    "//!FILTER NEAREST                                                      \n"
    "//!BORDER REPEAT                                                       \n"
    "ff0000ff00ff00ff0000ffff00ffffffff00ffffffff00ff4c4c4cff999999ffffffffff\n"

    // Test custom parameters
    "//!PARAM test                                                          \n"
    "//!DESC test parameter                                                 \n"
    "//!TYPE DYNAMIC float                                                  \n"
    "//!MINIMUM 0.0                                                         \n"
    "//!MAXIMUM 100.0                                                       \n"
    "1.0                                                                    \n"
    "                                                                       \n"
    "//!PARAM testconst                                                     \n"
    "//!TYPE CONSTANT uint                                                  \n"
    "//!MAXIMUM 16                                                          \n"
    "3                                                                      \n"
    "                                                                       \n"
    "//!PARAM testdefine                                                    \n"
    "//!TYPE DEFINE                                                         \n"
    "100                                                                    \n"
    "                                                                       \n"
    "//!PARAM testenum                                                      \n"
    "//!TYPE ENUM DEFINE                                                    \n"
    "FOO                                                                    \n"
    "BAR                                                                    \n"
    "                                                                       \n"
    "//!HOOK MAIN                                                           \n"
    "//!WHEN testconst 30 >                                                 \n"
    "#error should not be run                                               \n"
    "                                                                       \n"
    "//!HOOK MAIN                                                           \n"
    "//!WHEN testenum FOO =                                                 \n"
    "#if testenum == BAR                                                    \n"
    " #error bad                                                            \n"
    "#endif                                                                 \n"
    "vec4 hook() { return vec4(0.0); }                                      \n"
};

static const char *compute_shader_tests[] = {
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
    "//!STORAGE                                                             \n",

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

static bool frame_passthrough(pl_gpu gpu, pl_tex *tex,
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

static void render_info_cb(void *priv, const struct pl_render_info *info)
{
    printf("{%d} Executed shader: %s\n", info->index,
           info->pass->shader->description);
}

static void pl_render_tests(pl_gpu gpu)
{
    pl_tex img_tex = NULL, fbo = NULL;
    pl_renderer rr = NULL;
    printf("pl_render_tests:\n");

    enum { width = 50, height = 50 };
    static float data[width][height];
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++)
            data[y][x] = RANDOM;
    }

    struct pl_plane img_plane = {0};
    struct pl_plane_data plane_data = {
        .type = PL_FMT_FLOAT,
        .width = width,
        .height = height,
        .component_size = { 8 * sizeof(float) },
        .component_map  = { 0 },
        .pixel_stride = sizeof(float),
        .pixels = data,
    };

    if (!pl_recreate_plane(gpu, NULL, &fbo, &plane_data))
        return;

    if (!pl_upload_plane(gpu, &img_plane, &img_tex, &plane_data))
        goto error;

    rr = pl_renderer_create(gpu->log, gpu);
    pl_tex_clear_ex(gpu, fbo, (union pl_clear_color){0});

    struct pl_frame image = {
        .num_planes     = 1,
        .planes         = { img_plane },
        .repr = {
            .sys        = PL_COLOR_SYSTEM_BT_709,
            .levels     = PL_COLOR_LEVELS_FULL,
        },
        .color          = pl_color_space_srgb,
    };

    struct pl_frame target = {
        .num_planes     = 1,
        .planes         = {{
            .texture            = fbo,
            .components         = 3,
            .component_mapping  = {0, 1, 2},
        }},
        .repr = {
            .sys        = PL_COLOR_SYSTEM_RGB,
            .levels     = PL_COLOR_LEVELS_FULL,
            .bits.color_depth = 32,
        },
        .color          = pl_color_space_srgb,
    };

    REQUIRE(pl_render_image(rr, &image, &target, NULL));
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);

    // TODO: embed a reference texture and ensure it matches

    // Test a bunch of different params
#define TEST(SNAME, STYPE, DEFAULT, FIELD, LIMIT)                       \
    do {                                                                \
        for (int i = 0; i <= LIMIT; i++) {                              \
            printf("- testing `" #STYPE "." #FIELD " = %d`\n", i);        \
            struct pl_render_params params = pl_render_default_params;  \
            params.force_dither = true;                                 \
            struct STYPE tmp = DEFAULT;                                 \
            tmp.FIELD = i;                                              \
            params.SNAME = &tmp;                                        \
            REQUIRE(pl_render_image(rr, &image, &target, &params));     \
            pl_gpu_flush(gpu);                                          \
            REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE); \
        }                                                               \
    } while (0)

#define TEST_PARAMS(NAME, FIELD, LIMIT) \
    TEST(NAME##_params, pl_##NAME##_params, pl_##NAME##_default_params, FIELD, LIMIT)

    image.crop.x1 = width / 2.0;
    image.crop.y1 = height / 2.0;
    for (int i = 0; i < pl_num_scale_filters; i++) {
        struct pl_render_params params = pl_render_default_params;
        params.upscaler = pl_scale_filters[i].filter;
        printf("- testing `params.upscaler = /* %s */`\n", pl_scale_filters[i].name);
        REQUIRE(pl_render_image(rr, &image, &target, &params));
        pl_gpu_flush(gpu);
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
    }
    image.crop.x1 = image.crop.y1 = 0;

    target.crop.x1 = width / 2.0;
    target.crop.y1 = height / 2.0;
    for (int i = 0; i < pl_num_scale_filters; i++) {
        struct pl_render_params params = pl_render_default_params;
        params.downscaler = pl_scale_filters[i].filter;
        printf("- testing `params.downscaler = /* %s */`\n", pl_scale_filters[i].name);
        REQUIRE(pl_render_image(rr, &image, &target, &params));
        pl_gpu_flush(gpu);
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
    }
    target.crop.x1 = target.crop.y1 = 0;

    TEST_PARAMS(deband, iterations, 3);
    TEST_PARAMS(sigmoid, center, 1);
    TEST_PARAMS(color_map, intent, PL_INTENT_ABSOLUTE_COLORIMETRIC);
    TEST_PARAMS(dither, method, PL_DITHER_WHITE_NOISE);
    TEST_PARAMS(dither, temporal, true);
    TEST_PARAMS(distort, alpha_mode, PL_ALPHA_INDEPENDENT);
    TEST_PARAMS(distort, constrain, true);
    TEST_PARAMS(distort, bicubic, true);
    TEST(cone_params, pl_cone_params, pl_vision_deuteranomaly, strength, 0);

    // Test gamma-correct dithering
    target.repr.bits.color_depth = 2;
    TEST_PARAMS(dither, transfer, PL_COLOR_TRC_GAMMA22);
    target.repr.bits.color_depth = 32;

    // Test HDR tone mapping
    image.color = pl_color_space_hdr10;
    TEST_PARAMS(color_map, visualize_lut, true);
    if (gpu->limits.max_ssbo_size)
        TEST_PARAMS(peak_detect, allow_delayed, true);

    // Test inverse tone-mapping and pure BPC
    image.color.hdr.max_luma = 1000;
    target.color.hdr.max_luma = 4000;
    target.color.hdr.min_luma = 0.02;
    TEST_PARAMS(color_map, inverse_tone_mapping, true);

    image.color = pl_color_space_srgb;
    target.color = pl_color_space_srgb;

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
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
    params = pl_render_default_params;

    struct pl_frame inferred_image = image, inferred_target = target;
    pl_frames_infer(rr, &inferred_image, &inferred_target);
    REQUIRE(pl_render_image(rr, &inferred_image, &inferred_target, &params));
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);

    // Test background blending and alpha transparency
    params.blend_against_tiles = true;
    params.corner_rounding = 0.25f;
    REQUIRE(pl_render_image(rr, &image, &target, &params));
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
    params = pl_render_default_params;

    // Test film grain synthesis
    image.film_grain.type = PL_FILM_GRAIN_AV1;
    image.film_grain.params.av1 = av1_grain_data;
    REQUIRE(pl_render_image(rr, &image, &target, &params));
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);

    image.film_grain.type = PL_FILM_GRAIN_H274;
    image.film_grain.params.h274 = h274_grain_data;
    REQUIRE(pl_render_image(rr, &image, &target, &params));
    // H.274 film grain synthesis requires compute shaders
    if (gpu->glsl.compute) {
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
    } else {
        const struct pl_render_errors rr_err = pl_renderer_get_errors(rr);
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_FILM_GRAIN);
        pl_renderer_reset_errors(rr, &rr_err);
    }
    image.film_grain = (struct pl_film_grain_data) {0};

    // Test mpv-style custom shaders
    for (int i = 0; i < PL_ARRAY_SIZE(user_shader_tests); i++) {
        printf("- testing user shader:\n\n%s\n", user_shader_tests[i]);
        const struct pl_hook *hook;
        hook = pl_mpv_user_shader_parse(gpu, user_shader_tests[i],
                                        strlen(user_shader_tests[i]));
        REQUIRE(hook);

        params.hooks = &hook;
        params.num_hooks = 1;
        REQUIRE(pl_render_image(rr, &image, &target, &params));
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);

        pl_mpv_user_shader_destroy(&hook);
    }

    if (gpu->glsl.compute && gpu->limits.max_ssbo_size) {
        for (int i = 0; i < PL_ARRAY_SIZE(compute_shader_tests); i++) {
            printf("- testing user shader:\n\n%s\n", compute_shader_tests[i]);
            const struct pl_hook *hook;
            hook = pl_mpv_user_shader_parse(gpu, compute_shader_tests[i],
                                            strlen(compute_shader_tests[i]));
            REQUIRE(hook);

            params.hooks = &hook;
            params.num_hooks = 1;
            REQUIRE(pl_render_image(rr, &image, &target, &params));
            REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);

            pl_mpv_user_shader_destroy(&hook);
        }
    }
    params = pl_render_default_params;

    // Test custom LUTs
    for (int i = 0; i < PL_ARRAY_SIZE(test_luts); i++) {
        printf("- testing custom lut %d\n", i);
        struct pl_custom_lut *lut;
        lut = pl_lut_parse_cube(gpu->log, test_luts[i], strlen(test_luts[i]));
        REQUIRE(lut);

        bool has_3dlut = gpu->limits.max_tex_3d_dim && gpu->glsl.version > 100;
        if (lut->size[2] && !has_3dlut) {
            pl_lut_free(&lut);
            continue;
        }

        // Test all three at the same time to reduce the number of tests
        image.lut = target.lut = params.lut = lut;

        for (enum pl_lut_type t = PL_LUT_UNKNOWN; t <= PL_LUT_CONVERSION; t++) {
            printf("- testing LUT method %d\n", t);
            image.lut_type = target.lut_type = params.lut_type = t;
            REQUIRE(pl_render_image(rr, &image, &target, &params));
            REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
        }

        image.lut = target.lut = params.lut = NULL;
        pl_lut_free(&lut);
    }

#ifdef PL_HAVE_LCMS

    // It doesn't fit without use of 3D textures on GLES2
    if (gpu->glsl.version > 100) {
        // Test ICC profiles
        image.profile = TEST_PROFILE(sRGB_v2_nano_icc);
        REQUIRE(pl_render_image(rr, &image, &target, &params));
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
        image.profile = (struct pl_icc_profile) {0};

        target.profile = TEST_PROFILE(sRGB_v2_nano_icc);
        REQUIRE(pl_render_image(rr, &image, &target, &params));
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
        target.profile = (struct pl_icc_profile) {0};

        image.profile = TEST_PROFILE(sRGB_v2_nano_icc);
        target.profile = image.profile;
        REQUIRE(pl_render_image(rr, &image, &target, &params));
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
        image.profile = (struct pl_icc_profile) {0};
        target.profile = (struct pl_icc_profile) {0};
    }

#endif

    // Test overlays
    image.num_overlays = 1;
    image.overlays = &(struct pl_overlay) {
        .tex = img_plane.texture,
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
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
    params.disable_fbos = true;
    REQUIRE(pl_render_image(rr, &image, &target, &params));
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
    image.num_overlays = 0;
    params = pl_render_default_params;

    target.num_overlays = 1;
    target.overlays = &(struct pl_overlay) {
        .tex = img_plane.texture,
        .mode = PL_OVERLAY_MONOCHROME,
        .num_parts = 1,
        .parts = &(struct pl_overlay_part) {
            .src = {5, 5, 15, 15},
            .dst = {5, 5, 15, 15},
            .color = {1.0, 0.5, 0.0},
        },
    };
    REQUIRE(pl_render_image(rr, &image, &target, &params));
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
    REQUIRE(pl_render_image(rr, NULL, &target, &params));
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
    target.num_overlays = 0;

    // Test rotation
    for (pl_rotation rot = 0; rot < PL_ROTATION_360; rot += PL_ROTATION_90) {
        image.rotation = rot;
        REQUIRE(pl_render_image(rr, &image, &target, &params));
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
    }

    // Attempt frame mixing, using the mixer queue helper
    printf("- testing frame mixing\n");
    struct pl_render_params mix_params = {
        .frame_mixer = &pl_filter_mitchell_clamp,
        .info_callback = render_info_cb,
    };

    struct pl_queue_params qparams = {
        .radius = pl_frame_mix_radius(&mix_params),
        .vsync_duration = 1.0 / 60.0,
    };

    // Test large PTS jumps in frame mix
    struct pl_frame_mix mix = (struct pl_frame_mix) {
        .num_frames = 2,
        .frames = (const struct pl_frame *[]) { &image, &image },
        .signatures = (uint64_t[]) { 0xFFF1, 0xFFF2 },
        .timestamps = (float[]) { -100, 100 },
        .vsync_duration = 1.6,
    };
    REQUIRE(pl_render_image_mix(rr, &mix, &target, &mix_params));

    // Test inferring frame mix
    inferred_target = target;
    pl_frames_infer_mix(rr, &mix, &inferred_target, &inferred_image);
    REQUIRE(pl_render_image_mix(rr, &mix, &target, &mix_params));

    // Test empty frame mix
    mix = (struct pl_frame_mix) {0};
    REQUIRE(pl_render_image_mix(rr, &mix, &target, &mix_params));

    // Test inferring empty frame mix
    inferred_target = target;
    pl_frames_infer_mix(rr, &mix, &inferred_target, &inferred_image);
    REQUIRE(pl_render_image_mix(rr, &mix, &target, &mix_params));

    // Test mixer queue
#define NUM_MIX_FRAMES 20
    const float frame_duration = 1.0 / 24.0;
    struct pl_source_frame srcframes[NUM_MIX_FRAMES+1];
    srcframes[NUM_MIX_FRAMES] = (struct pl_source_frame) {0};
    for (int i = 0; i < NUM_MIX_FRAMES; i++) {
        srcframes[i] = (struct pl_source_frame) {
            .pts = i * frame_duration,
            .duration = frame_duration,
            .map = frame_passthrough,
            .frame_data = &image,
        };
    }

    pl_queue queue = pl_queue_create(gpu);
    enum pl_queue_status ret;

    // Test pre-pushing all frames, with delayed EOF.
    for (int i = 0; i < NUM_MIX_FRAMES; i++) {
        const struct pl_source_frame *src = &srcframes[i];
        if (i > 10) // test pushing in reverse order
            src = &srcframes[NUM_MIX_FRAMES + 10 - i];
        if (!pl_queue_push_block(queue, 1, src)) // mini-sleep
            pl_queue_push(queue, src); // push it anyway, for testing
    }

    while ((ret = pl_queue_update(queue, &mix, &qparams)) != PL_QUEUE_EOF) {
        if (ret == PL_QUEUE_MORE) {
            REQUIRE_CMP(qparams.pts, >, 0.0f, "f");
            pl_queue_push(queue, NULL); // push delayed EOF
            continue;
        }

        REQUIRE_CMP(ret, ==, PL_QUEUE_OK, "u");
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
        .get_frame = get_frame_ptr,
        .priv = &frame_ptr,
    };

    pl_queue_reset(queue);
    while ((ret = pl_queue_update(queue, &mix, &qparams)) != PL_QUEUE_EOF) {
        REQUIRE_CMP(ret, ==, PL_QUEUE_OK, "u");
        REQUIRE_CMP(mix.num_frames, <=, 2, "d");
        REQUIRE(pl_render_image_mix(rr, &mix, &target, &mix_params));
        qparams.pts += qparams.vsync_duration;
    }

    // Test large PTS jump
    pl_queue_reset(queue);
    REQUIRE(pl_queue_update(queue, &mix, &qparams) == PL_QUEUE_EOF);

    // Test deinterlacing
    pl_queue_reset(queue);
    printf("- testing deinterlacing\n");
    for (int i = 0; i < NUM_MIX_FRAMES; i++) {
        struct pl_source_frame *src = &srcframes[i];
        if (i > 10)
            src = &srcframes[NUM_MIX_FRAMES + 10 - i];
        src->first_field = PL_FIELD_EVEN;
        pl_queue_push(queue, src);
    }
    pl_queue_push(queue, NULL);

    qparams.pts = 0;
    qparams.get_frame = NULL;
    while ((ret = pl_queue_update(queue, &mix, &qparams)) != PL_QUEUE_EOF) {
        REQUIRE_CMP(ret, ==, PL_QUEUE_OK, "u");
        REQUIRE(pl_render_image_mix(rr, &mix, &target, &mix_params));
        qparams.pts += qparams.vsync_duration;
    }

    pl_queue_destroy(&queue);

error:
    pl_renderer_destroy(&rr);
    pl_tex_destroy(gpu, &img_tex);
    pl_tex_destroy(gpu, &fbo);
}

static struct pl_hook_res noop_hook(void *priv, const struct pl_hook_params *params)
{
    return (struct pl_hook_res) {0};
}

static void pl_ycbcr_tests(pl_gpu gpu)
{
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
                                    gpu->limits.align_tex_xfer_pitch),
        };
    }

    pl_fmt fmt = pl_plane_find_fmt(gpu, NULL, &data[0]);
    enum pl_fmt_caps caps = PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE;
    if (!fmt || (fmt->caps & caps) != caps)
        return;

    printf("pl_ycbcr_tests:\nfmt=%s\n", fmt->name);

    pl_renderer rr = pl_renderer_create(gpu->log, gpu);
    if (!rr)
        return;

    pl_tex src_tex[3] = {0};
    pl_tex dst_tex[3] = {0};
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
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);

    size_t buf_size = data[0].height * data[0].row_stride;
    dst_buffer = malloc(buf_size);
    if (!dst_buffer)
        goto error;

    for (int i = 0; i < 3; i++) {
        memset(dst_buffer, 0xAA, buf_size);
        REQUIRE(pl_tex_download(gpu, &(struct pl_tex_transfer_params) {
            .tex = dst_tex[i],
            .ptr = dst_buffer,
            .row_pitch = data[i].row_stride,
        }));

        for (int y = 0; y < data[i].height; y++) {
            for (int x = 0; x < data[i].width; x++) {
                size_t off = y * data[i].row_stride + x * data[i].pixel_stride;
                uint16_t *src_pixel = (uint16_t *) &src_buffer[i][off];
                uint16_t *dst_pixel = (uint16_t *) &dst_buffer[off];
                int diff = abs((int) *src_pixel - (int) *dst_pixel);
                REQUIRE_CMP(diff, <=, 150, "d"); // a little over 0.2%
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

static void pl_test_export_import(pl_gpu gpu,
                                  enum pl_handle_type handle_type)
{
    printf("pl_test_export_import:\n");
    // Test texture roundtrip

    if (!(gpu->export_caps.tex & handle_type) ||
        !(gpu->import_caps.tex & handle_type))
        goto skip_tex;

    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 4, 0, 0, PL_FMT_CAP_BLITTABLE);
    if (!fmt)
        goto skip_tex;

    printf("- testing texture import/export with fmt %s\n", fmt->name);

    pl_tex export = pl_tex_create(gpu, &(struct pl_tex_params) {
        .w = 32,
        .h = 32,
        .format = fmt,
        .export_handle = handle_type,
    });
    REQUIRE(export);
    REQUIRE_HANDLE(export->shared_mem, handle_type);

    pl_tex import = pl_tex_create(gpu, &(struct pl_tex_params) {
        .w = export->params.w,
        .h = export->params.h,
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

    printf("- testing buffer import/export\n");

    pl_buf exp_buf = pl_buf_create(gpu, &(struct pl_buf_params) {
        .size = 32,
        .export_handle = handle_type,
    });
    REQUIRE(exp_buf);
    REQUIRE_HANDLE(exp_buf->shared_mem, handle_type);

    pl_buf imp_buf = pl_buf_create(gpu, &(struct pl_buf_params) {
        .size = 32,
        .import_handle = handle_type,
        .shared_mem = exp_buf->shared_mem,
    });
    REQUIRE(imp_buf);

    pl_buf_destroy(gpu, &imp_buf);
    pl_buf_destroy(gpu, &exp_buf);
}

static void pl_test_host_ptr(pl_gpu gpu)
{
    if (!(gpu->import_caps.buf & PL_HANDLE_HOST_PTR))
        return;

    printf("pl_test_host_ptr:\n");

#ifdef __unix__

    printf("- testing host ptr\n");
    REQUIRE(gpu->limits.max_mapped_size);

    const size_t size = 2 << 20;
    const size_t offset = 2 << 10;
    const size_t slice = 2 << 16;

    uint8_t *data = aligned_alloc(0x1000, size);
    for (int i = 0; i < size; i++)
        data[i] = (uint8_t) i;

    pl_buf buf = pl_buf_create(gpu, &(struct pl_buf_params) {
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
    REQUIRE_MEMEQ(data + offset, buf->data, slice);

    pl_buf_destroy(gpu, &buf);
    free(data);

#endif // unix
}

void gpu_shader_tests(pl_gpu gpu)
{
    pl_buffer_tests(gpu);
    pl_texture_tests(gpu);
    pl_planar_tests(gpu);
    pl_shader_tests(gpu);
    pl_scaler_tests(gpu);
    pl_render_tests(gpu);
    pl_ycbcr_tests(gpu);

    REQUIRE(!pl_gpu_is_failed(gpu));
}

void gpu_interop_tests(pl_gpu gpu)
{
    pl_test_export_import(gpu, PL_HANDLE_DMA_BUF);
    pl_test_host_ptr(gpu);

    REQUIRE(!pl_gpu_is_failed(gpu));
}
