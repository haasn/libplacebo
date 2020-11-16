#include "gpu_tests.h"

#ifndef EPOXY_HAS_EGL
int main()
{
    return SKIP;
}
#else // EPOXY_HAS_EGL

#include "opengl/utils.h"
#include <epoxy/gl.h>
#include <epoxy/egl.h>

static void opengl_interop_tests(const struct pl_gpu *gpu)
{
    const struct pl_fmt *fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 1, 0, 0,
                                           PL_FMT_CAP_RENDERABLE |
                                           PL_FMT_CAP_LINEAR);
    if (!fmt)
        return;

    const struct pl_tex *export = pl_tex_create(gpu, &(struct pl_tex_params) {
        .w = 32,
        .h = 32,
        .format = fmt,
        .sampleable = true,
        .renderable = true,
        .blit_dst = fmt->caps & PL_FMT_CAP_BLITTABLE,
        .sample_mode = PL_TEX_SAMPLE_LINEAR,
        .address_mode = PL_TEX_ADDRESS_REPEAT,
    });

    REQUIRE(export);

    struct pl_opengl_wrap_params wrap = {
        .width = export->params.w,
        .height = export->params.h,
        .depth = export->params.d,
    };

    wrap.texture = pl_opengl_unwrap(gpu, export, &wrap.target, &wrap.iformat, NULL);
    REQUIRE(wrap.texture);

    const struct pl_tex *import = pl_opengl_wrap(gpu, &wrap);
    REQUIRE(import);
    REQUIRE(import->params.renderable);
    REQUIRE(import->params.blit_dst == export->params.blit_dst);

    pl_tex_destroy(gpu, &import);
    pl_tex_destroy(gpu, &export);
}

#define PBUFFER_WIDTH 640
#define PBUFFER_HEIGHT 480

struct swapchain_priv {
    EGLDisplay display;
    EGLSurface surface;
};

static void swap_buffers(void *priv)
{
    struct swapchain_priv *p = priv;
    eglSwapBuffers(p->display, p->surface);
}

static void opengl_swapchain_tests(const struct pl_opengl *gl,
                                   EGLDisplay display, EGLSurface surface)
{
    if (surface == EGL_NO_SURFACE)
        return;

    printf("testing opengl swapchain\n");
    const struct pl_gpu *gpu = gl->gpu;
    const struct pl_swapchain *sw;
    sw = pl_opengl_create_swapchain(gl, &(struct pl_opengl_swapchain_params) {
        .swap_buffers = swap_buffers,
        .priv = &(struct swapchain_priv) { display, surface },
    });
    REQUIRE(sw);

    int w = PBUFFER_WIDTH, h = PBUFFER_HEIGHT;
    REQUIRE(pl_swapchain_resize(sw, &w, &h));

    for (int i = 0; i < 10; i++) {
        struct pl_swapchain_frame frame;
        REQUIRE(pl_swapchain_start_frame(sw, &frame));
        if (frame.fbo->params.blit_dst)
            pl_tex_clear(gpu, frame.fbo, (float[4]){0});

        // TODO: test this with an actual pl_renderer instance
        struct pl_render_target target;
        pl_render_target_from_swapchain(&target, &frame);

        REQUIRE(pl_swapchain_submit_frame(sw));
        pl_swapchain_swap_buffers(sw);
    }

    pl_swapchain_destroy(&sw);
}

static void opengl_test_export_import(const struct pl_opengl *gl,
                                      enum pl_handle_type handle_type)
{
    const struct pl_gpu *gpu = gl->gpu;
    printf("testing opengl import/export\n");

    if (!(gpu->export_caps.tex & handle_type) ||
        !(gpu->import_caps.tex & handle_type)) {
        fprintf(stderr, "%s unsupported caps!\n", __func__);
        return;
    }

    const struct pl_fmt *fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 1, 0, 0,
                                           PL_FMT_CAP_BLITTABLE);
    if (!fmt) {
        fprintf(stderr, "%s unsupported format\n", __func__);
        return;
    }

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
}

int main()
{
    // Create the OpenGL context
    if (!epoxy_has_egl_extension(EGL_NO_DISPLAY, "EGL_MESA_platform_surfaceless"))
        return SKIP;

    EGLDisplay dpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA,
                                              EGL_DEFAULT_DISPLAY, NULL);
    if (dpy == EGL_NO_DISPLAY)
        return SKIP;

    EGLint major, minor;
    if (!eglInitialize(dpy, &major, &minor))
        return SKIP;

    printf("Initialized EGL v%d.%d\n", major, minor);
    int egl_ver = major * 10 + minor;

    struct {
        EGLenum api;
        EGLenum render;
        int major, minor;
        int glsl_ver;
        EGLenum profile;
    } egl_vers[] = {
        { EGL_OPENGL_API,       EGL_OPENGL_BIT,     4, 6, 460, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT },
        { EGL_OPENGL_API,       EGL_OPENGL_BIT,     4, 5, 450, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT },
        { EGL_OPENGL_API,       EGL_OPENGL_BIT,     4, 4, 440, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT },
        { EGL_OPENGL_API,       EGL_OPENGL_BIT,     4, 0, 400, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT },
        { EGL_OPENGL_API,       EGL_OPENGL_BIT,     3, 3, 330, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT },
        { EGL_OPENGL_API,       EGL_OPENGL_BIT,     3, 2, 150, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT },
        { EGL_OPENGL_API,       EGL_OPENGL_BIT,     3, 1, 140, EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT, },
        { EGL_OPENGL_API,       EGL_OPENGL_BIT,     3, 0, 130, EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT, },
        { EGL_OPENGL_API,       EGL_OPENGL_BIT,     2, 1, 120, EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT, },
        { EGL_OPENGL_API,       EGL_OPENGL_BIT,     2, 0, 110, EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT, },
        { EGL_OPENGL_ES_API,    EGL_OPENGL_ES3_BIT, 3, 0, 300, },
        { EGL_OPENGL_ES_API,    EGL_OPENGL_ES2_BIT, 2, 0, 100, },
    };

    pl_gpu_caps last_caps = 0;
    struct pl_glsl_desc last_glsl = {0};
    struct pl_gpu_limits last_limits = {0};

    struct pl_context *ctx = pl_test_context();

    for (int i = 0; i < PL_ARRAY_SIZE(egl_vers); i++) {

        const int cfg_attribs[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, egl_vers[i].render,
            EGL_NONE
        };

        EGLConfig config = 0;
        EGLint num_configs = 0;
        bool ok = eglChooseConfig(dpy, cfg_attribs, &config, 1, &num_configs);
        if (!ok || !num_configs)
            goto error;

        if (!eglBindAPI(egl_vers[i].api))
            goto error;

        EGLContext egl;
        if (egl_vers[i].api == EGL_OPENGL_ES_API) {
            // OpenGL ES
            const EGLint egl_attribs[] = {
                EGL_CONTEXT_CLIENT_VERSION, egl_vers[i].major,
                (egl_ver >= 15) ? EGL_CONTEXT_OPENGL_DEBUG : EGL_NONE, EGL_TRUE,
                EGL_NONE
            };

            printf("Attempting creation of OpenGL ES v%d context\n", egl_vers[i].major);
            egl = eglCreateContext(dpy, config, EGL_NO_CONTEXT, egl_attribs);
        } else {
            // Desktop OpenGL
            const int egl_attribs[] = {
                EGL_CONTEXT_MAJOR_VERSION, egl_vers[i].major,
                EGL_CONTEXT_MINOR_VERSION, egl_vers[i].minor,
                EGL_CONTEXT_OPENGL_PROFILE_MASK, egl_vers[i].profile,
                (egl_ver >= 15) ? EGL_CONTEXT_OPENGL_DEBUG : EGL_NONE, EGL_TRUE,
                EGL_NONE
            };

            printf("Attempting creation of Desktop OpenGL v%d.%d context\n",
                   egl_vers[i].major, egl_vers[i].minor);
            egl = eglCreateContext(dpy, config, EGL_NO_CONTEXT, egl_attribs);
        }

        if (!egl)
            goto error;

        const EGLint pbuffer_attribs[] = {
            EGL_WIDTH, PBUFFER_WIDTH,
            EGL_HEIGHT, PBUFFER_HEIGHT,
            EGL_NONE
        };

        EGLSurface surf = eglCreatePbufferSurface(dpy, config, pbuffer_attribs);

        if (!eglMakeCurrent(dpy, surf, surf, egl))
            goto error;

        struct pl_opengl_params params = pl_opengl_default_params;
        params.max_glsl_version = egl_vers[i].glsl_ver;
        params.debug = true;
        params.egl_display = dpy;
        params.egl_context = egl;

#ifdef CI_BLACKLIST_COMPUTE
        params.blacklist_caps = PL_GPU_CAP_COMPUTE;
#endif
#ifdef CI_ALLOW_SW
        params.allow_software = true;
#endif

        const struct pl_opengl *gl = pl_opengl_create(ctx, &params);
        if (!gl)
            goto next;

        // Skip repeat tests
        const struct pl_gpu *gpu = gl->gpu;
        if (last_caps == gpu->caps &&
            memcmp(&last_glsl, &gpu->glsl, sizeof(last_glsl)) == 0 &&
            memcmp(&last_limits, &gpu->limits, sizeof(last_limits)) == 0)
        {
            printf("Skipping tests due to duplicate capabilities/version\n");
            goto next;
        }

        last_caps = gpu->caps;
        last_glsl = gpu->glsl;
        last_limits = gpu->limits;

        gpu_tests(gpu);
        opengl_interop_tests(gpu);
        opengl_swapchain_tests(gl, dpy, surf);
        opengl_test_export_import(gl, PL_HANDLE_DMA_BUF);

        // Reduce log spam after first successful test
        pl_test_set_verbosity(ctx, PL_LOG_INFO);

next:
        pl_opengl_destroy(&gl);
        eglDestroySurface(dpy, surf);
        eglDestroyContext(dpy, egl);
        continue;

error: ;
        EGLint error = eglGetError();
        if (error != EGL_SUCCESS)
            fprintf(stderr, "EGL error: %s\n", egl_err_str(error));
    }

    eglTerminate(dpy);
    pl_context_destroy(&ctx);

    if (!last_glsl.version)
        return SKIP;
}

#endif // EPOXY_HAS_EGL
