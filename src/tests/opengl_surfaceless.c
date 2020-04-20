#include "gpu_tests.h"

#include <epoxy/gl.h>
#include <epoxy/egl.h>

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

    struct {
        EGLenum api;
        EGLenum render;
        int major, minor;
        EGLenum profile;
    } egl_vers[] = {
        { EGL_OPENGL_API,       EGL_OPENGL_BIT,     4, 6, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT },
        { EGL_OPENGL_API,       EGL_OPENGL_BIT,     4, 5, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT },
        { EGL_OPENGL_API,       EGL_OPENGL_BIT,     4, 4, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT },
        { EGL_OPENGL_API,       EGL_OPENGL_BIT,     4, 0, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT },
        { EGL_OPENGL_API,       EGL_OPENGL_BIT,     3, 3, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT },
        { EGL_OPENGL_API,       EGL_OPENGL_BIT,     3, 2, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT },
        { EGL_OPENGL_API,       EGL_OPENGL_BIT,     3, 1, EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT, },
        { EGL_OPENGL_API,       EGL_OPENGL_BIT,     3, 0, EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT, },
        { EGL_OPENGL_API,       EGL_OPENGL_BIT,     2, 1, EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT, },
        { EGL_OPENGL_ES_API,    EGL_OPENGL_ES3_BIT, 3, 0, },
        { EGL_OPENGL_ES_API,    EGL_OPENGL_ES2_BIT, 2, 0, },
    };

    pl_gpu_caps last_caps = 0;
    struct pl_glsl_desc last_glsl = {0};
    struct pl_gpu_limits last_limits = {0};

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
            continue;

        if (!eglBindAPI(egl_vers[i].api))
            continue;

        EGLContext egl;
        if (egl_vers[i].api == EGL_OPENGL_ES_API) {
            // OpenGL ES
            const int egl_attribs[] = {
                EGL_CONTEXT_OPENGL_DEBUG, EGL_TRUE,
                EGL_CONTEXT_CLIENT_VERSION, egl_vers[i].major,
                EGL_NONE
            };

            printf("Attempting creation of OpenGL ES v%d context\n", egl_vers[i].major);
            egl = eglCreateContext(dpy, config, EGL_NO_CONTEXT, egl_attribs);
        } else {
            // Desktop OpenGL
            const int egl_attribs[] = {
                EGL_CONTEXT_OPENGL_DEBUG, EGL_TRUE,
                EGL_CONTEXT_MAJOR_VERSION, egl_vers[i].major,
                EGL_CONTEXT_MINOR_VERSION, egl_vers[i].minor,
                EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT,
                EGL_NONE
            };

            printf("Attempting creation of Desktop OpenGL v%d.%d context\n",
                   egl_vers[i].major, egl_vers[i].minor);
            egl = eglCreateContext(dpy, config, EGL_NO_CONTEXT, egl_attribs);
        }

        if (!eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, egl))
            continue;

        struct pl_context *ctx = pl_test_context();
        struct pl_opengl_params params = pl_opengl_default_params;
        params.debug = true;

        const struct pl_opengl *gl = pl_opengl_create(ctx, &params);
        REQUIRE(gl);

        const struct pl_gpu *gpu = gl->gpu;

        // Skip repeat tests
        if (last_caps == gpu->caps &&
            memcmp(&last_glsl, &gpu->glsl, sizeof(last_glsl)) == 0 &&
            memcmp(&last_limits, &gpu->limits, sizeof(last_limits)) == 0)
        {
            printf("Skipping tests due to duplicate capabilities/version\n");
            continue;
        }

        last_caps = gpu->caps;
        last_glsl = gpu->glsl;
        last_limits = gpu->limits;

        gpu_tests(gpu);

        pl_opengl_destroy(&gl);
        pl_context_destroy(&ctx);

        eglDestroyContext(dpy, egl);
    }

    eglTerminate(dpy);

    if (!last_glsl.version)
        return SKIP;
}
