#include "gpu_tests.h"

#include <X11/Xlib.h>
#include <epoxy/gl.h>
#include <epoxy/glx.h>

int main()
{
    // Create the OpenGL context
    Display *display = XOpenDisplay(NULL);
    if (!display)
        return SKIP;

    static int visualAttribs[] = { None };
    int num_fbconfigs = 0;
    GLXFBConfig* fbconfigs = glXChooseFBConfig(display, DefaultScreen(display),
                                               visualAttribs, &num_fbconfigs);
    if (!fbconfigs) {
        XCloseDisplay(display);
        return SKIP;
    }

    static const int context_attribs[] = {
        GLX_CONTEXT_FLAGS_ARB, GLX_CONTEXT_DEBUG_BIT_ARB,
        GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
        None
    };

    GLXContext glx = glXCreateContextAttribsARB(display, fbconfigs[0], 0,
                                                true, context_attribs);
    if (!glx) {
        XCloseDisplay(display);
        return SKIP;
    }

    static const int pbuffer_attribs[] = {
        GLX_PBUFFER_WIDTH,  32,
        GLX_PBUFFER_HEIGHT, 32,
        None
    };

    GLXPbuffer pbuf = glXCreatePbuffer(display, fbconfigs[0], pbuffer_attribs);
    if (!pbuf) {
        XCloseDisplay(display);
        return SKIP;
    }

    REQUIRE(glXMakeContextCurrent(display, pbuf, pbuf, glx));

    struct pl_context *ctx = pl_test_context();
    struct pl_opengl_params params = pl_opengl_default_params;
    params.debug = true;

    const struct pl_opengl *gl = pl_opengl_create(ctx, &params);
    if (!gl)
        return SKIP;

    const struct pl_gpu *gpu = gl->gpu;
    gpu_tests(gpu);

    pl_opengl_destroy(&gl);
    pl_context_destroy(&ctx);

    glXDestroyPbuffer(display, pbuf);
    XCloseDisplay(display);
}
