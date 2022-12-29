/*
 * This file is part of libplacebo.
 *
 * libplacebo is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * libplacebo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libplacebo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "common.h"
#include "gpu.h"
#include "utils.h"

const char *gl_err_str(GLenum err)
{
    switch (err) {
#define CASE(name) case name: return #name
    CASE(GL_NO_ERROR);
    CASE(GL_INVALID_ENUM);
    CASE(GL_INVALID_VALUE);
    CASE(GL_INVALID_OPERATION);
    CASE(GL_INVALID_FRAMEBUFFER_OPERATION);
    CASE(GL_OUT_OF_MEMORY);
    CASE(GL_STACK_UNDERFLOW);
    CASE(GL_STACK_OVERFLOW);
#undef CASE

    default: return "unknown error";
    }
}

void gl_poll_callbacks(pl_gpu gpu)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    struct pl_gl *p = PL_PRIV(gpu);
    while (p->callbacks.num) {
        struct gl_cb cb = p->callbacks.elem[0];
        GLenum res = gl->ClientWaitSync(cb.sync, 0, 0);
        switch (res) {
        case GL_ALREADY_SIGNALED:
        case GL_CONDITION_SATISFIED:
            PL_ARRAY_REMOVE_AT(p->callbacks, 0);
            cb.callback(cb.priv);
            continue;

        case GL_WAIT_FAILED:
            PL_ARRAY_REMOVE_AT(p->callbacks, 0);
            gl->DeleteSync(cb.sync);
            p->failed = true;
            gl_check_err(gpu, "gl_poll_callbacks"); // NOTE: will recurse!
            return;

        case GL_TIMEOUT_EXPIRED:
            return;

        default:
            pl_unreachable();
        }
    }
}

bool gl_check_err(pl_gpu gpu, const char *fun)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    struct pl_gl *p = PL_PRIV(gpu);
    bool ret = true;

    while (true) {
        GLenum error = gl->GetError();
        if (error == GL_NO_ERROR)
            break;
        PL_ERR(gpu, "%s: OpenGL error: %s", fun, gl_err_str(error));
        ret = false;
        p->failed = true;
    }

    gl_poll_callbacks(gpu);
    return ret;
}

bool gl_is_software(pl_opengl pl_gl)
{
    struct gl_ctx *glctx = PL_PRIV(pl_gl);
    const gl_funcs *gl = &glctx->func;
    const char *renderer = (char *) gl->GetString(GL_RENDERER);
    return !renderer ||
           strcmp(renderer, "Software Rasterizer") == 0 ||
           strstr(renderer, "llvmpipe") ||
           strstr(renderer, "softpipe") ||
           strcmp(renderer, "Mesa X11") == 0 ||
           strcmp(renderer, "Apple Software Renderer") == 0;
}

bool gl_is_gles(pl_opengl pl_gl)
{
    struct gl_ctx *glctx = PL_PRIV(pl_gl);
    const gl_funcs *gl = &glctx->func;
    const char *version = (char *) gl->GetString(GL_VERSION);
    return pl_str_startswith0(pl_str0(version), "OpenGL ES");
}

bool gl_test_ext(pl_gpu gpu, const char *ext, int gl_ver, int gles_ver)
{
    struct pl_gl *p = PL_PRIV(gpu);
    if (gl_ver && p->gl_ver >= gl_ver)
        return true;
    if (gles_ver && p->gles_ver >= gles_ver)
        return true;

    return ext ? pl_opengl_has_ext(p->gl, ext) : false;
}

const char *egl_err_str(EGLenum err)
{
    switch (err) {
#define CASE(name) case name: return #name
    CASE(EGL_SUCCESS);
    CASE(EGL_NOT_INITIALIZED);
    CASE(EGL_BAD_ACCESS);
    CASE(EGL_BAD_ALLOC);
    CASE(EGL_BAD_ATTRIBUTE);
    CASE(EGL_BAD_CONFIG);
    CASE(EGL_BAD_CONTEXT);
    CASE(EGL_BAD_CURRENT_SURFACE);
    CASE(EGL_BAD_DISPLAY);
    CASE(EGL_BAD_MATCH);
    CASE(EGL_BAD_NATIVE_PIXMAP);
    CASE(EGL_BAD_NATIVE_WINDOW);
    CASE(EGL_BAD_PARAMETER);
    CASE(EGL_BAD_SURFACE);
#undef CASE

    default: return "unknown error";
    }
}

bool egl_check_err(pl_gpu gpu, const char *fun)
{
    struct pl_gl *p = PL_PRIV(gpu);
    bool ret = true;

    while (true) {
        GLenum error = eglGetError();
        if (error == EGL_SUCCESS)
            return ret;
        PL_ERR(gpu, "%s: EGL error: %s", fun, egl_err_str(error));
        ret = false;
        p->failed = true;
    }
}
