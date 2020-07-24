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

bool gl_check_err(const struct pl_gpu *gpu, const char *fun)
{
    struct pl_gl *gl = TA_PRIV(gpu);
    bool ret = true;

    while (true) {
        GLenum error = glGetError();
        if (error == GL_NO_ERROR)
            return ret;
        PL_ERR(gpu, "%s: OpenGL error: %s", fun, gl_err_str(error));
        ret = false;
        gl->failed = true;
    }
}

bool egl_check_err(const struct pl_gpu *gpu, const char *fun)
{
    struct pl_gl *gl = TA_PRIV(gpu);
    bool ret = true;

    while (true) {
        GLenum error = eglGetError();
        if (error == EGL_SUCCESS)
            return ret;
        PL_ERR(gpu, "%s: EGL error: %s", fun, egl_err_str(error));
        ret = false;
        gl->failed = true;
    }
}

bool gl_is_software(void)
{
    const char *renderer = glGetString(GL_RENDERER);
    const char *vendor = glGetString(GL_VENDOR);
    return !(renderer && vendor) ||
           strcmp(renderer, "Software Rasterizer") == 0 ||
           strstr(renderer, "llvmpipe") ||
           strstr(renderer, "softpipe") ||
           strcmp(vendor, "Microsoft Corporation") == 0 ||
           strcmp(renderer, "Mesa X11") == 0 ||
           strcmp(renderer, "Apple Software Renderer") == 0;
}
