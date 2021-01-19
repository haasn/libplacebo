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
 * License along with libplacebo.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "common.h"
#include "utils.h"
#include "gpu.h"

const struct pl_opengl_params pl_opengl_default_params = {0};

struct priv {
    struct pl_context *ctx;
    bool is_debug;
    bool is_debug_egl;
};

static void GLAPIENTRY debug_cb(GLenum source, GLenum type, GLuint id,
                                GLenum severity, GLsizei length,
                                const GLchar *message, const void *userParam)
{
    struct pl_context *ctx = (void *) userParam;
    enum pl_log_level level = PL_LOG_ERR;

    switch (severity) {
    case GL_DEBUG_SEVERITY_NOTIFICATION:level = PL_LOG_DEBUG; break;
    case GL_DEBUG_SEVERITY_LOW:         level = PL_LOG_INFO; break;
    case GL_DEBUG_SEVERITY_MEDIUM:      level = PL_LOG_WARN; break;
    case GL_DEBUG_SEVERITY_HIGH:        level = PL_LOG_ERR; break;
    }

    pl_msg(ctx, level, "GL: %s", message);
}

#ifdef EPOXY_HAS_EGL

static void debug_cb_egl(EGLenum error, const char *command,
                         EGLint messageType, EGLLabelKHR threadLabel,
                         EGLLabelKHR objectLabel, const char *message)
{
    struct pl_context *ctx = threadLabel;
    if (!ctx)
        return;

    enum pl_log_level level = PL_LOG_ERR;
    switch (messageType) {
    case EGL_DEBUG_MSG_CRITICAL_KHR:    level = PL_LOG_FATAL; break;
    case EGL_DEBUG_MSG_ERROR_KHR:       level = PL_LOG_ERR; break;
    case EGL_DEBUG_MSG_WARN_KHR:        level = PL_LOG_WARN; break;
    case EGL_DEBUG_MSG_INFO_KHR:        level = PL_LOG_DEBUG; break;
    }

    pl_msg(ctx, level, "EGL: %s: %s %s", command, egl_err_str(error),
           message);
}

#endif // EPOXY_HAS_EGL

void pl_opengl_destroy(const struct pl_opengl **ptr)
{
    const struct pl_opengl *pl_gl = *ptr;
    if (!pl_gl)
        return;

    struct priv *p = PL_PRIV(pl_gl);
    if (p->is_debug)
        glDebugMessageCallback(NULL, NULL);

#ifdef EPOXY_HAS_EGL
    if (p->is_debug_egl)
        eglDebugMessageControlKHR(NULL, NULL);
#endif

    pl_gpu_destroy(pl_gl->gpu);
    pl_free_ptr((void **) ptr);
}

const struct pl_opengl *pl_opengl_create(struct pl_context *ctx,
                                         const struct pl_opengl_params *params)
{
    params = PL_DEF(params, &pl_opengl_default_params);
    struct pl_opengl *pl_gl = pl_zalloc_priv(NULL, struct pl_opengl, struct priv);
    struct priv *p = PL_PRIV(pl_gl);
    p->ctx = ctx;

    int ver = epoxy_gl_version();
    if (!ver) {
        PL_FATAL(p, "No OpenGL version detected - make sure an OpenGL context "
                 "is bound to the current thread!");
        goto error;
    }

    PL_INFO(p, "Detected OpenGL version strings:");
    PL_INFO(p, "    GL_VERSION:  %s", glGetString(GL_VERSION));
    PL_INFO(p, "    GL_VENDOR:   %s", glGetString(GL_VENDOR));
    PL_INFO(p, "    GL_RENDERER: %s", glGetString(GL_RENDERER));

    if (pl_msg_test(ctx, PL_LOG_DEBUG)) {
        if (ver >= 30) {
            int num_exts = 0;
            glGetIntegerv(GL_NUM_EXTENSIONS, &num_exts);
            PL_DEBUG(p, "    GL_EXTENSIONS:");
            for (int i = 0; i < num_exts; i++) {
                const char *ext = glGetStringi(GL_EXTENSIONS, i);
                PL_DEBUG(p, "        %s", ext);
            }
        } else {
            PL_DEBUG(p, "    GL_EXTENSIONS: %s", glGetString(GL_EXTENSIONS));
        }
    }

    if (!params->allow_software && gl_is_software()) {
        PL_FATAL(p, "OpenGL context is suspected to be a software rasterizer, "
                 "but `allow_software` is false.");
        goto error;
    }

    if (params->debug) {
        if (epoxy_has_gl_extension("GL_ARB_debug_output")) {
            glDebugMessageCallback(debug_cb, ctx);
            p->is_debug = true;
        } else {
            PL_WARN(p, "OpenGL debugging requested but GL_ARB_debug_output "
                    "unavailable.. ignoring!");
        }

#ifdef EPOXY_HAS_EGL
        if (params->egl_display && epoxy_has_egl_extension(params->egl_display, "EGL_KHR_debug")) {
            static const EGLAttrib attribs[] = {
                // Enable everything under the sun, because the `pl_ctx` log
                // level may change at runtime.
                EGL_DEBUG_MSG_CRITICAL_KHR, EGL_TRUE,
                EGL_DEBUG_MSG_ERROR_KHR,    EGL_TRUE,
                EGL_DEBUG_MSG_WARN_KHR,     EGL_TRUE,
                EGL_DEBUG_MSG_INFO_KHR,     EGL_TRUE,
                EGL_NONE,
            };

            eglDebugMessageControlKHR(debug_cb_egl, attribs);
            eglLabelObjectKHR(NULL, EGL_OBJECT_THREAD_KHR, NULL, ctx);
            p->is_debug_egl = true;
        }
#endif // EPOXY_HAS_EGL
    }

    pl_gl->gpu = pl_gpu_create_gl(ctx, params);
    if (!pl_gl->gpu)
        goto error;

    // Restrict caps/version
    if (params->blacklist_caps) {
        pl_gpu_caps *caps = (pl_gpu_caps*) &pl_gl->gpu->caps;
        *caps &= ~(params->blacklist_caps);
        PL_INFO(p, "Restricting capabilities 0x%x... new caps are 0x%x",
                (unsigned int) params->blacklist_caps, (unsigned int) *caps);
    }

    if (params->max_glsl_version) {
        struct pl_glsl_desc *desc = (struct pl_glsl_desc *) &pl_gl->gpu->glsl;
        desc->version = PL_MIN(desc->version, params->max_glsl_version);
        PL_INFO(p, "Restricting GLSL version to %d... new version is %d",
                params->max_glsl_version, desc->version);
    }

    return pl_gl;

error:
    PL_FATAL(p, "Failed initializing opengl context!");
    pl_opengl_destroy((const struct pl_opengl **) &pl_gl);
    return NULL;
}
