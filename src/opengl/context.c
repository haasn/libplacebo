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
#include "gpu.h"

const struct pl_opengl_params pl_opengl_default_params = {0};

struct priv {
    struct pl_context *ctx;
    bool is_debug;
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

void pl_opengl_destroy(const struct pl_opengl **ptr)
{
    const struct pl_opengl *pl_gl = *ptr;
    if (!pl_gl)
        return;

    struct priv *p = TA_PRIV(pl_gl);
    if (p->is_debug)
        glDebugMessageCallback(NULL, NULL);

    pl_gpu_destroy(pl_gl->gpu);
    TA_FREEP((void **) ptr);
}

const struct pl_opengl *pl_opengl_create(struct pl_context *ctx,
                                         const struct pl_opengl_params *params)
{
    params = PL_DEF(params, &pl_opengl_default_params);
    struct pl_opengl *pl_gl = talloc_zero_priv(NULL, struct pl_opengl, struct priv);
    struct priv *p = TA_PRIV(pl_gl);
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

    if (params->debug) {
        if (epoxy_has_gl_extension("GL_ARB_debug_output")) {
            glDebugMessageCallback(debug_cb, ctx);
            p->is_debug = true;
        } else {
            PL_WARN(p, "OpenGL debugging requested but GL_ARB_debug_output "
                    "unavailable.. ignoring!");
        }
    }

    pl_gl->gpu = pl_gpu_create_gl(ctx);
    if (!pl_gl->gpu)
        goto error;

    return pl_gl;

error:
    PL_FATAL(p, "Failed initializing opengl context");
    pl_opengl_destroy((const struct pl_opengl **) &pl_gl);
    return NULL;
}
