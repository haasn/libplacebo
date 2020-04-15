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

#include <pthread.h>

#include "common.h"
#include "gpu.h"

const struct pl_opengl_params pl_opengl_default_params = {0};

struct priv {
    struct pl_context *ctx;
    bool is_debug;
};

// OpenGL can't destroy allocators, so we need to use global state to make sure
// we properly clean up after ourselves to avoid logging to contexts after they
// stopped existing
static pthread_mutex_t debug_ctx_mutex = PTHREAD_MUTEX_INITIALIZER;

static bool debug_cb_set;
static struct pl_context *debug_ctx;
static int debug_ctx_refcount;

void pl_opengl_destroy(const struct pl_opengl **ptr)
{
    const struct pl_opengl *pl_gl = *ptr;
    if (!pl_gl)
        return;

    struct priv *p = TA_PRIV(pl_gl);
    if (p->is_debug) {
        pthread_mutex_lock(&debug_ctx_mutex);
        if (--debug_ctx_refcount == 0)
            debug_ctx = NULL;
        pthread_mutex_unlock(&debug_ctx_mutex);
    }

    pl_gpu_destroy(pl_gl->gpu);
    TA_FREEP((void **) ptr);
}

static void GLAPIENTRY debug_cb(GLenum source, GLenum type, GLuint id,
                                GLenum severity, GLsizei length,
                                const GLchar *message, const void *userParam)
{
    pthread_mutex_lock(&debug_ctx_mutex);
    if (!debug_ctx)
        goto done;

    enum pl_log_level level = PL_LOG_ERR;
    switch (severity) {
    case GL_DEBUG_SEVERITY_NOTIFICATION:level = PL_LOG_DEBUG; break;
    case GL_DEBUG_SEVERITY_LOW:         level = PL_LOG_INFO; break;
    case GL_DEBUG_SEVERITY_MEDIUM:      level = PL_LOG_WARN; break;
    case GL_DEBUG_SEVERITY_HIGH:        level = PL_LOG_ERR; break;
    }

    pl_msg(debug_ctx, level, "GL: %s", message);

done:
    pthread_mutex_unlock(&debug_ctx_mutex);
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
            pthread_mutex_lock(&debug_ctx_mutex);
            if (debug_ctx && debug_ctx != ctx) {
                PL_WARN(p, "Tried creating multiple `pl_opengl` objects with "
                        "debugging enabled on different `pl_context`, this is "
                        "not supported.. ignoring!");
            } else {
                debug_ctx = ctx;
                debug_ctx_refcount++;
                p->is_debug = true;

                if (!debug_cb_set)
                    glDebugMessageCallback(debug_cb, NULL);
            }
            pthread_mutex_unlock(&debug_ctx_mutex);
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
