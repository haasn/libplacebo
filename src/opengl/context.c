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

#include <ctype.h>

#include "common.h"
#include "utils.h"
#include "gpu.h"

const struct pl_opengl_params pl_opengl_default_params = {0};

static void GLAPIENTRY debug_cb(GLenum source, GLenum type, GLuint id,
                                GLenum severity, GLsizei length,
                                const GLchar *message, const void *userParam)
{
    pl_log log = (void *) userParam;
    enum pl_log_level level = PL_LOG_ERR;

    switch (severity) {
    case GL_DEBUG_SEVERITY_NOTIFICATION:level = PL_LOG_DEBUG; break;
    case GL_DEBUG_SEVERITY_LOW:         level = PL_LOG_INFO; break;
    case GL_DEBUG_SEVERITY_MEDIUM:      level = PL_LOG_WARN; break;
    case GL_DEBUG_SEVERITY_HIGH:        level = PL_LOG_ERR; break;
    }

#ifndef MSAN
    pl_msg(log, level, "GL: %s", message);

    if (level <= PL_LOG_ERR)
        pl_log_stack_trace(log, level);
#endif
}

static void GLAPIENTRY debug_cb_egl(EGLenum error, const char *command,
                                    EGLint messageType, EGLLabelKHR threadLabel,
                                    EGLLabelKHR objectLabel, const char *message)
{
    pl_log log = threadLabel;
    enum pl_log_level level = PL_LOG_ERR;

    switch (messageType) {
    case EGL_DEBUG_MSG_CRITICAL_KHR:    level = PL_LOG_FATAL; break;
    case EGL_DEBUG_MSG_ERROR_KHR:       level = PL_LOG_ERR; break;
    case EGL_DEBUG_MSG_WARN_KHR:        level = PL_LOG_WARN; break;
    case EGL_DEBUG_MSG_INFO_KHR:        level = PL_LOG_DEBUG; break;
    }

#ifndef MSAN
    pl_msg(log, level, "EGL: %s: %s %s", command, egl_err_str(error),
           message);

    if (level <= PL_LOG_ERR)
        pl_log_stack_trace(log, level);
#endif
}

#ifdef PL_HAVE_GL_PROC_ADDR
// Guards access to the (thread-unsafe) glad internal loader
static pl_static_mutex glad_loader_mutex = PL_STATIC_MUTEX_INITIALIZER;
#endif

void pl_opengl_destroy(pl_opengl *ptr)
{
    pl_opengl pl_gl = *ptr;
    if (!pl_gl)
        return;

    struct gl_ctx *p = PL_PRIV(pl_gl);
    const gl_funcs *gl = &p->func;
    if (!gl_make_current(pl_gl)) {
        PL_WARN(p, "Failed uninitializing OpenGL context, leaking resources!");
        return;
    }

    if (p->is_debug)
        gl->DebugMessageCallback(NULL, NULL);

    if (p->is_debug_egl)
        eglDebugMessageControlKHR(NULL, NULL);

    pl_gpu_destroy(pl_gl->gpu);

#ifdef PL_HAVE_GL_PROC_ADDR
    if (p->gl_loaded) {
        pl_static_mutex_lock(&glad_loader_mutex);
        if (p->params.egl_display)
            gladLoaderUnloadEGL();
        gladLoaderUnloadGLES2();
        gladLoaderUnloadGL();
        pl_static_mutex_unlock(&glad_loader_mutex);
    }
#endif

    gl_release_current(pl_gl);
    pl_mutex_destroy(&p->lock);
    pl_free_ptr((void **) ptr);

}

typedef PL_ARRAY(const char *) ext_arr_t;
static void add_exts_str(void *alloc, ext_arr_t *arr, const char *extstr)
{
    pl_str rest = pl_str_strip(pl_str0(pl_strdup0(alloc, pl_str0(extstr))));
    while (rest.len) {
        pl_str ext = pl_str_split_char(rest, ' ', &rest);
        ext.buf[ext.len] = '\0'; // re-use separator for terminator
        PL_ARRAY_APPEND(alloc, *arr, (char *) ext.buf);
    }
}

pl_opengl pl_opengl_create(pl_log log, const struct pl_opengl_params *params)
{
    params = PL_DEF(params, &pl_opengl_default_params);
    struct pl_opengl_t *pl_gl = pl_zalloc_obj(NULL, pl_gl, struct gl_ctx);
    struct gl_ctx *p = PL_PRIV(pl_gl);
    gl_funcs *gl = &p->func;
    p->params = *params;
    p->log = log;

    pl_mutex_init_type(&p->lock, PL_MUTEX_RECURSIVE);
    if (!gl_make_current(pl_gl)) {
        pl_free(pl_gl);
        return NULL;
    }

    bool ok = false;
    if (params->get_proc_addr_ex) {
        ok |= gladLoadGLContextUserPtr(gl, params->get_proc_addr_ex, params->proc_ctx);
        ok |= gladLoadGLES2ContextUserPtr(gl, params->get_proc_addr_ex, params->proc_ctx);
    } else if (params->get_proc_addr) {
        ok |= gladLoadGLContext(gl, params->get_proc_addr);
        ok |= gladLoadGLES2Context(gl, params->get_proc_addr);
    } else {
#ifdef PL_HAVE_GL_PROC_ADDR
        pl_static_mutex_lock(&glad_loader_mutex);
        ok |= gladLoaderLoadGLContext(gl);
        ok |= gladLoaderLoadGLES2Context(gl);
        pl_static_mutex_unlock(&glad_loader_mutex);
        p->gl_loaded = true;
#else
        PL_FATAL(p, "No `glGetProcAddress` function provided, and libplacebo "
                 "built without its built-in OpenGL loader!");
        goto error;
#endif
    }

    if (!ok) {
        PL_FATAL(p, "Failed to initialize OpenGL context - make sure a valid "
                 "OpenGL context is bound to the current thread!");
        goto error;
    }

    const char *version = (const char *) gl->GetString(GL_VERSION);
    if (version) {
        const char *ver = version;
        while (!isdigit(*ver) && *ver != '\0')
            ver++;
        if (sscanf(ver, "%d.%d", &pl_gl->major, &pl_gl->minor) != 2) {
            PL_FATAL(p, "Invalid GL_VERSION string: %s\n", version);
            goto error;
        }
    }

    if (!pl_gl->major) {
        PL_FATAL(p, "No OpenGL version detected - make sure an OpenGL context "
                 "is bound to the current thread!");
        goto error;
    }

    PL_INFO(p, "Detected OpenGL version strings:");
    PL_INFO(p, "    GL_VERSION:  %s", version);
    PL_INFO(p, "    GL_VENDOR:   %s", (char *) gl->GetString(GL_VENDOR));
    PL_INFO(p, "    GL_RENDERER: %s", (char *) gl->GetString(GL_RENDERER));

    ext_arr_t exts = {0};
    if (pl_gl->major >= 3) {
        gl->GetIntegerv(GL_NUM_EXTENSIONS, &exts.num);
        PL_ARRAY_RESIZE(pl_gl, exts, exts.num);
        for (int i = 0; i < exts.num; i++)
            exts.elem[i] = (const char *) gl->GetStringi(GL_EXTENSIONS, i);
    } else {
        add_exts_str(pl_gl, &exts, (const char *) gl->GetString(GL_EXTENSIONS));
    }

    if (pl_msg_test(log, PL_LOG_DEBUG)) {
        PL_DEBUG(p, "    GL_EXTENSIONS:");
        for (int i = 0; i < exts.num; i++)
            PL_DEBUG(p, "        %s", exts.elem[i]);
    }

    if (params->egl_display) {
        if (params->get_proc_addr_ex) {
            ok = gladLoadEGLUserPtr(params->egl_display, params->get_proc_addr_ex,
                                    params->proc_ctx);
        } else if (params->get_proc_addr) {
            ok = gladLoadEGL(params->egl_display, params->get_proc_addr);
        } else {
#ifdef PL_HAVE_GL_PROC_ADDR
            pl_static_mutex_lock(&glad_loader_mutex);
            ok = gladLoaderLoadEGL(params->egl_display);
            pl_static_mutex_unlock(&glad_loader_mutex);
#else
            pl_unreachable();
#endif
        }

        if (!ok) {
            PL_FATAL(p, "Failed loading EGL functions - double check EGLDisplay?");
            goto error;
        }

        int start = exts.num;
        add_exts_str(pl_gl, &exts, eglQueryString(params->egl_display,
                                                  EGL_EXTENSIONS));
        if (exts.num > start) {
            PL_DEBUG(p, "    EGL_EXTENSIONS:");
            for (int i = start; i < exts.num; i++)
                PL_DEBUG(p, "        %s", exts.elem[i]);
        }
    }

    pl_gl->extensions = exts.elem;
    pl_gl->num_extensions = exts.num;

    if (!params->allow_software && gl_is_software(pl_gl)) {
        PL_FATAL(p, "OpenGL context is suspected to be a software rasterizer, "
                 "but `allow_software` is false.");
        goto error;
    }

    if (params->debug) {
        if (pl_opengl_has_ext(pl_gl, "GL_KHR_debug")) {
            gl->DebugMessageCallback(debug_cb, log);
            gl->Enable(GL_DEBUG_OUTPUT);
            p->is_debug = true;
        } else {
            PL_WARN(p, "OpenGL debugging requested, but GL_KHR_debug is not "
                    "available... ignoring!");
        }

        if (params->egl_display && pl_opengl_has_ext(pl_gl, "EGL_KHR_debug")) {
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
            eglLabelObjectKHR(NULL, EGL_OBJECT_THREAD_KHR, NULL, (void *) log);
            p->is_debug_egl = true;
        }
    }

    pl_gl->gpu = pl_gpu_create_gl(log, pl_gl, params);
    if (!pl_gl->gpu)
        goto error;

    // Restrict version
    if (params->max_glsl_version) {
        struct pl_glsl_version *glsl = (struct pl_glsl_version *) &pl_gl->gpu->glsl;
        glsl->version = PL_MIN(glsl->version, params->max_glsl_version);
        PL_INFO(p, "Restricting GLSL version to %d... new version is %d",
                params->max_glsl_version, glsl->version);
    }

    gl_release_current(pl_gl);
    return pl_gl;

error:
    PL_FATAL(p, "Failed initializing opengl context!");
    gl_release_current(pl_gl);
    pl_opengl_destroy((pl_opengl *) &pl_gl);
    return NULL;
}

bool gl_make_current(pl_opengl pl_gl)
{
    struct gl_ctx *p = PL_PRIV(pl_gl);
    pl_mutex_lock(&p->lock);
    if (!p->count && p->params.make_current) {
        if (!p->params.make_current(p->params.priv)) {
            PL_ERR(p, "Failed making OpenGL context current on calling thread!");
            pl_mutex_unlock(&p->lock);
            return false;
        }
    }

    p->count++;
    return true;
}

void gl_release_current(pl_opengl pl_gl)
{
    struct gl_ctx *p = PL_PRIV(pl_gl);
    p->count--;
    if (!p->count && p->params.release_current)
        p->params.release_current(p->params.priv);
    pl_mutex_unlock(&p->lock);
}
