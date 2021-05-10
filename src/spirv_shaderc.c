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

#include <stdlib.h>
#include <shaderc/shaderc.h>

#include "spirv.h"

struct priv {
    shaderc_compiler_t compiler;
    shaderc_compile_options_t opts;
};

static void shaderc_destroy(struct spirv_compiler *spirv)
{
    struct priv *p = PL_PRIV(spirv);
    shaderc_compile_options_release(p->opts);
    shaderc_compiler_release(p->compiler);
    pl_free(spirv);
}

static struct spirv_compiler *shaderc_create(pl_log log, uint32_t api_version)
{
    struct spirv_compiler *spirv = pl_alloc_obj(NULL, spirv, struct priv);
    struct priv *p = PL_PRIV(spirv);
    *spirv = (struct spirv_compiler) {0};

    p->compiler = shaderc_compiler_initialize();
    if (!p->compiler)
        goto error;

    p->opts = shaderc_compile_options_initialize();
    if (!p->opts)
        goto error;

    shaderc_compile_options_set_optimization_level(p->opts,
            shaderc_optimization_level_performance);
    shaderc_compile_options_set_target_env(p->opts,
            shaderc_target_env_vulkan,
            api_version);

    unsigned int ver = 0, rev = 0;
    shaderc_get_spv_version(&ver, &rev);
    pl_info(log, "shaderc SPIR-V version %u.%u rev %u",
            ver >> 16, (ver >> 8) & 0xff, rev);

    spirv->compiler_version = ver * 100 + rev;
    spirv->glsl = (struct pl_glsl_desc) {
        .version = 450, // this is impossible to query, so hard-code it
        .vulkan  = true,
    };
    return spirv;

error:
    shaderc_destroy(spirv);
    return NULL;
}

static shaderc_compilation_result_t compile(struct priv *p,
                                            enum glsl_shader_stage type,
                                            const char *glsl, bool debug)
{
    static const shaderc_shader_kind kinds[] = {
        [GLSL_SHADER_VERTEX]   = shaderc_glsl_vertex_shader,
        [GLSL_SHADER_FRAGMENT] = shaderc_glsl_fragment_shader,
        [GLSL_SHADER_COMPUTE]  = shaderc_glsl_compute_shader,
    };

    if (debug) {
        return shaderc_compile_into_spv_assembly(p->compiler, glsl, strlen(glsl),
                                        kinds[type], "input", "main", p->opts);
    } else {
        return shaderc_compile_into_spv(p->compiler, glsl, strlen(glsl),
                                        kinds[type], "input", "main", p->opts);
    }
}

static bool shaderc_compile(struct spirv_compiler *spirv, void *alloc,
                            enum glsl_shader_stage type, const char *glsl,
                            pl_str *out_spirv)
{
    struct priv *p = PL_PRIV(spirv);

    shaderc_compilation_result_t res = compile(p, type, glsl, false);
    int errs = shaderc_result_get_num_errors(res),
        warn = shaderc_result_get_num_warnings(res);

    enum pl_log_level lev = errs ? PL_LOG_ERR : warn ? PL_LOG_INFO : PL_LOG_DEBUG;

    int s = shaderc_result_get_compilation_status(res);
    bool success = s == shaderc_compilation_status_success;
    if (!success)
        lev = PL_LOG_ERR;

    const char *msg = shaderc_result_get_error_message(res);
    if (msg[0])
        PL_MSG(spirv, lev, "shaderc output:\n%s", msg);

    static const char *results[] = {
        [shaderc_compilation_status_success]            = "success",
        [shaderc_compilation_status_invalid_stage]      = "invalid stage",
        [shaderc_compilation_status_compilation_error]  = "error",
        [shaderc_compilation_status_internal_error]     = "internal error",
        [shaderc_compilation_status_null_result_object] = "no result",
        [shaderc_compilation_status_invalid_assembly]   = "invalid assembly",
    };

    const char *status = s < PL_ARRAY_SIZE(results) ? results[s] : "unknown";
    PL_MSG(spirv, lev, "shaderc compile status '%s' (%d errors, %d warnings)",
           status, errs, warn);

    if (success) {
        void *bytes = (void *) shaderc_result_get_bytes(res);
        pl_assert(bytes);
        out_spirv->len = shaderc_result_get_length(res);
        out_spirv->buf = pl_memdup(alloc, bytes, out_spirv->len);
    }

    // Also print SPIR-V disassembly for debugging purposes. Unfortunately
    // there doesn't seem to be a way to get this except compiling the shader
    // a second time..
    if (pl_msg_test(spirv->log, PL_LOG_TRACE)) {
        shaderc_compilation_result_t dis = compile(p, type, glsl, true);
        PL_TRACE(spirv, "Generated SPIR-V:\n%.*s",
                 (int) shaderc_result_get_length(dis),
                 shaderc_result_get_bytes(dis));
        shaderc_result_release(dis);
    }

    shaderc_result_release(res);
    return success;
}

const struct spirv_compiler_fns pl_spirv_shaderc = {
    .name = "shaderc",
    .compile_glsl = shaderc_compile,
    .create = shaderc_create,
    .destroy = shaderc_destroy,
};
