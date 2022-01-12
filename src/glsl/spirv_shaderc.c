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
#include "utils.h"

const struct spirv_compiler_impl pl_spirv_shaderc;

struct priv {
    shaderc_compiler_t compiler;
};

static void shaderc_destroy(struct spirv_compiler *spirv)
{
    struct priv *p = PL_PRIV(spirv);
    shaderc_compiler_release(p->compiler);
    pl_free(spirv);
}

static struct spirv_compiler *shaderc_create(pl_log log)
{
    struct spirv_compiler *spirv = pl_alloc_obj(NULL, spirv, shaderc_compiler_t);
    *spirv = (struct spirv_compiler) {
        .signature = pl_str0_hash(pl_spirv_shaderc.name),
        .impl = &pl_spirv_shaderc,
        .log = log,
    };

    struct priv *p = PL_PRIV(spirv);
    p->compiler = shaderc_compiler_initialize();
    if (!p->compiler)
        goto error;

    unsigned int ver = 0, rev = 0;
    shaderc_get_spv_version(&ver, &rev);
    pl_info(log, "shaderc SPIR-V version %u.%u rev %u",
            ver >> 16, (ver >> 8) & 0xff, rev);

    pl_hash_merge(&spirv->signature, (uint64_t) ver << 32 | rev);
    return spirv;

error:
    shaderc_destroy(spirv);
    return NULL;
}

static pl_str shaderc_compile(struct spirv_compiler *spirv, void *alloc,
                              const struct pl_glsl_version *glsl,
                              enum glsl_shader_stage stage,
                              const char *shader)
{
    struct priv *p = PL_PRIV(spirv);

    shaderc_compile_options_t opts = shaderc_compile_options_initialize();
    if (!opts)
        return (pl_str) {0};

    struct pl_spirv_version spirv_ver = pl_glsl_spv_version(glsl);
    shaderc_compile_options_set_optimization_level(opts,
            shaderc_optimization_level_performance);
    shaderc_compile_options_set_target_spirv(opts, spirv_ver.spv_version);
    shaderc_compile_options_set_target_env(opts,
            spirv_ver.vulkan ? shaderc_target_env_vulkan
                             : shaderc_target_env_opengl,
            spirv_ver.env_version);

    for (int i = 0; i < 3; i++) {
        shaderc_compile_options_set_limit(opts,
                shaderc_limit_max_compute_work_group_size_x + i,
                glsl->max_group_size[i]);
    }

    shaderc_compile_options_set_limit(opts,
            shaderc_limit_min_program_texel_offset,
            glsl->min_gather_offset);
    shaderc_compile_options_set_limit(opts,
            shaderc_limit_max_program_texel_offset,
            glsl->max_gather_offset);

    static const shaderc_shader_kind kinds[] = {
        [GLSL_SHADER_VERTEX]   = shaderc_glsl_vertex_shader,
        [GLSL_SHADER_FRAGMENT] = shaderc_glsl_fragment_shader,
        [GLSL_SHADER_COMPUTE]  = shaderc_glsl_compute_shader,
    };

    shaderc_compilation_result_t res;
    res = shaderc_compile_into_spv(p->compiler, shader, strlen(shader),
                                   kinds[stage], "input", "main", opts);

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

    pl_str ret = {0};
    if (success) {
        void *bytes = (void *) shaderc_result_get_bytes(res);
        pl_assert(bytes);
        ret.len = shaderc_result_get_length(res);
        ret.buf = pl_memdup(alloc, bytes, ret.len);
    }

    shaderc_result_release(res);
    shaderc_compile_options_release(opts);
    return ret;
}

const struct spirv_compiler_impl pl_spirv_shaderc = {
    .name       = "shaderc",
    .destroy    = shaderc_destroy,
    .create     = shaderc_create,
    .compile    = shaderc_compile,
};
