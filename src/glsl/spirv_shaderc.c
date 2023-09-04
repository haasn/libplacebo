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

const struct spirv_compiler pl_spirv_shaderc;

struct priv {
    shaderc_compiler_t compiler;
};

static void shaderc_destroy(pl_spirv spirv)
{
    struct priv *p = PL_PRIV(spirv);
    shaderc_compiler_release(p->compiler);
    pl_free((void *) spirv);
}

static pl_spirv shaderc_create(pl_log log, struct pl_spirv_version spirv_ver)
{
    struct pl_spirv_t *spirv = pl_alloc_obj(NULL, spirv, struct priv);
    *spirv = (struct pl_spirv_t) {
        .signature = pl_str0_hash(pl_spirv_shaderc.name),
        .impl      = &pl_spirv_shaderc,
        .version   = spirv_ver,
        .log       = log,
    };

    struct priv *p = PL_PRIV(spirv);
    p->compiler = shaderc_compiler_initialize();
    if (!p->compiler)
        goto error;

    unsigned int ver = 0, rev = 0;
    shaderc_get_spv_version(&ver, &rev);
    PL_INFO(spirv, "shaderc SPIR-V version %u.%u rev %u",
            ver >> 16, (ver >> 8) & 0xff, rev);

    // Clamp to supported version by shaderc
    if (ver < spirv->version.spv_version) {
        spirv->version.spv_version = ver;
        spirv->version.env_version = pl_spirv_version_to_vulkan(ver);
    }

    pl_hash_merge(&spirv->signature, (uint64_t) spirv->version.spv_version << 32 |
                                                spirv->version.env_version);
    pl_hash_merge(&spirv->signature, (uint64_t) ver << 32 | rev);
    return spirv;

error:
    shaderc_destroy(spirv);
    return NULL;
}

static pl_str shaderc_compile(pl_spirv spirv, void *alloc,
                              struct pl_glsl_version glsl_ver,
                              enum glsl_shader_stage stage,
                              const char *shader)
{
    struct priv *p = PL_PRIV(spirv);
    const size_t len = strlen(shader);

    shaderc_compile_options_t opts = shaderc_compile_options_initialize();
    if (!opts)
        return (pl_str) {0};

    shaderc_compile_options_set_optimization_level(opts,
            shaderc_optimization_level_performance);
    shaderc_compile_options_set_target_spirv(opts, spirv->version.spv_version);
    shaderc_compile_options_set_target_env(opts, shaderc_target_env_vulkan,
                                                 spirv->version.env_version);

    for (int i = 0; i < 3; i++) {
        shaderc_compile_options_set_limit(opts,
                shaderc_limit_max_compute_work_group_size_x + i,
                glsl_ver.max_group_size[i]);
    }

    shaderc_compile_options_set_limit(opts,
            shaderc_limit_min_program_texel_offset,
            glsl_ver.min_gather_offset);
    shaderc_compile_options_set_limit(opts,
            shaderc_limit_max_program_texel_offset,
            glsl_ver.max_gather_offset);

    static const shaderc_shader_kind kinds[] = {
        [GLSL_SHADER_VERTEX]   = shaderc_glsl_vertex_shader,
        [GLSL_SHADER_FRAGMENT] = shaderc_glsl_fragment_shader,
        [GLSL_SHADER_COMPUTE]  = shaderc_glsl_compute_shader,
    };

    static const char * const file_name = "input";
    static const char * const entry_point = "main";

    shaderc_compilation_result_t res;
    res = shaderc_compile_into_spv(p->compiler, shader, len, kinds[stage],
                                   file_name, entry_point, opts);

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

        if (pl_msg_test(spirv->log, PL_LOG_TRACE)) {
            shaderc_compilation_result_t dis;
            dis = shaderc_compile_into_spv_assembly(p->compiler, shader, len,
                                                    kinds[stage], file_name,
                                                    entry_point, opts);
            PL_TRACE(spirv, "Generated SPIR-V:\n%.*s",
                     (int) shaderc_result_get_length(dis),
                     shaderc_result_get_bytes(dis));
            shaderc_result_release(dis);
        }
    }

    shaderc_result_release(res);
    shaderc_compile_options_release(opts);
    return ret;
}

const struct spirv_compiler pl_spirv_shaderc = {
    .name       = "shaderc",
    .destroy    = shaderc_destroy,
    .create     = shaderc_create,
    .compile    = shaderc_compile,
};
