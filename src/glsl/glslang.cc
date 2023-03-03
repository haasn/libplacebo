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

#include "config_internal.h"

#include <assert.h>

extern "C" {
#include "pl_alloc.h"
#include "pl_thread.h"
}

#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>
#include <glslang/build_info.h>

#include "glslang.h"

#if (GLSLANG_VERSION_MAJOR * 1000 + GLSLANG_VERSION_MINOR) >= 11013
#include <glslang/Public/ResourceLimits.h>
#define DefaultTBuiltInResource *GetDefaultResources()
#endif

using namespace glslang;

static pl_static_mutex pl_glslang_mutex = PL_STATIC_MUTEX_INITIALIZER;
static int pl_glslang_refcount;

bool pl_glslang_init(void)
{
    bool ret = true;

    pl_static_mutex_lock(&pl_glslang_mutex);
    if (pl_glslang_refcount++ == 0)
        ret = InitializeProcess();
    pl_static_mutex_unlock(&pl_glslang_mutex);

    return ret;
}

void pl_glslang_uninit(void)
{
    pl_static_mutex_lock(&pl_glslang_mutex);
    if (--pl_glslang_refcount == 0)
        FinalizeProcess();
    pl_static_mutex_unlock(&pl_glslang_mutex);
}

struct pl_glslang_res *pl_glslang_compile(const struct pl_glsl_version *glsl,
                                          enum glsl_shader_stage stage,
                                          const char *text)
{
    assert(pl_glslang_refcount);
    struct pl_glslang_res *res = pl_zalloc_ptr(NULL, res);

    EShLanguage lang;
    switch (stage) {
    case GLSL_SHADER_VERTEX:     lang = EShLangVertex; break;
    case GLSL_SHADER_FRAGMENT:   lang = EShLangFragment; break;
    case GLSL_SHADER_COMPUTE:    lang = EShLangCompute; break;
    default: abort();
    }

    TShader *shader = new TShader(lang);

    struct pl_spirv_version spirv_ver = pl_glsl_spv_version(glsl);
    shader->setEnvClient(spirv_ver.vulkan ? EShClientVulkan : EShClientOpenGL,
                         (EShTargetClientVersion) spirv_ver.env_version);
    shader->setEnvTarget(EShTargetSpv, (EShTargetLanguageVersion) spirv_ver.spv_version);
    shader->setStrings(&text, 1);

    TBuiltInResource limits = DefaultTBuiltInResource;
    limits.maxComputeWorkGroupSizeX = glsl->max_group_size[0];
    limits.maxComputeWorkGroupSizeY = glsl->max_group_size[1];
    limits.maxComputeWorkGroupSizeZ = glsl->max_group_size[2];
    limits.minProgramTexelOffset = glsl->min_gather_offset;
    limits.maxProgramTexelOffset = glsl->max_gather_offset;

    if (!shader->parse(&limits, 0, true, EShMsgDefault)) {
        res->error_msg = pl_str0dup0(res, shader->getInfoLog());
        delete shader;
        return res;
    }

    TProgram *prog = new TProgram();
    prog->addShader(shader);
    if (!prog->link(EShMsgDefault)) {
        res->error_msg = pl_str0dup0(res, prog->getInfoLog());
        delete shader;
        delete prog;
        return res;
    }

    std::vector<unsigned int> spirv;
    GlslangToSpv(*prog->getIntermediate(lang), spirv);

    res->success = true;
    res->size = spirv.size() * sizeof(unsigned int);
    res->data = pl_memdup(res, spirv.data(), res->size),
    delete shader;
    delete prog;
    return res;
}
