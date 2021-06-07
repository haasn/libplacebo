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
#include <pthread.h>

extern "C" {
#include "pl_alloc.h"
}

#include <glslang/Include/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include "glslang.h"

#define GLSLANG_VERSION_CHECK(major, minor, patch) \
    (((major) < GLSLANG_VERSION_MAJOR) || ((major) == GLSLANG_VERSION_MAJOR && \
    (((minor) < GLSLANG_VERSION_MINOR) || ((minor) == GLSLANG_VERSION_MINOR && \
     ((patch) <= GLSLANG_VERSION_PATCH)))))

using namespace glslang;

static pthread_mutex_t pl_glslang_mutex = PTHREAD_MUTEX_INITIALIZER;
static int pl_glslang_refcount;

int pl_glslang_version(void)
{
    return (GLSLANG_VERSION_MAJOR & 0xFF) << 24 |
           (GLSLANG_VERSION_MINOR & 0xFF) << 16 |
           (GLSLANG_VERSION_PATCH & 0xFFFF);
}

bool pl_glslang_init(void)
{
    bool ret = true;

    pthread_mutex_lock(&pl_glslang_mutex);
    if (pl_glslang_refcount++ == 0)
        ret = InitializeProcess();
    pthread_mutex_unlock(&pl_glslang_mutex);

    return ret;
}

void pl_glslang_uninit(void)
{
    pthread_mutex_lock(&pl_glslang_mutex);
    if (--pl_glslang_refcount == 0)
        FinalizeProcess();
    pthread_mutex_unlock(&pl_glslang_mutex);
}

extern const TBuiltInResource DefaultTBuiltInResource;

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

// Taken from glslang's examples, which apparently generally bases the choices
// on OpenGL specification limits
const TBuiltInResource DefaultTBuiltInResource = {
    /* .MaxLights = */ 32,
    /* .MaxClipPlanes = */ 6,
    /* .MaxTextureUnits = */ 32,
    /* .MaxTextureCoords = */ 32,
    /* .MaxVertexAttribs = */ 64,
    /* .MaxVertexUniformComponents = */ 4096,
    /* .MaxVaryingFloats = */ 64,
    /* .MaxVertexTextureImageUnits = */ 32,
    /* .MaxCombinedTextureImageUnits = */ 80,
    /* .MaxTextureImageUnits = */ 32,
    /* .MaxFragmentUniformComponents = */ 4096,
    /* .MaxDrawBuffers = */ 32,
    /* .MaxVertexUniformVectors = */ 128,
    /* .MaxVaryingVectors = */ 8,
    /* .MaxFragmentUniformVectors = */ 16,
    /* .MaxVertexOutputVectors = */ 16,
    /* .MaxFragmentInputVectors = */ 15,
    /* .MinProgramTexelOffset = */ -8,
    /* .MaxProgramTexelOffset = */ 7,
    /* .MaxClipDistances = */ 8,
    /* .MaxComputeWorkGroupCountX = */ 65535,
    /* .MaxComputeWorkGroupCountY = */ 65535,
    /* .MaxComputeWorkGroupCountZ = */ 65535,
    /* .MaxComputeWorkGroupSizeX = */ 1024,
    /* .MaxComputeWorkGroupSizeY = */ 1024,
    /* .MaxComputeWorkGroupSizeZ = */ 64,
    /* .MaxComputeUniformComponents = */ 1024,
    /* .MaxComputeTextureImageUnits = */ 16,
    /* .MaxComputeImageUniforms = */ 8,
    /* .MaxComputeAtomicCounters = */ 8,
    /* .MaxComputeAtomicCounterBuffers = */ 1,
    /* .MaxVaryingComponents = */ 60,
    /* .MaxVertexOutputComponents = */ 64,
    /* .MaxGeometryInputComponents = */ 64,
    /* .MaxGeometryOutputComponents = */ 128,
    /* .MaxFragmentInputComponents = */ 128,
    /* .MaxImageUnits = */ 8,
    /* .MaxCombinedImageUnitsAndFragmentOutputs = */ 8,
    /* .MaxCombinedShaderOutputResources = */ 8,
    /* .MaxImageSamples = */ 0,
    /* .MaxVertexImageUniforms = */ 0,
    /* .MaxTessControlImageUniforms = */ 0,
    /* .MaxTessEvaluationImageUniforms = */ 0,
    /* .MaxGeometryImageUniforms = */ 0,
    /* .MaxFragmentImageUniforms = */ 8,
    /* .MaxCombinedImageUniforms = */ 8,
    /* .MaxGeometryTextureImageUnits = */ 16,
    /* .MaxGeometryOutputVertices = */ 256,
    /* .MaxGeometryTotalOutputComponents = */ 1024,
    /* .MaxGeometryUniformComponents = */ 1024,
    /* .MaxGeometryVaryingComponents = */ 64,
    /* .MaxTessControlInputComponents = */ 128,
    /* .MaxTessControlOutputComponents = */ 128,
    /* .MaxTessControlTextureImageUnits = */ 16,
    /* .MaxTessControlUniformComponents = */ 1024,
    /* .MaxTessControlTotalOutputComponents = */ 4096,
    /* .MaxTessEvaluationInputComponents = */ 128,
    /* .MaxTessEvaluationOutputComponents = */ 128,
    /* .MaxTessEvaluationTextureImageUnits = */ 16,
    /* .MaxTessEvaluationUniformComponents = */ 1024,
    /* .MaxTessPatchComponents = */ 120,
    /* .MaxPatchVertices = */ 32,
    /* .MaxTessGenLevel = */ 64,
    /* .MaxViewports = */ 16,
    /* .MaxVertexAtomicCounters = */ 0,
    /* .MaxTessControlAtomicCounters = */ 0,
    /* .MaxTessEvaluationAtomicCounters = */ 0,
    /* .MaxGeometryAtomicCounters = */ 0,
    /* .MaxFragmentAtomicCounters = */ 8,
    /* .MaxCombinedAtomicCounters = */ 8,
    /* .MaxAtomicCounterBindings = */ 1,
    /* .MaxVertexAtomicCounterBuffers = */ 0,
    /* .MaxTessControlAtomicCounterBuffers = */ 0,
    /* .MaxTessEvaluationAtomicCounterBuffers = */ 0,
    /* .MaxGeometryAtomicCounterBuffers = */ 0,
    /* .MaxFragmentAtomicCounterBuffers = */ 1,
    /* .MaxCombinedAtomicCounterBuffers = */ 1,
    /* .MaxAtomicCounterBufferSize = */ 16384,
    /* .MaxTransformFeedbackBuffers = */ 4,
    /* .MaxTransformFeedbackInterleavedComponents = */ 64,
    /* .MaxCullDistances = */ 8,
    /* .MaxCombinedClipAndCullDistances = */ 8,
    /* .MaxSamples = */ 4,
#if GLSLANG_VERSION_CHECK(0, 0, 2892)
    /* .maxMeshOutputVerticesNV = */ 256,
    /* .maxMeshOutputPrimitivesNV = */ 512,
    /* .maxMeshWorkGroupSizeX_NV = */ 32,
    /* .maxMeshWorkGroupSizeY_NV = */ 1,
    /* .maxMeshWorkGroupSizeZ_NV = */ 1,
    /* .maxTaskWorkGroupSizeX_NV = */ 32,
    /* .maxTaskWorkGroupSizeY_NV = */ 1,
    /* .maxTaskWorkGroupSizeZ_NV = */ 1,
    /* .maxMeshViewCountNV = */ 4,
#endif
#if GLSLANG_VERSION_CHECK(0, 0, 3763)
    /* .maxDualSourceDrawBuffersEXT = */ 1,
#endif

    /* .limits = */ {
        /* .nonInductiveForLoops = */ 1,
        /* .whileLoops = */ 1,
        /* .doWhileLoops = */ 1,
        /* .generalUniformIndexing = */ 1,
        /* .generalAttributeMatrixVectorIndexing = */ 1,
        /* .generalVaryingIndexing = */ 1,
        /* .generalSamplerIndexing = */ 1,
        /* .generalVariableIndexing = */ 1,
        /* .generalConstantMatrixVectorIndexing = */ 1,
    }
};
