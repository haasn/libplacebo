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

#include "gpu.h"
#include "glsl/spirv.h"

// For pl_pass.priv
struct pl_pass_vk {
    // Pipeline / render pass
    VkPipeline base;
    VkPipeline pipe;
    VkPipelineLayout pipeLayout;
    VkRenderPass renderPass;
    // Descriptor set (bindings)
    bool use_pushd;
    VkDescriptorSetLayout dsLayout;
    VkDescriptorPool dsPool;
    // To keep track of which descriptor sets are and aren't available, we
    // allocate a fixed number and use a bitmask of all available sets.
    VkDescriptorSet dss[16];
    uint16_t dmask;

    // For recompilation
    VkVertexInputAttributeDescription *attrs;
    VkPipelineCache cache;
    VkShaderModule vert;
    VkShaderModule shader;

    // For updating
    VkWriteDescriptorSet *dswrite;
    VkDescriptorImageInfo *dsiinfo;
    VkDescriptorBufferInfo *dsbinfo;
    VkSpecializationInfo specInfo;
    size_t spec_size;
};

int vk_desc_namespace(pl_gpu gpu, enum pl_desc_type type)
{
    return 0;
}

static void pass_destroy_cb(pl_gpu gpu, pl_pass pass)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_pass_vk *pass_vk = PL_PRIV(pass);

    vk->DestroyPipeline(vk->dev, pass_vk->pipe, PL_VK_ALLOC);
    vk->DestroyPipeline(vk->dev, pass_vk->base, PL_VK_ALLOC);
    vk->DestroyRenderPass(vk->dev, pass_vk->renderPass, PL_VK_ALLOC);
    vk->DestroyPipelineLayout(vk->dev, pass_vk->pipeLayout, PL_VK_ALLOC);
    vk->DestroyPipelineCache(vk->dev, pass_vk->cache, PL_VK_ALLOC);
    vk->DestroyDescriptorPool(vk->dev, pass_vk->dsPool, PL_VK_ALLOC);
    vk->DestroyDescriptorSetLayout(vk->dev, pass_vk->dsLayout, PL_VK_ALLOC);
    vk->DestroyShaderModule(vk->dev, pass_vk->vert, PL_VK_ALLOC);
    vk->DestroyShaderModule(vk->dev, pass_vk->shader, PL_VK_ALLOC);

    pl_free((void *) pass);
}

void vk_pass_destroy(pl_gpu gpu, pl_pass pass)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    pl_mutex_lock(&p->recording);
    if (p->cmd) {
        vk_cmd_callback(p->cmd, (vk_cb) pass_destroy_cb, gpu, pass);
    } else {
        vk_dev_callback(vk, (vk_cb) pass_destroy_cb, gpu, pass);
    }
    pl_mutex_unlock(&p->recording);
}

static const VkDescriptorType dsType[] = {
    [PL_DESC_SAMPLED_TEX] = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    [PL_DESC_STORAGE_IMG] = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    [PL_DESC_BUF_UNIFORM] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    [PL_DESC_BUF_STORAGE] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    [PL_DESC_BUF_TEXEL_UNIFORM] = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
    [PL_DESC_BUF_TEXEL_STORAGE] = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
};

#define CACHE_MAGIC {'P','L','V','K'}
#define CACHE_VERSION 4
static const char vk_cache_magic[4] = CACHE_MAGIC;

struct vk_cache_header {
    char magic[sizeof(vk_cache_magic)];
    int cache_version;
    uint64_t signature;
    size_t vert_spirv_len;
    size_t frag_spirv_len;
    size_t comp_spirv_len;
    size_t pipecache_len;
};

static uint64_t cache_signature(pl_gpu gpu, const struct pl_pass_params *params)
{
    struct pl_vk *p = PL_PRIV(gpu);
    uint64_t sig = p->spirv->signature;

    pl_hash_merge(&sig, pl_str0_hash(params->glsl_shader));
    if (params->type == PL_PASS_RASTER)
        pl_hash_merge(&sig, pl_str0_hash(params->vertex_shader));
    return sig;
}

static bool vk_use_cached_program(const struct pl_pass_params *params,
                                  const struct spirv_compiler *spirv,
                                  pl_str *vert_spirv, pl_str *frag_spirv,
                                  pl_str *comp_spirv, pl_str *pipecache,
                                  uint64_t signature)
{
    pl_str cache = {
        .buf = (uint8_t *) params->cached_program,
        .len = params->cached_program_len,
    };

    if (cache.len < sizeof(struct vk_cache_header))
        return false;

    struct vk_cache_header *header = (struct vk_cache_header *) cache.buf;
    cache = pl_str_drop(cache, sizeof(*header));

    if (strncmp(header->magic, vk_cache_magic, sizeof(vk_cache_magic)) != 0)
        return false;
    if (header->cache_version != CACHE_VERSION)
        return false;
    if (header->signature != signature)
        return false;

#define GET(ptr)                                        \
        if (cache.len < header->ptr##_len)              \
            return false;                               \
        *ptr = pl_str_take(cache, header->ptr##_len);   \
        cache = pl_str_drop(cache, ptr->len);

    GET(vert_spirv);
    GET(frag_spirv);
    GET(comp_spirv);
    GET(pipecache);
    return true;
}

static VkResult vk_compile_glsl(pl_gpu gpu, void *alloc,
                                enum glsl_shader_stage stage,
                                const char *shader,
                                pl_str *out_spirv)
{
    struct pl_vk *p = PL_PRIV(gpu);

    clock_t start = clock();
    *out_spirv = spirv_compile_glsl(p->spirv, alloc, &gpu->glsl, stage, shader);
    pl_log_cpu_time(gpu->log, start, clock(), "translating SPIR-V");

    return out_spirv->len ? VK_SUCCESS : VK_ERROR_INITIALIZATION_FAILED;
}

static const VkShaderStageFlags stageFlags[] = {
    [PL_PASS_RASTER]  = VK_SHADER_STAGE_FRAGMENT_BIT |
                        VK_SHADER_STAGE_VERTEX_BIT,
    [PL_PASS_COMPUTE] = VK_SHADER_STAGE_COMPUTE_BIT,
};

static void destroy_pipeline(struct vk_ctx *vk, void *pipeline)
{
    vk->DestroyPipeline(vk->dev, vk_unwrap_handle(pipeline), PL_VK_ALLOC);
}

static VkResult vk_recreate_pipelines(struct vk_ctx *vk, pl_pass pass,
                                      bool derivable, VkPipeline base,
                                      VkPipeline *out_pipe)
{
    struct pl_pass_vk *pass_vk = PL_PRIV(pass);
    const struct pl_pass_params *params = &pass->params;

    // The old pipeline might still be in use, so we have to destroy it
    // asynchronously with a device idle callback
    if (*out_pipe) {
        vk_dev_callback(vk, (vk_cb) destroy_pipeline, vk, vk_wrap_handle(*out_pipe));
        *out_pipe = VK_NULL_HANDLE;
    }

    VkPipelineCreateFlags flags = 0;
    if (derivable)
        flags |= VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT;
    if (base)
        flags |= VK_PIPELINE_CREATE_DERIVATIVE_BIT;

    const VkSpecializationInfo *specInfo = &pass_vk->specInfo;
    if (!specInfo->dataSize)
        specInfo = NULL;

    switch (params->type) {
    case PL_PASS_RASTER: {
        static const VkBlendFactor blendFactors[] = {
            [PL_BLEND_ZERO]                = VK_BLEND_FACTOR_ZERO,
            [PL_BLEND_ONE]                 = VK_BLEND_FACTOR_ONE,
            [PL_BLEND_SRC_ALPHA]           = VK_BLEND_FACTOR_SRC_ALPHA,
            [PL_BLEND_ONE_MINUS_SRC_ALPHA] = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        };

        VkPipelineColorBlendAttachmentState blendState = {
            .colorBlendOp = VK_BLEND_OP_ADD,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                              VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT |
                              VK_COLOR_COMPONENT_A_BIT,
        };

        const struct pl_blend_params *blend = params->blend_params;
        if (blend) {
            blendState.blendEnable = true;
            blendState.srcColorBlendFactor = blendFactors[blend->src_rgb];
            blendState.dstColorBlendFactor = blendFactors[blend->dst_rgb];
            blendState.srcAlphaBlendFactor = blendFactors[blend->src_alpha];
            blendState.dstAlphaBlendFactor = blendFactors[blend->dst_alpha];
        }

        static const VkPrimitiveTopology topologies[PL_PRIM_TYPE_COUNT] = {
            [PL_PRIM_TRIANGLE_LIST]  = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            [PL_PRIM_TRIANGLE_STRIP] = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
        };

        VkGraphicsPipelineCreateInfo cinfo = {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .flags = flags,
            .stageCount = 2,
            .pStages = (VkPipelineShaderStageCreateInfo[]) {
                {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_VERTEX_BIT,
                    .module = pass_vk->vert,
                    .pName = "main",
                }, {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                    .module = pass_vk->shader,
                    .pName = "main",
                    .pSpecializationInfo = specInfo,
                }
            },
            .pVertexInputState = &(VkPipelineVertexInputStateCreateInfo) {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &(VkVertexInputBindingDescription) {
                    .binding = 0,
                    .stride = params->vertex_stride,
                    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
                },
                .vertexAttributeDescriptionCount = params->num_vertex_attribs,
                .pVertexAttributeDescriptions = pass_vk->attrs,
            },
            .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo) {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .topology = topologies[params->vertex_type],
            },
            .pViewportState = &(VkPipelineViewportStateCreateInfo) {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .viewportCount = 1,
                .scissorCount = 1,
            },
            .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo) {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = VK_CULL_MODE_NONE,
                .lineWidth = 1.0f,
            },
            .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo) {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            },
            .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo) {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .attachmentCount = 1,
                .pAttachments = &blendState,
            },
            .pDynamicState = &(VkPipelineDynamicStateCreateInfo) {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                .dynamicStateCount = 2,
                .pDynamicStates = (VkDynamicState[]){
                    VK_DYNAMIC_STATE_VIEWPORT,
                    VK_DYNAMIC_STATE_SCISSOR,
                },
            },
            .layout = pass_vk->pipeLayout,
            .renderPass = pass_vk->renderPass,
            .basePipelineHandle = base,
            .basePipelineIndex = -1,
        };

        return vk->CreateGraphicsPipelines(vk->dev, pass_vk->cache, 1, &cinfo,
                                           PL_VK_ALLOC, out_pipe);
    }

    case PL_PASS_COMPUTE: {
        VkComputePipelineCreateInfo cinfo = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .flags = flags,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = pass_vk->shader,
                .pName = "main",
                .pSpecializationInfo = specInfo,
            },
            .layout = pass_vk->pipeLayout,
            .basePipelineHandle = base,
            .basePipelineIndex = -1,
        };

        return vk->CreateComputePipelines(vk->dev, pass_vk->cache, 1, &cinfo,
                                          PL_VK_ALLOC, out_pipe);
    }

    case PL_PASS_INVALID:
    case PL_PASS_TYPE_COUNT:
        break;
    }

    pl_unreachable();
}

pl_pass vk_pass_create(pl_gpu gpu, const struct pl_pass_params *params)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    bool success = false;

    struct pl_pass_t *pass = pl_zalloc_obj(NULL, pass, struct pl_pass_vk);
    pass->params = pl_pass_params_copy(pass, params);

    struct pl_pass_vk *pass_vk = PL_PRIV(pass);
    pass_vk->dmask = -1; // all descriptors available

    // temporary allocations
    void *tmp = pl_tmp(NULL);

    int num_desc = params->num_descriptors;
    if (!num_desc)
        goto no_descriptors;
    if (num_desc > vk->limits.maxPerStageResources) {
        PL_ERR(gpu, "Pass with %d descriptors exceeds the maximum number of "
               "per-stage resources %" PRIu32"!",
               num_desc, vk->limits.maxPerStageResources);
        goto error;
    }

    pass_vk->dswrite = pl_calloc(pass, num_desc, sizeof(VkWriteDescriptorSet));
    pass_vk->dsiinfo = pl_calloc(pass, num_desc, sizeof(VkDescriptorImageInfo));
    pass_vk->dsbinfo = pl_calloc(pass, num_desc, sizeof(VkDescriptorBufferInfo));

#define NUM_DS (PL_ARRAY_SIZE(pass_vk->dss))

    static int dsSize[PL_DESC_TYPE_COUNT] = {0};
    VkDescriptorSetLayoutBinding *bindings = pl_calloc_ptr(tmp, num_desc, bindings);

    uint32_t max_tex = vk->limits.maxPerStageDescriptorSampledImages,
             max_img = vk->limits.maxPerStageDescriptorStorageImages,
             max_ubo = vk->limits.maxPerStageDescriptorUniformBuffers,
             max_ssbo = vk->limits.maxPerStageDescriptorStorageBuffers;

    uint32_t *dsLimits[PL_DESC_TYPE_COUNT] = {
        [PL_DESC_SAMPLED_TEX] = &max_tex,
        [PL_DESC_STORAGE_IMG] = &max_img,
        [PL_DESC_BUF_UNIFORM] = &max_ubo,
        [PL_DESC_BUF_STORAGE] = &max_ssbo,
        [PL_DESC_BUF_TEXEL_UNIFORM] = &max_tex,
        [PL_DESC_BUF_TEXEL_STORAGE] = &max_img,
    };

    for (int i = 0; i < num_desc; i++) {
        struct pl_desc *desc = &params->descriptors[i];
        if (!(*dsLimits[desc->type])--) {
            PL_ERR(gpu, "Pass exceeds the maximum number of per-stage "
                   "descriptors of type %u!", (unsigned) desc->type);
            goto error;
        }

        dsSize[desc->type]++;
        bindings[i] = (VkDescriptorSetLayoutBinding) {
            .binding = desc->binding,
            .descriptorType = dsType[desc->type],
            .descriptorCount = 1,
            .stageFlags = stageFlags[params->type],
        };
    }

    VkDescriptorSetLayoutCreateInfo dinfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pBindings = bindings,
        .bindingCount = num_desc,
    };

    if (p->max_push_descriptors && num_desc <= p->max_push_descriptors) {
        dinfo.flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        pass_vk->use_pushd = true;
    } else if (p->max_push_descriptors) {
        PL_INFO(gpu, "Pass with %d descriptors exceeds the maximum push "
                "descriptor count (%d). Falling back to descriptor sets!",
                num_desc, p->max_push_descriptors);
    }

    VK(vk->CreateDescriptorSetLayout(vk->dev, &dinfo, PL_VK_ALLOC,
                                     &pass_vk->dsLayout));

    if (!pass_vk->use_pushd) {
        PL_ARRAY(VkDescriptorPoolSize) dsPoolSizes = {0};

        for (enum pl_desc_type t = 0; t < PL_DESC_TYPE_COUNT; t++) {
            if (dsSize[t] > 0) {
                PL_ARRAY_APPEND(tmp, dsPoolSizes, (VkDescriptorPoolSize) {
                    .type = dsType[t],
                    .descriptorCount = dsSize[t] * NUM_DS,
                });
            }
        }

        if (dsPoolSizes.num) {
            VkDescriptorPoolCreateInfo pinfo = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = NUM_DS,
                .pPoolSizes = dsPoolSizes.elem,
                .poolSizeCount = dsPoolSizes.num,
            };

            VK(vk->CreateDescriptorPool(vk->dev, &pinfo, PL_VK_ALLOC, &pass_vk->dsPool));

            VkDescriptorSetLayout layouts[NUM_DS];
            for (int i = 0; i < NUM_DS; i++)
                layouts[i] = pass_vk->dsLayout;

            VkDescriptorSetAllocateInfo ainfo = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = pass_vk->dsPool,
                .descriptorSetCount = NUM_DS,
                .pSetLayouts = layouts,
            };

            VK(vk->AllocateDescriptorSets(vk->dev, &ainfo, pass_vk->dss));
        }
    }

no_descriptors: ;

    bool has_spec = params->num_constants;
    if (has_spec) {
        PL_ARRAY(VkSpecializationMapEntry) entries = {0};
        PL_ARRAY_RESIZE(pass, entries, params->num_constants);
        size_t spec_size = 0;

        for (int i = 0; i < params->num_constants; i++) {
            const struct pl_constant *con = &params->constants[i];
            size_t con_size = pl_var_type_size(con->type);
            entries.elem[i] = (VkSpecializationMapEntry) {
                .constantID = con->id,
                .offset = con->offset,
                .size = con_size,
            };

            size_t req_size = con->offset + con_size;
            spec_size = PL_MAX(spec_size, req_size);
        }

        pass_vk->spec_size = spec_size;
        pass_vk->specInfo = (VkSpecializationInfo) {
            .mapEntryCount = params->num_constants,
            .pMapEntries = entries.elem,
        };

        if (params->constant_data) {
            pass_vk->specInfo.pData = pl_memdup(pass, params->constant_data, spec_size);
            pass_vk->specInfo.dataSize = spec_size;
        }
    }

    VkPipelineLayoutCreateInfo linfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = num_desc ? 1 : 0,
        .pSetLayouts = &pass_vk->dsLayout,
        .pushConstantRangeCount = params->push_constants_size ? 1 : 0,
        .pPushConstantRanges = &(VkPushConstantRange){
            .stageFlags = stageFlags[params->type],
            .offset = 0,
            .size = params->push_constants_size,
        },
    };

    VK(vk->CreatePipelineLayout(vk->dev, &linfo, PL_VK_ALLOC,
                                &pass_vk->pipeLayout));

    pl_str vert = {0}, frag = {0}, comp = {0}, pipecache = {0};
    uint64_t sig = cache_signature(gpu, params);
    if (vk_use_cached_program(params, p->spirv, &vert, &frag, &comp, &pipecache, sig)) {
        PL_DEBUG(gpu, "Using cached SPIR-V and VkPipeline");
    } else {
        pipecache.len = 0;
        switch (params->type) {
        case PL_PASS_RASTER:
            VK(vk_compile_glsl(gpu, tmp, GLSL_SHADER_VERTEX,
                               params->vertex_shader, &vert));
            VK(vk_compile_glsl(gpu, tmp, GLSL_SHADER_FRAGMENT,
                               params->glsl_shader, &frag));
            comp.len = 0;
            break;
        case PL_PASS_COMPUTE:
            VK(vk_compile_glsl(gpu, tmp, GLSL_SHADER_COMPUTE,
                               params->glsl_shader, &comp));
            frag.len = 0;
            vert.len = 0;
            break;
        case PL_PASS_INVALID:
        case PL_PASS_TYPE_COUNT:
            pl_unreachable();
        }
    }

    VkPipelineCacheCreateInfo pcinfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        .pInitialData = pipecache.buf,
        .initialDataSize = pipecache.len,
    };

    VK(vk->CreatePipelineCache(vk->dev, &pcinfo, PL_VK_ALLOC, &pass_vk->cache));

    VkShaderModuleCreateInfo sinfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    };

    clock_t start = clock();
    switch (params->type) {
    case PL_PASS_RASTER: {
        sinfo.pCode = (uint32_t *) vert.buf;
        sinfo.codeSize = vert.len;
        VK(vk->CreateShaderModule(vk->dev, &sinfo, PL_VK_ALLOC, &pass_vk->vert));
        PL_VK_NAME(SHADER_MODULE, pass_vk->vert, "vertex");

        sinfo.pCode = (uint32_t *) frag.buf;
        sinfo.codeSize = frag.len;
        VK(vk->CreateShaderModule(vk->dev, &sinfo, PL_VK_ALLOC, &pass_vk->shader));
        PL_VK_NAME(SHADER_MODULE, pass_vk->shader, "fragment");

        pass_vk->attrs = pl_calloc_ptr(pass, params->num_vertex_attribs, pass_vk->attrs);
        for (int i = 0; i < params->num_vertex_attribs; i++) {
            struct pl_vertex_attrib *va = &params->vertex_attribs[i];
            const struct vk_format **pfmt_vk = PL_PRIV(va->fmt);

            pass_vk->attrs[i] = (VkVertexInputAttributeDescription) {
                .binding  = 0,
                .location = va->location,
                .offset   = va->offset,
                .format   = PL_DEF((*pfmt_vk)->bfmt, (*pfmt_vk)->tfmt),
            };
        }

        VkRenderPassCreateInfo rinfo = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &(VkAttachmentDescription) {
                .format = (VkFormat) params->target_format->signature,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .loadOp = pass->params.load_target
                            ? VK_ATTACHMENT_LOAD_OP_LOAD
                            : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            },
            .subpassCount = 1,
            .pSubpasses = &(VkSubpassDescription) {
                .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                .colorAttachmentCount = 1,
                .pColorAttachments = &(VkAttachmentReference) {
                    .attachment = 0,
                    .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                },
            },
        };

        VK(vk->CreateRenderPass(vk->dev, &rinfo, PL_VK_ALLOC, &pass_vk->renderPass));
        break;
    }
    case PL_PASS_COMPUTE: {
        sinfo.pCode = (uint32_t *) comp.buf;
        sinfo.codeSize = comp.len;
        VK(vk->CreateShaderModule(vk->dev, &sinfo, PL_VK_ALLOC, &pass_vk->shader));
        PL_VK_NAME(SHADER_MODULE, pass_vk->shader, "compute");
        break;
    }
    case PL_PASS_INVALID:
    case PL_PASS_TYPE_COUNT:
        pl_unreachable();
    }

    clock_t after_compilation = clock();
    pl_log_cpu_time(gpu->log, start, after_compilation, "compiling shader");

    // Create the graphics/compute pipeline
    VkPipeline *pipe = has_spec ? &pass_vk->base : &pass_vk->pipe;
    VK(vk_recreate_pipelines(vk, pass, has_spec, VK_NULL_HANDLE, pipe));
    pl_log_cpu_time(gpu->log, after_compilation, clock(), "creating pipeline");

    if (!has_spec) {
        // We can free these if we no longer need them for specialization
        pl_free_ptr(&pass_vk->attrs);
        vk->DestroyShaderModule(vk->dev, pass_vk->vert, PL_VK_ALLOC);
        vk->DestroyShaderModule(vk->dev, pass_vk->shader, PL_VK_ALLOC);
        pass_vk->vert = VK_NULL_HANDLE;
        pass_vk->shader = VK_NULL_HANDLE;
    }

    // Update params->cached_program
    pl_str cache = {0};
    VK(vk->GetPipelineCacheData(vk->dev, pass_vk->cache, &cache.len, NULL));
    cache.buf = pl_alloc(tmp, cache.len);
    VK(vk->GetPipelineCacheData(vk->dev, pass_vk->cache, &cache.len, cache.buf));
    if (!has_spec) {
        vk->DestroyPipelineCache(vk->dev, pass_vk->cache, PL_VK_ALLOC);
        pass_vk->cache = VK_NULL_HANDLE;
    }

    struct vk_cache_header header = {
        .magic = CACHE_MAGIC,
        .cache_version = CACHE_VERSION,
        .signature = sig,
        .vert_spirv_len = vert.len,
        .frag_spirv_len = frag.len,
        .comp_spirv_len = comp.len,
        .pipecache_len = cache.len,
    };

    PL_DEBUG(vk, "Pass statistics: size %zu, SPIR-V: vert %zu frag %zu comp %zu",
             cache.len, vert.len, frag.len, comp.len);

    pl_str prog = {0};
    pl_str_append(pass, &prog, (pl_str){ (uint8_t *) &header, sizeof(header) });
    pl_str_append(pass, &prog, vert);
    pl_str_append(pass, &prog, frag);
    pl_str_append(pass, &prog, comp);
    pl_str_append(pass, &prog, cache);
    pass->params.cached_program = prog.buf;
    pass->params.cached_program_len = prog.len;

    success = true;

error:
    if (!success) {
        pass_destroy_cb(gpu, pass);
        pass = NULL;
    }

#undef NUM_DS

    pl_free(tmp);
    return pass;
}

static const VkPipelineStageFlags passStages[] = {
    [PL_PASS_RASTER]  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
    [PL_PASS_COMPUTE] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
};

static void vk_update_descriptor(pl_gpu gpu, struct vk_cmd *cmd, pl_pass pass,
                                 struct pl_desc_binding db,
                                 VkDescriptorSet ds, int idx)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct pl_pass_vk *pass_vk = PL_PRIV(pass);
    struct pl_desc *desc = &pass->params.descriptors[idx];

    VkWriteDescriptorSet *wds = &pass_vk->dswrite[idx];
    *wds = (VkWriteDescriptorSet) {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = desc->binding,
        .descriptorCount = 1,
        .descriptorType = dsType[desc->type],
    };

    static const VkAccessFlags access[PL_DESC_ACCESS_COUNT] = {
        [PL_DESC_ACCESS_READONLY]   = VK_ACCESS_SHADER_READ_BIT,
        [PL_DESC_ACCESS_WRITEONLY]  = VK_ACCESS_SHADER_WRITE_BIT,
        [PL_DESC_ACCESS_READWRITE]  = VK_ACCESS_SHADER_READ_BIT |
                                      VK_ACCESS_SHADER_WRITE_BIT,
    };

    switch (desc->type) {
    case PL_DESC_SAMPLED_TEX: {
        pl_tex tex = db.object;
        struct pl_tex_vk *tex_vk = PL_PRIV(tex);

        vk_tex_barrier(gpu, cmd, tex, passStages[pass->params.type],
                      VK_ACCESS_SHADER_READ_BIT,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, false);

        VkDescriptorImageInfo *iinfo = &pass_vk->dsiinfo[idx];
        *iinfo = (VkDescriptorImageInfo) {
            .sampler = p->samplers[db.sample_mode][db.address_mode],
            .imageView = tex_vk->view,
            .imageLayout = tex_vk->layout,
        };

        wds->pImageInfo = iinfo;
        return;
    }
    case PL_DESC_STORAGE_IMG: {
        pl_tex tex = db.object;
        struct pl_tex_vk *tex_vk = PL_PRIV(tex);

        vk_tex_barrier(gpu, cmd, tex, passStages[pass->params.type],
                       access[desc->access], VK_IMAGE_LAYOUT_GENERAL, false);

        VkDescriptorImageInfo *iinfo = &pass_vk->dsiinfo[idx];
        *iinfo = (VkDescriptorImageInfo) {
            .imageView = tex_vk->view,
            .imageLayout = tex_vk->layout,
        };

        wds->pImageInfo = iinfo;
        return;
    }
    case PL_DESC_BUF_UNIFORM:
    case PL_DESC_BUF_STORAGE: {
        pl_buf buf = db.object;
        struct pl_buf_vk *buf_vk = PL_PRIV(buf);

        vk_buf_barrier(gpu, cmd, buf, passStages[pass->params.type],
                       access[desc->access], 0, buf->params.size, false);

        VkDescriptorBufferInfo *binfo = &pass_vk->dsbinfo[idx];
        *binfo = (VkDescriptorBufferInfo) {
            .buffer = buf_vk->mem.buf,
            .offset = buf_vk->mem.offset,
            .range = buf->params.size,
        };

        wds->pBufferInfo = binfo;
        return;
    }
    case PL_DESC_BUF_TEXEL_UNIFORM:
    case PL_DESC_BUF_TEXEL_STORAGE: {
        pl_buf buf = db.object;
        struct pl_buf_vk *buf_vk = PL_PRIV(buf);

        vk_buf_barrier(gpu, cmd, buf, passStages[pass->params.type],
                       access[desc->access], 0, buf->params.size, false);

        wds->pTexelBufferView = &buf_vk->view;
        return;
    }
    case PL_DESC_INVALID:
    case PL_DESC_TYPE_COUNT:
        break;
    }

    pl_unreachable();
}

static void vk_release_descriptor(pl_gpu gpu, struct vk_cmd *cmd, pl_pass pass,
                                  struct pl_desc_binding db, int idx)
{
    const struct pl_desc *desc = &pass->params.descriptors[idx];

    switch (desc->type) {
    case PL_DESC_BUF_UNIFORM:
    case PL_DESC_BUF_STORAGE:
    case PL_DESC_BUF_TEXEL_UNIFORM:
    case PL_DESC_BUF_TEXEL_STORAGE:
        if (desc->access != PL_DESC_ACCESS_READONLY) {
            pl_buf buf = db.object;
            vk_buf_flush(gpu, cmd, buf, 0, buf->params.size);
        }
        return;
    case PL_DESC_SAMPLED_TEX:
    case PL_DESC_STORAGE_IMG:
        return;
    case PL_DESC_INVALID:
    case PL_DESC_TYPE_COUNT:
        break;
    }

    pl_unreachable();
}

static void set_ds(struct pl_pass_vk *pass_vk, void *dsbit)
{
    pass_vk->dmask |= (uintptr_t) dsbit;
}

static bool need_respec(pl_pass pass, const struct pl_pass_run_params *params)
{
    struct pl_pass_vk *pass_vk = PL_PRIV(pass);
    if (!pass_vk->spec_size || !params->constant_data)
        return false;

    VkSpecializationInfo *specInfo = &pass_vk->specInfo;
    size_t size = pass_vk->spec_size;
    if (!specInfo->pData) {
        // Shader was never specialized before
        specInfo->pData = pl_memdup((void *) pass, params->constant_data, size);
        specInfo->dataSize = size;
        return true;
    }

    // Shader is being re-specialized with new values
    if (memcmp(specInfo->pData, params->constant_data, size) != 0) {
        memcpy((void *) specInfo->pData, params->constant_data, size);
        return true;
    }

    return false;
}

void vk_pass_run(pl_gpu gpu, const struct pl_pass_run_params *params)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    pl_pass pass = params->pass;
    struct pl_pass_vk *pass_vk = PL_PRIV(pass);

    if (params->vertex_data || params->index_data)
        return pl_pass_run_vbo(gpu, params);

    // Check if we need to re-specialize this pipeline
    if (need_respec(pass, params)) {
        clock_t start = clock();
        VK(vk_recreate_pipelines(vk, pass, false, pass_vk->base, &pass_vk->pipe));
        pl_log_cpu_time(gpu->log, start, clock(), "re-specializing shader");
    }

    if (!pass_vk->use_pushd) {
        // Wait for a free descriptor set
        while (!pass_vk->dmask) {
            PL_TRACE(gpu, "No free descriptor sets! ...blocking (slow path)");
            vk_poll_commands(vk, 10000000); // 10 ms
        }
    }

    static const enum queue_type types[] = {
        [PL_PASS_RASTER]  = GRAPHICS,
        [PL_PASS_COMPUTE] = COMPUTE,
    };

    struct vk_cmd *cmd = CMD_BEGIN_TIMED(types[pass->params.type], params->timer);
    if (!cmd)
        goto error;

    // Find a descriptor set to use
    VkDescriptorSet ds = VK_NULL_HANDLE;
    if (!pass_vk->use_pushd) {
        for (int i = 0; i < PL_ARRAY_SIZE(pass_vk->dss); i++) {
            uint16_t dsbit = 1u << i;
            if (pass_vk->dmask & dsbit) {
                ds = pass_vk->dss[i];
                pass_vk->dmask &= ~dsbit; // unset
                vk_cmd_callback(cmd, (vk_cb) set_ds, pass_vk,
                                (void *)(uintptr_t) dsbit);
                break;
            }
        }
    }

    // Update the dswrite structure with all of the new values
    for (int i = 0; i < pass->params.num_descriptors; i++)
        vk_update_descriptor(gpu, cmd, pass, params->desc_bindings[i], ds, i);

    if (!pass_vk->use_pushd) {
        vk->UpdateDescriptorSets(vk->dev, pass->params.num_descriptors,
                                 pass_vk->dswrite, 0, NULL);
    }

    // Bind the pipeline, descriptor set, etc.
    static const VkPipelineBindPoint bindPoint[] = {
        [PL_PASS_RASTER]  = VK_PIPELINE_BIND_POINT_GRAPHICS,
        [PL_PASS_COMPUTE] = VK_PIPELINE_BIND_POINT_COMPUTE,
    };

    vk->CmdBindPipeline(cmd->buf, bindPoint[pass->params.type],
                        PL_DEF(pass_vk->pipe, pass_vk->base));

    if (ds) {
        vk->CmdBindDescriptorSets(cmd->buf, bindPoint[pass->params.type],
                                  pass_vk->pipeLayout, 0, 1, &ds, 0, NULL);
    }

    if (pass_vk->use_pushd) {
        vk->CmdPushDescriptorSetKHR(cmd->buf, bindPoint[pass->params.type],
                                    pass_vk->pipeLayout, 0,
                                    pass->params.num_descriptors,
                                    pass_vk->dswrite);
    }

    if (pass->params.push_constants_size) {
        vk->CmdPushConstants(cmd->buf, pass_vk->pipeLayout,
                             stageFlags[pass->params.type], 0,
                             pass->params.push_constants_size,
                             params->push_constants);
    }

    switch (pass->params.type) {
    case PL_PASS_RASTER: {
        pl_tex tex = params->target;
        struct pl_tex_vk *tex_vk = PL_PRIV(tex);
        pl_buf vert = params->vertex_buf;
        struct pl_buf_vk *vert_vk = PL_PRIV(vert);
        pl_buf index = params->index_buf;
        struct pl_buf_vk *index_vk = index ? PL_PRIV(index) : NULL;
        pl_assert(vert);

        // In the edge case that vert = index buffer, we need to synchronize
        // for both flags simultaneously
        VkAccessFlags vbo_flags = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        if (index == vert)
            vbo_flags |= VK_ACCESS_INDEX_READ_BIT;

        vk_buf_barrier(gpu, cmd, vert, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                       vbo_flags, 0, vert->params.size, false);

        VkDeviceSize offset = vert_vk->mem.offset + params->buf_offset;
        vk->CmdBindVertexBuffers(cmd->buf, 0, 1, &vert_vk->mem.buf, &offset);

        if (index) {
            if (index != vert) {
                vk_buf_barrier(gpu, cmd, index, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                               VK_ACCESS_INDEX_READ_BIT, 0, index->params.size,
                               false);
            }

            static const VkIndexType index_fmts[PL_INDEX_FORMAT_COUNT] = {
                [PL_INDEX_UINT16] = VK_INDEX_TYPE_UINT16,
                [PL_INDEX_UINT32] = VK_INDEX_TYPE_UINT32,
            };

            vk->CmdBindIndexBuffer(cmd->buf, index_vk->mem.buf,
                                   index_vk->mem.offset + params->index_offset,
                                   index_fmts[params->index_fmt]);
        }

        vk_tex_barrier(gpu, cmd, tex, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, false);

        VkViewport viewport = {
            .x = params->viewport.x0,
            .y = params->viewport.y0,
            .width  = pl_rect_w(params->viewport),
            .height = pl_rect_h(params->viewport),
        };

        VkRect2D scissor = {
            .offset = {params->scissors.x0, params->scissors.y0},
            .extent = {pl_rect_w(params->scissors), pl_rect_h(params->scissors)},
        };

        vk->CmdSetViewport(cmd->buf, 0, 1, &viewport);
        vk->CmdSetScissor(cmd->buf, 0, 1, &scissor);

        VkRenderPassBeginInfo binfo = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = pass_vk->renderPass,
            .framebuffer = tex_vk->framebuffer,
            .renderArea.extent = {tex->params.w, tex->params.h},
        };

        vk->CmdBeginRenderPass(cmd->buf, &binfo, VK_SUBPASS_CONTENTS_INLINE);

        if (index) {
            vk->CmdDrawIndexed(cmd->buf, params->vertex_count, 1, 0, 0, 0);
        } else {
            vk->CmdDraw(cmd->buf, params->vertex_count, 1, 0, 0);
        }

        vk->CmdEndRenderPass(cmd->buf);

        // The renderPass implicitly transitions the texture to this layout
        tex_vk->layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        break;
    }
    case PL_PASS_COMPUTE:
        vk->CmdDispatch(cmd->buf, params->compute_groups[0],
                        params->compute_groups[1],
                        params->compute_groups[2]);
        break;
    case PL_PASS_INVALID:
    case PL_PASS_TYPE_COUNT:
        pl_unreachable();
    };

    for (int i = 0; i < pass->params.num_descriptors; i++)
        vk_release_descriptor(gpu, cmd, pass, params->desc_bindings[i], i);

    // submit this command buffer for better intra-frame granularity
    CMD_SUBMIT(&cmd);

error:
    return;
}
