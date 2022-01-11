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
#include "formats.h"
#include "glsl/spirv.h"

struct stream_buf_slice {
    const void *data;
    unsigned int size;
    unsigned int offset;
};

// Upload one or more slices of single-use data to a suballocated dynamic
// buffer. Only call this once per-buffer per-pass, since it will discard or
// reallocate the buffer when full.
static bool stream_buf_upload(pl_gpu gpu, struct d3d_stream_buf *stream,
                              struct stream_buf_slice *slices, int num_slices)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;
    unsigned int align = PL_DEF(stream->align, sizeof(float));

    // Get total size, rounded up to the buffer's alignment
    size_t size = 0;
    for (int i = 0; i < num_slices; i++)
        size += PL_ALIGN2(slices[i].size, align);

    if (size > gpu->limits.max_buf_size) {
        PL_ERR(gpu, "Streaming buffer is too large");
        return -1;
    }

    // If the data doesn't fit, realloc the buffer
    if (size > stream->size) {
        size_t new_size = stream->size;
        // Arbitrary base size
        if (!new_size)
            new_size = 16 * 1024;
        while (new_size < size)
            new_size *= 2;
        new_size = PL_MIN(new_size, gpu->limits.max_buf_size);

        ID3D11Buffer *new_buf;
        D3D11_BUFFER_DESC vbuf_desc = {
            .ByteWidth = new_size,
            .Usage = D3D11_USAGE_DYNAMIC,
            .BindFlags = stream->bind_flags,
            .CPUAccessFlags = D3D11_CPU_ACCESS_WRITE,
        };
        D3D(ID3D11Device_CreateBuffer(p->dev, &vbuf_desc, NULL, &new_buf));

        SAFE_RELEASE(stream->buf);
        stream->buf = new_buf;
        stream->size = new_size;
        stream->used = 0;
    }

    bool discard = false;
    size_t offset = stream->used;
    if (offset + size > stream->size) {
        // We reached the end of the buffer, so discard and wrap around
        discard = true;
        offset = 0;
    }

    D3D11_MAPPED_SUBRESOURCE map = {0};
    UINT type = discard ? D3D11_MAP_WRITE_DISCARD : D3D11_MAP_WRITE_NO_OVERWRITE;
    D3D(ID3D11DeviceContext_Map(p->imm, (ID3D11Resource *) stream->buf, 0, type,
                                0, &map));

    // Upload each slice
    char *cdata = map.pData;
    stream->used = offset;
    for (int i = 0; i < num_slices; i++) {
        slices[i].offset = stream->used;
        memcpy(cdata + slices[i].offset, slices[i].data, slices[i].size);
        stream->used += PL_ALIGN2(slices[i].size, align);
    }

    ID3D11DeviceContext_Unmap(p->imm, (ID3D11Resource *) stream->buf, 0);

    return true;

error:
    return false;
}

static const char *get_shader_target(pl_gpu gpu, enum glsl_shader_stage stage)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    switch (p->fl) {
    default:
        switch (stage) {
        case GLSL_SHADER_VERTEX:   return "vs_5_0";
        case GLSL_SHADER_FRAGMENT: return "ps_5_0";
        case GLSL_SHADER_COMPUTE:  return "cs_5_0";
        }
        break;
    case D3D_FEATURE_LEVEL_10_1:
        switch (stage) {
        case GLSL_SHADER_VERTEX:   return "vs_4_1";
        case GLSL_SHADER_FRAGMENT: return "ps_4_1";
        case GLSL_SHADER_COMPUTE:  return "cs_4_1";
        }
        break;
    case D3D_FEATURE_LEVEL_10_0:
        switch (stage) {
        case GLSL_SHADER_VERTEX:   return "vs_4_0";
        case GLSL_SHADER_FRAGMENT: return "ps_4_0";
        case GLSL_SHADER_COMPUTE:  return "cs_4_0";
        }
        break;
    case D3D_FEATURE_LEVEL_9_3:
        switch (stage) {
        case GLSL_SHADER_VERTEX:   return "vs_4_0_level_9_3";
        case GLSL_SHADER_FRAGMENT: return "ps_4_0_level_9_3";
        case GLSL_SHADER_COMPUTE:  return NULL;
        }
        break;
    case D3D_FEATURE_LEVEL_9_2:
    case D3D_FEATURE_LEVEL_9_1:
        switch (stage) {
        case GLSL_SHADER_VERTEX:   return "vs_4_0_level_9_1";
        case GLSL_SHADER_FRAGMENT: return "ps_4_0_level_9_1";
        case GLSL_SHADER_COMPUTE:  return NULL;
        }
        break;
    }
    return NULL;
}

#define SC(cmd)                                                              \
    do {                                                                     \
        spvc_result res = (cmd);                                             \
        if (res != SPVC_SUCCESS) {                                           \
            PL_ERR(gpu, "%s: %s (%d) (%s:%d)",                               \
                   #cmd, pass_s->sc ?                                        \
                       spvc_context_get_last_error_string(pass_s->sc) : "",  \
                   res, __FILE__, __LINE__);                                 \
            goto error;                                                      \
        }                                                                    \
    } while (0)

static spvc_result mark_resources_used(pl_pass pass, spvc_compiler sc_comp,
                                       spvc_resources resources,
                                       spvc_resource_type res_type,
                                       enum glsl_shader_stage stage)
{
    struct pl_pass_d3d11 *pass_p = PL_PRIV(pass);
    const spvc_reflected_resource *res_list;
    size_t res_count;
    spvc_result res;

    res = spvc_resources_get_resource_list_for_type(resources, res_type,
                                                    &res_list, &res_count);
    if (res != SPVC_SUCCESS)
        return res;

    for (int i = 0; i < res_count; i++) {
        unsigned int binding = spvc_compiler_get_decoration(sc_comp,
            res_list[i].id, SpvDecorationBinding);
        unsigned int descriptor_set = spvc_compiler_get_decoration(sc_comp,
            res_list[i].id, SpvDecorationDescriptorSet);
        if (descriptor_set != 0)
            continue;

        // Find the pl_desc with this binding and mark it as used
        for (int j = 0; j < pass->params.num_descriptors; j++) {
            struct pl_desc *desc = &pass->params.descriptors[j];
            if (desc->binding != binding)
                continue;

            struct pl_desc_d3d11 *desc_p = &pass_p->descriptors[j];
            if (stage == GLSL_SHADER_VERTEX) {
                desc_p->vertex.used = true;
            } else {
                desc_p->main.used = true;
            }
        }
    }

    return res;
}

static const char *shader_names[] = {
    [GLSL_SHADER_VERTEX]   = "vertex",
    [GLSL_SHADER_FRAGMENT] = "fragment",
    [GLSL_SHADER_COMPUTE]  = "compute",
};

static bool shader_compile_glsl(pl_gpu gpu, pl_pass pass,
                                struct d3d_pass_stage *pass_s,
                                enum glsl_shader_stage stage, const char *glsl)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    void *tmp = pl_tmp(NULL);
    bool success = false;

    clock_t start = clock();
    pl_str spirv = spirv_compile_glsl(p->spirv, tmp, &gpu->glsl, stage, glsl);
    if (!spirv.len)
        goto error;

    pl_log_cpu_time(gpu->log, start, clock(), "translating GLSL to SPIR-V");

    SC(spvc_context_create(&pass_s->sc));

    spvc_parsed_ir sc_ir;
    SC(spvc_context_parse_spirv(pass_s->sc, (SpvId *) spirv.buf,
                                spirv.len / sizeof(SpvId), &sc_ir));

    SC(spvc_context_create_compiler(pass_s->sc, SPVC_BACKEND_HLSL, sc_ir,
                                    SPVC_CAPTURE_MODE_TAKE_OWNERSHIP,
                                    &pass_s->sc_comp));

    spvc_compiler_options sc_opts;
    SC(spvc_compiler_create_compiler_options(pass_s->sc_comp, &sc_opts));

    int sc_shader_model;
    if (p->fl >= D3D_FEATURE_LEVEL_11_0) {
        sc_shader_model = 50;
    } else if (p->fl >= D3D_FEATURE_LEVEL_10_1) {
        sc_shader_model = 41;
    } else {
        sc_shader_model = 40;
    }

    SC(spvc_compiler_options_set_uint(sc_opts,
        SPVC_COMPILER_OPTION_HLSL_SHADER_MODEL, sc_shader_model));

    // Unlike Vulkan and OpenGL, in D3D11, the clip-space is "flipped" with
    // respect to framebuffer-space. In other words, if you render to a pixel at
    // (0, -1), you have to sample from (0, 1) to get the value back. We unflip
    // it by setting the following option, which inserts the equivalent of
    // `gl_Position.y = -gl_Position.y` into the vertex shader
    if (stage == GLSL_SHADER_VERTEX) {
        SC(spvc_compiler_options_set_bool(sc_opts,
            SPVC_COMPILER_OPTION_FLIP_VERTEX_Y, SPVC_TRUE));
    }

    // Bind readonly images and imageBuffers as SRVs. This is done because a lot
    // of hardware (especially FL11_x hardware) has very poor format support for
    // reading values from UAVs. It allows the common case of readonly and
    // writeonly images to support more formats, though the less common case of
    // readwrite images still requires format support for UAV loads (represented
    // by the PL_FMT_CAP_READWRITE cap in libplacebo.)
    //
    // Note that setting this option comes at the cost of GLSL support. Readonly
    // and readwrite images are the same type in GLSL, but SRV and UAV bound
    // textures are different types in HLSL, so for example, a GLSL function
    // with an image parameter may fail to compile as HLSL if it's called with a
    // readonly image and a readwrite image at different call sites.
    SC(spvc_compiler_options_set_bool(sc_opts,
        SPVC_COMPILER_OPTION_HLSL_NONWRITABLE_UAV_TEXTURE_AS_SRV, SPVC_TRUE));

    SC(spvc_compiler_install_compiler_options(pass_s->sc_comp, sc_opts));

    spvc_set active = NULL;
    SC(spvc_compiler_get_active_interface_variables(pass_s->sc_comp, &active));
    spvc_resources resources = NULL;
    SC(spvc_compiler_create_shader_resources_for_active_variables(
        pass_s->sc_comp, &resources, active));

    // In D3D11, the vertex shader and fragment shader can have a different set
    // of bindings. At this point, SPIRV-Cross knows which resources are
    // statically used in each stage. We can use this information to optimize
    // HLSL register allocation by not binding resources to shader stages
    // they're not used in.
    mark_resources_used(pass, pass_s->sc_comp, resources,
                        SPVC_RESOURCE_TYPE_UNIFORM_BUFFER, stage);
    mark_resources_used(pass, pass_s->sc_comp, resources,
                        SPVC_RESOURCE_TYPE_STORAGE_BUFFER, stage);
    mark_resources_used(pass, pass_s->sc_comp, resources,
                        SPVC_RESOURCE_TYPE_STORAGE_IMAGE, stage);
    mark_resources_used(pass, pass_s->sc_comp, resources,
                        SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, stage);

    success = true;
error:;
    if (!success) {
        PL_ERR(gpu, "%s shader GLSL source:", shader_names[stage]);
        pl_msg_source(gpu->ctx, PL_LOG_ERR, glsl);

        if (pass_s->sc) {
            spvc_context_destroy(pass_s->sc);
            pass_s->sc = NULL;
        }
    }
    pl_free(tmp);

    return success;
}

static bool shader_compile_hlsl(pl_gpu gpu, pl_pass pass,
                                struct d3d_pass_stage *pass_s,
                                enum glsl_shader_stage stage, const char *glsl,
                                ID3DBlob **out)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct pl_pass_d3d11 *pass_p = PL_PRIV(pass);
    const char *hlsl = NULL;
    ID3DBlob *errors = NULL;
    bool success = false;
    HRESULT hr;

    int max_binding = -1;

    // This should not be called without first calling shader_compile_glsl
    pl_assert(pass_s->sc_comp);

    static const SpvExecutionModel spv_execution_model[] = {
        [GLSL_SHADER_VERTEX]   = SpvExecutionModelVertex,
        [GLSL_SHADER_FRAGMENT] = SpvExecutionModelFragment,
        [GLSL_SHADER_COMPUTE]  = SpvExecutionModelGLCompute,
    };

    // Assign the HLSL register numbers we want to use for each resource
    for (int i = 0; i < pass->params.num_descriptors; i++) {
        struct pl_desc *desc = &pass->params.descriptors[i];
        struct pl_desc_d3d11 *desc_p = &pass_p->descriptors[i];
        struct d3d_desc_stage *desc_s =
            stage == GLSL_SHADER_VERTEX ? &desc_p->vertex : &desc_p->main;

        // Skip resources that aren't in this shader stage
        if (!desc_s->used)
            continue;

        spvc_hlsl_resource_binding binding;
        spvc_hlsl_resource_binding_init(&binding);
        binding.stage = spv_execution_model[stage];
        binding.binding = desc->binding;
        max_binding = PL_MAX(max_binding, desc->binding);
        if (desc_s->cbv_slot > 0)
            binding.cbv.register_binding = desc_s->cbv_slot;
        if (desc_s->srv_slot > 0)
            binding.srv.register_binding = desc_s->srv_slot;
        if (desc_s->sampler_slot > 0)
            binding.sampler.register_binding = desc_s->sampler_slot;
        if (desc_s->uav_slot > 0)
            binding.uav.register_binding = desc_s->uav_slot;
        SC(spvc_compiler_hlsl_add_resource_binding(pass_s->sc_comp, &binding));
    }

    if (stage == GLSL_SHADER_COMPUTE) {
        // Check if the gl_NumWorkGroups builtin is used. If it is, we have to
        // emulate it with a constant buffer, so allocate it a CBV register.
        spvc_variable_id num_workgroups_id =
            spvc_compiler_hlsl_remap_num_workgroups_builtin(pass_s->sc_comp);
        if (num_workgroups_id) {
            pass_p->num_workgroups_used = true;

            spvc_hlsl_resource_binding binding;
            spvc_hlsl_resource_binding_init(&binding);
            binding.stage = spv_execution_model[stage];
            binding.binding = max_binding + 1;

            // Allocate a CBV register for the buffer
            binding.cbv.register_binding = pass_s->cbvs.num;
            PL_ARRAY_APPEND(pass, pass_s->cbvs, HLSL_BINDING_NUM_WORKGROUPS);
            if (pass_s->cbvs.num >
                    D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) {
                PL_ERR(gpu, "Not enough constant buffer slots for gl_NumWorkGroups");
                goto error;
            }

            spvc_compiler_set_decoration(pass_s->sc_comp, num_workgroups_id,
                                         SpvDecorationDescriptorSet, 0);
            spvc_compiler_set_decoration(pass_s->sc_comp, num_workgroups_id,
                                         SpvDecorationBinding, binding.binding);

            SC(spvc_compiler_hlsl_add_resource_binding(pass_s->sc_comp, &binding));
        }
    }

    clock_t start = clock();
    SC(spvc_compiler_compile(pass_s->sc_comp, &hlsl));

    clock_t after_spvc = clock();
    pl_log_cpu_time(gpu->log, start, after_spvc, "translating SPIR-V to HLSL");

    // Check if each resource binding was actually used by SPIRV-Cross in the
    // compiled HLSL. This information can be used to optimize resource binding
    // to the pipeline.
    for (int i = 0; i < pass->params.num_descriptors; i++) {
        struct pl_desc *desc = &pass->params.descriptors[i];
        struct pl_desc_d3d11 *desc_p = &pass_p->descriptors[i];
        struct d3d_desc_stage *desc_s =
            stage == GLSL_SHADER_VERTEX ? &desc_p->vertex : &desc_p->main;

        // Skip resources that aren't in this shader stage
        if (!desc_s->used)
            continue;

        bool used = spvc_compiler_hlsl_is_resource_used(pass_s->sc_comp,
            spv_execution_model[stage], 0, desc->binding);
        if (!used)
            desc_s->used = false;
    }

    hr = p->D3DCompile(hlsl, strlen(hlsl), NULL, NULL, NULL, "main",
        get_shader_target(gpu, stage),
        D3DCOMPILE_SKIP_VALIDATION | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, out,
        &errors);
    if (FAILED(hr)) {
        PL_ERR(gpu, "D3DCompile failed: %s\n%.*s", pl_hresult_to_str(hr),
               (int) ID3D10Blob_GetBufferSize(errors),
               (char *) ID3D10Blob_GetBufferPointer(errors));
        goto error;
    }

    pl_log_cpu_time(gpu->log, after_spvc, clock(), "translating HLSL to DXBC");

    success = true;
error:;
    int level = success ? PL_LOG_DEBUG : PL_LOG_ERR;
    PL_MSG(gpu, level, "%s shader GLSL source:", shader_names[stage]);
    pl_msg_source(gpu->ctx, level, glsl);
    if (hlsl) {
        PL_MSG(gpu, level, "%s shader HLSL source:", shader_names[stage]);
        pl_msg_source(gpu->ctx, level, hlsl);
    }

    if (pass_s->sc) {
        spvc_context_destroy(pass_s->sc);
        pass_s->sc = NULL;
    }
    SAFE_RELEASE(errors);
    return success;
}

#define CACHE_MAGIC {'P','L','D','3','D',11}
#define CACHE_VERSION 1
static const char d3d11_cache_magic[6] = CACHE_MAGIC;

struct d3d11_cache_header {
    char magic[sizeof(d3d11_cache_magic)];
    int cache_version;
    uint64_t hash;
    bool num_workgroups_used;
    int num_main_cbvs;
    int num_main_srvs;
    int num_main_samplers;
    int num_vertex_cbvs;
    int num_vertex_srvs;
    int num_vertex_samplers;
    int num_uavs;
    size_t vert_bc_len;
    size_t frag_bc_len;
    size_t comp_bc_len;
};

static inline uint64_t pass_cache_signature(pl_gpu gpu,
                                            const struct pl_pass_params *params)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);

    uint64_t hash = p->spirv->signature;

    unsigned spvc_major, spvc_minor, spvc_patch;
    spvc_get_version(&spvc_major, &spvc_minor, &spvc_patch);

    pl_hash_merge(&hash, spvc_major);
    pl_hash_merge(&hash, spvc_minor);
    pl_hash_merge(&hash, spvc_patch);

    pl_hash_merge(&hash, ((uint64_t)p->d3d_compiler_ver.major << 48)
                       | ((uint64_t)p->d3d_compiler_ver.minor << 32)
                       | ((uint64_t)p->d3d_compiler_ver.build << 16)
                       |  (uint64_t)p->d3d_compiler_ver.revision);
    pl_hash_merge(&hash, p->fl);

    pl_hash_merge(&hash, pl_str_hash(pl_str0(params->glsl_shader)));
    if (params->type == PL_PASS_RASTER)
        pl_hash_merge(&hash, pl_str_hash(pl_str0(params->vertex_shader)));

    return hash;
}

static inline size_t cache_payload_size(struct d3d11_cache_header *header)
{
    size_t required = (header->num_main_cbvs + header->num_main_srvs +
                       header->num_main_samplers + header->num_vertex_cbvs +
                       header->num_vertex_srvs + header->num_vertex_samplers +
                       header->num_uavs) * sizeof(int) + header->vert_bc_len +
                       header->frag_bc_len + header->comp_bc_len;

    return required;
}

static bool d3d11_use_cached_program(pl_gpu gpu, struct pl_pass *pass,
                                     const struct pl_pass_params *params,
                                     pl_str *vert_bc, pl_str *frag_bc, pl_str *comp_bc)
{
    struct pl_pass_d3d11 *pass_p = PL_PRIV(pass);

    pl_str cache = {
        .buf = (uint8_t *) params->cached_program,
        .len = params->cached_program_len,
    };

    if (cache.len < sizeof(struct d3d11_cache_header))
        return false;

    struct d3d11_cache_header *header = (struct d3d11_cache_header *) cache.buf;
    cache = pl_str_drop(cache, sizeof(*header));

    if (strncmp(header->magic, d3d11_cache_magic, sizeof(d3d11_cache_magic)) != 0)
        return false;
    if (header->cache_version != CACHE_VERSION)
        return false;
    if (header->hash != pass_cache_signature(gpu, params))
        return false;

    // determine required cache size before reading anything
    size_t required = cache_payload_size(header);

    if (cache.len < required)
        return false;

    pass_p->num_workgroups_used = header->num_workgroups_used;

#define GET_ARRAY(object, name, num_elements) {                     \
    PL_ARRAY_MEMDUP(pass, (object)->name, cache.buf, num_elements); \
    cache = pl_str_drop(cache, num_elements * sizeof(*(object)->name.elem)); }

#define GET_STAGE_ARRAY(stage, name) \
            GET_ARRAY(&pass_p->stage, name, header->num_##stage##_##name)

    GET_STAGE_ARRAY(main, cbvs);
    GET_STAGE_ARRAY(main, srvs);
    GET_STAGE_ARRAY(main, samplers);
    GET_STAGE_ARRAY(vertex, cbvs);
    GET_STAGE_ARRAY(vertex, srvs);
    GET_STAGE_ARRAY(vertex, samplers);
    GET_ARRAY(pass_p, uavs, header->num_uavs);

#define GET_SHADER(ptr)                               \
        *ptr = pl_str_take(cache, header->ptr##_len); \
        cache = pl_str_drop(cache, ptr->len);

    GET_SHADER(vert_bc);
    GET_SHADER(frag_bc);
    GET_SHADER(comp_bc);

    return true;
}

static void d3d11_update_program_cache(pl_gpu gpu, struct pl_pass *pass,
                                       const pl_str *vs_str, const pl_str *ps_str,
                                       const pl_str *cs_str)
{
    struct pl_pass_d3d11 *pass_p = PL_PRIV(pass);

    struct d3d11_cache_header header = {
        .magic = CACHE_MAGIC,
        .cache_version = CACHE_VERSION,
        .hash = pass_cache_signature(gpu, &pass->params),
        .num_workgroups_used = pass_p->num_workgroups_used,
        .num_main_cbvs = pass_p->main.cbvs.num,
        .num_main_srvs = pass_p->main.srvs.num,
        .num_main_samplers = pass_p->main.samplers.num,
        .num_vertex_cbvs = pass_p->vertex.cbvs.num,
        .num_vertex_srvs = pass_p->vertex.srvs.num,
        .num_vertex_samplers = pass_p->vertex.samplers.num,
        .num_uavs = pass_p->uavs.num,
        .vert_bc_len = vs_str ? vs_str->len : 0,
        .frag_bc_len = ps_str ? ps_str->len : 0,
        .comp_bc_len = cs_str ? cs_str->len : 0,
    };

    size_t cache_size = sizeof(header) + cache_payload_size(&header);
    pl_str cache = {0};
    pl_str_append(pass, &cache, (pl_str){ (uint8_t *) &header, sizeof(header) });

#define WRITE_ARRAY(name) pl_str_append(pass, &cache, \
        (pl_str){ (uint8_t *) pass_p->name.elem, \
                  sizeof(*pass_p->name.elem) * pass_p->name.num })
    WRITE_ARRAY(main.cbvs);
    WRITE_ARRAY(main.srvs);
    WRITE_ARRAY(main.samplers);
    WRITE_ARRAY(vertex.cbvs);
    WRITE_ARRAY(vertex.srvs);
    WRITE_ARRAY(vertex.samplers);
    WRITE_ARRAY(uavs);

    if (vs_str)
        pl_str_append(pass, &cache, *vs_str);

    if (ps_str)
        pl_str_append(pass, &cache, *ps_str);

    if (cs_str)
        pl_str_append(pass, &cache, *cs_str);

    pl_assert(cache_size == cache.len);

    pass->params.cached_program = cache.buf;
    pass->params.cached_program_len = cache.len;
}

void pl_d3d11_pass_destroy(pl_gpu gpu, pl_pass pass)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;
    struct pl_pass_d3d11 *pass_p = PL_PRIV(pass);

    if (pass_p->main.sc) {
        spvc_context_destroy(pass_p->main.sc);
        pass_p->main.sc = NULL;
    }
    if (pass_p->vertex.sc) {
        spvc_context_destroy(pass_p->vertex.sc);
        pass_p->vertex.sc = NULL;
    }

    SAFE_RELEASE(pass_p->vs);
    SAFE_RELEASE(pass_p->ps);
    SAFE_RELEASE(pass_p->cs);
    SAFE_RELEASE(pass_p->layout);
    SAFE_RELEASE(pass_p->bstate);

    pl_d3d11_flush_message_queue(ctx, "After pass destroy");

    pl_free((void *) pass);
}

static bool pass_create_raster(pl_gpu gpu, struct pl_pass *pass,
                               const struct pl_pass_params *params,
                               pl_str *vs_bc, pl_str *ps_bc)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;
    struct pl_pass_d3d11 *pass_p = PL_PRIV(pass);
    ID3DBlob *vs_blob = NULL;
    ID3DBlob *ps_blob = NULL;
    const void *vs_bytecode = NULL, *ps_bytecode = NULL;
    size_t vs_length = 0, ps_length = 0;
    D3D11_INPUT_ELEMENT_DESC *in_descs = NULL;
    bool success = false;

    pl_assert((vs_bc == NULL || vs_bc->len == 0) ==
              (ps_bc == NULL || ps_bc->len == 0));
    if (vs_bc == NULL || vs_bc->len == 0) {
        if (!shader_compile_hlsl(gpu, pass, &pass_p->vertex, GLSL_SHADER_VERTEX,
                                 params->vertex_shader, &vs_blob))
            goto error;

        vs_bytecode = ID3D10Blob_GetBufferPointer(vs_blob);
        vs_length = ID3D10Blob_GetBufferSize(vs_blob);

        if (!shader_compile_hlsl(gpu, pass, &pass_p->main, GLSL_SHADER_FRAGMENT,
                                 params->glsl_shader, &ps_blob))
            goto error;

        ps_bytecode = ID3D10Blob_GetBufferPointer(ps_blob);
        ps_length = ID3D10Blob_GetBufferSize(ps_blob);
    } else {
        vs_bytecode = vs_bc->buf;
        vs_length = vs_bc->len;
        ps_bytecode = ps_bc->buf;
        ps_length = ps_bc->len;
    }

    D3D(ID3D11Device_CreateVertexShader(p->dev,
        vs_bytecode, vs_length, NULL, &pass_p->vs));

    D3D(ID3D11Device_CreatePixelShader(p->dev,
        ps_bytecode, ps_length, NULL, &pass_p->ps));

    in_descs = pl_calloc_ptr(pass, params->num_vertex_attribs, in_descs);
    for (int i = 0; i < params->num_vertex_attribs; i++) {
        struct pl_vertex_attrib *va = &params->vertex_attribs[i];

        in_descs[i] = (D3D11_INPUT_ELEMENT_DESC) {
            // The semantic name doesn't mean much and is just used to verify
            // the input description matches the shader. SPIRV-Cross always
            // uses TEXCOORD, so we should too.
            .SemanticName = "TEXCOORD",
            .SemanticIndex = va->location,
            .AlignedByteOffset = va->offset,
            .Format = fmt_to_dxgi(va->fmt),
        };
    }
    D3D(ID3D11Device_CreateInputLayout(p->dev, in_descs,
        params->num_vertex_attribs, vs_bytecode, vs_length, &pass_p->layout));

    static const D3D11_BLEND blend_options[] = {
        [PL_BLEND_ZERO] = D3D11_BLEND_ZERO,
        [PL_BLEND_ONE] = D3D11_BLEND_ONE,
        [PL_BLEND_SRC_ALPHA] = D3D11_BLEND_SRC_ALPHA,
        [PL_BLEND_ONE_MINUS_SRC_ALPHA] = D3D11_BLEND_INV_SRC_ALPHA,
    };

    D3D11_BLEND_DESC bdesc = {
        .RenderTarget[0] = {
            .RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL,
        },
    };
    if (params->blend_params) {
        bdesc.RenderTarget[0] = (D3D11_RENDER_TARGET_BLEND_DESC) {
            .BlendEnable = TRUE,
            .SrcBlend = blend_options[params->blend_params->src_rgb],
            .DestBlend = blend_options[params->blend_params->dst_rgb],
            .BlendOp = D3D11_BLEND_OP_ADD,
            .SrcBlendAlpha = blend_options[params->blend_params->src_alpha],
            .DestBlendAlpha = blend_options[params->blend_params->dst_alpha],
            .BlendOpAlpha = D3D11_BLEND_OP_ADD,
            .RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL,
        };
    }
    D3D(ID3D11Device_CreateBlendState(p->dev, &bdesc, &pass_p->bstate));

    const pl_str vs_str = { (uint8_t *) vs_bytecode, vs_length };
    const pl_str ps_str = { (uint8_t *) ps_bytecode, ps_length };
    d3d11_update_program_cache(gpu, pass, &vs_str, &ps_str, NULL);

    success = true;
error:
    SAFE_RELEASE(vs_blob);
    SAFE_RELEASE(ps_blob);
    pl_free(in_descs);
    return success;
}

static bool pass_create_compute(pl_gpu gpu, struct pl_pass *pass,
                                const struct pl_pass_params *params,
                                pl_str *comp_bc)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;
    struct pl_pass_d3d11 *pass_p = PL_PRIV(pass);
    ID3DBlob *cs_blob = NULL;
    bool success = false;
    const void *cs_bytecode = NULL;
    size_t cs_length = 0;

    if (comp_bc == NULL || comp_bc->len == 0) {
        if (!shader_compile_hlsl(gpu, pass, &pass_p->main, GLSL_SHADER_COMPUTE,
                                 params->glsl_shader, &cs_blob))
            goto error;

        cs_bytecode = ID3D10Blob_GetBufferPointer(cs_blob);
        cs_length = ID3D10Blob_GetBufferSize(cs_blob);
    } else {
        cs_bytecode = comp_bc->buf;
        cs_length = comp_bc->len;
    }

    D3D(ID3D11Device_CreateComputeShader(p->dev,
        cs_bytecode, cs_length,
        NULL, &pass_p->cs));

    if (pass_p->num_workgroups_used) {
        D3D11_BUFFER_DESC bdesc = {
            .BindFlags = D3D11_BIND_CONSTANT_BUFFER,
            .ByteWidth = sizeof(pass_p->last_num_wgs),
        };
        D3D(ID3D11Device_CreateBuffer(p->dev, &bdesc, NULL,
                                      &pass_p->num_workgroups_buf));
    }

    pl_str cs_str = { (uint8_t *) cs_bytecode, cs_length };
    d3d11_update_program_cache(gpu, pass, NULL, NULL, &cs_str);

    success = true;
error:
    SAFE_RELEASE(cs_blob);
    return success;
}

const struct pl_pass *pl_d3d11_pass_create(pl_gpu gpu,
                                           const struct pl_pass_params *params)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;

    struct pl_pass *pass = pl_zalloc_obj(NULL, pass, struct pl_pass_d3d11);
    pass->params = pl_pass_params_copy(pass, params);

    struct pl_pass_d3d11 *pass_p = PL_PRIV(pass);

    pass_p->descriptors = pl_calloc_ptr(pass, params->num_descriptors,
                                        pass_p->descriptors);
    for (int i = 0; i < params->num_descriptors; i++) {
        struct pl_desc_d3d11 *desc_p = &pass_p->descriptors[i];
        *desc_p = (struct pl_desc_d3d11) {
            .main = {
                .cbv_slot = -1,
                .srv_slot = -1,
                .sampler_slot = -1,
                .uav_slot = -1,
            },
            .vertex = {
                .cbv_slot = -1,
                .srv_slot = -1,
                .sampler_slot = -1,
            },
        };
    }

    pl_str vert = {0}, frag = {0}, comp = {0};
    if (d3d11_use_cached_program(gpu, pass, params, &vert, &frag, &comp)) {
        PL_DEBUG(gpu, "Using cached DXBC shaders");
    } else {
        // Compile GLSL to SPIR-V. This also sets `desc_stage.used` based on which
        // resources are statically used in the shader for each pass.
        if (params->type == PL_PASS_RASTER) {
            if (!shader_compile_glsl(gpu, pass, &pass_p->vertex, GLSL_SHADER_VERTEX,
                                     params->vertex_shader))
                goto error;
            if (!shader_compile_glsl(gpu, pass, &pass_p->main, GLSL_SHADER_FRAGMENT,
                                     params->glsl_shader))
                goto error;
        } else {
            if (!shader_compile_glsl(gpu, pass, &pass_p->main, GLSL_SHADER_COMPUTE,
                                     params->glsl_shader))
                goto error;
        }

        // In a raster pass, one of the UAV slots is used by the runtime for the RTV
        int uav_offset = params->type == PL_PASS_COMPUTE ? 0 : 1;
        int max_uavs = p->max_uavs - uav_offset;

        for (int desc_idx = 0; desc_idx < params->num_descriptors; desc_idx++) {
            struct pl_desc *desc = &params->descriptors[desc_idx];
            struct pl_desc_d3d11 *desc_p = &pass_p->descriptors[desc_idx];

            bool has_cbv = false, has_srv = false, has_sampler = false, has_uav = false;

            switch (desc->type) {
            case PL_DESC_SAMPLED_TEX:
                has_sampler = true;
                has_srv = true;
                break;
            case PL_DESC_BUF_STORAGE:
            case PL_DESC_STORAGE_IMG:
            case PL_DESC_BUF_TEXEL_STORAGE:
                if (desc->access == PL_DESC_ACCESS_READONLY) {
                    has_srv = true;
                } else {
                    has_uav = true;
                }
                break;
            case PL_DESC_BUF_UNIFORM:
                has_cbv = true;
                break;
            case PL_DESC_BUF_TEXEL_UNIFORM:
                has_srv = true;
                break;
            case PL_DESC_INVALID:
            case PL_DESC_TYPE_COUNT:
                pl_unreachable();
            }

            // Allocate HLSL register numbers for each shader stage
            struct d3d_pass_stage *stages[] = { &pass_p->main, &pass_p->vertex };
            for (int j = 0; j < PL_ARRAY_SIZE(stages); j++) {
                struct d3d_pass_stage *pass_s = stages[j];
                struct d3d_desc_stage *desc_s =
                    pass_s == &pass_p->vertex ? &desc_p->vertex : &desc_p->main;
                if (!desc_s->used)
                    continue;

                if (has_cbv) {
                    desc_s->cbv_slot = pass_s->cbvs.num;
                    PL_ARRAY_APPEND(pass, pass_s->cbvs, desc_idx);
                    if (pass_s->cbvs.num > D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) {
                        PL_ERR(gpu, "Too many constant buffers in shader");
                        goto error;
                    }
                }

                if (has_srv) {
                    desc_s->srv_slot = pass_s->srvs.num;
                    PL_ARRAY_APPEND(pass, pass_s->srvs, desc_idx);
                    if (pass_s->srvs.num > p->max_srvs) {
                        PL_ERR(gpu, "Too many SRVs in shader");
                        goto error;
                    }
                }

                if (has_sampler) {
                    desc_s->sampler_slot = pass_s->samplers.num;
                    PL_ARRAY_APPEND(pass, pass_s->samplers, desc_idx);
                    if (pass_s->srvs.num > D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT) {
                        PL_ERR(gpu, "Too many samplers in shader");
                        goto error;
                    }
                }
            }

            // UAV bindings are shared between all shader stages
            if (has_uav && (desc_p->main.used || desc_p->vertex.used)) {
                desc_p->main.uav_slot = pass_p->uavs.num + uav_offset;
                PL_ARRAY_APPEND(pass, pass_p->uavs, desc_idx);
                if (pass_p->uavs.num > max_uavs) {
                    PL_ERR(gpu, "Too many UAVs in shader");
                    goto error;
                }
            }
        }
    }

    if (params->type == PL_PASS_COMPUTE) {
        if (!pass_create_compute(gpu, pass, params, &comp))
            goto error;
    } else {
        if (!pass_create_raster(gpu, pass, params, &vert, &frag))
            goto error;
    }

    // Pre-allocate resource arrays to use in pl_pass_run
    pass_p->cbv_arr = pl_calloc(pass,
        PL_MAX(pass_p->main.cbvs.num, pass_p->vertex.cbvs.num),
        sizeof(*pass_p->cbv_arr));
    pass_p->srv_arr = pl_calloc(pass,
        PL_MAX(pass_p->main.srvs.num, pass_p->vertex.srvs.num),
        sizeof(*pass_p->srv_arr));
    pass_p->sampler_arr = pl_calloc(pass,
        PL_MAX(pass_p->main.samplers.num, pass_p->vertex.samplers.num),
        sizeof(*pass_p->sampler_arr));
    pass_p->uav_arr = pl_calloc(pass, pass_p->uavs.num, sizeof(*pass_p->uav_arr));

    pl_d3d11_flush_message_queue(ctx, "After pass create");

    return pass;

error:
    pl_d3d11_pass_destroy(gpu, pass);
    return NULL;
}

// Shared logic between VS, PS and CS for filling the resource arrays that are
// passed to ID3D11DeviceContext methods
static void fill_resources(pl_gpu gpu, pl_pass pass,
                           struct d3d_pass_stage *pass_s,
                           const struct pl_pass_run_params *params,
                           ID3D11Buffer **cbvs, ID3D11ShaderResourceView **srvs,
                           ID3D11SamplerState **samplers)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct pl_pass_d3d11 *pass_p = PL_PRIV(pass);

    for (int i = 0; i < pass_s->cbvs.num; i++) {
        int binding = pass_s->cbvs.elem[i];
        if (binding == HLSL_BINDING_NOT_USED) {
            cbvs[i] = NULL;
            continue;
        } else if (binding == HLSL_BINDING_NUM_WORKGROUPS) {
            cbvs[i] = pass_p->num_workgroups_buf;
            continue;
        }

        pl_buf buf = params->desc_bindings[binding].object;
        pl_d3d11_buf_resolve(gpu, buf);
        struct pl_buf_d3d11 *buf_p = PL_PRIV(buf);
        cbvs[i] = buf_p->buf;
    }

    for (int i = 0; i < pass_s->srvs.num; i++) {
        int binding = pass_s->srvs.elem[i];
        if (binding == HLSL_BINDING_NOT_USED) {
            srvs[i] = NULL;
            continue;
        }

        pl_tex tex;
        struct pl_tex_d3d11 *tex_p;
        pl_buf buf;
        struct pl_buf_d3d11 *buf_p;
        switch (pass->params.descriptors[binding].type) {
        case PL_DESC_SAMPLED_TEX:
        case PL_DESC_STORAGE_IMG:
            tex = params->desc_bindings[binding].object;
            tex_p = PL_PRIV(tex);
            srvs[i] = tex_p->srv;
            break;
        case PL_DESC_BUF_STORAGE:
            buf = params->desc_bindings[binding].object;
            buf_p = PL_PRIV(buf);
            srvs[i] = buf_p->raw_srv;
            break;
        case PL_DESC_BUF_TEXEL_UNIFORM:
        case PL_DESC_BUF_TEXEL_STORAGE:
            buf = params->desc_bindings[binding].object;
            buf_p = PL_PRIV(buf);
            srvs[i] = buf_p->texel_srv;
            break;
        default:
            break;
        }
    }

    for (int i = 0; i < pass_s->samplers.num; i++) {
        int binding = pass_s->samplers.elem[i];
        if (binding == HLSL_BINDING_NOT_USED) {
            samplers[i] = NULL;
            continue;
        }

        struct pl_desc_binding *db = &params->desc_bindings[binding];
        samplers[i] = p->samplers[db->sample_mode][db->address_mode];
    }
}

static void fill_uavs(pl_pass pass, const struct pl_pass_run_params *params,
                      ID3D11UnorderedAccessView **uavs)
{
    struct pl_pass_d3d11 *pass_p = PL_PRIV(pass);

    for (int i = 0; i < pass_p->uavs.num; i++) {
        int binding = pass_p->uavs.elem[i];
        if (binding == HLSL_BINDING_NOT_USED) {
            uavs[i] = NULL;
            continue;
        }

        pl_tex tex;
        struct pl_tex_d3d11 *tex_p;
        pl_buf buf;
        struct pl_buf_d3d11 *buf_p;
        switch (pass->params.descriptors[binding].type) {
        case PL_DESC_BUF_STORAGE:
            buf = params->desc_bindings[binding].object;
            buf_p = PL_PRIV(buf);
            uavs[i] = buf_p->raw_uav;
            break;
        case PL_DESC_STORAGE_IMG:
            tex = params->desc_bindings[binding].object;
            tex_p = PL_PRIV(tex);
            uavs[i] = tex_p->uav;
            break;
        case PL_DESC_BUF_TEXEL_STORAGE:
            buf = params->desc_bindings[binding].object;
            buf_p = PL_PRIV(buf);
            uavs[i] = buf_p->texel_uav;
            break;
        default:
            break;
        }
    }
}

static void pass_run_raster(pl_gpu gpu, const struct pl_pass_run_params *params)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    pl_pass pass = params->pass;
    struct pl_pass_d3d11 *pass_p = PL_PRIV(pass);

    if (p->fl <= D3D_FEATURE_LEVEL_9_3 && params->index_buf) {
        // Index buffers are unsupported because we can't tell if they are an
        // index buffer or a vertex buffer on creation, and FL9_x allows only
        // one binding type per-buffer
        PL_ERR(gpu, "Index buffers are unsupported in FL9_x");
        return;
    }

    if (p->fl <= D3D_FEATURE_LEVEL_9_1 && params->index_data &&
        params->index_fmt != PL_INDEX_UINT16)
    {
        PL_ERR(gpu, "32-bit index format is unsupported in FL9_1");
        return;
    }

    // Figure out how much vertex/index data to upload, if any
    size_t vertex_alloc = params->vertex_data ? pl_vertex_buf_size(params) : 0;
    size_t index_alloc = params->index_data ? pl_index_buf_size(params) : 0;

    static const DXGI_FORMAT index_fmts[PL_INDEX_FORMAT_COUNT] = {
        [PL_INDEX_UINT16] = DXGI_FORMAT_R16_UINT,
        [PL_INDEX_UINT32] = DXGI_FORMAT_R32_UINT,
    };

    // Upload vertex data. On >=FL10_0 we use the same buffer for index data, so
    // upload that too.
    bool share_vertex_index_buf = p->fl > D3D_FEATURE_LEVEL_9_3;
    if (vertex_alloc || (share_vertex_index_buf && index_alloc)) {
        struct stream_buf_slice slices[] = {
            { .data = params->vertex_data, .size = vertex_alloc },
            { .data = params->index_data, .size = index_alloc },
        };

        if (!stream_buf_upload(gpu, &p->vbuf, slices,
                               share_vertex_index_buf ? 2 : 1)) {
            PL_ERR(gpu, "Failed to upload vertex data");
            return;
        }

        if (vertex_alloc) {
            ID3D11DeviceContext_IASetVertexBuffers(p->imm, 0, 1, &p->vbuf.buf,
                &(UINT) { pass->params.vertex_stride }, &slices[0].offset);
        }
        if (share_vertex_index_buf && index_alloc) {
            ID3D11DeviceContext_IASetIndexBuffer(p->imm, p->vbuf.buf,
                index_fmts[params->index_fmt], slices[1].offset);
        }
    }

    // Upload index data for <=FL9_3, which must be in its own buffer
    if (!share_vertex_index_buf && index_alloc) {
        struct stream_buf_slice slices[] = {
            { .data = params->index_data, .size = index_alloc },
        };

        if (!stream_buf_upload(gpu, &p->ibuf, slices, PL_ARRAY_SIZE(slices))) {
            PL_ERR(gpu, "Failed to upload index data");
            return;
        }

        ID3D11DeviceContext_IASetIndexBuffer(p->imm, p->ibuf.buf,
            index_fmts[params->index_fmt], slices[0].offset);
    }

    if (params->vertex_buf) {
        struct pl_buf_d3d11 *buf_p = PL_PRIV(params->vertex_buf);
        ID3D11DeviceContext_IASetVertexBuffers(p->imm, 0, 1, &buf_p->buf,
            &(UINT) { pass->params.vertex_stride },
            &(UINT) { params->buf_offset });
    }

    if (params->index_buf) {
        struct pl_buf_d3d11 *buf_p = PL_PRIV(params->index_buf);
        ID3D11DeviceContext_IASetIndexBuffer(p->imm, buf_p->buf,
            index_fmts[params->index_fmt], params->index_offset);
    }

    ID3D11DeviceContext_IASetInputLayout(p->imm, pass_p->layout);

    static const D3D_PRIMITIVE_TOPOLOGY prim_topology[] = {
        [PL_PRIM_TRIANGLE_LIST] = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
        [PL_PRIM_TRIANGLE_STRIP] = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
    };
    ID3D11DeviceContext_IASetPrimitiveTopology(p->imm,
        prim_topology[pass->params.vertex_type]);

    ID3D11DeviceContext_VSSetShader(p->imm, pass_p->vs, NULL, 0);

    ID3D11Buffer **cbvs = pass_p->cbv_arr;
    ID3D11ShaderResourceView **srvs = pass_p->srv_arr;
    ID3D11SamplerState **samplers = pass_p->sampler_arr;
    ID3D11UnorderedAccessView **uavs = pass_p->uav_arr;

    // Set vertex shader resources. The device context is called conditionally
    // because the debug layer complains if these are called with 0 resources.
    fill_resources(gpu, pass, &pass_p->vertex, params, cbvs, srvs, samplers);
    if (pass_p->vertex.cbvs.num)
        ID3D11DeviceContext_VSSetConstantBuffers(p->imm, 0, pass_p->vertex.cbvs.num, cbvs);
    if (pass_p->vertex.srvs.num)
        ID3D11DeviceContext_VSSetShaderResources(p->imm, 0, pass_p->vertex.srvs.num, srvs);
    if (pass_p->vertex.samplers.num)
        ID3D11DeviceContext_VSSetSamplers(p->imm, 0, pass_p->vertex.samplers.num, samplers);

    ID3D11DeviceContext_RSSetState(p->imm, p->rstate);
    ID3D11DeviceContext_RSSetViewports(p->imm, 1, (&(D3D11_VIEWPORT) {
        .TopLeftX = params->viewport.x0,
        .TopLeftY = params->viewport.y0,
        .Width = pl_rect_w(params->viewport),
        .Height = pl_rect_h(params->viewport),
        .MinDepth = 0,
        .MaxDepth = 1,
    }));
    ID3D11DeviceContext_RSSetScissorRects(p->imm, 1, (&(D3D11_RECT) {
        .left = params->scissors.x0,
        .top = params->scissors.y0,
        .right = params->scissors.x1,
        .bottom = params->scissors.y1,
    }));

    ID3D11DeviceContext_PSSetShader(p->imm, pass_p->ps, NULL, 0);

    // Set pixel shader resources
    fill_resources(gpu, pass, &pass_p->main, params, cbvs, srvs, samplers);
    if (pass_p->main.cbvs.num)
        ID3D11DeviceContext_PSSetConstantBuffers(p->imm, 0, pass_p->main.cbvs.num, cbvs);
    if (pass_p->main.srvs.num)
        ID3D11DeviceContext_PSSetShaderResources(p->imm, 0, pass_p->main.srvs.num, srvs);
    if (pass_p->main.samplers.num)
        ID3D11DeviceContext_PSSetSamplers(p->imm, 0, pass_p->main.samplers.num, samplers);

    ID3D11DeviceContext_OMSetBlendState(p->imm, pass_p->bstate, NULL,
                                        D3D11_DEFAULT_SAMPLE_MASK);
    ID3D11DeviceContext_OMSetDepthStencilState(p->imm, p->dsstate, 0);

    fill_uavs(pass, params, uavs);

    struct pl_tex_d3d11 *target_p = PL_PRIV(params->target);
    ID3D11DeviceContext_OMSetRenderTargetsAndUnorderedAccessViews(
        p->imm, 1, &target_p->rtv, NULL, 1, pass_p->uavs.num, uavs, NULL);

    if (params->index_data || params->index_buf) {
        ID3D11DeviceContext_DrawIndexed(p->imm, params->vertex_count, 0, 0);
    } else {
        ID3D11DeviceContext_Draw(p->imm, params->vertex_count, 0);
    }

    // Unbind everything. It's easier to do this than to actually track state,
    // and if we leave the RTV bound, it could trip up D3D's conflict checker.
    // Also, apparently unbinding SRVs can prevent a 10level9 bug?
    // https://docs.microsoft.com/en-us/windows/win32/direct3d11/overviews-direct3d-11-devices-downlevel-prevent-null-srvs
    for (int i = 0; i < PL_MAX(pass_p->main.cbvs.num, pass_p->vertex.cbvs.num); i++)
        cbvs[i] = NULL;
    for (int i = 0; i < PL_MAX(pass_p->main.srvs.num, pass_p->vertex.srvs.num); i++)
        srvs[i] = NULL;
    for (int i = 0; i < PL_MAX(pass_p->main.samplers.num, pass_p->vertex.samplers.num); i++)
        samplers[i] = NULL;
    for (int i = 0; i < pass_p->uavs.num; i++)
        uavs[i] = NULL;
    if (pass_p->vertex.cbvs.num)
        ID3D11DeviceContext_VSSetConstantBuffers(p->imm, 0, pass_p->vertex.cbvs.num, cbvs);
    if (pass_p->vertex.srvs.num)
        ID3D11DeviceContext_VSSetShaderResources(p->imm, 0, pass_p->vertex.srvs.num, srvs);
    if (pass_p->vertex.samplers.num)
        ID3D11DeviceContext_VSSetSamplers(p->imm, 0, pass_p->vertex.samplers.num, samplers);
    if (pass_p->main.cbvs.num)
        ID3D11DeviceContext_PSSetConstantBuffers(p->imm, 0, pass_p->main.cbvs.num, cbvs);
    if (pass_p->main.srvs.num)
        ID3D11DeviceContext_PSSetShaderResources(p->imm, 0, pass_p->main.srvs.num, srvs);
    if (pass_p->main.samplers.num)
        ID3D11DeviceContext_PSSetSamplers(p->imm, 0, pass_p->main.samplers.num, samplers);
    ID3D11DeviceContext_OMSetRenderTargetsAndUnorderedAccessViews(
        p->imm, 0, NULL, NULL, 1, pass_p->uavs.num, uavs, NULL);
}

static void pass_run_compute(pl_gpu gpu, const struct pl_pass_run_params *params)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    pl_pass pass = params->pass;
    struct pl_pass_d3d11 *pass_p = PL_PRIV(pass);

    // Update gl_NumWorkGroups emulation buffer if necessary
    if (pass_p->num_workgroups_used) {
        bool needs_update = false;
        for (int i = 0; i < 3; i++) {
            if (pass_p->last_num_wgs.num_wgs[i] != params->compute_groups[i])
                needs_update = true;
            pass_p->last_num_wgs.num_wgs[i] = params->compute_groups[i];
        }

        if (needs_update) {
            ID3D11DeviceContext_UpdateSubresource(p->imm,
                (ID3D11Resource *) pass_p->num_workgroups_buf, 0, NULL,
                &pass_p->last_num_wgs, 0, 0);
        }
    }

    ID3D11DeviceContext_CSSetShader(p->imm, pass_p->cs, NULL, 0);

    ID3D11Buffer **cbvs = pass_p->cbv_arr;
    ID3D11ShaderResourceView **srvs = pass_p->srv_arr;
    ID3D11UnorderedAccessView **uavs = pass_p->uav_arr;
    ID3D11SamplerState **samplers = pass_p->sampler_arr;

    fill_resources(gpu, pass, &pass_p->main, params, cbvs, srvs, samplers);
    fill_uavs(pass, params, uavs);

    if (pass_p->main.cbvs.num)
        ID3D11DeviceContext_CSSetConstantBuffers(p->imm, 0, pass_p->main.cbvs.num, cbvs);
    if (pass_p->main.srvs.num)
        ID3D11DeviceContext_CSSetShaderResources(p->imm, 0, pass_p->main.srvs.num, srvs);
    if (pass_p->main.samplers.num)
        ID3D11DeviceContext_CSSetSamplers(p->imm, 0, pass_p->main.samplers.num, samplers);
    if (pass_p->uavs.num)
        ID3D11DeviceContext_CSSetUnorderedAccessViews(p->imm, 0, pass_p->uavs.num, uavs, NULL);

    ID3D11DeviceContext_Dispatch(p->imm, params->compute_groups[0],
                                         params->compute_groups[1],
                                         params->compute_groups[2]);

    // Unbind everything
    for (int i = 0; i < pass_p->main.cbvs.num; i++)
        cbvs[i] = NULL;
    for (int i = 0; i < pass_p->main.srvs.num; i++)
        srvs[i] = NULL;
    for (int i = 0; i < pass_p->main.samplers.num; i++)
        samplers[i] = NULL;
    for (int i = 0; i < pass_p->uavs.num; i++)
        uavs[i] = NULL;
    if (pass_p->main.cbvs.num)
        ID3D11DeviceContext_CSSetConstantBuffers(p->imm, 0, pass_p->main.cbvs.num, cbvs);
    if (pass_p->main.srvs.num)
        ID3D11DeviceContext_CSSetShaderResources(p->imm, 0, pass_p->main.srvs.num, srvs);
    if (pass_p->main.samplers.num)
        ID3D11DeviceContext_CSSetSamplers(p->imm, 0, pass_p->main.samplers.num, samplers);
    if (pass_p->uavs.num)
        ID3D11DeviceContext_CSSetUnorderedAccessViews(p->imm, 0, pass_p->uavs.num, uavs, NULL);
}

void pl_d3d11_pass_run(pl_gpu gpu, const struct pl_pass_run_params *params)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;
    pl_pass pass = params->pass;

    pl_d3d11_timer_start(gpu, params->timer);

    if (pass->params.type == PL_PASS_COMPUTE) {
        pass_run_compute(gpu, params);
    } else {
        pass_run_raster(gpu, params);
    }

    pl_d3d11_timer_end(gpu, params->timer);
    pl_d3d11_flush_message_queue(ctx, "After pass run");
}
