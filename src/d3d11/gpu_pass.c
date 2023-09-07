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
#include "../cache.h"

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

static SpvExecutionModel stage_to_spv(enum glsl_shader_stage stage)
{
    static const SpvExecutionModel spv_execution_model[] = {
        [GLSL_SHADER_VERTEX]   = SpvExecutionModelVertex,
        [GLSL_SHADER_FRAGMENT] = SpvExecutionModelFragment,
        [GLSL_SHADER_COMPUTE]  = SpvExecutionModelGLCompute,
    };
    return spv_execution_model[stage];
}

#define SC(cmd)                                                             \
    do {                                                                    \
        spvc_result res = (cmd);                                            \
        if (res != SPVC_SUCCESS) {                                          \
            PL_ERR(gpu, "%s: %s (%d) (%s:%d)",                              \
                   #cmd, sc ? spvc_context_get_last_error_string(sc) : "",  \
                   res, __FILE__, __LINE__);                                \
            goto error;                                                     \
        }                                                                   \
    } while (0)

// Some decorations, like SpvDecorationNonWritable, are actually found on the
// members of a buffer block, rather than the buffer block itself. If all
// members have a certain decoration, SPIRV-Cross considers it to apply to the
// buffer block too, which determines things like whether a SRV or UAV is used
// for an SSBO. This function checks if SPIRV-Cross considers a decoration to
// apply to a buffer block.
static spvc_result buffer_block_has_decoration(spvc_compiler sc_comp,
                                               spvc_variable_id id,
                                               SpvDecoration decoration,
                                               bool *out)
{
    const SpvDecoration *decorations;
    size_t num_decorations = 0;

    spvc_result res = spvc_compiler_get_buffer_block_decorations(sc_comp, id,
        &decorations, &num_decorations);
    if (res != SPVC_SUCCESS)
        return res;

    for (size_t j = 0; j < num_decorations; j++) {
        if (decorations[j] == decoration) {
            *out = true;
            return res;
        }
    }

    *out = false;
    return res;
}

static bool alloc_hlsl_reg_bindings(pl_gpu gpu, pl_pass pass,
                                    struct d3d_pass_stage *pass_s,
                                    spvc_context sc,
                                    spvc_compiler sc_comp,
                                    spvc_resources resources,
                                    spvc_resource_type res_type,
                                    enum glsl_shader_stage stage)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct pl_pass_d3d11 *pass_p = PL_PRIV(pass);
    const spvc_reflected_resource *res_list;
    size_t res_count;

    SC(spvc_resources_get_resource_list_for_type(resources, res_type,
                                                 &res_list, &res_count));

    // In a raster pass, one of the UAV slots is used by the runtime for the RTV
    int uav_offset = stage == GLSL_SHADER_COMPUTE ? 0 : 1;
    int max_uavs = p->max_uavs - uav_offset;

    for (int i = 0; i < res_count; i++) {
        unsigned int binding = spvc_compiler_get_decoration(sc_comp,
            res_list[i].id, SpvDecorationBinding);
        unsigned int descriptor_set = spvc_compiler_get_decoration(sc_comp,
            res_list[i].id, SpvDecorationDescriptorSet);
        if (descriptor_set != 0)
            continue;

        pass_p->max_binding = PL_MAX(pass_p->max_binding, binding);

        spvc_hlsl_resource_binding hlslbind;
        spvc_hlsl_resource_binding_init(&hlslbind);
        hlslbind.stage = stage_to_spv(stage);
        hlslbind.binding = binding;
        hlslbind.desc_set = descriptor_set;

        bool has_cbv = false, has_sampler = false, has_srv = false, has_uav = false;
        switch (res_type) {
        case SPVC_RESOURCE_TYPE_UNIFORM_BUFFER:
            has_cbv = true;
            break;
        case SPVC_RESOURCE_TYPE_STORAGE_BUFFER:;
            bool non_writable_bb = false;
            SC(buffer_block_has_decoration(sc_comp, res_list[i].id,
                SpvDecorationNonWritable, &non_writable_bb));
            if (non_writable_bb) {
                has_srv = true;
            } else {
                has_uav = true;
            }
            break;
        case SPVC_RESOURCE_TYPE_STORAGE_IMAGE:;
            bool non_writable = spvc_compiler_has_decoration(sc_comp,
                res_list[i].id, SpvDecorationNonWritable);
            if (non_writable) {
                has_srv = true;
            } else {
                has_uav = true;
            }
            break;
        case SPVC_RESOURCE_TYPE_SEPARATE_IMAGE:
            has_srv = true;
            break;
        case SPVC_RESOURCE_TYPE_SAMPLED_IMAGE:;
            spvc_type type = spvc_compiler_get_type_handle(sc_comp,
                                                           res_list[i].type_id);
            SpvDim dimension = spvc_type_get_image_dimension(type);
            // Uniform texel buffers are technically sampled images, but they
            // aren't sampled from, so don't allocate a sampler
            if (dimension != SpvDimBuffer)
                has_sampler = true;
            has_srv = true;
            break;
        default:
            break;
        }

        if (has_cbv) {
            hlslbind.cbv.register_binding = pass_s->cbvs.num;
            PL_ARRAY_APPEND(pass, pass_s->cbvs, binding);
            if (pass_s->cbvs.num > D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) {
                PL_ERR(gpu, "Too many constant buffers in shader");
                goto error;
            }
        }

        if (has_sampler) {
            hlslbind.sampler.register_binding = pass_s->samplers.num;
            PL_ARRAY_APPEND(pass, pass_s->samplers, binding);
            if (pass_s->samplers.num > D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT) {
                PL_ERR(gpu, "Too many samplers in shader");
                goto error;
            }
        }

        if (has_srv) {
            hlslbind.srv.register_binding = pass_s->srvs.num;
            PL_ARRAY_APPEND(pass, pass_s->srvs, binding);
            if (pass_s->srvs.num > p->max_srvs) {
                PL_ERR(gpu, "Too many SRVs in shader");
                goto error;
            }
        }

        if (has_uav) {
            // UAV registers are shared between the vertex and fragment shaders
            // in a raster pass, so check if the UAV for this resource has
            // already been allocated
            bool uav_bound = false;
            for (int j = 0; j < pass_p->uavs.num; j++) {
                if (pass_p->uavs.elem[j] == binding) {
                    uav_bound = true;
                    break;
                }
            }

            if (!uav_bound) {
                hlslbind.uav.register_binding = pass_p->uavs.num + uav_offset;
                PL_ARRAY_APPEND(pass, pass_p->uavs, binding);
                if (pass_p->uavs.num > max_uavs) {
                    PL_ERR(gpu, "Too many UAVs in shader");
                    goto error;
                }
            }
        }

        SC(spvc_compiler_hlsl_add_resource_binding(sc_comp, &hlslbind));
    }

    return true;
error:
    return false;
}

static const char *shader_names[] = {
    [GLSL_SHADER_VERTEX]   = "vertex",
    [GLSL_SHADER_FRAGMENT] = "fragment",
    [GLSL_SHADER_COMPUTE]  = "compute",
};

static ID3DBlob *shader_compile_glsl(pl_gpu gpu, pl_pass pass,
                                     struct d3d_pass_stage *pass_s,
                                     enum glsl_shader_stage stage,
                                     const char *glsl)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct pl_pass_d3d11 *pass_p = PL_PRIV(pass);
    void *tmp = pl_tmp(NULL);
    spvc_context sc = NULL;
    spvc_compiler sc_comp = NULL;
    const char *hlsl = NULL;
    ID3DBlob *out = NULL;
    ID3DBlob *errors = NULL;
    HRESULT hr;

    pl_clock_t start = pl_clock_now();
    pl_str spirv = pl_spirv_compile_glsl(p->spirv, tmp, gpu->glsl, stage, glsl);
    if (!spirv.len)
        goto error;

    pl_clock_t after_glsl = pl_clock_now();
    pl_log_cpu_time(gpu->log, start, after_glsl, "translating GLSL to SPIR-V");

    SC(spvc_context_create(&sc));

    spvc_parsed_ir sc_ir;
    SC(spvc_context_parse_spirv(sc, (SpvId *) spirv.buf,
                                spirv.len / sizeof(SpvId), &sc_ir));

    SC(spvc_context_create_compiler(sc, SPVC_BACKEND_HLSL, sc_ir,
                                    SPVC_CAPTURE_MODE_TAKE_OWNERSHIP,
                                    &sc_comp));

    spvc_compiler_options sc_opts;
    SC(spvc_compiler_create_compiler_options(sc_comp, &sc_opts));

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

    SC(spvc_compiler_install_compiler_options(sc_comp, sc_opts));

    spvc_set active = NULL;
    SC(spvc_compiler_get_active_interface_variables(sc_comp, &active));
    spvc_resources resources = NULL;
    SC(spvc_compiler_create_shader_resources_for_active_variables(
        sc_comp, &resources, active));

    // Allocate HLSL registers for each resource type
    alloc_hlsl_reg_bindings(gpu, pass, pass_s, sc, sc_comp, resources,
                            SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, stage);
    alloc_hlsl_reg_bindings(gpu, pass, pass_s, sc, sc_comp, resources,
                            SPVC_RESOURCE_TYPE_SEPARATE_IMAGE, stage);
    alloc_hlsl_reg_bindings(gpu, pass, pass_s, sc, sc_comp, resources,
                            SPVC_RESOURCE_TYPE_UNIFORM_BUFFER, stage);
    alloc_hlsl_reg_bindings(gpu, pass, pass_s, sc, sc_comp, resources,
                            SPVC_RESOURCE_TYPE_STORAGE_BUFFER, stage);
    alloc_hlsl_reg_bindings(gpu, pass, pass_s, sc, sc_comp, resources,
                            SPVC_RESOURCE_TYPE_STORAGE_IMAGE, stage);

    if (stage == GLSL_SHADER_COMPUTE) {
        // Check if the gl_NumWorkGroups builtin is used. If it is, we have to
        // emulate it with a constant buffer, so allocate it a CBV register.
        spvc_variable_id num_workgroups_id =
            spvc_compiler_hlsl_remap_num_workgroups_builtin(sc_comp);
        if (num_workgroups_id) {
            pass_p->num_workgroups_used = true;

            spvc_hlsl_resource_binding binding;
            spvc_hlsl_resource_binding_init(&binding);
            binding.stage = stage_to_spv(stage);
            binding.binding = pass_p->max_binding + 1;

            // Allocate a CBV register for the buffer
            binding.cbv.register_binding = pass_s->cbvs.num;
            PL_ARRAY_APPEND(pass, pass_s->cbvs, HLSL_BINDING_NUM_WORKGROUPS);
            if (pass_s->cbvs.num >
                    D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) {
                PL_ERR(gpu, "Not enough constant buffer slots for gl_NumWorkGroups");
                goto error;
            }

            spvc_compiler_set_decoration(sc_comp, num_workgroups_id,
                                         SpvDecorationDescriptorSet, 0);
            spvc_compiler_set_decoration(sc_comp, num_workgroups_id,
                                         SpvDecorationBinding, binding.binding);

            SC(spvc_compiler_hlsl_add_resource_binding(sc_comp, &binding));
        }
    }

    SC(spvc_compiler_compile(sc_comp, &hlsl));

    pl_clock_t after_spvc = pl_clock_now();
    pl_log_cpu_time(gpu->log, after_glsl, after_spvc, "translating SPIR-V to HLSL");

    hr = p->D3DCompile(hlsl, strlen(hlsl), NULL, NULL, NULL, "main",
        get_shader_target(gpu, stage),
        D3DCOMPILE_SKIP_VALIDATION | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &out,
        &errors);
    if (FAILED(hr)) {
        SAFE_RELEASE(out);
        PL_ERR(gpu, "D3DCompile failed: %s\n%.*s", pl_hresult_to_str(hr),
               (int) ID3D10Blob_GetBufferSize(errors),
               (char *) ID3D10Blob_GetBufferPointer(errors));
        goto error;
    }

    pl_log_cpu_time(gpu->log, after_spvc, pl_clock_now(), "translating HLSL to DXBC");

error:;
    if (hlsl) {
        int level = out ? PL_LOG_DEBUG : PL_LOG_ERR;
        PL_MSG(gpu, level, "%s shader HLSL source:", shader_names[stage]);
        pl_msg_source(gpu->log, level, hlsl);
    }

    if (sc)
        spvc_context_destroy(sc);
    SAFE_RELEASE(errors);
    pl_free(tmp);
    return out;
}

struct d3d11_cache_header {
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

static inline uint64_t pass_cache_signature(pl_gpu gpu, uint64_t *key,
                                            const struct pl_pass_params *params)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);

    uint64_t hash = CACHE_KEY_D3D_DXBC; // seed to uniquely identify d3d11 shaders

    pl_hash_merge(&hash, pl_str0_hash(params->glsl_shader));
    if (params->type == PL_PASS_RASTER)
        pl_hash_merge(&hash, pl_str0_hash(params->vertex_shader));

    // store hash based on the shader bodys as the lookup key
    if (key)
        *key = hash;

    // and add the compiler version information into the verification signature
    pl_hash_merge(&hash, p->spirv->signature);

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

static bool d3d11_use_cached_program(pl_gpu gpu, struct pl_pass_t *pass,
                                     const struct pl_pass_params *params,
                                     pl_cache_obj *obj, uint64_t *out_sig,
                                     pl_str *vert_bc, pl_str *frag_bc, pl_str *comp_bc)
{
    struct pl_pass_d3d11 *pass_p = PL_PRIV(pass);
    const pl_cache gpu_cache = pl_gpu_cache(gpu);
    if (!gpu_cache)
        return false;

    *out_sig = pass_cache_signature(gpu, &obj->key, params);
    if (!pl_cache_get(gpu_cache, obj))
        return false;

    pl_str cache = (pl_str) { obj->data, obj->size };
    if (cache.len < sizeof(struct d3d11_cache_header))
        return false;

    struct d3d11_cache_header *header = (struct d3d11_cache_header *) cache.buf;
    cache = pl_str_drop(cache, sizeof(*header));

    if (header->hash != *out_sig)
        return false;

    // determine required cache size before reading anything
    size_t required = cache_payload_size(header);

    if (cache.len < required)
        return false;

    pass_p->num_workgroups_used = header->num_workgroups_used;

#define GET_ARRAY(object, name, num_elems)                                     \
    do {                                                                       \
        PL_ARRAY_MEMDUP(pass, (object)->name, cache.buf, num_elems);           \
        cache = pl_str_drop(cache, num_elems * sizeof(*(object)->name.elem));  \
    } while (0)

#define GET_STAGE_ARRAY(stage, name) \
            GET_ARRAY(&pass_p->stage, name, header->num_##stage##_##name)

    GET_STAGE_ARRAY(main, cbvs);
    GET_STAGE_ARRAY(main, srvs);
    GET_STAGE_ARRAY(main, samplers);
    GET_STAGE_ARRAY(vertex, cbvs);
    GET_STAGE_ARRAY(vertex, srvs);
    GET_STAGE_ARRAY(vertex, samplers);
    GET_ARRAY(pass_p, uavs, header->num_uavs);

#define GET_SHADER(ptr)                                    \
    do {                                                   \
        if (ptr)                                           \
            *ptr = pl_str_take(cache, header->ptr##_len);  \
        cache = pl_str_drop(cache, header->ptr##_len);     \
    } while (0)

    GET_SHADER(vert_bc);
    GET_SHADER(frag_bc);
    GET_SHADER(comp_bc);

    return true;
}

static void d3d11_update_program_cache(pl_gpu gpu, struct pl_pass_t *pass,
                                       uint64_t key, uint64_t sig,
                                       const pl_str *vs_str, const pl_str *ps_str,
                                       const pl_str *cs_str)
{
    struct pl_pass_d3d11 *pass_p = PL_PRIV(pass);
    const pl_cache gpu_cache = pl_gpu_cache(gpu);
    if (!gpu_cache)
        return;

    struct d3d11_cache_header header = {
        .hash = sig,
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
    pl_str_append(NULL, &cache, (pl_str){ (uint8_t *) &header, sizeof(header) });

#define WRITE_ARRAY(name) pl_str_append(NULL, &cache, \
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
        pl_str_append(NULL, &cache, *vs_str);

    if (ps_str)
        pl_str_append(NULL, &cache, *ps_str);

    if (cs_str)
        pl_str_append(NULL, &cache, *cs_str);

    pl_assert(cache_size == cache.len);
    pl_cache_str(gpu_cache, key, &cache);
}

void pl_d3d11_pass_destroy(pl_gpu gpu, pl_pass pass)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;
    struct pl_pass_d3d11 *pass_p = PL_PRIV(pass);

    SAFE_RELEASE(pass_p->vs);
    SAFE_RELEASE(pass_p->ps);
    SAFE_RELEASE(pass_p->cs);
    SAFE_RELEASE(pass_p->layout);
    SAFE_RELEASE(pass_p->bstate);
    SAFE_RELEASE(pass_p->num_workgroups_buf);

    pl_d3d11_flush_message_queue(ctx, "After pass destroy");

    pl_free((void *) pass);
}

static bool pass_create_raster(pl_gpu gpu, struct pl_pass_t *pass,
                               const struct pl_pass_params *params)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;
    struct pl_pass_d3d11 *pass_p = PL_PRIV(pass);
    ID3DBlob *vs_blob = NULL;
    pl_str vs_str = {0};
    ID3DBlob *ps_blob = NULL;
    pl_str ps_str = {0};
    D3D11_INPUT_ELEMENT_DESC *in_descs = NULL;
    pl_cache_obj obj = {0};
    uint64_t sig = 0;
    bool success = false;

    if (d3d11_use_cached_program(gpu, pass, params, &obj, &sig, &vs_str, &ps_str, NULL))
        PL_DEBUG(gpu, "Using cached DXBC shaders");

    pl_assert((vs_str.len == 0) == (ps_str.len == 0));
    if (vs_str.len == 0) {
        vs_blob = shader_compile_glsl(gpu, pass, &pass_p->vertex,
                                      GLSL_SHADER_VERTEX, params->vertex_shader);
        if (!vs_blob)
            goto error;

        vs_str = (pl_str) {
            .buf = ID3D10Blob_GetBufferPointer(vs_blob),
            .len = ID3D10Blob_GetBufferSize(vs_blob),
        };

        ps_blob = shader_compile_glsl(gpu, pass, &pass_p->main,
                                      GLSL_SHADER_FRAGMENT, params->glsl_shader);
        if (!ps_blob)
            goto error;

        ps_str = (pl_str) {
            .buf = ID3D10Blob_GetBufferPointer(ps_blob),
            .len = ID3D10Blob_GetBufferSize(ps_blob),
        };
    }

    D3D(ID3D11Device_CreateVertexShader(p->dev, vs_str.buf, vs_str.len, NULL,
                                        &pass_p->vs));

    D3D(ID3D11Device_CreatePixelShader(p->dev, ps_str.buf, ps_str.len, NULL,
                                       &pass_p->ps));

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
        params->num_vertex_attribs, vs_str.buf, vs_str.len, &pass_p->layout));

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

    d3d11_update_program_cache(gpu, pass, obj.key, sig, &vs_str, &ps_str, NULL);

    success = true;
error:
    SAFE_RELEASE(vs_blob);
    SAFE_RELEASE(ps_blob);
    pl_cache_obj_free(&obj);
    pl_free(in_descs);
    return success;
}

static bool pass_create_compute(pl_gpu gpu, struct pl_pass_t *pass,
                                const struct pl_pass_params *params)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;
    struct pl_pass_d3d11 *pass_p = PL_PRIV(pass);
    ID3DBlob *cs_blob = NULL;
    pl_str cs_str = {0};
    pl_cache_obj obj = {0};
    uint64_t sig = 0;
    bool success = false;

    if (d3d11_use_cached_program(gpu, pass, params, &obj, &sig, NULL, NULL, &cs_str))
        PL_DEBUG(gpu, "Using cached DXBC shader");

    if (cs_str.len == 0) {
        cs_blob = shader_compile_glsl(gpu, pass, &pass_p->main,
                                      GLSL_SHADER_COMPUTE, params->glsl_shader);
        if (!cs_blob)
            goto error;

        cs_str = (pl_str) {
            .buf = ID3D10Blob_GetBufferPointer(cs_blob),
            .len = ID3D10Blob_GetBufferSize(cs_blob),
        };
    }

    D3D(ID3D11Device_CreateComputeShader(p->dev, cs_str.buf, cs_str.len, NULL,
                                         &pass_p->cs));

    if (pass_p->num_workgroups_used) {
        D3D11_BUFFER_DESC bdesc = {
            .BindFlags = D3D11_BIND_CONSTANT_BUFFER,
            .ByteWidth = sizeof(pass_p->last_num_wgs),
        };
        D3D(ID3D11Device_CreateBuffer(p->dev, &bdesc, NULL,
                                      &pass_p->num_workgroups_buf));
    }

    d3d11_update_program_cache(gpu, pass, obj.key, sig, NULL, NULL, &cs_str);

    success = true;
error:
    pl_cache_obj_free(&obj);
    SAFE_RELEASE(cs_blob);
    return success;
}

const struct pl_pass_t *pl_d3d11_pass_create(pl_gpu gpu,
                                             const struct pl_pass_params *params)
{
    struct pl_gpu_d3d11 *p = PL_PRIV(gpu);
    struct d3d11_ctx *ctx = p->ctx;

    struct pl_pass_t *pass = pl_zalloc_obj(NULL, pass, struct pl_pass_d3d11);
    pass->params = pl_pass_params_copy(pass, params);
    struct pl_pass_d3d11 *pass_p = PL_PRIV(pass);
    *pass_p = (struct pl_pass_d3d11) {
        .max_binding = -1,
    };

    if (params->type == PL_PASS_COMPUTE) {
        if (!pass_create_compute(gpu, pass, params))
            goto error;
    } else {
        if (!pass_create_raster(gpu, pass, params))
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

    // Find the highest binding number used in `params->descriptors` if we
    // haven't found it already. (If the shader was compiled fresh rather than
    // loaded from cache, `pass_p->max_binding` should already be set.)
    if (pass_p->max_binding == -1) {
        for (int i = 0; i < params->num_descriptors; i++) {
            pass_p->max_binding = PL_MAX(pass_p->max_binding,
                                         params->descriptors[i].binding);
        }
    }

    // Build a mapping from binding numbers to descriptor array indexes
    int *binding_map = pl_calloc_ptr(pass, pass_p->max_binding + 1, binding_map);
    for (int i = 0; i <= pass_p->max_binding; i++)
        binding_map[i] = HLSL_BINDING_NOT_USED;
    for (int i = 0; i < params->num_descriptors; i++)
        binding_map[params->descriptors[i].binding] = i;

#define MAP_RESOURCES(array)                                 \
    do {                                                     \
        for (int i = 0; i < array.num; i++) {                \
            if (array.elem[i] > pass_p->max_binding) {       \
                array.elem[i] = HLSL_BINDING_NOT_USED;       \
            } else if (array.elem[i] >= 0) {                 \
                array.elem[i] = binding_map[array.elem[i]];  \
            }                                                \
        }                                                    \
    } while (0)

    // During shader compilation (or after loading a compiled shader from cache)
    // the entries of the following resource lists are shader binding numbers,
    // however, it's more efficient for `pl_pass_run` if they refer to indexes
    // of the `params->descriptors` array instead, so remap them here
    MAP_RESOURCES(pass_p->main.cbvs);
    MAP_RESOURCES(pass_p->main.samplers);
    MAP_RESOURCES(pass_p->main.srvs);
    MAP_RESOURCES(pass_p->vertex.cbvs);
    MAP_RESOURCES(pass_p->vertex.samplers);
    MAP_RESOURCES(pass_p->vertex.srvs);
    MAP_RESOURCES(pass_p->uavs);
    pl_free(binding_map);

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
        if (binding == HLSL_BINDING_NUM_WORKGROUPS) {
            cbvs[i] = pass_p->num_workgroups_buf;
            continue;
        } else if (binding < 0) {
            cbvs[i] = NULL;
            continue;
        }

        pl_buf buf = params->desc_bindings[binding].object;
        pl_d3d11_buf_resolve(gpu, buf);
        struct pl_buf_d3d11 *buf_p = PL_PRIV(buf);
        cbvs[i] = buf_p->buf;
    }

    for (int i = 0; i < pass_s->srvs.num; i++) {
        int binding = pass_s->srvs.elem[i];
        if (binding < 0) {
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
        if (binding < 0) {
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
        if (binding < 0) {
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
