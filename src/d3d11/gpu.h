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

#pragma once

#include <stdalign.h>
#include <d3d11_4.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <spirv_cross_c.h>

#include "../gpu.h"

#include "common.h"
#include "utils.h"

pl_gpu pl_gpu_create_d3d11(struct d3d11_ctx *ctx);

// --- pl_gpu internal structs and helpers

// Size of one constant in a constant buffer
#define CBUF_ELEM (sizeof(float[4]))

struct d3d_stream_buf {
    UINT bind_flags;
    ID3D11Buffer *buf;
    size_t size;
    size_t used;
    unsigned int align;
};

struct pl_gpu_d3d11 {
    struct pl_gpu_fns impl;
    struct d3d11_ctx *ctx;
    ID3D11Device *dev;
    ID3D11Device1 *dev1;
    ID3D11Device5 *dev5;
    ID3D11DeviceContext *imm;
    ID3D11DeviceContext1 *imm1;
    ID3D11DeviceContext4 *imm4;

    struct spirv_compiler *spirv;

    pD3DCompile D3DCompile;
    struct dll_version d3d_compiler_ver;

    // Device capabilities
    D3D_FEATURE_LEVEL fl;
    bool has_timestamp_queries;
    bool has_monitored_fences;

    int max_srvs;
    int max_uavs;

    // This is a pl_dispatch used on ourselves for the purposes of dispatching
    // shaders for performing various emulation tasks (e.g. blits).
    // Warning: As in pl_vk, care must be taken to avoid recursive calls.
    struct pl_dispatch *dp;

    // Streaming vertex and index buffers
    struct d3d_stream_buf vbuf;
    struct d3d_stream_buf ibuf;

    // Shared rasterizer state
    ID3D11RasterizerState *rstate;

    // Shared depth-stencil state
    ID3D11DepthStencilState *dsstate;

    // Array of ID3D11SamplerStates for every combination of sample/address modes
    ID3D11SamplerState *samplers[PL_TEX_SAMPLE_MODE_COUNT][PL_TEX_ADDRESS_MODE_COUNT];

    // Resources for finish()
    ID3D11Fence *finish_fence;
    uint64_t finish_value;
    HANDLE finish_event;
    ID3D11Query *finish_query;
    pl_buf finish_buf_src;
    pl_buf finish_buf_dst;
};

void pl_d3d11_setup_formats(struct pl_gpu *gpu);

void pl_d3d11_timer_start(pl_gpu gpu, pl_timer timer);
void pl_d3d11_timer_end(pl_gpu gpu, pl_timer timer);

struct pl_buf_d3d11 {
    ID3D11Buffer *buf;
    ID3D11Buffer *staging;
    ID3D11ShaderResourceView *raw_srv;
    ID3D11UnorderedAccessView *raw_uav;
    ID3D11ShaderResourceView *texel_srv;
    ID3D11UnorderedAccessView *texel_uav;

    char *data;
    bool dirty;
};

void pl_d3d11_buf_destroy(pl_gpu gpu, pl_buf buf);
pl_buf pl_d3d11_buf_create(pl_gpu gpu, const struct pl_buf_params *params);
void pl_d3d11_buf_write(pl_gpu gpu, pl_buf buf, size_t offset, const void *data,
                        size_t size);
bool pl_d3d11_buf_read(pl_gpu gpu, pl_buf buf, size_t offset, void *dest,
                       size_t size);
void pl_d3d11_buf_copy(pl_gpu gpu, pl_buf dst, size_t dst_offset, pl_buf src,
                       size_t src_offset, size_t size);

// Ensure a buffer is up-to-date with its system memory mirror before it is used
void pl_d3d11_buf_resolve(pl_gpu gpu, pl_buf buf);

struct pl_tex_d3d11 {
    // res mirrors one of tex1d, tex2d or tex3d for convenience. It does not
    // hold an additional reference to the texture object.
    ID3D11Resource *res;

    ID3D11Texture1D *tex1d;
    ID3D11Texture2D *tex2d;
    ID3D11Texture3D *tex3d;
    int array_slice;

    // Mirrors one of staging1d, staging2d, or staging3d, and doesn't hold a ref
    ID3D11Resource *staging;

    // Staging textures for pl_tex_download
    ID3D11Texture1D *staging1d;
    ID3D11Texture2D *staging2d;
    ID3D11Texture3D *staging3d;

    ID3D11ShaderResourceView *srv;
    ID3D11RenderTargetView *rtv;
    ID3D11UnorderedAccessView *uav;
};

void pl_d3d11_tex_destroy(pl_gpu gpu, pl_tex tex);
pl_tex pl_d3d11_tex_create(pl_gpu gpu, const struct pl_tex_params *params);
void pl_d3d11_tex_invalidate(pl_gpu gpu, pl_tex tex);
void pl_d3d11_tex_clear_ex(pl_gpu gpu, pl_tex tex,
                           const union pl_clear_color color);
void pl_d3d11_tex_blit(pl_gpu gpu, const struct pl_tex_blit_params *params);
bool pl_d3d11_tex_upload(pl_gpu gpu, const struct pl_tex_transfer_params *params);
bool pl_d3d11_tex_download(pl_gpu gpu, const struct pl_tex_transfer_params *params);

// Constant buffer layout used for gl_NumWorkGroups emulation
struct d3d_num_workgroups_buf {
    alignas(CBUF_ELEM) uint32_t num_wgs[3];
};

// Represents a descriptor binding to a specific shader stage (VS, PS, CS)
struct d3d_desc_stage {
    // The HLSL register number used for this resource
    int cbv_slot;     // register(bN)
    int srv_slot;     // register(tN)
    int sampler_slot; // register(sN)
    int uav_slot;     // register(uN)

    // Is the resource used in this shader pass? Used to optimize pipeline
    // binding for resources that are used in the vertex shader but not the
    // fragment shader or vice versa.
    bool used;
};

struct pl_desc_d3d11 {
    struct d3d_desc_stage main; // PS and CS
    struct d3d_desc_stage vertex;
};

enum {
    HLSL_BINDING_NOT_USED = -1, // Slot should always be bound as NULL
    HLSL_BINDING_NUM_WORKGROUPS = -2, // Slot used for gl_NumWorkGroups emulation
};

// Represents a specific shader stage in a pl_pass (VS, PS, CS)
struct d3d_pass_stage {
    // GLSL->HLSL translator state
    spvc_context sc;
    spvc_compiler sc_comp;

    // Lists for each resource type, to simplify binding in pl_pass_run. Indexes
    // match the index of the arrays passed to the ID3D11DeviceContext methods.
    // Entries are the index of pass->params.descriptors which should be bound
    // in that position, or a HLSL_BINDING_* special value.
    PL_ARRAY(int) cbvs;
    PL_ARRAY(int) srvs;
    PL_ARRAY(int) samplers;
};

struct pl_pass_d3d11 {
    ID3D11PixelShader *ps;
    ID3D11VertexShader *vs;
    ID3D11ComputeShader *cs;
    ID3D11InputLayout *layout;
    ID3D11BlendState *bstate;

    // gl_NumWorkGroups emulation
    struct d3d_num_workgroups_buf last_num_wgs;
    ID3D11Buffer *num_workgroups_buf;
    bool num_workgroups_used;

    struct pl_desc_d3d11 *descriptors;

    struct d3d_pass_stage main; // PS and CS
    struct d3d_pass_stage vertex;

    // List of resources, as in `struct pass_stage`, except UAVs are shared
    // between all shader stages
    PL_ARRAY(int) uavs;

    // Pre-allocated resource arrays to use in pl_pass_run
    ID3D11Buffer **cbv_arr;
    ID3D11ShaderResourceView **srv_arr;
    ID3D11SamplerState **sampler_arr;
    ID3D11UnorderedAccessView **uav_arr;
};

void pl_d3d11_pass_destroy(pl_gpu gpu, pl_pass pass);
const struct pl_pass *pl_d3d11_pass_create(pl_gpu gpu,
                                           const struct pl_pass_params *params);
void pl_d3d11_pass_run(pl_gpu gpu, const struct pl_pass_run_params *params);
