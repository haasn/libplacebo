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

#ifndef LIBPLACEBO_GPU_H_
#define LIBPLACEBO_GPU_H_

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#include <libplacebo/common.h>

// This file contains the definition of an API which is designed to abstract
// away from platform-specific APIs like the various OpenGL variants, Direct3D
// and Vulkan in a common way. It is a much more limited API than those APIs,
// since it tries targetting a very small common subset of features that is
// needed to implement libplacebo's rendering.
//
// NOTE: When speaking of "valid usage" or "must", invalid usage is assumed to
// result in undefined behavior. (Typically, an error message is printed to
// stderr and libplacebo aborts - but this is not guaranteed). So ensuring
// valid API usage by the API user is absolutely crucial. If you want to be
// freed from this reponsibility, use the higher level abstractions provided by
// libplacebo alongside gpu.h.

// Structure which wraps metadata describing GLSL capabilities.
struct pl_glsl_desc {
    int version;        // GLSL version (e.g. 450), for #version
    bool gles;          // GLSL ES semantics (ESSL)
    bool vulkan;        // GL_KHR_vulkan_glsl semantics
};

typedef uint64_t pl_gpu_caps;
enum {
    PL_GPU_CAP_COMPUTE          = 1 << 0, // supports compute shaders
    PL_GPU_CAP_PARALLEL_COMPUTE = 1 << 1, // supports multiple compute queues
    PL_GPU_CAP_INPUT_VARIABLES  = 1 << 2, // supports shader input variables
};

// Some `pl_gpu` operations allow sharing resources (memory or images) with
// external APIs - examples include interop with other graphics APIs such as
// CUDA, and also various hardware decoding APIs. This defines the mechanism
// underpinning the communication of such an interoperation.
typedef uint64_t pl_handle_types;
enum {
    PL_HANDLE_FD    = (1 << 0), // `int fd` for POSIX-style APIs
};

struct pl_gpu_handle {
    size_t size;    // the total size of the memory referenced by this handle
    // List of all requested handles:
    int fd;         // PL_HANDLE_FD
};

// Structure defining the physical limits of this GPU instance. If a limit is
// given as 0, that means that feature is unsupported.
struct pl_gpu_limits {
    uint32_t max_tex_1d_dim;    // maximum width for a 1D texture
    uint32_t max_tex_2d_dim;    // maximum width/height for a 2D texture (required)
    uint32_t max_tex_3d_dim;    // maximum width/height/depth for a 3D texture
    size_t max_pushc_size;      // maximum push_constants_size
    size_t max_xfer_size;       // maximum size of a PL_BUF_TEX_TRANSFER
    size_t max_ubo_size;        // maximum size of a PL_BUF_UNIFORM
    size_t max_ssbo_size;       // maximum size of a PL_BUF_STORAGE
    uint64_t max_buffer_texels; // maximum texels in a PL_BUF_TEXEL_*
    int16_t min_gather_offset;  // minimum textureGatherOffset offset
    int16_t max_gather_offset;  // maximum textureGatherOffset offset

    // Compute shader limits. Always available (non-zero) if PL_GPU_CAP_COMPUTE set
    size_t max_shmem_size;      // maximum compute shader shared memory size
    uint32_t max_group_threads; // maximum number of local threads per work group
    uint32_t max_group_size[3]; // maximum work group size per dimension
    uint32_t max_dispatch[3];   // maximum dispatch size per dimension

    // These don't represent hard limits but indicate performance hints for
    // optimal alignment. For best performance, the corresponding field
    // should be aligned to a multiple of these. They will always be a power
    // of two.
    uint32_t align_tex_xfer_stride; // optimal pl_tex_transfer_params.stride_w/h
    size_t align_tex_xfer_offset;   // optimal pl_tex_transfer_params.buf_offset
};

// Abstract device context which wraps an underlying graphics context and can
// be used to dispatch rendering commands.
struct pl_gpu {
    struct pl_context *ctx;  // the pl_context this GPU was initialized from
    struct pl_gpu_fns *impl; // the underlying implementation (unique per GPU)
    void *priv;

    pl_gpu_caps caps;            // PL_GPU_CAP_* bit field
    struct pl_glsl_desc glsl;    // GLSL version supported by this GPU
    struct pl_gpu_limits limits; // physical device limits
    pl_handle_types handle_caps; // supported handle types for external memory
    // Note: Every GPU must support at least one of PL_GPU_CAP_INPUT_VARIABLES
    // or uniform buffers (limits.max_ubo_size > 0).

    // Supported texture formats, in preference order. (If there are multiple
    // similar formats, the "better" ones come first)
    const struct pl_fmt **formats;
    int num_formats;
};

// Helper function to align the given dimension (e.g. width or height) to a
// multiple of the optimal texture transfer stride.
int pl_optimal_transfer_stride(const struct pl_gpu *gpu, int dimension);

enum pl_fmt_type {
    PL_FMT_UNKNOWN = 0, // also used for inconsistent multi-component formats
    PL_FMT_UNORM,       // unsigned, normalized integer format (sampled as float)
    PL_FMT_SNORM,       // signed, normalized integer format (sampled as float)
    PL_FMT_UINT,        // unsigned integer format (sampled as integer)
    PL_FMT_SINT,        // signed integer format (sampled as integer)
    PL_FMT_FLOAT,       // (signed) float formats, any bit size
    PL_FMT_TYPE_COUNT,
};

enum pl_fmt_caps {
    PL_FMT_CAP_SAMPLEABLE   = 1 << 0, // may be sampled from (PL_DESC_SAMPLED_TEX)
    PL_FMT_CAP_STORABLE     = 1 << 1, // may be used as storage image (PL_DESC_STORAGE_IMG)
    PL_FMT_CAP_LINEAR       = 1 << 2, // may be linearly samplied from (PL_TEX_SAMPLE_LINEAR)
    PL_FMT_CAP_RENDERABLE   = 1 << 3, // may be rendered to (pl_pass_params.target_fmt)
    PL_FMT_CAP_BLENDABLE    = 1 << 4, // may be blended to (pl_pass_params.enable_blend)
    PL_FMT_CAP_BLITTABLE    = 1 << 5, // may be blitted from/to (pl_tex_blit)
    PL_FMT_CAP_VERTEX       = 1 << 6, // may be used as a vertex attribute
    PL_FMT_CAP_TEXEL_UNIFORM = 1 << 7, // may be used as a texel uniform buffer
    PL_FMT_CAP_TEXEL_STORAGE = 1 << 8, // may be used as a texel storage buffer

    // Notes:
    // - PL_FMT_CAP_LINEAR also implies PL_FMT_CAP_SAMPLEABLE
    // - PL_FMT_CAP_STORABLE also implies PL_GPU_CAP_COMPUTE
    // - PL_FMT_CAP_VERTEX implies that the format is non-opaque
};

// Structure describing a texel/vertex format.
struct pl_fmt {
    const char *name;       // symbolic name for this format (e.g. rgba32f)
    const void *priv;

    enum pl_fmt_type type;  // the format's data type and interpretation
    enum pl_fmt_caps caps;  // the features supported by this format
    int num_components;     // number of components for this format
    int component_depth[4]; // meaningful bits per component, texture precision

    // This controls the relationship between the data as seen by the host and
    // the way it's interpreted by the texture. The host representation is
    // always tightly packed (no padding bits in between each component).
    //
    // If `opaque` is true, then there's no meaningful correspondence between
    // the two, and all of the remaining fields in this section are unset.
    //
    // If `emulated` is true, then this format doesn't actually exist on the
    // GPU as an uploadable texture format - and any apparent support is being
    // emulated (typically using compute shaders in the upload path).
    bool opaque;
    bool emulated;
    size_t texel_size;      // total size in bytes per texel
    int host_bits[4];       // number of meaningful bits in host memory
    int sample_order[4];    // sampled index for each component, e.g.
                            // {2, 1, 0, 3} for BGRA textures

    // If usable as a vertex or texel buffer format, this gives the GLSL type
    // corresponding to the data. (e.g. vec4)
    const char *glsl_type;

    // If usable as a storage image or texel storage buffer
    // (PL_FMT_CAP_STORABLE / PL_FMT_CAP_TEXEL_STORAGE), this gives the GLSL
    // texel format corresponding to the format. (e.g. rgba16ui)
    const char *glsl_format;
};

// Returns whether or not a pl_fmt's components are ordered sequentially
// in memory in the order RGBA.
bool pl_fmt_is_ordered(const struct pl_fmt *fmt);

// Helper function to find a format with a given number of components and
// minimum effective precision per component. If `host_bits` is set, then the
// format will always be non-opaque, unpadded, ordered and have exactly this
// bit depth for each component. Finally, all `caps` must be supported.
const struct pl_fmt *pl_find_fmt(const struct pl_gpu *gpu, enum pl_fmt_type type,
                                 int num_components, int min_depth,
                                 int host_bits, enum pl_fmt_caps caps);

// Finds a vertex format for a given configuration. The resulting vertex will
// have a component depth equivalent to to the sizeof() the equivalent host type.
// (e.g. PL_FMT_FLOAT will always have sizeof(float))
const struct pl_fmt *pl_find_vertex_fmt(const struct pl_gpu *gpu,
                                        enum pl_fmt_type type,
                                        int num_components);

// Find a format based on its name.
const struct pl_fmt *pl_find_named_fmt(const struct pl_gpu *gpu, const char *name);

enum pl_tex_sample_mode {
    PL_TEX_SAMPLE_NEAREST,  // nearest neighour sampling
    PL_TEX_SAMPLE_LINEAR,   // linear filtering
};

enum pl_tex_address_mode {
    PL_TEX_ADDRESS_CLAMP,  // clamp the nearest edge texel
    PL_TEX_ADDRESS_REPEAT, // repeat (tile) the texture
    PL_TEX_ADDRESS_MIRROR, // repeat (mirror) the texture
};

// Structure describing a texture.
struct pl_tex_params {
    int w, h, d;            // physical dimension; unused dimensions must be 0
    const struct pl_fmt *format;

    // The following bools describe what operations can be performed. The
    // corresponding pl_fmt capability must be set for every enabled
    // operation type.
    bool sampleable;    // usable as a PL_DESC_SAMPLED_TEX
    bool renderable;    // usable as a render target (pl_pass_run)
                        // (must only be used with 2D textures)
    bool storable;      // usable as a storage image (PL_DESC_IMG_*)
    bool blit_src;      // usable as a blit source
    bool blit_dst;      // usable as a blit destination
    bool host_writable; // may be updated with pl_tex_upload()
    bool host_readable; // may be fetched with pl_tex_download()

    // The following capabilities are only relevant for textures which have
    // either sampleable or blit_src enabled.
    enum pl_tex_sample_mode sample_mode;
    enum pl_tex_address_mode address_mode;

    // If non-NULL, the texture will be created with these contents. Using
    // this does *not* require setting host_writable. Otherwise, the initial
    // data is undefined.
    const void *initial_data;
};

static inline int pl_tex_params_dimension(const struct pl_tex_params params)
{
    return params.d ? 3 : params.h ? 2 : 1;
}

// Conflates the following typical GPU API concepts:
// - texture itself
// - sampler state
// - staging buffers for texture upload
// - framebuffer objects
// - wrappers for swapchain framebuffers
// - synchronization needed for upload/rendering/etc.
//
// Essentially a pl_tex can be anything ranging from a normal texture, a wrapped
// external/real framebuffer, a framebuffer object + texture pair, a mapped
// texture (via pl_hwdec), or other sorts of things that can be sampled from
// and/or rendered to.
struct pl_tex {
    struct pl_tex_params params;
    void *priv;
};

// Create a texture (with undefined contents). Returns NULL on failure. This is
// assumed to be an expensive/rare operation, and may need to perform memory
// allocation or framebuffer creation.
const struct pl_tex *pl_tex_create(const struct pl_gpu *gpu,
                                   const struct pl_tex_params *params);

void pl_tex_destroy(const struct pl_gpu *gpu, const struct pl_tex **tex);

// This works like `pl_tex_create`, but if the texture already exists and has
// incompatible texture parameters, it will get destroyed first. A texture is
// considered "compatible" if it has the same texture format and sample/address
// mode and it supports a superset of the features the user requested.
//
// Even if the texture is not recreated, calling this function will still
// invalidate the contents of the texture. (Note: Because of this,
// `initial_data` may not be used with `pl_tex_recreate`. Doing so is an error)
bool pl_tex_recreate(const struct pl_gpu *gpu, const struct pl_tex **tex,
                     const struct pl_tex_params *params);

// Invalidates the contents of a texture. After this, the contents are fully
// undefined.
void pl_tex_invalidate(const struct pl_gpu *gpu, const struct pl_tex *tex);

// Clear the dst texture with the given color (rgba). This is functionally
// identical to a blit operation, which means dst->params.blit_dst must be
// set.
void pl_tex_clear(const struct pl_gpu *gpu, const struct pl_tex *dst,
                  const float color[4]);

// Copy a sub-rectangle from one texture to another. The source/dest regions
// must be within the texture bounds. Areas outside the dest region are
// preserved. The formats of the textures must be loosely compatible - which
// essentially means that they must have the same texel size. Additionally,
// UINT textures can only be blitted to other UINT textures, and SINT textures
// can only be blitted to other SINT textures. Finally, src.blit_src and
// dst.blit_dst must be set, respectively.
//
// The rectangles may be "flipped", which leads to the image being flipped
// while blitting. If the src and dst rects have different sizes, the source
// image will be scaled according to src->params.sample_mode. That said, the
// src and dst rects must be fully contained within the src/dst dimensions.
void pl_tex_blit(const struct pl_gpu *gpu,
                 const struct pl_tex *dst, const struct pl_tex *src,
                 struct pl_rect3d dst_rc, struct pl_rect3d src_rc);

// Structure describing a texture transfer operation.
struct pl_tex_transfer_params {
    // Texture to transfer to/from. Depending on the type of the operation,
    // this must have params.host_writable (uploads) or params.host_readable
    // (downloads) set, respectively.
    const struct pl_tex *tex;

    // Note: Superfluous parameters are ignored, i.e. for a 1D texture, the y
    // and z fields of `rc`, as well as the corresponding strides, are ignored.
    // In all other cases, the stride must be >= the corresponding dimension of
    // `rc`, and the `rc` must be normalized and fully contained within the
    // image dimensions. Missing fields in the `rc` are inferred from the image
    // size. If unset, the strides are inferred from `rc` (that is, it's
    // assumed that the data is tightly packed in the buffer).
    struct pl_rect3d rc;   // region of the texture to transfer
    unsigned int stride_w; // the number of texels per horizontal row (x axis)
    unsigned int stride_h; // the number of texels per vertical column (y axis)

    // For the data source/target of a transfer operation, there are two valid
    // options:
    //
    // 1. Transferring to/from a buffer:
    const struct pl_buf *buf; // buffer to use (type must be PL_BUF_TEX_TRANSFER)
    size_t buf_offset;        // offset of data within buffer, must be a multiple
                              // of 4 as well as `tex->params.format->texel_size`
    // 2. Transferring to/from host memory directly:
    void *ptr;                // address of data
    // The contents of the memory region / buffer must exactly match the
    // texture format; i.e. there is no explicit conversion between formats.

    // For data uploads, which are typically "fire and forget" operations,
    // which method used does not matter much; although uploading from a host
    // mapped buffer requires fewer memory copy operations and is therefore
    // advised when uploading large amounts of data frequently.

    // For data downloads, downloading directly to host memory is a blocking
    // operation and should therefore be avoided as much as possible. It's
    // highyly recommended to always use a texture transfer buffer for texture
    // downloads if possible, which allows the transfer to happen
    // asynchronously.

    // When performing a texture transfer using a buffer, the buffer may be
    // marked as "in use" and should not used for a different type of operation
    // until pl_buf_poll returns false.
};

// Upload data to a texture. Returns whether successful.
bool pl_tex_upload(const struct pl_gpu *gpu,
                   const struct pl_tex_transfer_params *params);

// Download data from a texture. Returns whether successful.
bool pl_tex_download(const struct pl_gpu *gpu,
                     const struct pl_tex_transfer_params *params);

// Buffer usage type. This restricts what types of operations may be performed
// on a buffer.
enum pl_buf_type {
    PL_BUF_INVALID = 0,
    PL_BUF_TEX_TRANSFER,  // texture transfer buffer (for pl_tex_upload/download)
    PL_BUF_UNIFORM,       // UBO, for PL_DESC_BUF_UNIFORM
    PL_BUF_STORAGE,       // SSBO, for PL_DESC_BUF_STORAGE
    PL_BUF_TEXEL_UNIFORM, // texel buffer, for PL_DESC_BUF_TEXEL_UNIFORM
    PL_BUF_TEXEL_STORAGE, // texel buffer, for PL_DESC_BUF_TEXEL_STORAGE
    PL_BUF_PRIVATE,       // GPU-private usage (interpretation arbitrary)
    PL_BUF_TYPE_COUNT,
};

enum pl_buf_mem_type {
    PL_BUF_MEM_AUTO = 0, // use whatever seems most appropriate
    PL_BUF_MEM_HOST,     // try allocating from host memory (RAM)
    PL_BUF_MEM_DEVICE,   // try allocating from device memory (VRAM)
};

// Structure describing a buffer.
struct pl_buf_params {
    enum pl_buf_type type;
    size_t size;        // size in bytes
    bool host_mapped;   // create a persistent, RW mapping (pl_buf.data)
    bool host_writable; // contents may be updated via pl_buf_write()
    bool host_readable; // contents may be read back via pl_buf_read()

    // Provide a hint for the memory type you want to use when allocating
    // this buffer's memory. Currently, this field is ignored for all buffer
    // types except `PL_BUF_TEX_TRANSFER`, since uniform/storage buffers only
    // make sense when allocated from device memory.
    enum pl_buf_mem_type memory_type;

    // For texel buffers (PL_BUF_TEXEL_*), this gives the interpretation of the
    // buffer's contents. `format->caps` must include the corresponding
    // PL_FMT_CAP_TEXEL_* for the texel buffer type in use.
    const struct pl_fmt *format;

    // Setting this indicates that a buffer should be shared with external
    // APIs, and is treated as a bit mask of all handle types you want to
    // receive. This *must* be a subset of `pl_gpu.handle_caps`.
    pl_handle_types ext_handles;

    // If non-NULL, the buffer will be created with these contents. Otherwise,
    // the initial data is undefined. Using this does *not* require setting
    // host_writable.
    const void *initial_data;
};

// A generic buffer, which can be used for multiple purposes (texture transfer,
// storage buffer, uniform buffer, etc.)
//
// Note on efficiency: A pl_buf does not necessarily represent a true "buffer"
// object on the underlying graphics API. It may also refer to a sub-slice of
// a larger buffer, depending on the implementation details of the GPU. The
// bottom line is that users do not need to worry about the efficiency of using
// many small pl_buf objects. Having many small pl_bufs, even lots of few-byte
// vertex buffers, is designed to be completely fine.
struct pl_buf {
    struct pl_buf_params params;
    uint8_t *data; // for persistently mapped buffers, points to the first byte
    void *priv;

    // When using external memory handles, this contains the handles, plus the
    // offset of this paricular `pl_buf` within the handle. These handles are
    // owned by the `pl_gpu` - if a user wishes to use them in a way that takes
    // over ownership (e.g. importing into some APIs), they must clone the
    // handle before doing so (e.g. using `dup` for fds).
    //
    // If the `pl_buf` is destroyed (pl_buf_destroy), the contents of the
    // memory associated with these handles become undefined - including the
    // contents of any external API objects imported from them.
    struct pl_gpu_handle handles;
    size_t handle_offset;
};

// Create a buffer. The type of buffer depends on the parameters. The buffer
// parameters must adhere to the restrictions imposed by the pl_gpu_limits.
// Returns NULL on failure.
//
// For buffers with external handles, the buffer is considered to be in an
// "exported" state by default, and may be used directly by the external API
// after being created (until the first libplacebo operation on the buffer).
const struct pl_buf *pl_buf_create(const struct pl_gpu *gpu,
                                   const struct pl_buf_params *params);

// This behaves like `pl_buf_create`, but if the buffer already exists and has
// incompatible parameters, it will get destroyed first. A buffer is considered
// "compatible" if it has the same buffer type and texel format, a size greater
// than or equal to the requested size, and it has a superset of the features
// the user requested. After this operation, the contents of the buffer are
// undefined.
//
// Note: Due to its unpredictability, it's not allowed to use this with
// `params->initial_data` being set. Similarly, it's not allowed on a buffer
// with `params->ext_handles`. since this may invalidate the corresponding
// external API's handle. Conversely, it *is* allowed on a buffer with
// `params->host_mapped`, and the corresponding `buf->data` pointer *may*
// change as a result of doing so.
bool pl_buf_recreate(const struct pl_gpu *gpu, const struct pl_buf **buf,
                     const struct pl_buf_params *params);

void pl_buf_destroy(const struct pl_gpu *gpu, const struct pl_buf **buf);

// Update the contents of a buffer, starting at a given offset (must be a
// multiple of 4) and up to a given size, with the contents of *data.
void pl_buf_write(const struct pl_gpu *gpu, const struct pl_buf *buf,
                  size_t buf_offset, const void *data, size_t size);

// Read back the contents of a buffer, starting at a given offset (must be a
// multiple of 4) and up to a given size, storing the data into *dest.
// Returns whether successful.
bool pl_buf_read(const struct pl_gpu *gpu, const struct pl_buf *buf,
                 size_t buf_offset, void *dest, size_t size);

// Initiates a buffer export operation, allowing a buffer to be accessed by an
// external API. This is only valid for buffers with `params->ext_handles`.
// Calling this twice in a row is a harmless no-op. Returns whether successful.
//
// There is no corresponding "buffer import" operation, the next libplacebo
// operation that touches the buffer (e.g. pl_tex_upload, but also pl_buf_write
// and pl_buf_read) will implicitly import the buffer back to libplacebo. Users
// must ensure that all pending operations made by the external API are fully
// completed before using it in libplacebo again.
//
// Please note that this function returning does not mean the memory is
// immediately available as such. In general, it will mark a buffer as "in use"
// in the same way a read or write would, and it is the user's responsibility
// to wait until `pl_buf_poll` returns false before accessing the memory from
// the external API.
//
// In terms of the access performed by this operation, it is not considered a
// "read" or "write" and therefore does not technically conflict with reads or
// writes to the buffer performed by the host (via mapped memory - any use of
// `pl_buf_read` or `pl_buf_write` would defeat the purpose of the export).
// However, restrictions made by the external API may apply that prevent this.
//
// The recommended use pattern is something like this:
//
// while (loop) {
//    const struct pl_buf *buf = get_free_buffer(); // or block on pl_buf_poll
//    // write to the buffer using the external API
//    pl_tex_upload(gpu, /* ... buf ... */); // implicitly imports
//    pl_buf_export(gpu, buf);
// }
//
// i.e. perform an external API operation, then use and immediately export the
// buffer in libplacebo, and finally wait until `pl_buf_poll` is false before
// re-using it. (Or get a new, fresh buffer in the meantime)
bool pl_buf_export(const struct pl_gpu *gpu, const struct pl_buf *buf);

// Returns whether or not a buffer is currently "in use". This can either be
// because of a pending read operation, a pending write operation or a pending
// buffer import/export operation. Any access to the buffer by the user is
// forbidden while a buffer is "in use". This includes using `pl_buf_read` or
// `pl_buf_write` or accessing mapped memory directly. The only exception to
// this rule is multiple reads, for example reading from a buffer with
// `pl_tex_upload` while simultaneously reading from it using mapped memory.
//
// The `timeout`, specified in nanoseconds, indicates how long to block for
// before returning. If set to 0, this function will never block, and only
// returns the current status of the buffer. The actual precision of the
// timeout may be significantly longer than one nanosecond, and has no upper
// bound. This function does not provide hard latency guarantees.
//
// Note: Performing multiple libplacebo operations at the same time is always
// valid, for example it's perfectly valid to submit a `pl_tex_upload`
// immediately followed by a `pl_tex_download` to the same buffer. However,
// when this is the case, it's undefined as to when exactly the upload is
// happening versus when exactly the download is happening. So in this example,
// any access to the buffer by the host would be forbidden, including reads,
// until the user has verified that `pl_buf_poll` returned false. It's also
// always valid to call `pl_buf_destroy`, even on in-use buffers.
bool pl_buf_poll(const struct pl_gpu *gpu, const struct pl_buf *buf,
                 uint64_t timeout);

// Data type of a shader input variable (e.g. uniform, or UBO member)
enum pl_var_type {
    PL_VAR_INVALID = 0,
    PL_VAR_SINT,        // C: int           GLSL: int/ivec
    PL_VAR_UINT,        // C: unsigned int  GLSL: uint/uvec
    PL_VAR_FLOAT,       // C: float         GLSL: float/vec/mat
    PL_VAR_TYPE_COUNT
};

// Returns the host size (in bytes) of a pl_var_type.
size_t pl_var_type_size(enum pl_var_type type);

// Represents a shader input variable (concrete data, e.g. vector, matrix)
struct pl_var {
    const char *name;       // name as used in the shader
    enum pl_var_type type;
    // The total number of values is given by dim_v * dim_m. For example, a
    // vec2 would have dim_v = 2 and dim_m = 1. A mat3x4 would have dim_v = 4
    // and dim_m = 3.
    int dim_v;              // vector dimension
    int dim_m;              // matrix dimension (number of columns, see below)
    int dim_a;              // array dimension
};

// Returns a GLSL type name (e.g. vec4) for a given pl_var, or NULL if the
// variable is not legal. Not that the array dimension is ignored, since the
// array dimension is usually part of the variable name and not the type name.
const char *pl_var_glsl_type_name(struct pl_var var);

// Helper functions for constructing the most common pl_vars.
struct pl_var pl_var_uint(const char *name);
struct pl_var pl_var_float(const char *name);
struct pl_var pl_var_vec2(const char *name);
struct pl_var pl_var_vec3(const char *name);
struct pl_var pl_var_vec4(const char *name);
struct pl_var pl_var_mat2(const char *name);
struct pl_var pl_var_mat3(const char *name);
struct pl_var pl_var_mat4(const char *name);

// Converts a pl_fmt to an "equivalent" pl_var. Equivalent in this sense means
// that the pl_var's type will be the same as the vertex's sampled type (e.g.
// PL_FMT_UNORM gets turned into PL_VAR_FLOAT).
struct pl_var pl_var_from_fmt(const struct pl_fmt *fmt, const char *name);

// Describes the memory layout of a variable, relative to some starting location
// (typically the offset within a uniform/storage/pushconstant buffer)
//
// Note on matrices: All GPUs expect column major matrices, for both buffers and
// input variables. Care needs to be taken to avoid trying to use e.g. a
// pl_matrix3x3 (which is row major) directly as a pl_var_update.data!
//
// In terms of the host layout, a column-major matrix (e.g. matCxR) with C
// columns and R rows is treated like an array vecR[C]. The `stride` here refers
// to the separation between these array elements, i.e. the separation between
// the individual columns.
//
// Visualization of a mat4x3:
//
//       0   1   2   3  <- columns
// 0  [ (A) (D) (G) (J) ]
// 1  [ (B) (E) (H) (K) ]
// 2  [ (C) (F) (I) (L) ]
// ^ rows
//
// Layout in GPU memory: (stride=16, size=60)
//
// [ A B C ] X <- column 0, offset +0
// [ D E F ] X <- column 1, offset +16
// [ G H I ] X <- column 2, offset +32
// [ J K L ]   <- column 3, offset +48
//
// Note the lack of padding on the last column in this example.
// In general: size <= stride * dim_m
//
// C representation: (stride=12, size=48)
//
// { { A, B, C },
//   { D, E, F },
//   { G, H, I },
//   { J, K, L } }
//
// Note on arrays: `stride` represents both the stride between elements of a
// matrix, and the stride between elements of an array. That is, there is no
// distinction between the columns of a matrix and the rows of an array. For
// example, a mat2[10] and a vec2[20] share the same pl_var_layout - the stride
// would be sizeof(vec2) and the size would be sizeof(vec2) * 2 * 10.
//
// For non-array/matrix types, `stride` is equal to `size`.

struct pl_var_layout {
    size_t offset; // the starting offset of the first byte
    size_t stride; // the delta between two elements of an array/matrix
    size_t size;   // the total size of the input
};

// Returns the host layout of an input variable as required for a
// tightly-packed, byte-aligned C data type, given a starting offset.
struct pl_var_layout pl_var_host_layout(size_t offset, const struct pl_var *var);

// Returns the GLSL std140 layout of an input variable given a current buffer
// offset, as required for a buffer of type PL_BUF_UNIFORM.
//
// The normal way to use this function is when calculating the size and offset
// requirements of a uniform buffer in an incremental fashion, to calculate the
// new offset of the next variable in this buffer.
struct pl_var_layout pl_std140_layout(size_t offset, const struct pl_var *var);

// Returns the GLSL std430 layout of an input variable given a current buffer
// offset, as required for a buffer of type PL_BUF_STORAGE, and for push
// constants.
struct pl_var_layout pl_std430_layout(size_t offset, const struct pl_var *var);

// Like memcpy, but copies bytes from `src` to `dst` in a manner governed by
// the stride and size of `dst_layout` as well as `src_layout`. Also takes
// into account the respective `offset`.
void memcpy_layout(void *dst, struct pl_var_layout dst_layout,
                   const void *src, struct pl_var_layout src_layout);

// Represents a vertex attribute.
struct pl_vertex_attrib {
    const char *name;         // name as used in the shader
    const struct pl_fmt *fmt; // data format (must have PL_FMT_CAP_VERTEX)
    size_t offset;            // byte offset into the vertex struct
    int location;             // vertex location (as used in the shader)
};

// Type of a shader input descriptor.
enum pl_desc_type {
    PL_DESC_INVALID = 0,
    PL_DESC_SAMPLED_TEX,    // C: pl_tex*    GLSL: combined texture sampler
                            // (pl_tex->params.sampleable must be set)
    PL_DESC_STORAGE_IMG,    // C: pl_tex*    GLSL: storage image
                            // (pl_tex->params.storable must be set)
    PL_DESC_BUF_UNIFORM,    // C: pl_buf*    GLSL: uniform buffer
                            // (pl_buf->params.type must be PL_BUF_UNIFORM)
    PL_DESC_BUF_STORAGE,    // C: pl_buf*    GLSL: storage buffer
                            // (pl_buf->params.type must be PL_BUF_STORAGE)
    PL_DESC_BUF_TEXEL_UNIFORM,// C: pl_buf*  GLSL: uniform samplerBuffer
                              // (pl_buf->params.type must be PL_BUF_TEXEL_UNIFORM)
    PL_DESC_BUF_TEXEL_STORAGE,// C: pl_buf*  GLSL: uniform imageBuffer
                              // (pl_buf->params.type must be PL_BUF_TEXEL_STORAGE)
    PL_DESC_TYPE_COUNT
};

// Returns an abstract namespace index for a given descriptor type. This will
// always be a value >= 0 and < PL_DESC_TYPE_COUNT. Implementations can use
// this to figure out which descriptors may share the same value of `binding`.
// Bindings must only be unique for all descriptors within the same namespace.
int pl_desc_namespace(const struct pl_gpu *gpu, enum pl_desc_type type);

// Access mode of a shader input descriptor.
enum pl_desc_access {
    PL_DESC_ACCESS_READWRITE,
    PL_DESC_ACCESS_READONLY,
    PL_DESC_ACCESS_WRITEONLY,
};

// Returns the GLSL syntax for a given access mode (e.g. "readonly").
const char *pl_desc_access_glsl_name(enum pl_desc_access mode);

struct pl_buffer_var {
    struct pl_var var;
    struct pl_var_layout layout;
};

// Represents a shader descriptor (e.g. texture or buffer binding)
struct pl_desc {
    const char *name;       // name as used in the shader
    enum pl_desc_type type;

    // The binding of this descriptor, as used in the shader. All bindings
    // within a namespace must be unique. (see: pl_desc_namespace)
    int binding;

    // For storage images and storage buffers, this can be used to restrict
    // the type of access that may be performed on the descriptor. Ignored for
    // the other descriptor types (uniform buffers and sampled textures are
    // always read-only).
    enum pl_desc_access access;

    // For PL_DESC_BUF_UNIFORM/STORAGE, this specifies the layout of the
    // variables contained by a buffer. Ignored for the other descriptor types
    struct pl_buffer_var *buffer_vars;
    int num_buffer_vars;
};

// Framebuffer blending mode (for raster passes)
enum pl_blend_mode {
    PL_BLEND_ZERO,
    PL_BLEND_ONE,
    PL_BLEND_SRC_ALPHA,
    PL_BLEND_ONE_MINUS_SRC_ALPHA,
};

struct pl_blend_params {
    enum pl_blend_mode src_rgb;
    enum pl_blend_mode dst_rgb;
    enum pl_blend_mode src_alpha;
    enum pl_blend_mode dst_alpha;
};

enum pl_prim_type {
    PL_PRIM_TRIANGLE_LIST,
    PL_PRIM_TRIANGLE_STRIP,
    PL_PRIM_TRIANGLE_FAN,
};

enum pl_pass_type {
    PL_PASS_INVALID = 0,
    PL_PASS_RASTER,  // vertex+fragment shader
    PL_PASS_COMPUTE, // compute shader (requires PL_GPU_CAP_COMPUTE)
    PL_PASS_TYPE_COUNT,
};

// Description of a rendering pass. It conflates the following:
//  - GLSL shader(s) and its list of inputs
//  - target parameters (for raster passes)
struct pl_pass_params {
    enum pl_pass_type type;

    // Input variables. Only supported if PL_GPU_CAP_INPUT_VARIABLES is set.
    // Otherwise, num_variables must be 0.
    struct pl_var *variables;
    int num_variables;

    // Input descriptors. (Always supported)
    struct pl_desc *descriptors;
    int num_descriptors;

    // Push constant region. Must be be a multiple of 4 <= limits.max_pushc_size
    size_t push_constants_size;

    // The shader text in GLSL. For PL_PASS_RASTER, this is interpreted
    // as a fragment shader. For PL_PASS_COMPUTE, this is interpreted as
    // a compute shader.
    const char *glsl_shader;

    // Highly implementation-specific byte array storing a compiled version of
    // the same shader. Can be used to speed up pass creation on already
    // known/cached shaders.
    //
    // Note: There are no restrictions on this. Passing an out-of-date cache,
    // passing a cache corresponding to a different progam, or passing a cache
    // belonging to a different GPU, are all valid. But obviously, in such cases,
    // there is no benefit in doing so.
    const uint8_t *cached_program;
    size_t cached_program_len;

    // --- type==PL_PASS_RASTER only

    // Describes the interpretation and layout of the vertex data.
    enum pl_prim_type vertex_type;
    struct pl_vertex_attrib *vertex_attribs;
    int num_vertex_attribs;
    size_t vertex_stride;

    // The vertex shader itself.
    const char *vertex_shader;

    // The target dummy texture this renderpass is intended to be used with.
    // This doesn't have to be a real texture - the caller can also pass a
    // blank pl_tex object, as long as target_dummy.params.format is set. The
    // format must support PL_FMT_CAP_RENDERABLE, and if the target dummy is
    // an actual texture, it must have `renderable` enabled.
    //
    // If you pass a real texture here, the GPU backend may be able to optimize
    // the render pass better for the specific requirements of this texture.
    // This does not change the semantics of pl_pass_run, just perhaps the
    // performance. (The `priv` pointer will be cleared by pl_pass_create, so
    // there is no risk of a dangling reference)
    struct pl_tex target_dummy;

    // Target blending mode. If this is NULL, blending is disabled. Otherwise,
    // the `target_dummy.params.format` must have PL_FMT_CAP_BLENDABLE.
    const struct pl_blend_params *blend_params;

    // If false, the target's existing contents will be discarded before the
    // pass is run. (Semantically equivalent to calling pl_tex_invalidate
    // before every pl_pass_run, but slightly more efficient)
    bool load_target;
};

// Conflates the following typical GPU API concepts:
// - various kinds of shaders
// - rendering pipelines
// - descriptor sets, uniforms, other bindings
// - all synchronization necessary
// - the current values of all inputs
struct pl_pass {
    struct pl_pass_params params;
    void *priv;
};

// Compile a shader and create a render pass. This is a rare/expensive
// operation and may take a significant amount of time, even if a cached
// program is used. Returns NULL on failure.
//
// The resulting pl_pass->params.cached_program will be initialized by
// this function to point to a new, valid cached program (if any).
const struct pl_pass *pl_pass_create(const struct pl_gpu *gpu,
                                     const struct pl_pass_params *params);

void pl_pass_destroy(const struct pl_gpu *gpu, const struct pl_pass **pass);

struct pl_desc_binding {
    const void *object; // pl_* object with type corresponding to pl_desc_type
};

struct pl_var_update {
    int index;        // index into params.variables[]
    const void *data; // pointer to raw byte data corresponding to pl_var_host_layout()
};

struct pl_pass_run_params {
    const struct pl_pass *pass;

    // This list only contains descriptors/variables which have changed
    // since the previous invocation. All non-mentioned variables implicitly
    // preserve their state from the last invocation.
    struct pl_var_update *var_updates;
    int num_var_updates;

    // This list contains all descriptors used by this pass. It must
    // always be filled, even if the descriptors haven't changed. The order
    // must match that of pass->params.descriptors
    struct pl_desc_binding *desc_bindings;

    // The push constants for this invocation. This must always be set and
    // fully defined for every invocation if params.push_constants_size > 0.
    void *push_constants;

    // --- pass->params.type==PL_PASS_RASTER only

    // Target must be a 2D texture, target->params.renderable must be true, and
    // target->params.format must match pass->params.target_dummy.params.format.
    // If the viewport or scissors are left blank, they are inferred from
    // target->params.
    //
    // WARNING: Rendering to a *target that is being read from by the same
    // shader is undefined behavior. In general, trying to bind the same
    // resource multiple times to the same shader is undefined behavior.
    const struct pl_tex *target;
    struct pl_rect2d viewport; // screen space viewport (must be normalized)
    struct pl_rect2d scissors; // target render scissors (must be normalized)

    void *vertex_data;  // raw pointer to vertex data
    int vertex_count;   // number of vertices to render

    // --- pass->params.type==PL_PASS_COMPUTE only

    // Number of work groups to dispatch per dimension (X/Y/Z). Must be <= the
    // corresponding index of limits.max_dispatch
    int compute_groups[3];
};

// Execute a render pass.
void pl_pass_run(const struct pl_gpu *gpu, const struct pl_pass_run_params *params);

// This is semantically a no-op, but it provides a hint that you want to flush
// any partially queued up commands and begin execution. There is normally no
// need to call this, because queued commands will always be implicitly flushed
// whenever necessary to make forward progress on commands like `pl_buf_poll`,
// or when submitting a frame to a swapchain for display. In fact, calling this
// function can negatively impact performance, because some GPUs rely on being
// able to re-order and modify queued commands in order to enable optimizations
// retroactively.
//
// The only time this might be beneficial to call explicitly is if you're doing
// lots of offline processing where you submit a bunch of work and then
// use asynchronous texture downloads (via pl_tex_download) to retrieve the
// results. In that case you should call this function after each work item
// to ensure good parallelism between them.
//
// It's worth noting that this function may block if you're over-feeding the
// GPU without waiting for existing results to finish.
void pl_gpu_flush(const struct pl_gpu *gpu);

// This is like `pl_gpu_flush` but also blocks until the GPU is fully idle
// before returning. Using this in your rendering loop is seriously disadvised,
// and almost never the right solution. The intended use case is for deinit
// logic, where users may want to force the all pending GPU operations to
// finish so they can clean up their state more easily.
//
// After this operation is called, it's guaranteed that all pending buffer
// operations are complete - i.e. `pl_buf_poll` is guaranteed to return false.
// Also, if you only care about buffer operations, you can accomplish this more
// easily by using `pl_buf_poll` with the timeout set to `UINT64_MAX`. But
// if you have many buffers it may be more convenient to call this function
// instead. The difference is that this function will also affect e.g.
// renders to a `pl_swapchain`.
void pl_gpu_finish(const struct pl_gpu *gpu);

#endif // LIBPLACEBO_GPU_H_
