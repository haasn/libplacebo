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

#ifndef LIBPLACEBO_RA_H_
#define LIBPLACEBO_RA_H_

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#include "common.h"

// This file containings the definition of an API referred to as "RA", which
// stands for Rendering Abstraction (or Rendering API) and is designed to
// abstract away from platform-specific APIs like the various OpenGL variants,
// Direct3D and Vulkan in a common way. It is a much more limited API than
// those APIs, since it tries targetting a very small common subset of features
// that is needed to implement libplacebo's rendering.
//
// NOTE: When speaking of "valid usage" or "must", invalid usage is assumed
// to result in undefined behavior. (if libplacebo is compiled without NDEBUG,
// this will be checked and libplacebo will terminate safely instead)

// Structure which wraps metadata describing GLSL capabilities.
struct ra_glsl_desc {
    int version;        // GLSL version (e.g. 450), for #version
    bool gles;          // GLSL ES semantics (ESSL)
    bool vulkan;        // GL_KHR_vulkan_glsl semantics
};

typedef uint64_t ra_caps;
enum {
    RA_CAP_COMPUTE          = 1 << 0, // supports compute shaders
    RA_CAP_PARALLEL_COMPUTE = 1 << 1, // supports multiple compute queues
    RA_CAP_INPUT_VARIABLES  = 1 << 2, // supports shader input variables
};

// Structure defining the physical limits of this RA instance. If a limit is
// given as 0, that means that feature is unsupported.
struct ra_limits {
    int max_tex_1d_dim;    // maximum width for a 1D texture
    int max_tex_2d_dim;    // maximum width/height for a 2D texture (required)
    int max_tex_3d_dim;    // maximum width/height/depth for a 3D texture
    size_t max_pushc_size; // maximum push_constants_size
    size_t max_xfer_size;  // maximum size of a RA_BUF_TEX_TRANSFER
    size_t max_ubo_size;   // maximum size of a RA_BUF_UNIFORM
    size_t max_ssbo_size;  // maximum size of a RA_BUF_STORAGE
    int min_gather_offset; // minimum textureGatherOffset offset
    int max_gather_offset; // maximum textureGatherOffset offset

    // Compute shader limits. Always available (non-zero) if RA_CAP_COMPUTE set
    size_t max_shmem_size; // maximum compute shader shared memory size
    int max_group_threads; // maximum number of local threads per work group
    int max_group_size[3]; // maximum work group size per dimension
    int max_dispatch[3];   // maximum dispatch size per dimension

    // These don't represent hard limits but indicate performance hints for
    // optimal alignment. For best performance, the corresponding field
    // should be aligned to a multiple of these. They will always be a power
    // of two.
    int align_tex_xfer_stride;    // optimal ra_tex_transfer_params.stride_w/h
    size_t align_tex_xfer_offset; // optimal ra_tex_transfer_params.buf_offset
};

// Abstract device context which wraps an underlying graphics context and can
// be used to dispatch rendering commands.
struct ra {
    struct pl_context *ctx;     // the pl_context this RA was initialized from
    const struct ra_fns *impl;  // the underlying implementation (unique per RA)
    void *priv;

    ra_caps caps;               // RA_CAP_* bit field
    struct ra_glsl_desc glsl;   // GLSL version supported by this RA
    struct ra_limits limits;    // physical device limits
    // Note: Every RA must support at least one of RA_CAP_INPUT_VARIABLES or
    // uniform buffers (limits.max_ubo_size > 0).

    // Supported texture formats, in preference order. (If there are multiple
    // similar formats, the "better" ones come first)
    const struct ra_fmt **formats;
    int num_formats;
};

// Helper function to align the given dimension (e.g. width or height) to a
// multiple of the optimal texture transfer stride.
int ra_optimal_transfer_stride(const struct ra *ra, int dimension);

enum ra_fmt_type {
    RA_FMT_UNKNOWN = 0, // also used for inconsistent multi-component formats
    RA_FMT_UNORM,       // unsigned, normalized integer format (sampled as float)
    RA_FMT_SNORM,       // signed, normalized integer format (sampled as float)
    RA_FMT_UINT,        // unsigned integer format (sampled as integer)
    RA_FMT_SINT,        // signed integer format (sampled as integer)
    RA_FMT_FLOAT,       // (signed) float formats, any bit size
    RA_FMT_TYPE_COUNT,
};

enum ra_fmt_caps {
    RA_FMT_CAP_SAMPLEABLE   = 1 << 0, // may be sampled from (RA_DESC_SAMPLED_TEX)
    RA_FMT_CAP_STORABLE     = 1 << 1, // may be used as storage image (RA_DESC_STORAGE_IMG)
    RA_FMT_CAP_LINEAR       = 1 << 2, // may be linearly samplied from (RA_TEX_SAMPLE_LINEAR)
    RA_FMT_CAP_RENDERABLE   = 1 << 3, // may be rendered to (ra_pass_params.target_fmt)
    RA_FMT_CAP_BLENDABLE    = 1 << 4, // may be blended to (ra_pass_params.enable_blend)
    RA_FMT_CAP_BLITTABLE    = 1 << 5, // may be blitted from/to (ra_tex_blit)
    RA_FMT_CAP_VERTEX       = 1 << 6, // may be used as a vertex attribute

    // Notes:
    // - RA_FMT_CAP_LINEAR also implies RA_FMT_CAP_SAMPLEABLE
    // - RA_FMT_CAP_STORABLE also implies RA_CAP_COMPUTE
    // - RA_FMT_CAP_VERTEX implies that the format is non-opaque
};

// Structure describing a texel/vertex format.
struct ra_fmt {
    const char *name;       // symbolic name for this format (e.g. rgba32f)
    const void *priv;

    enum ra_fmt_type type;  // the format's data type and interpretation
    enum ra_fmt_caps caps;  // the features supported by this format
    int num_components;     // number of components for this format
    int component_depth[4]; // meaningful bits per component, texture precision

    // This controls the relationship between the data as seen by the host and
    // the way it's interpreted by the texture. If `opaque` is true, then
    // there's no meaningful correspondence between the two, and all of the
    // remaining fields in this section are unset. The host representation is
    // always tightly packed (no padding bits in between each component).
    bool opaque;
    size_t texel_size;      // total size in bytes per texel
    int host_bits[4];       // number of meaningful bits in host memory
    int sample_order[4];    // sampled index for each component, e.g.
                            // {2, 1, 0, 3} for BGRA textures

    // If usable as a vertex or texel buffer format, this gives the GLSL type
    // corresponding to the data. (e.g. vec4)
    const char *glsl_type;

    // If usable as a storage image (RA_FMT_CAP_STORABLE), this gives the
    // GLSL image format corresponding to the format. (e.g. rgba16ui)
    const char *glsl_format;
};

// Returns whether or not a ra_fmt's components are ordered sequentially
// in memory in the order RGBA.
bool ra_fmt_is_ordered(const struct ra_fmt *fmt);

// Helper function to find a format with a given number of components and
// minimum effective precision per component. If `host_bits` is set, then the
// format will always be non-opaque, unpadded, ordered and have exactly this
// bit depth for each component. Finally, all `caps` must be supported.
const struct ra_fmt *ra_find_fmt(const struct ra *ra, enum ra_fmt_type type,
                                 int num_components, int min_depth,
                                 int host_bits, enum ra_fmt_caps caps);

// Finds a vertex format for a given configuration. The resulting vertex will
// have a component depth equivalent to to the sizeof() the equivalent host type.
// (e.g. RA_FMT_FLOAT will always have sizeof(float))
const struct ra_fmt *ra_find_vertex_fmt(const struct ra *ra,
                                        enum ra_fmt_type type,
                                        int num_components);

// Find a format based on its name.
const struct ra_fmt *ra_find_named_fmt(const struct ra *ra, const char *name);

enum ra_tex_sample_mode {
    RA_TEX_SAMPLE_NEAREST,  // nearest neighour sampling
    RA_TEX_SAMPLE_LINEAR,   // linear filtering
};

enum ra_tex_address_mode {
    RA_TEX_ADDRESS_CLAMP,  // clamp the nearest edge texel
    RA_TEX_ADDRESS_REPEAT, // repeat (tile) the texture
    RA_TEX_ADDRESS_MIRROR, // repeat (mirror) the texture
};

// Structure describing a texture.
struct ra_tex_params {
    int w, h, d;            // physical dimension; unused dimensions must be 0
    const struct ra_fmt *format;

    // The following bools describe what operations can be performed. The
    // corresponding ra_fmt capability must be set for every enabled
    // operation type.
    bool sampleable;    // usable as a RA_DESC_SAMPLED_TEX
    bool renderable;    // usable as a render target (ra_pass_run)
                        // (must only be used with 2D textures)
    bool storable;      // usable as a storage image (RA_DESC_IMG_*)
    bool blit_src;      // usable as a blit source
    bool blit_dst;      // usable as a blit destination
    bool host_writable; // may be updated with ra_tex_upload()
    bool host_readable; // may be fetched with ra_tex_download()

    // The following capabilities are only relevant for textures which have
    // either sampleable or blit_src enabled.
    enum ra_tex_sample_mode sample_mode;
    enum ra_tex_address_mode address_mode;

    // If non-NULL, the texture will be created with these contents. Using
    // this does *not* require setting host_writable. Otherwise, the initial
    // data is undefined.
    const void *initial_data;
};

static inline int ra_tex_params_dimension(const struct ra_tex_params params)
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
// Essentially a ra_tex can be anything ranging from a normal texture, a wrapped
// external/real framebuffer, a framebuffer object + texture pair, a mapped
// texture (via ra_hwdec), or other sorts of things that can be sampled from
// and/or rendered to.
struct ra_tex {
    struct ra_tex_params params;
    void *priv;
};

// Create a texture (with undefined contents). Returns NULL on failure. This is
// assumed to be an expensive/rare operation, and may need to perform memory
// allocation or framebuffer creation.
const struct ra_tex *ra_tex_create(const struct ra *ra,
                                   const struct ra_tex_params *params);

void ra_tex_destroy(const struct ra *ra, const struct ra_tex **tex);

// Invalidates the contents of a texture. After this, the contents are fully
// undefined.
void ra_tex_invalidate(const struct ra *ra, const struct ra_tex *tex);

// Clear the dst texture with the given color (rgba). This is functionally
// identical to a blit operation, which means dst->params.blit_dst must be
// set.
void ra_tex_clear(const struct ra *ra, const struct ra_tex *dst,
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
void ra_tex_blit(const struct ra *ra,
                 const struct ra_tex *dst, const struct ra_tex *src,
                 struct pl_rect3d dst_rc, struct pl_rect3d src_rc);

// Structure describing a texture transfer operation.
struct ra_tex_transfer_params {
    // Texture to transfer to/from. Depending on the type of the operation,
    // this must have params.host_writable (uploads) or params.host_readable
    // (downloads) set, respectively.
    const struct ra_tex *tex;

    // Note: Superfluous parameters are ignored, i.e. for a 1D texture, the y
    // and z fields of `rc`, as well as the corresponding strides, are ignored.
    // In all other cases, the stride must be >= the corresponding dimension
    // of `rc`, and the `rc` must be normalized and fully contained within the
    // image dimensions. If any of these parameters are left away (0), they
    // are inferred from the texture's size.
    struct pl_rect3d rc;   // region of the texture to transfer
    unsigned int stride_w; // the number of texels per horizontal row (x axis)
    unsigned int stride_h; // the number of texels per vertical column (y axis)

    // For the data source/target of a transfer operation, there are two valid
    // options:
    //
    // 1. Transferring to/from a buffer:
    const struct ra_buf *buf; // buffer to use (type must be RA_BUF_TEX_TRANSFER)
    size_t buf_offset;        // offset of data within buffer, must be a multiple of 4
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
    // until ra_buf_poll returns false.
};

// Upload data to a texture. Returns whether successful.
bool ra_tex_upload(const struct ra *ra,
                   const struct ra_tex_transfer_params *params);

// Download data from a texture. Returns whether successful.
bool ra_tex_download(const struct ra *ra,
                     const struct ra_tex_transfer_params *params);

// Buffer usage type. This restricts what types of operations may be performed
// on a buffer.
enum ra_buf_type {
    RA_BUF_INVALID = 0,
    RA_BUF_TEX_TRANSFER, // texture transfer buffer (for ra_tex_upload/download)
    RA_BUF_UNIFORM,      // UBO, for RA_DESC_BUF_UNIFORM
    RA_BUF_STORAGE,      // SSBO, for RA_DESC_BUF_STORAGE
    RA_BUF_PRIVATE,      // RA-private usage (interpretation arbitrary)
    RA_BUF_TYPE_COUNT,
};

// Structure describing a buffer.
struct ra_buf_params {
    enum ra_buf_type type;
    size_t size;        // size in bytes
    bool host_mapped;   // create a persistent, RW mapping (ra_buf.data)
    bool host_writable; // contents may be updated via ra_buf_write()
    bool host_readable; // contents may be read back via ra_buf_read()

    // If non-NULL, the buffer will be created with these contents. Otherwise,
    // the initial data is undefined. Using this does *not* require setting
    // host_writable.
    const void *initial_data;
};

// A generic buffer, which can be used for multiple purposes (texture transfer,
// storage buffer, uniform buffer, etc.)
//
// Note on efficiency: A ra_buf does not necessarily represent a true "buffer"
// object on the underlying graphics API. It may also refer to a sub-slice of
// a larger buffer, depending on the implementation details of the RA. The
// bottom line is that users do not need to worry about the efficiency of using
// many small ra_buf objects. Having many small ra_bufs, even lots of few-byte
// vertex buffers, is designed to be completely fine.
struct ra_buf {
    struct ra_buf_params params;
    char *data; // for persistently mapped buffers, points to the first byte
    void *priv;
};

// Create a buffer. The type of buffer depends on the parameters. The buffer
// parameters must adhere to the restrictions imposed by the ra_limits. Returns
// NULL on failure.
const struct ra_buf *ra_buf_create(const struct ra *ra,
                                   const struct ra_buf_params *params);

void ra_buf_destroy(const struct ra *ra, const struct ra_buf **buf);

// Update the contents of a buffer, starting at a given offset (must be a
// multiple of 4) and up to a given size, with the contents of *data.
void ra_buf_write(const struct ra *ra, const struct ra_buf *buf,
                  size_t buf_offset, const void *data, size_t size);

// Read back the contents of a buffer, starting at a given offset (must be a
// multiple of 4) and up to a given size, storing the data into *dest.
// Returns whether successful.
bool ra_buf_read(const struct ra *ra, const struct ra_buf *buf,
                 size_t buf_offset, void *dest, size_t size);

// Returns whether or not a buffer is currently "in use". This can either be
// because of a pending read operation or because of a pending write operation.
// Coalescing multiple types of the same access (e.g. uploading the same buffer
// to multiple textures) is fine, but trying to read a buffer while it is being
// written to or trying to write to a buffer while it is being read from will
// almost surely result in graphical corruption. RA makes no attempt to enforce
// this, it is up to the user to check and adhere to whatever restrictions are
// necessary.
//
// The `timeout`, specified in nanoseconds, indicates how long to block for
// before returning. If set to 0, this function will never block, and only
// returns the current status of the buffer. The actual precision of the
// timeout may be significantly longer than one nanosecond, and has no upper
// bound. This function does not provide hard latency guarantees.
//
// Note: Destroying a buffer (ra_buf_destroy) is always valid, even if that
// buffer is in use.
bool ra_buf_poll(const struct ra *ra, const struct ra_buf *buf, uint64_t timeout);

// Data type of a shader input variable (e.g. uniform, or UBO member)
enum ra_var_type {
    RA_VAR_INVALID = 0,
    RA_VAR_SINT,        // C: int           GLSL: int/ivec
    RA_VAR_UINT,        // C: unsigned int  GLSL: uint/uvec
    RA_VAR_FLOAT,       // C: float         GLSL: float/vec/mat
    RA_VAR_TYPE_COUNT
};

// Returns the host size (in bytes) of a ra_var_type.
size_t ra_var_type_size(enum ra_var_type type);

// Represents a shader input variable (concrete data, e.g. vector, matrix)
struct ra_var {
    const char *name;       // name as used in the shader
    enum ra_var_type type;
    // The total number of values is given by dim_v * dim_m. For example, a
    // vec2 would have dim_v = 2 and dim_m = 1. A mat3x4 would have dim_v = 4
    // and dim_m = 3.
    int dim_v;              // vector dimension
    int dim_m;              // matrix dimension (number of columns, see below)
    int dim_a;              // array dimension
};

// Returns a GLSL type name (e.g. vec4) for a given ra_var, or NULL if the
// variable is not legal. Not that the array dimension is ignored, since the
// array dimension is usually part of the variable name and not the type name.
const char *ra_var_glsl_type_name(struct ra_var var);

// Helper functions for constructing the most common ra_vars.
struct ra_var ra_var_uint(const char *name);
struct ra_var ra_var_float(const char *name);
struct ra_var ra_var_vec2(const char *name);
struct ra_var ra_var_vec3(const char *name);
struct ra_var ra_var_vec4(const char *name);
struct ra_var ra_var_mat2(const char *name);
struct ra_var ra_var_mat3(const char *name);
struct ra_var ra_var_mat4(const char *name);

// Converts a ra_fmt to an "equivalent" ra_var. Equivalent in this sense means
// that the ra_var's type will be the same as the vertex's sampled type (e.g.
// RA_FMT_UNORM gets turned into RA_VAR_FLOAT).
struct ra_var ra_var_from_fmt(const struct ra_fmt *fmt, const char *name);

// Describes the memory layout of a variable, relative to some starting location
// (typically the offset within a uniform/storage/pushconstant buffer)
//
// Note on matrices: All RAs expect column major matrices, for both buffers and
// input variables. Care needs to be taken to avoid trying to use e.g. a
// pl_matrix3x3 (which is row major) directly as a ra_var_update.data!
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
// [ G H I ] X <- column 1, offset +32
// [ J K L ]   <- column 1, offset +48
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
// example, a mat2[10] and a vec2[20] share the same ra_var_layout - the stride
// would be sizeof(vec2) and the size would be sizeof(vec2) * 2 * 10.

struct ra_var_layout {
    size_t offset; // the starting offset of the first byte
    size_t stride; // the delta between two elements of an array/matrix
    size_t size;   // the total size of the input
};

// Returns the host layout of an input variable as required for a
// tightly-packed, byte-aligned C data type, given a starting offset.
struct ra_var_layout ra_var_host_layout(size_t offset, const struct ra_var *var);

// Returns the layout requirements of a uniform buffer element given a current
// buffer offset. If limits.max_ubo_size is 0, then this function returns {0}.
//
// Note: In terms of the GLSL, this is always *specified* as std140 layout, but
// because of the way GLSL gets translated to other APIs (notably D3D11), the
// actual buffer contents may vary considerably from std140. As such, the
// calling code should not make any assumptions about the buffer layout and
// instead query the layout requirements explicitly using this function.
//
// The normal way to use this function is when calculating the size and offset
// requirements of a uniform buffer in an incremental fashion, to calculate the
// new offset of the next variable in this buffer.
struct ra_var_layout ra_buf_uniform_layout(const struct ra *ra, size_t offset,
                                           const struct ra_var *var);

// Returns the layout requirements of a storage buffer element given a current
// buffer offset. If limits.max_ssbo_size is 0, then this function returns {0}.
//
// Note: In terms of the GLSL, this is always *specified* as std430 layout, but
// like with ra_buf_uniform_layout, the actual implementation may disagree.
struct ra_var_layout ra_buf_storage_layout(const struct ra *ra, size_t offset,
                                           const struct ra_var *var);

// Returns the layout requirements of a push constant element given a current
// push constant offset. If ra.max_pushc_size is 0, then this function returns
// {0}.
struct ra_var_layout ra_push_constant_layout(const struct ra *ra, size_t offset,
                                             const struct ra_var *var);

// Like memcpy, but copies bytes from `src` to `dst` in a manner governed by
// the stride and size of `dst_layout` as well as `src_layout`. Also takes
// into account the respective `offset`.
void memcpy_layout(void *dst, struct ra_var_layout dst_layout,
                   const void *src, struct ra_var_layout src_layout);

// Represents a vertex attribute.
struct ra_vertex_attrib {
    const char *name;         // name as used in the shader
    const struct ra_fmt *fmt; // data format (must have RA_FMT_CAP_VERTEX)
    size_t offset;            // byte offset into the vertex struct
    int location;             // vertex location (as used in the shader)
};

// Type of a shader input descriptor.
enum ra_desc_type {
    RA_DESC_INVALID = 0,
    RA_DESC_SAMPLED_TEX,    // C: ra_tex*    GLSL: combined texture sampler
                            // (ra_tex->params.sampleable must be set)
    RA_DESC_STORAGE_IMG,    // C: ra_tex*    GLSL: storage image
                            // (ra_tex->params.storable must be set)
    RA_DESC_BUF_UNIFORM,    // C: ra_buf*    GLSL: uniform buffer
                            // (ra_buf->params.type must be RA_BUF_UNIFORM)
    RA_DESC_BUF_STORAGE,    // C: ra_buf*    GLSL: storage buffer
                            // (ra_buf->params.type must be RA_BUF_STORAGE)
    RA_DESC_TYPE_COUNT
};

// Returns an abstract namespace index for a given descriptor type. This will
// always be a value >= 0 and < RA_DESC_TYPE_COUNT. Implementations can use
// this to figure out which descriptors may share the same value of `binding`.
// Bindings must only be unique for all descriptors within the same namespace.
int ra_desc_namespace(const struct ra *ra, enum ra_desc_type type);

// Access mode of a shader input descriptor.
enum ra_desc_access {
    RA_DESC_ACCESS_READWRITE,
    RA_DESC_ACCESS_READONLY,
    RA_DESC_ACCESS_WRITEONLY,
};

// Returns the GLSL syntax for a given access mode (e.g. "readonly").
const char *ra_desc_access_glsl_name(enum ra_desc_access mode);

struct ra_buffer_var {
    struct ra_var var;
    struct ra_var_layout layout;
};

// Represents a shader descriptor (e.g. texture or buffer binding)
struct ra_desc {
    const char *name;       // name as used in the shader
    enum ra_desc_type type;

    // The binding of this descriptor, as used in the shader. All bindings
    // within a namespace must be unique. (see: ra_desc_namespace)
    int binding;

    // For storage images and storage buffers, this can be used to restrict
    // the type of access that may be performed on the descriptor. Ignored for
    // the other descriptor types (uniform buffers and sampled textures are
    // always read-only).
    enum ra_desc_access access;

    // For RA_DESC_BUF_*, this specifies the layout of the variables contained
    // by a buffer. Ignored for the other descriptor types
    struct ra_buffer_var *buffer_vars;
    int num_buffer_vars;
};

// Framebuffer blending mode (for raster passes)
enum ra_blend_mode {
    RA_BLEND_ZERO,
    RA_BLEND_ONE,
    RA_BLEND_SRC_ALPHA,
    RA_BLEND_ONE_MINUS_SRC_ALPHA,
};

enum ra_prim_type {
    RA_PRIM_TRIANGLE_LIST,
    RA_PRIM_TRIANGLE_STRIP,
    RA_PRIM_TRIANGLE_FAN,
};

enum ra_pass_type {
    RA_PASS_INVALID = 0,
    RA_PASS_RASTER,  // vertex+fragment shader
    RA_PASS_COMPUTE, // compute shader (requires RA_CAP_COMPUTE)
    RA_PASS_TYPE_COUNT,
};

// Description of a rendering pass. It conflates the following:
//  - GLSL shader(s) and its list of inputs
//  - target parameters (for raster passes)
struct ra_pass_params {
    enum ra_pass_type type;

    // Input variables. Only supported if RA_CAP_INPUT_VARIABLES is set.
    // Otherwise, num_variables must be 0.
    struct ra_var *variables;
    int num_variables;

    // Input descriptors. (Always supported)
    struct ra_desc *descriptors;
    int num_descriptors;

    // Push constant region. Must be be a multiple of 4 <= limits.max_pushc_size
    size_t push_constants_size;

    // The shader text in GLSL. For RA_PASS_RASTER, this is interpreted
    // as a fragment shader. For RA_PASS_COMPUTE, this is interpreted as
    // a compute shader.
    const char *glsl_shader;

    // Highly implementation-specific byte array storing a compiled version of
    // the same shader. Can be used to speed up pass creation on already
    // known/cached shaders.
    //
    // Note: There are no restrictions on this. Passing an out-of-date cache,
    // passing a cache corresponding to a different progam, or passing a cache
    // belonging to a different RA, are all valid. But obviously, in such cases,
    // there is no benefit in doing so.
    const uint8_t *cached_program;
    size_t cached_program_len;

    // --- type==RA_PASS_RASTER only

    // Describes the interpretation and layout of the vertex data.
    enum ra_prim_type vertex_type;
    struct ra_vertex_attrib *vertex_attribs;
    int num_vertex_attribs;
    size_t vertex_stride;

    // The vertex shader itself.
    const char *vertex_shader;

    // The target dummy texture this renderpass is intended to be used with.
    // This doesn't have to be a real texture - the caller can also pass a
    // blank ra_tex object, as long as target_dummy.params.format is set. The
    // format must support RA_FMT_CAP_RENDERABLE, and the target dummy must
    // have `renderable` enabled.
    //
    // If you pass a real texture here, the RA backend may be able to optimize
    // the render pass better for the specific requirements of this texture.
    // This does not change the semantics of ra_pass_run, just perhaps the
    // performance. (The `priv` pointer will be cleared by ra_pass_create, so
    // there is no risk of a dangling reference)
    struct ra_tex target_dummy;

    // Target blending mode. If `enable_blend` is true, target_params.format
    // must have RA_FMT_CAP_BLENDABLE. Otherwise, the fields are ignored.
    bool enable_blend;
    enum ra_blend_mode blend_src_rgb;
    enum ra_blend_mode blend_dst_rgb;
    enum ra_blend_mode blend_src_alpha;
    enum ra_blend_mode blend_dst_alpha;

    // If false, the target's existing contents will be discarded before the
    // pass is run. (Semantically equivalent to calling ra_tex_invalidate
    // before every ra_pass_run, but slightly more efficient)
    bool load_target;
};

// Conflates the following typical GPU API concepts:
// - various kinds of shaders
// - rendering pipelines
// - descriptor sets, uniforms, other bindings
// - all synchronization necessary
// - the current values of all inputs
struct ra_pass {
    struct ra_pass_params params;
    void *priv;
};

// Compile a shader and create a render pass. This is a rare/expensive
// operation and may take a significant amount of time, even if a cached
// program is used. Returns NULL on failure.
//
// The resulting ra_pass->params.cached_program will be initialized by
// this function to point to a new, valid cached program (if any).
const struct ra_pass *ra_pass_create(const struct ra *ra,
                                     const struct ra_pass_params *params);

void ra_pass_destroy(const struct ra *ra, const struct ra_pass **pass);

struct ra_desc_binding {
    const void *object; // ra_* object with type corresponding to ra_desc_type
};

struct ra_var_update {
    int index;        // index into params.variables[]
    const void *data; // pointer to raw byte data corresponding to ra_var_host_layout()
};

struct ra_pass_run_params {
    const struct ra_pass *pass;

    // This list only contains descriptors/variables which have changed
    // since the previous invocation. All non-mentioned variables implicitly
    // preserve their state from the last invocation.
    struct ra_var_update *var_updates;
    int num_var_updates;

    // This list contains all descriptors used by this pass. It must
    // always be filled, even if the descriptors haven't changed. The order
    // must match that of pass->params.descriptors
    struct ra_desc_binding *desc_bindings;

    // The push constants for this invocation. This must always be set and
    // fully defined for every invocation if params.push_constants_size > 0.
    void *push_constants;

    // --- pass->params.type==RA_PASS_RASTER only

    // Target must be a 2D texture, target->params.renderable must be true, and
    // target->params.format must match pass->params.target_fmt. If the viewport
    // or scissors are left blank, they are inferred from target->params.
    //
    // WARNING: Rendering to a *target that is being read from by the same
    // shader is undefined behavior. In general, trying to bind the same
    // resource multiple times to the same shader is undefined behavior.
    const struct ra_tex *target;
    struct pl_rect2d viewport; // screen space viewport (must be normalized)
    struct pl_rect2d scissors; // target render scissors (must be normalized)

    void *vertex_data;  // raw pointer to vertex data
    int vertex_count;   // number of vertices to render

    // --- pass->params.type==RA_PASS_COMPUTE only

    // Number of work groups to dispatch per dimension (X/Y/Z). Must be <= the
    // corresponding index of limits.max_dispatch
    int compute_groups[3];
};

// Execute a render pass.
void ra_pass_run(const struct ra *ra, const struct ra_pass_run_params *params);

// This is semantically a no-op, but it provides a hint that you want to flush
// any partially queued up commands and begin execution. There is normally no
// need to call this, because queued commands will always be implicitly flushed
// whenever necessary to make forward progress on commands like `ra_buf_poll`,
// or when submitting a frame to a swapchain for display. In fact, calling this
// function can negatively impact performance, because some RAs rely on being
// able to re-order and modify queued commands in order to enable optimizations
// retroactively.
//
// The only time this might be beneficial to call explicitly is if you're doing
// lots of offline rendering over a long period of time, and only fetching the
// results (via ra_tex_download) at the very end.
void ra_flush(const struct ra *ra);

#endif // LIBPLACEBO_RA_H_
