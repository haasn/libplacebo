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
// NOTE: This abstraction layer is currently not hooked up to any implementation,
// but it is being used to provide types needed by the other abstractions.

typedef uint64_t ra_glsl_caps;
enum {
    RA_GLSL_CAP_SHARED_BINDING = 1 << 0, // descriptor namespaces are separate
};

// Structure which wraps metadata describing GLSL capabilities.
struct ra_glsl_desc {
    int glsl_version;   // GLSL version (e.g. 450), for #version
    bool gles;          // GLSL ES semantics (ESSL)
    bool vulkan;        // GL_KHR_vulkan_glsl semantics
    ra_glsl_caps caps;  // RA_GLSL_CAP_* bit field
};

typedef uint64_t ra_caps;
enum {
    RA_CAP_TEX_1D           = 1 << 0, // supports creating 1D textures
    RA_CAP_TEX_3D           = 1 << 1, // supports creating 3D textures
    RA_CAP_BLIT             = 1 << 2, // supports ra_fns.blit
    RA_CAP_COMPUTE          = 1 << 3, // supports compute shaders
    RA_CAP_UNIFORM_BUFFERS  = 1 << 4, // supports RA_BUF_TYPE_UNIFORM
    RA_CAP_STORAGE_BUFFERS  = 1 << 5, // supports RA_BUF_TYPE_STORAGE
    RA_CAP_INPUT_VARIABLES  = 1 << 6, // supports shader input variables
};

// Abstract device context which wraps an underlying graphics context and can
// be used to dispatch rendering commands.
struct ra {
    struct pl_context *ctx;     // the pl_context this RA was initialized from
    const struct ra_fns *fns;   // the functions corresponding to this RA
    void *priv;

    ra_caps caps;               // RA_CAP_* bit field
    struct ra_glsl_desc glsl;   // GLSL version supported by this RA
    int max_texture_wh;         // maximum W/H of a 2D texture
    size_t max_shmem;           // maximum shared memory for compute shaders
    size_t max_pushc;           // maximum push constant size

    // Supported texture formats, in preference order. (If there are multiple
    // similar formats, the "better" ones come first)
    const struct ra_format **formats;
    int num_formats;
};

enum ra_fmt_type {
    RA_FMT_UNKNOWN = 0, // also used for inconsistent multi-component formats
    RA_FMT_UNORM,       // unsigned, normalized integer format (sampled as float)
    RA_FMT_SNORM,       // signed, normalized integer format (sampled as float)
    RA_FMT_UINT,        // unsigned integer format (sampled as integer)
    RA_FMT_SINT,        // signed integer format (sampled as integer)
    RA_FMT_FLOAT,       // (signed) float formats, any bit size
    RA_FMT_TYPE_COUNT,
};

// Structure describing a texel/vertex format.
struct ra_format {
    const char *name;       // symbolic name for this format (e.g. rgba32f)
    void *priv;

    enum ra_fmt_type type;  // the format's data type and interpretation
    int num_components;     // number of components for this format

    // For the component metadata, the index refers to the index of the
    // component's physical layout - i.e. component_depth[2] refers to the depth
    // of the third physical channel, regardless of whether that's B or R.
    int component_index[4]; // for example, bgra would be {2, 1, 0, 3}
    int component_depth[4]; // meaningful bits for this component
    int component_pad[4];   // padding bits that come *before* this component
    size_t texel_size;      // the number of bytes per texel
    // Note: Trailing padding (e.g. RGBX) is implicitly indicated by pixel_size
    // being larger than the sum of component_depth + component_pad.

    // The features supported by this format
    bool vertex_format;     // may be used as a vertex attribute
    bool texture_format;    // may be used to create textures (ra_tex_create)
    bool sampleable;        // may be sampled from (RA_DESC_SAMPLED_TEX)
    bool linear_filterable; // supports linear filtering when sampling
    bool renderable;        // may be rendered to (ra_renderpass_params.target_format)

    // If usable as a vertex or texel buffer format, this gives the GLSL type
    // corresponding to the data. (e.g. vec4)
    // (Note: texel buffers are currently unsupported)
    const char *glsl_type;
};

// Returns whether or not a ra_format's components are ordered sequentially
// in memory in the order RGBA.
bool ra_format_is_ordered(const struct ra_format *fmt);

// Returns whether or not a ra_format is "regular"; i.e. it's ordered and
// unpadded. In other words, a regular format is any where the representation
// is "trivial" and doesn't require any special re-packing or re-ordering.
bool ra_format_is_regular(const struct ra_format *fmt);

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
    const struct ra_format *format;
    int w, h, d;            // physical dimension; unused dimensions must be 0

    // The following bools describe what operations can be performed
    bool sampleable;        // usable as a RA_DESC_SAMPLED_TEX
    bool renderable;        // usable as a render target (ra_renderpass_run.target)
    bool storage_image;     // must be usable as a storage image (RA_DESC_IMG_*)
    bool blit_src;          // must be usable as a blit source
    bool blit_dst;          // must be usable as a blit destination
    bool host_mutable;      // texture may be updated with tex_upload()

    // The following capabilities are only relevant for textures which have
    // either sampleable or blit_src enabled.
    enum ra_tex_sample_mode sample_mode;
    enum ra_tex_address_mode address_mode;

    // If non-NULL, the texture will be created with these contents. Using
    // this does *not* require setting host_mutable. Otherwise, the initial
    // data is undefined.
    void *initial_data;
};

// Conflates the following typical GPU API concepts:
// - texture itself
// - sampler state
// - staging buffers for texture upload
// - framebuffer objects
// - wrappers for swapchain framebuffers
// - synchronization needed for upload/rendering/etc.
struct ra_tex {
    struct ra_tex_params params;
    void *priv;
};

// Structure describing a texture upload operation.
struct ra_tex_upload_params {
    struct ra_tex *tex; // texture to upload to

    // Note: Superfluous parameters are ignored (e.g. specifying `h` for a 1D
    // image, or specifying `stride_h` for a 2D image.
    int w, h, d;        // extent of the data to upload
    int stride_w;       // the number of texels per horizontal row (x axis)
    int stride_h;       // the number of texels per vertical column (y axis)

    // For the data source of an upload operation, there are two valid options:
    // 1. Uploading from buffer:
    struct ra_buf *buf; // buffer to upload from
    size_t buf_offset;  // offset of data within buffer (bytes)
    // 2. Uploading from host memory:
    const void *src;    // address of data
    // Which data upload method is used is up to the convenience of the user,
    // but they are obviously mutually exclusive. Valid API usage requires
    // that exactly one of *buf or *src is set.

    // Host memory uploads are always supported, although they may be
    // internally translated to buffer pools depending on the capabilities of
    // the underlying API.
};

// Buffer usage type. This restricts what types of operations may be performed
// on a buffer.
enum ra_buf_type {
    RA_BUF_INVALID = 0,
    RA_BUF_TEX_UPLOAD,  // texture upload buffer (for ra_tex_upload)
    RA_BUF_STORAGE,     // SSBO, for RA_DESC_BUF_STORAGE
    RA_BUF_UNIFORM,     // UBO, for RA_DESC_BUF_UNIFORM
    RA_BUF_VERTEX,      // for vertex buffers, no public API (RA-internal)
    RA_BUF_TYPE_COUNT,
};

// Structure describing a buffer.
struct ra_buf_params {
    enum ra_buf_type type;
    size_t size;       // size in bytes
    bool host_mapped;  // create a read-writable persistent mapping (ra_buf.data)
    bool host_mutable; // contents may be updated via buf_update()

    // If non-NULL, the buffer will be created with these contents. Otherwise,
    // the initial data is undefined. Using this does *not* require setting
    // host_mutable.
    void *initial_data;
};

// A generic buffer, which can be used for multiple purposes (texture upload,
// storage buffer, uniform buffer, etc.)
struct ra_buf {
    struct ra_buf_params params;
    void *data; // for persistently mapped buffers, points to the first byte
    void *priv;
};

// Represents a vertex attribute.
struct ra_vertex_attrib {
    const char *name;            // name as used in the shader
    const struct ra_format *fmt; // data format (must have `vertex_format` set)
    size_t offset;               // byte offset into the vertex struct
};

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
    int dim_m;              // matrix dimension (number of columns)
};

// Represents the layout requirements of an input variable
struct ra_var_layout {
    size_t align;  // the alignment requirements (always a power of two)
    size_t stride; // the delta between two rows of an array/matrix
    size_t size;   // the total size of the input
};

// Returns the host layout of an input variable as required for a
// tightly-packed, byte-aligned C data type.
struct ra_var_layout ra_var_host_layout(const struct ra_var var);

// Type of a shader input descriptor.
enum ra_desc_type {
    RA_DESC_INVALID = 0,
    RA_DESC_SAMPLED_TEX,    // C: ra_tex*    GLSL: combined texture sampler
                            // (ra_tex->params.sampleable must be set)
    RA_DESC_STORAGE_IMG,    // C: ra_tex*    GLSL: readonly storage image
                            // (ra_tex->params.storage_image must be set)
    RA_DESC_BUF_UNIFORM,    // C: ra_buf*    GLSL: uniform buffer
                            // (ra_buf->params.type must be RA_BUF_UNIFORM)
    RA_DESC_BUF_STORAGE,    // C: ra_buf*    GLSL: storage buffer
                            // (ra_buf->params.type must be RA_BUF_STORAGE)
    RA_DESC_TYPE_COUNT
};

// Access mode of a shader input descriptor.
enum ra_desc_access {
    RA_DESC_ACCESS_READWRITE,
    RA_DESC_ACCESS_READONLY,
    RA_DESC_ACCESS_WRITEONLY,
};

// Represents a shader descriptor (e.g. texture or buffer binding)
struct ra_desc {
    const char *name;       // name as used in the shader
    enum ra_desc_type type;

    // If RA_GLSL_CAP_SHARED_BINDING is set, the bindings for each descriptor
    // type are separate, and the same binding point may be used for different
    // descriptors as long as they have a different type. Otherwise, all
    // bindings share the same namespace and must be unique for every
    // descriptor.
    int binding;

    // For storage images and storage buffers, this can be used to restrict
    // the type of access that may be performed on the descriptor. Ignored for
    // the other descriptor types
    enum ra_desc_access access;

    // For RA_DESC_BUF_*, this specifies the GLSL layout of the buffer (not
    // including the surrounding { } braces)
    const char *buffer_layout;
};

// Framebuffer blending mode (for raster renderpasses)
enum ra_blend_mode {
    RA_BLEND_ZERO,
    RA_BLEND_ONE,
    RA_BLEND_SRC_ALPHA,
    RA_BLEND_ONE_MINUS_SRC_ALPHA,
};

enum ra_renderpass_type {
    RA_RENDERPASS_INVALID = 0,
    RA_RENDERPASS_RASTER,  // vertex+fragment shader
    RA_RENDERPASS_COMPUTE, // compute shader
    RA_RENDERPASS_TYPE_COUNT,
};

// Description of a rendering pass. It conflates the following:
//  - GLSL shader and its list of inputs
//  - target parameters (for raster passes)
struct ra_renderpass_params {
    enum ra_renderpass_type type;

    // Input variables. Only supported if RA_CAP_INPUT_VARIABLES is set.
    struct ra_var *variables;
    int num_variables;

    // Input descriptors
    struct ra_desc *descriptors;
    int num_descriptors;

    // Push constant region. Must be <= ra.max_pushc, and a multiple of 4
    size_t push_constants_size;

    // The shader text in GLSL. For RA_RENDERPASS_RASTER, this is interpreted
    // as a fragment shader. For RA_RENDERPASS_COMPUTE, this is interpreted as
    // a compute shader.
    const char *glsl_shader;

    // --- type==RA_RENDERPASS_RASTER only

    // Describes the format of the vertex data.
    struct ra_vertex_attrib *vertex_attribs;
    int num_vertex_attribs;
    size_t vertex_stride;

    // The vertex shader itself.
    const char *vertex_shader;

    // Format of the target texture. Must have `renderable` set.
    const struct ra_format *target_format;

    // Target blending mode. If enable_blend is false, the blend_ fields are
    // ignored.
    bool enable_blend;
    enum ra_blend_mode blend_src_rgb;
    enum ra_blend_mode blend_dst_rgb;
    enum ra_blend_mode blend_src_alpha;
    enum ra_blend_mode blend_dst_alpha;

    // If true, the contents of `target` not written to will become undefined.
    bool invalidate_target;
};

// Conflates the following typical GPU API concepts:
// - various kinds of shaders
// - rendering pipelines
// - descriptor sets, uniforms, other bindings
// - all synchronization necessary
// - the current values of all inputs
struct ra_renderpass {
    // All fields are read-only after creation.
    struct ra_renderpass_params params;
    void *priv;
};

struct ra_desc_update {
    int index;  // index into params.descriptors[]
    void *data; // ra_* object with type corresponding to ra_desc_type
};

struct ra_var_update {
    int index;  // index into params.variables[]
    void *data; // pointer to raw byte data corresponding to ra_var_host_layout()
};

enum ra_prim_type {
    RA_PRIM_TRIANGLE_LIST,
    RA_PRIM_TRIANGLE_STRIP,
    RA_PRIM_TRIANGLE_FAN,
};

// Parameters for running a renderpass. These are expected to change often.
struct ra_renderpass_run_params {
    struct ra_renderpass *pass;

    // These lists only contain descriptors/variables which have changed
    // since the previous invocation. All non-mentioned inputs implicitly
    // preserve their state from the last invocation.
    struct ra_desc_update *desc_updates;
    struct ra_var_update *var_updates;
    int num_desc_updates;
    int num_var_updates;

    // The push constants for this invocation. This must always be set and
    // fully defined for every invocation iff params.push_constants_size > 0.
    void *push_constants;

    // --- pass->params.type==RA_RENDERPASS_TYPE_RASTER only

    // target->params.renderable must be true, and target->params.format must
    // match pass->params.target_format. Target must be a 2D texture.
    struct ra_tex *target;
    struct pl_rect2d viewport;
    struct pl_rect2d scissors;

    enum ra_prim_type vertex_type;
    void *vertex_data;  // raw pointer to vertex data
    int vertex_count;   // number of vertices to render

    // --- pass->params.type==RA_RENDERPASS_TYPE_COMPUTE only

    // Number of work groups to dispatch. (X/Y/Z)
    int compute_groups[3];
};

// This is a fully opaque type provided by the implementation, but we want to
// at least give it a saner name than void* for code readability purposes.
typedef void ra_timer;

// RA functions. These give the operations supported by a RA. All functions
// must be set except where otherwise noted.
struct ra_fns {
    // This also frees the RA.
    void (*destroy)(const struct ra **ra);

    // Create a texture (with undefined contents). Return NULL on failure.
    // This is a rare operation, and normally textures and even FBOs for
    // temporary rendering intermediate data are cached.
    const struct ra_tex *(*tex_create)(const struct ra *ra,
                                       const struct ra_tex_params *params);

    void (*tex_destroy)(const struct ra *ra, const struct ra_tex *tex);

    // Upload data to a texture. This is an extremely common operation. When
    // using a buffer, the contants of the buffer must exactly match the format
    // as described by the texture's ra_format - conversions between bit depths
    // and representations are not supported. This operation may mark the buffer
    // as "in use" while the copy is going on. Returns whether successful.
    bool (*tex_upload)(const struct ra *ra,
                       const struct ra_tex_upload_params *params);

    // Create a buffer. The type of buffer depends on the parameters. Not all
    // buffer usage types must be supported; may return NULL if unavailable.
    const struct ra_buf *(*buf_create)(const struct ra *ra,
                                       const struct ra_buf_params *params);

    void (*buf_destroy)(const struct ra *ra, const struct ra_buf *buf);

    // Update the contents of a buffer, starting at a given offset (*must* be a
    // multiple of 4) and up to a given size, with the contents of *data. This
    // is an extremely common operation. Calling this while the buffer is
    // considered "in use" is an error. (See: buf_poll)
    void (*buf_update)(const struct ra *ra, const struct ra_buf *buf,
                       size_t buf_offset, const void *data, size_t size);

    // Returns if a buffer is currently "in use" or not. Updating the contents
    // of a buffer (via buf_update or writes to buf->data) while it is still
    // in use is an error and may result in graphical corruption. Optional, if
    // NULL then all buffers are always usable.
    bool (*buf_poll)(const struct ra *ra, const struct ra_buf *buf);

    // Returns the layout requirements of a uniform buffer element. Optional,
    // but must be implemented if RA_CAP_UNIFORM_BUFFERS is supported.
    // Note: This is normally equivalent to std140 layout, but not necessarily
    // (for example, RAs based on d3d11 may internally translate std140 to a
    // different layout). As such, the calling code should not make any
    // assumptions about the buffer layout and instead query the layout
    // requirements using this function.
    struct ra_var_layout (*uniform_layout)(const struct ra_var *var);

    // Returns the layout requirements of a push constant element. Optional,
    // but must be implemented if ra.max_pushc > 0.
    struct ra_var_layout (*push_constant_layout)(const struct ra_var *var);

    // Clear the dst texture with the given color (rgba) and within the given
    // region. This is functionally identical to a blit operation, which means
    // dst->params.blit_dst must be set. Content outside of the scissor is
    // preserved. Note: Clearing a partial region of an image may perform
    // significantly worse than clearing the entire image, and should be
    // avoided.
    void (*clear)(const struct ra *ra, const struct ra_tex *dst,
                  struct pl_rect3d rect, const float color[4]);

    // Copy a sub-rectangle from one texture to another. The source/dest
    // regions must be within the texture bounds. Areas outside the dest region
    // are preserved. The formats of the textures must be loosely compatible -
    // which essentially means that they must have the same texel size.
    // src.blit_src and dst.blit_dst must be set, respectively. The rectangles
    // may be "negative", which leads to the image being flipped while
    // blitting. If the src and dst rects have different sizes, the source
    // image will be scaled according to src->params.sample_mode. Required if
    // RA_CAP_BLIT is present. If RA_CAP_BLIT is not set, this function is
    // optional - and must not be called even if it's non-NULL.
    void (*blit)(const struct ra *ra,
                 const struct ra_tex *dst, const struct ra_tex *src,
                 struct pl_rect3d dst_rc, struct pl_rect3d src_rc);

    // Compile a shader and create a render pass. This is a rare operation.
    const struct ra_renderpass *(*renderpass_create)(const struct ra *ra,
                                    const struct ra_renderpass_params *params);

    void (*renderpass_destroy)(const struct ra *ra,
                               const struct ra_renderpass *pass);

    // Execute a render pass. This is an extremely common operation.
    void (*renderpass_run)(const struct ra *ra,
                           const struct ra_renderpass_run_params *params);

    // Create a timer object. Returns NULL on failure, or if timers are
    // unavailable for some reason. Optional.
    const ra_timer *(*timer_create)(const struct ra *ra);

    void (*timer_destroy)(const struct ra *ra, const ra_timer *timer);

    // Start recording a timer. Note that valid usage requires you to pair
    // every start with a stop. Trying to start a timer twice, or trying to
    // stop a timer before having started it, consistutes invalid usage.
    void (*timer_start)(const struct ra *ra, const ra_timer *timer);

    // Stop recording a timer. This also returns any results that have been
    // measured since the last usage of this ra_timer, in nanoseconds. It's
    // important to note that GPU timer measurement are asynchronous, so this
    // function does not always produce a value - and the values it does
    // produce are typically delayed by a few frames. When no value is
    // available, this returns 0.
    uint64_t (*timer_stop)(const struct ra *ra, const ra_timer *timer);
};

// Utility functions for common operations, to alleviate the need for the
// ra->fns-> boilerplate.
const struct ra_tex *ra_tex_create(const struct ra *ra,
                                   const struct ra_tex_params *params);
void ra_tex_destroy(const struct ra *ra, const struct ra_tex **tex);

const struct ra_buf *ra_buf_create(const struct ra *ra,
                                   const struct ra_buf_params *params);
void ra_buf_destroy(const struct ra *ra, const struct ra_buf **buf);

// Helper function to find a format with a given number of components and depth
// per component. The format must be usable as a texture format. If `regular`
// is true, ra_format_is_regular() must be true.
const struct ra_format *ra_find_texture_format(const struct ra *ra,
                                               enum ra_fmt_type type,
                                               int num_components,
                                               int bits_per_component,
                                               bool regular);

// Find a format based on its name.
const struct ra_format *ra_find_named_format(const struct ra *ra,
                                             const char *name);

#endif // LIBPLACEBO_RA_H_
