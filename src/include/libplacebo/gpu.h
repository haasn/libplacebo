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
// NOTE: Most, but not all, parameter conditions (phrases such as "must" or
// "valid usage" are explicitly tested and result in error messages followed by
// graceful failure. Exceptions are noted where they exist.

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
    PL_GPU_CAP_MAPPED_BUFFERS   = 1 << 3, // supports host-mapped buffers
    PL_GPU_CAP_BLITTABLE_1D_3D  = 1 << 4, // supports blittable 1D/3D textures
    PL_GPU_CAP_SUBGROUPS        = 1 << 5, // supports subgroups

    // Note on subgroup support: PL_GPU_CAP_SUBGROUPS implies subgroup support
    // for both fragment and compute shaders, but not necessarily any other
    // stages. In terms of the supported feature set, currently at least the
    // following capabilities are required:
    // - basic
    // - voting
    // - arithmetic
    // - ballot
    // - shuffle
};

// Some `pl_gpu` operations allow sharing GPU resources with external APIs -
// examples include interop with other graphics APIs such as CUDA, and also
// various hardware decoding APIs. This defines the mechanism underpinning the
// communication of such an interoperation.
typedef uint64_t pl_handle_caps;
enum pl_handle_type {
    PL_HANDLE_FD        = (1 << 0), // `int fd` for POSIX-style APIs
    PL_HANDLE_WIN32     = (1 << 1), // `HANDLE` for win32 API
    PL_HANDLE_WIN32_KMT = (1 << 2), // `HANDLE` for pre-Windows-8 win32 API
    PL_HANDLE_DMA_BUF   = (1 << 3), // 'int fd' for a dma_buf fd
    PL_HANDLE_HOST_PTR  = (1 << 4), // `void *` for a host-allocated pointer
};

struct pl_gpu_handle_caps {
    pl_handle_caps tex;  // supported handles for `pl_tex` + `pl_shared_mem`
    pl_handle_caps buf;  // supported handles for `pl_buf` + `pl_shared_mem`
    pl_handle_caps sync; // supported handles for `pl_sync`
};

// Wrapper for the handle used to communicate a shared resource externally.
// This handle is owned by the `pl_gpu` - if a user wishes to use it in a way
// that takes over ownership (e.g. importing into some APIs), they must clone
// the handle before doing so (e.g. using `dup` for fds). It is important to
// read the external API documentation _very_ carefully as different handle
// types may be managed in different ways. (eg: CUDA takes ownership of an fd,
// but does not take ownership of a win32 handle).
union pl_handle {
    int fd;         // PL_HANDLE_FD / PL_HANDLE_DMA_BUF
    void *handle;   // PL_HANDLE_WIN32 / PL_HANDLE_WIN32_KMT
    void *ptr;      // PL_HANDLE_HOST_PTR
};

// Structure encapsulating memory that is shared between libplacebo and the
// user. This memory can be imported into external APIs using the handle.
//
// If the object a `pl_shared_mem` belongs to is destroyed (e.g. via
// `pl_buf_destroy`), the handle becomes undefined, as do the contents of the
// memory it points to, as well as any external API objects imported from it.
struct pl_shared_mem {
    union pl_handle handle;
    size_t size;   // the total size of the memory referenced by this handle
    size_t offset; // the offset of the object within the referenced memory

    // For PL_HANDLE_DMA_BUF, this specifies the DRM format modifier that
    // describes this resource. Note that when importing `pl_buf`, this must
    // be DRM_FORMAT_MOD_LINEAR. For importing `pl_tex`, it can be any
    // format modifier supported by the implementation.
    uint64_t drm_format_mod;
};

// Structure defining the physical limits of this GPU instance. If a limit is
// given as 0, that means that feature is unsupported.
struct pl_gpu_limits {
    uint32_t max_tex_1d_dim;    // maximum width for a 1D texture
    uint32_t max_tex_2d_dim;    // maximum width/height for a 2D texture (required)
    uint32_t max_tex_3d_dim;    // maximum width/height/depth for a 3D texture
    size_t max_pushc_size;      // maximum `push_constants_size`
    size_t max_buf_size;        // maximum size of any buffer
    size_t max_ubo_size;        // maximum size of a `uniform` buffer
    size_t max_ssbo_size;       // maximum size of a `storable` buffer
    size_t max_vbo_size;        // maximum size of a `drawable` buffer
    uint64_t max_buffer_texels; // maximum number of texels in a texel buffer
    int16_t min_gather_offset;  // minimum `textureGatherOffset` offset
    int16_t max_gather_offset;  // maximum `textureGatherOffset` offset
    uint32_t subgroup_size;     // number of threads in a subgroup

    // Compute shader limits. Always available (non-zero) if PL_GPU_CAP_COMPUTE set
    size_t max_shmem_size;      // maximum compute shader shared memory size
    uint32_t max_group_threads; // maximum number of local threads per work group
    uint32_t max_group_size[3]; // maximum work group size per dimension
    uint32_t max_dispatch[3];   // maximum dispatch size per dimension

    // These don't represent hard limits but indicate performance hints for
    // optimal alignment. For best performance, the corresponding field
    // should be aligned to a multiple of these. They will always be a power
    // of two.
    uint32_t align_tex_xfer_stride; // optimal `pl_tex_transfer_params.stride_w/h`
    size_t align_tex_xfer_offset;   // optimal `pl_tex_transfer_params.buf_offset`
};

// Backwards compatibility alias
#define max_xfer_size max_buf_size

// Structure grouping PCI bus address fields for GPU devices
struct pl_gpu_pci_address {
    uint32_t domain;
    uint32_t bus;
    uint32_t device;
    uint32_t function;
};

// Abstract device context which wraps an underlying graphics context and can
// be used to dispatch rendering commands.
struct pl_gpu {
    struct pl_context *ctx;  // the `pl_context` this GPU was initialized from

    pl_gpu_caps caps;            // PL_GPU_CAP_* bit field
    struct pl_glsl_desc glsl;    // GLSL version supported by this GPU
    struct pl_gpu_limits limits; // physical device limits
    // Note: Every GPU must support at least one of PL_GPU_CAP_INPUT_VARIABLES
    // or uniform buffers (limits.max_ubo_size > 0).

    // Fields relevant to external API interop. If the underlying device does
    // not support interop with other APIs, these will all be {0}.
    struct pl_gpu_handle_caps export_caps; // supported handles for exporting
    struct pl_gpu_handle_caps import_caps; // supported handles for importing
    uint8_t uuid[16];                      // underlying device UUID

    // Supported texture formats, in preference order. (If there are multiple
    // similar formats, the "better" ones come first)
    const struct pl_fmt **formats;
    int num_formats;

    // PCI Bus address of the underlying device, to help with interop.
    // This will only be filled in if interop is supported.
    struct pl_gpu_pci_address pci;
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
    PL_FMT_CAP_HOST_READABLE = 1 << 9, // may be used with `host_readable` textures

    // Notes:
    // - PL_FMT_CAP_LINEAR also implies PL_FMT_CAP_SAMPLEABLE
    // - PL_FMT_CAP_STORABLE also implies PL_GPU_CAP_COMPUTE
    // - PL_FMT_CAP_BLENDABLE implies PL_FMT_CAP_RENDERABLE
    // - PL_FMT_CAP_VERTEX implies that the format is non-opaque
    // - PL_FMT_CAP_HOST_READABLE implies that the format is non-opaque
};

// Structure describing a texel/vertex format.
struct pl_fmt {
    const char *name;       // symbolic name for this format (e.g. rgba32f)

    enum pl_fmt_type type;  // the format's data type and interpretation
    enum pl_fmt_caps caps;  // the features supported by this format
    int num_components;     // number of components for this format
    int component_depth[4]; // meaningful bits per component, texture precision
    size_t internal_size;   // internal texel size (for blit compatibility)

    // This controls the relationship between the data as seen by the host and
    // the way it's interpreted by the texture. The host representation is
    // always tightly packed (no padding bits in between each component).
    //
    // This representation assumes little endian ordering, i.e. components
    // being ordered from LSB to MSB in memory. Note that for oddly packed
    // formats like rgb10a2 or rgb565, this is inconsistent with the naming.
    // (That is to say, rgb565 has sample order {2, 1, 0} under this convention
    // - because rgb565 treats the R channel as the *most* significant bits)
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
    // texel format corresponding to the format (e.g. rgba16ui), if any. This
    // field may be NULL, in which case the format modifier may be left
    // unspecified.
    const char *glsl_format;

    // If non-opaque, this gives the fourcc associated with the host
    // representation. In particular, this is intended for use with
    // PL_HANDLE_DMA_BUF, where this field will match the DRM format from
    // <drm_fourcc.h>. May be 0, for formats without matching DRM fourcc.
    uint32_t fourcc;
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

// Find a format based on its fourcc.
const struct pl_fmt *pl_find_fourcc(const struct pl_gpu *gpu, uint32_t fourcc);

enum pl_tex_sample_mode {
    PL_TEX_SAMPLE_NEAREST,  // nearest neighour sampling
    PL_TEX_SAMPLE_LINEAR,   // linear filtering
    PL_TEX_SAMPLE_MODE_COUNT,
};

enum pl_tex_address_mode {
    PL_TEX_ADDRESS_CLAMP,  // clamp the nearest edge texel
    PL_TEX_ADDRESS_REPEAT, // repeat (tile) the texture
    PL_TEX_ADDRESS_MIRROR, // repeat (mirror) the texture
    PL_TEX_ADDRESS_MODE_COUNT,
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

    // Note: For `blit_src`, `blit_dst`, the texture must either be
    // 2-dimensional or `PL_GPU_CAP_BLITTABLE_1D_3D` must set.

    // The following capabilities are only relevant for textures which have
    // either sampleable or blit_src enabled.
    enum pl_tex_sample_mode sample_mode;
    enum pl_tex_address_mode address_mode;

    // At most one of `export_handle` and `import_handle` can be set for a
    // texture.

    // Setting this indicates that the memory backing this texture should be
    // shared with external APIs, If so, this must be exactly *one* of
    // `pl_gpu.export_caps.tex`.
    enum pl_handle_type export_handle;

    // Setting this indicates that the memory backing this texture will be
    // imported from an external API. If so, this must be exactly *one* of
    // `pl_gpu.import_caps.tex`. Mutually exclusive with `initial_data`.
    enum pl_handle_type import_handle;

    // If the shared memory is being imported, the import handle must be
    // specified here. Otherwise, this is ignored.
    struct pl_shared_mem shared_mem;

    // If non-NULL, the texture will be created with these contents (tightly
    // packed). Using this does *not* require setting host_writable. Otherwise,
    // the initial data is undefined. Mutually exclusive with `import_handle`.
    const void *initial_data;

    // Arbitrary user data. libplacebo does not use this at all.
    void *user_data;
};

static inline int pl_tex_params_dimension(const struct pl_tex_params params)
{
    return params.d ? 3 : params.h ? 2 : 1;
}

enum pl_sampler_type {
    PL_SAMPLER_NORMAL,      // gsampler2D, gsampler3D etc.
    PL_SAMPLER_RECT,        // gsampler2DRect
    PL_SAMPLER_EXTERNAL,    // gsamplerExternalOES
    PL_SAMPLER_TYPE_COUNT,
};

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

    // If `params.export_handle` is set, this structure references the shared
    // memory backing this buffer, via the requested handle type.
    //
    // While this texture is not in an "exported" state, the contents of the
    // memory are undefined. (See: `pl_tex_export`)
    //
    // Note: Due to vulkan driver limitations, `shared_mem.drm_format_mod` will
    // currently always be set to DRM_FORMAT_MOD_INVALID. No guarantee can be
    // made about the cross-driver compatibility of textures exported this way.
    struct pl_shared_mem shared_mem;

    // If `params.sampleable` is true, this indicates the correct sampler type
    // to use when sampling from this texture.
    enum pl_sampler_type sampler_type;
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
//
// Note: If the `user_data` alone changes, this does not trigger a texture
// recreation. In theory, this can be used to detect when the texture ended
// up being recreated.
bool pl_tex_recreate(const struct pl_gpu *gpu, const struct pl_tex **tex,
                     const struct pl_tex_params *params);

// Invalidates the contents of a texture. After this, the contents are fully
// undefined.
void pl_tex_invalidate(const struct pl_gpu *gpu, const struct pl_tex *tex);

// Clear the dst texture with the given color (rgba). This is functionally
// identical to a blit operation, which means `dst->params.blit_dst` must be
// set.
void pl_tex_clear(const struct pl_gpu *gpu, const struct pl_tex *dst,
                  const float color[4]);

// Copy a sub-rectangle from one texture to another. The source/dest regions
// must be within the texture bounds. Areas outside the dest region are
// preserved. The formats of the textures must be loosely compatible - which
// essentially means that they must have the same `internal_size`. Additionally,
// UINT textures can only be blitted to other UINT textures, and SINT textures
// can only be blitted to other SINT textures. Finally, `src.blit_src` and
// `dst.blit_dst` must be set, respectively.
//
// The rectangles may be "flipped", which leads to the image being flipped
// while blitting. If the src and dst rects have different sizes, the source
// image will be scaled according to `src->params.sample_mode`. That said, the
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

    // An optional timer to report the approximate duration of the texture
    // transfer to. Note that this is only an approximation, since the actual
    // texture transfer may happen entirely in the background (in particular,
    // for implementations with asynchronous transfer capabilities). It's also
    // not guaranteed that all GPUs support this.
    struct pl_timer *timer;

    // For the data source/target of a transfer operation, there are two valid
    // options:
    //
    // 1. Transferring to/from a buffer:
    const struct pl_buf *buf; // buffer to use
    size_t buf_offset;        // offset of data within buffer, should be a
                              // multiple of `tex->params.format->texel_size`
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
};

// Upload data to a texture. Returns whether successful.
bool pl_tex_upload(const struct pl_gpu *gpu,
                   const struct pl_tex_transfer_params *params);

// Download data from a texture. Returns whether successful.
bool pl_tex_download(const struct pl_gpu *gpu,
                     const struct pl_tex_transfer_params *params);

// (Deprecated) Buffer usage type. This defines what types of operations may be
// performed on a buffer. They are defined merely for backwards compatibility,
// and correspond to merely enabling the respective usage flags.
enum pl_buf_type {
    PL_BUF_TEX_TRANSFER,  // no extra usage flags
    PL_BUF_UNIFORM,       // enables `uniform`
    PL_BUF_STORAGE,       // enables `storable`
    PL_BUF_TEXEL_UNIFORM, // equivalent to PL_BUF_UNIFORM (when `format` is set)
    PL_BUF_TEXEL_STORAGE, // equivalent to PL_BUF_STORAGE (when `format` is set)
    PL_BUF_TYPE_COUNT,
};

enum pl_buf_mem_type {
    PL_BUF_MEM_AUTO = 0, // use whatever seems most appropriate
    PL_BUF_MEM_HOST,     // try allocating from host memory (RAM)
    PL_BUF_MEM_DEVICE,   // try allocating from device memory (VRAM)
    PL_BUF_MEM_TYPE_COUNT,

    // Note: This distinction only matters for discrete GPUs
};

// Structure describing a buffer.
struct pl_buf_params {
    size_t size;        // size in bytes (must be <= `pl_gpu_limits.max_buf_size`)
    bool host_writable; // contents may be updated via pl_buf_write()
    bool host_readable; // contents may be read back via pl_buf_read()
    bool host_mapped;   // create a persistent, RW mapping (pl_buf.data)
                        // (requires PL_GPU_CAP_MAPPED_BUFFERS)

    // May be used as PL_DESC_BUF_UNIFORM or PL_DESC_BUF_TEXEL_UNIFORM.
    // Requires `size <= pl_gpu_limits.max_ubo_size`
    bool uniform;

    // May be used as PL_DESC_BUF_STORAGE or PL_DESC_BUF_TEXEL_STORAGE.
    // Requires `size <= pl_gpu_limits.max_ssbo_size`
    bool storable;

    // May be used as the source of vertex data for `pl_pass_run`.
    bool drawable;

    // Provide a hint for the memory type you want to use when allocating
    // this buffer's memory.
    //
    // Note: Restrictions may apply depending on the usage flags. In
    // particular, allocating buffers with `uniform` or `storable` enabled from
    // non-device memory will almost surely fail.
    enum pl_buf_mem_type memory_type;

    // Setting this to a format with the `PL_FMT_CAP_TEXEL_*` capability allows
    // this buffer to be used as a `PL_DESC_BUF_TEXEL_*`, when `uniform` and
    // `storage` are respectively also enabled.
    const struct pl_fmt *format;

    // At most one of `export_handle` and `import_handle` can be set for a
    // buffer.

    // Setting this indicates that the memory backing this buffer should be
    // shared with external APIs, If so, this must be exactly *one* of
    // `pl_gpu.export_caps.buf`.
    enum pl_handle_type export_handle;

    // Setting this indicates that the memory backing this buffer will be
    // imported from an external API. If so, this must be exactly *one* of
    // `pl_gpu.import_caps.buf`.
    enum pl_handle_type import_handle;

    // If the shared memory is being imported, the import handle must be
    // specified here. Otherwise, this is ignored.
    struct pl_shared_mem shared_mem;

    // If non-NULL, the buffer will be created with these contents. Otherwise,
    // the initial data is undefined. Using this does *not* require setting
    // host_writable.
    const void *initial_data;

    // Arbitrary user data. libplacebo does not use this at all.
    void *user_data;

    // Deprecated. Setting a type now effectively just enables some of the
    // buffer usage flags. See `pl_buf_type`. This field will be removed
    // in the future.
    enum pl_buf_type type PL_DEPRECATED;
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

    // If `params.handle_type` is set, this structure references the shared
    // memory backing this buffer, via the requested handle type.
    //
    // While this buffer is not in an "exported" state, the contents of the
    // memory are undefined. (See: `pl_buf_export`)
    struct pl_shared_mem shared_mem;
};

// Create a buffer. The type of buffer depends on the parameters. The buffer
// parameters must adhere to the restrictions imposed by the pl_gpu_limits.
// Returns NULL on failure.
//
// For buffers with shared memory, the buffer is considered to be in an
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
// with `params->export_handle`. since this may invalidate the corresponding
// external API's handle. Conversely, it *is* allowed on a buffer with
// `params->host_mapped`, and the corresponding `buf->data` pointer *may*
// change as a result of doing so.
//
// Note: If the `user_data` alone changes, this does not trigger a buffer
// recreation. In theory, this can be used to detect when the buffer ended
// up being recreated.
bool pl_buf_recreate(const struct pl_gpu *gpu, const struct pl_buf **buf,
                     const struct pl_buf_params *params);

void pl_buf_destroy(const struct pl_gpu *gpu, const struct pl_buf **buf);

// Update the contents of a buffer, starting at a given offset (must be a
// multiple of 4) and up to a given size, with the contents of *data.
//
// This function will block until the buffer is no longer in use. Use
// `pl_buf_poll` to perform non-blocking queries of buffer availability.
//
// Note: This function can incur synchronization overhead, so it shouldn't be
// used in tight loops. If you do need to loop (e.g. to perform a strided
// write), consider using host-mapped buffers, or fixing the memory in RAM,
// before calling this function.
void pl_buf_write(const struct pl_gpu *gpu, const struct pl_buf *buf,
                  size_t buf_offset, const void *data, size_t size);

// Read back the contents of a buffer, starting at a given offset, storing the
// data into *dest. Returns whether successful.
//
// This function will block until the buffer is no longer in use. Use
// `pl_buf_poll` to perform non-blocking queries of buffer availability.
bool pl_buf_read(const struct pl_gpu *gpu, const struct pl_buf *buf,
                 size_t buf_offset, void *dest, size_t size);

// Copy `size` bytes from one buffer to another, reading from and writing to
// the respective offsets.
void pl_buf_copy(const struct pl_gpu *gpu,
                 const struct pl_buf *dst, size_t dst_offset,
                 const struct pl_buf *src, size_t src_offset,
                 size_t size);

// Initiates a buffer export operation, allowing a buffer to be accessed by an
// external API. This is only valid for buffers with `params.handle_type`.
// Calling this twice in a row is a harmless no-op. Returns whether successful.
//
// There is no corresponding "buffer import" operation, the next libplacebo
// operation that touches the buffer (e.g. pl_tex_upload, but also pl_buf_write
// and pl_buf_read) will implicitly import the buffer back to libplacebo. Users
// must ensure that all pending operations made by the external API are fully
// completed before using it in libplacebo again. (Otherwise, the behaviour
// is undefined)
//
// Please note that this function returning does not mean the memory is
// immediately available as such. In general, it will mark a buffer as "in use"
// in the same way any other buffer operation would, and it is the user's
// responsibility to wait until `pl_buf_poll` returns false before accessing
// the memory from the external API.
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
// re-using it in the external API. (Or get a new buffer in the meantime)
bool pl_buf_export(const struct pl_gpu *gpu, const struct pl_buf *buf);

// Returns whether or not a buffer is currently "in use". This can either be
// because of a pending read operation, a pending write operation or a pending
// buffer export operation. Any access to the buffer by external APIs or via
// the host pointer (for host-mapped buffers) is forbidden while a buffer is
// "in use". The only exception to this rule is multiple reads, for example
// reading from a buffer with `pl_tex_upload` while simultaneously reading from
// it using mapped memory.
//
// The `timeout`, specified in nanoseconds, indicates how long to block for
// before returning. If set to 0, this function will never block, and only
// returns the current status of the buffer. The actual precision of the
// timeout may be significantly longer than one nanosecond, and has no upper
// bound. This function does not provide hard latency guarantees. This function
// may also return at any time, even if the buffer is still in use. If the user
// wishes to block until the buffer is definitely no longer in use, the
// recommended usage is:
//
// while (pl_buf_poll(gpu, buf, UINT64_MAX))
//      ; // do nothing
//
// Note: libplacebo operations on buffers are always internally synchronized,
// so this is only needed for host-mapped or externally exported buffers.
// However, it may be used to do non-blocking queries before calling blocking
// functions such as `pl_buf_read`.
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

// Helper functions for constructing the most common pl_vars, with names
// corresponding to their corresponding GLSL built-in types.
struct pl_var pl_var_float(const char *name);
struct pl_var pl_var_vec2(const char *name);
struct pl_var pl_var_vec3(const char *name);
struct pl_var pl_var_vec4(const char *name);
struct pl_var pl_var_mat2(const char *name);
struct pl_var pl_var_mat2x3(const char *name);
struct pl_var pl_var_mat2x4(const char *name);
struct pl_var pl_var_mat3(const char *name);
struct pl_var pl_var_mat3x4(const char *name);
struct pl_var pl_var_mat4x2(const char *name);
struct pl_var pl_var_mat4x3(const char *name);
struct pl_var pl_var_mat4(const char *name);
struct pl_var pl_var_int(const char *name);
struct pl_var pl_var_ivec2(const char *name);
struct pl_var pl_var_ivec3(const char *name);
struct pl_var pl_var_ivec4(const char *name);
struct pl_var pl_var_uint(const char *name);
struct pl_var pl_var_uvec2(const char *name);
struct pl_var pl_var_uvec3(const char *name);
struct pl_var pl_var_uvec4(const char *name);

struct pl_named_var {
    const char *glsl_name;
    struct pl_var var;
};

// The same list as above, tagged by name and terminated with a {0} entry.
extern const struct pl_named_var pl_var_glsl_types[];

// Efficient helper function for performing a lookup in the above array.
// Returns NULL if the variable is not legal. Note that the array dimension is
// ignored, since it's usually part of the variable name and not the type name.
const char *pl_var_glsl_type_name(struct pl_var var);

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
// offset, as required for a buffer descriptor of type PL_DESC_BUF_UNIFORM
//
// The normal way to use this function is when calculating the size and offset
// requirements of a uniform buffer in an incremental fashion, to calculate the
// new offset of the next variable in this buffer.
struct pl_var_layout pl_std140_layout(size_t offset, const struct pl_var *var);

// Returns the GLSL std430 layout of an input variable given a current buffer
// offset, as required for a buffer descriptor of type PL_DESC_BUF_STORAGE, and
// for push constants.
struct pl_var_layout pl_std430_layout(size_t offset, const struct pl_var *var);

// Convenience definitions / friendly names for these
#define pl_buf_uniform_layout pl_std140_layout
#define pl_buf_storage_layout pl_std430_layout
#define pl_push_constant_layout pl_std430_layout

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
                            // (`pl_tex->params.sampleable` must be set)
    PL_DESC_STORAGE_IMG,    // C: pl_tex*    GLSL: storage image
                            // (`pl_tex->params.storable` must be set)
    PL_DESC_BUF_UNIFORM,    // C: pl_buf*    GLSL: uniform buffer
                            // (`pl_buf->params.uniform` must be set)
    PL_DESC_BUF_STORAGE,    // C: pl_buf*    GLSL: storage buffer
                            // (`pl_buf->params.storable` must be set)
    PL_DESC_BUF_TEXEL_UNIFORM,// C: pl_buf*  GLSL: uniform samplerBuffer
                              // (`pl_buf->params.uniform` and `format` must be set)
    PL_DESC_BUF_TEXEL_STORAGE,// C: pl_buf*  GLSL: uniform imageBuffer
                              // (`pl_buf->params.uniform` and `format` must be set)
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
    PL_DESC_ACCESS_COUNT,
};

// Returns the GLSL syntax for a given access mode (e.g. "readonly").
const char *pl_desc_access_glsl_name(enum pl_desc_access mode);

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
};

// Framebuffer blending mode (for raster passes)
enum pl_blend_mode {
    PL_BLEND_ZERO,
    PL_BLEND_ONE,
    PL_BLEND_SRC_ALPHA,
    PL_BLEND_ONE_MINUS_SRC_ALPHA,
    PL_BLEND_MODE_COUNT,
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
    PL_PRIM_TYPE_COUNT,
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

    // The target texture this render pass is intended to be used with. This
    // doesn't have to come from a real texture - the caller can also invent
    // values or pass a blank struct, as long as `target_dummy.params.format`
    // is set. The format must support `PL_FMT_CAP_RENDERABLE`. If any other
    // fields are set, the GPU may be able to further optimize the render pass
    // for this particular type of texture.
    struct pl_tex target_dummy;

    // Target blending mode. If this is NULL, blending is disabled. Otherwise,
    // the `target_dummy.params.format` must have PL_FMT_CAP_BLENDABLE.
    const struct pl_blend_params *blend_params;

    // If false, the target's existing contents will be discarded before the
    // pass is run. (Semantically equivalent to calling pl_tex_invalidate
    // before every pl_pass_run, but slightly more efficient)
    //
    // Specifying `blend_params` requires `load_target` to be true.
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

    // An optional timer to report the approximate runtime of this shader pass
    // invocation to. Note that this is only an approximation, since shaders
    // may overlap their execution times and contend for GPU time.
    struct pl_timer *timer;

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

    // Number of vertices to render
    int vertex_count;

    // Vertex data may be provided in one of two forms:
    //
    // 1. Drawing from host memory directly
    void *vertex_data;
    // 2. Drawing from a vertex buffer (requires `vertex_buf->params.drawable`)
    const struct pl_buf *vertex_buf;
    size_t buf_offset;

    // --- pass->params.type==PL_PASS_COMPUTE only

    // Number of work groups to dispatch per dimension (X/Y/Z). Must be <= the
    // corresponding index of limits.max_dispatch
    int compute_groups[3];
};

// Execute a render pass.
void pl_pass_run(const struct pl_gpu *gpu, const struct pl_pass_run_params *params);

// A generic synchronization object intended for use with an external API. This
// is not required when solely using libplacebo API functions, as all required
// synchronisation is done internally. This comes in the form of a pair of
// semaphores - one to synchronize access in each direction.
struct pl_sync {
    enum pl_handle_type handle_type;

    // This handle is signalled by the `pl_gpu`, and waited on by the user. It
    // fires when it is safe for the user to access the shared resource.
    union pl_handle wait_handle;

    // This handle is signalled by the user, and waited on by the `pl_gpu`. It
    // must fire when the user has finished accessing the shared resource.
    union pl_handle signal_handle;
};

// Create a synchronization object. Returns NULL on failure.
//
// `handle_type` must be exactly *one* of `pl_gpu.export_caps.sync`, and
// indicates which type of handle to generate for sharing this sync object.
const struct pl_sync *pl_sync_create(const struct pl_gpu *gpu,
                                     enum pl_handle_type handle_type);

// Destroy a `pl_sync`. Note that this invalidates the externally imported
// semaphores. Users should therefore make sure that all operations that
// wait on or signal any of the semaphore have been fully submitted and
// processed by the external API before destroying the `pl_sync`.
//
// Despite this, it's safe to destroy a `pl_sync` if the only pending
// operations that involve it are internal to libplacebo.
void pl_sync_destroy(const struct pl_gpu *gpu,
                     const struct pl_sync **sync);

// Initiates a texture export operation, allowing a texture to be accessed by
// an external API. Returns whether successful. After this operation
// successfully returns, it is guaranteed that `sync->wait_handle` will
// eventually be signalled. For APIs where this is relevant, the image layout
// should be specified as "general", e.g. `GL_LAYOUT_GENERAL_EXT` for OpenGL.
//
// There is no corresponding "import" operation - the next operation that uses
// a texture will implicitly import the texture. Valid API usage requires that
// the user *must* submit a semaphore signal operation on `sync->signal_handle`
// before doing so. Not doing so is undefined behavior and may very well
// deadlock the calling process and/or the graphics card!
//
// Note that despite this restriction, it is always valid to call
// `pl_tex_destroy`, even if the texture is in an exported state, without
// having to signal the corresponding sync object first.
bool pl_tex_export(const struct pl_gpu *gpu, const struct pl_tex *tex,
                   const struct pl_sync *sync);

// A generic 'timer query' object. These can be used to measure an
// approximation of the GPU execution time of a given operation. Due to the
// highly asynchronous nature of GPUs, the actual results of any individual
// timer query may be delayed by quite a bit. As such, users should avoid
// trying to pair any particular GPU command with any particular timer query
// result, and only reuse `pl_timer` objects with identical operations. The
// results of timer queries are guaranteed to be in-order, but individual
// queries may be dropped, and some operations might not record timer results
// at all. (For example, if the underlying hardware does not support timer
// queries for a given operation type)
struct pl_timer;

// Creates a new timer object. This may return NULL, for example if the
// implementation does not support timers, but since passing NULL to
// `pl_timer_destroy` and `pl_timer_query` is safe, users generally need not
// concern themselves with handling this.
struct pl_timer *pl_timer_create(const struct pl_gpu *gpu);

void pl_timer_destroy(const struct pl_gpu *gpu, struct pl_timer **);

// Queries any results that have been measured since the last execution of
// `pl_timer_query`. There may be more than one result, in which case the user
// should simply call the function again to get the subsequent values. This
// function returns a value of 0 in the event that there are no more
// unprocessed results.
//
// The results are reported in nanoseconds, but the actual precision of the
// timestamp queries may be significantly lower.
//
// Note: Results do not queue up indefinitely. Generally, the implementation
// will only keep track of a small, fixed number of results internally. Make
// sure to include this function as part of your main rendering loop to process
// all of its results, or older results will be overwritten by newer ones.
uint64_t pl_timer_query(const struct pl_gpu *gpu, struct pl_timer *);

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
// lots of offline processing, i.e. you aren't rendering to a swapchain but to
// textures that you download from again. In that case you should call this
// function after each "work item" to ensure good parallelism between them.
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
// It's also guaranteed that any outstanding timer query results are available.
//
// Note: If you only care about buffer operations, you can accomplish this more
// easily by using `pl_buf_poll` with the timeout set to `UINT64_MAX`. But if
// you have many buffers it may be more convenient to call this function
// instead. The difference is that this function will also affect e.g. renders
// to a `pl_swapchain`.
void pl_gpu_finish(const struct pl_gpu *gpu);

// Returns true if the GPU is considered to be in a "failed" state, which
// during normal operation is typically the result of things like the device
// being lost (due to e.g. power management).
//
// If this returns true, users *should* destroy and recreate the `pl_gpu`,
// including all associated resources, via the appropriate mechanism.
bool pl_gpu_is_failed(const struct pl_gpu *gpu);

#endif // LIBPLACEBO_GPU_H_
