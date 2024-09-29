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

#ifndef LIBPLACEBO_D3D11_H_
#define LIBPLACEBO_D3D11_H_

#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <libplacebo/gpu.h>
#include <libplacebo/swapchain.h>

PL_API_BEGIN

// Structure representing the actual D3D11 device and associated GPU instance
typedef const struct pl_d3d11_t {
    pl_gpu gpu;

    // The D3D11 device in use. The user is free to use this for their own
    // purposes, including taking a reference to the device (with AddRef) and
    // using it beyond the lifetime of the pl_d3d11 that created it (though if
    // this is done with debug enabled, it will confuse the leak checker.)
    ID3D11Device *device;

    // True if the device is using a software (WARP) adapter
    bool software;
} *pl_d3d11;

struct pl_d3d11_params {
    // The Direct3D 11 device to use. Optional, if NULL then libplacebo will
    // create its own ID3D11Device using the options below. If set, all the
    // options below will be ignored.
    ID3D11Device *device;

    // --- Adapter selection options

    // The adapter to use. This overrides adapter_luid.
    IDXGIAdapter *adapter;

    // The LUID of the adapter to use. If adapter and adapter_luid are unset,
    // the default adapter will be used instead.
    LUID adapter_luid;

    // Allow a software (WARP) adapter when selecting the adapter automatically.
    // Note that sometimes the default adapter will be a software adapter. This
    // is because, on Windows 8 and up, if there are no hardware adapters,
    // Windows will pretend the WARP adapter is the default hardware adapter.
    bool allow_software;

    // Always use a software adapter. This is mainly for testing purposes.
    bool force_software;

    // --- Device creation options

    // Enable the debug layer (D3D11_CREATE_DEVICE_DEBUG)
    // Also logs IDXGIInfoQueue messages
    bool debug;

    // Disables the use of compute shaders. Some devices/drivers perform better
    // without them. This may also help prevent image corruption in cases where
    // the driver is misbehaving. Some features may be disabled if this is set.
    bool no_compute;

    // Extra flags to pass to D3D11CreateDevice (D3D11_CREATE_DEVICE_FLAG).
    // libplacebo should be compatible with any flags passed here.
    UINT flags;

    // The minimum and maximum allowable feature levels for the created device.
    // libplacebo will attempt to create a device with the highest feature level
    // between min_feature_level and max_feature_level (inclusive.) If there are
    // no supported feature levels in this range, `pl_d3d11_create` will either
    // return NULL or fall back to the software adapter, depending on whether
    // `allow_software` is set.
    //
    // Normally there is no reason to set `max_feature_level` other than to test
    // if a program works at lower feature levels.
    //
    // Note that D3D_FEATURE_LEVEL_9_3 and below (known as 10level9) are highly
    // restrictive. These feature levels are supported on a best-effort basis.
    // They represent very old DirectX 9 compatible PC and laptop hardware
    // (2001-2007, GeForce FX, 6, 7, ATI R300-R500, GMA 950-X3000) and some
    // less-old mobile devices (Surface RT, Surface 2.) Basic video rendering
    // should work, but the full pl_gpu API will not be available and advanced
    // shaders will probably fail. The hardware is probably too slow for these
    // anyway.
    //
    // Known restrictions of 10level9 devices include:
    //   D3D_FEATURE_LEVEL_9_3 and below:
    //   - `pl_pass_run_params->index_buf` will not work (but `index_data` will)
    //   - Dimensions of 3D textures must be powers of two
    //   - Shaders cannot use gl_FragCoord
    //   - Shaders cannot use texelFetch
    //   D3D_FEATURE_LEVEL_9_2 and below:
    //   - Fragment shaders have no dynamic flow control and very strict limits
    //     on the number of constants, temporary registers and instructions.
    //     Whether a shader meets the requirements will depend on how it's
    //     compiled and optimized, but it's likely that only simple shaders will
    //     work.
    //   D3D_FEATURE_LEVEL_9_1:
    //   - No high-bit-depth formats with PL_FMT_CAP_RENDERABLE or
    //     PL_FMT_CAP_LINEAR
    //
    // If these restrictions are undesirable and you don't need to support
    // ancient hardware, set `min_feature_level` to D3D_FEATURE_LEVEL_10_0.
    int min_feature_level; // Defaults to D3D_FEATURE_LEVEL_9_1 if unset
    int max_feature_level; // Defaults to D3D_FEATURE_LEVEL_12_1 if unset

    // Allow up to N in-flight frames. Similar to swapchain_depth for Vulkan and
    // OpenGL, though with DXGI this is a device-wide setting that affects all
    // swapchains (except for waitable swapchains.) See the documentation for
    // `pl_swapchain_latency` for more information.
    int max_frame_latency;
};

// Default/recommended parameters. Should generally be safe and efficient.
#define PL_D3D11_DEFAULTS   \
    .allow_software = true,

#define pl_d3d11_params(...) (&(struct pl_d3d11_params) { PL_D3D11_DEFAULTS __VA_ARGS__ })
PL_API extern const struct pl_d3d11_params pl_d3d11_default_params;

// Creates a new Direct3D 11 device based on the given parameters, or wraps an
// existing device, and initializes a new GPU instance. If params is left as
// NULL, it defaults to &pl_d3d11_default_params. If an existing device is
// provided in params->device, `pl_d3d11_create` will take a reference to it
// that will be released in `pl_d3d11_destroy`.
PL_API pl_d3d11 pl_d3d11_create(pl_log log, const struct pl_d3d11_params *params);

// Release the D3D11 device.
//
// Note that all libplacebo objects allocated from this pl_d3d11 object (e.g.
// via `d3d11->gpu` or using `pl_d3d11_create_swapchain`) *must* be explicitly
// destroyed by the user before calling this.
PL_API void pl_d3d11_destroy(pl_d3d11 *d3d11);

// For a `pl_gpu` backed by `pl_d3d11`, this function can be used to retrieve
// the underlying `pl_d3d11`. Returns NULL for any other type of `gpu`.
PL_API pl_d3d11 pl_d3d11_get(pl_gpu gpu);

struct pl_d3d11_swapchain_params {
    // The Direct3D 11 swapchain to wrap. Optional. If NULL, libplacebo will
    // create its own swapchain using the options below. If set, all the
    // swapchain creation options will be ignored.
    //
    // The provided swapchain must have been created by the same device used
    // by `gpu` and must not have multisampled backbuffers.
    IDXGISwapChain *swapchain;

    // --- Swapchain creation options

    // Initial framebuffer width and height. If both width and height are set to
    // 0 and window is non-NULL, the client area of the window is used instead.
    // For convenience, if either component would be 0, it is set to 1 instead.
    // This is because Windows can have 0-sized windows, but not 0-sized
    // swapchains.
    int width;
    int height;

    // The handle of the output window. In Windows 8 and up this is optional
    // because you can output to a CoreWindow or create a composition swapchain
    // instead.
    HWND window;

    // A pointer to the CoreWindow to output to. If both this and `window` are
    // NULL, CreateSwapChainForComposition will be used to create the swapchain.
    IUnknown *core_window;

    // If set, libplacebo will create a swapchain that uses the legacy bitblt
    // presentation model (with the DXGI_SWAP_EFFECT_DISCARD swap effect.) This
    // tends to give worse performance and frame pacing in windowed mode and it
    // prevents borderless fullscreen optimizations, but it might be necessary
    // to work around buggy drivers, especially with DXGI 1.2 in the Platform
    // Update for Windows 7. When unset, libplacebo will try to use the flip
    // presentation model and only fall back to bitblt if flip is unavailable.
    bool blit;

    // additional swapchain flags
    // No validation on these flags is being performed, and swapchain creation
    // may fail if an unsupported combination is requested.
    UINT flags;

    // --- Swapchain usage behavior options

    // Disable using a 10-bit swapchain format for SDR output
    bool disable_10bit_sdr;
};

#define pl_d3d11_swapchain_params(...) (&(struct pl_d3d11_swapchain_params) { __VA_ARGS__ })

// Creates a new Direct3D 11 swapchain, or wraps an existing one. If an existing
// swapchain is provided in params->swapchain, `pl_d3d11_create_swapchain` will
// take a reference to it that will be released in `pl_swapchain_destroy`.
PL_API pl_swapchain pl_d3d11_create_swapchain(pl_d3d11 d3d11,
    const struct pl_d3d11_swapchain_params *params);

// Takes a `pl_swapchain` created by pl_d3d11_create_swapchain and returns a
// reference to the underlying IDXGISwapChain. This increments the refcount, so
// call IDXGISwapChain::Release when finished with it.
PL_API IDXGISwapChain *pl_d3d11_swapchain_unwrap(pl_swapchain sw);

struct pl_d3d11_wrap_params {
    // The D3D11 texture to wrap, or a texture array containing the texture to
    // wrap. Must be a ID3D11Texture1D, ID3D11Texture2D or ID3D11Texture3D
    // created by the same device used by `gpu`, must have D3D11_USAGE_DEFAULT,
    // and must not be mipmapped or multisampled.
    ID3D11Resource *tex;

    // If tex is a texture array, this is the array member to use as the pl_tex.
    int array_slice;

    // If tex is a video resource (eg. DXGI_FORMAT_AYUV, DXGI_FORMAT_NV12,
    // DXGI_FORMAT_P010, etc.,) it can be wrapped as a pl_tex by specifying the
    // type and size of the shader view. For planar video formats, the plane
    // that is wrapped depends on the chosen format.
    //
    // If tex is not a video resource, these fields are unnecessary. The correct
    // format will be determined automatically. If tex is not 2D, these fields
    // are ignored.
    //
    // For a list of supported video formats and their corresponding view
    // formats and sizes, see:
    // https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm#VideoViews
    DXGI_FORMAT fmt;
    int w;
    int h;
};

#define pl_d3d11_wrap_params(...) (&(struct pl_d3d11_wrap_params) { __VA_ARGS__ })

// Wraps an external texture into a pl_tex abstraction. `pl_d3d11_wrap` takes a
// reference to the texture, which is released when `pl_tex_destroy` is called.
//
// This function may fail due to incompatible formats, incompatible flags or
// other reasons, in which case it will return NULL.
PL_API pl_tex pl_d3d11_wrap(pl_gpu gpu, const struct pl_d3d11_wrap_params *params);

PL_API_END

#endif // LIBPLACEBO_D3D11_H_
