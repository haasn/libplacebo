# libplacebo

![travis-ci badge](https://travis-ci.org/haasn/libplacebo.svg?branch=master)

**libplacebo** is essentially the core rendering algorithms and ideas of
[mpv](https://mpv.io) turned into a library. This grew out of an interest to
accomplish the following goals:

- Clean up mpv's internal [RA](#tier-1-rendering-abstraction) API and make it
  reusable for other projects.
- Provide a standard library of useful GPU-accelerated image processing
  primitives based on GLSL, so projects like VLC or Firefox can use them
  without incurring a heavy dependency on `libmpv`.
- Rewrite core parts of mpv's GPU-accelerated video renderer on top of
  redesigned abstractions. (Basically, I wanted to eliminate code smell like
  `shader_cache.c` and totally redesign `gpu/video.c`)

**NOTE**: libplacebo is currently in an early stage. Expect the API to be
very unstable, and many parts to be missing. To reflect this, the API version
as exported by common.h will **NOT** change except on new beta releases
(v0.x). So using libplacebo directly from git master is not advised; always
use one of the tagged releases if you want to play around with libplacebo in
its current stage.

Once the version number hits 1.X, which will mark the first stable release,
the API version will be bumped for every change to the public API - even
changes that happen on git master.

## Authors

libplacebo's main developer is Niklas Haas
([@haasn](https://github.com/haasn)), but the project would not be possible
without the development of mpv, which was done primarily by Vincent Lang
([@wm4](https://github.com/wm4)).

For a full list of past contributors to mpv, see the [mpv authorship
page](https://github.com/mpv-player/mpv/graphs/contributors).

### License

Since the code derives from several LGPLv2.1+-licensed parts of mpv, there's
little choice but to license libplacebo the same way. It's worth pointing out
that, except for some minor exceptions (e.g. filters.c and colorspace.c), most
of the code is either original work or can be attributed to only a small
number of developers, so a relicensing to a more permissive license might be
possible in principle.

## API Overview

The public API of libplacebo is currently split up into the following
components, the header files (and documentation) for which are available
inside the [`src/include/libplacebo`](src/include/libplacebo) directory. The
API is available in different "tiers", representing levels of abstraction
inside libplacebo. The APIs in higher tiers depend on those in lower tiers.
Which tier is used by a user depends on how much power/control they want over
the actual rendering. The low-level tiers are more suitable for big projects
that need strong control over the entire rendering pipeline; whereas the
high-level tiers are more suitable for smaller or simpler projects that want
libplacebo to take care of everything.

### Tier 0 (context, raw math primitives)

- `colorspace.h`: A collection of enums and structs for describing color
  spaces, as well as a collection of helper functions for computing various
  color space transformation matrices.
- `common.h`: A collection of miscellaneous utility types and macros that are
  shared among multiple subsystems. Usually does not need to be included
  directly.
- `context.h`: The main entry-point into the library. Controls memory
  allocation, logging. and guards ABI/thread safety.
- `config.h`: Macros defining information about the way libplacebo was built,
  including the version strings and compiled-in features/dependencies. Usually
  does not need to be included directly. May be useful for feature tests.
- `dither.h`: Some helper functions for generating various noise and dithering
  matrices. Might be useful for somebody else.
- `filters.h`: A collection of reusable reconstruction filter kernels, which
  can be used for scaling. The generated weights arrays are semi-tailored to
  the needs of libplacebo, but may be useful to somebody else regardless. Also
  contains the structs needed to define a filter kernel for the purposes of
  libplacebo's upscaling routines.

The API functions in this tier are either used throughout the program
(context, common etc.) or are low-level implementations of filter kernels,
color space conversion logic etc.; which are entirely independent of GLSL
and even the GPU in general.

### Tier 1 (rendering abstraction)

- `ra.h`: Exports the RA API used by libplacebo internally.
- `vulkan.h`: RA implementation based on Vulkan.

As part of the public API, libplacebo exports the **RA** API ("Rendering
Abstraction"). Basically, this is the API libplacebo uses internally to wrap
OpenGL, Vulkan, Direct3D etc. into a single unifying API subset that abstracts
away state, messy details, synchronization etc. into a fairly high-level API
suitable for libplacebo's image processing tasks.

It's made public both because it constitutes part of the public API of various
image processing functions, but also in the hopes that it will be useful for
other developers of GPU-accelerated image processing software. RA can be used
entirely independently of libplacebo's image processing, which is why it
uses its own namespace (`ra_` instead of `pl_`).

**NOTE**: The port of RA into libplacebo is currently very WIP, and right now
only the vulkan-based interface is exported.

### Tier 2 (GLSL generating primitives)

- `shaders.h`: The low-level interface to shader generation. This can be used
  to generate GLSL stubs suitable for inclusion in other programs, as part of
  larger shaders. For example, a program might use this interface to generate
  a specialized tone-mapping function for performing color space conversions,
  then call that from their own fragment shader code. This abstraction has an
  optional dependency on `RA`, but can also be used independently from it.

In addition to this low-level interface, there are several available shader
routines which libplacebo exports:

- `shaders/colorspace.h`: Shader routines for decoding and transforming
  colors, tone mapping, dithering, and so forth.
- `shaders/sampling.h`: Shader routines for various algorithms that sample
  from images, such as debanding and scaling.

### Tier 3 (shader dispatch)

- `dispatch.h`: A higher-level interface to the `pl_shader` system, based on
  RA. This dispatch mechanism generates+executes complete GLSL shaders,
  subject to the constraints and limitations of the underlying RA.

This shader dispatch mechanism is designed to be combined with the shader
processing routines exported by `shaders/*.h`, but takes care of the low-level
translation of the resulting `pl_shader_res` objects into legal GLSL. It also
takes care of resource binding, shader input placement, as well as shader
caching and resource pooling; and makes sure all generated shaders have unique
identifiers (so they can be freely merged together).

### Tier 4 (high level renderer)

- `renderer.h`: A high-level renderer which combines the shader primitives
  and dispatch mechanism into a fully-fledged rendering pipeline that takes
  raw texture data and transforms it into the desired output image.

This is the "primary" interface to libplacebo, and the one most users will be
interested in. It takes care of internal details such as degrading to simpler
algorithms depending on the hardware's capabilities, combining the correct
sequence of colorspace transformations and shader passes in order to get the
best overall image quality, and so forth.

**WARNING**: The `renderer` is a WIP component and currently doesn't really do
much other than forced bilinear scaling. If you trigger any unimplemented
paths, it will abort().

## Installing

### Gentoo

An [ebuild](etc/libplacebo-9999.ebuild) is available.

### Building from source

libplacebo is built using the [meson build system](http://mesonbuild.com/).
You can build the project using the following steps:

```bash
$ DIR=./build
$ meson $DIR
$ ninja -C$DIR
```

To rebuild the project on changes, re-run `ninja -Cbuild`. If you wish to
install the build products to the configured prefix (typically `/usr/local/`),
you can run `ninja -Cbuild install`. Note that this is normally ill- advised
except for developers who know what they're doing. Regular users should rely
on distro packages.

### Configuring

To get a list of configuration options supported by libplacebo, after running
`meson $DIR` you can run `meson configure $DIR`, e.g.:

```bash
$ meson $DIR
$ meson configure $DIR
```

If you want to disable a component, for example Vulkan support, you can
explicitly set it to `false`, i.e.:

```bash
$ meson configure $DIR -Dvulkan=false -Dshaderc=false
$ ninja -C$DIR
```

### Testing

To enable building and executing the tests, you need to build with
`tests` enabled, i.e.:

```bash
$ meson configure $DIR -Dtests=true
$ ninja -C$DIR test
```

## Using

Building a trivial project using libplacebo is straightforward:

```c
// build with -lplacebo

#include <libplacebo/context.h>

void main()
{
    struct pl_context *ctx;
    ctx = pl_context_create(PL_API_VER, &(struct pl_context_params) {
        .log_cb    = pl_log_color,
        .log_level = PL_LOG_INFO,
    });

    // do something..

    pl_context_destroy(&ctx);
}
```

For a full documentation of the API, refer to the above [API
Overview](#api-overview) as well as the [public header
files](src/include/libplacebo). You can find additional examples of how to use
the various components, in the [unit tests](src/tests).

I will create more and expanded samples once the project has a bit more
functionality worth writing home about.

## Support

If you like what I am doing with libplacebo, and would like to help see this
project grow beyond its initial scope, feel free to
[support me on Patreon](https://www.patreon.com/haasn).
