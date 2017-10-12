# libplacebo

**libplacebo** is essentially the core rendering engine of
[mpv](https://mpv.io) turned into a library. This grew out of an interest to
accomplish the following goals:

- Clean up mpv's internal [RA](#rendering-abstraction) API and make it reusable for other projects.
- Provide a standard library of useful GPU-accelerated image processing
  primitives based on GLSL, so projects like VLC or Firefox can use them
  without incurring a heavy dependency on `libmpv`.
- Rewrite core parts of mpv's GPU-accelerated video renderer on top of
  redesigned abstractions. (Basically, I wanted to eliminate code smell like
  `shader_cache.c` and totally redesign `gpu/video.c`)

**NOTE**: libplacebo is currently in a very early stage. Expect the API to be
extremely unstable, and many parts to be missing. The API version as exported
by `common.h` will **NOT** change until I declare the API stable, which will
coincide with the first release. As such, libplacebo should currently only
be used for testing purposes. It is not a finished product by any means.

## Authors

libplacebo's main developer is Niklas Haas
([@haasn](https://github.com/haasn)), but the project would not be possible
without the immense contributions of Vincent Lang
([@wm4](https://github.com/wm4)), who laid the groundwork for most of the code
that ended up in libplacebo.

For a full list of past contributors to mpv, see the [mpv authorship
page](https://github.com/mpv-player/mpv/graphs/contributors).

### License

Since the code heavily derives from LGPLv2.1+-licensed parts of mpv, there's
little choice but to license libplacebo the same way.

## API Overview

The public API of libplacebo is currently split up into the following
components, the header files (and documentation) for which are available
inside the [`src/public/`](src/public/) directory.

- `context.h`: The main entry-point into the library. Controls memory
  allocation, logging. and guards ABI/thread safety.
- `colorspace.h`: A collection of enums and structs for describing color
  spaces, as well as a collection of helper functions for computing various
  color space transformation matrices.
- `common.h`: A collection of miscellaneous utility types and macros that are
  shared among multiple subsystems. Usually does not need to be included
  directly.
- `config.h`: Macros defining information about the way libplacebo was built,
  including the version strings and compiled-in features/dependencies. Usually
  does not need to be included directly. May be useful for feature tests.
- `filters.h`: A collection of reusable reconstruction filter kernels, which
  can be used for scaling. The generated weights arrays are semi-tailored to
  the needs of libplacebo, but may be useful to somebody else regardless. Also
  contains the structs needed to define a filter kernel for the purposes of
  libplacebo's upscaling routines.
- `ra.h`: Exports the RA API used by libplacebo internally. For more
  information, see the [rendering abstraction](#rendering-abstraction)
  section.
- `shaders.h`: A collection of reusable GLSL primitives for various individual
  tasks including color space transformations and (eventually) image sampling,
  debanding, etc. These have an optional dependency on RA (ra.h), but can also
  be used independently (with more restrictions).
- `vulkan.h`: The main interface to the vulkan-based libplacebo code. This
  API essentially lets you create a vulkan-based RA instance.

### Rendering Abstraction

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
only the vulkan-based interface is exported. It's also not very tested/stable.

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
    struct pl_context *ctx = pl_context_create(PL_API_VER);

    // do something..

    pl_context_destroy(&ctx);
}
```

For a full documentation of the API, refer to the above [API
Overview](#api-overview) as well as the [public header files](src/public/). I
will create more and expanded samples once the project has a bit more
functionality worth writing home about.

## Support

If you like what I am doing with libplacebo, and would like to help see this
project grow beyond its initial scope, feel free to
[support me on Patreon](https://www.patreon.com/haasn).
