# Custom Shaders (mpv .hook syntax)

libplacebo supports the same [custom shader syntax used by
mpv](https://mpv.io/manual/master/#options-glsl-shader), with some important
changes. This document will serve as a complete reference for this syntax.

## Overview

In general, user shaders are divided into distinct *blocks*. Each block can
define a shader, a texture, a buffer, or a tunable parameter. Each block
starts with a collection of header directives, which are lines starting with
the syntax `//!`.

As an example, here is a simple shader that simply inverts the video signal:

``` glsl linenums="1"
//!HOOK LUMA
//!HOOK RGB
//!BIND HOOKED

vec4 hook()
{
    vec4 color = HOOKED_texOff(0);
    color.rgb = vec3(1.0) - color.rgb;
    return color;
}
```

This shader defines one block - a shader block which hooks into the two
texture stages `LUMA` and `RGB`, binds the hooked texture, inverts the value
of the `rgb` channels, and then returns the modified color.

### Expressions

In a few contexts, shader directives accept arithmetic expressions, denoted by
`<expr>` in the listing below. For historical reasons, all expressions are
given in [reverse polish notation
(RPN)](https://en.wikipedia.org/wiki/Reverse_Polish_notation), and the only
value type is a floating point number. The following value types and
arithmetic operations are available:

* `1.234`: Literal float constant, evaluates to itself.
* `NAME.w`, `NAME.width`: Evaluates to the width of a texture with name `NAME`.
* `NAME.h`, `NAME.height`: Evaluates to the height of a texture with name `NAME`.
* `PAR`: Evaluates to the value of a tunable shader parameter with name `PAR`.
* `+`: Evaluates to `X+Y`.
* `-`: Evaluates to `X-Y`.
* `*`: Evaluates to `X*Y`.
* `/`: Evaluates to `X/Y`.
* `%`: Evaluates to `fmod(X, Y)`.
* `>`: Evaluates to `(X > Y) ? 1.0 : 0.0`.
* `<`: Evaluates to `(X < Y) ? 1.0 : 0.0`.
* `=`: Evaluates to `fuzzy_eq(X, Y) ? 1.0 : 0.0`, with some tolerance to
  allow for floating point inaccuracy. (Around 1 ppm)
* `!`: Evaluates to `X ? 0.0 : 1.0`.

Note that `+` and `*` can be used as suitable replacements for the otherwise
absent boolean logic expressions (`||` and `&&`).

## Shaders

Shaders are the default block type, and have no special syntax to indicate
their presence. Shader stages contain raw GLSL code that will be
(conditionally) executed. This GLSL snippet must define a single function
`vec4 hook()`, or `void hook()` for compute shaders.

During the execution of any shader, the following global variables are made
available:

* `int frame`: A raw counter tracking the number of executions of this shader
  stage.
* `float random`: A pseudo-random float uniformly distributed in the range
  `[0,1)`.
* `vec2 input_size`: The nominal size (in pixels) of the original input image.
* `vec2 target_size`: The nominal size (in pixels) of the output rectangle.
* `vec2 tex_offset`: The nominal offset (in pixels), of the original input crop.
* `vec4 linearize(vec4 color)`: Linearize the input color according to the
  image's tagged gamma function.
* `vec4 delinearize(vec4 color)`: Opposite counterpart to `linearize`.

Shader stages accept the following directives:

### `HOOK <texture>`

A `HOOK` directive determines when a shader stage is run. During internal
processing, libplacebo goes over a number of pre-defined *hook points* at set
points in the processing pipeline. It is only possible to intercept the image,
and run custom shaders, at these fixed hook points.

Here is a current list of hook points:

* `RGB`: Input plane containing RGB values
* `LUMA`: Input plane containing a Y value
* `CHROMA`: Input plane containing chroma values (one or both)
* `ALPHA`: Input plane containing a single alpha value
* `XYZ`: Input plane containing XYZ values
* `CHROMA_SCALED`: Chroma plane, after merging and upscaling to luma size
* `ALPHA_SCALED`: Alpha plane, after upscaling to luma size
* `NATIVE`: Merged input planes, before any sort of color conversion (as-is)
* `MAIN`: After conversion to RGB, before linearization/scaling
* `LINEAR`: After conversion to linear light (for scaling purposes)
* `SIGMOID`: After conversion to sigmoidized light (for scaling purposes)
* `PREKERNEL`: Immediately before the execution of the main scaler kernel
* `POSTKERNEL`: Immediately after the execution of the main scaler kernel
* `SCALED`: After scaling, in either linear or non-linear light RGB
* `OUTPUT`: After color conversion to the output display's native colorspace

!!! warning "`MAINPRESUB`"
    In mpv, `MAIN` and `MAINPRESUB` are separate shader stages, because the
    mpv option `--blend-subtitles=video` allows rendering overlays directly
    onto the pre-scaled video stage. libplacebo does not support this feature,
    and as such, the `MAINPRESUB` shader stage does not exist. It is still
    valid to refer to this name in shaders, but it is handled identically to
    `MAIN`.

It's possible for a hook point to never fire. For example, `SIGMOID` will not
fire when downscaling, as sigmoidization only happens when upscaling.
Similarly, `LUMA`/`CHROMA` will not fire on an RGB video and vice versa.

A single shader stage may hook multiple hook points simultaneously, for
example, to cover both `LUMA` and `RGB` cases with the same logic. (See the
example shader in the introduction)

### `BIND <texture>`

The `BIND` directive makes a texture available for use in the shader. This can
be any of the previously named hook points, a custom texture define by a
`TEXTURE` block, a custom texture saved by a `SAVE` directive, or the special
value `HOOKED` which allows binding whatever texture hook dispatched this
shader stage.

A bound texture will define the following GLSL functions (as macros):

* `sampler2D NAME_raw`: A reference to the raw texture sampler itself.
* `vec2 NAME_pos`: The texel coordinates of the current pixel.
* `vec2 NAME_map(ivec2 id)`: A function that maps from `gl_GlobalInvocationID`
  to texel coordinates. (Compute shaders)
* `vec2 NAME_size`: The size (in pixels) of the texture.
* `vec2 NAME_pt`: Convenience macro for `1.0 / NAME_size`. The size of a
  single pixel (in texel coordinates).
* `vec2 NAME_off`: The sample offset of the texture. Basically, the pixel
  coordinates of the top-left corner of the sampled area.
* `float NAME_mul`: The coefficient that must be multiplied into sampled
  values in order to rescale them to `[0,1]`.
* `vec4 NAME_tex(vec2 pos)`: A wrapper around `NAME_mul * texture(NAME_raw,
  pos)`, which picks the correct `texture` function for the version of GLSL in
  use.
* `vec4 NAME_texOff(vec2 offset)`: A wrapper around `NAME_tex(NAME_pos + NAME_pt * offset)`.
  This can be used to easily access adjacent pixels, e.g. `NAME_texOff(-1,2)`
  samples a pixel one to the left and two to the bottom of the current
  location.
* `vec4 NAME_gather(vec2 pos, int c)`: A wrapper around
  `NAME_mul * textureGather(pos, c)`, with appropriate scaling. (Only when
  supported[^ifdef])

!!! note "Rotation matrix"
    For compatibility with mpv, we also define a `mat2 NAME_rot` which is
    simply equal to a 2x2 identity matrix. libplacebo never rotates input
    planes - all rotation happens during the final output to the display.

[^ifdef]: Because these are macros, their presence can be tested for using
  `#ifdef` inside the GLSL preprocessor.

### `SAVE <texture>`

By default, after execution of a shader stage, the resulting output is
captured back into the same hooked texture that triggered the shader. This
behavior can be overridden using the explicit `SAVE` directive. For example,
a shader might need access to a low-res version of the luma input texture in
order to process chroma:

``` glsl linenums="1"
//!HOOK CHROMA
//!BIND CHROMA
//!BIND LUMA
//!SAVE LUMA_LOWRES
//!WIDTH CHROMA.w
//!HEIGHT CHROMA.h

vec4 hook()
{
    return LUMA_texOff(0);
}
```

This shader binds both luma and chroma and resizes the luma plane down to the
size of the chroma plane, saving the result as a new texture `LUMA_LOWRES`. In
general, you can pick any name you want, here.

### `DESC <description>`

This purely informative directive simply gives the shader stage a name. This
is the name that will be reported to the shader stage and execution time
metrics.

### `OFFSET <xo yo | ALIGN>`

This directive indicates a pixel shift (offset) introduced by this pass. These
pixel offsets will be accumulated and corrected automatically as part of plane
alignment / main scaling.

A special value of `ALIGN` will attempt to counteract any existing offset of
the hooked texture by aligning it with reference plane (i.e. luma). This can
be used to e.g. introduce custom chroma scaling in a way that doesn't break
chroma subtexel offsets.

An example:

``` glsl linenums="1"
//!HOOK LUMA
//!BIND HOOKED
//!OFFSET 100.5 100.5

vec4 hook()
{
    // Constant offset by N pixels towards the bottom right
    return HOOKED_texOff(-vec2(100.5));
}
```

This (slightly silly) shader simply shifts the entire sampled region to the
bottom right by 100.5 pixels, and propagates this shift to the main scaler
using the `OFFSET` directive. As such, the end result of this is that there is
no visible shift of the overall image, but some detail (~100 pixels) near the
bottom-right border is lost due to falling outside the bounds of the texture.

### `WIDTH <expr>`, `HEIGHT <expr>`

These directives can be used to override the dimensions of the resulting
texture. Note that not all textures can be resized this way. Currently, only
`RGB`, `LUMA`, `CHROMA`, `XYZ`, `NATIVE` and `MAIN` are resizable. Trying to
save a texture with an incompatible size to any other shader stage will result
in an error.

### `WHEN <expr>`

This directive takes an expression that can be used to make shader stages
conditionally executed. If this evaluates to 0, the shader stage will be
skipped.

Example:

``` glsl linenums="1"
//!PARAM strength
//!TYPE float
//!MINIMUM 0
1.0

//!HOOK MAIN
//!BIND HOOKED
//!WHEN intensity 0 >
//!DESC do something based on 'intensity'
...
```

This example defines a shader stage that only conditionally executes itself
if the value of the `intensity` shader parameter is non-zero.

### `COMPONENTS <num>`

This directive overrides the number of components present in a texture.
For example, if you want to extract a one-dimensional feature map from the
otherwise 3 or 4 dimensional `MAIN` texture, you can use this directive to
save on memory bandwidth and consumption by having libplacebo only allocate a
one-component texture to store the feature map in:

``` glsl linenums="1"
//!HOOK MAIN
//!BIND HOOKED
//!SAVE featuremap
//!COMPONENTS 1
```

### `COMPUTE <bw> <bh> [<tw> <th>]`

This directive specifies that the shader should be treated as a compute
shader, with the block size `bw` and `bh`. The compute shader will be
dispatched with however many blocks are necessary to completely tile over the
output. Within each block, there will be `tw*th` threads, forming a single
work group. In other words: `tw` and `th` specify the work group size, which
can be different from the block size. So for example, a compute shader with
`bw = bh = 32` and `tw = th = 8` running on a `500x500` texture would dispatch
`16x16` blocks (rounded up), each with `8x8` threads.

Instead of defining a `vec4 hook()`, compute shaders must define a `void
hook()` which results directly to the output texture, a `writeonly image2D
out_image` made available to the shader stage.

For example, here is a shader executing a single-pass 41x41 convolution
(average blur) on the luma plane, using a compute shader to share sampling
work between adjacent threads in a work group:

``` glsl linenums="1"
//!HOOK LUMA
//!BIND HOOKED
//!COMPUTE 32 32
//!DESC avg convolution

// Kernel size, 41x41 as an example
const ivec2 ksize = ivec2(41, 41);
const ivec2 offset = ksize / 2;

// We need to load extra source texels to account for padding due to kernel
// overhang
const ivec2 isize = ivec2(gl_WorkGroupSize) + ksize - 1;

shared float inp[isize.y][isize.x];

void hook()
{
    // load texels into shmem
    ivec2 base = ivec2(gl_WorkGroupID) * ivec2(gl_WorkGroupSize);
    for (uint y = gl_LocalInvocationID.y; y < isize.y; y += gl_WorkGroupSize.y) {
        for (uint x = gl_LocalInvocationID.x; x < isize.x; x += gl_WorkGroupSize.x)
            inp[y][x] = texelFetch(HOOKED_raw, base + ivec2(x,y) - offset, 0).x;
    }

    // synchronize threads
    barrier();

    // do convolution
    float sum;
    for (uint y = 0; y < ksize.y; y++) {
        for (uint x = 0; x < ksize.x; x++)
            sum += inp[gl_LocalInvocationID.y+y][gl_LocalInvocationID.x+x];
    }

    vec4 color = vec4(HOOKED_mul * sum / (ksize.x * ksize.y), 0, 0, 1);
    imageStore(out_image, ivec2(gl_GlobalInvocationID), color);
}
```

## Textures

Custom textures can be defined and made available to shader stages using
`TEXTURE` blocks. These can be used to provide e.g. LUTs or pre-trained
weights.

The data for a texture is provided as a raw hexadecimal string encoding the
in-memory representation of a texture, according to its given texture format,
for example:

``` glsl linenums="1"
//!TEXTURE COLORS
//!SIZE 3 3
//!FORMAT rgba32f
//!FILTER NEAREST
//!BORDER REPEAT
0000803f000000000000000000000000000000000000803f00000000000000000000000
0000000000000803f00000000000000000000803f0000803f000000000000803f000000
000000803f000000000000803f0000803f00000000000000009a99993e9a99993e9a999
93e000000009a99193F9A99193f9a99193f000000000000803f0000803f0000803f0000
0000
```

Texture blocks accept the following directives:

### `TEXTURE <name>`

This must be the first directive in a texture block, and marks it as such. The
name given is the name that the texture will be referred to (via `BIND`
directives).

### `SIZE <width> [<height> [<depth>]]`

This directive gives the size of the texture, as integers. For example,
`//!SIZE 512 512` marks a 512x512 texture block. Textures can be 1D, 2D or 3D
depending on the number of coordinates specified.

### `FORMAT <fmt>`

This directive specifies the texture format. A complete list of known textures
is exposed as part of the `pl_gpu` struct metadata, but they follow the format
convention `rgba8`, `rg16hf`, `rgba32f`, `r64i` and so on.

### `FILTER <LINEAR | NEAREST>`

This directive specifies the texture magnification/minification filter.

### `BORDER <CLAMP | REPEAT | MIRROR>`

This directive specifies the border clamping method of the texture.

### `STORAGE`

If present, this directive marks the texture as a storage image. It will still
be initialized with the initial values, but rather than being bound as a
read-only and immutable `sampler2D`, it is bound as a `readwrite coherent
image2D`. Such texture scan be used to, for example, store persistent state
across invocations of the shader.

## Buffers

Custom uniform / storage shader buffer  blocks can be defined using `BUFFER`
directives.

The (initial) data for a buffer is provided as a raw hexadecimal string
encoding the in-memory representation of a buffer in the corresponding GLSL
packing layout (std140 or std430 for uniform and storage blocks,
respectively):

``` glsl linenums="1"
//!BUFFER buf_uniform
//!VAR float foo
//!VAR float bar
0000000000000000

//!BUFFER buf_storage
//!VAR vec2 bat
//!VAR int big[32];
//!STORAGE
```

Buffer blocks accept the following directives:

### `BUFFER <name>`

This must be the first directive in a buffer block, and marks it as such. The
name given is mostly cosmetic, as individual variables can be accessed
directly using the names given in the corresponding `VAR` directives.

### `STORAGE`

If present, this directive marks the buffer as a (readwrite coherent) shader
storage block, instead of a readonly uniform buffer block. Such storage blocks
can be used to track and evolve state across invocations of this shader.

Storage blocks may also be initialized with default data, but this is
optional. They can also be initialized as part of the first shader execution
(e.g. by testing for `frame == 0`).

### `VAR <type> <name>`

This directive appends a new variable to the shader block, with GLSL type
`<type>` and shader name `<name>`. For example, `VAR float foo` introduces a
`float foo;` member into the buffer block, and `VAR mat4 transform` introduces
a `mat4 transform;` member.

It is also possible to introduce array variables, using `[N]` as part of the
variable name.

## Tunable parameters

Finally, the `PARAM` directive allows introducing tunable shader parameters,
which are exposed programmatically as part of the C API (`pl_hook`).[^mpv]

[^mpv]: In mpv using `--vo=gpu-next`, these can be set using the
  [`--glsl-shader-opts` option](https://mpv.io/manual/master/#options-glsl-shader-opts).

The default value of a parameter is given as the block body, for example:

``` glsl linenums="1"
//!PARAM contrast
//!DESC Gain to apply to image brightness
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 100.0
1.0
```

Parameters accept the following directives:

### `PARAM <name>`

This must be the first directive in a parameter block, and marks it as such.
The name given is the name that will be used to refer to this parameter in
GLSL code.

### `DESC <description>`

This directive can be used to provide a friendlier description of the shader
parameter, exposed as part of the C API to end users.

### `MINIMUM <value>`, `MAXIMUM <value>`

Provides the minimum/maximum value bound of this parameter. If absent, no
minimum/maximum is enforced.

### `TYPE <DEFINE | [DYNAMIC | CONSTANT] <type>>`

This gives the type of the parameter, which determines what type of values it
can hold and how it will be made available to the shader. `<type>` must be
a scalar GLSL numeric type, such as `int`, `float` or `uint`.

The optional qualifiers `DYNAMIC` or `CONSTANT` mark the parameter as
dynamically changing and compile-time constant, respectively. A `DYNAMIC`
variable is assumed to change frequently, and will be grouped with other
frequently-changing input parameters. A `CONSTANT` parameter will be
introduced as a compile-time constant into the shader header, which means thy
can be used in e.g. constant expressions such as array sizes.[^spec]

[^spec]: On supported platforms, these are implemented using specialization
  constants, which can be updated at run-time without requiring a full shader
  recompilation.

Finally, the special type `TYPE DEFINE` marks a variable as a preprocessor
define, which can be used inside `#if` preprocessor expressions. For example:

``` glsl linenums="1"
//!PARAM taps
//!DESC Smoothing taps
//!TYPE DEFINE
//!MINIMUM 0
//!MAXIMUM 5
2

//!HOOK LUMA
//!BIND HOOKED
const uint row_size = 2 * taps + 1;
const float weights[row_size] = {
#if taps == 0
    1.0,
#endif

#if taps == 1
    0.10650697891920,
    0.78698604216159,
    0.10650697891920,
#endif

#if taps == 2
    0.05448868454964,
    0.24420134200323,
    0.40261994689424,
    0.24420134200323,
    0.05448868454964,
#endif

    // ...
};
```

## Full example

A collection of full examples can be found in the [mpv user shaders
wiki](https://github.com/mpv-player/mpv/wiki/User-Scripts#user-shaders), but
here is an example of a parametrized Gaussian smoothed film grain compute
shader:

``` glsl linenums="1"
//!PARAM intensity
//!DESC Film grain intensity
//!TYPE float
//!MINIMUM 0
0.1

//!PARAM taps
//!DESC Film grain smoothing taps
//!TYPE DEFINE
//!MINIMUM 0
//!MAXIMUM 5
2

//!HOOK LUMA
//!BIND HOOKED
//!DESC Apply gaussian smoothed film grain
//!WHEN intensity 0 >
//!COMPUTE 32 32

const uint row_size = 2 * taps + 1;
const float weights[row_size] = {
#if taps == 0
    1.0,
#endif

#if taps == 1
    0.10650697891920,
    0.78698604216159,
    0.10650697891920,
#endif

#if taps == 2
    0.05448868454964,
    0.24420134200323,
    0.40261994689424,
    0.24420134200323,
    0.05448868454964,
#endif

#if taps == 3
    0.03663284536919,
    0.11128075847888,
    0.21674532140370,
    0.27068214949642,
    0.21674532140370,
    0.11128075847888,
    0.03663284536919,
#endif

#if taps == 4
    0.02763055063889,
    0.06628224528636,
    0.12383153680577,
    0.18017382291138,
    0.20416368871516,
    0.18017382291138,
    0.12383153680577,
    0.06628224528636,
    0.02763055063889,
#endif

#if taps == 5
    0.02219054849244,
    0.04558899978527,
    0.07981140824009,
    0.11906462996609,
    0.15136080967773,
    0.16396720767670,
    0.15136080967773,
    0.11906462996609,
    0.07981140824009,
    0.04558899978527,
    0.02219054849244,
#endif
};

const uvec2 isize = uvec2(gl_WorkGroupSize) + uvec2(2 * taps);
shared float grain[isize.y][isize.x];

// PRNG
float permute(float x)
{
    x = (34.0 * x + 1.0) * x;
    return fract(x * 1.0/289.0) * 289.0;
}

float seed(uvec2 pos)
{
    const float phi = 1.61803398874989;
    vec3 m = vec3(fract(phi * vec2(pos)), random) + vec3(1.0);
    return permute(permute(m.x) + m.y) + m.z;
}

float rand(inout float state)
{
    state = permute(state);
    return fract(state * 1.0/41.0);
}

// Turns uniform white noise into gaussian white noise by passing it
// through an approximation of the gaussian quantile function
float rand_gaussian(inout float state) {
    const float a0 = 0.151015505647689;
    const float a1 = -0.5303572634357367;
    const float a2 = 1.365020122861334;
    const float b0 = 0.132089632343748;
    const float b1 = -0.7607324991323768;

    float p = 0.95 * rand(state) + 0.025;
    float q = p - 0.5;
    float r = q * q;

    float g = q * (a2 + (a1 * r + a0) / (r*r + b1*r + b0));
    g *= 0.255121822830526; // normalize to [-1,1)
    return g;
}

void hook()
{
    // generate grain in `grain`
    uint num_threads = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
    for (uint i = gl_LocalInvocationIndex; i < isize.y * isize.x; i += num_threads) {
        uvec2 pos = uvec2(i % isize.y, i / isize.y);
        float state = seed(gl_WorkGroupID.xy * gl_WorkGroupSize.xy + pos);
        grain[pos.y][pos.x] = rand_gaussian(state);
    }

    // make writes visible
    barrier();

    // convolve horizontally
    for (uint y = gl_LocalInvocationID.y; y < isize.y; y += gl_WorkGroupSize.y) {
        float hsum = 0;
        for (uint x = 0; x < row_size; x++) {
            float g = grain[y][gl_LocalInvocationID.x + x];
            hsum += weights[x] * g;
        }

        // update grain LUT
        grain[y][gl_LocalInvocationID.x + taps] = hsum;
    }

    barrier();

    // convolve vertically
    float vsum = 0.0;
    for (uint y = 0; y < row_size; y++) {
        float g = grain[gl_LocalInvocationID.y + y][gl_LocalInvocationID.x + taps];
        vsum += weights[y] * g;
    }

    vec4 color = HOOKED_tex(HOOKED_pos);
    color.rgb += vec3(intensity * vsum);
    imageStore(out_image, ivec2(gl_GlobalInvocationID), color);
}
```
