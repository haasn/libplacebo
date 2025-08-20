# GLSL shader system

## Overall design

Shaders in libplacebo are all written in GLSL, and built up incrementally, on
demand. Generally, all shaders for each frame are generated *per frame*. So
functions like `pl_shader_color_map` etc. are run anew for every frame. This
makes the renderer very stateless and allows us to directly embed relevant
constants, uniforms etc. as part of the same code that generates the actual
GLSL shader.

To avoid this from becoming wasteful, libplacebo uses an internal string
building abstraction
([`pl_str_builder`](https://code.videolan.org/videolan/libplacebo/-/blob/master/src/pl_string.h#L263)).
Rather than building up a string directly, a `pl_str_builder` is like a list of
string building functions/callbacks to execute in order to generate the actual
shader. Combined with an efficient `pl_str_builder_hash`, this allows us to
avoid the bulk of the string templating work for already-cached shaders.

## Legacy API

For the vast majority of libplacebo's history, the main entry-point into the
shader building mechanism was the `GLSL()` macro ([and
variants](#shader-sections-glsl-glslh-glslf)), which works like a
`printf`-append:

```c linenums="1"
void pl_shader_extract_features(pl_shader sh, struct pl_color_space csp)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    sh_describe(sh, "feature extraction");
    pl_shader_linearize(sh, &csp);
    GLSL("// pl_shader_extract_features             \n"
         "{                                         \n"
         "vec3 lms = %f * "$" * color.rgb;          \n"
         "lms = pow(max(lms, 0.0), vec3(%f));       \n"
         "lms = (vec3(%f) + %f * lms)               \n"
         "        / (vec3(1.0) + %f * lms);         \n"
         "lms = pow(lms, vec3(%f));                 \n"
         "float I = dot(vec3(%f, %f, %f), lms);     \n"
         "color = vec4(I, 0.0, 0.0, 1.0);           \n"
         "}                                         \n",
         PL_COLOR_SDR_WHITE / 10000,
         SH_MAT3(pl_ipt_rgb2lms(pl_raw_primaries_get(csp.primaries))),
         PQ_M1, PQ_C1, PQ_C2, PQ_C3, PQ_M2,
         pl_ipt_lms2ipt.m[0][0], pl_ipt_lms2ipt.m[0][1], pl_ipt_lms2ipt.m[0][2]);
}
```

The special macro `$` is a stand-in for an *identifier* (`ident_t`), which is
the internal type used to pass references to loaded uniforms, descriptors and
so on:

```c
typedef unsigned short ident_t;
#define $           "_%hx"
#define NULL_IDENT  0u

// ...

ident_t sh_var_mat3(pl_shader sh, const char *name, pl_matrix3x3 val);
#define SH_MAT3(val) sh_var_mat3(sh, "mat", val)
```

In general, constants in libplacebo are divided into three categories:

### Literal shader constants

These are values that are expected to change very infrequently (or never), or
for which we want to generate a different shader variant per value. Such values
should be directly formatted as numbers into the shader text: `%d`, `%f` and so
on. This is commonly used for array sizes, constants that depend only on
hardware limits, constants that never change (but which have a friendly name,
like `PQ_C2` above), and so on.

As an example, the debanding iterations weights are hard-coded like this,
because the debanding shader is expected to change as a result of a different
number of iterations anyway:

```c linenums="1"
// For each iteration, compute the average at a given distance and
// pick it instead of the color if the difference is below the threshold.
for (int i = 1; i <= params->iterations; i++) {
    GLSL(// Compute a random angle and distance
         "d = "$".xy * vec2(%d.0 * "$", %f);    \n" // (1)
         "d = d.x * vec2(cos(d.y), sin(d.y));   \n"
         // Sample at quarter-turn intervals around the source pixel
         "avg = T(0.0);                         \n"
         "avg += GET(+d.x, +d.y);               \n"
         "avg += GET(-d.x, +d.y);               \n"
         "avg += GET(-d.x, -d.y);               \n"
         "avg += GET(+d.x, -d.y);               \n"
         "avg *= 0.25;                          \n"
         // Compare the (normalized) average against the pixel
         "diff = abs(res - avg);                \n"
         "bound = T("$" / %d.0);                \n",
         prng, i, radius, M_PI * 2,
         threshold, i);

    if (num_comps > 1) {
        GLSL("res = mix(avg, res, greaterThan(diff, bound)); \n");
    } else {
        GLSL("res = mix(avg, res, diff > bound); \n");
    }
}
```

1.  The `%d.0` here corresponds to the iteration index `i`, while the `%f`
    corresponds to the fixed constant `M_PI * 2`.

### Specializable shader constants

These are used for tunable parameters that are expected to change infrequently
during normal playback. These constitute by far the biggest category, and most
parameters coming from the various `_params` structs should be loaded like
this.

They are loaded using the `sh_const_*()` functions, which generate a
specialization constant on supported platforms, falling back to a literal
shader `#define` otherwise. For anoymous parameters, you can use the
short-hands `SH_FLOAT`, `SH_INT` etc.:

```c
ident_t sh_const_int(pl_shader sh, const char *name, int val);
ident_t sh_const_uint(pl_shader sh, const char *name, unsigned int val);
ident_t sh_const_float(pl_shader sh, const char *name, float val);
#define SH_INT(val)     sh_const_int(sh, "const", val)
#define SH_UINT(val)    sh_const_uint(sh, "const", val)
#define SH_FLOAT(val)   sh_const_float(sh, "const", val)
```

Here is an example of them in action:

```c linenums="1"
void pl_shader_sigmoidize(pl_shader sh, const struct pl_sigmoid_params *params)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    params = PL_DEF(params, &pl_sigmoid_default_params);
    float center = PL_DEF(params->center, 0.75);
    float slope  = PL_DEF(params->slope, 6.5);

    // This function needs to go through (0,0) and (1,1), so we compute the
    // values at 1 and 0, and then scale/shift them, respectively.
    float offset = 1.0 / (1 + expf(slope * center));
    float scale  = 1.0 / (1 + expf(slope * (center - 1))) - offset;

    GLSL("// pl_shader_sigmoidize                               \n"
         "color = clamp(color, 0.0, 1.0);                       \n"
         "color = vec4("$") - vec4("$") *                       \n"
         "    log(vec4(1.0) / (color * vec4("$") + vec4("$"))   \n"
         "        - vec4(1.0));                                 \n",
         SH_FLOAT(center), SH_FLOAT(1.0 / slope),
         SH_FLOAT(scale), SH_FLOAT(offset));
}
```

The advantage of this type of shader constant is that they will be
transparently replaced by dynamic uniforms whenever
`pl_render_params.dynamic_constants` is true, which allows the renderer to
respond more instantly to changes in the parameters (e.g. as a result of a user
dragging a slider around). During "normal" playback, they will then be
"promoted" to actual shader constants to prevent them from taking up registers.

### Dynamic variables

For anything else, e.g. variables which are expected to change very frequently,
you can use the generic `sh_var()` mechanism, which sends constants either as
elements of a uniform buffer, or directly as push constants:

```c
ident_t sh_var_int(pl_shader sh, const char *name, int val, bool dynamic);
ident_t sh_var_uint(pl_shader sh, const char *name, unsigned int val, bool dynamic);
ident_t sh_var_float(pl_shader sh, const char *name, float val, bool dynamic);
#define SH_INT_DYN(val)   sh_var_int(sh, "const", val, true)
#define SH_UINT_DYN(val)  sh_var_uint(sh, "const", val, true)
#define SH_FLOAT_DYN(val) sh_var_float(sh, "const", val, true)
```

These are used primarily when a variable is expected to change very frequently,
e.g. as a result of randomness, or for constants which depend on dynamically
computed, source-dependent variables (e.g. input frame characteristics):

```c linenums="1"
if (params->show_clipping) {
    const float eps = 1e-6f;
    GLSL("bool clip_hi, clip_lo;                            \n"
         "clip_hi = any(greaterThan(color.rgb, vec3("$"))); \n"
         "clip_lo = any(lessThan(color.rgb, vec3("$")));    \n"
         "clip_hi = clip_hi || ipt.x > "$";                 \n"
         "clip_lo = clip_lo || ipt.x < "$";                 \n",
         SH_FLOAT_DYN(pl_hdr_rescale(PL_HDR_PQ, PL_HDR_NORM, tone.input_max) + eps),
         SH_FLOAT(pl_hdr_rescale(PL_HDR_PQ, PL_HDR_NORM, tone.input_min) - eps),
         SH_FLOAT_DYN(tone.input_max + eps),
         SH_FLOAT(tone.input_min - eps));
}
```

### Shader sections (GLSL, GLSLH, GLSLF)

Shader macros come in three main flavors, depending on where the resulting text
should be formatted:

- `GLSL`: Expanded in the scope of the current `main` function,
  and is related to code directly processing the current pixel value.
- `GLSLH`: Printed to the 'header', before the first function, but after
  variables, uniforms etc. This is used for global definitions, helper
  functions, shared memory variables, and so on.
- `GLSLF`: Printed to the `footer`, which is always at the end of the current
  `main` function, but before returning to the caller / writing to the
  framebuffer. Used to e.g. update SSBO state in preparation for the next
  frame.

Finally, there is a fourth category `GLSLP` (prelude), which is currently only
used internally to generate preambles during e.g. compute shader translation.

## New #pragma GLSL macro

Starting with libplacebo v6, the internal shader system has been augmented by a
custom macro preprocessor, which is designed to ease the boilerplate of writing
shaders (and also strip redundant whitespace from generated shaders). The code
for this is found in the
[tools/glsl_preproc](https://code.videolan.org/videolan/libplacebo/-/tree/master/tools/glsl_preproc)
directory.

In a nutshell, this allows us to embed GLSL snippets directly as `#pragma GLSL`
macros (resp. `#pragma GLSLH`, `#pragma GLSLF`):

```c linenums="1"
bool pl_shader_sample_bicubic(pl_shader sh, const struct pl_sample_src *src)
{
    ident_t tex, pos, pt;
    float rx, ry, scale;
    if (!setup_src(sh, src, &tex, &pos, &pt, &rx, &ry, NULL, &scale, true, LINEAR))
        return false;

    if (rx < 1 || ry < 1) {
        PL_TRACE(sh, "Using fast bicubic sampling when downscaling. This "
                 "will most likely result in nasty aliasing!");
    }

    // Explanation of how bicubic scaling with only 4 texel fetches is done:
    //   http://www.mate.tue.nl/mate/pdfs/10318.pdf
    //   'Efficient GPU-Based Texture Interpolation using Uniform B-Splines'

    sh_describe(sh, "bicubic");
#pragma GLSL /* pl_shader_sample_bicubic */         \
    vec4 color;                                     \
    {                                               \
    vec2 pos = $pos;                                \
    vec2 size = vec2(textureSize($tex, 0));         \
    vec2 frac  = fract(pos * size + vec2(0.5));     \
    vec2 frac2 = frac * frac;                       \
    vec2 inv   = vec2(1.0) - frac;                  \
    vec2 inv2  = inv * inv;                         \
    /* compute basis spline */                      \
    vec2 w0 = 1.0/6.0 * inv2 * inv;                 \
    vec2 w1 = 2.0/3.0 - 0.5 * frac2 * (2.0 - frac); \
    vec2 w2 = 2.0/3.0 - 0.5 * inv2  * (2.0 - inv);  \
    vec2 w3 = 1.0/6.0 * frac2 * frac;               \
    vec4 g = vec4(w0 + w1, w2 + w3);                \
    vec4 h = vec4(w1, w3) / g + inv.xyxy;           \
    h.xy -= vec2(2.0);                              \
    /* sample four corners, then interpolate */     \
    vec4 p = pos.xyxy + $pt.xyxy * h;               \
    vec4 c00 = textureLod($tex, p.xy, 0.0);         \
    vec4 c01 = textureLod($tex, p.xw, 0.0);         \
    vec4 c0 = mix(c01, c00, g.y);                   \
    vec4 c10 = textureLod($tex, p.zy, 0.0);         \
    vec4 c11 = textureLod($tex, p.zw, 0.0);         \
    vec4 c1 = mix(c11, c10, g.y);                   \
    color = ${float:scale} * mix(c1, c0, g.x);      \
    }

    return true;
}
```

This gets transformed, by the GLSL macro preprocessor, into an optimized shader
template invocation like the following:

```c linenums="1"
{
    // ...
    sh_describe(sh, "bicubic");
    const struct __attribute__((__packed__)) {
        ident_t pos;
        ident_t tex;
        ident_t pt;
        ident_t scale;
    } _glsl_330_args = {
        .pos = pos,
        .tex = tex,
        .pt = pt,
        .scale = sh_const_float(sh, "scale", scale),
    };
    size_t _glsl_330_fn(void *, pl_str *, const uint8_t *);
    pl_str_builder_append(sh->buffers[SH_BUF_BODY], _glsl_330_fn,
                          &_glsl_330_args, sizeof(_glsl_330_args));
    // ...
}

size_t _glsl_330_fn(void *alloc, pl_str *buf, const uint8_t *ptr)
{
    struct __attribute__((__packed__)) {
        ident_t pos;
        ident_t tex;
        ident_t pt;
        ident_t scale;
    } vars;
    memcpy(&vars, ptr, sizeof(vars));

    pl_str_append_asprintf_c(alloc, buf,
        "/* pl_shader_sample_bicubic */\n"
        "    vec4 color;\n"
        "    {\n"
        "    vec2 pos = /*pos*/_%hx;\n"
        "    vec2 size = vec2(textureSize(/*tex*/_%hx, 0));\n"
        "    vec2 frac  = fract(pos * size + vec2(0.5));\n"
        "    vec2 frac2 = frac * frac;\n"
        "    vec2 inv   = vec2(1.0) - frac;\n"
        "    vec2 inv2  = inv * inv;\n"
        "    /* compute basis spline */\n"
        "    vec2 w0 = 1.0/6.0 * inv2 * inv;\n"
        "    vec2 w1 = 2.0/3.0 - 0.5 * frac2 * (2.0 - frac);\n"
        "    vec2 w2 = 2.0/3.0 - 0.5 * inv2  * (2.0 - inv);\n"
        "    vec2 w3 = 1.0/6.0 * frac2 * frac;\n"
        "    vec4 g = vec4(w0 + w1, w2 + w3);\n"
        "    vec4 h = vec4(w1, w3) / g + inv.xyxy;\n"
        "    h.xy -= vec2(2.0);\n"
        "    /* sample four corners, then interpolate */\n"
        "    vec4 p = pos.xyxy + /*pt*/_%hx.xyxy * h;\n"
        "    vec4 c00 = textureLod(/*tex*/_%hx, p.xy, 0.0);\n"
        "    vec4 c01 = textureLod(/*tex*/_%hx, p.xw, 0.0);\n"
        "    vec4 c0 = mix(c01, c00, g.y);\n"
        "    vec4 c10 = textureLod(/*tex*/_%hx, p.zy, 0.0);\n"
        "    vec4 c11 = textureLod(/*tex*/_%hx, p.zw, 0.0);\n"
        "    vec4 c1 = mix(c11, c10, g.y);\n"
        "    color = /*scale*/_%hx * mix(c1, c0, g.x);\n"
        "    }\n",
        vars.pos,
        vars.tex,
        vars.pt,
        vars.tex,
        vars.tex,
        vars.tex,
        vars.tex,
        vars.scale
    );

    return sizeof(vars);
}
```

To support this style of shader programming, special syntax was invented:

### Shader variables

Instead of being formatted with `"$"`, `%f` etc. and supplied in a big list,
printf style, GLSL macros may directly embed shader variables:

```c
ident_t pos, tex = sh_bind(sh, texture, ..., &pos, ...);
#pragma GLSL vec4 color = texture($tex, $pos);
```

The simplest possible shader variable is just `$name`, which corresponds to
any variable of type `ident_t`. More complicated expression are also possible:

```glsl
#define RAND3 ${sh_prng(sh, false, NULL)}
color.rgb += ${float:params->noise} * RAND3;
```

In the expression `${float:params->noise}`, the `float:` prefix here transforms
the shader variable into the equivalent of `SH_FLOAT()` in the legacy API,
that is, a generic float (specialization) constant. Other possible types are:

```glsl
TYPE  i = ${ident: sh_desc(...)};
float f = ${float: M_PI};
int   i = ${int:   params->width};
uint  u = ${uint:  sizeof(ssbo)};
```

In addition to a type specifier, the optional qualifiers `dynamic` and `const`
will modify the variable, turning it into (respectively) a dynamically loaded
uniform (`SH_FLOAT_DYN` etc.), or a hard-coded shader literal (`%d`, `%f`
etc.):

```glsl
const float base = ${const float: M_LOG10E};
int seed = ${dynamic int: rand()};
```

For sampling from component masks, the special types `swizzle` and
`(b|u|i)vecType` can be used to generate the appropriate texture swizzle and
corresponding vector type:

```glsl
${vecType: comp_mask} tmp = color.${swizzle: comp_mask};
```

### Macro directives

Lines beginning with `@` are not included in the GLSL as-is, but instead parsed
as macro directives, to control the code flow inside the macro expansion:

#### @if / @else

Standard-purpose conditional. Example:

```glsl
float alpha = ...;
@if (repr.alpha == PL_ALPHA_INDEPENDENT)
    color.a *= alpha;
@else
    color.rgba *= alpha;
```

The condition is evaluated outside the macro (in the enclosing scope) and
the resulting boolean variable is directly passed to the template.

An `@if` block can also enclose multiple lines:

```glsl
@if (threshold > 0) {
    float thresh = ${float:threshold};
    coeff = mix(coeff, vec2(0.0), lessThan(coeff, vec2(thresh)));
    coeff = mix(coeff, vec2(1.0), greaterThan(coeff, vec2(1.0 - thresh)));
@}
```

#### @for

This can be used to generate (unrolled) loops:

```glsl
int offset = ${const int: params->kernel_width / 2};
float sum = 0.0;
@for (x < params->kernel_width)
    sum += textureLodOffset($luma, $pos, 0.0, int(@sum - offset)).r;
```

This introduces a local variable, `@x`, which expands to an integer containing
the current loop index. Loop indices always start at 0. Valid terminating
conditions include `<` and `<=`, and the loop stop condition is also evaluated
as an integer.

Alternatively, this can be used to iterate over a bitmask (as commonly used for
e.g. components in a color mask):

```glsl
float weight = /* ... */;
vec4 color = textureLod($tex, $pos, 0.0);
@for (c : params->component_mask)
    sum[@c] += weight * color[@c];
```

Finally, to combine loops with conditionals, the special syntax `@if @(cond)`
may be used to evaluate expressions inside the template loop:

```glsl
@for (i < 10) {
    float weight = /* ... */;
    @if @(i < 5)
        weight = -weight;
    sum += weight * texture(...);
@}
```

In this case, the `@if` conditional may only reference local (loop) variables.

#### @switch / @case

This corresponds fairly straightforwardly to a normal switch/case from C:

```glsl
@switch (color->transfer) {
@case PL_COLOR_TRC_SRGB:
    color.rgb = mix(color.rgb * 1.0/12.92,
                    pow((color.rgb + vec3(0.055)) / 1.055, vec3(2.4)),
                    lessThan(vec3(0.04045), color.rgb));
    @break;
@case PL_COLOR_TRC_GAMMA18:
    color.rgb = pow(color.rgb, vec3(1.8));
    @break;
@case PL_COLOR_TRC_GAMMA20:
    color.rgb = pow(color.rgb, vec3(2.0));
    @break;
@case PL_COLOR_TRC_GAMMA22:
    color.rgb = pow(color.rgb, vec3(2.2));
    @break;
/* ... */
@}
```

The switch body is always evaluated as an `unsigned int`.
