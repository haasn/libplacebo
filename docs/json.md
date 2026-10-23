# JSON scenes

A `pl_scene` is a [compositor](https://code.videolan.org/videolan/libplacebo/-/blob/master/src/include/libplacebo/compositor.h)
configured from a JSON description: the placement and timing of each input,
plus the render options of every input and of the canvas as a whole.

```c
pl_scene scene = pl_scene_create_json(log, json, strlen(json));
if (!scene)
    return;

// Scenes must be initialized with a GPU before they can be rendered
if (!pl_compositor_init(scene->comp, gpu))
    return;

// Provide the frames for each input, in the order they appear in the JSON
for (int i = 0; i < scene->comp->num_inputs; i++) {
    pl_input input = scene->comp->inputs[i];
    input->user_data = &sources[i];
    input->get_frame = my_get_frame;
}

pl_scene_render(scene, &target, &queue_params, NULL);

pl_scene_destroy(&scene);
```

The scene owns the compositor, and with it `pl_compositor.params` as well as
the `params` of every input. Inputs may not be added to or removed from a
scene.

## Structure

A scene is a JSON object with the following optional keys:

```json
{
    "params": { "border": "tiles", "dither": false },
    "color": { "primaries": "bt.2020", "transfer": "pq" },
    "inputs": [
        {
            "start_pts": "1001/30000",
            "stop_pts": 12.5,
            "fit": "cover",
            "dst_rect": { "x0": 0.0, "y0": 0.0, "x1": 0.5, "y1": 0.5 },
            "color": { "transfer": "pq", "hdr": { "max_luma": 1000 } },
            "params": { "upscaler": "ewa_lanczossharp", "deband": true }
        },
        {
            "z_index": 1,
            "fit": "contain",
            "dst_rect": { "x0": 0.5, "x1": 1.0 }
        }
    ]
}
```

- `params`: render options for the canvas, i.e. for composing the inputs and
  rendering the result onto the target frame. See [render options](#render-options).
- `inputs`: an array with one object per input, describing its timing,
  placement and frame property overrides, as well as its own `params`. See
  [inputs](#inputs).
- `color`, `repr`, `chroma_location`: overrides for the color space,
  representation and chroma location of the target frame. See [color overrides](#color-overrides).

## Values

Every value is given as a JSON number, boolean or string:

- Numbers are passed on as written, so `1000`, `1e3` and `1000.0` are all
  equivalent for a floating point option.
- `true` and `false` are valid only for boolean options.
- Strings are accepted for every option. This is how named values (e.g.
  `"fit": "cover"`) are given, and it also allows writing floating point
  values as fractions, such as `"1001/30000"`.

Nested structs, such as the rects or the color overrides, are written as nested
objects.

Keys are applied in the order they appear. Repeating a key is allowed, with
the last occurrence taking effect.

## Inputs

Each input object takes the following keys, all optional. Their meanings are
documented in full on the corresponding fields of `pl_input`.

| Key          | Type                   | Default    | Description                                            |
|--------------|------------------------|------------|--------------------------------------------------------|
| `start_pts`  | number                 | `0`        | Output timestamp at which the input appears            |
| `stop_pts`   | number                 | unbounded  | Output timestamp at which the input disappears again   |
| `start_mode` | `nearest`, `up`, `down`| `nearest`  | How `start_pts` is rounded onto the output frames      |
| `stop_mode`  | `nearest`, `up`, `down`| `nearest`  | How `stop_pts` is rounded onto the output frames       |
| `seek_pts`   | number                 | `0`        | Position within the input's own timeline at `start_pts`|
| `z_index`    | integer                | `0`        | Compositing order, higher values are drawn on top      |
| `rotation`   | integer                | `0`        | Extra rotation, in clockwise multiples of 90°          |
| `fit`        | see below              | `contain`  | How the image is fitted into `dst_rect`                |
| `src_rect`   | [rect](#rects)         | full frame | Part of the image to display                           |
| `dst_rect`   | [rect](#rects)         | full frame | Part of the canvas to display it in                    |
| `focus`      | `{ "x", "y" }`         | `0.5, 0.5` | Focus point, deciding how the image is aligned within `dst_rect` |
| `color`      | [color space](#color-overrides) | unset | Overrides the color space of the input's frames |
| `repr`       | [color repr](#color-overrides)  | unset | Overrides the color representation of the input's frames |
| `chroma_location` | [chroma location](#color-overrides) | unset | Overrides the chroma location of the input's frames |
| `params`     | [render options](#render-options) | `pl_render_default_params` | Render options for this input |

`start_pts` and `stop_pts` are given in the output timeline, while `seek_pts`
is given in the input timeline (the input's own timestamps).

The available `fit` modes are:

- `contain`: Largest size contained within `dst_rect`, preserving aspect.
- `cover`: Smallest size that covers `dst_rect`, preserving aspect.
- `fill`: Stretched to the exact size of `dst_rect`, ignoring aspect.
- `scale_down`: Like `place`, but scaled down like `contain` if it would not
  fit otherwise.
- `place`: Unscaled, i.e. displayed at its native size.
- `raw`: Like `place`, but rounded to exact pixels, mapping image pixels
  one-to-one onto canvas pixels.

### Rects

`src_rect` and `dst_rect` are given as an object with the keys `x0`, `y0`, `x1`
and `y1`, relative to the image and the canvas respectively, in the range
`[0, 1]`. Both are expressed in the displayed (i.e. rotated) orientation.
Leaving out a key keeps its default, so e.g. `{ "x1": 0.5 }` selects the left
half. Swapping `x0` and `x1` (or `y0` and `y1`) mirrors the image.

### Color overrides

Any value set in `color`, `repr` or `chroma_location` replaces the corresponding value reported by
the input's frames (or, at the top level, of the target frame); unset values
(or `auto`) leave them as they are.

`color` takes the following keys:

- `primaries`: `auto`, `bt.601-525`, `bt.601-625`, `bt.709`, `bt.470m`,
  `ebu3213`, `bt.2020`, `apple`, `adobe`, `prophoto`, `cie1931`, `dci-p3`,
  `display-p3`, `v-gamut`, `s-gamut`, `film-c`, `aces-ap0`, `aces-ap1`
- `transfer`: `auto`, `bt.1886`, `srgb`, `linear`, `gamma1.8`, `gamma2.0`,
  `gamma2.2`, `gamma2.4`, `gamma2.6`, `gamma2.8`, `prophoto`, `st428`, `pq`,
  `hlg`, `v-log`, `s-log1`, `s-log2`, `scrgb`
- `hdr`: an object with the keys `min_luma`, `max_luma`, `max_cll` and
  `max_fall`, all in cd/m²

`repr` takes the following keys:

- `sys`: `auto`, `bt.601`, `bt.709`, `smpte-240m`, `bt.2020-ncl`,
  `bt.2020-cl`, `bt.2100-pq`, `bt.2100-hlg`, `dolbyvision`, `ycgco`,
  `ycgco-re`, `ycgco-ro`, `rgb`, `xyz`
- `levels`: `auto`, `limited`, `full`
- `alpha`: `unknown`, `independent`, `premultiplied`, `none`
- `bits`: an object with the keys `sample_depth`, `color_depth` and
  `bit_shift`

`chroma_location` takes one of `auto`, `left`, `center`, `top-left`,
`top-center`, `bottom-left` or `bottom-center`.

## Render options

Each `params` object takes the [options](options.md) documented for
`pl_options`, under the same keys as the corresponding option names.

```json
"params": {
    "preset": "high_quality",
    "upscaler": "ewa_lanczossharp",
    "deband": true,
    "deband_iterations": 2,
    "brightness": 0.1
}
```

As usual, the options of a section only take effect while that section is
enabled; for example, `deband_iterations` does nothing without
`"deband": true`. Since keys are applied in order, a `preset` should come
before any options it is meant to be combined with.

### Canvas and input options

Every option takes effect in exactly one place, either on the canvas or on
each input, whichever it applies to:

- Options affecting the canvas as a whole are taken from the top-level
  `params`, and are ignored in the `params` of an input. These include
  things like `background`, `border`, `dither`, etc.
- Options affecting the image are taken from the `params` of each input, and
  are ignored in the top-level `params`. These include things like
  `frame_mixer`, `deband`, the color adjustment, etc.
- All other options, such as the scalers or the color mapping, apply to
  whichever of the two they are given for.

The `border` is the part of the canvas not covered by the lowest visible input,
which is also the input it is blurred from with `border=blur`. The `background`
is drawn below all inputs.

## Animations

Any option, whether a field of an input (such as the rects, `focus` or the
timestamps) or one of its render options (such as `brightness`), can be
animated by giving it an array of keyframes instead of a single value:

```json
"dst_rect": {
    "x1": [
        { "pts": 0.0, "value": 0.5, "easing": [0.42, 0, 0.58, 1] },
        { "pts": 2.0, "value": 1.0 }
    ]
}
```

Each keyframe takes the following keys:

- `pts`: The output timestamp of the keyframe, in the same timebase as
  `pl_queue_params.pts`. Required.
- `value`: The value at that timestamp, given and range checked like any other
  value of the option. Required.
- `easing`: How to transition from this keyframe to the next one, as the
  control points `[x1, y1, x2, y2]` of a cubic Bézier curve from `(0, 0)` to
  `(1, 1)`, like the CSS `cubic-bezier()` function, or as the name of one of
  the predefined curves below. `x1` and `x2` must lie in the range `[0, 1]`.
  If omitted, the value is held until the next keyframe.
  Only supported for numeric options, with the values of integer options
  rounded to the nearest integer; all other options always hold their value
  until the next keyframe.

The predefined easing curves are:

| Name          | Control points           |
|---------------|--------------------------|
| linear        | `[0, 0, 1, 1]`           |
| ease          | `[0.25, 0.1, 0.25, 1]`   |
| ease-in       | `[0.42, 0, 1, 1]`        |
| ease-out      | `[0, 0, 0.58, 1]`        |
| ease-in-out   | `[0.42, 0, 0.58, 1]`     |

Keyframes may be listed in any order, however, keyframes with the same `pts`
keep the order they were given in, which can be used to jump from one value
to another instantly. Before the first keyframe, and after the last one, the
option keeps the value of the corresponding boundary keyframe.

The animated values are only updated by `pl_scene_update()`, which should be
called for every output frame before rendering it, with the same `pts` as
passed to the compositor. Until the first update, every animated option has the
value of its first keyframe.

The values in between keyframes are clamped to the valid range of the option,
so a curve that overshoots (with `y1` or `y2` outside of `[0, 1]`) stops at the
limits of that range.

Presets, such as `preset`, cannot be animated.

Since changes to a render option may imply recompiling the affected shaders,
animating any render option also enables `dynamic_constants` for the
`params` it belongs to.
