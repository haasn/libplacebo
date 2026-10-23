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

A scene is a JSON object with two optional keys:

```json
{
    "params": { "border": "tiles", "dither": false },
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

Any value set in `color` or `repr` replaces the corresponding value reported by
the input's frames; unset values (or `auto`) leave them as they are.

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
