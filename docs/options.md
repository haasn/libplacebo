# Options

The following provides an overview of all options available via the built-in
`pl_options` system.

## Global preset

### `preset=<default|fast|high_quality>`

Override all options from all sections by the values from the given
preset. The following presets are available:

- `default`: Default settings, tuned to provide a balance of performance and
  quality. Should be fine on almost all systems.
- `fast`: Disable all advanced rendering, equivalent to passing `no` to every
  option. Increases performance on very slow / old integrated GPUs.
- `high_quality`: Reset all structs to their `high_quality` presets (where
  available), set the upscaler to `ewa_lanczossharp`, and enable `deband=yes`.
  Suitable for use on machines with a discrete GPU.

## Scaling

### `upscaler=<filter>`

Sets the filter used for upscaling. Defaults to `lanczos`. Pass `upscaler=help`
to see a full list of filters. The most relevant options, roughly ordered from
fastest to slowest:

- `none`: No filter, only use basic GPU texture sampling
- `nearest`: Nearest-neighbour (box) sampling (very fast)
- `bilinear`: Bilinear sampling (very fast)
- `oversample`: Aspect-ratio preserving nearest neighbour sampling (very fast)
- `bicubic`: Bicubic interpolation (fast)
- `gaussian`: Gaussian smoothing (fast)
- `catmull_rom`: Catmull-Rom cubic spline
- `lanczos`: Lanczos reconstruction
- `ewa_lanczos`: EWA Lanczos ("Jinc") reconstruction (slow)
- `ewa_lanczossharp`: Sharpened version of `ewa_lanczos` (slow)
- `ewa_lanczos4sharpest`: Very sharp version of `ewa_lanczos`, with
  anti-ringing (very slow)

### `downscaler=<filter>`

Sets the filter used for downscaling. Defaults to `hermite`. Pass
`downscaler=help` to see a full list of filters. The most relevant options,
roughly ordered from fastest to slowest:

- `none`: Use the same filter as specified for `upscaler`
- `box`: Box averaging (very fast)
- `hermite`: Hermite-weighted averaging (fast)
- `bilinear`: Bilinear (triangle) averaging (fast)
- `bicubic`: Bicubic interpolation (fast)
- `gaussian`: Gaussian smoothing (fast)
- `catmull_rom`: Catmull-Rom cubic spline
- `mitchell`: Mitchell-Netravalia cubic spline
- `lanczos`: Lanczos reconstruction

### `plane_upscaler=<filter>`, `plane_downscaler=<filter>`

Override the filter used for upscaling/downscaling planes, e.g. chroma/alpha.
If set to `none`, use the same setting as `upscaler` and `downscaler`,
respectively. Defaults to `none` for both.

### `frame_mixer=<filter>`

Sets the filter used for frame mixing (temporal interpolation). Defaults to
`oversample`. Pass `frame_mixer=help` to see a full list of filters. The most
relevant options, roughly ordered from fastest to slowest:

- `none`: Disable frame mixing, show nearest frame to target PTS
- `oversample`: Oversampling, only mix "edge" frames while preserving FPS
- `hermite`: Hermite-weighted frame mixing
- `linear`: Linear frame mixing
- `cubic`: Cubic B-spline frame mixing

### `antiringing_strength=<0.0..1.0>`

Antiringing strength to use for all filters. A value of `0.0` disables
antiringing, and a value of `1.0` enables full-strength antiringing. Defaults
to `0.0`.

!!! note
    Specific filter presets may override this option.

### Custom scalers

Custom filter kernels can be created by setting the filter to `custom`, in
addition to setting the respective options, replacing `<scaler>` by the
corresponding scaler (`upscaler`, `downscaler`, etc.)

#### `<scaler>_preset=<filter>`

Overrides the value of all options in this section by their default values from
the given filter preset.

#### `<scaler>_kernel=<kernel>`, `<scaler>_window=<kernel>`

Choose the filter kernel and window function, rspectively. Pass `help` to
get a full list of filter kernels. Defaults to `none`.

#### `<scaler>_radius=<0.0..16.0>`

Override the filter kernel radius. Has no effect if the filter kernel
is not resizeable. Defaults to `0.0`, meaning "no override".

#### `<scaler>_clamp=<0.0..1.0>`

Represents an extra weighting/clamping coefficient for negative weights. A
value of `0.0` represents no clamping. A value of `1.0` represents full
clamping, i.e. all negative lobes will be removed. Defaults to `0.0`.

#### `<scaler>_blur=<0.0..100.0>`

Additional blur coefficient. This effectively stretches the kernel, without
changing the effective radius of the filter radius. Setting this to a value of
`0.0` is equivalent to disabling it. Values significantly below `1.0` may
seriously degrade the visual output, and should be used with care. Defaults to
`0.0`.

#### `<scaler>_taper=<0.0..1.0>`

Additional taper coefficient. This essentially flattens the function's center.
The values within `[-taper, taper]` will return `1.0`, with the actual function
being squished into the remainder of `[taper, radius]`. Defaults to `0.0`.

#### `<scaler>_antiring=<0.0..1.0>`

Antiringing override for this filter. Defaults to `0.0`, which infers the value
from `antiringing_strength`.

#### `<scaler>_param1`, `<scaler>_param2` `<scaler>_wparam1`, `<scaler>_wparam2`

Parameters for the respective filter function. Ignored if not tunable. Defaults
to `0.0`.

#### `<scaler>_polar=<yes|no>`

If true, this filter is a polar/2D filter (EWA), instead of a separable/1D
(orthogonal) filter. Defaults to `no`.

## Debanding

These options control the optional debanding step. Debanding can be used to
reduce the prevalence of quantization artefacts in low quality sources, but
can be heavy to compute on weaker devices.

!!! note
    This can also be used as a pure grain generator, by setting
    `deband_iterations=0`.

### `deband=<yes|no>`

Enables debanding. Defaults to `no`.

### `deband_preset=<default>`

Overrides the value of all options in this section by their default values from
the given preset.

### `deband_iterations=<0..16>`

The number of debanding steps to perform per sample. Each
step reduces a bit more banding, but takes time to compute.
Note that the strength of each step falls off very quickly,
so high numbers (>4) are practically useless. Defaults to `1`.

### `deband_threshold=<0.0..1000.0>`

The debanding filter's cut-off threshold. Higher numbers
increase the debanding strength dramatically, but
progressively diminish image details. Defaults to `3.0`.

### `deband_radius=<0.0..1000.0>`

The debanding filter's initial radius. The radius increases
linearly for each iteration. A higher radius will find more
gradients, but a lower radius will smooth more aggressively.
Defaults to `16.0`.

### `deband_grain=<0.0..1000.0>`

Add some extra noise to the image. This significantly helps
cover up remaining quantization artifacts. Higher numbers add
more noise. Defaults to `4.0`, which is very mild.

### `deband_grain_neutral_r, deband_grain_neutral_g, deband_grain_neutral_b`

'Neutral' grain value for each channel being debanded. Grain
application will be modulated to avoid disturbing colors
close to this value. Set this to a value corresponding to
black in the relevant colorspace.

!!! note
    This is done automatically by `pl_renderer` and should not need to be
    touched by the user. This is purely a debug option.

## Sigmoidization

These options control the sigmoidization parameters. Sigmoidization is an
optional step during upscaling which reduces the prominence of ringing
artifacts.

### `sigmoid=<yes|no>`

Enables sigmoidization. Defaults to `yes`.

### `sigmoid_preset=<default>`

Overrides the value of all options in this section by their default values from
the given preset.

### `sigmoid_center=<0.0..1.0>`

The center (bias) of the sigmoid curve. Defaults to `0.75`.

### `sigmoid_slope=<1.0..20.0>`

The slope (steepness) of the sigmoid curve. Defaults to `6.5`.

## Color adjustment

These options affect the decoding of the source color values, and can be used
to subjectively alter the appearance of the video.

### `color_adjustment=<yes|no>`

Enables color adjustment. Defaults to `yes`.

### `color_adjustment_preset=<neutral>`

Overrides the value of all options in this section by their default values from
the given preset.

### `brightness=<-1.0..1.0>`

Brightness boost. Adds a constant bias onto the source
luminance signal. `0.0` = neutral, `1.0` = solid white,
`-1.0` = solid black. Defaults to `0.0`.

### `contrast=<0.0..100.0>`

Contrast gain. Multiplies the source luminance signal by a
constant factor. `1.0` = neutral, `0.0` = solid black.
Defaults to `1.0`.

### `saturation=<0.0..100.0>`

Saturation gain. Multiplies the source chromaticity signal by
a constant factor. `1.0` = neutral, `0.0` = grayscale.
Defaults to `1.0`.

### `hue=<angle>`

Hue shift. Corresponds to a rotation of the UV subvector
around the neutral axis. Specified in radians. Defaults to
`0.0` (neutral).

### `gamma=<0.0..100.0>`

Gamma lift. Subjectively brightnes or darkens the scene while
preserving overall contrast. `1.0` = neutral, `0.0` = solid
black. Defaults to `1.0`.

### `temperature=<-1.143..5.286>`

Color temperature shift. Relative to 6500 K, a value of `0.0` gives you 6500 K
(no change), a value of `-1.0` gives you 3000 K, and a value of `1.0` gives you
10000 K. Defaults to `0.0`.

## HDR peak detection

These options affect the HDR peak detection step. This can be used to greatly
improve the HDR tone-mapping process in the absence of dynamic video metadata,
but may be prohibitively slow on some devices (e.g. weaker integrated GPUs).

### `peak_detect=<yes|no>`

Enables HDR peak detection. Defaults to `yes`.

### `peak_detection_preset=<default|high_quality>`

Overrides the value of all options in this section by their default values from
the given preset. `high_quality` also enables frame histogram measurement.

### `peak_smoothing_period=<0.0..1000.0>`

Smoothing coefficient for the detected values. This controls the time parameter
(tau) of an IIR low pass filter. In other words, it represent the cutoff period
(= 1 / cutoff frequency) in frames. Frequencies below this length will be
suppressed. This helps block out annoying "sparkling" or "flickering" due to
small variations in frame-to-frame brightness. If left as `0.0`, this smoothing
is completely disabled. Defaults to `20.0`.

### `scene_threshold_low=<0.0..100.0>`, `scene_threshold_high=<0.0..100.0>`

In order to avoid reacting sluggishly on scene changes as a result of the
low-pass filter, we disable it when the difference between the current frame
brightness and the average frame brightness exceeds a given threshold
difference. But rather than a single hard cutoff, which would lead to weird
discontinuities on fades, we gradually disable it over a small window of
brightness ranges. These parameters control the lower and upper bounds of this
window, in units of 1% PQ.

Setting either one of these to 0.0 disables this logic. Defaults to `1.0` and
`3.0`, respectively.

### `peak_percentile=<0.0..100.0>`

Which percentile of the input image brightness histogram to consider as the
true peak of the scene. If this is set to `100` (or `0`), the brightest pixel
is measured. Otherwise, the top of the frequency distribution is progressively
cut off. Setting this too low will cause clipping of very bright details, but
can improve the dynamic brightness range of scenes with very bright isolated
highlights.

Defaults to `100.0`. The `high_quality` preset instead sets this to `99.995`,
which is very conservative and should cause no major issues in typical content.

### `allow_delayed_peak=<yes|no>`

Allows the peak detection result to be delayed by up to a single frame, which
can sometimes improve thoughput, at the cost of introducing the possibility of
1-frame flickers on transitions. Defaults to `no`.

## Color mapping

These options affect the way colors are transformed between color spaces,
including tone- and gamut-mapping where needed.

### `color_map=<yes|no>`

Enables the use of these color mapping settings. Defaults to `yes`.

!!! note
    Disabling this option does *not* disable color mapping, it just means "use
    the default options for everything".

### `color_map_preset=<default|high_quality>`

Overrides the value of all options in this section by their default values from
the given preset. `high_quality` also enables HDR contrast recovery.

### `gamut_mapping=<function>`

Gamut mapping function to use to handle out-of-gamut colors, including colors
which are out-of-gamut as a consequence of tone mapping. Defaults to
`perceptual`. The following options are available:

- `clip`: Performs no gamut-mapping, just hard clips out-of-range colors
  per-channel.
- `perceptual`: Performs a perceptually balanced (saturation) gamut mapping,
  using a soft knee function to preserve in-gamut colors, followed by a final
  softclip operation. This works bidirectionally, meaning it can both compress
  and expand the gamut. Behaves similar to a blend of `saturation` and
  `softclip`.
- `softclip`: Performs a perceptually balanced gamut mapping using a soft knee
  function to roll-off clipped regions, and a hue shifting function to preserve
  saturation.
- `relative`: Performs relative colorimetric clipping, while maintaining an
  exponential relationship between brightness and chromaticity.
- `saturation`: Performs simple RGB->RGB saturation mapping. The input R/G/B
  channels are mapped directly onto the output R/G/B channels. Will never clip,
  but will distort all hues and/or result in a faded look.
- `absolute`: Performs absolute colorimetric clipping. Like `relative`, but
  does not adapt the white point.
- `desaturate`: Performs constant-luminance colorimetric clipping, desaturing
  colors towards white until they're in-range.
- `darken`: Uniformly darkens the input slightly to prevent clipping on
  blown-out highlights, then clamps colorimetrically to the input gamut
  boundary, biased slightly to preserve chromaticity over luminance.
- `highlight`: Performs no gamut mapping, but simply highlights out-of-gamut
  pixels.
- `linear`: Linearly/uniformly desaturates the image in order to bring the
  entire image into the target gamut.

### Gamut mapping constants

These settings can be used to fine-tune the constants used for the various
gamut mapping algorithms.

#### `perceptual_deadzone=<0.0..1.0>`

(Relative) chromaticity protection zone for `perceptual` mapping. Defaults to
`0.30`.

#### `perceptual_strength<0.0..1.0>`

Strength of the `perceptual` saturation mapping component. Defaults to `0.80`.

#### `colorimetric_gamma=<0.0..10.0>`

I vs C curve gamma to use for colorimetric clipping (`relative`, `absolute`
and `darken`). Defaults to `1.80`.

#### `softclip_knee=<0.0..1.0>`

Knee point to use for soft-clipping methods (`perceptual`, `softclip`).
Defaults to `0.70`.

#### `softclip_desat=<0.0..1.0>`

Desaturation strength for `softclip`. Defaults to `0.35`.

### `lut3d_size_I=<0..1024>`, `lut3d_size_C=<0..1024>`, `lut3d_size_h=<0..1024>`

Gamut mapping 3DLUT size. Setting a dimension to `0` picks the default value.
Defaults to `48`, `32` and `256`, respectively, for channels `I`, `C` and `h`.

### `lut3d_tricubic=<yes|no>`

Use higher quality, but slower, tricubic interpolation for gamut mapping
3DLUTs. May substantially improve the 3DLUT gamut mapping accuracy, in
particular at smaller 3DLUT sizes. Shouldn't have much effect at the default
size. Defaults to `no`.

### `gamut_expansion=<yes|no>`

If enabled, allows the gamut mapping function to expand the gamut, in cases
where the target gamut exceeds that of the source. If disabled, the source
gamut will never be enlarged, even when using a gamut mapping function capable
of bidirectional mapping. Defaults to `no`.

### `tone_mapping=<function>`

Tone mapping function to use for adapting between difference luminance ranges,
including black point adaptation. Defaults to `spline`. The following functions
are available:

- `clip`: Performs no tone-mapping, just clips out-of-range colors. Retains
  perfect color accuracy for in-range colors but completely destroys
  out-of-range information. Does not perform any black point adaptation.
- `spline`: Simple spline consisting of two polynomials, joined by a single
  pivot point, which is tuned based on the source scene average brightness
  (taking into account dynamic metadata if available). This function can be
  used for both forward and inverse tone mapping.
- `st2094-40`: EETF from SMPTE ST 2094-40 Annex B, which uses the provided OOTF
  based on Bezier curves to perform tone-mapping. The OOTF used is adjusted
  based on the ratio between the targeted and actual display peak luminances.
  In the absence of HDR10+ metadata, falls back to a simple constant bezier
  curve.
- `st2094-10`: EETF from SMPTE ST 2094-10 Annex B.2, which takes into account
  the input signal average luminance in addition to the maximum/minimum.
!!! warning
    This does *not* currently include the subjective gain/offset/gamma controls
    defined in Annex B.3. (Open an issue with a valid sample file if you want
    such parameters to be respected.)
- `bt2390`: EETF from the ITU-R Report BT.2390, a hermite spline roll-off with
  linear segment.
- `bt2446a`: EETF from ITU-R Report BT.2446, method A. Can be used for both
  forward and inverse tone mapping.
- `reinhard:` Very simple non-linear curve. Named after Erik Reinhard.
- `mobius`: Generalization of the `reinhard` tone mapping algorithm to support
  an additional linear slope near black. The name is derived from its function
  shape `(ax+b)/(cx+d)`, which is known as a Möbius transformation. This
  function is considered legacy/low-quality, and should not be used.
- `hable`: Piece-wise, filmic tone-mapping algorithm developed by John Hable
  for use in Uncharted 2, inspired by a similar tone-mapping algorithm used by
  Kodak. Popularized by its use in video games with HDR rendering. Preserves
  both dark and bright details very well, but comes with the drawback of
  changing the average brightness quite significantly. This is sort of similar
  to `reinhard` with `reinhard_contrast=0.24`. This function is considered
  legacy/low-quality, and should not be used.
- `gamma`: Fits a gamma (power) function to transfer between the source and
  target color spaces, effectively resulting in a perceptual hard-knee joining
  two roughly linear sections. This preserves details at all scales, but can
  result in an image with a muted or dull appearance. This function
  is considered legacy/low-quality and should not be used.
- `linear`: Linearly stretches the input range to the output range, in PQ
  space. This will preserve all details accurately, but results in a
  significantly different average brightness. Can be used for inverse
  tone-mapping in addition to regular tone-mapping.
- `linearlight`: Like `linear`, but in linear light (instead of PQ). Works well
  for small range adjustments but may cause severe darkening when
  downconverting from e.g. 10k nits to SDR.

### Tone-mapping constants

These settings can be used to fine-tune the constants used for the various
tone mapping algorithms.

#### `knee_adaptation=<0.0..1.0>`

Configures the knee point, as a ratio between the source average and target
average (in PQ space). An adaptation of `1.0` always adapts the source scene
average brightness to the (scaled) target average, while a value of `0.0` never
modifies scene brightness.

Affects all methods that use the ST2094 knee point determination (currently
`spline`, `st2094-40` and `st2094-10`). Defaults to `0.4`.

#### `knee_minimum=<0.0..0.5>`, `knee_maximum=<0.5..1.0>`

Configures the knee point minimum and maximum, respectively, as a percentage of
the PQ luminance range. Provides a hard limit on the knee point chosen by
`knee_adaptation`. Defaults to `0.1` and `0.8`, respectively.

#### `knee_default=<0.0..1.0>`

Default knee point to use in the absence of source scene average metadata.
Normally, this is ignored in favor of picking the knee point as the (relative)
source scene average brightness level. Defaults to `0.4`.

#### `knee_offset=<0.5..2.0>`

Knee point offset (for `bt2390` only). Note that a value of `0.5` is the
spec-defined default behavior, which differs from the libplacebo default of
`1.0`.

#### `slope_tuning=<0.0..10.0>`, `slope_offset=<0.0..1.0>`

For the single-pivot polynomial (spline) function, this controls the
coefficients used to tune the slope of the curve. This tuning is designed to
make the slope closer to `1.0` when the difference in peaks is low, and closer
to linear when the difference between peaks is high. Defaults to `1.5`, with
offset `0.2`.

#### `spline_contrast=<0.0..1.5>`

Contrast setting for the `spline` function. Higher values make the curve
steeper (closer to `clip`), preserving midtones at the cost of losing
shadow/highlight details, while lower values make the curve shallowed (closer
to `linear`), preserving highlights at the cost of losing midtone contrast.
Values above `1.0` are possible, resulting in an output with more contrast than
the input. Defaults to `0.5`.

#### `reinhard_contrast=<0.0..1.0>`

For the `reinhard` function, this specifies the local contrast coefficient at
the display peak. Essentially, a value of `0.5` implies that the reference
white will be about half as bright as when clipping. Defaults to `0.5`.

#### `linear_knee=<0.0..1.0>`

For legacy functions (`mobius`, `gamma`) which operate on linear light, this
directly sets the corresponding knee point. Defaults to `0.3`.

#### `exposure=<0.0..10.0>`

For linear methods (`linear`, `linearlight`), this controls the linear
exposure/gain applied to the image. Defaults to `1.0`.

### `inverse_tone_mapping=<yes|no>`

If enabled, and supported by the given tone mapping function, will perform
inverse tone mapping to expand the dynamic range of a signal. libplacebo is not
liable for any HDR-induced eye damage. Defaults to `no`.

### `tone_map_metadata=<any|none|hdr10|hdr10plus|cie_y>`

Data source to use when tone-mapping. Setting this to a specific value allows
overriding the default metadata preference logic. Defaults to `any`.

### `tone_lut_size=<0..4096>`

Tone mapping LUT size. Setting `0` picks the default size. Defaults to `256`.

### `contrast_recovery=<0.0..2.0>`

HDR contrast recovery strength. If set to a value above `0.0`, the source image
will be divided into high-frequency and low-frequency components, and a portion
of the high-frequency image is added back onto the tone-mapped output. May
cause excessive ringing artifacts for some HDR sources, but can improve the
subjective sharpness and detail left over in the image after tone-mapping.

Defaults to `0.0`. The `high_quality` preset sets this to `0.3`, which is a
fairly conservativee value and should subtly enhance the image quality without
creating too many obvious artefacts.

### `contrast_smoothness=<1.0..32.0>`

HDR contrast recovery lowpass kernel size. Increasing or decreasing this will
affect the visual appearance substantially. Defaults to `3.5`.

### Debug options

Miscellaneous debugging and display options related to tone/gamut mapping.

#### `force_tone_mapping_lut=<yes|no>`

Force the use of a full tone-mapping LUT even for functions that have faster
pure GLSL replacements (e.g. `clip`, `linear`, `saturation`). This is a debug
option. Defaults to `no`.

#### `visualize_lut=<yes|no>`

Visualize the color mapping LUTs. Displays a (PQ-PQ) graph of the active
tone-mapping LUT. The X axis shows PQ input values, the Y axis shows PQ output
values. The tone-mapping curve is shown in green/yellow. Yellow means the
brightness has been boosted from the source, dark blue regions show where the
brightness has been reduced. The extra colored regions and lines indicate
various monitor limits, as well a reference diagonal (neutral tone-mapping) and
source scene average brightness information (if available). The background
behind this shows a visualization of the gamut mapping 3DLUT, in IPT space.
Iso-luminance, iso-chromaticity and iso-hue lines are highlighted (depending on
the exact value of `visualize_theta`). Defaults to `no`.

#### `visualize_lut_x0`, `visualize_lut_y0`, `visualize_lut_x0`, `visualize_lut_y1`

Controls where to draw the LUt visualization, relative to the rendered video.
Defaults to `0.0` for `x0`/`y0`, and `1.0` for `x1`/`y1`.

#### `visualize_hue=<angle>`, `visualize_theta=<angle>`

Controls the rotation of the gamut 3DLUT visualization. The `hue` parameter
rotates the gamut through hue space (around the `I` axis), while the `theta`
parameter vertically rotates the cross section (around the `C` axis), in
radians. Defaults to `0.0` for both.

#### `show_clipping=<yes|no>`

Graphically highlight hard-clipped pixels during tone-mapping (i.e. pixels that
exceed the claimed source luminance range). Defaults to `no`.

## Dithering

These options affect the way colors are dithered before output. Dithering is
always required to avoid introducing banding artefacts as a result of
quantization to a lower bit depth output texture.

### `dither=<yes|no>`

Enables dithering. Defaults to `yes`.

### `dither_preset=<default>`

Overrides the value of all options in this section by their default values from
the given preset.

### `dither_method=<method>`

Chooses the dithering method to use. Defaults to `blue`. The following methods
are available:

- `blue`: Dither with blue noise. Very high quality, but requires the use of a
  LUT.
!!! warning
    Computing a blue noise texture with a large size can be very slow, however
    this only needs to be performed once. Even so, using this with a
    `dither_lut_size` greater than `6` is generally ill-advised.
- `ordered_lut`: Dither with an ordered (bayer) dither matrix, using a LUT. Low
  quality, and since this also uses a LUT, there's generally no advantage to
  picking this instead of `blue`. It's mainly there for testing.
- `ordered`: The same as `ordered`, but uses fixed function math instead of a
  LUT. This is faster, but only supports a fixed dither matrix size of 16x16
  (equivalent to `dither_lut_size=4`).
- `white`: Dither with white noise. This does not require a LUT and is fairly
  cheap to compute. Unlike the other modes it doesn't show any repeating
  patterns either spatially or temporally, but the downside is that this is
  visually fairly jarring due to the presence of low frequencies in the noise
  spectrum.

### `dither_lut_size=<1..8>`

For the dither methods which require the use of a LUT (`blue`, `ordered_lut`),
this controls the size of the LUT (base 2). Defaults to `6`.

### `dither_temporal=<yes|no>`

Enables temporal dithering. This reduces the persistence of dithering artifacts
by perturbing the dithering matrix per frame. Defaults to `no`.

!!! warning
    This can cause nasty aliasing artifacts on some LCD screens.

## Cone distortion

These options can be optionally used to modulate the signal in LMS space, in
particular, to simulate color blindiness.

### `cone=<yes|no>`

Enables cone distortion. Defaults to `no`.

### `cone_preset=<preset>`

Overrides the value of all options in this section by their default values from
the given preset. The following presets are available:

- `normal`: No distortion (92% of population)
- `protanomaly`: Red cone deficiency (0.66% of population)
- `protanopia`: Red cone absence (0.59% of population)
- `deuteranomaly`: Green cone deficiency (2.7% of population)
- `deuteranopia`: Green cone absence (0.56% of population)
- `tritanomaly`: Blue cone deficiency (0.01% of population)
- `tritanopia`: Blue cone absence (0.016% of population)
- `monochromacy`: Blue cones only (<0.001% of population)
- `achromatopsia`: Rods only (<0.0001% of population)

### `cones=<none|l|m|s|lm|ms|ls|lms>`

Choose the set of cones to modulate. Defaults to `none`.

### `cone_strength=<gain>`

Defect/gain coefficient to apply to these cones. `1.0` = unaffected, `0.0` =
full blindness. Defaults to `1.0`. Values above `1.0` can be used to instead
boost the signal going to this cone. For example, to partially counteract
deuteranomaly, you could set `cones=m`, `cone_strength=2.0`. Defaults to `0.0`.

## Output blending

These options affect the way the image is blended onto the output framebuffer.

### `blend=<yes|no>`

Enables output blending. Defaults to `no`.

### `blend_preset=<alpha_overlay>`

Overrides the value of all options in this section by their default values from
the given preset. Currently, the only preset is `alpha_overlay`, which
corresponds to normal alpha blending.

### `blend_src_rgb`, `blend_src_alpha`, `blend_dst_rgb`, `blend_dst_alpha`

Choose the blending mode for each component. Defaults to `zero` for all. The
following modes are available:

- `zero`: Component will be unused.
- `one`: Component will be added at full strength.
- `alpha`: Component will be multiplied by the source alpha value.
- `one_minus_alpha`: Component will be multiplied by 1 minus the source alpha.

## Deinterlacing

Configures the settings used to deinterlace frames, if required.

!!! note
    The use of these options requires the caller to pass extra metadata to
    incoming frames to link them together / mark them as fields.

### `deinterlace=<yes|no>`

Enables deinterlacing. Defaults to `no`.

### `deinterlace_preset=<default>`

Overrides the value of all options in this section by their default values from
the given preset.

### `deinterlace_algo=<algorithm>`

Chooses the algorithm to use for deinterlacing. Defaults to `yadif`. The
following algorithms are available:

- `weave`: No-op deinterlacing, just sample the weaved frame un-touched.
- `bob`: Naive bob deinterlacing. Doubles the field lines vertically.
- `yadif`: "Yet another deinterlacing filter". Deinterlacer with temporal and
  spatial information. Based on FFmpeg's Yadif filter algorithm, but adapted
  slightly for the GPU.

### `deinterlace_skip_spatial=<yes|no>`

Skip the spatial interlacing check for `yadif`. Defaults to `no`.

## Distortion

The settings in this section can be used to distort/transform the output image.

### `distort=<yes|no>`

Enables distortion. Defaults to `no`.

### `distort_preset=<default>`

Overrides the value of all options in this section by their default values from
the given preset.

### `distort_scale_x`, `distort_scale_y`

Scale the image in the X/Y dimension by an arbitrary factor. Corresponds to the
main diagonal of the transformation matrix. Defaults to `1.0` for both.

### `distort_shear_x`, `distort_shear_y`

Adds the X/Y dimension onto the Y/X dimension (respectively), scaled by an
arbitrary amount. Corresponds to the anti-diagonal of the 2x2 transformation
matrix. Defaults to `0.0` for both.

### `distort_offset_x`, `distort_offset_y`

Offsets the X/Y dimensions by an arbitrary offset, relative to the image size.
Corresponds to the bottom row of a 3x3 affine transformation matrix. Defaults
to `0.0` for both.

### `distort_unscaled=<yes|no>`

If enabled, the texture is placed inside the center of the canvas without
scaling. Otherwise, it is effectively stretched to the canvas size. Defaults
to `no`.

!!! note
    This option has no effect when using `pl_renderer`.

### `distort_constrain=<yes|no>`

If enabled, the transformation is automatically scaled down and shifted to
ensure that the resulting image fits inside the output canvas. Defaults to
`no`.

### `distort_bicubic=<yes|no>`

If enabled, use bicubic interpolation rather than faster bilinear
interpolation. Higher quality but slower. Defaults to `no`.

### `distort_addreess_mode=<clamp|repeat|mirror>`

Specifies the texture address mode to use when sampling out of bounds. Defaults
to `clamp`.

### `distort_alpha_mode=<none|independent|premultiplied>`

If set to something other than `none`, all out-of-bounds accesses will instead
be treated as transparent, according to the given alpha mode.

## Miscellaneous renderer settings

### `error_diffusion=<kernel>`

Enables error diffusion dithering. Error diffusion is a very slow and memory
intensive method of dithering without the use of a fixed dither pattern. If
set, this will be used instead of `dither_method` whenever possible. It's
highly recommended to use this only for still images, not moving video.
Defaults to `none`. The following options are available:

- `simple`: Simple error diffusion (fast)
- `false-fs`: False Floyd-Steinberg kernel (fast)
- `sierra-lite`: Sierra Lite kernel (slow)
- `floyd-steinberg`: Floyd-Steinberg kernel (slow)
- `atkinson`: Atkinson kernel (slow)
- `jarvis-judice-ninke`: Jarvis, Judice & Ninke kernel (very slow)
- `stucki`: Stucki kernel (very slow)
- `burkes`: Burkes kernel (very slow)
- `sierra-2`: Two-row Sierra (very slow)
- `sierra-3`: Three-row Sierra (very slow)

### `lut_type=<type>`

Overrides the color mapping LUT type. Defaults to `unknown`. The following
options are available:

- `unknown`: Unknown LUT type, try and guess from metadata
- `native`: LUT is applied to raw image contents
- `normalized`: LUT is applied to normalized (HDR) RGB values
- `conversion`: LUT fully replaces color conversion step

!!! note
    There is no way to load LUTs via the options mechanism, so this option only
    has an effect if the LUT is loaded via external means.

### `background_r=<0.0..1.0>`, `background_g=<0.0..1.0>`, `background_b=<0.0..1.0>`

If the image being rendered does not span the entire size of the target, it
will be cleared explicitly using this background color (RGB). Defaults to `0.0`
for all.

### `background_transparency=<0.0..1.0>`

The (inverted) alpha value of the background clear color. Defaults to `0.0`.

### `skip_target_clearing=<yes|no>`

If set, skips clearing the background backbuffer entirely. Defaults to `no`.

!!! note
    This is automatically skipped if the image to be rendered would completely
    cover the backbuffer.

### `corner_rounding=<0.0..1.0>`

If set to a value above `0.0`, the output will be rendered with rounded
corners, as if an alpha transparency mask had been applied. The value indicates
the relative fraction of the side length to round - a value of `1.0` rounds the
corners as much as possible. Defaults to `0.0`.

### `blend_against_tiles=<yes|no>`

If true, then transparent images will made opaque by painting them against a
checkerboard pattern consisting of alternating colors. Defaults to `no`.

### `tile_color_hi_r`, `tile_color_hi_g`, `tile_color_hi_b`, `tile_color_lo_r`, `tile_color_lo_g`, `tile_color_l_b`

The colors of the light/dark tiles used for `blend_against_tiles`. Defaults to
`0.93` for light R/G/B and `0.87` for dark R/G/B, respectively.

### `tile_size=<2..256>`

The size, in output pixels, of the tiles used for `blend_against_tiles`.
Defaults to `32`.

## Performance / quality trade-offs

These should generally be left off where quality is desired, as they can
degrade the result quite noticeably; but may be useful for older or slower
hardware. Note that libplacebo will automatically disable advanced features on
hardware where they are unsupported, regardless of these settings. So only
enable them if you need a performance bump.

### `skip_anti_aliasing=<yes|no>`

Disables anti-aliasing on downscaling. This will result in moiré artifacts and
nasty, jagged pixels when downscaling, except for some very limited special
cases (e.g. bilinear downsampling to exactly 0.5x). Significantly speeds up
downscaling with high downscaling ratios. Defaults to `no`.

### `preserve_mixing_cache=<yes|no>`

Normally, when the size of the target framebuffer changes, or the render
parameters are updated, the internal cache of mixed frames must be discarded in
order to re-render all required frames. Setting this option to `yes` will skip
the cache invalidation and instead re-use the existing frames (with bilinear
scaling to the new size if necessary). This comes at a hefty quality loss
shortly after a resize, but should make it much more smooth. Defaults to `no`.

## Debugging, tuning and testing

These may affect performance or may make debugging problems easier, but
shouldn't have any effect on the quality (except where otherwise noted).

### `skip_caching_single_frame=<yes|no>`

Normally, single frames will also get pushed through the mixer cache, in order
to speed up re-draws. Enabling this option disables that logic, causing single
frames to bypass being written to the cache. Defaults to `no`.

!!! note
    If a frame is *already* cached, it will be re-used, regardless.

### `disable_linear_scaling=<yes|no>`

Disables linearization / sigmoidization before scaling. This might be useful
when tracking down unexpected image artifacts or excessing ringing, but it
shouldn't normally be necessary. Defaults to `no`.

### `disable_builtin_scalers=<yes|no>`

Forces the use of the slower, "general" scaling algorithms even when faster
built-in replacements exist. Defaults to `no`.

### `correct_subpixel_offsets=<yes|no>`

Forces correction of subpixel offsets (using the configured `upscaler`).
Defaults to `no`.

!!! warning
    Enabling this may cause such images to get noticeably blurrier, especially
    when using a polar scaler. It's not generally recommended to enable this.

### `force_dither=<yes|no>`

Forces the use of dithering, even when rendering to 16-bit FBOs. This is
generally pretty pointless because most 16-bit FBOs have high enough depth that
rounding errors are below the human perception threshold, but this can be used
to test the dither code. Defaults to `no`.

### `disable_dither_gamma_correction=<yes|no>`

Disables the gamma-correct dithering logic which normally applies when
dithering to low bit depths. No real use, outside of testing. Defaults to `no`.

### `disable_fbos=<yes|no>`

Completely overrides the use of FBOs, as if there were no renderable texture
format available. This disables most features. Defaults to `no`.

### `force_low_bit_depth_fbos=<yes|no>`

Use only low-bit-depth FBOs (8 bits). Note that this also implies disabling
linear scaling and sigmoidization. Defaults to `no`.

### `dynamic_constants=<yes|no>`

If this is enabled, all shaders will be generated as "dynamic" shaders, with
any compile-time constants being replaced by runtime-adjustable values. This is
generally a performance loss, but has the advantage of being able to freely
change parameters without triggering shader recompilations. It's a good idea to
enable this if you will change these options very frequently, but it should be
disabled once those values are "dialed in". Defaults to `no`.
