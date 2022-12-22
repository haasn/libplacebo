# Rendering content: pl_frame, pl_renderer, and pl_queue

This example roughly builds off the [previous entry](./basic-rendering.md),
and as such will not cover the basics of how to create a window, initialize a
`pl_gpu` and get pixels onto the screen.

## Renderer

The `pl_renderer` set of APIs represents the highest-level interface into
libplacebo, and is what most users who simply want to display e.g. a video
feed on-screen will want to be using.

The basic initialization is straightforward, requiring no extra parameters:

``` c linenums="1"
pl_renderer renderer;

init()
{
    renderer = pl_renderer_create(pllog, gpu);
    if (!renderer)
        goto error;

    // ...
}

uninit()
{
    pl_renderer_destroy(&renderer);
}
```

What makes the renderer powerful is the large number of `pl_render_params` it
exposes. By default, libplacebo provides several presets to use:

* **pl_render_fast_params**: Disables everything except for defaults. This is
  the fastest possible configuration.
* **pl_render_default_params**: Contains the recommended default parameters,
  including some slightly higher quality scaling, as well as dithering.
* **pl_render_high_quality_params**: A preset of reasonable defaults for a
  higher-end machine (i.e. anything with a discrete GPU). This enables most
  of the basic functionality, including upscaling, downscaling, debanding
  and better HDR tone mapping.

Covering all of the possible options exposed by `pl_render_params` is
out-of-scope of this example and would be better served by looking at [the API
documentation](https://code.videolan.org/videolan/libplacebo/-/blob/master/src/include/libplacebo/renderer.h#L94).

### Frames

[`pl_frame`](https://code.videolan.org/videolan/libplacebo/-/blob/master/src/include/libplacebo/renderer.h#L503)
is the struct libplacebo uses to group textures and their metadata together
into a coherent unit that can be rendered using the renderer. This is not
currently a dynamically allocated or refcounted heap object, it is merely a
struct that can live on the stack (or anywhere else). The actual data lives in
corresponding `pl_tex` objects referenced in each of the frame's planes.

``` c linenums="1"
bool render_frame(const struct pl_frame *image,
                  const struct pl_swapchain_frame *swframe)
{
    struct pl_frame target;
    pl_frame_from_swapchain(&target, swframe);

    return pl_render_image(renderer, image, target,
                           &pl_render_default_params);
}
```

!!! note "Renderer state"
    The `pl_renderer` is conceptually (almost) stateless. The only thing that
    is needed to get a different result is to change the render params, which
    can be varied freely on every call, if the user desires.

    The one case where this is not entirely true is when using frame mixing
    (see below), or when using HDR peak detection. In this case, the renderer
    can be explicitly reset using `pl_renderer_flush_cache`.

To upload frames, the easiest methods are made available as dedicated helpers
in
[`<libplacebo/utils/upload.h>`](https://code.videolan.org/videolan/libplacebo/-/blob/master/src/include/libplacebo/utils/upload.h),
and
[`<libplacebo/utils/libav.h>`](https://code.videolan.org/videolan/libplacebo/-/blob/master/src/include/libplacebo/utils/libav.h)
(for AVFrames). In general, I recommend checking out the [demo
programs](https://code.videolan.org/videolan/libplacebo/-/tree/master/demos)
for a clearer illustration of how to use them in practice.

### Shader cache

The renderer internally generates, compiles and caches a potentially large
number of shader programs, some of which can be complex. On some platforms
(notably D3D11), these can be quite costly to recompile on every program
launch.

As such, the renderer offers a way to save/restore its internal shader cache
from some external location (managed by the API user). The use of this API is
highly recommended:

``` c linenums="1" hl_lines="1-2 10-14 21-27"
static uint8_t *load_saved_cache();
static void store_saved_cache(uint8_t *cache, size_t bytes);

void init()
{
    renderer = pl_renderer_create(pllog, gpu);
    if (!renderer)
        goto error;

    uint8_t *cache = load_saved_cache();
    if (cache) {
        pl_renderer_load(renderer, cache);
        free(cache);
    }

    // ...
}

void uninit()
{
    size_t cache_bytes = pl_renderer_save(renderer, NULL);
    uint8_t *cache = malloc(cache_bytes);
    if (cache) {
        pl_renderer_save(renderer, cache);
        store_saved_cache(cache, cache_bytes);
        free(cache);
    }

    pl_renderer_destroy(&renderer);
}
```

!!! warning "Cache safety"
    libplacebo performs only minimal validity checking on the shader cache,
    and in general, cannot possibly guard against malicious alteration of such
    files. Loading a cache from an untrusted source represents a remote code
    execution vector.

## Frame mixing

One of the renderer's most powerful features is its ability to compensate
for differences in framerates between the source and display by using [frame
mixing](https://github.com/mpv-player/mpv/wiki/Interpolation) to blend
adjacent frames together.

Using this API requires presenting the renderer, at each vsync, with a
`pl_frame_mix` struct, describing the current state of the vsync. In
principle, such structs can be constructed by hand. To do this, all of the
relevant frames (nearby the vsync timestamp) must be collected, and their
relative distances to the vsync determined, by normalizing all PTS values such
that the vsync represents time `0.0` (and a distance of `1.0` represents the
nominal duration between adjacent frames). Note that timing vsyncs, and
determining the correct vsync duration, are both left as problems for the user
to solve.[^timing]. Here could be an example of a valid struct:

[^timing]: However, this may change in the future, as the recent introduction of
  the Vulkan display timing extension may result in display timing feedback
  being added to the `pl_swapchain` API. That said, as of writing, this has
  not yet happened.

``` c
(struct pl_frame_mix) {
    .num_frames = 6
    .frames = (const struct pl_frame *[]) {
        /* frame 0 */
        /* frame 1 */
        /* ... */
        /* frame 5 */
    },
    .signatures = (uint64_t[]) {
        0x0, 0x1, 0x2, 0x3, 0x4, 0x5 // (1)
    },
    .timestamps = (float[]) {
        -2.4, -1.4, -0.4, 0.6, 1.6, 2.6, // (2)
    },
    .vsync_duration = 0.4, // 24 fps video on 60 fps display
}
```

1.  These must be unique per frame, but always refer to the same frame. For
    example, this could be based on the frame's PTS, the frame's numerical ID
    (in order of decoding), or some sort of hash. The details don't matter,
    only that this uniquely identifies specific frames.

2.  Typically, for CFR sources, frame timestamps will always be separated in
    this list by a distance of 1.0. In this example, the vsync falls roughly
    halfway (but not quite) in between two adjacent frames (with IDs 0x2 and
    0x3).

!!! note "Frame mixing radius"
    In this example, the frame mixing radius (as determined by
    `pl_frame_mix_radius` is `3.0`, so we include all frames that fall within
    the timestamp interval of `[-3, 3)`. In general, you should consult this
    function to determine what frames need to be included in the
    `pl_frame_mix` - though including more frames than needed is not an error.

### Frame queue

Because this API is rather unwieldy and clumsy to use directly, libplacebo
provides a helper abstraction known as `pl_queue` to assist in transforming
some arbitrary source of frames (such as a video decoder) into nicely packed
`pl_frame_mix` structs ready for consumption by the `pl_renderer`:

``` c linenums="1"
#include <libplacebo/utils/frame_queue.h>

pl_queue queue;

void init()
{
    queue = pl_queue_create(gpu);
}

void uninit()
{
    pl_queue_destroy(&queue);
    // ...
}
```

This queue can be interacted with through a number of mechanisms: either
pushing frames (blocking or non-blocking), or by having the queue poll frames
(via blocking or non-blocking callback) as-needed. For a full overview of the
various methods of pushing and polling frames, check the [API
documentation](https://code.videolan.org/videolan/libplacebo/-/blob/master/src/include/libplacebo/utils/frame_queue.h#L115).

In this example, I will assume that we have a separate decoder thread pushing
frames into the `pl_queue` in a blocking manner:

``` c linenums="1"
static void decoder_thread(void)
{
    void *frame;

    while ((frame = /* decode new frame */)) {
        pl_queue_push_block(queue, UINT64_MAX, &(struct pl_source_frame) {
            .pts        = /* frame pts */,
            .duration   = /* frame duration */,
            .map        = /* map callback */,
            .unmap      = /* unmap callback */,
            .frame_data = frame,
        });
    }

    pl_queue_push(queue, NULL); // signal EOF
}
```

Now, in our render loop, we want to call `pl_queue_update` with appropriate
values to retrieve the correct frame mix for each vsync:

``` c linenums="1" hl_lines="3-10 12-21 27"
bool render_frame(const struct pl_swapchain_frame *swframe)
{
    struct pl_frame_mix mix;
    enum pl_queue_status res;
    res = pl_queue_update(queue, &mix, pl_queue_params(
        .pts            = /* time of next vsync */,
        .radius         = pl_frame_mix_radius(&render_params),
        .vsync_duration = /* if known */,
        .timeout        = UINT64_MAX, // (2)
    ));

    switch (res) {
    case PL_QUEUE_OK:
        break;
    case PL_QUEUE_EOF:
        /* no more frames */
        return false;
    case PL_QUEUE_ERR:
        goto error;
    // (1)
    }


    struct pl_frame target;
    pl_frame_from_swapchain(&target, swframe);

    return pl_render_image_mix(renderer, &mix, target,
                               &pl_render_default_params);
}
```

1.  There is a fourth status, `PL_QUEUE_MORE`, which is returned only if the
    resulting frame mix is incomplete (and the timeout was reached) -
    basically this can only happen if the queue runs dry due to frames not
    being supplied fast enough.

    In this example, since we are setting `timeout` to `UINT64_MAX`, we will
    never get this return value.

2.  Setting this makes `pl_queue_update` block indefinitely until sufficiently
    many frames have been pushed into the `pl_queue` from our separate
    decoding thread.

### Deinterlacing

The frame queue also vastly simplifies the process of performing
motion-adaptive temporal deinterlacing, by automatically linking together
adjacent fields/frames. To take advantage of this, all you need to do is set
the appropriate field (`pl_source_frame.first_frame`), as well as enabling
[deinterlacing
parameters](https://code.videolan.org/videolan/libplacebo/-/blob/master/src/include/libplacebo/renderer.h#L186).
