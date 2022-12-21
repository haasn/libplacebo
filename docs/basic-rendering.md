# Basic rendering example

We will demonstrate the basics of the libplacebo GPU output API with a worked
example. The goal is to show a simple color on screen.

## Creating a `pl_log`

Almost all major entry-points into libplacebo require providing a log
callback (or `NULL` to disable logging). This is abstracted into the `pl_log`
object type, which we can create with
`pl_log_create`:

``` c linenums="1"
#include <libplacebo/log.h>

pl_log pllog;

int main()
{
    pllog = pl_log_create(PL_API_VER, pl_log_params(
        .log_cb = pl_log_color,
        .log_level = PL_LOG_INFO,
    ));

    // ...

    pl_log_destroy(&pllog);
    return 0;
}
```

!!! note "Compiling"

    You can compile this example with:

    ``` bash
    $ gcc example.c -o example `pkg-config --cflags --libs libplacebo`
    ```

The parameter `PL_API_VER` has no special significance and is merely included
for historical reasons. Aside from that, this snippet introduces a number of
core concepts of the libplacebo API:

### Parameter structs

For extensibility, almost all libplacebo calls take a pointer to a `const
struct pl_*_params`, into which all extensible parameters go. For convenience,
libplacebo provides macros which create anonymous params structs on the stack
(and also fill in default parameters). Note that this only works for C99 and
above, users of C89 and C++ must initialize parameter structs manually.

Under the hood, `pl_log_params(...)` just translates to `&((struct
pl_log_params) { /* default params */, ... })`. This style of API allows
libplacebo to effectively simulate optional named parameters.

!!! note "On default parameters"

    Wherever possible, parameters are designed in such a way that `{0}` gives
    you a minimal parameter structure, with default behavior and no optional
    features enabled. This is done for forwards compatibility - as new
    features are introduced, old struct initializers will simply opt out of
    them.

### Destructors

All libplacebo objects must be destroyed manually using the corresponding
`pl_*_destroy` call, which takes a pointer to the variable the object is
stored in. The resulting variable is written to `NULL`. This helps prevent
use-after-free bugs.

!!! note "NULL"

    As a general rule, all libplacebo destructors are safe to call on
    variables containing `NULL`. So, users need not explicitly `NULL`-test
    before calling destructors on variables.

## Creating a window

While libplacebo can work in isolation, to render images offline, for the sake
of this guide we want to provide something graphical on-screen. As such, we
need to create some sort of window. Libplacebo provides no built-in mechanism
for this, it assumes the API user will already have a windowing system
in-place.

Complete examples (based on GLFW and SDL) can be found [in the libplacebo
demos](https://code.videolan.org/videolan/libplacebo/-/tree/master/demos). But
for now, we will focus on getting a very simple window on-screen using GLFW:

``` c linenums="1" hl_lines="3 5 6 7 9 17 18 20 21 22 24 25 26 28 29"
// ...

#include <GLFW/glfw3.h>

const char * const title = "libplacebo demo";
int width = 800;
int height = 600;

GLFWwindow *window;

int main()
{
    pllog = pl_log_create(PL_API_VER, pl_log_params(
        .log_level = PL_LOG_INFO,
    ));

    if (!glfwInit())
        return 1;

    window = glfwCreateWindow(width, height, title, NULL, NULL);
    if (!window)
        return 1;

    while (!glfwWindowShouldClose(window)) {
        glfwWaitEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    pl_log_destroy(&pllog);
    return 0;
}
```

!!! note "Compiling"

    We now also need to include the glfw3 library to compile this example.

    ``` bash
    $ gcc example.c -o example `pkg-config --cflags --libs glfw3 libplacebo`
    ```

## Creating the `pl_gpu`

All GPU operations are abstracted into an internal `pl_gpu` object, which
serves as the primary entry-point to any sort of GPU interaction. This object
cannot be created directly, but must be obtained from some graphical API:
currently there are Vulkan, OpenGL or D3D11. A `pl_gpu` can be accessed from
an API-specific object like `pl_vulkan`, `pl_opengl` and `pl_d3d11`.

In this guide, for simplicity, we will be using OpenGL, simply because that's
what GLFW initializes by default.

``` c linenums="1" hl_lines="3 5-6 15-23 29 36-45"
// ...

pl_opengl opengl;

static bool make_current(void *priv);
static void release_current(void *priv);

int main()
{
    // ...
    window = glfwCreateWindow(width, height, title, NULL, NULL);
    if (!window)
        return 1;

    opengl = pl_opengl_create(pllog, pl_opengl_params(
        .get_proc_addr      = glfwGetProcAddress,
        .allow_software     = true,         // allow software rasterers
        .debug              = true,         // enable error reporting
        .make_current       = make_current, // (1)
        .release_current    = release_current,
    ));
    if (!opengl)
        return 2;

    while (!glfwWindowShouldClose(window)) {
        glfwWaitEvents();
    }

    pl_opengl_destroy(&opengl);
    glfwDestroyWindow(window);
    glfwTerminate();
    pl_log_destroy(&pllog);
    return 0;
}

static bool make_current(void *priv)
{
    glfwMakeContextCurrent(window);
    return true;
}

static void release_current(void *priv)
{
    glfwMakeContextCurrent(NULL);
}
```

1.  Setting this allows the resulting `pl_gpu` to be thread-safe, which
    enables asynchronous transfers to be used. The alternative is to simply
    call `glfwMakeContextCurrent` once after creating the window.

    This method of making the context current is generally preferred,
    however, so we've demonstrated it here for completeness' sake.

## Creating a swapchain

All access to window-based rendering commands are abstracted into an object
known as a "swapchain" (from Vulkan terminology), including the default
backbuffers on D3D11 and OpenGL. If we want to present something to screen,
we need to first create a `pl_swapchain`.

We can use this swapchain to perform the equivalent of `gl*SwapBuffers`:

``` c linenums="1" hl_lines="2 4-9 17-22 24-27 30-31 34"
// ...
pl_swapchain swchain;

static void resize_cb(GLFWwindow *win, int new_w, int new_h)
{
    width  = new_w;
    height = new_h;
    pl_swapchain_resize(swchain, &width, &height);
}

int main()
{
    // ...
    if (!opengl)
        return 2;

    swchain = pl_opengl_create_swapchain(opengl, pl_opengl_swapchain_params(
        .swap_buffers   = (void (*)(void *)) glfwSwapBuffers,
        .priv           = window,
    ));
    if (!swchain)
        return 2;

    // (2)
    if (!pl_swapchain_resize(swchain, &width, &height))
        return 2;
    glfwSetFramebufferSizeCallback(window, resize_cb);

    while (!glfwWindowShouldClose(window)) {
        pl_swapchain_swap_buffers(swchain);
        glfwPollEvents(); // (1)
    }

    pl_swapchain_destroy(&swchain);
    pl_opengl_destroy(&opengl);
    glfwDestroyWindow(window);
    glfwTerminate();
    pl_log_destroy(&pllog);
    return 0;
}
```

1.  We change this from `glfwWaitEvents` to `glfwPollEvents` because
    we now want to re-run our main loop once per vsync, rather than only when
    new events arrive.  The `pl_swapchain_swap_buffers` call will ensure
    that this does not execute too quickly.

2.  The swapchain needs to be resized to fit the size of the window, which in
    GLFW is handled by listening to a callback. In addition to setting this
    callback, we also need to inform the swapchain of the initial window size.

    Note that the `pl_swapchain_resize` function handles both resize requests
    and size queries - hence, the actual swapchain size is returned back to
    the passed variables.

## Getting pixels on the screen

With a swapchain in hand, we're now equipped to start drawing pixels to the
screen:

``` c linenums="1" hl_lines="3-8 15-20"
// ...

static void render_frame(struct pl_swapchain_frame frame)
{
    pl_gpu gpu = opengl->gpu;

    pl_tex_clear(gpu, frame.fbo, (float[4]){ 1.0, 0.5, 0.0, 1.0 });
}

int main()
{
    // ...

    while (!glfwWindowShouldClose(window)) {
        struct pl_swapchain_frame frame;
        while (!pl_swapchain_start_frame(swchain, &frame))
            glfwWaitEvents(); // (1)
        render_frame(frame);
        if (!pl_swapchain_submit_frame(swchain))
            break; // (2)

        pl_swapchain_swap_buffers(swchain);
        glfwPollEvents();
    }

    // ...
}
```

1.  If `pl_swapchain_start_frame` fails, it typically means the window is
    hidden, minimized or blocked. This is not a fatal condition, and as such
    we simply want to process window events until we can resume rendering.

2.  If `pl_swapchain_submit_frame` fails, it typically means the window has
    been lost, and further rendering commands are not expected to succeed.
    As such, in this case, we simply terminate the example program.

Our main render loop has changed into a combination of
`pl_swapchain_start_frame`, rendering, and `pl_swapchain_submit_frame`. To
start with, we simply use the `pl_tex_clear` function to blit a constant
orange color to the framebuffer.

### Interlude: Rendering commands

The previous code snippet represented our first foray into the `pl_gpu` API.
For more detail on this API, see the [GPU API](#TODO) section. But as a
general rule of thumb, all `pl_gpu`-level operations are thread safe,
asynchronous (except when returning something to the CPU), and internally
refcounted (so you can destroy all objects as soon as you no longer need the
reference).

In the example loop, `pl_swapchain_swap_buffers` is the only operation that
actually flushes commands to the GPU. You can force an early flush with
`pl_gpu_flush()` or `pl_gpu_finish()`, but other than that, commands will
"queue" internally and complete asynchronously at some unknown point in time,
until forward progress is needed (e.g. `pl_tex_download`).

## Conclusion

We have demonstrated how to create a window, how to initialize the libplacebo
API, create a GPU instance based on OpenGL, and how to write a basic rendering
loop that blits a single color to the framebuffer.

Here is a complete transcript of the example we built in this section:

??? example "Basic rendering"
    ``` c linenums="1"
    #include <GLFW/glfw3.h>
    
    #include <libplacebo/log.h>
    #include <libplacebo/opengl.h>
    #include <libplacebo/gpu.h>
    
    const char * const title = "libplacebo demo";
    int width = 800;
    int height = 600;
    
    GLFWwindow *window;
    
    pl_log pllog;
    pl_opengl opengl;
    pl_swapchain swchain;
    
    static bool make_current(void *priv);
    static void release_current(void *priv);
    
    static void resize_cb(GLFWwindow *win, int new_w, int new_h)
    {
        width  = new_w;
        height = new_h;
        pl_swapchain_resize(swchain, &width, &height);
    }
    
    static void render_frame(struct pl_swapchain_frame frame)
    {
        pl_gpu gpu = opengl->gpu;
    
        pl_tex_clear(gpu, frame.fbo, (float[4]){ 1.0, 0.5, 0.0, 1.0 });
    }
    
    int main()
    {
        pllog = pl_log_create(PL_API_VER, pl_log_params(
            .log_cb = pl_log_color,
            .log_level = PL_LOG_INFO,
        ));
    
        if (!glfwInit())
            return 1;
    
        window = glfwCreateWindow(width, height, title, NULL, NULL);
        if (!window)
            return 1;
    
        opengl = pl_opengl_create(pllog, pl_opengl_params(
            .get_proc_addr      = glfwGetProcAddress,
            .allow_software     = true,         // allow software rasterers
            .debug              = true,         // enable error reporting
            .make_current       = make_current,
            .release_current    = release_current,
        ));
    
        swchain = pl_opengl_create_swapchain(opengl, pl_opengl_swapchain_params(
            .swap_buffers   = (void (*)(void *)) glfwSwapBuffers,
            .priv           = window,
        ));
        if (!swchain)
            return 2;
    
        if (!pl_swapchain_resize(swchain, &width, &height))
            return 2;
        glfwSetFramebufferSizeCallback(window, resize_cb);
    
        while (!glfwWindowShouldClose(window)) {
            struct pl_swapchain_frame frame;
            while (!pl_swapchain_start_frame(swchain, &frame))
                glfwWaitEvents();
            render_frame(frame);
            if (!pl_swapchain_submit_frame(swchain))
                break;
    
            pl_swapchain_swap_buffers(swchain);
            glfwPollEvents();
        }
    
        pl_swapchain_destroy(&swchain);
        pl_opengl_destroy(&opengl);
        glfwDestroyWindow(window);
        glfwTerminate();
        pl_log_destroy(&pllog);
        return 0;
    }
    
    static bool make_current(void *priv)
    {
        glfwMakeContextCurrent(window);
        return true;
    }
    
    static void release_current(void *priv)
    {
        glfwMakeContextCurrent(NULL);
    }
    ```
