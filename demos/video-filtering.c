/* Presented are two hypothetical scenarios of how one might use libplacebo
 * as something like an FFmpeg or mpv video filter. We examine two example
 * APIs (loosely modeled after real video filtering APIs) and how each style
 * would like to use libplacebo.
 *
 * For sake of a simple example, let's assume this is a debanding filter.
 * For those of you too lazy to compile/run this file but still want to see
 * results, these are from my machine (RX 560 with mesa/amdgpu git 2018-09-27):
 *
 * RADV:
 *   api1: 10000 frames in 21.113829 s => 2.111383 ms/frame (473.62 fps)
 *   api2: 10000 frames in 14.314918 s => 1.431492 ms/frame (698.57 fps)
 *
 * AMDVLK:
 *   api1: 10000 frames in 17.027370 s => 1.702737 ms/frame (587.29 fps)
 *   api2: api2: 10000 frames in 8.882183 s => 0.888218 ms/frame (1125.85 fps)
 *
 * You can see that AMDVLK is much better at doing texture streaming than
 * RADV - this is because as of writing RADV still does not support
 * asynchronous texture queues / DMA engine transfers. If we disable the
 * `async_transfer` option with AMDVLK we get this:
 *
 *   api1: 10000 frames in 20.377153 s => 2.037715 ms/frame (490.75 fps)
 *   api2: 10000 frames in 14.874546 s => 1.487455 ms/frame (672.29 fps)
 *
 * License: CC0 / Public Domain
 */

#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include <libplacebo/dispatch.h>
#include <libplacebo/shaders/sampling.h>
#include <libplacebo/utils/upload.h>
#include <libplacebo/vulkan.h>

///////////////////////
/// API definitions ///
///////////////////////

// Stuff that would be common to each API

void *init();
void uninit(void *priv);

struct format {
    // For simplicity let's make a few assumptions here, since configuring the
    // texture format is not the point of this example. (In practice you can
    // go nuts with the `utils/upload.h` helpers)
    //
    // - All formats contain unsigned integers only
    // - All components have the same size in bits
    // - All components are in the "canonical" order
    // - All formats have power of two sizes only (2 or 4 components, not 3)
    // - All plane strides are a multiple of the pixel size
    int num_comps;
    int bitdepth;
};

struct plane {
    int subx, suby; // subsampling shift
    struct format fmt;
    size_t stride;
    void *data;
};

#define MAX_PLANES 4

struct image {
    int width, height;
    int num_planes;
    struct plane planes[MAX_PLANES];

    // For API #2, the associated mapped buffer (if any)
    struct api2_buf *associated_buf;
};


// Example API design #1: synchronous, blocking, double-copy (bad!)
//
// In this API, `api1_filter` must immediately return with the new data.
// This prevents parallelism on the GPU and should be avoided if possible,
// but sometimes that's what you have to work with. So this is what it
// would look like.
//
// Also, let's assume this API design reconfigures the filter chain (using
// a blank `proxy` image every time the image format or dimensions change,
// and doesn't expect us to fail due to format mismatches or resource
// exhaustion afterwards.

bool api1_reconfig(void *priv, const struct image *proxy);
bool api1_filter(void *priv, struct image *dst, struct image *src);


// Example API design #2: asynchronous, streaming, queued, zero-copy (good!)
//
// In this API, `api2_process` will run by the calling code every so often
// (e.g. when new data is available or expected). This function has access
// to non-blocking functions `get_image` and `put_image` that interface
// with the video filtering engine's internal queueing system.
//
// This API is also designed to feed multiple frames ahead of time, i.e.
// it will feed us as many frames as it can while we're still returning
// `API2_WANT_MORE`. To drain the filter chain, it would continue running
// the process function until `API2_HAVE_MORE` is no longer present
// in the output.
//
// This API is also designed to do zero-copy where possible. When it wants
// to create a data buffer of a given size, it will call our function
// `api2_alloc` which will return a buffer that we can process directly.
// We can use this to do zero-copy uploading to the GPU, by creating
// host-visible persistently mapped buffers. In order to prevent the video
// filtering system from re-using our buffers while copies are happening, we
// use special functions `image_lock` and `image_unlock` to increase a
// refcount on the image's backing storage. (As is typical of such APIs)
//
// Finally, this API is designed to be fully dynamic: The image parameters
// could change at any time, and we must be equipped to handle that.

enum api2_status {
    // Negative values are used to signal error conditions
    API2_ERR_FMT = -2,          // incompatible / unsupported format
    API2_ERR_UNKNOWN = -1,      // some other error happened
    API2_OK = 0,                // no error, no status - everything's good

    // Positive values represent a mask of status conditions
    API2_WANT_MORE = (1 << 0),  // we want more frames, please feed some more!
    API2_HAVE_MORE = (1 << 1),  // we have more frames but they're not ready
};

enum api2_status api2_process(void *priv);

// Functions for creating persistently mapped buffers
struct api2_buf {
    void *data;
    size_t size;
    void *priv;
};

bool api2_alloc(void *priv, size_t size, struct api2_buf *out);
void api2_free(void *priv, const struct api2_buf *buf);

// These functions are provided by the API. The exact details of how images
// are enqueued, dequeued and locked are not really important here, so just
// do something unrealistic but simple to demonstrate with.
struct image *get_image();
void put_image(struct image img);
void image_lock(struct image *img);
void image_unlock(struct image *img);


/////////////////////////////////
/// libplacebo implementation ///
/////////////////////////////////


// For API #2:
#define PARALLELISM 8

struct entry {
    const struct pl_buf *buf; // to stream the download
    const struct pl_tex *tex_in[MAX_PLANES];
    const struct pl_tex *tex_out[MAX_PLANES];
    struct image image;

    // For entries that are associated with a held image, so we can unlock them
    // as soon as possible
    struct image *held_image;
    const struct pl_buf *held_buf;
};

// For both APIs:
struct priv {
    struct pl_context *ctx;
    const struct pl_vulkan *vk;
    const struct pl_gpu *gpu;
    struct pl_dispatch *dp;

    // API #1: A simple pair of input and output textures
    const struct pl_tex *tex_in[MAX_PLANES];
    const struct pl_tex *tex_out[MAX_PLANES];

    // API #2: A ring buffer of textures/buffers for streaming
    int idx_in;  // points the next free entry
    int idx_out; // points to the first entry still in progress
    struct entry entries[PARALLELISM];
};

void *init() {
    struct priv *p = malloc(sizeof(struct priv));
    if (!p)
        return NULL;

    *p = (struct priv) {0};
    p->ctx = pl_context_create(PL_API_VER, NULL);
    if (!p->ctx) {
        fprintf(stderr, "Failed initializing libplacebo\n");
        goto error;
    }

    p->vk = pl_vulkan_create(p->ctx, &(struct pl_vulkan_params) {
        // Note: This is for API #2. In API #1 you could just pass params=NULL
        // and it wouldn't really matter. (In fact, it would even be faster)
        .async_transfer = true,
        .async_compute = true,
        .queue_count = PARALLELISM,
    });

    if (!p->vk) {
        fprintf(stderr, "Failed creating vulkan context\n");
        goto error;
    }

    // Give this a shorter name for convenience
    p->gpu = p->vk->gpu;

    p->dp = pl_dispatch_create(p->ctx, p->gpu);
    if (!p->vk) {
        fprintf(stderr, "Failed creating shader dispatch object\n");
        goto error;
    }

    return p;

error:
    uninit(p);
    return NULL;
}

void uninit(void *priv)
{
    struct priv *p = priv;

    // API #1
    for (int i = 0; i < MAX_PLANES; i++) {
        pl_tex_destroy(p->gpu, &p->tex_in[i]);
        pl_tex_destroy(p->gpu, &p->tex_out[i]);
    }

    // API #2
    for (int i = 0; i < PARALLELISM; i++) {
        pl_buf_destroy(p->gpu, &p->entries[i].buf);
        for (int j = 0; j < MAX_PLANES; j++) {
            pl_tex_destroy(p->gpu, &p->entries[i].tex_in[j]);
            pl_tex_destroy(p->gpu, &p->entries[i].tex_out[j]);
        }
        if (p->entries[i].held_image)
            image_unlock(p->entries[i].held_image);
    }

    pl_dispatch_destroy(&p->dp);
    pl_vulkan_destroy(&p->vk);
    pl_context_destroy(&p->ctx);

    free(p);
}

// Helper function to set up the `pl_plane_data` struct from the image params
void setup_plane_data(const struct image *img,
                      struct pl_plane_data out[MAX_PLANES])
{
    for (int i = 0; i < img->num_planes; i++) {
        const struct plane *plane = &img->planes[i];

        out[i] = (struct pl_plane_data) {
            .type = PL_FMT_UNORM,
            .width = img->width >> plane->subx,
            .height = img->height >> plane->suby,
            .pixel_stride = plane->fmt.num_comps * plane->fmt.bitdepth / 8,
            .row_stride = plane->stride,
            .pixels = plane->data,
        };

        for (int c = 0; c < plane->fmt.num_comps; c++) {
            out[i].component_size[c] = plane->fmt.bitdepth;
            out[i].component_pad[c] = 0;
            out[i].component_map[c] = c;
        }
    }
}

// API #1 implementation:
//
// In this design, we will create all GPU resources inside `reconfig`, based on
// the texture format configured from the proxy image. This will avoid failing
// later on due to e.g. resource exhaustion or texture format mismatch, and
// thereby falls within the intended semantics of this style of API.

bool api1_reconfig(void *priv, const struct image *proxy)
{
    struct priv *p = priv;
    struct pl_plane_data data[MAX_PLANES];
    setup_plane_data(proxy, data);

    for (int i = 0; i < proxy->num_planes; i++) {
        const struct plane *plane = &proxy->planes[i];
        const struct pl_fmt *fmt = pl_plane_find_fmt(p->gpu, NULL, &data[i]);
        if (!fmt) {
            fprintf(stderr, "Failed configuring filter: no good texture format!\n");
            return false;
        }

        bool ok = true;
        ok &= pl_tex_recreate(p->gpu, &p->tex_in[i], &(struct pl_tex_params) {
            .w = data[i].width,
            .h = data[i].height,
            .format = fmt,
            .sampleable = true,
            .host_writable = true,
            .sample_mode = PL_TEX_SAMPLE_LINEAR,
        });

        ok &= pl_tex_recreate(p->gpu, &p->tex_out[i], &(struct pl_tex_params) {
            .w = data[i].width,
            .h = data[i].height,
            .format = fmt,
            .renderable = true,
            .host_readable = true,
        });

        if (!ok) {
            fprintf(stderr, "Failed creating GPU textures!\n");
            return false;
        }
    }

    return true;
}


// API #1 implementation:

bool api1_filter(void *priv, struct image *dst, struct image *src)
{
    struct priv *p = priv;
    struct pl_plane_data data[MAX_PLANES];
    setup_plane_data(src, data);

    // Upload planes
    for (int i = 0; i < src->num_planes; i++) {
        bool ok = pl_tex_upload(p->gpu, &(struct pl_tex_transfer_params) {
            .tex = p->tex_in[i],
            .stride_w = data[i].row_stride / data[i].pixel_stride,
            .ptr = src->planes[i].data,
        });

        if (!ok) {
            fprintf(stderr, "Failed uploading data to the GPU!\n");
            return false;
        }
    }

    // Process planes
    for (int i = 0; i < src->num_planes; i++) {
        struct pl_shader *sh = pl_dispatch_begin(p->dp);
        pl_shader_deband(sh, &(struct pl_sample_src){ .tex = p->tex_in[i] }, NULL);
        if (!pl_dispatch_finish(p->dp, &sh, p->tex_out[i], NULL, NULL)) {
            fprintf(stderr, "Failed processing planes!\n");
            return false;
        }
    }

    // Download planes
    for (int i = 0; i < src->num_planes; i++) {
        bool ok = pl_tex_download(p->gpu, &(struct pl_tex_transfer_params) {
            .tex = p->tex_out[i],
            .stride_w = dst->planes[i].stride / data[i].pixel_stride,
            .ptr = dst->planes[i].data,
        });

        if (!ok) {
            fprintf(stderr, "Failed downloading data from the GPU!\n");
            return false;
        }
    }

    return true;
}


// API #2 implementation:

// Align up to the nearest multiple of a power of two
#define ALIGN2(x, align) (((x) + (align) - 1) & ~((align) - 1))

static enum api2_status submit_work(struct priv *p, struct entry *e,
                                    struct image *img)
{
    // If the image comes from a mapped buffer, we have to take a lock
    // while our upload is in progress
    if (img->associated_buf) {
        assert(!e->held_image);
        image_lock(img);
        e->held_image = img;
        e->held_buf = img->associated_buf->priv;
    }

    // Upload this image's data
    struct pl_plane_data data[MAX_PLANES];
    setup_plane_data(img, data);

    for (int i = 0; i < img->num_planes; i++) {
        const struct pl_fmt *fmt = pl_plane_find_fmt(p->gpu, NULL, &data[i]);
        if (!fmt)
            return API2_ERR_FMT;

        if (!pl_upload_plane(p->gpu, NULL, &e->tex_in[i], &data[i]))
            return API2_ERR_UNKNOWN;

        // Re-create the target FBO as well with this format if necessary
        bool ok = pl_tex_recreate(p->gpu, &e->tex_out[i], &(struct pl_tex_params) {
            .w = data[i].width,
            .h = data[i].height,
            .format = fmt,
            .renderable = true,
            .host_readable = true,
        });
        if (!ok)
            return API2_ERR_UNKNOWN;
    }

    // Dispatch the work for this image
    for (int i = 0; i < img->num_planes; i++) {
        struct pl_shader *sh = pl_dispatch_begin(p->dp);
        pl_shader_deband(sh, &(struct pl_sample_src){ .tex = e->tex_in[i] }, NULL);
        if (!pl_dispatch_finish(p->dp, &sh, e->tex_out[i], NULL, NULL))
            return API2_ERR_UNKNOWN;
    }

    // Set up the resulting `struct image` that will hold our target
    // data. We just copy the format etc. from the source image
    memcpy(&e->image, img, sizeof(struct image));

    size_t offset[MAX_PLANES], stride[MAX_PLANES], total_size = 0;
    for (int i = 0; i < img->num_planes; i++) {
        // For performance, we want to make sure we align the stride
        // to a multiple of the GPU's preferred texture transfer stride
        // (This is entirely optional)
        stride[i] = ALIGN2(img->planes[i].stride,
                           p->gpu->limits.align_tex_xfer_stride);
        int height = img->height >> img->planes[i].suby;

        // Round up the offset to the nearest multiple of the optimal
        // transfer alignment. (This is also entirely optional)
        offset[i] = ALIGN2(total_size, p->gpu->limits.align_tex_xfer_offset);
        total_size = offset[i] + stride[i] * height;
    }

    // Dispatch the asynchronous download into a mapped buffer
    bool ok = pl_buf_recreate(p->gpu, &e->buf, &(struct pl_buf_params) {
        .type = PL_BUF_TEX_TRANSFER,
        .size = total_size,
        .host_mapped = true,
    });
    if (!ok)
        return API2_ERR_UNKNOWN;

    for (int i = 0; i < img->num_planes; i++) {
        ok = pl_tex_download(p->gpu, &(struct pl_tex_transfer_params) {
            .tex = e->tex_out[i],
            .stride_w = stride[i] / data[i].pixel_stride,
            .buf = e->buf,
            .buf_offset = offset[i],
        });
        if (!ok)
            return API2_ERR_UNKNOWN;

        // Update the output fields
        e->image.planes[i].data = e->buf->data + offset[i];
        e->image.planes[i].stride = stride[i];
    }

    return API2_OK;
}

enum api2_status api2_process(void *priv)
{
    struct priv *p = priv;
    enum api2_status ret = 0;

    // Opportunistically release any held images. We do this across the ring
    // buffer, rather than doing this as part of the following loop, because
    // we want to release images ahead-of-time (no FIFO constraints)
    for (int i = 0; i < PARALLELISM; i++) {
        struct entry *e = &p->entries[i];
        if (e->held_image && !pl_buf_poll(p->gpu, e->held_buf, 0)) {
            // upload buffer is no longer in use, release it
            image_unlock(e->held_image);
            e->held_image = NULL;
            e->held_buf = NULL;
        }
    }

    // Poll the status of existing entries and dequeue the ones that are done
    while (p->idx_out != p->idx_in) {
        struct entry *e = &p->entries[p->idx_out];
        if (pl_buf_poll(p->gpu, e->buf, 0))
            break;

        // download buffer is no longer busy, dequeue the frame
        put_image(e->image);
        p->idx_out = (p->idx_out + 1) % PARALLELISM;
    }

    // Fill up the queue with more work
    int last_free_idx = (p->idx_out ? p->idx_out : PARALLELISM) - 1;
    while (p->idx_in != last_free_idx) {
        struct image *img = get_image();
        if (!img) {
            ret |= API2_WANT_MORE;
            break;
        }

        enum api2_status err = submit_work(p, &p->entries[p->idx_in], img);
        if (err < 0)
            return err;

        p->idx_in = (p->idx_in + 1) % PARALLELISM;
    }

    if (p->idx_out != p->idx_in)
        ret |= API2_HAVE_MORE;

    // Note: It might be faster to call this at the end of `submit_work`
    // instead of at the end of this function. Specific testing recommended.
    pl_gpu_flush(p->gpu);
    return ret;
}

bool api2_alloc(void *priv, size_t size, struct api2_buf *out)
{
    struct priv *p = priv;

    const struct pl_buf *buf = pl_buf_create(p->gpu, &(struct pl_buf_params) {
        .type = PL_BUF_TEX_TRANSFER,
        .size = size,
        .host_mapped = true,
    });

    if (!buf)
        return false;

    *out = (struct api2_buf) {
        .data = buf->data,
        .size = size,
        .priv = (void *) buf,
    };
    return true;
}

void api2_free(void *priv, const struct api2_buf *buf)
{
    struct priv *p = priv;
    const struct pl_buf *plbuf = buf->priv;
    pl_buf_destroy(p->gpu, &plbuf);
}


////////////////////////////////////
/// Proof of Concept / Benchmark ///
////////////////////////////////////

#define FRAMES 10000

// Let's say we're processing a 1920x1080 4:2:0 8-bit NV12 video, arbitrarily
// with a stride of 2048 pixels per row
#define TEXELSZ sizeof(uint8_t)
#define WIDTH   1920
#define HEIGHT  1080
#define STRIDE  (2048 * TEXELSZ)
// Subsampled planes
#define SWIDTH  (WIDTH >> 1)
#define SHEIGHT (HEIGHT >> 1)
#define SSTRIDE (STRIDE >> 1)
// Plane offsets / sizes
#define SIZE    (HEIGHT * STRIDE * TEXELSZ)
#define SSIZE   (SHEIGHT * SSTRIDE * TEXELSZ)

#define SIZE0   (HEIGHT * STRIDE * TEXELSZ)
#define SIZE1   (2 * SHEIGHT * SSTRIDE * TEXELSZ)
#define OFFSET0 0
#define OFFSET1 SIZE0
#define BUFSIZE (OFFSET1 + SIZE1)

// Skeleton of an example image
static const struct image example_image = {
    .width = WIDTH,
    .height = HEIGHT,
    .num_planes = 2,
    .planes = {
        {
            .subx = 0,
            .suby = 0,
            .stride = STRIDE,
            .fmt = {
                .num_comps = 1,
                .bitdepth = 8 * TEXELSZ,
            },
        }, {
            .subx = 1,
            .suby = 1,
            .stride = SSTRIDE * 2,
            .fmt = {
                .num_comps = 2,
                .bitdepth = 8 * TEXELSZ,
            },
        },
    },
};

// API #1: Nice and simple (but slow)
void api1_example()
{
    void *vf = init();
    if (!vf)
        return;

    if (!api1_reconfig(vf, &example_image)) {
        fprintf(stderr, "api1: Failed configuring video filter!\n");
        return;
    }

    // Allocate two buffers to hold the example data, and fill the source
    // buffer arbitrarily with a "simple" pattern. (Decoding the data into
    // the buffer is not meant to be part of this benchmark)
    uint8_t *srcbuf = malloc(BUFSIZE),
            *dstbuf = malloc(BUFSIZE);
    if (!srcbuf || !dstbuf)
        goto done;

    for (int i = 0; i < BUFSIZE; i++)
        srcbuf[i] = i;

    struct image src = example_image, dst = example_image;
    src.planes[0].data = srcbuf + OFFSET0;
    src.planes[1].data = srcbuf + OFFSET1;
    dst.planes[0].data = dstbuf + OFFSET0;
    dst.planes[1].data = dstbuf + OFFSET1;

    struct timeval start = {0}, stop = {0};
    gettimeofday(&start, NULL);

    // Process this dummy frame a bunch of times
    unsigned frames = 0;
    for (frames = 0; frames < FRAMES; frames++) {
        if (!api1_filter(vf, &dst, &src)) {
            fprintf(stderr, "api1: Failed filtering frame... aborting\n");
            break;
        }
    }

    gettimeofday(&stop, NULL);
    float secs = (float) (stop.tv_sec - start.tv_sec) +
                 1e-6 * (stop.tv_usec - start.tv_usec);

    printf("api1: %4u frames in %1.6f s => %2.6f ms/frame (%5.2f fps)\n",
           frames, secs, 1000 * secs / frames, frames / secs);

done:
    if (srcbuf)
        free(srcbuf);
    if (dstbuf)
        free(dstbuf);
    uninit(vf);
}


// API #2: Pretend we have some fancy pool of images.
#define POOLSIZE 32

static struct api2_buf buffers[POOLSIZE] = {0};
static struct image images[POOLSIZE] = {0};
static int refcount[POOLSIZE] = {0};
static unsigned api2_frames_in = 0;
static unsigned api2_frames_out = 0;

void api2_example()
{
    void *vf = init();
    if (!vf)
        return;

    // Set up a bunch of dummy images
    for (int i = 0; i < POOLSIZE; i++) {
        uint8_t *data;
        images[i] = example_image;
        if (api2_alloc(vf, BUFSIZE, &buffers[i])) {
            data = buffers[i].data;
            images[i].associated_buf = &buffers[i];
        } else {
            // Fall back in case mapped buffers are unsupported
            data = malloc(BUFSIZE);
        }
        // Fill with some "data" (like in API #1)
        for (int i = 0; i < BUFSIZE; i++)
            data[i] = i;
        images[i].planes[0].data = data + OFFSET0;
        images[i].planes[1].data = data + OFFSET1;
    }

    struct timeval start = {0}, stop = {0};
    gettimeofday(&start, NULL);

    // Just keep driving the event loop regardless of the return status
    // until we reach the critical number of frames. (Good enough for this PoC)
    while (api2_frames_out < FRAMES) {
        enum api2_status ret = api2_process(vf);
        if (ret < 0) {
            fprintf(stderr, "api2: Failed processing... aborting\n");
            break;
        }

        // Sleep a short time (100us) to prevent busy waiting the CPU
        nanosleep(&(struct timespec) { .tv_nsec = 100000 }, NULL);
    }

    gettimeofday(&stop, NULL);
    float secs = (float) (stop.tv_sec - start.tv_sec) +
                 1e-6 * (stop.tv_usec - start.tv_usec);

    printf("api2: %4u frames in %1.6f s => %2.6f ms/frame (%5.2f fps)\n",
           api2_frames_out, secs, 1000 * secs / api2_frames_out,
           api2_frames_out / secs);

done:

    for (int i = 0; i < POOLSIZE; i++) {
        if (images[i].associated_buf) {
            api2_free(vf, images[i].associated_buf);
        } else {
            // This is what we originally malloc'd
            free(images[i].planes[0].data);
        }
    }

    uninit(vf);
}

struct image *get_image()
{
    if (api2_frames_in == FRAMES)
        return NULL; // simulate EOF, to avoid queueing up "extra" work

    // if we can find a free (unlocked) image, give it that
    for (int i = 0; i < POOLSIZE; i++) {
        if (refcount[i] == 0) {
            api2_frames_in++;
            return &images[i];
        }
    }

    return NULL; // no free image available
}

void put_image(struct image img)
{
    api2_frames_out++;
}

void image_lock(struct image *img)
{
    int index = img - images; // cheat, for lack of having actual image management
    refcount[index]++;
}

void image_unlock(struct image *img)
{
    int index = img - images;
    refcount[index]--;
}

void main()
{
    printf("Running benchmarks...\n");
    api1_example();
    api2_example();
}
