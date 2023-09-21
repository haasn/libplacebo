/* GPU->GPU transfer benchmarks. Requires some manual setup.
 *
 * License: CC0 / Public Domain
 */

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <libplacebo/gpu.h>
#include <libplacebo/vulkan.h>

#include "pl_clock.h"

#define ALIGN2(x, align) (((x) + (align) - 1) & ~((align) - 1))

enum {
    // Image configuration
    NUM_TEX = 16,
    WIDTH   = 1920,
    HEIGHT  = 1080,
    DEPTH   = 16,
    COMPS   = 1,

    // Queue configuration
    NUM_QUEUES = 16,
    ASYNC_TX   = 1,
    ASYNC_COMP = 1,

    // Buffer configuration
    PTR_ALIGN    = 4096,
    PIXEL_PITCH  = DEPTH / 8,
    ROW_PITCH    = ALIGN2(WIDTH * PIXEL_PITCH, 256),
    IMAGE_SIZE   = ROW_PITCH * HEIGHT,
    BUFFER_SIZE  = IMAGE_SIZE + PTR_ALIGN - 1,

    // Test configuration
    TEST_MS    = 1500,
    WARMUP_MS  = 500,
    POLL_FREQ  = 10,
};

static uint8_t* page_align(uint8_t *data)
{
    return (uint8_t *) ALIGN2((uintptr_t) data, PTR_ALIGN);
}

enum mem_owner {
    CPU,
    SRC,
    DST,
    NUM_MEM_OWNERS,
};

enum mem_type {
    RAM,
    GPU,
    NUM_MEM_TYPES,
};

// This is attached to every `pl_tex.params.user_data`
struct buffers {
    pl_gpu gpu;
    pl_buf buf[NUM_MEM_TYPES];
    pl_buf exported[NUM_MEM_TYPES];
    pl_buf imported[NUM_MEM_TYPES];
    struct pl_tex_transfer_params async;
};

static struct buffers *alloc_buffers(pl_gpu gpu)
{
    struct buffers *buffers = malloc(sizeof(*buffers));
    *buffers = (struct buffers) { .gpu = gpu };

    for (enum mem_type type = 0; type < NUM_MEM_TYPES; type++) {
        buffers->buf[type] = pl_buf_create(gpu, pl_buf_params(
            .size          = BUFFER_SIZE,
            .memory_type   = type == RAM ? PL_BUF_MEM_HOST : PL_BUF_MEM_DEVICE,
            .host_mapped   = true,
        ));
        if (!buffers->buf[type])
            exit(2);

        if (gpu->export_caps.buf & PL_HANDLE_DMA_BUF) {
            buffers->exported[type] = pl_buf_create(gpu, pl_buf_params(
                .size          = BUFFER_SIZE,
                .memory_type   = type == RAM ? PL_BUF_MEM_HOST : PL_BUF_MEM_DEVICE,
                .export_handle = PL_HANDLE_DMA_BUF,
            ));
        }
    }

    return buffers;
}

static void free_buffers(struct buffers *buffers)
{
    for (enum mem_type type = 0; type < NUM_MEM_TYPES; type++) {
        pl_buf_destroy(buffers->gpu, &buffers->buf[type]);
        pl_buf_destroy(buffers->gpu, &buffers->exported[type]);
        pl_buf_destroy(buffers->gpu, &buffers->imported[type]);
    }
    free(buffers);
}

static void link_buffers(pl_gpu gpu, struct buffers *buffers,
                         const struct buffers *import)
{
    if (!(gpu->import_caps.buf & PL_HANDLE_DMA_BUF))
        return;

    for (enum mem_type type = 0; type < NUM_MEM_TYPES; type++) {
        if (!import->exported[type])
            continue;
        buffers->imported[type] = pl_buf_create(gpu, pl_buf_params(
            .size          = BUFFER_SIZE,
            .memory_type   = type == RAM ? PL_BUF_MEM_HOST : PL_BUF_MEM_DEVICE,
            .import_handle = PL_HANDLE_DMA_BUF,
            .shared_mem    = import->exported[type]->shared_mem,
        ));
    }
}

struct ctx {
    pl_gpu srcgpu, dstgpu;
    pl_tex src, dst;

    // for copy-based methods
    enum mem_owner  owner;
    enum mem_type   type;
    bool noimport;
    bool async;
};

static void await_buf(pl_gpu gpu, pl_buf buf)
{
    while (pl_buf_poll(gpu, buf, UINT64_MAX))
        ; // do nothing
}

static void async_upload(void *priv)
{
    struct buffers *buffers = priv;
    pl_tex_upload(buffers->gpu, &buffers->async);
}

static inline void copy_ptr(struct ctx ctx)
{
    const pl_gpu srcgpu = ctx.srcgpu, dstgpu = ctx.dstgpu;
    const pl_tex src = ctx.src, dst = ctx.dst;
    struct buffers *srcbuffers = src->params.user_data;
    struct buffers *dstbuffers = dst->params.user_data;
    pl_buf buf = NULL;
    uint8_t *data = NULL;

    if (ctx.owner == CPU) {
        static uint8_t static_buffer[BUFFER_SIZE];
        data = page_align(static_buffer);
    } else {
        struct buffers *b = ctx.owner == SRC ? srcbuffers : dstbuffers;
        buf = b->buf[ctx.type];
        data = page_align(buf->data);
        await_buf(b->gpu, buf);
    }

    struct pl_tex_transfer_params src_params = {
        .tex       = src,
        .row_pitch = ROW_PITCH,
        .no_import = ctx.noimport,
    };

    if (ctx.owner == SRC) {
        src_params.buf = buf;
        src_params.buf_offset = data - buf->data;
    } else {
        src_params.ptr = data;
    }

    struct pl_tex_transfer_params dst_params = {
        .tex       = dst,
        .row_pitch = ROW_PITCH,
        .no_import = ctx.noimport,
    };

    if (ctx.owner == DST) {
        dst_params.buf = buf;
        dst_params.buf_offset = data - buf->data;
    } else {
        dst_params.ptr = data;
    }

    if (ctx.async) {
        src_params.callback = async_upload;
        src_params.priv = dstbuffers;
        dstbuffers->async = dst_params;
        pl_tex_download(srcgpu, &src_params);
    } else {
        pl_tex_download(srcgpu, &src_params);
        pl_tex_upload(dstgpu, &dst_params);
    }
}

static inline void copy_interop(struct ctx ctx)
{
    const pl_gpu srcgpu = ctx.srcgpu, dstgpu = ctx.dstgpu;
    const pl_tex src = ctx.src, dst = ctx.dst;
    struct buffers *srcbuffers = src->params.user_data;
    struct buffers *dstbuffers = dst->params.user_data;

    struct pl_tex_transfer_params src_params = {
        .tex       = src,
        .row_pitch = ROW_PITCH,
    };

    struct pl_tex_transfer_params dst_params = {
        .tex       = dst,
        .row_pitch = ROW_PITCH,
    };

    if (ctx.owner == SRC) {
        src_params.buf = srcbuffers->exported[ctx.type];
        dst_params.buf = dstbuffers->imported[ctx.type];
    } else {
        src_params.buf = srcbuffers->imported[ctx.type];
        dst_params.buf = dstbuffers->exported[ctx.type];
    }

    await_buf(srcgpu, src_params.buf);
    if (ctx.async) {
        src_params.callback = async_upload;
        src_params.priv = dstbuffers;
        dstbuffers->async = dst_params;
        pl_tex_download(srcgpu, &src_params);
    } else {
        pl_tex_download(srcgpu, &src_params);
        await_buf(srcgpu, src_params.buf); // manual cross-GPU synchronization
        pl_tex_upload(dstgpu, &dst_params);
    }
}

typedef void method(struct ctx ctx);

static double bench(struct ctx ctx, pl_tex srcs[], pl_tex dsts[], method fun)
{
    const pl_gpu srcgpu = ctx.srcgpu, dstgpu = ctx.dstgpu;
    pl_clock_t start_warmup = 0, start_test = 0;
    uint64_t frames = 0, frames_warmup = 0;

    start_warmup = pl_clock_now();
    do {
        const int idx = frames % NUM_TEX;
        ctx.src = srcs[idx];
        ctx.dst = dsts[idx];

        // Generate some quasi-unique data in the source
        float x = M_E * (frames / 100.0);
        pl_tex_clear(srcgpu, ctx.src, (float[4]) {
            sinf(x + 0.0) / 2.0 + 0.5,
            sinf(x + 2.0) / 2.0 + 0.5,
            sinf(x + 4.0) / 2.0 + 0.5,
            1.0,
        });

        if (fun)
            fun(ctx);

        pl_gpu_flush(srcgpu); // to rotate queues
        pl_gpu_flush(dstgpu);
        frames++;

        if (frames % POLL_FREQ == 0) {
            pl_clock_t now = pl_clock_now();
            if (start_test) {
                if (pl_clock_diff(now, start_test) > TEST_MS * 1e-3)
                    break;
            } else if (pl_clock_diff(now, start_warmup) > WARMUP_MS * 1e-3) {
                start_test = now;
                frames_warmup = frames;
            }
        }
    } while (true);

    pl_gpu_finish(srcgpu);
    pl_gpu_finish(dstgpu);

    return pl_clock_diff(pl_clock_now(), start_test) / (frames - frames_warmup);
}

static void run_tests(pl_gpu srcgpu, pl_gpu dstgpu)
{
    const enum pl_fmt_caps caps = PL_FMT_CAP_HOST_READABLE;
    pl_fmt srcfmt = pl_find_fmt(srcgpu, PL_FMT_UNORM, COMPS, DEPTH, DEPTH, caps);
    pl_fmt dstfmt = pl_find_fmt(dstgpu, PL_FMT_UNORM, COMPS, DEPTH, DEPTH, caps);
    if (!srcfmt || !dstfmt)
        exit(2);

    pl_tex src[NUM_TEX], dst[NUM_TEX];
    for (int i = 0; i < NUM_TEX; i++) {
        struct buffers *srcbuffers = alloc_buffers(srcgpu);
        struct buffers *dstbuffers = alloc_buffers(dstgpu);
        if (!memcmp(srcgpu->uuid, dstgpu->uuid, sizeof(srcgpu->uuid))) {
            link_buffers(srcgpu, srcbuffers, dstbuffers);
            link_buffers(dstgpu, dstbuffers, srcbuffers);
        }

        src[i] = pl_tex_create(srcgpu, pl_tex_params(
            .w             = WIDTH,
            .h             = HEIGHT,
            .format        = srcfmt,
            .host_readable = true,
            .blit_dst      = true,
            .user_data     = srcbuffers,
        ));

        dst[i] = pl_tex_create(dstgpu, pl_tex_params(
            .w             = WIDTH,
            .h             = HEIGHT,
            .format        = dstfmt,
            .host_writable = true,
            .blit_dst      = true,
            .user_data     = dstbuffers,
        ));

        if (!src[i] || !dst[i])
            exit(2);
    }

    struct ctx ctx = {
        .srcgpu = srcgpu,
        .dstgpu = dstgpu,
    };

    static const char *owners[] = {
        [CPU] = "cpu",
        [SRC] = "src",
        [DST] = "dst",
    };

    static const char *types[] = {
        [RAM] = "ram",
        [GPU] = "gpu",
    };

    double baseline = bench(ctx, src, dst, NULL);

    // Test all possible generic copy methods
    for (enum mem_owner owner = 0; owner < NUM_MEM_OWNERS; owner++) {
        for (enum mem_type type = 0; type < NUM_MEM_TYPES; type++) {
            for (int async = 0; async <= 1; async++) {
                for (int noimport = 0; noimport <= 1; noimport++) {
                    // Blacklist undesirable configurations:
                    if (owner == CPU && type != RAM)
                        continue; // impossible
                    if (owner == CPU && async)
                        continue; // no synchronization on static buffer
                    if (owner == SRC && type == GPU)
                        continue; // GPU readback is orders of magnitude too slow
                    if (owner == DST && !noimport)
                        continue; // exhausts source address space

                    struct ctx cfg = ctx;
                    cfg.noimport = noimport;
                    cfg.owner    = owner;
                    cfg.type     = type;
                    cfg.async    = async;

                    printf("  %s %s %s %s : ",
                           owners[owner], types[type],
                           noimport ? "memcpy" : "      ",
                           async    ? "async" : "     ");

                    double dur = bench(cfg, src, dst, copy_ptr) - baseline;
                    printf("avg %.0f μs\t%.3f fps\n",
                           1e6 * dur, 1.0 / dur);
                }
            }
        }
    }

    // Test DMABUF interop when supported
    for (enum mem_owner owner = 0; owner < NUM_MEM_OWNERS; owner++) {
        for (enum mem_type type = 0; type < NUM_MEM_TYPES; type++) {
            for (int async = 0; async <= 1; async++) {
                struct buffers *buffers;
                switch (owner) {
                case SRC:
                    buffers = dst[0]->params.user_data;
                    if (!buffers->imported[type])
                        continue;
                    break;
                case DST:
                    buffers = src[0]->params.user_data;
                    if (!buffers->imported[type])
                        continue;
                    break;
                default: continue;
                }

                struct ctx cfg = ctx;
                cfg.owner = owner;
                cfg.type = type;

                printf("  %s %s %s %s : ",
                       owners[owner], types[type], "dmabuf",
                       async ? "async" : "     ");

                double dur = bench(cfg, src, dst, copy_interop) - baseline;
                        printf("avg %.0f μs\t%.3f fps\n",
                               1e6 * dur, 1.0 / dur);
            }
        }
    }

    for (int i = 0; i < NUM_TEX; i++) {
        free_buffers(src[i]->params.user_data);
        free_buffers(dst[i]->params.user_data);
        pl_tex_destroy(srcgpu, &src[i]);
        pl_tex_destroy(dstgpu, &dst[i]);
    }
}

int main(int argc, const char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s 'Device 1' 'Device 2'\n\n", argv[0]);
        fprintf(stderr, "(Use `vulkaninfo` for a list of devices)\n");
        exit(1);
    }

    pl_log log = pl_log_create(PL_API_VER, pl_log_params(
        .log_cb    = pl_log_color,
        .log_level = PL_LOG_WARN,
    ));

    pl_vk_inst inst = pl_vk_inst_create(log, pl_vk_inst_params(
        .debug = false,
    ));

    pl_vulkan dev1 = pl_vulkan_create(log, pl_vulkan_params(
        .device_name    = argv[1],
        .queue_count    = NUM_QUEUES,
        .async_transfer = ASYNC_TX,
        .async_compute  = ASYNC_COMP,
    ));

    pl_vulkan dev2 = pl_vulkan_create(log, pl_vulkan_params(
        .device_name    = argv[2],
        .queue_count    = NUM_QUEUES,
        .async_transfer = ASYNC_TX,
        .async_compute  = ASYNC_COMP,
    ));

    if (!dev1 || !dev2) {
        fprintf(stderr, "Failed creating Vulkan device!\n");
        exit(1);
    }

    if (ROW_PITCH % dev1->gpu->limits.align_tex_xfer_pitch) {
        fprintf(stderr, "Warning: Row pitch %d is not a multiple of optimal "
                "transfer pitch (%zu) for GPU '%s'\n", ROW_PITCH,
                dev1->gpu->limits.align_tex_xfer_pitch, argv[1]);
    }

    if (ROW_PITCH % dev2->gpu->limits.align_tex_xfer_pitch) {
        fprintf(stderr, "Warning: Row pitch %d is not a multiple of optimal "
                "transfer pitch (%zu) for GPU '%s'\n", ROW_PITCH,
                dev2->gpu->limits.align_tex_xfer_pitch, argv[2]);
    }

    printf("%s -> %s:\n", argv[1], argv[2]);
    run_tests(dev1->gpu, dev2->gpu);
    if (strcmp(argv[1], argv[2])) {
        printf("%s -> %s:\n", argv[2], argv[1]);
        run_tests(dev2->gpu, dev1->gpu);
    }

    pl_vulkan_destroy(&dev1);
    pl_vulkan_destroy(&dev2);
    pl_vk_inst_destroy(&inst);
    pl_log_destroy(&log);
}
