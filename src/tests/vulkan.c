#include "gpu_tests.h"
#include "vulkan/command.h"
#include "vulkan/gpu.h"
#include <vulkan/vulkan.h>

static void vulkan_interop_tests(const struct pl_vulkan *pl_vk,
                                 enum pl_handle_type handle_type)
{
    const struct pl_gpu *gpu = pl_vk->gpu;

    if (gpu->export_caps.buf & handle_type) {
        const struct pl_buf *buf = pl_buf_create(gpu, &(struct pl_buf_params) {
            .type = PL_BUF_TEX_TRANSFER,
            .size = 1024,
            .handle_type = handle_type,
        });

        REQUIRE(buf);
        REQUIRE(buf->shared_mem.handle.fd > -1);
        REQUIRE(buf->shared_mem.size >= buf->params.size);
        REQUIRE(pl_buf_export(gpu, buf));
        pl_buf_destroy(gpu, &buf);
    }

    const struct pl_fmt *fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 1, 0, 0,
                                           PL_FMT_CAP_BLITTABLE);
    if (!fmt)
        return;

    if (gpu->export_caps.sync & handle_type) {
        const struct pl_sync *sync = pl_sync_create(gpu, handle_type);
        const struct pl_tex *tex = pl_tex_create(gpu, &(struct pl_tex_params) {
            .w = 32,
            .h = 32,
            .format = fmt,
            .blit_dst = true,
        });

        REQUIRE(sync);
        REQUIRE(tex);
        REQUIRE(pl_tex_export(gpu, tex, sync));

        // Re-use our internal helpers to signal this VkSemaphore
        struct vk_ctx *vk = TA_PRIV(pl_vk);
        struct vk_cmd *cmd = vk_cmd_begin(vk, vk->pool_graphics);
        VkSemaphore signal;
        REQUIRE(cmd);
        pl_vk_sync_unwrap(sync, NULL, &signal);
        vk_cmd_sig(cmd, signal);
        vk_cmd_queue(vk, &cmd);
        REQUIRE(vk_flush_commands(vk));

        // Do something with the image again to "import" it
        pl_tex_clear(gpu, tex, (float[4]){0});
        pl_gpu_finish(gpu);

        pl_sync_destroy(gpu, &sync);
        pl_tex_destroy(gpu, &tex);
    }
}

static void vulkan_test_export_import(const struct pl_vulkan *pl_vk,
                                      enum pl_handle_type handle_type)
{
    const struct pl_gpu *gpu = pl_vk->gpu;

    if (!(gpu->export_caps.tex & handle_type) ||
        !(gpu->import_caps.tex & handle_type))
        return;

    const struct pl_fmt *fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 1, 0, 0,
                                           PL_FMT_CAP_BLITTABLE);
    if (!fmt)
        return;

    const struct pl_tex *export = pl_tex_create(gpu, &(struct pl_tex_params) {
        .w = 32,
        .h = 32,
        .format = fmt,
        .export_handle = handle_type,
    });
    REQUIRE(export);
    REQUIRE(export->shared_mem.handle.fd > -1);

    const struct pl_tex *import = pl_tex_create(gpu, &(struct pl_tex_params) {
        .w = 32,
        .h = 32,
        .format = fmt,
        .import_handle = handle_type,
        .shared_mem = export->shared_mem,
    });
    REQUIRE(import);

    pl_tex_destroy(gpu, &import);
    pl_tex_destroy(gpu, &export);
}

static void vulkan_swapchain_tests(const struct pl_vulkan *vk, VkSurfaceKHR surf)
{
    if (!surf)
        return;

    const struct pl_gpu *gpu = vk->gpu;
    const struct pl_swapchain *sw;
    sw = pl_vulkan_create_swapchain(vk, &(struct pl_vulkan_swapchain_params) {
        .surface = surf,
    });
    REQUIRE(sw);

    // Attempt actually initializing the swapchain
    int w = 640, h = 480;
    REQUIRE(pl_swapchain_resize(sw, &w, &h));

    for (int i = 0; i < 10; i++) {
        struct pl_swapchain_frame frame;
        REQUIRE(pl_swapchain_start_frame(sw, &frame));
        if (frame.fbo->params.blit_dst)
            pl_tex_clear(gpu, frame.fbo, (float[4]){0});

        // TODO: test this with an actual pl_renderer instance
        struct pl_render_target target;
        pl_render_target_from_swapchain(&target, &frame);

        REQUIRE(pl_swapchain_submit_frame(sw));
        pl_swapchain_swap_buffers(sw);

        // Try resizing the swapchain in the middle of rendering
        if (i == 5) {
            w = 320;
            h = 240;
            REQUIRE(pl_swapchain_resize(sw, &w, &h));
        }
    }

    pl_swapchain_destroy(&sw);
}

int main()
{
    struct pl_context *ctx = pl_test_context();
    const struct pl_vk_inst *inst;
    inst = pl_vk_inst_create(ctx, &(struct pl_vk_inst_params) {
        .debug = true,
        .debug_extra = true,
        .opt_extensions = (const char *[]){
            VK_KHR_SURFACE_EXTENSION_NAME,
            "VK_EXT_headless_surface", // in case it isn't defined
        },
        .num_opt_extensions = 2,
    });

    if (!inst)
        return SKIP;

    VK_LOAD_FUN(inst->instance, EnumeratePhysicalDevices, vkGetInstanceProcAddr);
    VK_LOAD_FUN(inst->instance, GetPhysicalDeviceProperties, vkGetInstanceProcAddr);

    uint32_t num = 0;
    EnumeratePhysicalDevices(inst->instance, &num, NULL);
    if (!num)
        return SKIP;

    VkPhysicalDevice *devices = calloc(num, sizeof(*devices));
    if (!devices)
        return 1;
    EnumeratePhysicalDevices(inst->instance, &num, devices);

    VkSurfaceKHR surf = NULL;

#ifdef VK_EXT_headless_surface
    VK_LOAD_FUN(inst->instance, CreateHeadlessSurfaceEXT, vkGetInstanceProcAddr);
    if (CreateHeadlessSurfaceEXT) {
        VkHeadlessSurfaceCreateInfoEXT info = {
            .sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT,
        };

        VkResult res = CreateHeadlessSurfaceEXT(inst->instance, &info, NULL, &surf);
        REQUIRE(res == VK_SUCCESS);
    }
#endif // VK_EXT_headless_surface

    // Make sure choosing any device works
    VkPhysicalDevice dev;
    dev = pl_vulkan_choose_device(ctx, &(struct pl_vulkan_device_params) {
        .instance = inst->instance,
        .allow_software = true,
        .surface = surf,
    });
    REQUIRE(dev);

    // Test all attached devices
    for (int i = 0; i < num; i++) {
        VkPhysicalDeviceProperties props = {0};
        GetPhysicalDeviceProperties(devices[i], &props);
        printf("Testing device %d: %s\n", i, props.deviceName);

        // Make sure we can choose this device by name
        dev = pl_vulkan_choose_device(ctx, &(struct pl_vulkan_device_params) {
            .instance = inst->instance,
            .device_name = props.deviceName,
        });
        REQUIRE(dev == devices[i]);

        struct pl_vulkan_params params = pl_vulkan_default_params;
        params.instance = inst->instance;
        params.device = devices[i];
        params.queue_count = 8; // test inter-queue stuff
        params.surface = surf;

        if (props.vendorID == 0x8086 &&
            props.deviceID == 0x3185 &&
            props.driverVersion <= 79695878)
        {
            // Blacklist compute shaders for the CI's old intel iGPU..
            params.blacklist_caps = PL_GPU_CAP_COMPUTE;
        }

        const struct pl_vulkan *vk = pl_vulkan_create(ctx, &params);
        if (!vk)
            continue;

        gpu_tests(vk->gpu);
        vulkan_swapchain_tests(vk, surf);

        // Run these tests last because they disable some validation layers
#ifdef VK_HAVE_UNIX
        vulkan_interop_tests(vk, PL_HANDLE_FD);
        vulkan_interop_tests(vk, PL_HANDLE_DMA_BUF);
        vulkan_test_export_import(vk, PL_HANDLE_DMA_BUF);
#endif
#ifdef VK_HAVE_WIN32
        vulkan_interop_tests(vk, PL_HANDLE_WIN32);
        vulkan_interop_tests(vk, PL_HANDLE_WIN32_KMT);
#endif
        pl_vulkan_destroy(&vk);

        // Re-run the same export/import tests with async queues disabled
        params.async_compute = false;
        params.async_transfer = false;
        vk = pl_vulkan_create(ctx, &params);
        REQUIRE(vk); // it succeeded the first time

#ifdef VK_HAVE_UNIX
        vulkan_interop_tests(vk, PL_HANDLE_FD);
        vulkan_interop_tests(vk, PL_HANDLE_DMA_BUF);
        vulkan_test_export_import(vk, PL_HANDLE_DMA_BUF);
#endif
#ifdef VK_HAVE_WIN32
        vulkan_interop_tests(vk, PL_HANDLE_WIN32);
        vulkan_interop_tests(vk, PL_HANDLE_WIN32_KMT);
#endif
        pl_vulkan_destroy(&vk);

        // Reduce log spam after first tested device
        pl_test_set_verbosity(ctx, PL_LOG_INFO);
    }

    vkDestroySurfaceKHR(inst->instance, surf, NULL);
    pl_vk_inst_destroy(&inst);
    pl_context_destroy(&ctx);
    free(devices);
}
