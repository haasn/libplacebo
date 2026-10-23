#include "utils.h"

#include <libplacebo/compositor.h>
#include <libplacebo/dummy.h>

// Copies `json` into an exactly-sized heap buffer with no NUL terminator, so
// that any reliance on one shows up under valgrind/ASan.
static pl_scene create(pl_log log, const char *json)
{
    const size_t len = strlen(json);
    char *buf = malloc(len);
    memcpy(buf, json, len);
    pl_scene scene = pl_scene_create_json(log, buf, len);
    free(buf);
    return scene;
}

static void require_invalid(const char *json)
{
    pl_scene scene = create(NULL, json);
    if (scene) {
        fprintf(stderr, "Scene unexpectedly parsed: %s\n", json);
        exit(1);
    }
}

int main(void)
{
    pl_log log = pl_test_logger();

    // Parsing a scene does not require a GPU
    pl_scene scene = create(log,
        "{"
        "  \"params\": { \"border\": \"tiles\", \"dither\": false },"
        "  \"inputs\": ["
        "    {"
        "      \"start_pts\": \"1001/30000\","
        "      \"stop_pts\": 12.5,"
        "      \"stop_mode\": \"down\","
        "      \"z_index\": -2,"
        "      \"fit\": \"cover\","
        "      \"src_rect\": { \"x0\": \"1/1000\", \"x1\": 0.999 },"
        "      \"focus\": { \"x\": 0.25 },"
        "      \"color\": { \"transfer\": \"pq\", \"hdr\": { \"max_luma\": 600 } },"
        "      \"repr\": { \"sys\": \"bt.709\", \"bits\": { \"sample_depth\": 8 } },"
        "      \"params\": {"
        "        \"upscaler\": \"bilinear\","
        "        \"deband\": true,"
        "        \"deband_iterations\": 3,"
        "        \"brightness\": 0.3"
        "      }"
        "    },"
        "    { \"fit\": \"contain\" }"
        "  ]"
        "}");
    REQUIRE(scene);
    pl_compositor comp = scene->comp;
    REQUIRE_CMP(comp->num_inputs, ==, 2, "d");

    // Input fields, including nested structs and fractions
    pl_input in = comp->inputs[0];
    REQUIRE_CMP(in->start_pts, ==, 1001.0 / 30000.0, "f");
    REQUIRE_CMP(in->stop_pts, ==, 12.5, "f");
    REQUIRE_CMP(in->stop_mode, ==, PL_PTS_ROUND_DOWN, "d");
    REQUIRE_CMP(in->z_index, ==, -2, "d");
    REQUIRE_CMP(in->fit, ==, PL_FIT_COVER, "d");
    REQUIRE_CMP(in->src_rect.x0, ==, 1.0f / 1000.0f, "f");
    REQUIRE_CMP(in->src_rect.x1, ==, 0.999f, "f");
    REQUIRE_CMP(in->src_rect.y1, ==, 1.0f, "f"); // unset fields keep defaults
    REQUIRE_CMP(in->focus.x, ==, 0.25f, "f");
    REQUIRE_CMP(in->focus.y, ==, 0.5f, "f");
    REQUIRE_CMP(in->color.transfer, ==, PL_COLOR_TRC_PQ, "d");
    REQUIRE_CMP(in->color.hdr.max_luma, ==, 600.0f, "f");
    REQUIRE_CMP(in->repr.sys, ==, PL_COLOR_SYSTEM_BT_709, "d");
    REQUIRE_CMP(in->repr.bits.sample_depth, ==, 8, "d");

    // Render params of the input
    const struct pl_render_params *par = in->params;
    REQUIRE(par);
    REQUIRE_STREQ(par->upscaler->name, "bilinear");
    REQUIRE(par->deband_params);
    REQUIRE_CMP(par->deband_params->iterations, ==, 3, "d");
    REQUIRE_CMP(par->color_adjustment->brightness, ==, 0.3f, "f");

    // ... which are independent of the other input, using the defaults
    const struct pl_render_params *par1 = comp->inputs[1]->params;
    REQUIRE(par1 && par1 != par);
    REQUIRE_CMP(comp->inputs[1]->fit, ==, PL_FIT_CONTAIN, "d");
    REQUIRE_STREQ(par1->upscaler->name, pl_render_default_params.upscaler->name);
    REQUIRE(!par1->deband_params);

    // ... and of the canvas
    REQUIRE(comp->params && comp->params != par && comp->params != par1);
    REQUIRE_CMP(comp->params->border, ==, PL_CLEAR_TILES, "d");
    REQUIRE(!comp->params->dither_params);
    REQUIRE(par->dither_params);

    // The compositor starts out uninitialized ...
    const struct pl_queue_params qparams = { PL_QUEUE_DEFAULTS .pts = 1.0 };
    for (int i = 0; i < comp->num_inputs; i++) {
        REQUIRE(!comp->inputs[i]->queue);
        REQUIRE(!comp->inputs[i]->renderer);
    }
    REQUIRE(!pl_compositor_current_frame(comp, &qparams, NULL));
    REQUIRE(!pl_compositor_render(comp, NULL, &qparams, NULL));
    pl_compositor_reset_errors(comp, NULL);
    pl_scene_update(scene, 1.0);

    // ... and can be initialized after the fact
    pl_gpu gpu = pl_gpu_dummy_create(log, NULL);
    REQUIRE(pl_compositor_init(comp, gpu));
    for (int i = 0; i < comp->num_inputs; i++) {
        REQUIRE(comp->inputs[i]->queue);
        REQUIRE(comp->inputs[i]->renderer);
    }
    REQUIRE(!pl_compositor_init(comp, gpu)); // only once

    pl_scene_destroy(&scene);

    // Any error at all fails the whole scene, including after valid inputs
    require_invalid("not json");
    require_invalid("{ \"nope\": 1 }");
    require_invalid("{ \"params\": { \"nope\": 1 } }");
    require_invalid("{ \"inputs\": [ { \"src_rect\": { \"x2\": 1 } } ] }");
    require_invalid("{ \"inputs\": [ { \"fit\": \"cover\" }, { \"fit\": \"squish\" } ] }");

    pl_gpu_dummy_destroy(&gpu);
    pl_log_destroy(&log);
    return 0;
}
