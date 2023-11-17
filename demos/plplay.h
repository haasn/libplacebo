#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>

#include <libplacebo/options.h>
#include <libplacebo/utils/frame_queue.h>

#include "common.h"
#include "pl_thread.h"

#define MAX_FRAME_PASSES 256
#define MAX_BLEND_PASSES 8
#define MAX_BLEND_FRAMES 8

enum {
    ZOOM_PAD = 0,
    ZOOM_CROP,
    ZOOM_STRETCH,
    ZOOM_FIT,
    ZOOM_RAW,
    ZOOM_400,
    ZOOM_200,
    ZOOM_100,
    ZOOM_50,
    ZOOM_25,
    ZOOM_COUNT,
};

struct plplay_args {
    const struct pl_render_params *preset;
    enum pl_log_level verbosity;
    const char *window_impl;
    const char *filename;
    bool hwdec;
};

bool parse_args(struct plplay_args *args, int argc, char *argv[]);

struct plplay {
    struct plplay_args args;
    struct window *win;
    struct ui *ui;
    char cache_file[512];

    // libplacebo
    pl_log log;
    pl_renderer renderer;
    pl_queue queue;
    pl_cache cache;
    uint64_t cache_sig;

    // libav*
    AVFormatContext *format;
    AVCodecContext *codec;
    const AVStream *stream; // points to first video stream of `format`
    pl_thread decoder_thread;
    bool decoder_thread_created;
    bool exit_thread;

    // settings / ui state
    pl_options opts;
    pl_rotation target_rot;
    int target_zoom;
    bool colorspace_hint;
    bool colorspace_hint_dynamic;
    bool ignore_dovi;
    bool toggle_fullscreen;
    bool advanced_scalers;

    bool target_override; // if false, fields below are ignored
    struct pl_color_repr force_repr;
    enum pl_color_primaries force_prim;
    enum pl_color_transfer force_trc;
    struct pl_hdr_metadata force_hdr;
    bool force_hdr_enable;
    bool fps_override;
    float fps;

    // ICC profile
    pl_icc_object icc;
    char *icc_name;
    bool use_icc_luma;
    bool force_bpc;

    // custom shaders
    const struct pl_hook **shader_hooks;
    char **shader_paths;
    size_t shader_num;
    size_t shader_size;

    // pass metadata
    struct pl_dispatch_info blend_info[MAX_BLEND_FRAMES][MAX_BLEND_PASSES];
    struct pl_dispatch_info frame_info[MAX_FRAME_PASSES];
    int num_frame_passes;
    int num_blend_passes[MAX_BLEND_FRAMES];

    // playback statistics
    struct {
        _Atomic uint32_t decoded;
        uint32_t rendered;
        uint32_t mapped;
        uint32_t dropped;
        uint32_t missed;
        uint32_t stalled;
        double missed_ms;
        double stalled_ms;
        double current_pts;

        struct timing {
            double sum, sum2, peak;
            uint64_t count;
        } acquire, update, render, draw_ui, sleep, submit, swap,
          vsync_interval, pts_interval;
    } stats;
};

void update_settings(struct plplay *p, const struct pl_frame *target);

static inline void apply_csp_overrides(struct plplay *p, struct pl_color_space *csp)
{
    if (p->force_prim) {
        csp->primaries = p->force_prim;
        csp->hdr.prim = *pl_raw_primaries_get(csp->primaries);
    }
    if (p->force_trc)
        csp->transfer = p->force_trc;
    if (p->force_hdr_enable) {
        struct pl_hdr_metadata fix = p->force_hdr;
        fix.prim = csp->hdr.prim;
        csp->hdr = fix;
    } else if (p->colorspace_hint_dynamic) {
        pl_color_space_nominal_luma_ex(pl_nominal_luma_params(
            .color      = csp,
            .metadata   = PL_HDR_METADATA_ANY,
            .scaling    = PL_HDR_NITS,
            .out_min    = &csp->hdr.min_luma,
            .out_max    = &csp->hdr.max_luma,
        ));
    }
}
