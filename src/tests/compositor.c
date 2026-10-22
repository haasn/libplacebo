#include "utils.h"

#include <float.h>

#include <libplacebo/compositor.h>
#include <libplacebo/dummy.h>
#include <libplacebo/vulkan.h>

// Tests for pl_compositor: input placement (fit modes, rects, rotation, focus),
// rendering (canvas params, equivalence with the renderer, status reporting),
// compositing order, color overrides, queue management, and the scheduling
// semantics of pl_input_update() on a synthetic output vsync grid.
//
// Reference models are written from <libplacebo/compositor.h> rather than
// mirrored from the implementation, so that the two can disagree and say which
// one is wrong. Set PL_TEST_VERBOSE=1 for trace logging.

#define MAX_SRC     256
#define MAX_OUT     192
#define MAX_INPUTS  3
#define EPS         1e-9

#define FAIL(...)                                                               \
    do {                                                                        \
        fprintf(stderr, "=== FAILED: ");                                        \
        fprintf(stderr, __VA_ARGS__);                                           \
        exit(1);                                                                \
    } while (0)

//----------------------------------------------------------------------------
// Reference model
//----------------------------------------------------------------------------

// Round `pts` to a boundary on the sorted grid `grid`. A PTS already lying on
// a boundary is never adjusted; values outside the grid clamp to its ends.
static double round_grid(double pts, const double *grid, int n,
                         enum pl_pts_round_mode mode)
{
    pl_assert(n > 0);
    if (pts <= grid[0] + EPS)
        return grid[0];
    if (pts >= grid[n - 1] - EPS)
        return grid[n - 1];

    int i = 0;
    while (i + 1 < n && grid[i + 1] <= pts + EPS)
        i++;
    if (fabs(grid[i] - pts) < EPS)
        return grid[i];

    const double lo = grid[i], hi = grid[i + 1];
    switch (mode) {
    case PL_PTS_ROUND_DOWN:    return lo;
    case PL_PTS_ROUND_UP:      return hi;
    case PL_PTS_ROUND_NEAREST: return (pts - lo) >= (hi - pts) ? hi : lo; // ties up
    case PL_PTS_ROUND_COUNT:   break;
    }
    pl_unreachable();
}

// Round an output timestamp. Both ends of a cut are quantized by the same
// function against the same grid, which is what makes the sequential playback
// guarantee hold. Without a known vsync duration only UP is resolvable, so
// everything else degrades to UP - uniformly, preserving the guarantee.
static double round_out(double pts, const double *grid, int n,
                        enum pl_pts_round_mode mode, bool have_vsync)
{
    if (!have_vsync)
        mode = PL_PTS_ROUND_UP;
    return round_grid(pts, grid, n, mode);
}

struct resolved {
    double start, stop;
};

//----------------------------------------------------------------------------
// Scenario description
//----------------------------------------------------------------------------

struct input_spec {
    // Synthetic source stream
    double first_pts;   // PTS of the first source frame
    double fps;         // source frame rate
    int count;          // number of source frames

    // Scheduling. `start`/`stop` are output timestamps, `seek` is an input one.
    double start, stop, seek;
    enum pl_pts_round_mode start_mode, stop_mode;

    bool trickle;       // feed the queue lazily rather than pre-pushing
};

struct scenario {
    const char *name;
    double out_fps;
    int num_out;
    const struct pl_filter_config *mixer; // NULL = no frame mixing
    bool no_vsync_hint;                   // leave pl_queue_params.vsync_duration unset
    struct input_spec in[MAX_INPUTS];
    int num_in;
};

//----------------------------------------------------------------------------
// Simulator
//----------------------------------------------------------------------------

struct sim_input {
    const struct input_spec *spec;
    struct pl_frame frames[MAX_SRC];
    struct pl_source_frame srcs[MAX_SRC];
    double pts[MAX_SRC];
    int pushed;
    pl_input input;

    // Reference model state
    struct resolved ref;

    // Per-output-frame records
    char ref_line[MAX_OUT + 1];
    char act_line[MAX_OUT + 1];
    double act_pts[MAX_OUT];
    bool act_vis[MAX_OUT];
    enum pl_queue_status act_status[MAX_OUT];

    // Values reported via pl_input_status, per output frame. The implementation
    // only sees frames up to the current instant, so these are not known up
    // front; NAN means "not yet determinable" and the checks below verify
    // monotone refinement rather than eager equality.
    double act_start[MAX_OUT];
    double act_stop[MAX_OUT];
};

static bool map_frame(pl_gpu gpu, pl_tex *tex, const struct pl_source_frame *src,
                      struct pl_frame *out_frame)
{
    *out_frame = *(const struct pl_frame *) src->frame_data;
    return true;
}

static char advance_char(int prev, int cur, bool visible)
{
    if (!visible)
        return '.';
    if (prev < 0)
        return '^'; // first visible frame of a run
    if (cur == prev)
        return '='; // duplicated source frame
    if (cur == prev + 1)
        return '>'; // advanced by exactly one source frame
    if (cur > prev + 1)
        return '+'; // skipped one or more source frames
    return '<';     // went backwards
}

// The source frame the renderer displays at `pts`, or -1 if none is available
// yet. This models pl_render_image_mix(): a frame only becomes available once
// `pts` reaches it, but among the frames actually in the mix the renderer picks
// refimg via pl_frame_mix_nearest() - nearest neighbour, not zero-order hold.
//
// The availability gate applies to still images too, so a single frame whose
// PTS lies ahead of the mapped input time never becomes visible at all. Only
// its *duration* is unbounded; see `build_reference`.
static int nearest_index(const struct sim_input *si, double pts)
{
    if (!si->spec->count || pts + EPS < si->pts[0])
        return -1; // nothing to show yet

    int best = 0;
    for (int i = 1; i < si->spec->count; i++) {
        if (fabs(si->pts[i] - pts) < fabs(si->pts[best] - pts))
            best = i;
    }
    return best;
}

static void build_reference(struct sim_input *si, const double *out_grid,
                            int out_n, int num_out, double vsync,
                            bool have_vsync)
{
    const struct input_spec *s = si->spec;
    const double frame_dur = 1.0 / s->fps;

    struct resolved *r = &si->ref;
    r->start = round_out(s->start, out_grid, out_n, s->start_mode, have_vsync);

    if (s->stop < DBL_MAX) {
        r->stop = round_out(s->stop, out_grid, out_n, s->stop_mode, have_vsync);
    } else if (s->count == 1) {
        r->stop = DBL_MAX; // still image: unbounded, never runs out
    } else {
        // "Until the stream runs out": the queue holds the last frame for one
        // extra frame duration before reporting EOF.
        r->stop = r->start - s->seek + si->pts[s->count - 1] + frame_dur;
        // A seek past the end of the stream never becomes visible at all, in
        // which case it stops at the same instant it was scheduled to start.
        r->stop = PL_MAX(r->stop, r->start);
    }

    int prev = -1;
    for (int o = 0; o < num_out; o++) {
        const double p = o * vsync;
        // Being scheduled is not sufficient: the input also needs a frame to
        // show, which it lacks while `seek_pts` maps to a point preceding the
        // stream. Such an input simply appears late.
        const bool sched = p >= r->start - EPS && p < r->stop - EPS;
        const int cur = sched ? nearest_index(si, p - r->start + s->seek) : -1;
        const bool vis = sched && cur >= 0;
        si->ref_line[o] = advance_char(prev, cur, vis);
        prev = vis ? cur : -1;
    }
    si->ref_line[num_out] = '\0';
}

// Run the scenario, filling in each input's `act_*` records. `repeat` controls
// how many times pl_input_update() is called per output frame, which must not
// affect the result.
static void simulate(const struct scenario *sc, struct sim_input *sis,
                     int repeat, pl_log log, pl_gpu gpu)
{
    pl_compositor comp = pl_compositor_create(log);
    pl_compositor_init(comp, gpu);

    struct pl_render_params rparams = { PL_RENDER_DEFAULTS };
    rparams.frame_mixer = sc->mixer;

    struct pl_queue_params params = { PL_QUEUE_DEFAULTS
        .vsync_duration = sc->no_vsync_hint ? 0.0 : 1.0 / sc->out_fps,
    };

    const double vsync = 1.0 / sc->out_fps;

    for (int n = 0; n < sc->num_in; n++) {
        struct sim_input *si = &sis[n];
        const struct input_spec *s = si->spec;

        si->input = pl_compositor_input_create(comp);
        si->input->start_pts  = s->start;
        si->input->stop_pts   = s->stop;
        si->input->seek_pts   = s->seek;
        si->input->start_mode = s->start_mode;
        si->input->stop_mode  = s->stop_mode;
        si->input->params     = &rparams;

        si->pushed = 0;
        if (!s->trickle) {
            for (int i = 0; i < s->count; i++)
                pl_queue_push(si->input->queue, &si->srcs[i]);
            pl_queue_push(si->input->queue, NULL);
            si->pushed = s->count + 1;
        }
    }

    int prev_idx[MAX_INPUTS];
    for (int n = 0; n < MAX_INPUTS; n++)
        prev_idx[n] = -1;

    for (int o = 0; o < sc->num_out; o++) {
        const double p = o * vsync;
        params.pts = p;

        for (int n = 0; n < sc->num_in; n++) {
            struct sim_input *si = &sis[n];
            const struct input_spec *s = si->spec;

            // Keep the queue fed relative to the *reference* timeline, so the
            // feed schedule does not depend on the implementation.
            if (s->trickle) {
                const double want = p - si->ref.start + s->seek + 5.0 / s->fps;
                while (si->pushed < s->count && si->pts[si->pushed] < want)
                    pl_queue_push(si->input->queue, &si->srcs[si->pushed++]);
                if (si->pushed == s->count) {
                    pl_queue_push(si->input->queue, NULL);
                    si->pushed++;
                }
            }

            struct pl_input_status st = {0};
            enum pl_queue_status ret = PL_QUEUE_OK;
            for (int rep = 0; rep < repeat; rep++)
                ret = pl_input_update(si->input, &st, &params);

            if (ret == PL_QUEUE_ERR)
                FAIL("input %d: PL_QUEUE_ERR at out[%d]\n", n, o);

            const bool vis = st.mix != NULL;
            int cur = -1;
            if (vis) {
                // Documented guarantee: a non-NULL mix holds at least one frame
                if (!st.mix->num_frames)
                    FAIL("input %d: mix non-NULL but empty at out[%d]\n", n, o);
                const struct pl_frame *f = pl_frame_mix_nearest(st.mix);
                cur = (int) (intptr_t) f->user_data;
            }

            si->act_status[o] = ret;
            si->act_vis[o]    = vis;
            si->act_pts[o]    = st.pts;
            si->act_start[o]  = st.start_pts;
            si->act_stop[o]   = st.stop_pts;
            si->act_line[o]   = advance_char(prev_idx[n], cur, vis);
            prev_idx[n] = vis ? cur : -1;
        }
    }

    for (int n = 0; n < sc->num_in; n++)
        sis[n].act_line[sc->num_out] = '\0';

    pl_compositor_destroy(&comp);
}

//----------------------------------------------------------------------------
// Checks
//----------------------------------------------------------------------------

static const char *fmt_pts(double pts, char buf[32])
{
    if (pts >= DBL_MAX)
        snprintf(buf, 32, "+inf");
    else
        snprintf(buf, 32, "%.4f", pts);
    return buf;
}

// Invariants that hold regardless of the rounding semantics in force.
static void check_invariants(const struct scenario *sc,
                             const struct sim_input *si, int idx)
{
    // The visible set must be a single contiguous run. An input that flickers
    // on and off, or comes back after stopping, is always a bug.
    int runs = 0;
    for (int o = 0; o < sc->num_out; o++) {
        if (si->act_vis[o] && (o == 0 || !si->act_vis[o - 1]))
            runs++;
    }
    if (runs > 1) {
        FAIL("input %d visible in %d disjoint runs\n timeline = %s\n",
             idx, runs, si->act_line);
    }

    // NAN means "not yet determinable", and each value becomes determinable at
    // a specific, observable moment:
    //
    //   start_pts  => the input has been visible at least once
    //   stop_pts  <=> pl_input_update() has returned PL_QUEUE_EOF
    //
    // `start_pts` is only an implication: it latches when the *scheduled* start
    // is reached, which can precede the first visible frame if the queue has
    // nothing to show yet. The converse - visible but unresolved - is a bug,
    // since the output-to-input mapping is anchored on it.
    bool was_visible = false, was_eof = false;
    for (int o = 0; o < sc->num_out; o++) {
        was_visible |= si->act_vis[o];
        was_eof |= si->act_status[o] == PL_QUEUE_EOF;

        if (was_visible && isnan(si->act_start[o])) {
            FAIL("input %d start_pts is NAN at out[%d], but the input has been "
                 "visible\n timeline = %s\n", idx, o, si->act_line);
        }
        if (isnan(si->act_stop[o]) == was_eof) {
            FAIL("input %d stop_pts is %s at out[%d], but EOF has%s been "
                 "reported\n timeline = %s\n", idx, was_eof ? "NAN" : "set",
                 o, was_eof ? "" : " not", si->act_line);
        }
    }

    // The reported input-domain PTS advances monotonically while visible.
    for (int o = 1; o < sc->num_out; o++) {
        if (!si->act_vis[o] || !si->act_vis[o - 1])
            continue;
        if (si->act_pts[o] < si->act_pts[o - 1] - EPS) {
            FAIL("input %d PTS went backwards at out[%d]: %f -> %f\n",
                 idx, o, si->act_pts[o - 1], si->act_pts[o]);
        }
    }
}

// Verify one reported value across the run: NAN for as long as it is not yet
// determinable, but once it commits to a concrete value that value must be
// correct and must never change or revert.
//
// `need_by` is the output frame by which it must have resolved, or -1 if the
// run never reaches that point.
static void check_value(const struct scenario *sc, int idx, const char *name,
                        const double *got, double ref, int need_by)
{
    char b0[32], b1[32];
    int first = -1;
    for (int o = 0; o < sc->num_out; o++) {
        if (!isnan(got[o])) {
            first = o;
            break;
        }
    }

    if (first < 0) {
        if (need_by >= 0)
            FAIL("input %d: %s never resolved (needed by out[%d])\n",
                 idx, name, need_by);
        return;
    }

    for (int o = first + 1; o < sc->num_out; o++) {
        if (isnan(got[o])) {
            FAIL("input %d: %s reverted to NAN at out[%d] after resolving at "
                 "out[%d]\n", idx, name, o, first);
        }
        if (fabs(got[o] - got[first]) > EPS) {
            FAIL("input %d: %s changed after resolving: %s at out[%d] -> %s at "
                 "out[%d]\n", idx, name, fmt_pts(got[first], b0), first,
                 fmt_pts(got[o], b1), o);
        }
    }

    if (fabs(got[first] - ref) > EPS) {
        FAIL("input %d: %s resolved to %s, reference says %s\n",
             idx, name, fmt_pts(got[first], b0), fmt_pts(ref, b1));
    }

    if (first > need_by && need_by >= 0) {
        FAIL("input %d: %s only resolved at out[%d], but was needed by "
             "out[%d]\n", idx, name, first, need_by);
    }
}

static void check_resolved(const struct scenario *sc, const struct sim_input *si,
                           int idx)
{
    // `start_pts` must be resolved by the time the input first appears, and
    // `stop_pts` by the time it disappears again. If the run never reaches
    // those points, leaving them unresolved is legitimate.
    int first_vis = -1, last_vis = -1;
    for (int o = 0; o < sc->num_out; o++) {
        if (si->act_vis[o]) {
            if (first_vis < 0)
                first_vis = o;
            last_vis = o;
        }
    }
    const int stop_at = (last_vis >= 0 && last_vis + 1 < sc->num_out)
                            ? last_vis + 1 : -1;

    check_value(sc, idx, "start_pts", si->act_start, si->ref.start, first_vis);
    check_value(sc, idx, "stop_pts",  si->act_stop,  si->ref.stop,  stop_at);
}

static void check_reference(const struct scenario *sc, const struct sim_input *si,
                            int idx)
{
    char b0[32], b1[32];

    if (strcmp(si->ref_line, si->act_line) != 0) {
        int at = 0;
        while (si->ref_line[at] && si->ref_line[at] == si->act_line[at])
            at++;
        FAIL("input %d: timeline diverges from reference at out[%d]\n"
             " ref: %s\n act: %s\n      %*s^\n",
             idx, at, si->ref_line, si->act_line, at, "");
    }

    printf("      input %d: %s  (start=%s stop=%s)\n", idx, si->act_line,
           fmt_pts(si->ref.start, b0), fmt_pts(si->ref.stop, b1));

    check_resolved(sc, si, idx);
}

// The sequential playback guarantee: for any input pair where one's `stop_pts`
// equals the other's `start_pts` under matching rounding modes, the handover
// must be exact - no output frame with both visible, none with neither.
//
// This holds by construction: both ends are quantized on the same grid by the
// same function.
static void check_sequential(const struct scenario *sc,
                             const struct sim_input *sis)
{
    for (int a = 0; a < sc->num_in; a++) {
        for (int b = 0; b < sc->num_in; b++) {
            if (a == b)
                continue;
            const struct input_spec *sa = sis[a].spec, *sb = sis[b].spec;
            if (sa->stop >= DBL_MAX || fabs(sa->stop - sb->start) > EPS)
                continue;
            if (sa->stop_mode != sb->start_mode)
                continue;

            for (int o = 0; o < sc->num_out; o++) {
                if (sis[a].act_vis[o] && sis[b].act_vis[o]) {
                    FAIL("inputs %d and %d overlap at out[%d], but %d->stop_pts"
                         " == %d->start_pts == %f\n %d = %s\n %d = %s\n",
                         a, b, o, a, b, sa->stop, a, sis[a].act_line,
                         b, sis[b].act_line);
                }
            }

            // No gap either: b must pick up on exactly the frame a drops out.
            int last_a = -1, first_b = -1;
            for (int o = 0; o < sc->num_out; o++) {
                if (sis[a].act_vis[o])
                    last_a = o;
                if (sis[b].act_vis[o] && first_b < 0)
                    first_b = o;
            }
            if (last_a >= 0 && first_b >= 0 && first_b != last_a + 1) {
                FAIL("gap between inputs %d and %d: %d ends at out[%d], %d "
                     "starts at out[%d]\n %d = %s\n %d = %s\n", a, b, a,
                     last_a, b, first_b, a, sis[a].act_line, b, sis[b].act_line);
            }
        }
    }
}

static void run_scenario(const struct scenario *sc, pl_log log, pl_gpu gpu)
{
    static struct sim_input sis[MAX_INPUTS];
    static struct sim_input sis2[MAX_INPUTS];

    const double vsync = 1.0 / sc->out_fps;
    pl_assert(sc->name && sc->out_fps > 0);
    pl_assert(sc->num_in > 0 && sc->num_in <= MAX_INPUTS);
    pl_assert(sc->num_out > 0 && sc->num_out <= MAX_OUT);

    // Output vsync grid, extended past the simulated range so that a stop can
    // round up beyond the last simulated frame.
    static double out_grid[MAX_OUT * 2];
    const int out_n = MAX_OUT * 2;
    for (int i = 0; i < out_n; i++)
        out_grid[i] = i * vsync;

    printf("--- %s (%g fps x%d)\n", sc->name, sc->out_fps, sc->num_out);

    for (int n = 0; n < sc->num_in; n++) {
        const struct input_spec *s = &sc->in[n];
        pl_assert(s->count > 0 && s->count <= MAX_SRC);

        struct sim_input *si = &sis[n];
        *si = (struct sim_input) { .spec = s };
        for (int i = 0; i < s->count; i++) {
            si->pts[i] = s->first_pts + i / s->fps;
            si->frames[i] = (struct pl_frame) {
                .user_data = (void *) (intptr_t) i,
            };
            si->srcs[i] = (struct pl_source_frame) {
                .pts        = si->pts[i],
                .duration   = 1.0 / s->fps,
                .map        = map_frame,
                .frame_data = &si->frames[i],
            };
        }
        build_reference(si, out_grid, out_n, sc->num_out, vsync,
                        !sc->no_vsync_hint);
    }

    simulate(sc, sis, 1, log, gpu);

    for (int n = 0; n < sc->num_in; n++) {
        check_invariants(sc, &sis[n], n);
        check_reference(sc, &sis[n], n);
    }
    check_sequential(sc, sis);

    // Repeated pl_input_update() calls at an identical PTS must not change the
    // outcome; plplay issues 2-3 per output frame.
    for (int n = 0; n < sc->num_in; n++)
        sis2[n] = sis[n];
    simulate(sc, sis2, 3, log, gpu);
    for (int n = 0; n < sc->num_in; n++) {
        if (strcmp(sis[n].act_line, sis2[n].act_line) != 0) {
            FAIL("input %d not idempotent under repeated pl_input_update()\n"
                 " 1x = %s\n 3x = %s\n", n, sis[n].act_line, sis2[n].act_line);
        }
    }
}

//----------------------------------------------------------------------------
// Scenarios
//----------------------------------------------------------------------------

#define SRC(fps_, first_, count_) .fps = fps_, .first_pts = first_, .count = count_
#define NOSTOP  .stop = DBL_MAX

static const struct scenario scenarios[] = {
    {
        .name = "baseline: 30 -> 30, fully aligned",
        .out_fps = 30, .num_out = 12, .num_in = 1,
        .in = {{ SRC(30, 0.0, 20), NOSTOP }},
    }, {
        .name = "start aligned to the output grid (0.1 = 3 vsyncs)",
        .out_fps = 30, .num_out = 12, .num_in = 1,
        .in = {{ SRC(30, 0.0, 20), NOSTOP, .start = 0.1 }},
    }, {
        .name = "start misaligned by half a vsync, NEAREST (ties round up)",
        .out_fps = 30, .num_out = 12, .num_in = 1,
        .in = {{ SRC(30, 0.0, 20), NOSTOP, .start = 1/60.0 }},
    }, {
        .name = "start misaligned by half a vsync, DOWN",
        .out_fps = 30, .num_out = 12, .num_in = 1,
        .in = {{ SRC(30, 0.0, 20), NOSTOP, .start = 1/60.0,
                 .start_mode = PL_PTS_ROUND_DOWN }},
    }, {
        .name = "start misaligned by half a vsync, UP",
        .out_fps = 30, .num_out = 12, .num_in = 1,
        .in = {{ SRC(30, 0.0, 20), NOSTOP, .start = 1/60.0,
                 .start_mode = PL_PTS_ROUND_UP }},
    }, {
        .name = "start misaligned, DOWN but no vsync_duration (falls back to UP)",
        .out_fps = 30, .num_out = 12, .num_in = 1, .no_vsync_hint = true,
        .in = {{ SRC(30, 0.0, 20), NOSTOP, .start = 1/60.0,
                 .start_mode = PL_PTS_ROUND_DOWN }},
    }, {
        .name = "seek mid-frame: input phase preserved exactly",
        .out_fps = 30, .num_out = 12, .num_in = 1,
        .in = {{ SRC(30, 0.0, 20), NOSTOP, .seek = 0.055 }},
    }, {
        // Deliberate: the renderer selects frames by nearest neighbour in all
        // cases, so a mid-frame seek into a low-fps source shifts every cut by
        // up to half an input frame (0.125s here) and can skip the first frame
        // outright - at 4 fps a seek to 0.13 is already closer to frame 1 than
        // frame 0. ZOH would hold each frame for its full interval instead;
        // that trade was made knowingly in favour of a single uniform rule.
        .name = "very low fps input: nearest selection shifts cut points",
        .out_fps = 30, .num_out = 24, .num_in = 1,
        .in = {{ SRC(4, 0.0, 8), NOSTOP, .seek = 0.13 }},
    }, {
        // A single frame is a still image of unbounded duration: it never runs
        // out, regardless of any tagged frame duration, and so is visible for
        // exactly [start_pts, stop_pts). Its own PTS still gates when it first
        // becomes available, as for any other frame.
        .name = "still: single frame, no stop -> visible for the whole run",
        .out_fps = 30, .num_out = 12, .num_in = 1,
        .in = {{ SRC(30, 0.0, 1), NOSTOP }},
    }, {
        .name = "still: low tagged frame duration must not end it",
        .out_fps = 30, .num_out = 12, .num_in = 1,
        .in = {{ SRC(4, 0.0, 1), NOSTOP }},
    }, {
        .name = "still: visible only within [start, stop)",
        .out_fps = 30, .num_out = 12, .num_in = 1,
        .in = {{ SRC(30, 0.0, 1), .start = 0.1, .stop = 0.2 }},
    }, {
        // No clamping: the caller is responsible for normalizing the timebase,
        // since libplacebo cannot tell an unnormalized stream apart from one
        // that genuinely begins late.
        .name = "unnormalized stream @3.7, seek = 3.7 (caller normalizes)",
        .out_fps = 30, .num_out = 8, .num_in = 1,
        .in = {{ SRC(30, 3.7, 20), NOSTOP, .seek = 3.7 }},
    }, {
        .name = "stream @0.1, seek = 0: appears 3 vsyncs late, no clamp",
        .out_fps = 30, .num_out = 12, .num_in = 1,
        .in = {{ SRC(30, 0.1, 20), NOSTOP }},
    }, {
        .name = "seek past the end of the stream: never becomes visible",
        .out_fps = 30, .num_out = 8, .num_in = 1,
        .in = {{ SRC(30, 0.0, 5), NOSTOP, .seek = 100.0 }},
    }, {
        .name = "stop aligned to the output grid (3 frames)",
        .out_fps = 30, .num_out = 8, .num_in = 1,
        .in = {{ SRC(30, 0.0, 20), .stop = 0.1 }},
    }, {
        .name = "stop misaligned, NEAREST",
        .out_fps = 30, .num_out = 10, .num_in = 1,
        .in = {{ SRC(30, 0.0, 20), .stop = 0.115 }},
    }, {
        .name = "stop misaligned, UP",
        .out_fps = 30, .num_out = 10, .num_in = 1,
        .in = {{ SRC(30, 0.0, 20), .stop = 0.115,
                 .stop_mode = PL_PTS_ROUND_UP }},
    }, {
        .name = "stop = DBL_MAX: plays until the stream runs out",
        .out_fps = 30, .num_out = 28, .num_in = 1,
        .in = {{ SRC(30, 0.0, 15), NOSTOP }},
    }, {
        .name = "rate mismatch: 24 -> 60, no mixer",
        .out_fps = 60, .num_out = 16, .num_in = 1,
        .in = {{ SRC(24, 0.0, 20), NOSTOP }},
    }, {
        .name = "rate mismatch: 24 -> 60, mitchell mixer",
        .out_fps = 60, .num_out = 16, .num_in = 1,
        .mixer = &pl_filter_mitchell_clamp,
        .in = {{ SRC(24, 0.0, 20), NOSTOP }},
    }, {
        .name = "trickle-fed queue instead of pre-pushed",
        .out_fps = 30, .num_out = 12, .num_in = 1,
        .in = {{ SRC(30, 0.0, 20), NOSTOP, .trickle = true }},
    },

    // The sequential playback guarantee: input0->stop_pts == input1->start_pts
    // with matching modes must hand over with no gap and no overlap.
    {
        .name = "SEQ: cut on a vsync boundary",
        .out_fps = 30, .num_out = 16, .num_in = 2,
        .in = {
            { SRC(30, 0.0, 20), .stop = 0.2 },
            { SRC(30, 0.0, 20), NOSTOP, .start = 0.2 },
        },
    }, {
        .name = "SEQ: cut mid-vsync, NEAREST both sides",
        .out_fps = 30, .num_out = 16, .num_in = 2,
        .in = {
            { SRC(30, 0.0, 20), .stop = 0.215 },
            { SRC(30, 0.0, 20), NOSTOP, .start = 0.215 },
        },
    }, {
        .name = "SEQ: cut mid-vsync, DOWN both sides",
        .out_fps = 30, .num_out = 16, .num_in = 2,
        .in = {
            { SRC(30, 0.0, 20), .stop = 0.215,
              .stop_mode = PL_PTS_ROUND_DOWN },
            { SRC(30, 0.0, 20), NOSTOP, .start = 0.215,
              .start_mode = PL_PTS_ROUND_DOWN },
        },
    }, {
        .name = "SEQ: cut mid-vsync, UP both sides",
        .out_fps = 30, .num_out = 16, .num_in = 2,
        .in = {
            { SRC(30, 0.0, 20), .stop = 0.215,
              .stop_mode = PL_PTS_ROUND_UP },
            { SRC(30, 0.0, 20), NOSTOP, .start = 0.215,
              .start_mode = PL_PTS_ROUND_UP },
        },
    }, {
        .name = "SEQ: cut mid-vsync, mismatched frame rates either side",
        .out_fps = 30, .num_out = 16, .num_in = 2,
        .in = {
            { SRC(24, 0.0, 20), .stop = 0.215 },
            { SRC(50, 0.0, 30), NOSTOP, .start = 0.215 },
        },
    }, {
        .name = "SEQ: cut mid-vsync, no vsync_duration (both fall back to UP)",
        .out_fps = 30, .num_out = 16, .num_in = 2, .no_vsync_hint = true,
        .in = {
            { SRC(30, 0.0, 20), .stop = 0.215 },
            { SRC(30, 0.0, 20), NOSTOP, .start = 0.215 },
        },
    }, {
        // Degenerate case of the same guarantee: a zero-length cut is empty
        // regardless of where it falls or which mode quantizes it.
        .name = "EMPTY: start == stop on a vsync boundary",
        .out_fps = 30, .num_out = 10, .num_in = 1,
        .in = {{ SRC(30, 0.0, 20), .start = 0.1, .stop = 0.1 }},
    }, {
        .name = "EMPTY: start == stop mid-vsync, NEAREST both",
        .out_fps = 30, .num_out = 10, .num_in = 1,
        .in = {{ SRC(30, 0.0, 20), .start = 0.215, .stop = 0.215 }},
    }, {
        .name = "EMPTY: start == stop mid-vsync, UP both",
        .out_fps = 30, .num_out = 10, .num_in = 1,
        .in = {{ SRC(30, 0.0, 20), .start = 0.215, .stop = 0.215,
                 .start_mode = PL_PTS_ROUND_UP,
                 .stop_mode  = PL_PTS_ROUND_UP }},
    }, {
        .name = "EMPTY: start == stop mid-vsync, DOWN both",
        .out_fps = 30, .num_out = 10, .num_in = 1,
        .in = {{ SRC(30, 0.0, 20), .start = 0.215, .stop = 0.215,
                 .start_mode = PL_PTS_ROUND_DOWN,
                 .stop_mode  = PL_PTS_ROUND_DOWN }},
    }, {
        .name = "EMPTY: start == stop at 0.0",
        .out_fps = 30, .num_out = 10, .num_in = 1,
        .in = {{ SRC(30, 0.0, 20), .start = 0.0, .stop = 0.0 }},
    },
};

//----------------------------------------------------------------------------
// Fit modes
//----------------------------------------------------------------------------

// pl_input_place() is a pure function of the fields below, so these cases need
// no queue, no renderer and no GPU work - just two bare frames.
struct fit_case {
    const char *name;
    enum pl_fit_mode fit;
    float image_w, image_h, image_par;
    pl_rotation image_rot;
    float canvas_w, canvas_h, canvas_par;
    pl_rotation canvas_rot;
    pl_rotation rotation;                   // pl_input.rotation
    struct pl_rect2df src_rect, dst_rect;   // pl_input.*_rect, {0} = full frame
    pl_vec2df focus;                        // pl_input.focus, if `has_focus`
    bool has_focus;
    int focus_centered;                     // FOCUS_X/Y: axes on which the focus
                                            // must land on the dst_rect center
    float expect_aspect;                    // physical aspect, 0 = unchecked
};

enum { FOCUS_X = 1 << 0, FOCUS_Y = 1 << 1 };
#define FOCUS(fx, fy) .focus = { fx, fy }, .has_focus = true

static pl_vec2df focus_of(const struct fit_case *c)
{
    return c->has_focus ? c->focus : (pl_vec2df) { 0.5f, 0.5f };
}

static const struct pl_rect2df full_rect = { 0, 0, 1, 1 };

static struct pl_rect2df rect_or_full(struct pl_rect2df rc)
{
    return pl_rect2d_eq(rc, (struct pl_rect2df) {0}) ? full_rect : rc;
}

// Map `rc`, given relative to `outer`, into the frame `outer` is relative to.
static struct pl_rect2df mix_rect(struct pl_rect2df outer, struct pl_rect2df rc)
{
    return (struct pl_rect2df) {
        .x0 = PL_MIX(outer.x0, outer.x1, rc.x0),
        .y0 = PL_MIX(outer.y0, outer.y1, rc.y0),
        .x1 = PL_MIX(outer.x0, outer.x1, rc.x1),
        .y1 = PL_MIX(outer.y0, outer.y1, rc.y1),
    };
}

// Mirror `rc` within `bounds`, along every axis on which `ref` is flipped.
static void flip_like(struct pl_rect2df *rc, struct pl_rect2df ref,
                      struct pl_rect2df bounds)
{
    if (ref.x0 > ref.x1) {
        rc->x0 = bounds.x0 + bounds.x1 - rc->x0;
        rc->x1 = bounds.x0 + bounds.x1 - rc->x1;
    }
    if (ref.y0 > ref.y1) {
        rc->y0 = bounds.y0 + bounds.y1 - rc->y0;
        rc->y1 = bounds.y0 + bounds.y1 - rc->y1;
    }
}

// Clip `dst` to `bounds`, cutting `src` down by the same fraction so that the
// mapping between the two stays exactly the same.
static void clip_rect(struct pl_rect2df *src, struct pl_rect2df *dst,
                      struct pl_rect2df bounds)
{
    const float sx = pl_rect_w(*src) / pl_rect_w(*dst);
    const float sy = pl_rect_h(*src) / pl_rect_h(*dst);
    if (dst->x0 < bounds.x0) {
        src->x0 += (bounds.x0 - dst->x0) * sx;
        dst->x0 = bounds.x0;
    }
    if (dst->x1 > bounds.x1) {
        src->x1 -= (dst->x1 - bounds.x1) * sx;
        dst->x1 = bounds.x1;
    }
    if (dst->y0 < bounds.y0) {
        src->y0 += (bounds.y0 - dst->y0) * sy;
        dst->y0 = bounds.y0;
    }
    if (dst->y1 > bounds.y1) {
        src->y1 -= (dst->y1 - bounds.y1) * sy;
        dst->y1 = bounds.y1;
    }
}

// "The smaller of the two is centered on the focus point of the larger, as far
// as possible without exceeding it": the offset of a span of `size`, relative
// to the larger one, centered on `focus`.
static float align(float size, float focus)
{
    return PL_CLAMP(focus - size / 2, 0.0f, 1.0f - size);
}

// Logical size, i.e. after folding in the pixel aspect ratio and rotation.
static void logical(float *w, float *h, float pw, float ph, float par,
                    pl_rotation rot)
{
    *w = pw;
    *h = ph;
    if (par > 1.0f)
        *w *= par;
    else if (par > 0.0f && par < 1.0f)
        *h /= par;
    if (rot % PL_ROTATION_180 == PL_ROTATION_90)
        PL_SWAP(*w, *h);
}

// Frame of reference: everything is expressed in the *displayed* orientation
// of each frame. For the image, that is after both its own rotation and
// `pl_input.rotation`; for the canvas, after its own rotation. A 16:9 camera
// frame tagged 90 degrees fits like a 9:16 one, `src_rect = {0, 0, 0.5, 1}`
// crops away the right half of what the viewer would see, and the rects that
// pl_input_place() returns are in that same orientation. Mapping all of this
// back onto the actual pixels is pl_compositor_render()'s job, which is what
// the rendered variant of these tests below checks.

// Reference model, written from the documented meaning of each fit mode in
// <libplacebo/compositor.h> rather than from pl_input_place().
static void expect_fit(const struct fit_case *c, struct pl_rect2df *src,
                       struct pl_rect2df *dst)
{
    // A flipped rect is the same rect with its contents mirrored, so fit
    // using the plain rects, and mirror the result within them at the very
    // end. The focus is thus measured from the `x0`, `y0` corner of whichever
    // rect it is relative to.
    const struct pl_rect2df sr = rect_or_full(c->src_rect), dr = rect_or_full(c->dst_rect);
    struct pl_rect2df src_rect = sr, dst_rect = dr;
    pl_rect2df_normalize(&src_rect);
    pl_rect2df_normalize(&dst_rect);
    const pl_vec2df focus = focus_of(c);

    // "Like PL_FIT_PLACE, but rounding to exact pixels": raw pixels means the
    // pixel aspect ratio of neither side applies.
    float ipar = c->image_par, cpar = c->canvas_par;
    if (c->fit == PL_FIT_RAW)
        ipar = cpar = 1.0f;

    float iw, ih, cw, ch;
    logical(&iw, &ih, c->image_w, c->image_h, ipar, c->image_rot);
    logical(&cw, &ch, c->canvas_w, c->canvas_h, cpar, c->canvas_rot);
    if (c->rotation % PL_ROTATION_180 == PL_ROTATION_90)
        PL_SWAP(iw, ih);
    const float full_cw = cw, full_ch = ch;

    if (c->fit == PL_FIT_RAW) {
        // "Rounding to exact pixels": image pixels map 1:1 onto canvas
        // pixels, so both rects are cut down to whole pixels first
        src_rect = (struct pl_rect2df) {
            roundf(src_rect.x0 * iw) / iw, roundf(src_rect.y0 * ih) / ih,
            roundf(src_rect.x1 * iw) / iw, roundf(src_rect.y1 * ih) / ih,
        };
        dst_rect = (struct pl_rect2df) {
            roundf(dst_rect.x0 * cw) / cw, roundf(dst_rect.y0 * ch) / ch,
            roundf(dst_rect.x1 * cw) / cw, roundf(dst_rect.y1 * ch) / ch,
        };
    }

    // The fit is performed as though `src_rect` were the entire image, and
    // `dst_rect` the entire canvas
    iw *= pl_rect_w(src_rect);
    ih *= pl_rect_h(src_rect);
    cw *= pl_rect_w(dst_rect);
    ch *= pl_rect_h(dst_rect);

    enum pl_fit_mode mode = c->fit;
    if (mode == PL_FIT_SCALE_DOWN) // "scaled only as needed to fit inside"
        mode = (iw > cw || ih > ch) ? PL_FIT_CONTAIN : PL_FIT_PLACE;

    *src = *dst = full_rect;
    switch (mode) {
    case PL_FIT_FILL: // "stretched to the exact output size, ignoring aspect"
        break;
    case PL_FIT_CONTAIN: { // "largest size contained within the canvas"
        // The image is the smaller one, aligned within the canvas
        const float scale = fminf(cw / iw, ch / ih);
        const float dw = iw * scale / cw, dh = ih * scale / ch;
        dst->x0 = align(dw, focus.x);
        dst->y0 = align(dh, focus.y);
        dst->x1 = dst->x0 + dw;
        dst->y1 = dst->y0 + dh;
        break;
    }
    case PL_FIT_COVER: { // "smallest size that covers the canvas"
        // The canvas is the smaller one, aligned within the image
        const float scale = fmaxf(cw / iw, ch / ih);
        const float sw = cw / (iw * scale), sh = ch / (ih * scale);
        src->x0 = align(sw, focus.x);
        src->y0 = align(sh, focus.y);
        src->x1 = src->x0 + sw;
        src->y1 = src->y0 + sh;
        break;
    }
    case PL_FIT_PLACE: // "source is unscaled (as-is)"
    case PL_FIT_RAW: {
        const float dw = iw / cw, dh = ih / ch;
        if (c->fit == PL_FIT_SCALE_DOWN) {
            // The image fits inside the canvas, and is aligned within it
            dst->x0 = align(dw, focus.x);
            dst->y0 = align(dh, focus.y);
        } else {
            // The free parameter is the position on the canvas: the image is
            // centered on the focus point, however far off the canvas that is
            dst->x0 = focus.x - dw / 2;
            dst->y0 = focus.y - dh / 2;
        }
        dst->x1 = dst->x0 + dw;
        dst->y1 = dst->y0 + dh;
        break;
    }
    default:
        pl_unreachable();
    }

    *src = mix_rect(src_rect, *src);
    *dst = mix_rect(dst_rect, *dst);

    if (c->fit == PL_FIT_RAW) {
        // ... and placed, at that exact size, on whole canvas pixels. Exact
        // (half pixel) ties may be broken either way, so this is only
        // compared to within a pixel.
        const float dx = roundf(dst->x0 * full_cw) / full_cw - dst->x0;
        const float dy = roundf(dst->y0 * full_ch) / full_ch - dst->y0;
        *dst = (struct pl_rect2df) {
            dst->x0 + dx, dst->y0 + dy, dst->x1 + dx, dst->y1 + dy,
        };
    }

    // Hard invariant, for every mode: nothing is ever drawn outside of
    // `dst_rect`. An image that would overflow it is cropped instead.
    clip_rect(src, dst, dst_rect);
    if (dst->x0 >= dst->x1 || dst->y0 >= dst->y1) {
        // Entirely outside of `dst_rect`, so nothing remains at all
        *src = *dst = (struct pl_rect2df) {0};
        return;
    }

    flip_like(src, sr, src_rect);
    flip_like(dst, dr, dst_rect);
}

// Displayed sizes in pixels, ignoring the pixel aspect ratio, as used by RAW
static void raw_sizes(const struct fit_case *c, float *iw, float *ih,
                      float *cw, float *ch)
{
    logical(iw, ih, c->image_w, c->image_h, 1.0f, c->image_rot);
    logical(cw, ch, c->canvas_w, c->canvas_h, 1.0f, c->canvas_rot);
    if (c->rotation % PL_ROTATION_180 == PL_ROTATION_90)
        PL_SWAP(*iw, *ih);
}

// Aspect the placed image physically occupies on the real (unrotated) canvas.
static float physical_aspect(const struct fit_case *c, struct pl_rect2df dst)
{
    pl_rect2df_rotate(&dst, -c->canvas_rot);
    pl_rect2df_normalize(&dst);
    return (pl_rect_w(dst) * c->canvas_w) / (pl_rect_h(dst) * c->canvas_h);
}

static void check_rect(const char *name, const char *which,
                       struct pl_rect2df got, struct pl_rect2df ref,
                       float tol_x, float tol_y)
{
    if (fabsf(got.x0 - ref.x0) <= tol_x && fabsf(got.y0 - ref.y0) <= tol_y &&
        fabsf(got.x1 - ref.x1) <= tol_x && fabsf(got.y1 - ref.y1) <= tol_y)
        return;

    FAIL("%s: %s rect is (%.4f %.4f)-(%.4f %.4f), reference says "
         "(%.4f %.4f)-(%.4f %.4f)\n", name, which,
         got.x0, got.y0, got.x1, got.y1, ref.x0, ref.y0, ref.x1, ref.y1);
}

// PL_FIT_RAW must map pixels exactly onto pixels: at unit scale, with every
// canvas pixel center landing on an image pixel center. This is checked on
// pl_input_place() directly, since the reference model merely restates it.
static void check_raw_exact(const struct fit_case *c, struct pl_rect2df src,
                            struct pl_rect2df dst)
{
    float iw, ih, cw, ch;
    raw_sizes(c, &iw, &ih, &cw, &ch);

    // Every edge on whole pixels, leaving the renderer nothing to round
    const float edges[8] = {
        src.x0 * iw, src.y0 * ih, src.x1 * iw, src.y1 * ih,
        dst.x0 * cw, dst.y0 * ch, dst.x1 * cw, dst.y1 * ch,
    };
    for (int i = 0; i < PL_ARRAY_SIZE(edges); i++) {
        if (fabsf(edges[i] - roundf(edges[i])) > 1e-3f) {
            FAIL("%s: RAW %s edge at %.4f is not on a whole pixel\n",
                 c->name, i < 4 ? "src" : "dst", edges[i]);
        }
    }

    const float sx = pl_rect_w(src) * iw / (pl_rect_w(dst) * cw);
    const float sy = pl_rect_h(src) * ih / (pl_rect_h(dst) * ch);
    if (fabsf(fabsf(sx) - 1.0f) > 1e-4f || fabsf(fabsf(sy) - 1.0f) > 1e-4f)
        FAIL("%s: RAW scales the image by %.5f x %.5f\n", c->name, sx, sy);

    // Map the first canvas pixel center inside `dst` back onto the image
    const float px = floorf(fminf(dst.x0, dst.x1) * cw) + 1.5f;
    const float py = floorf(fminf(dst.y0, dst.y1) * ch) + 1.5f;
    const float ix = src.x0 * iw + (px - dst.x0 * cw) * sx;
    const float iy = src.y0 * ih + (py - dst.y0 * ch) * sy;
    if (fabsf(ix - floorf(ix) - 0.5f) > 1e-3f || fabsf(iy - floorf(iy) - 0.5f) > 1e-3f) {
        FAIL("%s: RAW maps canvas pixel center (%.1f, %.1f) to (%.4f, %.4f), "
             "between image pixels\n", c->name, px, py, ix, iy);
    }
}

// Wherever the image is larger than `dst_rect`, its focus point should land
// exactly on the center of `dst_rect`; wherever it is smaller, its center
// should land exactly on the focus point of `dst_rect`. (For cases where this
// is not prevented by an edge.) Unscaled images are always centered on the
// focus point, even where they overflow. Checked on pl_input_place() directly.
static void check_focus(const struct fit_case *c, struct pl_rect2df src,
                        struct pl_rect2df dst)
{
    const struct pl_rect2df src_rect = rect_or_full(c->src_rect);
    const struct pl_rect2df dst_rect = rect_or_full(c->dst_rect);
    const pl_vec2df focus = focus_of(c);

    // RAW only gets to within a canvas pixel: half for snapping `dst_rect`
    // to whole pixels, and another half for rounding the offset
    float cw, ch;
    logical(&cw, &ch, c->canvas_w, c->canvas_h, 1.0f, c->canvas_rot);
    const float tx = c->fit == PL_FIT_RAW ? 1.0f / cw + 1e-5f : 1e-5f;
    const float ty = c->fit == PL_FIT_RAW ? 1.0f / ch + 1e-5f : 1e-5f;
    const bool unscaled = c->fit == PL_FIT_PLACE || c->fit == PL_FIT_RAW;

    for (int axis = 0; axis < 2; axis++) {
        if (!(c->focus_centered & (axis ? FOCUS_Y : FOCUS_X)))
            continue;
        const float f   = axis ? focus.y : focus.x;
        const float tol = axis ? ty : tx;
        const float s0 = axis ? src.y0 : src.x0, s1 = axis ? src.y1 : src.x1;
        const float d0 = axis ? dst.y0 : dst.x0, d1 = axis ? dst.y1 : dst.x1;
        const float r0 = axis ? src_rect.y0 : src_rect.x0;
        const float r1 = axis ? src_rect.y1 : src_rect.x1;
        const float b0 = axis ? dst_rect.y0 : dst_rect.x0;
        const float b1 = axis ? dst_rect.y1 : dst_rect.x1;

        // The image may be cropped, so map points via `src` -> `dst`
        float got, want;
        if (unscaled || fabsf(d1 - d0) < fabsf(b1 - b0) - tol) {
            const float p = (r0 + r1) / 2;
            got  = PL_MIX(d0, d1, (p - s0) / (s1 - s0)); // the image's center...
            want = PL_MIX(b0, b1, f); // ... on the focus point of dst_rect
        } else {
            const float p = PL_MIX(r0, r1, f);
            got  = PL_MIX(d0, d1, (p - s0) / (s1 - s0)); // the focus point...
            want = (b0 + b1) / 2;                        // ... on the center
        }
        if (fabsf(got - want) > tol) {
            FAIL("%s: aligned at %c = %.5f, expected %.5f\n",
                 c->name, axis ? 'y' : 'x', got, want);
        }
    }
}

static void setup_fit_input(pl_input input, const struct fit_case *c)
{
    input->fit      = c->fit;
    input->rotation = c->rotation;
    input->src_rect = rect_or_full(c->src_rect);
    input->dst_rect = rect_or_full(c->dst_rect);
    input->focus    = focus_of(c);
}

static void run_fit_case(pl_input input, const struct fit_case *c)
{
    const struct pl_frame image = {
        .crop = { 0, 0, c->image_w, c->image_h },
        .pixel_aspect_ratio = c->image_par,
        .rotation = c->image_rot,
    };

    const struct pl_frame target = {
        .crop = { 0, 0, c->canvas_w, c->canvas_h },
        .pixel_aspect_ratio = c->canvas_par,
        .rotation = c->canvas_rot,
    };

    setup_fit_input(input, c);

    struct pl_rect2df src, dst, ref_src, ref_dst;
    pl_input_place(input, &image, &target, &src, &dst);
    expect_fit(c, &ref_src, &ref_dst);

    // RAW may break exact (half pixel) ties either way, so allow it a pixel
    float tol[4] = { 1e-5f, 1e-5f, 1e-5f, 1e-5f };
    if (c->fit == PL_FIT_RAW) {
        float iw, ih, cw, ch;
        raw_sizes(c, &iw, &ih, &cw, &ch);
        tol[0] += 1.0f / iw;
        tol[1] += 1.0f / ih;
        tol[2] += 1.0f / cw;
        tol[3] += 1.0f / ch;
    }

    if (!pl_rect_w(ref_dst)) {
        // "Both rects are empty if the image is placed entirely outside"
        if (pl_rect_w(src) || pl_rect_h(src) || pl_rect_w(dst) || pl_rect_h(dst))
            FAIL("%s: image placed off-canvas, but the rects are not empty\n", c->name);
        printf("    %-60s off-canvas\n", c->name);
        return;
    }

    check_rect(c->name, "src", src, ref_src, tol[0], tol[1]);
    check_rect(c->name, "dst", dst, ref_dst, tol[2], tol[3]);
    if (c->fit == PL_FIT_RAW)
        check_raw_exact(c, src, dst);
    check_focus(c, src, dst);

    if (c->expect_aspect) {
        const float got = physical_aspect(c, dst);
        if (fabsf(got - c->expect_aspect) > 1e-4f) {
            FAIL("%s: image physically occupies aspect %.4f, expected %.4f\n",
                 c->name, got, c->expect_aspect);
        }
    }

    printf("    %-60s src (%.3f %.3f)-(%.3f %.3f)  dst (%.3f %.3f)-(%.3f %.3f)\n",
           c->name, src.x0, src.y0, src.x1, src.y1, dst.x0, dst.y0, dst.x1, dst.y1);
}

#define IMG(w, h)  .image_w = w, .image_h = h
#define CANVAS(w, h) .canvas_w = w, .canvas_h = h

static const struct fit_case fit_cases[] = {
    // CONTAIN: shrink to fit, letterbox or pillarbox as needed
    { "CONTAIN: 16:9 image on a 1:1 canvas (letterboxed)",
      PL_FIT_CONTAIN, IMG(160, 90), CANVAS(200, 200) },
    { "CONTAIN: 9:16 image on a 1:1 canvas (pillarboxed)",
      PL_FIT_CONTAIN, IMG(90, 160), CANVAS(200, 200) },
    { "CONTAIN: matching aspect fills exactly",
      PL_FIT_CONTAIN, IMG(160, 90), CANVAS(320, 180) },
    { "CONTAIN: 1:1 image on a 2:1 canvas",
      PL_FIT_CONTAIN, IMG(100, 100), CANVAS(400, 200) },

    // COVER: crop the source instead, destination always full
    { "COVER: 16:9 image on a 1:1 canvas (crops width)",
      PL_FIT_COVER, IMG(160, 90), CANVAS(200, 200) },
    { "COVER: 9:16 image on a 1:1 canvas (crops height)",
      PL_FIT_COVER, IMG(90, 160), CANVAS(200, 200) },
    { "COVER: matching aspect crops nothing",
      PL_FIT_COVER, IMG(160, 90), CANVAS(320, 180) },

    // FILL: ignore aspect entirely
    { "FILL: ignores aspect, both rects full",
      PL_FIT_FILL, IMG(160, 90), CANVAS(200, 200) },

    // PLACE: unscaled, centered; cropped rather than overflowing the canvas
    { "PLACE: small image is centered unscaled",
      PL_FIT_PLACE, IMG(100, 50), CANVAS(200, 200) },
    { "PLACE: oversized image is cropped to the canvas",
      PL_FIT_PLACE, IMG(400, 300), CANVAS(200, 200) },

    // SCALE_DOWN: PLACE when it fits, CONTAIN when it does not
    { "SCALE_DOWN: image that fits is left unscaled",
      PL_FIT_SCALE_DOWN, IMG(100, 50), CANVAS(200, 200) },
    { "SCALE_DOWN: oversized image is contained",
      PL_FIT_SCALE_DOWN, IMG(400, 300), CANVAS(200, 200) },
    { "SCALE_DOWN: overflow on one axis only still contains",
      PL_FIT_SCALE_DOWN, IMG(100, 300), CANVAS(200, 200) },

    // RAW: like PLACE, but snapped to whole canvas pixels and ignoring PAR
    { "RAW: exact pixel placement needs no rounding",
      PL_FIT_RAW, IMG(100, 50), CANVAS(200, 200) },
    { "RAW: odd sizes are rounded to the pixel grid",
      PL_FIT_RAW, IMG(101, 51), CANVAS(200, 200) },
    { "RAW: ignores the image pixel aspect ratio",
      PL_FIT_RAW, IMG(100, 50), .image_par = 2.0f, CANVAS(200, 200) },
    { "RAW: ignores the canvas pixel aspect ratio",
      PL_FIT_RAW, IMG(100, 50), CANVAS(200, 200), .canvas_par = 2.0f },

    // Pixel aspect ratios
    { "PAR: anamorphic image (2.0) widens the logical size",
      PL_FIT_CONTAIN, IMG(100, 100), .image_par = 2.0f, CANVAS(200, 200) },
    { "PAR: sub-unit image ratio (0.5) heightens the logical size",
      PL_FIT_CONTAIN, IMG(100, 100), .image_par = 0.5f, CANVAS(200, 200) },
    { "PAR: anamorphic canvas (2.0)",
      PL_FIT_CONTAIN, IMG(100, 100), CANVAS(200, 200), .canvas_par = 2.0f },
    { "PAR: anamorphic on both sides cancels out",
      PL_FIT_CONTAIN, IMG(100, 100), .image_par = 2.0f,
      CANVAS(200, 200), .canvas_par = 2.0f },

    // Rotation
    { "ROT: input rotated 90 swaps the fitted aspect",
      PL_FIT_CONTAIN, IMG(160, 90), CANVAS(200, 200),
      .rotation = PL_ROTATION_90 },
    { "ROT: input rotated 180 is unchanged",
      PL_FIT_CONTAIN, IMG(160, 90), CANVAS(200, 200),
      .rotation = PL_ROTATION_180 },
    { "ROT: canvas rotation swaps the canvas aspect",
      PL_FIT_CONTAIN, IMG(160, 90), CANVAS(400, 200),
      .canvas_rot = PL_ROTATION_90 },
    { "ROT: frame and input rotation cancel",
      PL_FIT_CONTAIN, IMG(160, 90), .image_rot = PL_ROTATION_90,
      CANVAS(200, 200), .rotation = PL_ROTATION_90 },
    { "ROT: rotated COVER crops the other axis",
      PL_FIT_COVER, IMG(160, 90), CANVAS(200, 200),
      .rotation = PL_ROTATION_90 },

    // Rotation, checked end-to-end: the aspect the viewer actually sees, after
    // the inverse rotation pl_compositor_render() applies.
    { "ROT: 3:2 image on a 16:9 canvas rotated 90 -> occupies 2:3",
      PL_FIT_CONTAIN, IMG(300, 200), CANVAS(1600, 900),
      .canvas_rot = PL_ROTATION_90, .expect_aspect = 2.0f / 3.0f },
    { "ROT: same image on an unrotated canvas -> occupies 3:2",
      PL_FIT_CONTAIN, IMG(300, 200), CANVAS(1600, 900),
      .expect_aspect = 3.0f / 2.0f },
    { "ROT: 16:9 image tagged 90 fits as 9:16",
      PL_FIT_CONTAIN, IMG(160, 90), .image_rot = PL_ROTATION_90,
      CANVAS(200, 200), .expect_aspect = 9.0f / 16.0f },
    { "ROT: input rotated 270 matches 90 for a centered fit",
      PL_FIT_CONTAIN, IMG(160, 90), CANVAS(200, 200),
      .rotation = PL_ROTATION_270, .expect_aspect = 9.0f / 16.0f },
    { "ROT: capture 90 displayed on a canvas held at 270 still reads 16:9",
      PL_FIT_CONTAIN, IMG(160, 90), .image_rot = PL_ROTATION_90,
      CANVAS(200, 200), .canvas_rot = PL_ROTATION_270,
      .expect_aspect = 16.0f / 9.0f },
    { "ROT: creative rotation on top of a rotated capture cancels it",
      PL_FIT_CONTAIN, IMG(160, 90), .image_rot = PL_ROTATION_90,
      CANVAS(200, 200), .rotation = PL_ROTATION_270,
      .expect_aspect = 16.0f / 9.0f },

    // Rects: the fit treats `src_rect` as the whole image and `dst_rect` as
    // the whole canvas. FILL thus maps one onto the other verbatim.
    { "RECT: FILL maps src_rect onto dst_rect verbatim",
      PL_FIT_FILL, IMG(160, 90), CANVAS(200, 200),
      .src_rect = { 0.25f, 0.25f, 0.75f, 0.75f },
      .dst_rect = { 0.1f, 0.2f, 0.9f, 0.8f } },
    { "RECT: FILL ignores aspect and pixel aspect ratio",
      PL_FIT_FILL, IMG(160, 90), .image_par = 2.0f, CANVAS(200, 200),
      .dst_rect = { 0, 0, 0.5f, 0.5f } },
    { "RECT: CONTAIN fits the cropped image, not the whole one",
      PL_FIT_CONTAIN, IMG(160, 90), CANVAS(200, 200),
      .src_rect = { 0, 0, 0.5f, 1 } },
    { "RECT: CONTAIN within a sub-rect of the canvas",
      PL_FIT_CONTAIN, IMG(160, 90), CANVAS(400, 200),
      .dst_rect = { 0.5f, 0, 1, 1 } },
    { "RECT: CONTAIN a cropped image within a sub-rect",
      PL_FIT_CONTAIN, IMG(160, 90), CANVAS(400, 200),
      .src_rect = { 0.25f, 0, 1, 1 }, .dst_rect = { 0, 0.25f, 0.5f, 1 } },
    { "RECT: COVER fills exactly the sub-rect, no more",
      PL_FIT_COVER, IMG(160, 90), CANVAS(400, 200),
      .dst_rect = { 0, 0, 0.5f, 1 } },
    { "RECT: COVER a cropped image",
      PL_FIT_COVER, IMG(160, 90), CANVAS(160, 90),
      .src_rect = { 0.25f, 0, 0.75f, 1 } },
    { "RECT: PLACE a cropped image at its cropped size",
      PL_FIT_PLACE, IMG(200, 100), CANVAS(200, 200),
      .src_rect = { 0, 0, 0.5f, 0.5f } },
    { "RECT: PLACE oversized into a sub-rect is cropped to it",
      PL_FIT_PLACE, IMG(400, 300), CANVAS(400, 400),
      .dst_rect = { 0.25f, 0.25f, 0.75f, 0.75f } },
    { "RECT: SCALE_DOWN against the sub-rect, not the canvas",
      PL_FIT_SCALE_DOWN, IMG(150, 100), CANVAS(200, 200),
      .dst_rect = { 0, 0, 0.5f, 0.5f } },
    { "RECT: SCALE_DOWN against the cropped image",
      PL_FIT_SCALE_DOWN, IMG(400, 300), CANVAS(200, 200),
      .src_rect = { 0.25f, 0.25f, 0.5f, 0.5f } },
    { "RECT: RAW rounds to the full canvas pixel grid",
      PL_FIT_RAW, IMG(51, 31), CANVAS(200, 200),
      .dst_rect = { 0.1f, 0.1f, 0.6f, 0.6f } },
    { "RECT: RAW oversized into a sub-rect is cropped to it",
      PL_FIT_RAW, IMG(101, 51), CANVAS(200, 200),
      .dst_rect = { 0.1f, 0.1f, 0.6f, 0.6f } },

    // RAW with fractional rects: no scaling, and no sub-pixel offsets either
    { "RAW: fractional src_rect is cut to whole pixels",
      PL_FIT_RAW, IMG(100, 60), CANVAS(200, 200),
      .src_rect = { 0.105f, 0.0f, 0.783f, 0.71f } },
    { "RAW: dst_rect on half pixels crops whole pixels",
      PL_FIT_RAW, IMG(101, 51), CANVAS(200, 200),
      .dst_rect = { 0.1025f, 0.1025f, 0.6025f, 0.6025f } },
    { "RAW: arbitrary fractional dst_rect",
      PL_FIT_RAW, IMG(51, 31), CANVAS(200, 200),
      .dst_rect = { 0.1033f, 0.1077f, 0.7011f, 0.6091f } },
    { "RAW: fractional rects on both sides",
      PL_FIT_RAW, IMG(150, 120), CANVAS(200, 160),
      .src_rect = { 0.0333f, 0.1111f, 0.9444f, 0.8765f },
      .dst_rect = { 0.0123f, 0.0456f, 0.6789f, 0.9876f } },
    { "RAW: ignores pixel aspect ratios, with fractional rects",
      PL_FIT_RAW, IMG(101, 51), .image_par = 1.5f, CANVAS(200, 200),
      .canvas_par = 0.8f, .src_rect = { 0.11f, 0.07f, 0.9f, 0.93f },
      .dst_rect = { 0.2025f, 0.1f, 0.85f, 0.7025f } },
    { "RAW: input rotated 90, with fractional rects",
      PL_FIT_RAW, IMG(101, 51), CANVAS(200, 200), .rotation = PL_ROTATION_90,
      .src_rect = { 0.1f, 0.05f, 0.93f, 0.97f },
      .dst_rect = { 0.0525f, 0.0f, 0.5525f, 1.0f } },
    { "RAW: frame rotated 270, with fractional rects",
      PL_FIT_RAW, IMG(77, 45), .image_rot = PL_ROTATION_270, CANVAS(160, 90),
      .src_rect = { 0.0123f, 0.2f, 0.81f, 0.987f },
      .dst_rect = { 0.0333f, 0.1f, 0.9f, 0.95f } },
    { "RAW: canvas rotated 90, with a fractional dst_rect",
      PL_FIT_RAW, IMG(101, 51), CANVAS(150, 100), .canvas_rot = PL_ROTATION_90,
      .dst_rect = { 0.0125f, 0.1f, 0.9f, 0.55f } },
    { "RAW: flipped fractional src_rect",
      PL_FIT_RAW, IMG(101, 51), CANVAS(200, 200),
      .src_rect = { 0.95f, 0.0f, 0.07f, 1.0f } },
    { "RAW: oversized into a flipped dst_rect on half pixels",
      PL_FIT_RAW, IMG(201, 151), CANVAS(200, 200),
      .dst_rect = { 0.6025f, 0.1025f, 0.1025f, 0.6025f } },
    { "RAW: flipped on both sides, rotated, fractional",
      PL_FIT_RAW, IMG(120, 80), CANVAS(200, 200), .rotation = PL_ROTATION_270,
      .src_rect = { 0.9f, 0.03f, 0.1f, 0.97f },
      .dst_rect = { 0.0f, 0.8525f, 0.7025f, 0.1f } },

    // Focus: decides what remains visible wherever the image overflows
    { "FOCUS: COVER keeps the focus point centered",
      PL_FIT_COVER, IMG(160, 90), CANVAS(200, 200), FOCUS(0.35f, 0.5f),
      .focus_centered = FOCUS_X },
    { "FOCUS: COVER stops at the left edge of the image",
      PL_FIT_COVER, IMG(160, 90), CANVAS(200, 200), FOCUS(0.1f, 0.5f) },
    { "FOCUS: COVER stops at the right edge of the image",
      PL_FIT_COVER, IMG(160, 90), CANVAS(200, 200), FOCUS(1.0f, 0.5f) },
    { "FOCUS: COVER ignores it along the axis that matches",
      PL_FIT_COVER, IMG(160, 90), CANVAS(200, 200), FOCUS(0.5f, 0.0f) },
    { "FOCUS: COVER of a tall image, vertically",
      PL_FIT_COVER, IMG(90, 160), CANVAS(200, 200), FOCUS(0.5f, 0.7f),
      .focus_centered = FOCUS_Y },
    { "FOCUS: FILL ignores it",
      PL_FIT_FILL, IMG(160, 90), CANVAS(200, 200), FOCUS(0.2f, 0.8f) },
    { "FOCUS: is relative to src_rect",
      PL_FIT_COVER, IMG(160, 90), CANVAS(200, 200), FOCUS(0.4f, 0.5f),
      .src_rect = { 0.25f, 0, 1, 1 }, .focus_centered = FOCUS_X },
    { "FOCUS: follows a flipped src_rect",
      PL_FIT_COVER, IMG(160, 90), CANVAS(200, 200), FOCUS(0.4f, 0.5f),
      .src_rect = { 1, 0, 0.25f, 1 }, .focus_centered = FOCUS_X },
    { "FOCUS: is in the displayed orientation (input rotated 90)",
      PL_FIT_COVER, IMG(160, 90), CANVAS(200, 200), FOCUS(0.5f, 0.3f),
      .rotation = PL_ROTATION_90, .focus_centered = FOCUS_Y },
    { "FOCUS: is in the displayed orientation (frame rotated 270)",
      PL_FIT_COVER, IMG(160, 90), .image_rot = PL_ROTATION_270,
      CANVAS(200, 200), FOCUS(0.5f, 0.65f), .focus_centered = FOCUS_Y },
    { "FOCUS: COVER into a sub-rect centers it there",
      PL_FIT_COVER, IMG(160, 90), CANVAS(400, 200), FOCUS(0.35f, 0.5f),
      .dst_rect = { 0.5f, 0, 1, 1 }, .focus_centered = FOCUS_X },
    { "FOCUS: on a rotated canvas",
      PL_FIT_COVER, IMG(160, 90), CANVAS(200, 400), FOCUS(0.5f, 0.45f),
      .canvas_rot = PL_ROTATION_90, .focus_centered = FOCUS_Y },
    { "FOCUS: a flipped dst_rect mirrors the picture in place",
      PL_FIT_COVER, IMG(160, 90), CANVAS(200, 200), FOCUS(0.35f, 0.5f),
      .dst_rect = { 1, 0, 0, 1 }, .focus_centered = FOCUS_X },

    // ... and, mirroring that, where the image falls short of `dst_rect`,
    // decides where within `dst_rect` the image is placed
    { "FOCUS: CONTAIN aligns to the left",
      PL_FIT_CONTAIN, IMG(100, 100), CANVAS(320, 180), FOCUS(0.0f, 0.5f) },
    { "FOCUS: CONTAIN aligns to the right",
      PL_FIT_CONTAIN, IMG(100, 100), CANVAS(320, 180), FOCUS(1.0f, 0.5f) },
    { "FOCUS: CONTAIN centers the image on the focus point",
      PL_FIT_CONTAIN, IMG(100, 100), CANVAS(320, 180), FOCUS(0.3f, 0.5f),
      .focus_centered = FOCUS_X },
    { "FOCUS: CONTAIN aligns a letterboxed image to the bottom",
      PL_FIT_CONTAIN, IMG(160, 90), CANVAS(200, 200), FOCUS(0.0f, 1.0f) },
    { "FOCUS: CONTAIN aligns within a sub-rect",
      PL_FIT_CONTAIN, IMG(100, 100), CANVAS(400, 200), FOCUS(1.0f, 0.5f),
      .dst_rect = { 0, 0, 0.75f, 1 } },
    { "FOCUS: CONTAIN is measured from the x0 of a flipped dst_rect",
      PL_FIT_CONTAIN, IMG(100, 100), CANVAS(320, 180), FOCUS(0.3f, 0.5f),
      .dst_rect = { 1, 0, 0, 1 }, .focus_centered = FOCUS_X },
    { "FOCUS: CONTAIN ignores the flip of src_rect",
      PL_FIT_CONTAIN, IMG(100, 100), CANVAS(320, 180), FOCUS(0.3f, 0.5f),
      .src_rect = { 1, 0, 0, 1 }, .focus_centered = FOCUS_X },
    { "FOCUS: CONTAIN on a rotated canvas",
      PL_FIT_CONTAIN, IMG(100, 100), CANVAS(180, 320), FOCUS(0.3f, 0.5f),
      .canvas_rot = PL_ROTATION_90, .focus_centered = FOCUS_X },
    { "FOCUS: SCALE_DOWN aligns to the corner",
      PL_FIT_SCALE_DOWN, IMG(400, 300), CANVAS(200, 200), FOCUS(0.0f, 0.0f) },
    { "FOCUS: SCALE_DOWN keeps a fitting image inside the canvas",
      PL_FIT_SCALE_DOWN, IMG(40, 40), CANVAS(200, 200), FOCUS(-1.0f, 0.5f) },

    // Unscaled images are instead always centered on the focus point of
    // `dst_rect`, however far off it that places them
    { "FOCUS: oversized PLACE is centered on the focus point",
      PL_FIT_PLACE, IMG(400, 300), CANVAS(200, 200), FOCUS(0.3f, 0.6f),
      .focus_centered = FOCUS_X | FOCUS_Y },
    { "FOCUS: oversized PLACE is centered on a corner of the canvas",
      PL_FIT_PLACE, IMG(400, 300), CANVAS(200, 200), FOCUS(0.0f, 1.0f),
      .focus_centered = FOCUS_X | FOCUS_Y },
    { "FOCUS: PLACE that fits is centered on a corner of the canvas",
      PL_FIT_PLACE, IMG(100, 50), CANVAS(200, 200), FOCUS(0.0f, 0.0f),
      .focus_centered = FOCUS_X | FOCUS_Y },
    { "FOCUS: PLACE that fits centers the image on the focus point",
      PL_FIT_PLACE, IMG(100, 50), CANVAS(200, 200), FOCUS(0.3f, 0.8f),
      .focus_centered = FOCUS_X | FOCUS_Y },
    { "FOCUS: RAW that fits centers the image on the focus point",
      PL_FIT_RAW, IMG(51, 31), CANVAS(200, 200), FOCUS(0.25f, 0.75f),
      .focus_centered = FOCUS_X | FOCUS_Y },
    { "FOCUS: oversized RAW is centered on the focus point",
      PL_FIT_RAW, IMG(400, 300), CANVAS(200, 200), FOCUS(0.3f, 0.6f),
      .focus_centered = FOCUS_X | FOCUS_Y },
    { "FOCUS: oversized RAW is centered on a corner of the canvas",
      PL_FIT_RAW, IMG(201, 151), CANVAS(200, 100), FOCUS(1.0f, 0.0f),
      .focus_centered = FOCUS_X | FOCUS_Y },
    { "FOCUS: PLACE on the left edge shows only the right half",
      PL_FIT_PLACE, IMG(200, 100), CANVAS(200, 200), FOCUS(0.0f, 0.5f),
      .focus_centered = FOCUS_X | FOCUS_Y },
    { "FOCUS: PLACE goes partially off the canvas",
      PL_FIT_PLACE, IMG(40, 40), CANVAS(200, 200), FOCUS(-0.05f, 1.08f),
      .focus_centered = FOCUS_X | FOCUS_Y },
    { "FOCUS: PLACE goes entirely off the canvas",
      PL_FIT_PLACE, IMG(40, 40), CANVAS(200, 200), FOCUS(-1.0f, 0.5f) },
    { "FOCUS: PLACE goes entirely off the canvas, past the far edge",
      PL_FIT_PLACE, IMG(40, 40), CANVAS(200, 200), FOCUS(0.5f, 2.0f) },
    { "FOCUS: RAW goes partially off the canvas",
      PL_FIT_RAW, IMG(41, 31), CANVAS(200, 200), FOCUS(1.04f, -0.03f),
      .focus_centered = FOCUS_X | FOCUS_Y },
    { "FOCUS: oversized RAW goes entirely off the canvas",
      PL_FIT_RAW, IMG(301, 151), CANVAS(200, 200), FOCUS(2.0f, 0.5f) },
    { "FOCUS: PLACE goes off a sub-rect, without leaving it",
      PL_FIT_PLACE, IMG(100, 100), CANVAS(400, 200), FOCUS(-0.1f, 0.5f),
      .dst_rect = { 0.5f, 0, 1, 1 }, .focus_centered = FOCUS_X | FOCUS_Y },
    { "FOCUS: PLACE off a flipped dst_rect shows the mirrored overhang",
      PL_FIT_PLACE, IMG(40, 40), CANVAS(200, 200), FOCUS(-0.05f, 0.5f),
      .dst_rect = { 1, 0, 0, 1 }, .focus_centered = FOCUS_X | FOCUS_Y },
    { "FOCUS: RAW off a flipped src_rect shows the mirrored overhang",
      PL_FIT_RAW, IMG(41, 31), CANVAS(200, 200), FOCUS(1.04f, 0.5f),
      .src_rect = { 1, 0, 0, 1 }, .focus_centered = FOCUS_X | FOCUS_Y },
    { "FOCUS: PLACE is measured from the x0 of a flipped dst_rect",
      PL_FIT_PLACE, IMG(40, 40), CANVAS(200, 200), FOCUS(0.2f, 0.5f),
      .dst_rect = { 1, 0, 0, 1 }, .focus_centered = FOCUS_X },
    { "FOCUS: PLACE ignores the flip of src_rect",
      PL_FIT_PLACE, IMG(400, 300), CANVAS(200, 200), FOCUS(0.2f, 0.5f),
      .src_rect = { 1, 0, 0, 1 }, .focus_centered = FOCUS_X },
    { "FOCUS: PLACE on a rotated canvas",
      PL_FIT_PLACE, IMG(40, 40), CANVAS(100, 200), FOCUS(0.0f, 0.25f),
      .canvas_rot = PL_ROTATION_90, .focus_centered = FOCUS_X | FOCUS_Y },
    { "FOCUS: PLACE of a rotated input",
      PL_FIT_PLACE, IMG(300, 60), CANVAS(200, 200), FOCUS(0.1f, 0.9f),
      .rotation = PL_ROTATION_90, .focus_centered = FOCUS_X | FOCUS_Y },
    { "FOCUS: RAW, overflowing on one axis and aligned on the other",
      PL_FIT_RAW, IMG(300, 100), CANVAS(200, 200), FOCUS(0.4f, 0.3f),
      .focus_centered = FOCUS_X | FOCUS_Y },
    { "FOCUS: RAW into a flipped, fractional dst_rect",
      PL_FIT_RAW, IMG(201, 151), CANVAS(200, 200), FOCUS(0.6f, 0.4f),
      .dst_rect = { 0.6025f, 0.1025f, 0.1025f, 0.6025f },
      .focus_centered = FOCUS_X | FOCUS_Y },

    // Rects under rotation: both are in the displayed orientation, so an
    // off-center rect must select the same content the viewer sees there.
    { "RECT+ROT: src_rect crops the input-rotated (90) image",
      PL_FIT_FILL, IMG(160, 90), CANVAS(200, 200),
      .rotation = PL_ROTATION_90, .src_rect = { 0, 0, 0.5f, 1 } },
    { "RECT+ROT: src_rect crops the input-rotated (270) image",
      PL_FIT_FILL, IMG(160, 90), CANVAS(200, 200),
      .rotation = PL_ROTATION_270, .src_rect = { 0, 0, 0.5f, 1 } },
    { "RECT+ROT: src_rect crops the input-rotated (180) image",
      PL_FIT_FILL, IMG(160, 90), CANVAS(200, 200),
      .rotation = PL_ROTATION_180, .src_rect = { 0, 0, 0.5f, 0.5f } },
    { "RECT+ROT: src_rect crops the frame-rotated (90) image",
      PL_FIT_FILL, IMG(160, 90), .image_rot = PL_ROTATION_90,
      CANVAS(200, 200), .src_rect = { 0, 0, 0.5f, 1 } },
    { "RECT+ROT: src_rect crops the doubly rotated (90+90) image",
      PL_FIT_FILL, IMG(160, 90), .image_rot = PL_ROTATION_90,
      CANVAS(200, 200), .rotation = PL_ROTATION_90,
      .src_rect = { 0, 0, 0.5f, 0.5f } },
    { "RECT+ROT: dst_rect is relative to the rotated canvas",
      PL_FIT_CONTAIN, IMG(160, 90), CANVAS(400, 200),
      .canvas_rot = PL_ROTATION_90, .dst_rect = { 0, 0, 1, 0.5f } },
    { "RECT+ROT: COVER a rotated, cropped image into a sub-rect",
      PL_FIT_COVER, IMG(160, 90), CANVAS(400, 200),
      .rotation = PL_ROTATION_90, .src_rect = { 0, 0.5f, 1, 1 },
      .dst_rect = { 0.5f, 0, 1, 1 } },

    // Flipped rects: the same rect, with its contents mirrored
    { "FLIP: src_rect flipped horizontally mirrors the image",
      PL_FIT_FILL, IMG(160, 90), CANVAS(200, 200),
      .src_rect = { 1, 0, 0, 1 } },
    { "FLIP: src_rect flipped vertically mirrors the image",
      PL_FIT_FILL, IMG(160, 90), CANVAS(200, 200),
      .src_rect = { 0, 1, 1, 0 } },
    { "FLIP: dst_rect flipped horizontally mirrors the image",
      PL_FIT_FILL, IMG(160, 90), CANVAS(200, 200),
      .dst_rect = { 1, 0, 0, 1 } },
    { "FLIP: flipping both rects cancels out",
      PL_FIT_FILL, IMG(160, 90), CANVAS(200, 200),
      .src_rect = { 1, 0, 0, 1 }, .dst_rect = { 1, 0, 0, 1 } },
    { "FLIP: CONTAIN a mirrored crop within a sub-rect",
      PL_FIT_CONTAIN, IMG(160, 90), CANVAS(400, 200),
      .src_rect = { 0.75f, 0, 0, 1 }, .dst_rect = { 0, 0.25f, 0.5f, 1 } },
    { "FLIP: COVER a mirrored crop into a sub-rect",
      PL_FIT_COVER, IMG(160, 90), CANVAS(400, 200),
      .src_rect = { 0.5f, 0, 0, 1 }, .dst_rect = { 0.5f, 0, 1, 1 } },
    { "FLIP: COVER into a doubly flipped sub-rect",
      PL_FIT_COVER, IMG(160, 90), CANVAS(400, 200),
      .dst_rect = { 1, 1, 0.5f, 0 } },
    { "FLIP: oversized PLACE into a flipped sub-rect is cropped to it",
      PL_FIT_PLACE, IMG(400, 300), CANVAS(400, 400),
      .dst_rect = { 0.75f, 0.25f, 0.25f, 0.75f } },
    { "FLIP: SCALE_DOWN against a flipped sub-rect",
      PL_FIT_SCALE_DOWN, IMG(150, 100), CANVAS(200, 200),
      .dst_rect = { 0.5f, 0.5f, 0, 0 } },
    { "FLIP: RAW into a flipped sub-rect",
      PL_FIT_RAW, IMG(51, 31), CANVAS(200, 200),
      .dst_rect = { 0.6f, 0.1f, 0.1f, 0.6f } },
    { "FLIP: oversized RAW into a flipped sub-rect is cropped to it",
      PL_FIT_RAW, IMG(101, 51), CANVAS(200, 200),
      .dst_rect = { 0.1f, 0.6f, 0.6f, 0.1f } },
    { "FLIP+ROT: mirrored crop of an input-rotated (90) image",
      PL_FIT_FILL, IMG(160, 90), CANVAS(200, 200),
      .rotation = PL_ROTATION_90, .src_rect = { 0.5f, 0, 0, 1 } },
    { "FLIP+ROT: mirrored crop of a frame-rotated (270) image",
      PL_FIT_FILL, IMG(160, 90), .image_rot = PL_ROTATION_270,
      CANVAS(200, 200), .src_rect = { 0, 0.5f, 1, 0 } },
    { "FLIP+ROT: flipped dst_rect on a rotated canvas",
      PL_FIT_CONTAIN, IMG(160, 90), CANVAS(400, 200),
      .canvas_rot = PL_ROTATION_90, .dst_rect = { 1, 0, 0, 0.5f } },
};

// Animating the focus point of an unscaled image from -1 to 2 must scroll it
// onto the canvas and off it again, at a constant size
static void run_scroll_test(pl_input input, enum pl_fit_mode fit)
{
    enum { IW = 40, IH = 20, CW = 200, CH = 100 };
    const struct pl_frame image  = { .crop = { 0, 0, IW, IH } };
    const struct pl_frame target = { .crop = { 0, 0, CW, CH } };
    const char *name = fit == PL_FIT_RAW ? "RAW" : "PLACE";
    const float w = (float) IW / CW, tol = fit == PL_FIT_RAW ? 1.0f / CW : 1e-5f;

    input->fit      = fit;
    input->src_rect = input->dst_rect = full_rect;
    input->rotation = PL_ROTATION_0;

    bool seen = false;
    for (int i = 0; i <= 60; i++) {
        const float f = -1.0f + i * 0.05f;
        input->focus = (pl_vec2df) { f, 0.5f };
        struct pl_rect2df src, dst;
        pl_input_place(input, &image, &target, &src, &dst);

        // The visible part of an image of width `w`, centered on `f`
        const float x0 = PL_MAX(f - w / 2, 0.0f), x1 = PL_MIN(f + w / 2, 1.0f);
        if (x1 - x0 <= tol) {
            if (pl_rect_w(dst) > tol)
                FAIL("%s scroll: visible at focus %.2f, expected off-canvas\n", name, f);
            continue;
        }

        seen = true;
        const float scale = pl_rect_w(dst) * CW / (pl_rect_w(src) * IW);
        if (fabsf(dst.x0 - x0) > tol || fabsf(dst.x1 - x1) > tol ||
            fabsf(src.x0 - (x0 - f + w / 2) / w) > tol / w ||
            fabsf(scale - 1.0f) > 1e-3f || fabsf(pl_rect_h(dst) - (float) IH / CH) > tol)
        {
            FAIL("%s scroll: at focus %.2f, placed src (%.4f)-(%.4f) dst "
                 "(%.4f)-(%.4f), expected dst (%.4f)-(%.4f) at its own size\n",
                 name, f, src.x0, src.x1, dst.x0, dst.x1, x0, x1);
        }
    }

    if (!seen)
        FAIL("%s scroll: never visible\n", name);
    printf("    %-60s ok\n", fit == PL_FIT_RAW ? "RAW scrolls across the canvas"
                                                : "PLACE scrolls across the canvas");
}

static void run_fit_tests(pl_log log, pl_gpu gpu)
{
    pl_compositor comp = pl_compositor_create(log);
    pl_input input = pl_compositor_input_create(comp);
    if (input->focus.x != 0.5f || input->focus.y != 0.5f)
        FAIL("default focus is (%f, %f), not the center\n", input->focus.x, input->focus.y);

    printf("--- fit modes\n");
    for (int i = 0; i < PL_ARRAY_SIZE(fit_cases); i++)
        run_fit_case(input, &fit_cases[i]);
    printf("\n");

    printf("--- scrolling unscaled images via the focus point\n");
    run_scroll_test(input, PL_FIT_PLACE);
    run_scroll_test(input, PL_FIT_RAW);
    printf("\n");

    pl_compositor_destroy(&comp);
}

//----------------------------------------------------------------------------
// Fit modes, rendered
//----------------------------------------------------------------------------

// The rects above are only half the story: pl_compositor_render() still has to
// map them back onto the frames' actual pixels, and a mistake there (say, a
// mirrored axis) leaves every rect and aspect ratio above intact. So render
// each case for real, with an image split into four distinctly colored
// quadrants, and compare every canvas pixel against the reference.

enum { BG = '.', RED = 'R', GREEN = 'G', BLUE = 'B', WHITE = 'W' };

static const char quadrant_colors[2][2] = {
    { RED, GREEN },
    { BLUE, WHITE },
};

// Map a normalized point in a frame's displayed orientation back onto its
// actual pixels, one clockwise 90 degree turn at a time. (Rotating a frame
// clockwise puts its bottom left corner at the top left.)
static void unrotate(float *x, float *y, pl_rotation rot)
{
    for (int i = 0; i < pl_rotation_normalize(rot); i++) {
        const float u = *x, v = *y;
        *x = v;
        *y = 1.0f - u;
    }
}

// Inverse of `unrotate`.
static void rotate(float *x, float *y, pl_rotation rot)
{
    for (int i = 0; i < pl_rotation_normalize(rot); i++) {
        const float px = *x, py = *y;
        *x = 1.0f - py;
        *y = px;
    }
}

static char classify(const uint8_t *p)
{
    const bool r = p[0] > 128, g = p[1] > 128, b = p[2] > 128;
    if (!r && !g && !b) return BG;
    if (r && g && b)    return WHITE;
    if (r && !g && !b)  return RED;
    if (!r && g && !b)  return GREEN;
    if (!r && !g && b)  return BLUE;
    return '?';
}

// What the reference model says canvas pixel (x, y) should show, or 0 if it
// lies too close to an edge for nearest-neighbour sampling to be unambiguous.
// Sets `*outside` if the pixel is outside of `dst_rect` entirely.
static char expect_pixel(const struct fit_case *c, struct pl_rect2df src,
                         struct pl_rect2df dst, int x, int y, bool *outside)
{
    static const float margin = 1.5f; // in pixels

    // Canvas pixel -> displayed canvas orientation
    float u = (x + 0.5f) / c->canvas_w, v = (y + 0.5f) / c->canvas_h;
    rotate(&u, &v, c->canvas_rot);

    const bool swap_c = c->canvas_rot % PL_ROTATION_180 == PL_ROTATION_90;
    const float eu = margin / (swap_c ? c->canvas_h : c->canvas_w);
    const float ev = margin / (swap_c ? c->canvas_w : c->canvas_h);
    struct pl_rect2df dst_rect = rect_or_full(c->dst_rect);
    pl_rect2df_normalize(&dst_rect);
    *outside = u < dst_rect.x0 - eu || u > dst_rect.x1 + eu ||
               v < dst_rect.y0 - ev || v > dst_rect.y1 + ev;

    if (fabsf(u - dst.x0) < eu || fabsf(u - dst.x1) < eu ||
        fabsf(v - dst.y0) < ev || fabsf(v - dst.y1) < ev)
        return 0;
    struct pl_rect2df bounds = dst;
    pl_rect2df_normalize(&bounds);
    if (u < bounds.x0 || u > bounds.x1 || v < bounds.y0 || v > bounds.y1)
        return BG;

    // Displayed canvas -> displayed image orientation -> image pixels. Any
    // flip in either rect is a sign change in this linear map.
    float s = PL_MIX(src.x0, src.x1, (u - dst.x0) / pl_rect_w(dst));
    float t = PL_MIX(src.y0, src.y1, (v - dst.y0) / pl_rect_h(dst));
    unrotate(&s, &t, c->image_rot + c->rotation);

    if (fabsf(s - 0.5f) < margin / c->image_w ||
        fabsf(t - 0.5f) < margin / c->image_h)
        return 0;
    return quadrant_colors[t >= 0.5f][s >= 0.5f];
}

// Print the canvas as seen and as expected, downsampled, to make failures
// legible at a glance.
static void dump_canvas(const struct fit_case *c, struct pl_rect2df src,
                        struct pl_rect2df dst, const uint8_t *data)
{
    const int cols = 48, rows = 24;
    fprintf(stderr, "    got:%*s expected:\n", cols - 3, "");
    for (int j = 0; j < rows; j++) {
        const int y = (j + 0.5f) * c->canvas_h / rows;
        fprintf(stderr, "    ");
        for (int i = 0; i < cols; i++) {
            const int x = (i + 0.5f) * c->canvas_w / cols;
            fputc(classify(&data[(y * (int) c->canvas_w + x) * 4]), stderr);
        }
        fprintf(stderr, "  ");
        for (int i = 0; i < cols; i++) {
            const int x = (i + 0.5f) * c->canvas_w / cols;
            bool outside;
            const char e = expect_pixel(c, src, dst, x, y, &outside);
            fputc(e ? e : ' ', stderr);
        }
        fputc('\n', stderr);
    }
}

// For PL_FIT_RAW, which must not resample at all, every texel instead gets a
// pseudo-random color, so that blending any two (from a sub-pixel offset or
// the slightest scaling) produces a color that matches neither of them.
static void texel_color(uint8_t *p, int x, int y)
{
    uint32_t h = (uint32_t) x * 0x9E3779B1u ^ (uint32_t) y * 0x85EBCA77u;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    p[0] = h;
    p[1] = h >> 8;
    p[2] = h >> 16;
    p[3] = 255;
}

// The image texel that canvas pixel (x, y) should show exactly, if any. Pixels
// on the edges of `dst` are skipped, since the renderer gets to round those.
static bool expect_texel(const struct fit_case *c, struct pl_rect2df src,
                         struct pl_rect2df dst, int x, int y, int *tx, int *ty)
{
    float u = (x + 0.5f) / c->canvas_w, v = (y + 0.5f) / c->canvas_h;
    rotate(&u, &v, c->canvas_rot);

    const bool swap_c = c->canvas_rot % PL_ROTATION_180 == PL_ROTATION_90;
    const float eu = 1.0f / (swap_c ? c->canvas_h : c->canvas_w);
    const float ev = 1.0f / (swap_c ? c->canvas_w : c->canvas_h);
    struct pl_rect2df bounds = dst;
    pl_rect2df_normalize(&bounds);
    if (u < bounds.x0 + eu || u > bounds.x1 - eu ||
        v < bounds.y0 + ev || v > bounds.y1 - ev)
        return false;

    float s = PL_MIX(src.x0, src.x1, (u - dst.x0) / pl_rect_w(dst));
    float t = PL_MIX(src.y0, src.y1, (v - dst.y0) / pl_rect_h(dst));
    unrotate(&s, &t, c->image_rot + c->rotation);
    *tx = floorf(s * c->image_w);
    *ty = floorf(t * c->image_h);
    return true;
}

static void render_fit_case(pl_log log, pl_gpu gpu, const struct fit_case *c,
                            bool exact)
{
    const int iw = c->image_w, ih = c->image_h;
    const int cw = c->canvas_w, ch = c->canvas_h;
    uint8_t *pixels = malloc(PL_MAX(iw * ih, cw * ch) * 4);
    if (!pixels)
        FAIL("%s: out of memory\n", c->name);

    for (int y = 0; y < ih; y++) {
        for (int x = 0; x < iw; x++) {
            if (exact) {
                texel_color(&pixels[(y * iw + x) * 4], x, y);
                continue;
            }
            const char q = quadrant_colors[y >= ih / 2.0f - 0.5f][x >= iw / 2.0f - 0.5f];
            uint8_t *p = &pixels[(y * iw + x) * 4];
            p[0] = (q == RED   || q == WHITE) ? 255 : 0;
            p[1] = (q == GREEN || q == WHITE) ? 255 : 0;
            p[2] = (q == BLUE  || q == WHITE) ? 255 : 0;
            p[3] = 255;
        }
    }

    pl_fmt fmt = pl_find_named_fmt(gpu, "rgba8");
    pl_tex img_tex = pl_tex_create(gpu, pl_tex_params(
        .w = iw, .h = ih, .format = fmt, .sampleable = true,
        .initial_data = pixels,
    ));
    pl_tex dst_tex = pl_tex_create(gpu, pl_tex_params(
        .w = cw, .h = ch, .format = fmt, .renderable = true,
        .host_readable = true, .blit_dst = true,
    ));
    if (!img_tex || !dst_tex)
        FAIL("%s: failed creating textures\n", c->name);
    pl_tex_clear(gpu, dst_tex, (float[4]) {0});

    static struct pl_frame image;
    image = (struct pl_frame) {
        .num_planes = 1,
        .planes[0] = { .texture = img_tex, .components = 4,
                       .component_mapping = {0, 1, 2, 3} },
        .repr  = pl_color_repr_rgb,
        .color = pl_color_space_srgb,
        .crop  = { 0, 0, iw, ih },
        .pixel_aspect_ratio = c->image_par,
        .rotation = c->image_rot,
    };

    const struct pl_frame target = {
        .num_planes = 1,
        .planes[0] = { .texture = dst_tex, .components = 4,
                       .component_mapping = {0, 1, 2, 3} },
        .repr  = pl_color_repr_rgb,
        .color = pl_color_space_srgb,
        .crop  = { 0, 0, cw, ch },
        .pixel_aspect_ratio = c->canvas_par,
        .rotation = c->canvas_rot,
    };

    pl_compositor comp = pl_compositor_create(log);
    pl_input input = pl_compositor_input_create(comp);
    pl_compositor_init(comp, gpu);
    setup_fit_input(input, c);

    struct pl_source_frame src = {
        .pts = 0.0, .duration = 1.0, .map = map_frame, .frame_data = &image,
    };
    pl_queue_push(input->queue, &src);
    pl_queue_push(input->queue, NULL);

    // Nearest neighbour sampling keeps every quadrant a solid color. For
    // the exact check, use a filter that would visibly blend texels instead.
    const struct pl_filter_config *filter = exact ? &pl_filter_bilinear
                                                  : &pl_filter_nearest;
    const struct pl_render_params rparams = { PL_RENDER_DEFAULTS
        .upscaler   = filter,
        .downscaler = filter,
    };
    input->params = &rparams;

    const struct pl_queue_params params = { PL_QUEUE_DEFAULTS
        .pts = 0.0, .vsync_duration = 1/30.0,
    };

    if (!pl_compositor_render(comp, &target, &params, NULL))
        FAIL("%s: rendering failed\n", c->name);
    if (!pl_tex_download(gpu, pl_tex_transfer_params(.tex = dst_tex, .ptr = pixels)))
        FAIL("%s: failed downloading the canvas\n", c->name);

    struct pl_rect2df ref_src, ref_dst;
    expect_fit(c, &ref_src, &ref_dst);

    // The exact check is about whether these rects are rendered pixel for
    // pixel, so use the ones actually placed rather than the reference -
    // which may break exact ties the other way
    struct pl_rect2df got_src, got_dst;
    pl_input_place(input, &image, &target, &got_src, &got_dst);

    for (int y = 0; exact && y < ch; y++) {
        for (int x = 0; x < cw; x++) {
            int tx, ty;
            if (!expect_texel(c, got_src, got_dst, x, y, &tx, &ty))
                continue;

            uint8_t want[4];
            texel_color(want, tx, ty);
            const uint8_t *got = &pixels[(y * cw + x) * 4];
            if (abs(got[0] - want[0]) <= 1 && abs(got[1] - want[1]) <= 1 &&
                abs(got[2] - want[2]) <= 1)
                continue;

            FAIL("%s: canvas pixel (%d, %d) is #%02x%02x%02x, expected "
                 "image pixel (%d, %d) = #%02x%02x%02x exactly\n", c->name,
                 x, y, got[0], got[1], got[2], tx, ty, want[0], want[1], want[2]);
        }
    }

    for (int y = 0; !exact && y < ch; y++) {
        for (int x = 0; x < cw; x++) {
            bool outside;
            const char want = expect_pixel(c, ref_src, ref_dst, x, y, &outside);
            const char got = classify(&pixels[(y * cw + x) * 4]);
            if (!want || got == want)
                continue;

            fprintf(stderr, "=== FAILED: %s: canvas pixel (%d, %d) is '%c', "
                    "expected '%c'%s\n", c->name, x, y, got, want,
                    outside && got != BG ? " - drawn outside of dst_rect" : "");
            dump_canvas(c, ref_src, ref_dst, pixels);
            exit(1);
        }
    }

    printf("    %-60s %s\n", c->name, exact ? "exact" : "ok");
    pl_compositor_destroy(&comp);
    pl_tex_destroy(gpu, &img_tex);
    pl_tex_destroy(gpu, &dst_tex);
    free(pixels);
}

//----------------------------------------------------------------------------
// Render params, rendered
//----------------------------------------------------------------------------

// Shared by the tests below
static const struct pl_color_adjustment desaturate = {
    PL_COLOR_ADJUSTMENT_NEUTRAL
    .saturation = 0.5f,
};

static const struct pl_distort_params tilt = {
    .transform.mat.m = {{ 0.9f, -0.3f }, { 0.3f, 0.9f }},
};

// Canvas-level options, which are ignored per input
static const struct pl_render_params params_own_canvas = { PL_RENDER_DEFAULTS
    .background       = PL_CLEAR_COLOR,
    .background_color = { 1.0f, 0.0f, 1.0f },
    .border           = PL_CLEAR_TILES,
};

// Up to two layers of solid color covering the canvas, red below green. The
// canvas has a blue background and border, and the target starts out magenta.
struct params_case {
    const char *name;
    int num_layers;                   // defaults to 2
    uint8_t alpha[2];
    float red, green, blue;           // expected, in linear light
    enum pl_clear_mode background;    // canvas, defaults to PL_CLEAR_COLOR
    enum pl_clear_mode border;        // canvas, defaults to PL_CLEAR_COLOR
    bool half;                        // layers only cover the left half
    bool stale;                       // render once with a blue border first
    bool direct;                      // without an intermediate FBO
    bool deprecated;                  // skip the border using `skip_target_clearing`
    const struct pl_render_params *base_params, *top_params;
    bool expect_tile;
};

static const struct params_case params_cases[] = {
    { "opaque top layer replaces the base",
      .alpha = { 255, 255 }, .green = 1.0f },
    { "top layer alpha blends in linear light",
      .alpha = { 255, 128 }, .red = 0.498f, .green = 0.502f },
    { "... or in gamma light without an FBO",
      .alpha = { 255, 128 }, .red = 0.2122f, .green = 0.2158f, .direct = true },
    { "canvas background shows below a single transparent layer",
      .num_layers = 1, .alpha = { 0 }, .blue = 1.0f },
    { "... and below transparent overlays, in linear light",
      .alpha = { 0, 128 }, .green = 0.502f, .blue = 0.498f },
    { "canvas border may be tiles",
      .num_layers = 1, .alpha = { 255 }, .half = true,
      .border = PL_CLEAR_TILES, .expect_tile = true },
    { "canvas border is blurred from the lowest layer",
      .alpha = { 255, 255 }, .half = true, .border = PL_CLEAR_BLUR, .red = 1.0f },
    { "skipped canvas border leaves the target alone",
      .alpha = { 255, 255 }, .half = true, .border = PL_CLEAR_SKIP,
      .red = 1.0f, .blue = 1.0f },
    { "... even after drawing a border before",
      .alpha = { 255, 255 }, .half = true, .border = PL_CLEAR_SKIP,
      .stale = true, .red = 1.0f, .blue = 1.0f },
    { "... and blends the canvas onto the target",
      .alpha = { 0, 0 }, .border = PL_CLEAR_SKIP, .background = PL_CLEAR_SKIP,
      .red = 1.0f, .blue = 1.0f },
    { "... with the background still drawn below the inputs",
      .alpha = { 0, 0 }, .border = PL_CLEAR_SKIP, .blue = 1.0f },
    { "... also when using `skip_target_clearing`",
      .alpha = { 0, 0 }, .deprecated = true, .red = 1.0f, .blue = 1.0f },
    { "per-input backgrounds are ignored",
      .alpha = { 0, 0 }, .blue = 1.0f,
      .base_params = &params_own_canvas, .top_params = &params_own_canvas },
    { "per-input borders are ignored",
      .num_layers = 1, .alpha = { 255 }, .half = true, .blue = 1.0f,
      .base_params = &params_own_canvas },
};

static float srgb_encode(float x)
{
    return x <= 0.0031308f ? 12.92f * x : 1.055f * powf(x, 1 / 2.4f) - 0.055f;
}

static pl_tex canvas_tex(pl_gpu gpu, int w, int h)
{
    return pl_tex_create(gpu, pl_tex_params(
        .w = w, .h = h, .format = pl_find_named_fmt(gpu, "rgba8"),
        .renderable = true, .host_readable = true, .blit_dst = true,
    ));
}

// An sRGB canvas, ignoring the texture's alpha channel
static struct pl_frame canvas_frame(pl_tex tex)
{
    return (struct pl_frame) {
        .num_planes = 1,
        .planes[0]  = { .texture = tex, .components = 3,
                        .component_mapping = {0, 1, 2} },
        .repr       = pl_color_repr_rgb,
        .color      = pl_color_space_srgb,
    };
}

// An sRGB image, with independent alpha if `alpha`
static struct pl_frame image_frame(pl_tex tex, bool alpha)
{
    struct pl_frame frame = {
        .num_planes = 1,
        .planes[0]  = { .texture = tex, .components = 4,
                        .component_mapping = {0, 1, 2, 3} },
        .repr       = pl_color_repr_rgb,
        .color      = pl_color_space_srgb,
    };
    frame.repr.alpha = alpha ? PL_ALPHA_INDEPENDENT : PL_ALPHA_NONE;
    return frame;
}

static pl_tex solid_tex(pl_gpu gpu, int r, int g, int b, int a)
{
    enum { N = 16 };
    uint8_t data[N * N * 4];
    for (int i = 0; i < N * N; i++) {
        data[i * 4 + 0] = r;
        data[i * 4 + 1] = g;
        data[i * 4 + 2] = b;
        data[i * 4 + 3] = a;
    }
    return pl_tex_create(gpu, pl_tex_params(
        .w = N, .h = N, .format = pl_find_named_fmt(gpu, "rgba8"),
        .sampleable = true, .initial_data = data,
    ));
}

// Composites `c` and returns the resulting (sRGB-encoded) color at the center
// of the canvas, or the center of its right half if `c->half`
static void composite_pixel(pl_log log, pl_gpu gpu, const struct params_case *c,
                            uint8_t px[3])
{
    enum { CW = 32, CH = 32 };
    const int num_layers = PL_DEF(c->num_layers, 2);
    pl_tex tex[2] = { solid_tex(gpu, 255, 0, 0, c->alpha[0]),
                      solid_tex(gpu, 0, 255, 0, c->alpha[1]) };
    pl_tex dst = canvas_tex(gpu, CW, CH);
    if (!tex[0] || !tex[1] || !dst)
        FAIL("%s: failed creating textures\n", c->name);

    pl_compositor comp = pl_compositor_create(log);
    pl_compositor_init(comp, gpu);

    static struct pl_frame frames[2];
    struct pl_source_frame srcs[2];
    for (int i = 0; i < num_layers; i++) {
        frames[i] = image_frame(tex[i], c->alpha[i] < 255);
        srcs[i] = (struct pl_source_frame) {
            .pts = 0.0, .duration = 1.0, .map = map_frame, .frame_data = &frames[i],
        };

        pl_input in = pl_compositor_input_create(comp);
        in->fit     = PL_FIT_FILL;
        in->z_index = i;
        in->params  = i ? c->top_params : c->base_params;
        if (c->half)
            in->dst_rect = (pl_rect2df) { 0, 0, 0.5f, 1 };
        pl_queue_push(in->queue, &srcs[i]);
        pl_queue_push(in->queue, NULL);
    }

    const struct pl_frame target = canvas_frame(dst);
    struct pl_render_params rparams = { PL_RENDER_DEFAULTS
        .background       = c->background,
        .background_color = { 0.0f, 0.0f, 1.0f },
        .disable_fbos     = c->direct,
    };
    comp->params = &rparams;
    const struct pl_queue_params params = { PL_QUEUE_DEFAULTS
        .pts = 0.0, .vsync_duration = 1/30.0,
    };

    for (int pass = c->stale ? 0 : 1; pass < 2; pass++) {
        pl_tex_clear(gpu, dst, (float[4]) { 1.0f, 0.0f, 1.0f, 1.0f });
        rparams.border = pass ? c->border : PL_CLEAR_COLOR;
        rparams.skip_target_clearing = pass && c->deprecated;
        if (!pl_compositor_render(comp, &target, &params, NULL))
            FAIL("%s: rendering failed\n", c->name);
    }

    uint8_t out[CW * CH * 4];
    if (!pl_tex_download(gpu, pl_tex_transfer_params(.tex = dst, .ptr = out)))
        FAIL("%s: failed downloading the canvas\n", c->name);

    const int x = c->half ? CW * 3 / 4 : CW / 2;
    memcpy(px, &out[((CH / 2) * CW + x) * 4], 3);

    pl_compositor_destroy(&comp);
    pl_tex_destroy(gpu, &tex[0]);
    pl_tex_destroy(gpu, &tex[1]);
    pl_tex_destroy(gpu, &dst);
}

static void render_params_case(pl_log log, pl_gpu gpu, const struct params_case *c)
{
    uint8_t px[3];
    composite_pixel(log, gpu, c, px);

    if (c->expect_tile) {
        // Either tile color, both a light grey
        if (abs(px[0] - px[1]) > 2 || abs(px[1] - px[2]) > 2 || px[0] < 200) {
            FAIL("%s: composited to #%02x%02x%02x, expected a tile\n",
                 c->name, px[0], px[1], px[2]);
        }
    } else {
        const int want_r = lrintf(srgb_encode(c->red)   * 255);
        const int want_g = lrintf(srgb_encode(c->green) * 255);
        const int want_b = lrintf(srgb_encode(c->blue)  * 255);
        if (abs(px[0] - want_r) > 2 || abs(px[1] - want_g) > 2 || abs(px[2] - want_b) > 2) {
            FAIL("%s: composited to #%02x%02x%02x, expected #%02x%02x%02x\n",
                 c->name, px[0], px[1], px[2], want_r, want_g, want_b);
        }
    }

    printf("    %-60s #%02x%02x%02x\n", c->name, px[0], px[1], px[2]);
}

static void render_params_tests(pl_log log, pl_gpu gpu)
{
    for (int i = 0; i < PL_ARRAY_SIZE(params_cases); i++)
        render_params_case(log, gpu, &params_cases[i]);
}

//----------------------------------------------------------------------------
// Equivalence with the renderer
//----------------------------------------------------------------------------

// After `pl_compositor_options_set()`, a single input covering the canvas must
// render exactly as it would using `pl_render_image_mix()` with the same
// options, i.e. without applying any of them twice. Inputs with partial alpha
// are only rendered directly with `disable_linear_composition`.
struct direct_case {
    const char *name;
    const struct pl_render_params *params; // options
    const struct pl_render_params *canvas; // replaces the canvas params, if set
    bool transparent;                       // image with an alpha channel
    bool blended;                           // partial alpha for other reasons
};

static const struct direct_case direct_cases[] = {
    { "default params", pl_render_params() },
    { "color adjustment", pl_render_params(.color_adjustment = &desaturate) },
    { "debanding", pl_render_params(.deband_params = &pl_deband_default_params) },
    { "cone distortion", pl_render_params(.cone_params = &pl_vision_protanomaly) },
    { "corner rounding", pl_render_params(.corner_rounding = 0.5f), .blended = true },
    { "distortion", pl_render_params(.distort_params = &tilt), .blended = true },
    { "upscaler", pl_render_params(.upscaler = &pl_filter_ewa_lanczossharp,
                                   .sigmoid_params = NULL) },
    { "transparent image onto tiles", pl_render_params(.background = PL_CLEAR_TILES),
      .transparent = true },
    { "transparent image onto a color", pl_render_params(
        .background_color = { 0.0f, 0.0f, 1.0f }), .transparent = true },
    { "image processing on the canvas is ignored", pl_render_params(),
      pl_render_params(
        .deband_params    = &pl_deband_default_params,
        .color_adjustment = &desaturate,
        .cone_params      = &pl_vision_protanomaly,
        .distort_params   = &tilt,
        .corner_rounding  = 0.5f,
      ) },
};

static pl_tex gradient_tex(pl_gpu gpu, int w, int h, bool alpha)
{
    uint8_t *data = malloc(w * h * 4);
    if (!data)
        return NULL;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint8_t *p = &data[(y * w + x) * 4];
            p[0] = x * 255 / (w - 1);
            p[1] = y * 255 / (h - 1);
            p[2] = 255 - p[0];
            p[3] = alpha ? (p[0] + p[1]) / 2 : 255;
        }
    }
    pl_tex tex = pl_tex_create(gpu, pl_tex_params(
        .w = w, .h = h, .format = pl_find_named_fmt(gpu, "rgba8"),
        .sampleable = true, .initial_data = data,
    ));
    free(data);
    return tex;
}

// Upscales the image, to make the choice of scaler matter
enum { DIRECT_IW = 24, DIRECT_IH = 20, DIRECT_CW = 48, DIRECT_CH = 40 };

// Returns the maximum difference between the two renders, and writes the
// direct render to `out`
static int render_direct_case(pl_log log, pl_gpu gpu, const struct direct_case *c,
                              uint8_t *out)
{
    enum { CW = DIRECT_CW, CH = DIRECT_CH };
    pl_tex img = gradient_tex(gpu, DIRECT_IW, DIRECT_IH, c->transparent);
    pl_tex dst[2] = { canvas_tex(gpu, CW, CH), canvas_tex(gpu, CW, CH) };
    if (!img || !dst[0] || !dst[1])
        FAIL("%s: failed creating textures\n", c->name);

    static struct pl_frame frame;
    frame = image_frame(img, c->transparent);
    struct pl_source_frame src = {
        .pts = 0.0, .duration = 1.0, .map = map_frame, .frame_data = &frame,
    };

    pl_compositor comp = pl_compositor_create(log);
    pl_input in = pl_compositor_input_create(comp);
    in->fit = PL_FIT_FILL;
    comp->disable_linear_composition = c->transparent || c->blended;
    pl_compositor_init(comp, gpu);
    pl_queue_push(in->queue, &src);
    pl_queue_push(in->queue, NULL);

    pl_options opts = pl_options_alloc(log);
    pl_options_reset(opts, c->params);
    pl_compositor_options_set(comp, opts);
    if (c->canvas)
        comp->params = c->canvas;

    const struct pl_queue_params params = { PL_QUEUE_DEFAULTS
        .pts = 0.0, .vsync_duration = 1/30.0,
    };

    const struct pl_frame target[2] = { canvas_frame(dst[0]), canvas_frame(dst[1]) };
    if (!pl_compositor_render(comp, &target[0], &params, NULL))
        FAIL("%s: compositing failed\n", c->name);

    struct pl_input_status status;
    if (pl_input_update(in, &status, &params) == PL_QUEUE_ERR || !status.mix)
        FAIL("%s: no frame mix\n", c->name);
    pl_renderer rr = pl_renderer_create(log, gpu);
    if (!pl_render_image_mix(rr, status.mix, &target[1], &opts->params))
        FAIL("%s: direct rendering failed\n", c->name);

    static uint8_t got[CW * CH * 4];
    if (!pl_tex_download(gpu, pl_tex_transfer_params(.tex = dst[0], .ptr = got)) ||
        !pl_tex_download(gpu, pl_tex_transfer_params(.tex = dst[1], .ptr = out)))
    {
        FAIL("%s: failed downloading the canvas\n", c->name);
    }

    int max_diff = 0;
    for (int i = 0; i < CW * CH * 4; i++) {
        if (i % 4 != 3)
            max_diff = PL_MAX(max_diff, abs(got[i] - out[i]));
    }

    pl_renderer_destroy(&rr);
    pl_compositor_destroy(&comp);
    pl_options_free(&opts);
    pl_tex_destroy(gpu, &img);
    pl_tex_destroy(gpu, &dst[0]);
    pl_tex_destroy(gpu, &dst[1]);
    return max_diff;
}

static void render_direct_tests(pl_log log, pl_gpu gpu)
{
    enum { SIZE = DIRECT_CW * DIRECT_CH * 4 };
    static uint8_t ref[SIZE], out[SIZE];
    for (int i = 0; i < PL_ARRAY_SIZE(direct_cases); i++) {
        const struct direct_case *c = &direct_cases[i];
        const int diff = render_direct_case(log, gpu, c, i ? out : ref);
        if (diff)
            FAIL("%s: differs from rendering directly, by up to %d\n", c->name, diff);

        // Make sure each case actually tests something
        if (i && !c->transparent && !c->canvas && !memcmp(out, ref, SIZE))
            FAIL("%s: has no effect\n", c->name);
        printf("    %-60s max diff %d\n", c->name, diff);
    }
}

static void count_pass(void *priv, const struct pl_render_info *info)
{
    (*(int *) priv)++;
}

// Counts into the same slots as `count_pass`, using `user_data` for inputs
static void count_comp_pass(void *priv, pl_input input,
                            const struct pl_render_info *info)
{
    int *passes = priv;
    passes[input ? (int) (intptr_t) input->user_data : 0]++;
}

// Renders two layers, counting the passes of the canvas and each input, as
// reported by the params of each, and by the compositor
static void render_info(pl_log log, pl_gpu gpu, int passes[3], int comp_passes[3])
{
    pl_tex tex[2] = { solid_tex(gpu, 255, 0, 0, 255), solid_tex(gpu, 0, 255, 0, 128) };
    pl_tex dst = canvas_tex(gpu, 32, 32);
    if (!tex[0] || !tex[1] || !dst)
        FAIL("info callback: failed creating textures\n");

    pl_compositor comp = pl_compositor_create(log);
    pl_compositor_init(comp, gpu);

    struct pl_render_params cpar = { PL_RENDER_DEFAULTS
        .info_callback = count_pass,
        .info_priv     = &passes[0],
    };
    comp->params        = &cpar;
    comp->info_callback = count_comp_pass;
    comp->info_priv     = comp_passes;

    static struct pl_frame frames[2];
    struct pl_render_params par[2];
    struct pl_source_frame srcs[2];
    for (int i = 0; i < 2; i++) {
        frames[i] = image_frame(tex[i], i);
        srcs[i] = (struct pl_source_frame) {
            .pts = 0.0, .duration = 1.0, .map = map_frame, .frame_data = &frames[i],
        };
        par[i] = (struct pl_render_params) { PL_RENDER_DEFAULTS
            .info_callback = count_pass,
            .info_priv     = &passes[i + 1],
        };

        pl_input in = pl_compositor_input_create(comp);
        in->fit       = PL_FIT_FILL;
        in->params    = &par[i];
        in->user_data = (void *) (intptr_t) (i + 1);
        pl_queue_push(in->queue, &srcs[i]);
        pl_queue_push(in->queue, NULL);
    }

    const struct pl_queue_params params = { PL_QUEUE_DEFAULTS
        .pts = 0.0, .vsync_duration = 1/30.0,
    };

    const struct pl_frame target = canvas_frame(dst);
    if (!pl_compositor_render(comp, &target, &params, NULL))
        FAIL("info callback: rendering failed\n");

    pl_compositor_destroy(&comp);
    pl_tex_destroy(gpu, &tex[0]);
    pl_tex_destroy(gpu, &tex[1]);
    pl_tex_destroy(gpu, &dst);
}

static void render_info_tests(pl_log log, pl_gpu gpu)
{
    int info[3] = {0}, comp_info[3] = {0};
    render_info(log, gpu, info, comp_info);
    if (!info[0] || !info[1] || !info[2]) {
        FAIL("info callback: missing passes (canvas %d, inputs %d, %d)\n",
             info[0], info[1], info[2]);
    }
    printf("    %-60s %d + %d + %d\n", "info callbacks report each input and the canvas",
           info[1], info[2], info[0]);

    // The compositor's callback sees all of these passes, attributed correctly
    for (int i = 0; i < 3; i++) {
        if (comp_info[i] != info[i]) {
            FAIL("compositor info callback: %d passes for %s, expected %d\n",
                 comp_info[i], i ? "an input" : "the canvas", info[i]);
        }
    }
    printf("    %-60s ok\n", "compositor info callback reports the input");
}

// `in_status` gives the expected status of each of the three inputs
static void expect_status(const char *name, const struct pl_compositor_status *st,
                          const pl_input *in, enum pl_queue_status status,
                          const enum pl_queue_status *in_status,
                          int num, const pl_input *visible)
{
    if (st->status != status)
        FAIL("%s: status %d, expected %d\n", name, st->status, status);
    for (int i = 0; i < 3; i++) {
        const struct pl_input_status *is = &st->input_status[i];
        if (is->status != in_status[i] || !is->mix != (in_status[i] != PL_QUEUE_OK))
            FAIL("%s: wrong status for input %d\n", name, i);
    }
    if (st->num_visible != num)
        FAIL("%s: %d visible inputs, expected %d\n", name, st->num_visible, num);
    for (int i = 0; i < num; i++) {
        if (st->visible[i] != visible[i])
            FAIL("%s: wrong input at position %d\n", name, i);
    }
    const struct pl_frame *base = NULL;
    for (int i = 0; i < 3 && num; i++) {
        if (in[i] == visible[0])
            base = pl_frame_mix_nearest(st->input_status[i].mix);
    }
    if (st->base != base)
        FAIL("%s: wrong base frame\n", name);
    printf("    %-60s ok\n", name);
}

// Renders, after checking that pl_compositor_update() agrees with the result
static void render_status(pl_compositor comp, const struct pl_frame *target,
                          const struct pl_queue_params *params,
                          struct pl_compositor_status *st)
{
    const struct pl_canvas_size canvas = { 32, 32 };
    const struct pl_compositor_status up = pl_compositor_update(comp, params, &canvas);
    const pl_input first = up.num_visible ? up.visible[0] : NULL;
    const int num = up.num_visible;
    const enum pl_queue_status status = up.status;

    if (!pl_compositor_render(comp, target, params, st))
        FAIL("status: rendering failed\n");
    if (st->status != status || st->num_visible != num ||
        (num && st->visible[0] != first))
        FAIL("status: pl_compositor_update() disagrees with rendering\n");
}

static void render_status_tests(pl_log log, pl_gpu gpu)
{
    pl_tex tex[2] = { solid_tex(gpu, 255, 0, 0, 255), solid_tex(gpu, 0, 255, 0, 128) };
    pl_tex dst = canvas_tex(gpu, 32, 32);
    if (!tex[0] || !tex[1] || !dst)
        FAIL("status: failed creating textures\n");

    pl_compositor comp = pl_compositor_create(log);
    pl_compositor_init(comp, gpu);

    // An opaque input, a transparent one below it, and one without frames
    static struct pl_frame frames[2];
    struct pl_source_frame srcs[2];
    pl_input in[3];
    for (int i = 0; i < 3; i++) {
        in[i] = pl_compositor_input_create(comp);
        in[i]->fit = PL_FIT_FILL;
        in[i]->stop_pts = 1.0;
        if (i == 2)
            continue;
        frames[i] = image_frame(tex[i], i);
        srcs[i] = (struct pl_source_frame) {
            .pts = 0.0, .duration = 1.0, .map = map_frame, .frame_data = &frames[i],
        };
        pl_queue_push(in[i]->queue, &srcs[i]);
        pl_queue_push(in[i]->queue, NULL);
    }
    in[1]->z_index = -1;
    in[0]->dst_rect = (pl_rect2df) { 0.0f, 0.0f, 0.5f, 1.0f };

    struct pl_queue_params params = { PL_QUEUE_DEFAULTS
        .pts = 0.0, .vsync_duration = 1/30.0,
    };

    const struct pl_frame target = canvas_frame(dst);
    struct pl_compositor_status st;
    render_status(comp, &target, &params, &st);
    expect_status("status lists the visible inputs in order", &st, in, PL_QUEUE_MORE,
                  (enum pl_queue_status[]) { PL_QUEUE_OK, PL_QUEUE_OK, PL_QUEUE_MORE },
                  2, (pl_input[]) { in[1], in[0] });

    pl_queue_push(in[2]->queue, NULL);
    in[0]->dst_rect = (pl_rect2df) { 0.0f, 0.0f, 1.0f, 1.0f };
    render_status(comp, &target, &params, &st);
    expect_status("status excludes obscured inputs", &st, in, PL_QUEUE_OK,
                  (enum pl_queue_status[]) { PL_QUEUE_OK, PL_QUEUE_OK, PL_QUEUE_EOF },
                  1, (pl_input[]) { in[0] });

    params.pts = 1.0;
    render_status(comp, &target, &params, &st);
    expect_status("status reports EOF once all inputs are done", &st, in, PL_QUEUE_EOF,
                  (enum pl_queue_status[]) { PL_QUEUE_EOF, PL_QUEUE_EOF, PL_QUEUE_EOF },
                  0, NULL);

    pl_compositor_destroy(&comp);
    pl_tex_destroy(gpu, &tex[0]);
    pl_tex_destroy(gpu, &tex[1]);
    pl_tex_destroy(gpu, &dst);
}

static void run_render_tests(pl_log log)
{
#ifdef PL_HAVE_VK_PROC_ADDR
    pl_vulkan vk = pl_vulkan_create(log, pl_vulkan_params(.allow_software = true));
    if (vk) {
        printf("--- fit modes, rendered\n");
        for (int i = 0; i < PL_ARRAY_SIZE(fit_cases); i++)
            render_fit_case(log, vk->gpu, &fit_cases[i], false);
        for (int i = 0; i < PL_ARRAY_SIZE(fit_cases); i++) {
            if (fit_cases[i].fit == PL_FIT_RAW)
                render_fit_case(log, vk->gpu, &fit_cases[i], true);
        }
        printf("\n--- render params, rendered\n");
        render_params_tests(log, vk->gpu);
        printf("\n--- equivalence with the renderer\n");
        render_direct_tests(log, vk->gpu);
        render_info_tests(log, vk->gpu);
        printf("\n--- composition status\n");
        render_status_tests(log, vk->gpu);
        pl_vulkan_destroy(&vk);
        printf("\n");
        return;
    }
#endif
    printf("--- rendered tests skipped: no GPU available\n\n");
}

//----------------------------------------------------------------------------
// Compositing order
//----------------------------------------------------------------------------

// Compositing order is observable headlessly: pl_compositor_current_frame()
// returns the lowest layer that nothing above it fully obscures, so these cases
// assert which layer that is.
struct layer {
    int z_index;
    enum pl_fit_mode fit;
    float width, height;
    bool transparent;
    struct pl_rect2df dst_rect; // {0} = full canvas
    float corner_rounding;
    bool distorted;
};

struct zorder_case {
    const char *name;
    float canvas_w, canvas_h;
    struct layer layers[4];
    int num_layers;
    int expect;             // attachment index of the expected base layer
    const struct pl_render_params *canvas;
};

static void run_zorder_case(pl_log log, pl_gpu gpu, const struct zorder_case *c)
{
    pl_compositor comp = pl_compositor_create(log);
    pl_compositor_init(comp, gpu);

    comp->params = c->canvas;

    static struct pl_frame frames[4];
    static struct pl_source_frame srcs[4];
    static struct pl_render_params rparams[4];

    for (int i = 0; i < c->num_layers; i++) {
        const struct layer *l = &c->layers[i];
        frames[i] = (struct pl_frame) {
            .crop = { 0, 0, l->width, l->height },
            .repr.alpha = l->transparent ? PL_ALPHA_INDEPENDENT : PL_ALPHA_NONE,
            .user_data = (void *) (intptr_t) i,
        };
        srcs[i] = (struct pl_source_frame) {
            .pts = 0.0, .duration = 1/30.0,
            .map = map_frame, .frame_data = &frames[i],
        };

        pl_input in = pl_compositor_input_create(comp);
        in->z_index = l->z_index;
        in->fit     = l->fit;
        in->dst_rect = rect_or_full(l->dst_rect);
        rparams[i] = (struct pl_render_params) { PL_RENDER_DEFAULTS
            .corner_rounding = l->corner_rounding,
            .distort_params  = l->distorted ? &tilt : NULL,
        };
        in->params = &rparams[i];
        pl_queue_push(in->queue, &srcs[i]);
        pl_queue_push(in->queue, NULL);
    }

    const struct pl_queue_params params = { PL_QUEUE_DEFAULTS
        .pts = 0.0, .vsync_duration = 1/30.0,
    };

    // A zero canvas size exercises the fallback for an unknown canvas
    const struct pl_canvas_size canvas = { .width = c->canvas_w, .height = c->canvas_h };
    const struct pl_frame *cur = pl_compositor_current_frame(comp, &params,
                                                             c->canvas_w ? &canvas : NULL);

    const int got = cur ? (int) (intptr_t) cur->user_data : -1;
    if (got != c->expect) {
        FAIL("%s: base layer is input %d, expected input %d\n",
             c->name, got, c->expect);
    }

    printf("    %-56s base = input %d\n", c->name, got);
    pl_compositor_destroy(&comp);
}

#define LAYER(z, fit_, w, h, ...) \
    { .z_index = z, .fit = fit_, .width = w, .height = h, __VA_ARGS__ }

static const struct zorder_case zorder_cases[] = {
    // PL_FIT_COVER always covers the canvas, so an opaque one obscures
    // everything below it; the default z_index of 0 leaves attachment order.
    { "default z_index: the last attached layer wins",
      .canvas_w = 200, .canvas_h = 200, .num_layers = 2, .expect = 1,
      .layers = { LAYER(0, PL_FIT_COVER, 160, 90),
                  LAYER(0, PL_FIT_COVER, 160, 90) } },
    { "higher z_index wins against attachment order",
      .canvas_w = 200, .canvas_h = 200, .num_layers = 2, .expect = 0,
      .layers = { LAYER(1, PL_FIT_COVER, 160, 90),
                  LAYER(0, PL_FIT_COVER, 160, 90) } },
    { "equal z_index falls back to attachment order",
      .canvas_w = 200, .canvas_h = 200, .num_layers = 2, .expect = 1,
      .layers = { LAYER(5, PL_FIT_COVER, 160, 90),
                  LAYER(5, PL_FIT_COVER, 160, 90) } },
    { "negative z_index sinks below the default",
      .canvas_w = 200, .canvas_h = 200, .num_layers = 2, .expect = 0,
      .layers = { LAYER(0, PL_FIT_COVER, 160, 90),
                  LAYER(-1, PL_FIT_COVER, 160, 90) } },
    { "z_index orders three layers independently of attachment",
      .canvas_w = 200, .canvas_h = 200, .num_layers = 3, .expect = 1,
      .layers = { LAYER(0, PL_FIT_COVER, 160, 90),
                  LAYER(9, PL_FIT_COVER, 160, 90),
                  LAYER(-5, PL_FIT_COVER, 160, 90) } },

    // A layer only obscures what is beneath it if it actually covers the
    // canvas and is opaque - being on top is not sufficient.
    { "letterboxed top layer does not obscure the one below",
      .canvas_w = 400, .canvas_h = 200, .num_layers = 2, .expect = 0,
      .layers = { LAYER(0, PL_FIT_COVER, 160, 90),
                  LAYER(10, PL_FIT_CONTAIN, 100, 100) } },
    // Backgrounds are only drawn below the canvas as a whole, so they never
    // make a transparent layer obscure anything
    { "transparent top layer does not obscure the one below",
      .canvas_w = 200, .canvas_h = 200, .num_layers = 2, .expect = 0,
      .layers = { LAYER(0, PL_FIT_COVER, 160, 90),
                  LAYER(10, PL_FIT_COVER, 160, 90, .transparent = true) } },
    { "... whatever the canvas background",
      .canvas_w = 200, .canvas_h = 200, .num_layers = 2, .expect = 0,
      .canvas = pl_render_params(.background = PL_CLEAR_TILES),
      .layers = { LAYER(0, PL_FIT_COVER, 160, 90),
                  LAYER(10, PL_FIT_COVER, 160, 90, .transparent = true) } },
    { "rounded corners do not obscure the layer below",
      .canvas_w = 200, .canvas_h = 200, .num_layers = 2, .expect = 0,
      .layers = { LAYER(0, PL_FIT_COVER, 160, 90),
                  LAYER(10, PL_FIT_COVER, 160, 90, .corner_rounding = 0.5f) } },
    { "canvas corner rounding does not affect occlusion",
      .canvas_w = 200, .canvas_h = 200, .num_layers = 2, .expect = 1,
      .canvas = pl_render_params(.corner_rounding = 0.5f),
      .layers = { LAYER(0, PL_FIT_COVER, 160, 90),
                  LAYER(10, PL_FIT_COVER, 160, 90) } },
    { "distorted layers do not obscure the layer below",
      .canvas_w = 200, .canvas_h = 200, .num_layers = 2, .expect = 0,
      .layers = { LAYER(0, PL_FIT_COVER, 160, 90),
                  LAYER(10, PL_FIT_COVER, 160, 90, .distorted = true) } },

    // `dst_rect` bounds what a layer can obscure, whatever its fit mode
    { "FILL onto the whole canvas obscures the layer below",
      .canvas_w = 200, .canvas_h = 200, .num_layers = 2, .expect = 1,
      .layers = { LAYER(0, PL_FIT_COVER, 160, 90),
                  LAYER(10, PL_FIT_FILL, 160, 90) } },
    { "COVER into a sub-rect does not obscure the layer below",
      .canvas_w = 200, .canvas_h = 200, .num_layers = 2, .expect = 0,
      .layers = { LAYER(0, PL_FIT_COVER, 160, 90),
                  LAYER(10, PL_FIT_COVER, 160, 90, .dst_rect = { 0, 0, 0.5f, 1 }) } },
    { "oversized PLACE into a sub-rect does not obscure the layer below",
      .canvas_w = 200, .canvas_h = 200, .num_layers = 2, .expect = 0,
      .layers = { LAYER(0, PL_FIT_COVER, 160, 90),
                  LAYER(10, PL_FIT_PLACE, 400, 400, .dst_rect = { 0, 0, 1, 0.5f }) } },
    { "COVER into a flipped full canvas obscures the layer below",
      .canvas_w = 200, .canvas_h = 200, .num_layers = 2, .expect = 1,
      .layers = { LAYER(0, PL_FIT_COVER, 160, 90),
                  LAYER(10, PL_FIT_COVER, 160, 90, .dst_rect = { 1, 1, 0, 0 }) } },
    { "COVER into a flipped sub-rect does not obscure the layer below",
      .canvas_w = 200, .canvas_h = 200, .num_layers = 2, .expect = 0,
      .layers = { LAYER(0, PL_FIT_COVER, 160, 90),
                  LAYER(10, PL_FIT_COVER, 160, 90, .dst_rect = { 0.5f, 0, 0, 1 }) } },
    { "unknown canvas: COVER obscures the layer below",
      .num_layers = 2, .expect = 1,
      .layers = { LAYER(0, PL_FIT_COVER, 160, 90),
                  LAYER(10, PL_FIT_COVER, 160, 90) } },
    { "unknown canvas: COVER into a sub-rect does not",
      .num_layers = 2, .expect = 0,
      .layers = { LAYER(0, PL_FIT_COVER, 160, 90),
                  LAYER(10, PL_FIT_COVER, 160, 90, .dst_rect = { 0, 0, 0.5f, 1 }) } },
};

static void run_zorder_tests(pl_log log, pl_gpu gpu)
{
    printf("--- compositing order\n");
    for (int i = 0; i < PL_ARRAY_SIZE(zorder_cases); i++)
        run_zorder_case(log, gpu, &zorder_cases[i]);
    printf("\n");
}

//----------------------------------------------------------------------------
// Color overrides
//----------------------------------------------------------------------------

// Attaches an input covering the canvas, showing only `frame`
static pl_input override_input(pl_compositor comp, struct pl_source_frame *src,
                               struct pl_frame *frame)
{
    pl_input in = pl_compositor_input_create(comp);
    in->fit = PL_FIT_COVER;
    *src = (struct pl_source_frame) {
        .pts = 0.0, .duration = 1/30.0, .map = map_frame, .frame_data = frame,
    };
    pl_queue_push(in->queue, src);
    pl_queue_push(in->queue, NULL);
    return in;
}

static void run_override_tests(pl_log log, pl_gpu gpu)
{
    const struct pl_queue_params params = { PL_QUEUE_DEFAULTS
        .pts = 0.0, .vsync_duration = 1/30.0,
    };
    const struct pl_canvas_size *canvas = pl_canvas_size(.width = 200, .height = 200);

    printf("--- color overrides\n");

    // A set field replaces the frame's; an unset one leaves it alone.
    {
        pl_compositor comp = pl_compositor_create(log);
        pl_compositor_init(comp, gpu);

        static struct pl_frame frame;
        static struct pl_source_frame src;
        frame = (struct pl_frame) {
            .crop = { 0, 0, 160, 90 },
            .repr = { .sys = PL_COLOR_SYSTEM_BT_601, .levels = PL_COLOR_LEVELS_LIMITED },
            .color = { .transfer = PL_COLOR_TRC_BT_1886 },
        };
        pl_input in = override_input(comp, &src, &frame);

        // Override only `sys` and `primaries`; the rest must survive
        in->repr.sys = PL_COLOR_SYSTEM_BT_709;
        in->color.primaries = PL_COLOR_PRIM_BT_2020;

        const struct pl_frame *cur = pl_compositor_current_frame(comp, &params, canvas);
        if (!cur)
            FAIL("color override: no current frame\n");
        if (cur->repr.sys != PL_COLOR_SYSTEM_BT_709)
            FAIL("color override: sys is %d, expected BT.709\n", cur->repr.sys);
        if (cur->repr.levels != PL_COLOR_LEVELS_LIMITED)
            FAIL("color override: unset levels was clobbered (%d)\n", cur->repr.levels);
        if (cur->color.primaries != PL_COLOR_PRIM_BT_2020)
            FAIL("color override: primaries is %d, expected BT.2020\n", cur->color.primaries);
        if (cur->color.transfer != PL_COLOR_TRC_BT_1886)
            FAIL("color override: unset transfer was clobbered (%d)\n", cur->color.transfer);
        printf("    %-56s ok\n", "set fields replace, unset fields survive");
        pl_compositor_destroy(&comp);
    }

    // Overrides must not be destructive: setting one, then clearing it, has to
    // restore the frame's original value rather than leave the override baked
    // into the queue's copy of the frame.
    {
        pl_compositor comp = pl_compositor_create(log);
        pl_compositor_init(comp, gpu);

        static struct pl_frame frame;
        static struct pl_source_frame src;
        frame = (struct pl_frame) {
            .crop = { 0, 0, 160, 90 },
            .color = { .primaries = PL_COLOR_PRIM_BT_709 },
        };
        pl_input in = override_input(comp, &src, &frame);

        const struct pl_frame *cur = pl_compositor_current_frame(comp, &params, canvas);
        if (!cur || cur->color.primaries != PL_COLOR_PRIM_BT_709)
            FAIL("override round-trip: initial primaries wrong\n");

        in->color.primaries = PL_COLOR_PRIM_BT_2020;
        cur = pl_compositor_current_frame(comp, &params, canvas);
        if (!cur || cur->color.primaries != PL_COLOR_PRIM_BT_2020)
            FAIL("override round-trip: override did not apply\n");

        in->color.primaries = PL_COLOR_PRIM_UNKNOWN;
        cur = pl_compositor_current_frame(comp, &params, canvas);
        if (!cur || cur->color.primaries != PL_COLOR_PRIM_BT_709) {
            FAIL("override round-trip: clearing the override left %d behind, "
                 "expected the frame's original BT.709\n",
                 cur ? cur->color.primaries : -1);
        }
        printf("    %-56s ok\n", "set, then cleared, restores the frame's value");
        pl_compositor_destroy(&comp);
    }

    // `repr.alpha` feeds the occlusion test as well as rendering, so an
    // override of it must reach both. This is why the merge happens where the
    // mix is built rather than alongside the crop fixups.
    {
        pl_compositor comp = pl_compositor_create(log);
        pl_compositor_init(comp, gpu);

        static struct pl_frame frames[2];
        static struct pl_source_frame srcs[2];
        pl_input in[2];
        for (int i = 0; i < 2; i++) {
            frames[i] = (struct pl_frame) {
                .crop = { 0, 0, 160, 90 },
                // The top layer claims transparency, so by itself it cannot
                // obscure the one below it
                .repr.alpha = i ? PL_ALPHA_INDEPENDENT : PL_ALPHA_NONE,
                .user_data = (void *) (intptr_t) i,
            };
            in[i] = override_input(comp, &srcs[i], &frames[i]);
        }
        in[1]->repr.alpha = PL_ALPHA_NONE;

        const struct pl_frame *cur = pl_compositor_current_frame(comp, &params, canvas);
        const int got = cur ? (int) (intptr_t) cur->user_data : -1;
        if (got != 1)
            FAIL("alpha override: base layer is input %d, expected 1\n", got);
        printf("    %-56s ok\n", "alpha override reaches occlusion");
        pl_compositor_destroy(&comp);
    }

    printf("\n");
}

//----------------------------------------------------------------------------
// Frame mixing
//----------------------------------------------------------------------------

// Returns the number of frames mixed by `input` at a PTS between two frames
static int mix_size(pl_input input, const struct pl_queue_params *params)
{
    struct pl_input_status status;
    if (pl_input_update(input, &status, params) == PL_QUEUE_ERR)
        FAIL("frame mixer: failed updating the input\n");
    return status.mix ? status.mix->num_frames : 0;
}

static void run_mixer_tests(pl_log log, pl_gpu gpu)
{
    printf("--- frame mixing\n");

    pl_compositor comp = pl_compositor_create(log);
    pl_input in = pl_compositor_input_create(comp);
    static struct pl_frame frames[4];
    static struct pl_source_frame srcs[4];
    pl_compositor_init(comp, gpu);
    for (int i = 0; i < 4; i++) {
        frames[i] = (struct pl_frame) { .crop = { 0, 0, 160, 90 } };
        srcs[i] = (struct pl_source_frame) {
            .pts = i, .duration = 1.0, .map = map_frame, .frame_data = &frames[i],
        };
        pl_queue_push(in->queue, &srcs[i]);
    }
    pl_queue_push(in->queue, NULL);

    // The input's frame mixer determines its mix radius
    const struct pl_queue_params params = { PL_QUEUE_DEFAULTS
        .pts = 1.5, .vsync_duration = 0.4,
    };
    struct pl_render_params rparams = { PL_RENDER_DEFAULTS };
    in->params = &rparams;
    const int nearest = mix_size(in, &params);
    rparams.frame_mixer = &pl_filter_mitchell_clamp;
    const int mixed = mix_size(in, &params);
    printf("    %-56s %d -> %d frames\n", "frame mixer widens the mix",
           nearest, mixed);
    if (mixed <= nearest)
        FAIL("frame mixer: did not widen the mix\n");

    pl_compositor_destroy(&comp);
    printf("\n");
}

//----------------------------------------------------------------------------
// Unset crops
//----------------------------------------------------------------------------

// An unset crop means the full frame, as it does for the renderer
static void run_crop_tests(pl_log log, pl_gpu gpu)
{
    printf("--- unset crops\n");

    pl_fmt fmt = pl_find_named_fmt(gpu, "rgba8");
    pl_tex img_tex = pl_tex_create(gpu, pl_tex_params(.w = 160, .h = 90, .format = fmt));
    pl_tex dst_tex = pl_tex_create(gpu, pl_tex_params(.w = 200, .h = 200, .format = fmt));
    if (!img_tex || !dst_tex)
        FAIL("unset crop: failed creating textures\n");

    const struct pl_frame image = {
        .num_planes = 1,
        .planes[0]  = { .texture = img_tex, .components = 4 },
    };
    const struct pl_frame target = {
        .num_planes = 1,
        .planes[0]  = { .texture = dst_tex, .components = 4 },
    };

    pl_compositor comp = pl_compositor_create(log);
    pl_input in = pl_compositor_input_create(comp);
    pl_compositor_init(comp, gpu);
    in->fit = PL_FIT_CONTAIN;

    // 160x90 letterboxed into 200x200 is 200x112.5, centered vertically
    struct pl_rect2df src, dst;
    pl_input_place(in, &image, &target, &src, &dst);
    const struct pl_rect2df want = { 0, 0.21875f, 1, 0.78125f };
    if (fabsf(dst.x0 - want.x0) > 1e-4 || fabsf(dst.y0 - want.y0) > 1e-4 ||
        fabsf(dst.x1 - want.x1) > 1e-4 || fabsf(dst.y1 - want.y1) > 1e-4)
    {
        FAIL("unset crop: placed dst (%.4f %.4f)-(%.4f %.4f), expected "
             "(%.4f %.4f)-(%.4f %.4f)\n", dst.x0, dst.y0, dst.x1, dst.y1,
             want.x0, want.y0, want.x1, want.y1);
    }
    printf("    %-56s ok\n", "CONTAIN between frames with unset crops");

    pl_compositor_destroy(&comp);
    pl_tex_destroy(gpu, &img_tex);
    pl_tex_destroy(gpu, &dst_tex);
    printf("\n");
}

//----------------------------------------------------------------------------
// Queue management
//----------------------------------------------------------------------------

static bool map_fail(pl_gpu gpu, pl_tex *tex, const struct pl_source_frame *src,
                     struct pl_frame *out_frame)
{
    return false;
}

static void count_discard(const struct pl_source_frame *src)
{
    (*(int *) src->frame_data)++;
}

static void run_queue_tests(pl_log log, pl_gpu gpu)
{
    printf("--- queue management\n");

    enum { NUM_FRAMES = 8 };
    const double dur = 0.125;
    static struct pl_frame frames[NUM_FRAMES];
    struct pl_source_frame srcs[NUM_FRAMES];
    for (int i = 0; i < NUM_FRAMES; i++) {
        frames[i] = (struct pl_frame) { .user_data = (void *) (intptr_t) i };
        srcs[i] = (struct pl_source_frame) {
            .pts = i * dur, .duration = dur,
            .map = map_frame, .frame_data = &frames[i],
        };
    }

    pl_compositor comp = pl_compositor_create(log);
    pl_compositor_init(comp, gpu);

    // Frames pushed ahead of time, without EOF. The first input needs frames
    // from 0.5 at its start, the second one runs out of frames before its start
    pl_input in[2];
    for (int n = 0; n < 2; n++) {
        in[n] = pl_compositor_input_create(comp);
        for (int i = 0; i < NUM_FRAMES; i++)
            pl_queue_push(in[n]->queue, &srcs[i]);
    }
    in[0]->start_pts = 1.0;
    in[0]->seek_pts  = 0.5;
    in[1]->start_pts = 3.0;
    in[1]->seek_pts  = 1.0;

    struct pl_queue_params params = { PL_QUEUE_DEFAULTS .vsync_duration = dur };
    struct pl_input_status st;
    for (int o = 0; o < 8; o++) {
        params.pts = o * dur;
        if (pl_input_update(in[0], &st, &params) != PL_QUEUE_OK || st.mix)
            FAIL("preroll: input not idle at %f, before its start\n", params.pts);
    }

    // The input's own timeline is at 0.875 - 1.0 + 0.5 = 0.375
    struct pl_source_frame first;
    if (!pl_queue_peek(in[0]->queue, 0, &first) || first.pts != 0.375)
        FAIL("preroll: oldest frame at %f, expected 0.375\n", first.pts);

    params.pts = 1.0;
    if (pl_input_update(in[0], &st, &params) != PL_QUEUE_OK || !st.mix ||
        pl_frame_mix_nearest(st.mix)->user_data != (void *) 4)
    {
        FAIL("preroll: input does not start at seek_pts\n");
    }
    printf("    %-56s ok\n", "frames from before seek_pts are culled before start");

    // Once an input reaches EOF, frames pushed to it are discarded
    in[0]->stop_pts = params.pts = 1.125;
    if (pl_input_update(in[0], &st, &params) != PL_QUEUE_EOF)
        FAIL("eof: input did not stop\n");

    int discarded = 0;
    pl_queue_push(in[0]->queue, &(struct pl_source_frame) {
        .pts = 2.0, .duration = dur, .map = map_frame,
        .discard = count_discard, .frame_data = &discarded,
    });
    if (discarded != 1 || pl_queue_num_frames(in[0]->queue))
        FAIL("eof: frame pushed after EOF was not discarded\n");
    printf("    %-56s ok\n", "frames pushed after EOF are discarded");

    // The second input's own timeline is at 2.875 - 3.0 + 1.0 = 0.875
    for (int o = 0; o < 24; o++) {
        params.pts = o * dur;
        const enum pl_queue_status want = o < 23 ? PL_QUEUE_OK : PL_QUEUE_MORE;
        if (pl_input_update(in[1], &st, &params) != want || st.status != want || st.mix)
            FAIL("preroll: wrong status at %f, before start\n", params.pts);
    }
    if (pl_compositor_update(comp, &params, NULL).status != PL_QUEUE_MORE)
        FAIL("preroll: missing frames not reported by the compositor\n");
    printf("    %-56s ok\n", "missing frames are reported before start");

    // A failing input is reported, without affecting the others
    pl_input bad = pl_compositor_input_create(comp);
    pl_queue_push(bad->queue, &(struct pl_source_frame) {
        .pts = 0.0, .duration = dur, .map = map_fail,
    });
    const struct pl_compositor_status cst = pl_compositor_update(comp, &params, NULL);
    if (cst.status != PL_QUEUE_ERR ||
        cst.input_status[1].status != PL_QUEUE_MORE ||
        cst.input_status[2].status != PL_QUEUE_ERR)
    {
        FAIL("error: failing input not reported correctly\n");
    }
    printf("    %-56s ok\n", "failing inputs are reported individually");

    pl_compositor_destroy(&comp);
    printf("\n");
}

int main(void)
{
    pl_log log = pl_log_create(PL_API_VER, pl_log_params(
        .log_cb    = pl_log_color,
        .log_level = getenv("PL_TEST_VERBOSE") ? PL_LOG_TRACE : PL_LOG_WARN,
    ));
    pl_gpu gpu = pl_gpu_dummy_create(log, NULL);

    run_fit_tests(log, gpu);
    run_render_tests(log);
    run_zorder_tests(log, gpu);
    run_override_tests(log, gpu);
    run_mixer_tests(log, gpu);
    run_crop_tests(log, gpu);
    run_queue_tests(log, gpu);

    printf("legend: '.' hidden  '^' first visible  '>' advance 1  '=' repeat  "
           "'+' skip  '<' rewind\n\n");
    for (int i = 0; i < PL_ARRAY_SIZE(scenarios); i++) {
        run_scenario(&scenarios[i], log, gpu);
        printf("\n");
    }

    pl_gpu_dummy_destroy(&gpu);
    pl_log_destroy(&log);
    return 0;
}
