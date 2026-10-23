/*
 * This file is part of libplacebo.
 *
 * libplacebo is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * libplacebo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libplacebo.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <limits.h>
#include <math.h>

#include "common.h"
#include "log.h"
#include "json.h"
#include "options.h"

#include <libplacebo/compositor.h>

struct priv {
    pl_options opts; // compositor-level options
    PL_ARRAY(pl_options) input_opts;
    PL_ARRAY(struct animation) animations;
};

void pl_scene_destroy(pl_scene *pscene)
{
    struct pl_scene_t *scene = (struct pl_scene_t *) *pscene;
    if (!scene)
        return;

    struct priv *p = PL_PRIV(scene);
    pl_assert(scene->comp->num_inputs == p->input_opts.num);
    pl_compositor_destroy(&scene->comp);
    for (int i = 0; i < p->input_opts.num; i++)
        pl_options_free(&p->input_opts.elem[i]);
    pl_options_free(&p->opts);
    pl_free(scene);
    *pscene = NULL;
}

// Animation / tweening

struct keyframe {
    double pts;
    double value;
    double easing[4]; // [x1, y1, x2, y2]
    void *data;       // raw field value, for non-numeric options
};

struct animation {
    pl_opt opt;
    void *value;

    // Sorted list of keyframes
    PL_ARRAY(struct keyframe) keyframes;
};

static double cubic_bezier(double t, double v0, double v1, const double p[4])
{
    const double x1 = p[0], y1 = p[1], x2 = p[2], y2 = p[3];
    if (x1 < 0.0 || x1 > 1.0 || x2 < 0.0 || x2 > 1.0)
        return v0; // fall back to step hold
    if (x1 == y1 && x2 == y2)
        return v0 + (v1 - v0) * t;

    /* Polynomial form with P0 = (0,0), P3 = (1,1):
     * B(s) = ((a*s + b)*s + c)*s */
    const double cx = 3.0 * x1;
    const double bx = 3.0 * (x2 - x1) - cx;
    const double ax = 1.0 - cx - bx;
    const double cy = 3.0 * y1;
    const double by = 3.0 * (y2 - y1) - cy;
    const double ay = 1.0 - cy - by;

    const double eps = 1e-7;
    double s = t; /* good initial guess: x(s) is close to s */
    int found = 0;

    /* Newton-Raphson on x(s) - t = 0 */
    for (int i = 0; i < 8; i++) {
        const double err = ((ax * s + bx) * s + cx) * s - t;
        if (fabs(err) < eps) { found = 1; break; }
        const double d = (3.0 * ax * s + 2.0 * bx) * s + cx;   /* dx/ds */
        if (fabs(d) < 1e-9) break;      /* flat spot, Newton unreliable */
        s -= err / d;
        if (s < 0.0 || s > 1.0) break;  /* left the domain */
    }

    /* Bisection fallback: x(s) is monotonic on [0,1] because x1, x2 lie in [0,1] */
    if (!found) {
        double lo = 0.0, hi = 1.0;
        s = t;
        for (int i = 0; i < 64; i++) {
            const double x = ((ax * s + bx) * s + cx) * s;
            if (fabs(x - t) < eps) break;
            if (x < t) lo = s; else hi = s;
            s = 0.5 * (lo + hi);
        }
    }

    const double y = ((ay * s + by) * s + cy) * s;
    return v0 + (v1 - v0) * y;
}

static inline bool opt_is_numeric(pl_opt opt)
{
    return opt->type == PL_OPT_FLOAT || opt->type == PL_OPT_INT;
}

static struct keyframe animation_sample(const struct animation *anim, double pts)
{
    pl_assert(anim && anim->keyframes.num > 0);
    const struct keyframe *next = &anim->keyframes.elem[0], *prev = NULL;
    if (pts < next->pts)
        return *next;

    for (int i = 1; i < anim->keyframes.num; i++) {
        prev = next;
        next = &anim->keyframes.elem[i];
        if (pts < next->pts)
            break;
    }

    if (pts >= next->pts)
        return *next;

    pl_assert(pts >= prev->pts && pts <= next->pts);
    if (!opt_is_numeric(anim->opt))
        return *prev; // can't be interpolated

    const double t = (pts - prev->pts) / (next->pts - prev->pts);
    return (struct keyframe) {
        .pts   = pts,
        .value = cubic_bezier(t, prev->value, next->value, prev->easing),
    };
}

static void animation_write(const struct animation *anim,
                            const struct keyframe *kf)
{
    pl_opt opt = anim->opt;
    if (!opt_is_numeric(opt)) {
        memcpy(anim->value, kf->data, pl_opt_size(opt));
        return;
    }

    double value = kf->value;
    if (opt->min != opt->max)
        value = PL_CLAMP(value, opt->min, opt->max);

    if (opt->type == PL_OPT_INT) {
        *(int *) anim->value = lround(PL_CLAMP(value, INT_MIN, INT_MAX));
        return;
    }

    switch (pl_opt_size(opt)) {
    case sizeof(float):  *(float *)  anim->value = value; return;
    case sizeof(double): *(double *) anim->value = value; return;
    }

    pl_unreachable();
}

void pl_scene_update(pl_scene scene, const double pts)
{
    struct priv *p = PL_PRIV(scene);
    for (int i = 0; i < p->animations.num; i++) {
        const struct animation *anim = &p->animations.elem[i];
        const struct keyframe kf = animation_sample(anim, pts);
        animation_write(anim, &kf);
    }
}

// JSON scene graph parsing

struct parse_ctx {
    pl_log log;
    pl_str path; // for error messages
    void *tmp;
    struct pl_scene_t *scene;
};

static bool json_value_str(const struct json_value_s *val, pl_str *out)
{
    switch (val->type) {
    case json_type_string:
        *out = json_str(val->payload);
        return true;
    case json_type_number: {
        const struct json_number_s *num = val->payload;
        *out = (pl_str) { (uint8_t *) num->number, num->number_size };
        return true;
    }
    case json_type_true:
        *out = pl_str0("yes");
        return true;
    case json_type_false:
        *out = pl_str0("no");
        return true;
    default:
        return false;
    }
}

static bool parse_number(struct parse_ctx *ctx, const struct json_value_s *val,
                         double *out)
{
    pl_str str;
    switch (val->type) {
    case json_type_string:
        str = json_str(val->payload);
        break;
    case json_type_number: {
        const struct json_number_s *num = val->payload;
        str = (pl_str) { (uint8_t *) num->number, num->number_size };
        break;
    }
    default:
        PL_ERR(ctx, "Invalid value type for '%.*s'. Expected number.",
               PL_STR_FMT(ctx->path));
        return false;
    }

    if (!pl_str_parse_double_fraction(str, out)) {
        PL_ERR(ctx, "Failed parsing '%.*s' from value: '%.*s'",
               PL_STR_FMT(ctx->path), PL_STR_FMT(str));
        return false;
    }

    return true;
}

static bool parse_easing(struct parse_ctx *ctx, double easing[4],
                         const struct json_value_s *val)
{
    static const struct {
        const char *name;
        double easing[4];
    } curves[] = {
        { "linear",      { 0.0,  0.0, 1.0,  1.0 } },
        { "ease",        { 0.25, 0.1, 0.25, 1.0 } },
        { "ease-in",     { 0.42, 0.0, 1.0,  1.0 } },
        { "ease-out",    { 0.0,  0.0, 0.58, 1.0 } },
        { "ease-in-out", { 0.42, 0.0, 0.58, 1.0 } },
    };

    if (val->type == json_type_string) {
        const pl_str name = json_str(val->payload);
        for (int i = 0; i < PL_ARRAY_SIZE(curves); i++) {
            if (pl_str_equals0(name, curves[i].name)) {
                memcpy(easing, curves[i].easing, sizeof(curves[i].easing));
                return true;
            }
        }

        PL_ERR(ctx, "Unrecognized easing curve '%.*s' for '%.*s'",
               PL_STR_FMT(name), PL_STR_FMT(ctx->path));
        return false;
    }

    const struct json_array_s *arr = json_value_as_array((void *) val);
    if (!arr || arr->length != 4) {
        PL_ERR(ctx, "Expected a curve name or an array of four numbers "
               "[x1, y1, x2, y2] for '%.*s'", PL_STR_FMT(ctx->path));
        return false;
    }

    int idx = 0;
    bool ok = true;
    for (const struct json_array_element_s *e = arr->start; e; e = e->next) {
        const size_t len_prefix = ctx->path.len;
        pl_str_append_asprintf(ctx->tmp, &ctx->path, "[%d]", idx);
        ok &= parse_number(ctx, e->value, &easing[idx]);
        ctx->path = pl_str_take(ctx->path, len_prefix);
        idx++;
    }

    if (ok && (easing[0] < 0.0 || easing[0] > 1.0 ||
               easing[2] < 0.0 || easing[2] > 1.0))
    {
        PL_ERR(ctx, "Easing '%.*s' must have x1 and x2 in the range [0, 1]",
               PL_STR_FMT(ctx->path));
        return false;
    }

    return ok;
}

static bool parse_keyframe_value(struct parse_ctx *ctx, void *base, pl_opt opt,
                                 void *field, struct keyframe *kf,
                                 const struct json_value_s *val)
{
    if (opt->type == PL_OPT_FLOAT) {
        if (!parse_number(ctx, val, &kf->value))
            return false;
        if (opt->min != opt->max && (kf->value < opt->min || kf->value > opt->max)) {
            PL_ERR(ctx, "Value of %f for '%.*s' out of range: [%.2f, %.2f]",
                   kf->value, PL_STR_FMT(ctx->path), opt->min, opt->max);
            return false;
        }
        return true;
    }

    pl_str str;
    if (!json_value_str(val, &str)) {
        PL_ERR(ctx, "Invalid value type for '%.*s'. Expected string, number "
               "or boolean.", PL_STR_FMT(ctx->path));
        return false;
    }

    // Parse into a copy of the field, so it can be written later
    const size_t size = pl_opt_size(opt);
    kf->data = pl_memdup(ctx->scene, field, size);
    if (!pl_opt_parse(ctx->log, base, opt, kf->data, str)) {
        PL_ERR(ctx, "Failed parsing '%.*s' from value: '%.*s'",
               PL_STR_FMT(ctx->path), PL_STR_FMT(str));
        return false;
    }

    if (opt->type == PL_OPT_INT)
        kf->value = *(int *) kf->data;
    return true;
}

static bool parse_keyframe(struct parse_ctx *ctx, void *base, pl_opt opt,
                           void *field, struct keyframe *kf,
                           const struct json_value_s *val)
{
    const struct json_object_s *obj = json_value_as_object((void *) val);
    if (!obj) {
        PL_ERR(ctx, "Expected an object for '%.*s'", PL_STR_FMT(ctx->path));
        return false;
    }

    *kf = (struct keyframe) { .easing = { -1.0 } }; // step hold by default
    bool has_pts = false, has_value = false, ok = true;
    for (const struct json_object_element_s *e = obj->start; e; e = e->next) {
        const size_t len_prefix = ctx->path.len;
        const pl_str key = json_str(e->name);
        pl_str_append_asprintf(ctx->tmp, &ctx->path, ".%.*s", PL_STR_FMT(key));
        if (pl_str_equals0(key, "pts")) {
            has_pts = parse_number(ctx, e->value, &kf->pts);
            ok &= has_pts;
        } else if (pl_str_equals0(key, "value")) {
            has_value = parse_keyframe_value(ctx, base, opt, field, kf, e->value);
            ok &= has_value;
        } else if (pl_str_equals0(key, "easing")) {
            if (!opt_is_numeric(opt)) {
                PL_ERR(ctx, "Easing '%.*s' is only supported for numeric "
                       "options", PL_STR_FMT(ctx->path));
                ok = false;
            } else {
                ok &= parse_easing(ctx, kf->easing, e->value);
            }
        } else {
            PL_ERR(ctx, "Unrecognized key '%.*s'", PL_STR_FMT(ctx->path));
            ok = false;
        }
        ctx->path = pl_str_take(ctx->path, len_prefix);
    }

    if (ok && (!has_pts || !has_value)) {
        PL_ERR(ctx, "Keyframe '%.*s' requires both 'pts' and 'value'",
               PL_STR_FMT(ctx->path));
        return false;
    }

    return ok;
}

static bool parse_animation(struct parse_ctx *ctx, void *base, pl_opt opt,
                            void *field, const struct json_array_s *arr)
{
    if (opt->preset) {
        PL_ERR(ctx, "Preset '%.*s' cannot be animated", PL_STR_FMT(ctx->path));
        return false;
    } else if (!arr->length) {
        PL_ERR(ctx, "Animation '%.*s' requires at least one keyframe",
               PL_STR_FMT(ctx->path));
        return false;
    }

    struct pl_scene_t *scene = ctx->scene;
    struct animation anim = { .value = field, .opt = opt };
    int idx = 0;
    bool ok = true;
    for (const struct json_array_element_s *e = arr->start; e; e = e->next) {
        const size_t len_prefix = ctx->path.len;
        pl_str_append_asprintf(ctx->tmp, &ctx->path, "[%d]", idx);
        struct keyframe kf;
        if (parse_keyframe(ctx, base, opt, field, &kf, e->value)) {
            PL_ARRAY_APPEND(scene, anim.keyframes, kf);
        } else {
            ok = false;
        }
        ctx->path = pl_str_take(ctx->path, len_prefix);
        idx++;
    }

    if (!ok)
        return false;

    // Sort by pts, keeping keyframes with equal pts in the order given
    for (int i = 1; i < anim.keyframes.num; i++) {
        const struct keyframe kf = anim.keyframes.elem[i];
        int j = i;
        for (; j > 0 && anim.keyframes.elem[j - 1].pts > kf.pts; j--)
            anim.keyframes.elem[j] = anim.keyframes.elem[j - 1];
        anim.keyframes.elem[j] = kf;
    }

    // Hold the first value until the first pl_scene_update()
    animation_write(&anim, &anim.keyframes.elem[0]);
    struct priv *p = PL_PRIV(scene);
    PL_ARRAY_APPEND(scene, p->animations, anim);
    return true;
}

static bool parse_value(struct parse_ctx *ctx, void *base, pl_opt opt,
                        void *field, const struct json_value_s *val)
{
    if (val->type == json_type_array)
        return parse_animation(ctx, base, opt, field, val->payload);

    pl_str str;
    if (!json_value_str(val, &str)) {
        PL_ERR(ctx, "Invalid value type for option '%.*s'. Expected string, "
               "number, boolean or array.", PL_STR_FMT(ctx->path));
        return false;
    }

    if (!pl_opt_parse(ctx->log, base, opt, field, str)) {
        PL_ERR(ctx, "Failed parsing option '%.*s' from value: '%.*s'",
               PL_STR_FMT(ctx->path), PL_STR_FMT(str));
        return false;
    }

    return true;
}

static bool parse_struct(struct parse_ctx *ctx, void *base,
                         const struct pl_opt_t *list, void *field,
                         const struct json_value_s *val);

static bool parse_key(struct parse_ctx *ctx, void *base,
                      const struct pl_opt_t *list, void *parent,
                      pl_str key, const struct json_value_s *val)
{
    pl_opt opt = NULL;
    for (pl_opt o = list; o->key; o++) {
        if (pl_str_equals0(key, o->key)) {
            opt = o;
            break;
        }
    }

    if (!opt) {
        PL_ERR(ctx, "Unrecognized option '%.*s'", PL_STR_FMT(ctx->path));
        return false;
    }

    void *field = pl_opt_field(opt, parent);
    if (opt->type == PL_OPT_STRUCT)
        return parse_struct(ctx, base, opt->fields, field, val);
    return parse_value(ctx, base, opt, field, val);
}

static bool parse_struct(struct parse_ctx *ctx, void *base,
                         const struct pl_opt_t *list, void *field,
                         const struct json_value_s *val)
{
    const struct json_object_s *obj = json_value_as_object((void *) val);
    if (!obj) {
        PL_ERR(ctx, "Expected an object for '%.*s'", PL_STR_FMT(ctx->path));
        return false;
    }

    bool ok = true;
    for (const struct json_object_element_s *e = obj->start; e; e = e->next) {
        const size_t len_prefix = ctx->path.len;
        const pl_str key = json_str(e->name);
        pl_str_append_asprintf(ctx->tmp, &ctx->path, ".%.*s", PL_STR_FMT(key));
        ok &= parse_key(ctx, base, list, field, key, e->value);
        ctx->path = pl_str_take(ctx->path, len_prefix);
    }

    return ok;
}

static bool parse_params(struct parse_ctx *ctx, pl_options opts,
                         const struct json_value_s *val)
{
    struct priv *p = PL_PRIV(ctx->scene);
    const int num_animations = p->animations.num;
    bool ok = parse_struct(ctx, opts, pl_option_list, opts, val);

    // Avoid recompiling shaders whenever an animated value changes
    if (p->animations.num > num_animations)
        opts->params.dynamic_constants = true;
    return ok;
}

static bool parse_input(struct parse_ctx *ctx, struct pl_scene_t *scene,
                        const struct json_value_s *val)
{
    const struct json_object_s *obj = json_value_as_object((void *) val);
    if (!obj) {
        PL_ERR(ctx, "Expected an object for '%.*s'", PL_STR_FMT(ctx->path));
        return false;
    }

    struct priv *p = PL_PRIV(scene);
    pl_options opts = pl_options_alloc(ctx->log);
    pl_options_reset(opts, &pl_render_default_params);
    PL_ARRAY_APPEND(scene, p->input_opts, opts);

    pl_input input = pl_compositor_input_create(scene->comp);
    input->params = &opts->params;

    bool ok = true;
    for (const struct json_object_element_s *e = obj->start; e; e = e->next) {
        const size_t len_prefix = ctx->path.len;
        const pl_str key = json_str(e->name);
        pl_str_append_asprintf(ctx->tmp, &ctx->path, ".%.*s", PL_STR_FMT(key));
        if (pl_str_equals0(key, "params")) {
            ok &= parse_params(ctx, opts, e->value);
        } else {
            ok &= parse_key(ctx, input, pl_input_opts, input, key, e->value);
        }
        ctx->path = pl_str_take(ctx->path, len_prefix);
    }

    return ok;
}

static bool parse_inputs(struct parse_ctx *ctx, struct pl_scene_t *scene,
                         const struct json_value_s *val)
{
    const struct json_array_s *arr = json_value_as_array((void *) val);
    if (!arr) {
        PL_ERR(ctx, "Expected an array for '%.*s'", PL_STR_FMT(ctx->path));
        return false;
    }

    int idx = 0;
    bool ok = true;
    for (const struct json_array_element_s *e = arr->start; e; e = e->next) {
        const size_t len_prefix = ctx->path.len;
        pl_str_append_asprintf(ctx->tmp, &ctx->path, "[%d]", idx);
        ok &= parse_input(ctx, scene, e->value);
        ctx->path = pl_str_take(ctx->path, len_prefix);
        idx++; // increment even if parse_input() fails, for error reporting
    }

    return ok;
}

static bool parse_scene(struct parse_ctx *ctx, struct pl_scene_t *scene,
                        const struct json_value_s *val)
{
    struct priv *p = PL_PRIV(scene);
    const struct json_object_s *obj = json_value_as_object((void *) val);
    if (!obj) {
        PL_ERR(ctx, "Expected a JSON object at the top level");
        return false;
    }

    bool ok = true;
    for (const struct json_object_element_s *e = obj->start; e; e = e->next) {
        const pl_str key = json_str(e->name);
        pl_str_append(ctx->tmp, &ctx->path, key);
        if (pl_str_equals0(key, "params")) {
            ok &= parse_params(ctx, p->opts, e->value);
        } else if (pl_str_equals0(key, "inputs")) {
            ok &= parse_inputs(ctx, scene, e->value);
        } else {
            ok &= parse_key(ctx, scene->comp, pl_compositor_opts, scene->comp,
                            key, e->value);
        }
        ctx->path.len = 0;
    }

    return ok;
}

static void *json_alloc(void *priv, size_t size)
{
    struct parse_ctx *ctx = priv;
    return pl_alloc(ctx->tmp, size);
}

pl_scene pl_scene_create_json(pl_log log, const char *json, size_t len)
{
    struct pl_scene_t *scene = pl_zalloc_obj(NULL, scene, struct priv);
    struct priv *p = PL_PRIV(scene);
    p->opts = pl_options_alloc(log);
    pl_options_reset(p->opts, &pl_render_default_params);

    scene->comp = pl_compositor_create(log);
    scene->comp->params = &p->opts->params;

    struct parse_ctx ctx = {
        .log   = log,
        .tmp   = pl_tmp(NULL),
        .scene = scene,
    };

    struct json_value_s *root;
    struct json_parse_result_s res = {0};
    root = json_parse_ex(json, len, 0, json_alloc, &ctx, &res);

    bool ok;
    if (root) {
        ok = parse_scene(&ctx, scene, root);
    } else {
        PL_ERR(&ctx, "Failed parsing JSON: %s (line %zu, column %zu)",
               json_errstr(res.error), res.error_line_no, res.error_row_no);
        ok = false;
    }

    if (!ok)
        pl_scene_destroy((pl_scene *) &scene);

    pl_free(ctx.tmp);
    return scene;
}
