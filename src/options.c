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

#include <math.h>

#include "common.h"
#include "log.h"

#include <libplacebo/options.h>

struct priv {
    pl_log log;

    // for pl_options_get
    struct pl_opt_data_t data;
    pl_str data_text;

    // for pl_options_save
    pl_str saved;

    // internally managed hooks array
    PL_ARRAY(const struct pl_hook *) hooks;
};

static const struct pl_options_t defaults = {
    .params             = { PL_RENDER_DEFAULTS },
    .deband_params      = { PL_DEBAND_DEFAULTS },
    .sigmoid_params     = { PL_SIGMOID_DEFAULTS },
    .color_adjustment   = { PL_COLOR_ADJUSTMENT_NEUTRAL },
    .peak_detect_params = { PL_PEAK_DETECT_DEFAULTS },
    .color_map_params   = { PL_COLOR_MAP_DEFAULTS },
    .dither_params      = { PL_DITHER_DEFAULTS },
    .icc_params         = { PL_ICC_DEFAULTS },
    .cone_params        = { PL_CONE_NONE, 1.0 },
    .deinterlace_params = { PL_DEINTERLACE_DEFAULTS },
    .distort_params     = { PL_DISTORT_DEFAULTS },
    .upscaler = {
        .name           = "custom",
        .description    = "Custom upscaler",
        .allowed        = PL_FILTER_UPSCALING,
    },
    .downscaler = {
        .name           = "custom",
        .description    = "Custom downscaler",
        .allowed        = PL_FILTER_DOWNSCALING,
    },
    .plane_upscaler = {
        .name           = "custom",
        .description    = "Custom plane upscaler",
        .allowed        = PL_FILTER_UPSCALING,
    },
    .plane_downscaler = {
        .name           = "custom",
        .description    = "Custom plane downscaler",
        .allowed        = PL_FILTER_DOWNSCALING,
    },
    .frame_mixer = {
        .name           = "custom",
        .description    = "Custom frame mixer",
        .allowed        = PL_FILTER_FRAME_MIXING,
    },
};

// Copies only whitelisted fields
static inline void copy_filter(struct pl_filter_config *dst,
                               const struct pl_filter_config *src)
{
    dst->kernel = src->kernel;
    dst->window = src->window;
    dst->radius = src->radius;
    dst->clamp  = src->clamp;
    dst->blur   = src->blur;
    dst->taper  = src->taper;
    dst->polar  = src->polar;
    for (int i = 0; i < PL_FILTER_MAX_PARAMS; i++) {
        dst->params[i]  = src->params[i];
        dst->wparams[i] = src->wparams[i];
    }
}

static inline void redirect_params(pl_options opts)
{
    // Copy all non-NULL params structs into pl_options and redirect them
#define REDIRECT_PARAMS(field) do          \
{                                          \
    if (opts->params.field) {              \
        opts->field = *opts->params.field; \
        opts->params.field = &opts->field; \
    }                                      \
} while (0)

    REDIRECT_PARAMS(deband_params);
    REDIRECT_PARAMS(sigmoid_params);
    REDIRECT_PARAMS(color_adjustment);
    REDIRECT_PARAMS(peak_detect_params);
    REDIRECT_PARAMS(color_map_params);
    REDIRECT_PARAMS(dither_params);
    REDIRECT_PARAMS(icc_params);
    REDIRECT_PARAMS(cone_params);
    REDIRECT_PARAMS(deinterlace_params);
    REDIRECT_PARAMS(distort_params);
}

void pl_options_reset(pl_options opts, const struct pl_render_params *preset)
{
    *opts = defaults;
    if (preset)
        opts->params = *preset;
    redirect_params(opts);

    // Make a copy of all scaler configurations that aren't built-in filters
    struct {
        bool upscaler;
        bool downscaler;
        bool plane_upscaler;
        bool plane_downscaler;
        bool frame_mixer;
    } fixed = {0};

    for (int i = 0; i < pl_num_filter_configs; i++) {
        const struct pl_filter_config *f = pl_filter_configs[i];
        fixed.upscaler         |= f == opts->params.upscaler;
        fixed.downscaler       |= f == opts->params.downscaler;
        fixed.plane_upscaler   |= f == opts->params.plane_upscaler;
        fixed.plane_downscaler |= f == opts->params.plane_downscaler;
        fixed.frame_mixer      |= f == opts->params.frame_mixer;
    }

#define REDIRECT_SCALER(scaler) do                       \
{                                                        \
    if (opts->params.scaler && !fixed.scaler) {          \
        copy_filter(&opts->scaler, opts->params.scaler); \
        opts->params.scaler = &opts->scaler;             \
    }                                                    \
} while (0)

    REDIRECT_SCALER(upscaler);
    REDIRECT_SCALER(downscaler);
    REDIRECT_SCALER(plane_upscaler);
    REDIRECT_SCALER(plane_downscaler);
    REDIRECT_SCALER(frame_mixer);
}

pl_options pl_options_alloc(pl_log log)
{
    struct pl_options_t *opts = pl_zalloc_obj(NULL, opts, struct priv);
    struct priv *p = PL_PRIV(opts);
    pl_options_reset(opts, NULL);
    p->log = log;
    return opts;
}

void pl_options_free(pl_options *popts)
{
    pl_free_ptr((void **) popts);
}

static void make_hooks_internal(pl_options opts)
{
    struct priv *p = PL_PRIV(opts);
    struct pl_render_params *params = &opts->params;
    if (params->num_hooks && params->hooks != p->hooks.elem) {
        PL_ARRAY_MEMDUP(opts, p->hooks, params->hooks, params->num_hooks);
        params->hooks = p->hooks.elem;
    }
}

void pl_options_add_hook(pl_options opts, const struct pl_hook *hook)
{
    struct priv *p = PL_PRIV(opts);
    make_hooks_internal(opts);
    PL_ARRAY_APPEND(opts, p->hooks, hook);
    opts->params.hooks = p->hooks.elem;
}

void pl_options_insert_hook(pl_options opts, const struct pl_hook *hook, int idx)
{
    struct priv *p = PL_PRIV(opts);
    make_hooks_internal(opts);
    PL_ARRAY_INSERT_AT(opts, p->hooks, idx, hook);
    opts->params.hooks = p->hooks.elem;
}

void pl_options_remove_hook_at(pl_options opts, int idx)
{
    struct priv *p = PL_PRIV(opts);
    make_hooks_internal(opts);
    PL_ARRAY_REMOVE_AT(p->hooks, idx);
    opts->params.hooks = p->hooks.elem;
}

// Options printing/parsing context
typedef const struct opt_ctx_t {
    pl_log log; // as a convenience, only needed when parsing
    pl_opt opt;
    void *alloc; // for printing only
    pl_options opts; // current base ptr
} *opt_ctx;

struct enum_val {
    const char *name;
    unsigned val;
};

struct preset {
    const char *name;
    const void *val;
};

struct named {
    const char *name;
};

typedef const struct opt_priv_t {
    int (*compare)(opt_ctx p, const void *a, const void *b); // optional
    void (*print)(opt_ctx p, pl_str *out, const void *val); // apends to `out`
    bool (*parse)(opt_ctx p, pl_str str, void *out_val);
    const struct enum_val *values; // for enums, terminated by {0}
    const struct preset *presets; // for preset lists, terminated by {0}
    const struct named * const *names; // for array-backed options, terminated by NULL

    // Offset and size of option in `struct pl_options_t`
    size_t offset;
    size_t size;
    size_t offset_params; // offset of actual struct (for params toggles)
} *opt_priv;

static pl_opt_data get_opt_data(opt_ctx ctx)
{
    pl_options opts = ctx->opts;
    struct priv *p = PL_PRIV(opts);
    opt_priv priv = ctx->opt->priv;
    const void *val = (void *) ((uintptr_t) opts + priv->offset);

    p->data_text.len = 0;
    priv->print(ctx, &p->data_text, val);
    p->data = (struct pl_opt_data_t) {
        .opts  = opts,
        .opt   = ctx->opt,
        .value = val,
        .text  = (char *) p->data_text.buf,
    };

    return &p->data;
}

pl_opt_data pl_options_get(pl_options opts, const char *key)
{
    struct priv *p = PL_PRIV(opts);

    pl_opt opt = pl_find_option(key);
    if (!opt || opt->preset) {
        PL_ERR(p, "Unrecognized or invalid option '%s'", key);
        return NULL;
    }

    return get_opt_data(&(struct opt_ctx_t) {
        .alloc = opts,
        .opts  = opts,
        .opt   = opt,
    });
}

void pl_options_iterate(pl_options opts,
                        void (*cb)(void *priv, pl_opt_data data),
                        void *cb_priv)
{
    for (pl_opt opt = pl_option_list; opt->key; opt++) {
        if (opt->preset)
            continue;

        struct opt_ctx_t ctx = {
            .alloc = opts,
            .opts  = opts,
            .opt   = opt,
        };

        opt_priv priv = opt->priv;
        const void *val = (void *) ((uintptr_t) opts + priv->offset);
        const void *ref = (void *) ((uintptr_t) &defaults + priv->offset);
        int cmp = priv->compare ? priv->compare(&ctx, val, ref)
                                : memcmp(val, ref, priv->size);
        if (cmp != 0)
            cb(cb_priv, get_opt_data(&ctx));
    }
}

static void save_cb(void *priv, pl_opt_data data)
{
    pl_opt opt = data->opt;
    void *alloc = data->opts;
    pl_str *out = priv;

    if (out->len)
        pl_str_append_raw(alloc, out, ",", 1);
    pl_str_append_raw(alloc, out, opt->key, strlen(opt->key));
    pl_str_append_raw(alloc, out, "=", 1);
    pl_str_append(alloc, out, pl_str0(data->text));
}

const char *pl_options_save(pl_options opts)
{
    struct priv *p = PL_PRIV(opts);

    p->saved.len = 0;
    pl_options_iterate(opts, save_cb, &p->saved);
    return p->saved.len ? (char *) p->saved.buf : "";
}

static bool option_set_raw(pl_options opts, pl_str k, pl_str v)
{
    struct priv *p = PL_PRIV(opts);
    k = pl_str_strip(k);
    v = pl_str_strip(v);

    pl_opt opt;
    for (opt = pl_option_list; opt->key; opt++) {
        if (pl_str_equals0(k, opt->key))
            goto found;
    }

    PL_ERR(p, "Unrecognized option '%.*s', in '%.*s=%.*s'",
           PL_STR_FMT(k), PL_STR_FMT(k), PL_STR_FMT(v));
    return false;

found:
    PL_TRACE(p, "Parsing option '%s' = '%.*s'", opt->key, PL_STR_FMT(v));
    if (opt->deprecated)
        PL_WARN(p, "Option '%s' is deprecated", opt->key);

    struct opt_ctx_t ctx = {
        .log  = p->log,
        .opts = opts,
        .opt  = opt,
    };

    opt_priv priv = opt->priv;
    void *val = (void *) ((uintptr_t) opts + priv->offset);
    return priv->parse(&ctx, v, val);
}

bool pl_options_set_str(pl_options opts, const char *key, const char *value)
{
    return option_set_raw(opts, pl_str0(key), pl_str0(value));
}

bool pl_options_load(pl_options opts, const char *str)
{
    bool ret = true;

    pl_str rest = pl_str0(str);
    while (rest.len) {
        pl_str kv = pl_str_strip(pl_str_split_chars(rest, " ,;:\n", &rest));
        if (!kv.len)
            continue;
        pl_str v, k = pl_str_split_char(kv, '=', &v);
        ret &= option_set_raw(opts, k, v);
    }

    return ret;
}

// Individual option types

static void print_bool(opt_ctx p, pl_str *out, const void *ptr)
{
    const bool *val = ptr;
    if (*val) {
        pl_str_append(p->alloc, out, pl_str0("yes"));
    } else {
        pl_str_append(p->alloc, out, pl_str0("no"));
    }
}

static bool parse_bool(opt_ctx p, pl_str str, void *out)
{
    bool *res = out;
    if (pl_str_equals0(str, "yes") ||
        pl_str_equals0(str, "y") ||
        pl_str_equals0(str, "on") ||
        pl_str_equals0(str, "true") ||
        pl_str_equals0(str, "enabled") ||
        !str.len) // accept naked option name as well
    {
        *res = true;
        return true;
    } else if (pl_str_equals0(str, "no") ||
               pl_str_equals0(str, "n") ||
               pl_str_equals0(str, "off") ||
               pl_str_equals0(str, "false") ||
               pl_str_equals0(str, "disabled"))
    {
        *res = false;
        return true;
    }

    PL_ERR(p, "Invalid value '%.*s' for option '%s', expected boolean",
           PL_STR_FMT(str), p->opt->key);
    return false;
}

static void print_int(opt_ctx p, pl_str *out, const void *ptr)
{
    pl_opt opt = p->opt;
    const int *val = ptr;
    pl_assert(opt->min == opt->max || (*val >= opt->min && *val <= opt->max));
    pl_str_append_asprintf_c(p->alloc, out, "%d", *val);
}

static bool parse_int(opt_ctx p, pl_str str, void *out)
{
    pl_opt opt = p->opt;
    int val;
    if (!pl_str_parse_int(str, &val)) {
        PL_ERR(p, "Invalid value '%.*s' for option '%s', expected integer",
               PL_STR_FMT(str), opt->key);
        return false;
    }

    if (opt->min != opt->max) {
        if (val < opt->min || val > opt->max) {
            PL_ERR(p, "Value of %d out of range for option '%s': [%d, %d]",
                   val, opt->key, (int) opt->min, (int) opt->max);
            return false;
        }
    }

    *(int *) out = val;
    return true;
}

static void print_float(opt_ctx p, pl_str *out, const void *ptr)
{
    pl_opt opt = p->opt;
    const float *val = ptr;
    pl_assert(opt->min == opt->max || (*val >= opt->min && *val <= opt->max));
    pl_str_append_asprintf_c(p->alloc, out, "%f", *val);
}

static bool parse_fraction(pl_str str, float *val)
{
    pl_str denom, num = pl_str_split_char(str, '/', &denom);
    float n, d;
    bool ok = denom.buf && denom.len && pl_str_parse_float(num, &n) &&
                                        pl_str_parse_float(denom, &d);
    if (ok)
        *val = n / d;
    return ok;
}

static bool parse_float(opt_ctx p, pl_str str, void *out)
{
    pl_opt opt = p->opt;
    float val;
    if (!parse_fraction(str, &val) && !pl_str_parse_float(str, &val)) {
        PL_ERR(p, "Invalid value '%.*s' for option '%s', expected floating point "
                  "or fraction", PL_STR_FMT(str), opt->key);
        return false;
    }

    switch (fpclassify(val)) {
    case FP_NAN:
    case FP_INFINITE:
    case FP_SUBNORMAL:
        PL_ERR(p, "Invalid value '%f' for option '%s', non-normal float",
               val, opt->key);
        return false;

    case FP_ZERO:
    case FP_NORMAL:
        break;
    }

    if (opt->min != opt->max) {
        if (val < opt->min || val > opt->max) {
            PL_ERR(p, "Value of %.3f out of range for option '%s': [%.2f, %.2f]",
                   val, opt->key, opt->min, opt->max);
            return false;
        }
    }

    *(float *) out = val;
    return true;
}

static int compare_params(opt_ctx p, const void *pa, const void *pb)
{
    const bool a = *(const void * const *) pa;
    const bool b = *(const void * const *) pb;
    return PL_CMP(a, b);
}

static void print_params(opt_ctx p, pl_str *out, const void *ptr)
{
    const bool value = *(const void * const *) ptr;
    print_bool(p, out, &value);
}

static bool parse_params(opt_ctx p, pl_str str, void *out)
{
    pl_opt opt = p->opt;
    opt_priv priv = opt->priv;
    const void **res = out;
    bool set;
    if (!parse_bool(p, str, &set))
        return false;
    if (set) {
        *res = (const void *) ((uintptr_t) p->opts + priv->offset_params);
    } else {
        *res = NULL;
    }
    return true;
}

static void print_enum(opt_ctx p, pl_str *out, const void *ptr)
{
    pl_opt opt = p->opt;
    opt_priv priv = opt->priv;
    const unsigned value = *(const unsigned *) ptr;
    for (int i = 0; priv->values[i].name; i++) {
        if (priv->values[i].val == value) {
            pl_str_append(p->alloc, out, pl_str0(priv->values[i].name));
            return;
        }
    }

    pl_unreachable();
}

static bool parse_enum(opt_ctx p, pl_str str, void *out)
{
    pl_opt opt = p->opt;
    opt_priv priv = opt->priv;
    for (int i = 0; priv->values[i].name; i++) {
        if (pl_str_equals0(str, priv->values[i].name)) {
            *(unsigned *) out = priv->values[i].val;
            return true;
        }
    }

    PL_ERR(p, "Value of '%.*s' unrecognized for option '%s', valid values:",
           PL_STR_FMT(str), opt->key);
    for (int i = 0; priv->values[i].name; i++)
        PL_ERR(p, "  %s", priv->values[i].name);
    return false;
}

static bool parse_preset(opt_ctx p, pl_str str, void *out)
{
    pl_opt opt = p->opt;
    opt_priv priv = opt->priv;
    for (int i = 0; priv->presets[i].name; i++) {
        if (pl_str_equals0(str, priv->presets[i].name)) {
            if (priv->offset == offsetof(struct pl_options_t, params)) {
                const struct pl_render_params *preset = priv->presets[i].val;
                pl_assert(priv->size == sizeof(*preset));

                // Redirect params structs into internal system after loading
                struct pl_render_params *params = out, prev = *params;
                *params = *preset;
                redirect_params(p->opts);

                // Re-apply excluded options
                params->lut = prev.lut;
                params->hooks = prev.hooks;
                params->num_hooks = prev.num_hooks;
                params->info_callback = prev.info_callback;
                params->info_priv = prev.info_priv;
            } else {
                memcpy(out, priv->presets[i].val, priv->size);
            }
            return true;
        }
    }

    PL_ERR(p, "Value of '%.*s' unrecognized for option '%s', valid values:",
           PL_STR_FMT(str), opt->key);
    for (int i = 0; priv->presets[i].name; i++)
        PL_ERR(p, "  %s", priv->presets[i].name);
    return false;
}

static void print_named(opt_ctx p, pl_str *out, const void *ptr)
{
    const struct named *value = *(const struct named **) ptr;
    if (value) {
        pl_str_append(p->alloc, out, pl_str0(value->name));
    } else {
        pl_str_append(p->alloc, out, pl_str0("none"));
    }
}

static bool parse_named(opt_ctx p, pl_str str, void *out)
{
    pl_opt opt = p->opt;
    opt_priv priv = opt->priv;
    const struct named **res = out;
    if (pl_str_equals0(str, "none")) {
        *res = NULL;
        return true;
    }

    for (int i = 0; priv->names[i]; i++) {
        if (pl_str_equals0(str, priv->names[i]->name)) {
            *res = priv->names[i];
            return true;
        }
    }

    PL_ERR(p, "Value of '%.*s' unrecognized for option '%s', valid values:",
           PL_STR_FMT(str), opt->key);
    PL_ERR(p, "  none");
    for (int i = 0; priv->names[i]; i++)
        PL_ERR(p, "  %s", priv->names[i]->name);
    return false;
}

static void print_scaler(opt_ctx p, pl_str *out, const void *ptr)
{
    const struct pl_filter_config *f = *(const struct pl_filter_config **) ptr;
    if (f) {
        pl_assert(f->name); // this is either a built-in scaler or ptr to custom
        pl_str_append(p->alloc, out, pl_str0(f->name));
    } else {
        pl_str_append(p->alloc, out, pl_str0("none"));
    }
}

static enum pl_filter_usage scaler_usage(pl_opt opt)
{
    opt_priv priv = opt->priv;
    switch (priv->offset) {
    case offsetof(struct pl_options_t, params.upscaler):
    case offsetof(struct pl_options_t, params.plane_upscaler):
    case offsetof(struct pl_options_t, upscaler):
    case offsetof(struct pl_options_t, plane_upscaler):
        return PL_FILTER_UPSCALING;

    case offsetof(struct pl_options_t, params.downscaler):
    case offsetof(struct pl_options_t, params.plane_downscaler):
    case offsetof(struct pl_options_t, downscaler):
    case offsetof(struct pl_options_t, plane_downscaler):
        return PL_FILTER_DOWNSCALING;

    case offsetof(struct pl_options_t, params.frame_mixer):
    case offsetof(struct pl_options_t, frame_mixer):
        return PL_FILTER_FRAME_MIXING;
    }

    pl_unreachable();
}

static bool parse_scaler(opt_ctx p, pl_str str, void *out)
{
    pl_opt opt = p->opt;
    opt_priv priv = opt->priv;
    const struct pl_filter_config **res = out;
    if (pl_str_equals0(str, "none")) {
        *res = NULL;
        return true;
    } else if (pl_str_equals0(str, "custom")) {
        *res = (void *) ((uintptr_t) p->opts + priv->offset_params);
        return true;
    }

    const enum pl_filter_usage usage = scaler_usage(opt);
    for (int i = 0; i < pl_num_filter_configs; i++) {
        if (!(pl_filter_configs[i]->allowed & usage))
            continue;
        if (pl_str_equals0(str, pl_filter_configs[i]->name)) {
            *res = pl_filter_configs[i];
            return true;
        }
    }

    PL_ERR(p, "Value of '%.*s' unrecognized for option '%s', valid values:",
           PL_STR_FMT(str), opt->key);
    PL_ERR(p, "  none");
    PL_ERR(p, "  custom");
    for (int i = 0; i < pl_num_filter_configs; i++) {
        if (pl_filter_configs[i]->allowed & usage)
            PL_ERR(p, "  %s", pl_filter_configs[i]->name);
    }
    return false;
}

static bool parse_scaler_preset(opt_ctx p, pl_str str, void *out)
{
    pl_opt opt = p->opt;
    struct pl_filter_config *res = out;
    if (pl_str_equals0(str, "none")) {
        *res = (struct pl_filter_config) { .name = "custom" };
        return true;
    }

    const enum pl_filter_usage usage = scaler_usage(opt);
    for (int i = 0; i < pl_num_filter_configs; i++) {
        if (!(pl_filter_configs[i]->allowed & usage))
            continue;
        if (pl_str_equals0(str, pl_filter_configs[i]->name)) {
            copy_filter(res, pl_filter_configs[i]);
            return true;
        }
    }

    PL_ERR(p, "Value of '%.*s' unrecognized for option '%s', valid values:",
           PL_STR_FMT(str), opt->key);
    PL_ERR(p, "  none");
    for (int i = 0; i < pl_num_filter_configs; i++) {
        if (pl_filter_configs[i]->allowed & usage)
            PL_ERR(p, "  %s", pl_filter_configs[i]->name);
    }
    return false;
}

#define OPT_BOOL(KEY, NAME, FIELD, ...)                                         \
    {                                                                           \
        .key  = KEY,                                                            \
        .name = NAME,                                                           \
        .type = PL_OPT_BOOL,                                                    \
        .priv = &(struct opt_priv_t) {                                          \
            .print  = print_bool,                                               \
            .parse  = parse_bool,                                               \
            .offset = offsetof(struct pl_options_t, FIELD),                     \
            .size   = sizeof(struct {                                           \
                bool dummy;                                                     \
                pl_static_assert(sizeof(defaults.FIELD) == sizeof(bool));       \
            }),                                                                 \
        },                                                                      \
        __VA_ARGS__                                                             \
    }

#define OPT_INT(KEY, NAME, FIELD, ...)                                          \
    {                                                                           \
        .key  = KEY,                                                            \
        .name = NAME,                                                           \
        .type = PL_OPT_INT,                                                     \
        .priv = &(struct opt_priv_t) {                                          \
            .print  = print_int,                                                \
            .parse  = parse_int,                                                \
            .offset = offsetof(struct pl_options_t, FIELD),                     \
            .size   = sizeof(struct {                                           \
                int dummy;                                                      \
                pl_static_assert(sizeof(defaults.FIELD) == sizeof(int));        \
            }),                                                                 \
        },                                                                      \
        __VA_ARGS__                                                             \
    }

#define OPT_FLOAT(KEY, NAME, FIELD, ...)                                        \
    {                                                                           \
        .key  = KEY,                                                            \
        .name = NAME,                                                           \
        .type = PL_OPT_FLOAT,                                                   \
        .priv = &(struct opt_priv_t) {                                          \
            .print  = print_float,                                              \
            .parse  = parse_float,                                              \
            .offset = offsetof(struct pl_options_t, FIELD),                     \
            .size   = sizeof(struct {                                           \
                float dummy;                                                    \
                pl_static_assert(sizeof(defaults.FIELD) == sizeof(float));      \
            }),                                                                 \
        },                                                                      \
        __VA_ARGS__                                                             \
    }

#define OPT_ENABLE_PARAMS(KEY, NAME, PARAMS, ...)                               \
    {                                                                           \
        .key  = KEY,                                                            \
        .name = NAME,                                                           \
        .type = PL_OPT_BOOL,                                                    \
        .priv = &(struct opt_priv_t) {                                          \
            .compare       = compare_params,                                    \
            .print         = print_params,                                      \
            .parse         = parse_params,                                      \
            .offset        = offsetof(struct pl_options_t, params.PARAMS),      \
            .offset_params = offsetof(struct pl_options_t, PARAMS),             \
            .size          = sizeof(struct {                                    \
                void *dummy;                                                    \
                pl_static_assert(sizeof(defaults.params.PARAMS) == sizeof(void*));\
            }),                                                                 \
        },                                                                      \
        __VA_ARGS__                                                             \
    }

#define OPT_ENUM(KEY, NAME, FIELD, VALUES, ...)                                 \
    {                                                                           \
        .key  = KEY,                                                            \
        .name = NAME,                                                           \
        .type = PL_OPT_STRING,                                                  \
        .priv = &(struct opt_priv_t) {                                          \
            .print  = print_enum,                                               \
            .parse  = parse_enum,                                               \
            .offset = offsetof(struct pl_options_t, FIELD),                     \
            .size   = sizeof(struct {                                           \
                unsigned dummy;                                                 \
                pl_static_assert(sizeof(defaults.FIELD) == sizeof(unsigned));   \
            }),                                                                 \
            .values = (struct enum_val[]) { VALUES }                            \
        },                                                                      \
        __VA_ARGS__                                                             \
    }

#define OPT_PRESET(KEY, NAME, PARAMS, PRESETS, ...)                             \
    {                                                                           \
        .key    = KEY,                                                          \
        .name   = NAME,                                                         \
        .type   = PL_OPT_STRING,                                                \
        .preset = true,                                                         \
        .priv   = &(struct opt_priv_t) {                                        \
            .parse   = parse_preset,                                            \
            .offset  = offsetof(struct pl_options_t, PARAMS),                   \
            .size    = sizeof(defaults.PARAMS),                                 \
            .presets = (struct preset[]) { PRESETS },                           \
        },                                                                      \
        __VA_ARGS__                                                             \
    }

#define OPT_NAMED(KEY, NAME, FIELD, NAMES, ...)                                 \
    {                                                                           \
        .key  = KEY,                                                            \
        .name = NAME,                                                           \
        .type = PL_OPT_STRING,                                                  \
        .priv = &(struct opt_priv_t) {                                          \
            .print  = print_named,                                              \
            .parse  = parse_named,                                              \
            .offset = offsetof(struct pl_options_t, FIELD),                     \
            .names  = (const struct named * const * ) NAMES,                    \
            .size   = sizeof(struct {                                           \
                const struct named *dummy;                                      \
                pl_static_assert(offsetof(__typeof__(*NAMES[0]), name) == 0);   \
                pl_static_assert(sizeof(defaults.FIELD) ==                      \
                                 sizeof(const struct named *));                 \
            }),                                                                 \
        },                                                                      \
        __VA_ARGS__                                                             \
    }

#define OPT_SCALER(KEY, NAME, SCALER, ...)                                      \
    {                                                                           \
        .key  = KEY,                                                            \
        .name = NAME,                                                           \
        .type = PL_OPT_STRING,                                                  \
        .priv = &(struct opt_priv_t) {                                          \
            .print         = print_scaler,                                      \
            .parse         = parse_scaler,                                      \
            .offset        = offsetof(struct pl_options_t, params.SCALER),      \
            .offset_params = offsetof(struct pl_options_t, SCALER),             \
            .size          = sizeof(struct {                                    \
                const struct pl_filter_config *dummy;                           \
                pl_static_assert(sizeof(defaults.SCALER) ==                     \
                                 sizeof(struct pl_filter_config));              \
            }),                                                                 \
        },                                                                      \
        __VA_ARGS__                                                             \
    }

#define OPT_SCALER_PRESET(KEY, NAME, SCALER, ...)                               \
    {                                                                           \
        .key    = KEY,                                                          \
        .name   = NAME,                                                         \
        .type   = PL_OPT_STRING,                                                \
        .preset = true,                                                         \
        .priv   = &(struct opt_priv_t) {                                        \
            .parse         = parse_scaler_preset,                               \
            .offset        = offsetof(struct pl_options_t, SCALER),             \
            .size          = sizeof(struct {                                    \
                struct pl_filter_config dummy;                                  \
                pl_static_assert(sizeof(defaults.SCALER) ==                     \
                                 sizeof(struct pl_filter_config));              \
            }),                                                                 \
        },                                                                      \
        __VA_ARGS__                                                             \
    }

#define LIST(...) __VA_ARGS__, {0}

#define SCALE_OPTS(PREFIX, NAME, FIELD)                                               \
    OPT_SCALER(PREFIX, NAME, FIELD),                                                  \
    OPT_SCALER_PRESET(PREFIX"_preset", NAME "preset", FIELD),                         \
    OPT_NAMED(PREFIX"_kernel", NAME" kernel", FIELD.kernel, pl_filter_functions),     \
    OPT_NAMED(PREFIX"_window", NAME" window", FIELD.window, pl_filter_functions),     \
    OPT_FLOAT(PREFIX"_radius", NAME" radius", FIELD.radius, .min = 1.0, .max = 16.0), \
    OPT_FLOAT(PREFIX"_clamp", NAME" clamping", FIELD.clamp, .max = 1.0),              \
    OPT_FLOAT(PREFIX"_blur", NAME" blur factor", FIELD.blur, .max = 100.0),           \
    OPT_FLOAT(PREFIX"_taper", NAME" taper factor", FIELD.taper, .max = 1.0),          \
    OPT_FLOAT(PREFIX"_antiring", NAME" antiringing", FIELD.antiring, .max = 1.0),     \
    OPT_FLOAT(PREFIX"_param1", NAME" parameter 1", FIELD.params[0]),                  \
    OPT_FLOAT(PREFIX"_param2", NAME" parameter 2", FIELD.params[1]),                  \
    OPT_FLOAT(PREFIX"_wparam1", NAME" window parameter 1", FIELD.wparams[0]),         \
    OPT_FLOAT(PREFIX"_wparam2", NAME" window parameter 2", FIELD.wparams[1]),         \
    OPT_BOOL(PREFIX"_polar", NAME" polar", FIELD.polar)

const struct pl_opt_t pl_option_list[] = {
    OPT_PRESET("preset", "Global preset", params, LIST(
               {"default",      &pl_render_default_params},
               {"fast",         &pl_render_fast_params},
               {"high_quality", &pl_render_high_quality_params})),

    // Scalers
    SCALE_OPTS("upscaler", "Upscaler", upscaler),
    SCALE_OPTS("downscaler", "Downscaler", downscaler),
    SCALE_OPTS("plane_upscaler", "Plane upscaler", plane_upscaler),
    SCALE_OPTS("plane_downscaler", "Plane downscaler", plane_downscaler),
    SCALE_OPTS("frame_mixer", "Frame mixer", frame_mixer),
    OPT_INT("lut_entries", "Scaler LUT entries", params.lut_entries, .max = 256),
    OPT_FLOAT("antiringing_strength", "Anti-ringing strength", params.antiringing_strength, .max = 1.0),

    // Debanding
    OPT_ENABLE_PARAMS("deband", "Enable debanding", deband_params),
    OPT_PRESET("deband_preset", "Debanding preset", deband_params, LIST(
               {"default", &pl_deband_default_params})),
    OPT_INT("deband_iterations", "Debanding iterations", deband_params.iterations, .max = 16),
    OPT_FLOAT("deband_threshold", "Debanding threshold", deband_params.threshold, .max = 1024.0),
    OPT_FLOAT("deband_radius", "Debanding radius", deband_params.radius, .max = 1024.0),
    OPT_FLOAT("deband_grain", "Debanding grain", deband_params.grain, .max = 1024.0),
    OPT_FLOAT("deband_grain_neutral_r", "Debanding grain neutral R", deband_params.grain_neutral[0]),
    OPT_FLOAT("deband_grain_neutral_g", "Debanding grain neutral G", deband_params.grain_neutral[1]),
    OPT_FLOAT("deband_grain_neutral_b", "Debanding grain neutral B", deband_params.grain_neutral[2]),

    // Sigmodization
    OPT_ENABLE_PARAMS("sigmoid", "Enable sigmoidization", sigmoid_params),
    OPT_PRESET("sigmoid_preset", "Sigmoidization preset", sigmoid_params, LIST(
               {"default", &pl_sigmoid_default_params})),
    OPT_FLOAT("sigmoid_center", "Sigmoidization center", sigmoid_params.center, .max = 1.0),
    OPT_FLOAT("sigmoid_slope", "Sigmoidization slope", sigmoid_params.slope, .min = 1.0, .max = 20.0),

    // Color adjustment
    OPT_ENABLE_PARAMS("color_adjustment", "Enable color adjustment", color_adjustment),
    OPT_PRESET("color_adjustment_preset", "Color adjustment preset", color_adjustment, LIST(
               {"neutral", &pl_color_adjustment_neutral})),
    OPT_FLOAT("brightness", "Brightness boost", color_adjustment.brightness, .min = -1.0, .max = 1.0),
    OPT_FLOAT("contrast", "Contrast boost", color_adjustment.contrast, .max = 100.0),
    OPT_FLOAT("saturation", "Saturation gain", color_adjustment.saturation, .max = 100.0),
    OPT_FLOAT("hue", "Hue shift", color_adjustment.hue),
    OPT_FLOAT("gamma", "Gamma adjustment", color_adjustment.gamma, .max = 100.0),
    OPT_FLOAT("temperature", "Color temperature shift", color_adjustment.temperature,
              .min = (2500  - 6500) / 3500.0, // see `pl_white_from_temp`
              .max = (25000 - 6500) / 3500.0),

    // Peak detection
    OPT_ENABLE_PARAMS("peak_detect", "Enable peak detection", peak_detect_params),
    OPT_PRESET("peak_detect_preset", "Peak detection preset", peak_detect_params, LIST(
               {"default",      &pl_peak_detect_default_params},
               {"high_quality", &pl_peak_detect_high_quality_params})),
    OPT_FLOAT("peak_smoothing_period", "Peak detection smoothing coefficient", peak_detect_params.smoothing_period, .max = 1000.0),
    OPT_FLOAT("scene_threshold_low", "Scene change threshold low", peak_detect_params.scene_threshold_low, .max = 100.0),
    OPT_FLOAT("scene_threshold_high", "Scene change threshold high", peak_detect_params.scene_threshold_high, .max = 100.0),
    OPT_FLOAT("minimum_peak", "Minimum detected peak", peak_detect_params.minimum_peak, .max = 100.0, .deprecated = true),
    OPT_FLOAT("peak_percentile", "Peak detection percentile", peak_detect_params.percentile, .max = 100.0),
    OPT_BOOL("allow_delayed_peak", "Allow delayed peak detection", peak_detect_params.allow_delayed),

    // Color mapping
    OPT_ENABLE_PARAMS("color_map", "Enable color mapping", color_map_params),
    OPT_PRESET("color_map_preset", "Color mapping preset", color_map_params, LIST(
               {"default",      &pl_color_map_default_params},
               {"high_quality", &pl_color_map_high_quality_params})),
    OPT_NAMED("gamut_mapping", "Gamut mapping function", color_map_params.gamut_mapping,
              pl_gamut_map_functions),
    OPT_FLOAT("perceptual_deadzone", "Gamut mapping perceptual deadzone", color_map_params.gamut_constants.perceptual_deadzone, .max = 1.0f),
    OPT_FLOAT("perceptual_strength", "Gamut mapping perceptual strength", color_map_params.gamut_constants.perceptual_strength, .max = 1.0f),
    OPT_FLOAT("colorimetric_gamma", "Gamut mapping colorimetric gamma", color_map_params.gamut_constants.colorimetric_gamma, .max = 10.0f),
    OPT_FLOAT("softclip_knee", "Gamut mapping softclip knee point", color_map_params.gamut_constants.softclip_knee, .max = 1.0f),
    OPT_FLOAT("softclip_desat", "Gamut mapping softclip desaturation strength", color_map_params.gamut_constants.softclip_desat, .max = 1.0f),
    OPT_INT("lut3d_size_I", "Gamut 3DLUT size I", color_map_params.lut3d_size[0], .max = 1024),
    OPT_INT("lut3d_size_C", "Gamut 3DLUT size C", color_map_params.lut3d_size[1], .max = 1024),
    OPT_INT("lut3d_size_h", "Gamut 3DLUT size h", color_map_params.lut3d_size[2], .max = 1024),
    OPT_BOOL("lut3d_tricubic", "Gamut 3DLUT tricubic interpolation", color_map_params.lut3d_tricubic),
    OPT_BOOL("gamut_expansion", "Gamut expansion", color_map_params.gamut_expansion),
    OPT_NAMED("tone_mapping", "Tone mapping function", color_map_params.tone_mapping_function,
              pl_tone_map_functions),
    OPT_FLOAT("knee_adaptation", "Tone mapping knee point adaptation", color_map_params.tone_constants.knee_adaptation, .max = 1.0f),
    OPT_FLOAT("knee_minimum", "Tone mapping knee point minimum", color_map_params.tone_constants.knee_minimum, .max = 0.5f),
    OPT_FLOAT("knee_maximum", "Tone mapping knee point maximum", color_map_params.tone_constants.knee_maximum, .min = 0.5f, .max = 1.0f),
    OPT_FLOAT("knee_default", "Tone mapping knee point default", color_map_params.tone_constants.knee_default, .max = 1.0f),
    OPT_FLOAT("knee_offset", "BT.2390 knee point offset", color_map_params.tone_constants.knee_offset, .min = 0.5f, .max = 2.0f),
    OPT_FLOAT("slope_tuning", "Spline slope tuning strength", color_map_params.tone_constants.slope_tuning, .max = 10.0f),
    OPT_FLOAT("slope_offset", "Spline slope tuning offset", color_map_params.tone_constants.slope_offset, .max = 1.0f),
    OPT_FLOAT("spline_contrast", "Spline slope contrast", color_map_params.tone_constants.spline_contrast, .max = 1.5f),
    OPT_FLOAT("reinhard_contrast", "Reinhard contrast", color_map_params.tone_constants.reinhard_contrast, .max = 1.0f),
    OPT_FLOAT("linear_knee", "Tone mapping linear knee point", color_map_params.tone_constants.linear_knee, .max = 1.0f),
    OPT_FLOAT("exposure", "Tone mapping linear exposure", color_map_params.tone_constants.exposure, .max = 10.0f),
    OPT_BOOL("inverse_tone_mapping", "Inverse tone mapping", color_map_params.inverse_tone_mapping),
    OPT_ENUM("tone_map_metadata", "Source of HDR metadata to use", color_map_params.metadata, LIST(
             {"any",       PL_HDR_METADATA_ANY},
             {"none",      PL_HDR_METADATA_NONE},
             {"hdr10",     PL_HDR_METADATA_HDR10},
             {"hdr10plus", PL_HDR_METADATA_HDR10PLUS},
             {"cie_y",     PL_HDR_METADATA_CIE_Y})),
    OPT_INT("tone_lut_size", "Tone mapping LUT size", color_map_params.lut_size, .max = 4096),
    OPT_FLOAT("contrast_recovery", "HDR contrast recovery strength", color_map_params.contrast_recovery, .max = 2.0),
    OPT_FLOAT("contrast_smoothness", "HDR contrast recovery smoothness", color_map_params.contrast_smoothness, .min = 1.0, .max = 32.0),
    OPT_BOOL("force_tone_mapping_lut", "Force tone mapping LUT", color_map_params.force_tone_mapping_lut),
    OPT_BOOL("visualize_lut", "Visualize tone mapping LUTs", color_map_params.visualize_lut),
    OPT_FLOAT("visualize_lut_x0", "Visualization rect x0", color_map_params.visualize_rect.x0),
    OPT_FLOAT("visualize_lut_y0", "Visualization rect y0", color_map_params.visualize_rect.y0),
    OPT_FLOAT("visualize_lut_x1", "Visualization rect x0", color_map_params.visualize_rect.x1),
    OPT_FLOAT("visualize_lut_y1", "Visualization rect y0", color_map_params.visualize_rect.y1),
    OPT_FLOAT("visualize_hue", "Visualization hue slice", color_map_params.visualize_hue),
    OPT_FLOAT("visualize_theta", "Visualization rotation", color_map_params.visualize_theta),
    OPT_BOOL("show_clipping", "Highlight clipped pixels", color_map_params.show_clipping),
    OPT_FLOAT("tone_mapping_param", "Tone mapping function parameter", color_map_params.tone_mapping_param, .deprecated = true),

    // Dithering
    OPT_ENABLE_PARAMS("dither", "Enable dithering", dither_params),
    OPT_PRESET("dither_preset", "Dithering preset", dither_params, LIST(
               {"default", &pl_dither_default_params})),
    OPT_ENUM("dither_method", "Dither method", dither_params.method, LIST(
             {"blue",         PL_DITHER_BLUE_NOISE},
             {"ordered_lut",  PL_DITHER_ORDERED_LUT},
             {"ordered",      PL_DITHER_ORDERED_FIXED},
             {"white",        PL_DITHER_WHITE_NOISE})),
    OPT_INT("dither_lut_size", "Dither LUT size", dither_params.lut_size, .min = 1, .max = 8),
    OPT_BOOL("dither_temporal", "Temporal dithering", dither_params.temporal),

    // ICC
    OPT_ENABLE_PARAMS("icc", "Enable ICC settings", icc_params, .deprecated = true),
    OPT_PRESET("icc_preset", "ICC preset", icc_params, LIST(
               {"default", &pl_icc_default_params}), .deprecated = true),
    OPT_ENUM("icc_intent", "ICC rendering intent", icc_params.intent, LIST(
             {"auto",       PL_INTENT_AUTO},
             {"perceptual", PL_INTENT_PERCEPTUAL},
             {"relative",   PL_INTENT_RELATIVE_COLORIMETRIC},
             {"saturation", PL_INTENT_SATURATION},
             {"absolute",   PL_INTENT_ABSOLUTE_COLORIMETRIC}), .deprecated = true),
    OPT_INT("icc_size_r", "ICC 3DLUT size R", icc_params.size_r, .max = 256, .deprecated = true),
    OPT_INT("icc_size_g", "ICC 3DLUT size G", icc_params.size_g, .max = 256, .deprecated = true),
    OPT_INT("icc_size_b", "ICC 3DLUT size G", icc_params.size_b, .max = 256, .deprecated = true),
    OPT_FLOAT("icc_max_luma", "ICC profile luma override", icc_params.max_luma, .max = 10000, .deprecated = true),
    OPT_BOOL("icc_force_bpc", "Force ICC black point compensation", icc_params.force_bpc, .deprecated = true),

    // Cone distortion
    OPT_ENABLE_PARAMS("cone", "Enable cone distortion", cone_params),
    OPT_PRESET("cone_preset", "Cone distortion preset", cone_params, LIST(
               {"normal",        &pl_vision_normal},
               {"protanomaly",   &pl_vision_protanomaly},
               {"protanopia",    &pl_vision_protanopia},
               {"deuteranomaly", &pl_vision_deuteranomaly},
               {"deuteranopia",  &pl_vision_deuteranopia},
               {"tritanomaly",   &pl_vision_tritanomaly},
               {"tritanopia",    &pl_vision_tritanopia},
               {"monochromacy",  &pl_vision_monochromacy},
               {"achromatopsia", &pl_vision_achromatopsia})),
    OPT_ENUM("cones", "Cone selection", cone_params.cones, LIST(
             {"none", PL_CONE_NONE},
             {"l",    PL_CONE_L},
             {"m",    PL_CONE_M},
             {"s",    PL_CONE_S},
             {"lm",   PL_CONE_LM},
             {"ms",   PL_CONE_MS},
             {"ls",   PL_CONE_LS},
             {"lms",  PL_CONE_LMS})),
    OPT_FLOAT("cone_strength", "Cone distortion gain", cone_params.strength),

    // Blending
#define BLEND_VALUES LIST(                       \
        {"zero",            PL_BLEND_ZERO},      \
        {"one",             PL_BLEND_ONE},       \
        {"alpha",           PL_BLEND_SRC_ALPHA}, \
        {"one_minus_alpha", PL_BLEND_ONE_MINUS_SRC_ALPHA})

    OPT_ENABLE_PARAMS("blend", "Enable output blending", blend_params),
    OPT_PRESET("blend_preset", "Output blending preset", blend_params, LIST(
               {"alpha_overlay", &pl_alpha_overlay})),
    OPT_ENUM("blend_src_rgb", "Source RGB blend mode", blend_params.src_rgb, BLEND_VALUES),
    OPT_ENUM("blend_src_alpha", "Source alpha blend mode", blend_params.src_alpha, BLEND_VALUES),
    OPT_ENUM("blend_dst_rgb", "Target RGB blend mode", blend_params.dst_rgb, BLEND_VALUES),
    OPT_ENUM("blend_dst_alpha", "Target alpha blend mode", blend_params.dst_alpha, BLEND_VALUES),

    // Deinterlacing
    OPT_ENABLE_PARAMS("deinterlace", "Enable deinterlacing", deinterlace_params),
    OPT_PRESET("deinterlace_preset", "Deinterlacing preset", deinterlace_params, LIST(
               {"default", &pl_deinterlace_default_params})),
    OPT_ENUM("deinterlace_algo", "Deinterlacing algorithm", deinterlace_params.algo, LIST(
             {"weave", PL_DEINTERLACE_WEAVE},
             {"bob",   PL_DEINTERLACE_BOB},
             {"yadif", PL_DEINTERLACE_YADIF})),
    OPT_BOOL("deinterlace_skip_spatial", "Skip spatial interlacing check", deinterlace_params.skip_spatial_check),

    // Distortion
    OPT_ENABLE_PARAMS("distort", "Enable distortion", distort_params),
    OPT_PRESET("distort_preset", "Distortion preset", distort_params, LIST(
               {"default", &pl_distort_default_params})),
    OPT_FLOAT("distort_scale_x", "Distortion X scale", distort_params.transform.mat.m[0][0]),
    OPT_FLOAT("distort_scale_y", "Distortion Y scale", distort_params.transform.mat.m[1][1]),
    OPT_FLOAT("distort_shear_x", "Distortion X shear", distort_params.transform.mat.m[0][1]),
    OPT_FLOAT("distort_shear_y", "Distortion Y shear", distort_params.transform.mat.m[1][0]),
    OPT_FLOAT("distort_offset_x", "Distortion X offset", distort_params.transform.c[0]),
    OPT_FLOAT("distort_offset_y", "Distortion Y offset", distort_params.transform.c[1]),
    OPT_BOOL("distort_unscaled", "Distortion unscaled", distort_params.unscaled),
    OPT_BOOL("distort_constrain", "Constrain distortion", distort_params.constrain),
    OPT_BOOL("distort_bicubic", "Distortion bicubic interpolation", distort_params.bicubic),
    OPT_ENUM("distort_address_mode", "Distortion texture address mode", distort_params.address_mode, LIST(
             {"clamp",  PL_TEX_ADDRESS_CLAMP},
             {"repeat", PL_TEX_ADDRESS_REPEAT},
             {"mirror", PL_TEX_ADDRESS_MIRROR})),
    OPT_ENUM("distort_alpha_mode", "Distortion alpha blending mode", distort_params.alpha_mode, LIST(
             {"none",          PL_ALPHA_UNKNOWN},
             {"independent",   PL_ALPHA_INDEPENDENT},
             {"premultiplied", PL_ALPHA_PREMULTIPLIED})),

    // Misc renderer settings
    OPT_NAMED("error_diffusion", "Error diffusion kernel", params.error_diffusion,
              pl_error_diffusion_kernels),
    OPT_ENUM("lut_type", "Color mapping LUT type", params.lut_type, LIST(
             {"unknown",    PL_LUT_UNKNOWN},
             {"native",     PL_LUT_NATIVE},
             {"normalized", PL_LUT_NORMALIZED},
             {"conversion", PL_LUT_CONVERSION})),
    OPT_FLOAT("background_r", "Background color R", params.background_color[0], .max = 1.0),
    OPT_FLOAT("background_g", "Background color G", params.background_color[1], .max = 1.0),
    OPT_FLOAT("background_b", "Background color B", params.background_color[2], .max = 1.0),
    OPT_FLOAT("background_transparency", "Background color transparency", params.background_transparency, .max = 1),
    OPT_BOOL("skip_target_clearing", "Skip target clearing", params.skip_target_clearing),
    OPT_FLOAT("corner_rounding", "Corner rounding", params.corner_rounding, .max = 1.0),
    OPT_BOOL("blend_against_tiles", "Blend against tiles", params.blend_against_tiles),
    OPT_FLOAT("tile_color_hi_r", "Bright tile R", params.tile_colors[0][0], .max = 1.0),
    OPT_FLOAT("tile_color_hi_g", "Bright tile G", params.tile_colors[0][1], .max = 1.0),
    OPT_FLOAT("tile_color_hi_b", "Bright tile B", params.tile_colors[0][2], .max = 1.0),
    OPT_FLOAT("tile_color_lo_r", "Dark tile R", params.tile_colors[1][0], .max = 1.0),
    OPT_FLOAT("tile_color_lo_g", "Dark tile G", params.tile_colors[1][1], .max = 1.0),
    OPT_FLOAT("tile_color_lo_b", "Dark tile B", params.tile_colors[1][2], .max = 1.0),
    OPT_INT("tile_size", "Tile size", params.tile_size, .min = 2, .max = 256),

    // Performance / quality trade-offs and debugging options
    OPT_BOOL("skip_anti_aliasing", "Skip anti-aliasing", params.skip_anti_aliasing),
    OPT_FLOAT("polar_cutoff", "Polar LUT cutoff", params.polar_cutoff, .max = 1.0),
    OPT_BOOL("preserve_mixing_cache", "Preserve mixing cache", params.preserve_mixing_cache),
    OPT_BOOL("skip_caching_single_frame", "Skip caching single frame", params.skip_caching_single_frame),
    OPT_BOOL("disable_linear_scaling", "Disable linear scaling", params.disable_linear_scaling),
    OPT_BOOL("disable_builtin_scalers", "Disable built-in scalers", params.disable_builtin_scalers),
    OPT_BOOL("correct_subpixel_offset", "Correct subpixel offsets", params.correct_subpixel_offsets),
    OPT_BOOL("ignore_icc_profiles", "Ignore ICC profiles", params.ignore_icc_profiles),
    OPT_BOOL("force_dither", "Force-enable dithering", params.force_dither),
    OPT_BOOL("disable_dither_gamma_correction", "Disable gamma-correct dithering", params.disable_dither_gamma_correction),
    OPT_BOOL("disable_fbos", "Disable FBOs", params.disable_fbos),
    OPT_BOOL("force_low_bit_depth_fbos", "Force 8-bit FBOs", params.force_low_bit_depth_fbos),
    OPT_BOOL("dynamic_constants", "Dynamic constants", params.dynamic_constants),
    {0},
};

const int pl_option_count = PL_ARRAY_SIZE(pl_option_list) - 1;

pl_opt pl_find_option(const char *key)
{
    for (int i = 0; i < pl_option_count; i++) {
        if (!strcmp(key, pl_option_list[i].key))
            return &pl_option_list[i];
    }

    return NULL;
}
