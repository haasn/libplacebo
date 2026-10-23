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

#include "common.h"
#include "log.h"
#include "json.h"
#include "options.h"

#include <libplacebo/compositor.h>

struct priv {
    pl_options opts; // compositor-level options
    PL_ARRAY(pl_options) input_opts;
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

void pl_scene_update(pl_scene scene, const double pts)
{
    // TODO
}

// JSON scene graph parsing

struct parse_ctx {
    pl_log log;
    pl_str path; // for error messages
    void *tmp;
};

static bool parse_value(struct parse_ctx *ctx, void *base, pl_opt opt,
                        void *field, const struct json_value_s *val)
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
    case json_type_true:
        str = pl_str0("yes");
        break;
    case json_type_false:
        str = pl_str0("no");
        break;
    case json_type_array:
    case json_type_object:
    case json_type_null:
        PL_ERR(ctx, "Invalid value type for option '%.*s'. Expected string, "
               "number or boolean.", PL_STR_FMT(ctx->path));
        return false;
    default:
        pl_unreachable();
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
            ok &= parse_struct(ctx, opts, pl_option_list, opts, e->value);
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
            ok &= parse_struct(ctx, p->opts, pl_option_list, p->opts, e->value);
        } else if (pl_str_equals0(key, "inputs")) {
            ok &= parse_inputs(ctx, scene, e->value);
        } else {
            PL_ERR(ctx, "Unrecognized key '%.*s'", PL_STR_FMT(ctx->path));
            ok = false;
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
        .log = log,
        .tmp = pl_tmp(NULL),
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
