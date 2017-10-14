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
 * License along with libplacebo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <math.h>
#include <stdio.h>
#include "bstr/bstr.h"

#include "common.h"
#include "context.h"

// Represents a blank placeholder for the purposes of namespace substitution.
// This is picked to be a literal string that is impossible to ever occur in
// valid code.
#define PLACEHOLDER "\1\2\3"

typedef const char * ident_t;

struct pl_shader {
    // Read-only fields
    struct pl_context *ctx;
    const struct ra *ra;

    // Internal state
    bool mutable;
    struct pl_shader_res res; // for accumulating vertex_attribs etc.
    struct bstr buffer_head;
    struct bstr buffer_body;
    bool flexible_work_groups;
    int fresh;
    int namespace;
    void *tmp;

    // For vertex attributes, since we need to keep track of their location
    int current_va_location;
    size_t current_va_offset;

    // For bindings, since we need to keep the namespaces unique
    int *current_binding;
};

struct pl_shader *pl_shader_alloc(struct pl_context *ctx,
                                  const struct ra *ra)
{
    struct pl_shader *sh = talloc_ptrtype(ctx, sh);
    *sh = (struct pl_shader) {
        .ctx = ctx,
        .ra = ra,
        .mutable = true,
        .tmp = talloc_new(sh),
    };

    if (ra) {
        int num_namespaces = ra_desc_namespace(ra, 0);
        sh->current_binding = talloc_zero_array(sh, int, num_namespaces);
    }

    return sh;
}

void pl_shader_free(struct pl_shader **sh)
{
    TA_FREEP(sh);
}

void pl_shader_reset(struct pl_shader *sh)
{
    struct pl_shader new = {
        .ctx = sh->ctx,
        .ra  = sh->ra,
        .tmp = talloc_new(sh),
        .mutable = true,

        // Preserve array allocations
        .buffer_head = { sh->buffer_head.start },
        .buffer_body = { sh->buffer_body.start },
        .current_binding = sh->current_binding,
        .res = {
            .variables      = sh->res.variables,
            .descriptors    = sh->res.descriptors,
            .vertex_attribs = sh->res.vertex_attribs,
        },
    };

    // Clear the bindings array
    if (new.current_binding) {
        assert(sh->ra);
        int num_namespaces = ra_desc_namespace(sh->ra, 0);
        for (int i = 0; i < num_namespaces; i++)
            new.current_binding[i] = 0;
    }

    talloc_free(sh->tmp);
    *sh = new;
}

bool pl_shader_is_compute(const struct pl_shader *sh)
{
    bool ret = true;
    for (int i = 0; i < PL_ARRAY_SIZE(sh->res.compute_work_groups); i++)
        ret &= !!sh->res.compute_work_groups[i];
    return ret;
}

// Helpers for adding new variables/descriptors/etc. with fresh, unique
// identifier names. These will never conflcit with other identifiers, even
// if the shaders are merged together.
static ident_t fresh(struct pl_shader *sh, const char *name)
{
    return talloc_asprintf(sh->tmp, "_" PLACEHOLDER "_%s_%d",
                           PL_DEF(name, "var"), sh->fresh++);
}

// Add a new shader var and return its identifier
static ident_t var(struct pl_shader *sh, struct pl_shader_var sv)
{
    sv.var.name = fresh(sh, sv.var.name);
    sv.data = talloc_memdup(sh->tmp, sv.data, ra_var_host_layout(0, sv.var).size);
    TARRAY_APPEND(sh, sh->res.variables, sh->res.num_variables, sv);
    return sv.var.name;
}

// Add a new shader desc and return its identifier. This function takes care of
// setting the binding to a fresh bind point according to the namespace
// requirements, so the caller may leave it blank.
static ident_t desc(struct pl_shader *sh, struct pl_shader_desc sd)
{
    assert(sh->ra);
    int namespace = ra_desc_namespace(sh->ra, sd.desc.type);

    sd.desc.name = fresh(sh, sd.desc.name);
    sd.desc.binding = sh->current_binding[namespace]++;

    TARRAY_APPEND(sh, sh->res.descriptors, sh->res.num_descriptors, sd);
    return sd.desc.name;
}

// Add a new vec2 vertex attribute from a pl_rect2df, or returns NULL on failure.
static ident_t attr_vec2(struct pl_shader *sh, const char *name,
                         const struct pl_rect2df *rc)
{
    if (!sh->ra) {
        PL_ERR(sh, "Failed adding vertex attr '%s': No RA available!", name);
        return NULL;
    }

    const struct ra_fmt *fmt = ra_find_vertex_fmt(sh->ra, RA_FMT_FLOAT, 2);
    if (!fmt) {
        PL_ERR(sh, "Failed adding vertex attr '%s': no vertex fmt!", name);
        return NULL;
    }

    float vals[4][2] = {
        // Clockwise from top left
        { rc->x0, rc->y0 },
        { rc->x1, rc->y0 },
        { rc->x1, rc->y1 },
        { rc->x0, rc->y1 },
    };

    float *data = talloc_memdup(sh->tmp, &vals[0][0], sizeof(vals));
    struct pl_shader_va va = {
        .attr = {
            .name     = fresh(sh, name),
            .fmt      = ra_find_vertex_fmt(sh->ra, RA_FMT_FLOAT, 2),
            .offset   = sh->current_va_offset,
            .location = sh->current_va_location,
        },
        .data = { &data[0], &data[2], &data[4], &data[6] },
    };

    TARRAY_APPEND(sh, sh->res.vertex_attribs, sh->res.num_vertex_attribs, va);
    sh->current_va_offset += sizeof(float[2]);
    sh->current_va_location += 1; // vec2 always consumes one location
    return va.attr.name;
}

// Bind a texture under a given transformation and make its attributes
// available as well. If an output pointer for one of the attributes is left
// as NULL, that attribute will not be added. Returns NULL on failure.
static ident_t bind(struct pl_shader *sh, const struct ra_tex *tex,
                    const char *name, const struct pl_transform2x2 *tf,
                    ident_t *out_pos, ident_t *out_size, ident_t *out_pt)
{
    if (!sh->ra) {
        PL_ERR(sh, "Failed binding texture '%s': No RA available!", name);
        return NULL;
    }

    assert(ra_tex_params_dimension(tex->params) == 2);
    ident_t itex = desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name = name,
            .type = RA_DESC_SAMPLED_TEX,
        },
        .binding = tex,
    });

    if (out_pos) {
        float xy0[2] = {0};
        float xy1[2] = {tex->params.w, tex->params.h};
        pl_transform2x2_apply(tf, xy0);
        pl_transform2x2_apply(tf, xy1);
        *out_pos = attr_vec2(sh, "pos", &(struct pl_rect2df) {
            .x0 = xy0[0], .y0 = xy0[1],
            .x1 = xy1[0], .y1 = xy1[1],
        });
    }

    if (out_size) {
        *out_size = var(sh, (struct pl_shader_var) {
            .var  = ra_var_vec2("size"),
            .data = &(float[2]) {tex->params.w, tex->params.h},
        });
    }

    if (out_pt) {
        *out_pt = var(sh, (struct pl_shader_var) {
            .var  = ra_var_vec2("pt"),
            .data = &(float[2]) {1.0 / tex->params.w, 1.0 / tex->params.h},
        });
    }

    return itex;
}

PRINTF_ATTRIBUTE(3, 4)
static void pl_shader_append(struct pl_shader *sh, struct bstr *buf,
                             const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    bstr_xappend_vasprintf(sh, buf, fmt, ap);
    va_end(ap);
}

#define GLSLH(...) pl_shader_append(sh, &sh->buffer_head, __VA_ARGS__)
#define GLSL(...)  pl_shader_append(sh, &sh->buffer_body, __VA_ARGS__)

// Performs the free variable rename. `buf` must point to a buffer with at
// least strlen(PLACEHOLDER) valid replacement characters.
static void rename_str(char *str, const char *buf)
{
    if (!str)
        return;

    while ((str = strstr(str, PLACEHOLDER))) {
        for (int i = 0; i < sizeof(PLACEHOLDER) - 1; i++)
            str[i] = buf[i];
    }
}

// Replace all of the free variables in the body and input list by literally
// string replacing it with an encoded representation of the given namespace
static void rename_vars(struct pl_shader *sh, int namespace)
{
    char buf[sizeof(PLACEHOLDER)] = {0};
    int num = snprintf(buf, sizeof(buf), "%d", namespace);

    // Clear out the remainder of `buf` with a safe character
    for (int i = num; i < sizeof(buf) - 1; i++)
        buf[i] = 'z';

    // This is safe, because we never shrink or splice the buffers; and they're
    // always null-terminated (for the same reason we can directly return them)
    rename_str(sh->buffer_head.start, buf);
    rename_str(sh->buffer_body.start, buf);

    // These are all safe to directly mutate, because we've allocated all
    // identifiers via `fresh`.
    rename_str((char *) sh->res.name, buf);
    for (int i = 0; i < sh->res.num_vertex_attribs; i++)
        rename_str((char *) sh->res.vertex_attribs[i].attr.name, buf);
    for (int i = 0; i < sh->res.num_variables; i++)
        rename_str((char *) sh->res.variables[i].var.name, buf);
    for (int i = 0; i < sh->res.num_descriptors; i++)
        rename_str((char *) sh->res.descriptors[i].desc.name, buf);
}

const struct pl_shader_res *pl_shader_finalize(struct pl_shader *sh)
{
    if (!sh->mutable) {
        PL_WARN(sh, "Attempted to finalize a shader twice?");
        return &sh->res;
    }

    static const char *outsigs[] = {
        [PL_SHADER_SIG_NONE]  = "void",
        [PL_SHADER_SIG_COLOR] = "vec4",
    };

    static const char *insigs[] = {
        [PL_SHADER_SIG_NONE]  = "",
        [PL_SHADER_SIG_COLOR] = "vec4 color",
    };

    ident_t name = sh->res.name = fresh(sh, "main");
    GLSLH("%s %s(%s) {\n", outsigs[sh->res.output], name, insigs[sh->res.input]);

    if (sh->buffer_body.len) {
        GLSLH("%s", sh->buffer_body.start);
        sh->buffer_body.len = 0;
        sh->buffer_body.start[0] = '\0'; // sanity, and for rename_vars
    }

    switch (sh->res.output) {
    case PL_SHADER_SIG_NONE: break;
    case PL_SHADER_SIG_COLOR:
        GLSLH("return color;\n");
        break;
    }

    GLSLH("}\n");
    rename_vars(sh, sh->namespace);

    // Update the result pointer
    sh->res.glsl = sh->buffer_head.start;
    sh->mutable = false;

    return &sh->res;
}

// Requires that the share is mutable and has an output signature compatible
// with the given input signature. Errors and returns false otherwise.
static bool require_input(struct pl_shader *sh, enum pl_shader_sig insig)
{
    if (!sh->mutable) {
        PL_ERR(sh, "Attempted to modify an immutable shader!");
        return false;
    }

    static const char *names[] = {
        [PL_SHADER_SIG_NONE]  = "PL_SHADER_SIG_NONE",
        [PL_SHADER_SIG_COLOR] = "PL_SHADER_SIG_COLOR",
    };

    // If we require an input, but there is none available - just get it from
    // the user by turning it into an explicit input signature.
    if (!sh->res.output && insig) {
        assert(!sh->res.input);
        sh->res.input = insig;
    } else if (sh->res.output != insig) {
        PL_ERR(sh, "Illegal sequence of shader operations! Current output "
               "signature is '%s', but called operation expects '%s'!",
               names[sh->res.output], names[insig]);
        return false;
    }

    // All of our shaders end up returning a vec4 color
    sh->res.output = PL_SHADER_SIG_COLOR;
    return true;
}

// Colorspace operations

void pl_shader_decode_color(struct pl_shader *sh, struct pl_color_repr *repr,
                            struct pl_color_adjustment params, int texture_bits)
{
    if (!require_input(sh, PL_SHADER_SIG_COLOR))
        return;

    GLSL("// pl_shader_decode_color\n");

    // For the non-linear color systems we need some special input handling
    // to make sure we don't accidentally screw everything up because of the
    // alpha multiplication, which only commutes with linear operations.
    bool is_nonlinear = !pl_color_system_is_linear(repr->sys);
    if (is_nonlinear && repr->alpha == PL_ALPHA_PREMULTIPLIED) {
        GLSL("color.rgb /= vec3(color.a);\n");
        repr->alpha = PL_ALPHA_INDEPENDENT;
    }

    // XYZ needs special handling due to the input gamma logic
    if (repr->sys == PL_COLOR_SYSTEM_XYZ) {
        float scale = pl_color_repr_normalize(repr);
        GLSL("color.rgb = pow(%f * color.rgb, vec3(2.6));\n", scale);
    }

    enum pl_color_system orig_sys = repr->sys;
    struct pl_transform3x3 tr = pl_color_repr_decode(repr, params);

    ident_t cmat = var(sh, (struct pl_shader_var) {
        .var  = ra_var_mat3("cmat"),
        .data = PL_TRANSPOSE_3X3(tr.mat.m),
    });

    ident_t cmat_c = var(sh, (struct pl_shader_var) {
        .var  = ra_var_vec3("cmat_m"),
        .data = tr.c,
    });

    GLSL("color.rgb = %s * color.rgb + %s;\n", cmat, cmat_c);

    if (orig_sys == PL_COLOR_SYSTEM_BT_2020_C) {
        // Conversion for C'rcY'cC'bc via the BT.2020 CL system:
        // C'bc = (B'-Y'c) / 1.9404  | C'bc <= 0
        //      = (B'-Y'c) / 1.5816  | C'bc >  0
        //
        // C'rc = (R'-Y'c) / 1.7184  | C'rc <= 0
        //      = (R'-Y'c) / 0.9936  | C'rc >  0
        //
        // as per the BT.2020 specification, table 4. This is a non-linear
        // transformation because (constant) luminance receives non-equal
        // contributions from the three different channels.
        GLSL("// constant luminance conversion                            \n"
             "color.br = color.br * mix(vec2(1.5816, 0.9936),             \n"
             "                          vec2(1.9404, 1.7184),             \n"
             "                          lessThanEqual(color.br, vec2(0))) \n"
             "           + color.gg;                                      \n"
        // Expand channels to camera-linear light. This shader currently just
        // assumes everything uses the BT.2020 12-bit gamma function, since the
        // difference between 10 and 12-bit is negligible for anything other
        // than 12-bit content.
             "color.rgb = mix(color.rgb * vec3(1.0/4.5),                       \n"
             "                pow((color.rgb + vec3(0.0993))*vec3(1.0/1.0993), \n"
             "                    vec3(1.0/0.45)),                             \n"
             "                lessThanEqual(vec3(0.08145), color.rgb));        \n"
        // Calculate the green channel from the expanded RYcB
        // The BT.2020 specification says Yc = 0.2627*R + 0.6780*G + 0.0593*B
             "color.g = (color.g - 0.2627*color.r - 0.0593*color.b)*1.0/0.6780; \n"
        // Recompress to receive the R'G'B' result, same as other systems
             "color.rgb = mix(color.rgb * vec3(4.5),                    \n"
             "                vec3(1.0993) * pow(color.rgb, vec3(0.45)) \n"
             "                   - vec3(0.0993),                        \n"
             "                lessThanEqual(vec3(0.0181), color.rgb));  \n");
    }

    if (repr->alpha == PL_ALPHA_INDEPENDENT) {
        GLSL("color.rgb *= vec3(color.a)\n");
        repr->alpha = PL_ALPHA_PREMULTIPLIED;
    }
}

// Common constants for SMPTE ST.2084 (PQ)
static const float PQ_M1 = 2610./4096 * 1./4,
                   PQ_M2 = 2523./4096 * 128,
                   PQ_C1 = 3424./4096,
                   PQ_C2 = 2413./4096 * 32,
                   PQ_C3 = 2392./4096 * 32;

// Common constants for ARIB STD-B67 (HLG)
static const float HLG_A = 0.17883277,
                   HLG_B = 0.28466892,
                   HLG_C = 0.55991073;

// Common constants for Panasonic V-Log
static const float VLOG_B = 0.00873,
                   VLOG_C = 0.241514,
                   VLOG_D = 0.598206;

// Common constants for Sony S-Log
static const float SLOG_A = 0.432699,
                   SLOG_B = 0.037584,
                   SLOG_C = 0.616596 + 0.03,
                   SLOG_P = 3.538813,
                   SLOG_Q = 0.030001,
                   SLOG_K2 = 155.0 / 219.0;

void pl_shader_linearize(struct pl_shader *sh, enum pl_color_transfer trc)
{
    if (!require_input(sh, PL_SHADER_SIG_COLOR))
        return;

    if (trc == PL_COLOR_TRC_LINEAR)
        return;

    // Note that this clamp may technically violate the definition of
    // ITU-R BT.2100, which allows for sub-blacks and super-whites to be
    // displayed on the display where such would be possible. That said, the
    // problem is that not all gamma curves are well-defined on the values
    // outside this range, so we ignore it and just clip anyway for sanity.
    GLSL("// pl_shader_linearize                  \n"
         "color.rgb = clamp(color.rgb, 0.0, 1.0); \n");

    switch (trc) {
    case PL_COLOR_TRC_SRGB:
        GLSL("color.rgb = mix(color.rgb * vec3(1.0/12.92),               \n"
             "                pow((color.rgb + vec3(0.055))/vec3(1.055), \n"
             "                    vec3(2.4)),                            \n"
             "                lessThan(vec3(0.04045), color.rgb));       \n");
        break;
    case PL_COLOR_TRC_BT_1886:
        GLSL("color.rgb = pow(color.rgb, vec3(2.4));\n");
        break;
    case PL_COLOR_TRC_GAMMA18:
        GLSL("color.rgb = pow(color.rgb, vec3(1.8));\n");
        break;
    case PL_COLOR_TRC_UNKNOWN:
    case PL_COLOR_TRC_GAMMA22:
        GLSL("color.rgb = pow(color.rgb, vec3(2.2));\n");
        break;
    case PL_COLOR_TRC_GAMMA28:
        GLSL("color.rgb = pow(color.rgb, vec3(2.8));\n");
        break;
    case PL_COLOR_TRC_PRO_PHOTO:
        GLSL("color.rgb = mix(color.rgb * vec3(1.0/16.0),          \n"
             "                pow(color.rgb, vec3(1.8)),           \n"
             "                lessThan(vec3(0.03125), color.rgb)); \n");
        break;
    case PL_COLOR_TRC_PQ:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/%f));         \n"
             "color.rgb = max(color.rgb - vec3(%f), vec3(0.0))  \n"
             "             / (vec3(%f) - vec3(%f) * color.rgb); \n"
             "color.rgb = pow(color.rgb, vec3(1.0/%f));         \n"
             // PQ's output range is 0-10000, but we need it to be relative to
             // to PL_COLOR_REF_WHITE instead, so rescale
             "color.rgb *= vec3(%f);\n",
             PQ_M2, PQ_C1, PQ_C2, PQ_C3, PQ_M1, 10000 / PL_COLOR_REF_WHITE);
        break;
    case PL_COLOR_TRC_HLG:
        GLSL("color.rgb = mix(vec3(4.0) * color.rgb * color.rgb,         \n"
             "                exp((color.rgb - vec3(%f)) * vec3(1.0/%f)) \n"
             "                    + vec3(%f),                            \n"
             "                lessThan(vec3(0.5), color.rgb));           \n",
             HLG_C, HLG_A, HLG_B);
        break;
    case PL_COLOR_TRC_V_LOG:
        GLSL("color.rgb = mix((color.rgb - vec3(0.125)) * vec3(1.0/5.6), \n"
             "    pow(vec3(10.0), (color.rgb - vec3(%f)) * vec3(1.0/%f)) \n"
             "              - vec3(%f),                                  \n"
             "    lessThanEqual(vec3(0.181), color.rgb));                \n",
             VLOG_D, VLOG_C, VLOG_B);
        break;
    case PL_COLOR_TRC_S_LOG1:
        GLSL("color.rgb = pow(vec3(10.0), (color.rgb - vec3(%f)) * vec3(1.0/%f)) \n"
             "            - vec3(%f);                                            \n",
             SLOG_C, SLOG_A, SLOG_B);
        break;
    case PL_COLOR_TRC_S_LOG2:
        GLSL("color.rgb = mix((color.rgb - vec3(%f)) * vec3(1.0/%f),      \n"
             "    (pow(vec3(10.0), (color.rgb - vec3(%f)) * vec3(1.0/%f)) \n"
             "              - vec3(%f)) * vec3(1.0/%f),                   \n"
             "    lessThanEqual(vec3(%f), color.rgb));                    \n",
             SLOG_Q, SLOG_P, SLOG_C, SLOG_A, SLOG_B, SLOG_K2, SLOG_Q);
        break;
    default:
        abort();
    }

    // Rescale to prevent clipping on non-float textures
    GLSL("color.rgb *= vec3(1.0/%f);\n", pl_color_transfer_nominal_peak(trc));
}

void pl_shader_delinearize(struct pl_shader *sh, enum pl_color_transfer trc)
{
    if (!require_input(sh, PL_SHADER_SIG_COLOR))
        return;

    if (trc == PL_COLOR_TRC_LINEAR)
        return;

    GLSL("// pl_shader_delinearize\n"
         "color.rgb = clamp(color.rgb, 0.0, 1.0);\n"
         "color.rgb *= vec3(%f);\n", pl_color_transfer_nominal_peak(trc));

    switch (trc) {
    case PL_COLOR_TRC_SRGB:
        GLSL("color.rgb = mix(color.rgb * vec3(12.92),                    \n"
             "                vec3(1.055) * pow(color.rgb, vec3(1.0/2.4)) \n"
             "                    - vec3(0.055),                          \n"
             "                lessThanEqual(vec3(0.0031308), color.rgb)); \n");
        break;
    case PL_COLOR_TRC_BT_1886:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/2.4));\n");
        break;
    case PL_COLOR_TRC_GAMMA18:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/1.8));\n");
        break;
    case PL_COLOR_TRC_UNKNOWN:
    case PL_COLOR_TRC_GAMMA22:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/2.2));\n");
        break;
    case PL_COLOR_TRC_GAMMA28:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/2.8));\n");
        break;
    case PL_COLOR_TRC_PRO_PHOTO:
        GLSL("color.rgb = mix(color.rgb * vec3(16.0),                    \n"
             "                pow(color.rgb, vec3(1.0/1.8)),             \n"
             "                lessThanEqual(vec3(0.001953), color.rgb)); \n");
        break;
    case PL_COLOR_TRC_PQ:
        GLSL("color.rgb *= vec3(1.0/%f);                         \n"
             "color.rgb = pow(color.rgb, vec3(%f));              \n"
             "color.rgb = (vec3(%f) + vec3(%f) * color.rgb)      \n"
             "             / (vec3(1.0) + vec3(%f) * color.rgb); \n"
             "color.rgb = pow(color.rgb, vec3(%f));              \n",
             10000 / PL_COLOR_REF_WHITE, PQ_M1, PQ_C1, PQ_C2, PQ_C3, PQ_M2);
        break;
    case PL_COLOR_TRC_HLG:
        GLSL("color.rgb = mix(vec3(0.5) * sqrt(color.rgb),                     \n"
             "                vec3(%f) * log(color.rgb - vec3(%f)) + vec3(%f), \n"
             "                lessThan(vec3(1.0), color.rgb));                 \n",
             HLG_A, HLG_B, HLG_C);
        break;
    case PL_COLOR_TRC_V_LOG:
        GLSL("color.rgb = mix(vec3(5.6) * color.rgb + vec3(0.125),   \n"
             "                vec3(%f) * log(color.rgb + vec3(%f))   \n"
             "                    + vec3(%f),                        \n"
             "                lessThanEqual(vec3(0.01), color.rgb)); \n",
             VLOG_C / M_LN10, VLOG_B, VLOG_D);
        break;
    case PL_COLOR_TRC_S_LOG1:
        GLSL("color.rgb = vec3(%f) * log(color.rgb + vec3(%f)) + vec3(%f);\n",
             SLOG_A / M_LN10, SLOG_B, SLOG_C);
        break;
    case PL_COLOR_TRC_S_LOG2:
        GLSL("color.rgb = mix(vec3(%f) * color.rgb + vec3(%f),                \n"
             "                vec3(%f) * log(vec3(%f) * color.rgb + vec3(%f)) \n"
             "                    + vec3(%f),                                 \n"
             "                lessThanEqual(vec3(0.0), color.rgb));           \n",
             SLOG_P, SLOG_Q, SLOG_A / M_LN10, SLOG_K2, SLOG_B, SLOG_C);
        break;
    default:
        abort();
    }
}

// Applies the OOTF / inverse OOTF. `peak` corresponds to the nominal peak
// (needed to scale the functions correctly)
static void pl_shader_ootf(struct pl_shader *sh, enum pl_color_light light,
                           float peak, ident_t luma)
{
    if (!require_input(sh, PL_SHADER_SIG_COLOR))
        return;

    if (!light || light == PL_COLOR_LIGHT_DISPLAY)
        return;

    GLSL("// pl_shader_ootf      \n"
         "color.rgb *= vec3(%f); \n", peak);

    switch (light)
    {
    case PL_COLOR_LIGHT_SCENE_HLG:
        // HLG OOTF from BT.2100, assuming a reference display with a
        // peak of 1000 cd/m² -> gamma = 1.2
        GLSL("color.rgb *= vec3(%f * pow(dot(%s, color.rgb), 0.2));\n",
             (1000 / PL_COLOR_REF_WHITE) / pow(12, 1.2), luma);
        break;
    case PL_COLOR_LIGHT_SCENE_709_1886:
        // This OOTF is defined by encoding the result as 709 and then decoding
        // it as 1886; although this is called 709_1886 we actually use the
        // more precise (by one decimal) values from BT.2020 instead
        GLSL("color.rgb = mix(color.rgb * vec3(4.5),                    \n"
             "                vec3(1.0993) * pow(color.rgb, vec3(0.45)) \n"
             "                             - vec3(0.0993),              \n"
             "                lessThan(vec3(0.0181), color.rgb));       \n"
             "color.rgb = pow(color.rgb, vec3(2.4));                    \n");
        break;
    case PL_COLOR_LIGHT_SCENE_1_2:
        GLSL("color.rgb = pow(color.rgb, vec3(1.2));\n");
        break;
    default:
        abort();
    }

    GLSL("color.rgb *= vec3(1.0/%f);\n", peak);
}

static void pl_shader_inverse_ootf(struct pl_shader *sh, enum pl_color_light light,
                                   float peak, ident_t luma)
{
    if (!light || light == PL_COLOR_LIGHT_DISPLAY)
        return;

    GLSL("// pl_shader_inverse_ootf\n"
         "color.rgb *= vec3(%f);\n", peak);

    switch (light)
    {
    case PL_COLOR_LIGHT_SCENE_HLG:
        GLSL("color.rgb *= vec3(1.0/%f);                                \n"
             "color.rgb /= vec3(max(1e-6, pow(dot(%s, color.rgb),       \n"
             "                                0.2/1.2)));               \n",
             (1000 / PL_COLOR_REF_WHITE) / pow(12, 1.2), luma);
        break;
    case PL_COLOR_LIGHT_SCENE_709_1886:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/2.4));                         \n"
             "color.rgb = mix(color.rgb * vec3(1.0/4.5),                         \n"
             "                pow((color.rgb + vec3(0.0993)) * vec3(1.0/1.0993), \n"
             "                    vec3(1/0.45)),                                 \n"
             "                lessThan(vec3(0.08145), color.rgb));               \n");
        break;
    case PL_COLOR_LIGHT_SCENE_1_2:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/1.2));\n");
        break;
    default:
        abort();
    }

    GLSL("color.rgb *= vec3(1.0/%f);\n", peak);
}

const struct pl_color_map_params pl_color_map_default_params = {
    .intent                  = PL_INTENT_RELATIVE_COLORIMETRIC,
    .tone_mapping_algo       = PL_TONE_MAPPING_MOBIUS,
    .tone_mapping_desaturate = 2.0,
};

static void pl_shader_tone_map(struct pl_shader *sh, float ref_peak, ident_t luma,
                               const struct pl_color_map_params *params)
{
    GLSL("// pl_shader_tone_map\n");

    // Desaturate the color using a coefficient dependent on the luminance
    if (params->tone_mapping_desaturate > 0) {
        GLSL("float luma = dot(%s, color.rgb);                           \n"
             "float overbright = max(luma - %f, 1e-6) / max(luma, 1e-6); \n"
             "color.rgb = mix(color.rgb, vec3(luma), overbright);        \n",
             luma, params->tone_mapping_desaturate);
    }

    // To prevent discoloration due to out-of-bounds clipping, we need to make
    // sure to reduce the value range as far as necessary to keep the entire
    // signal in range, so tone map based on the brightest component.
    GLSL("float sig = max(max(color.r, color.g), color.b); \n"
         "float sig_orig = sig;                            \n");

    GLSL("const float sig_peak = %f;\n", ref_peak);

    float param = params->tone_mapping_param;
    switch (params->tone_mapping_algo) {
    case PL_TONE_MAPPING_CLIP:
        GLSL("sig = %f * sig;\n", PL_DEF(param, 1.0));
        break;

    case PL_TONE_MAPPING_MOBIUS:
        GLSL("const float j = %f;                                           \n"
             // solve for M(j) = j; M(sig_peak) = 1.0; M'(j) = 1.0
             // where M(x) = scale * (x+a)/(x+b)
             "float a = -j*j * (sig_peak - 1.0) / (j*j - 2.0*j + sig_peak); \n"
             "float b = (j*j - 2.0*j*sig_peak + sig_peak) /                 \n"
             "          max(1e-6, sig_peak - 1.0);                          \n"
             "float scale = (b*b + 2.0*b*j + j*j) / (b-a);                  \n"
             "sig = sig > j ? (scale * (sig + a) / (sig + b)) : sig;         \n",
             PL_DEF(param, 0.3));
        break;

    case PL_TONE_MAPPING_REINHARD: {
        float contrast = PL_DEF(param, 0.5),
              offset = (1.0 - contrast) / contrast;
        GLSL("sig = sig / (sig + %f);                   \n"
             "float scale = (sig_peak + %f) / sig_peak; \n"
             "sig *= scale;                             \n",
             offset, offset);
        break;
    }

    case PL_TONE_MAPPING_HABLE: {
        float A = 0.15, B = 0.50, C = 0.10, D = 0.20, E = 0.02, F = 0.30;
        ident_t hable = fresh(sh, "hable");
        GLSLH("float %s(float x) {                                        \n"
              "return ((x * (%f*x + %f)+%f)/(x * (%f*x + %f) + %f)) - %f; \n"
              "}                                                          \n",
              hable, A, C*B, D*E, A, B, D*F, E/F);
        GLSL("sig = %s(sig) / %s(sig_peak);\n", hable, hable);
        break;
    }

    case PL_TONE_MAPPING_GAMMA: {
        GLSL("const float cutoff = 0.05, gamma = 1.0/%f;                     \n"
             "float scale = pow(cutoff / sig_peak, gamma) / cutoff;          \n"
             "sig = sig > cutoff ? pow(sig / sig_peak, gamma) : scale * sig; \n",
             PL_DEF(param, 1.8));
        break;
    }

    case PL_TONE_MAPPING_LINEAR: {
        GLSL("sig = %f / sig_peak * sig;\n", PL_DEF(param, 1.0));
        break;
    }

    default:
        abort();
    }

    // Apply the computed scale factor to the color, linearly to prevent
    // discoloration
    GLSL("color.rgb *= sig / sig_orig;\n");
}

void pl_shader_color_map(struct pl_shader *sh,
                         const struct pl_color_map_params *params,
                         struct pl_color_space src, struct pl_color_space dst,
                         bool prelinearized)
{
    if (!require_input(sh, PL_SHADER_SIG_COLOR))
        return;

    GLSL("// pl_shader_color_map\n");
    GLSL("{\n");

    // To be as conservative as possible, color mapping is disabled by default
    // except for special cases which are considered to be "sufficiently
    // different" from the source space. For primaries, this means anything
    // wide gamut; and for transfers, this means anything radically different
    // from the typical SDR curves.
    if (!dst.primaries) {
        dst.primaries = src.primaries;
        if (pl_color_primaries_is_wide_gamut(dst.primaries))
            dst.primaries = PL_COLOR_PRIM_BT_709;
    }

    if (!dst.transfer) {
        dst.transfer = src.transfer;
        if (pl_color_transfer_is_hdr(dst.transfer) ||
                dst.transfer == PL_COLOR_TRC_LINEAR)
            dst.transfer = PL_COLOR_TRC_GAMMA22;
    }

    // If the source signal peak information is unknown, infer it from the
    // transfer function. (Note: The sig peak of the dst space is irrelevant)
    if (!src.sig_peak)
        src.sig_peak = pl_color_transfer_nominal_peak(src.transfer);

    // Compute the highest encodable level
    float src_range = pl_color_transfer_nominal_peak(src.transfer),
          dst_range = pl_color_transfer_nominal_peak(dst.transfer);
    float ref_peak = src.sig_peak / dst_range;

    // All operations from here on require linear light as a starting point,
    // so we linearize even if src.gamma == dst.gamma when one of the other
    // operations needs it
    bool need_linear = src.transfer != dst.transfer ||
                       src.primaries != dst.primaries ||
                       src_range != dst_range ||
                       src.sig_peak > dst_range ||
                       src.light != dst.light;
    bool is_linear = prelinearized;

    // Various operations need access to the src_luma and dst_luma respectively,
    // so just always make them available if we're doing anything at all
    ident_t src_luma = NULL, dst_luma = NULL;
    if (need_linear) {
        struct pl_matrix3x3 rgb2xyz;
        rgb2xyz = pl_get_rgb2xyz_matrix(pl_raw_primaries_get(src.primaries));
        src_luma = var(sh, (struct pl_shader_var) {
            .var  = ra_var_vec3("src_luma"),
            .data = rgb2xyz.m[1], // RGB->Y vector
        });
        rgb2xyz = pl_get_rgb2xyz_matrix(pl_raw_primaries_get(dst.primaries));
        dst_luma = var(sh, (struct pl_shader_var) {
            .var  = ra_var_vec3("dst_luma"),
            .data = rgb2xyz.m[1], // RGB->Y vector
        });
    }

    if (need_linear && !is_linear) {
        pl_shader_linearize(sh, src.transfer);
        is_linear = true;
    }

    if (src.light != dst.light)
        pl_shader_ootf(sh, src.light, src_range, src_luma);

    // Rescale the signal to compensate for differences in the encoding range
    // and reference white level. This is necessary because of the 0-1 value
    // normalization for HDR signals.
    if (src_range != dst_range) {
        GLSL("// rescale value range;\n");
        GLSL("color.rgb *= vec3(%f);\n", src_range / dst_range);
    }

    // Adapt to the right colorspace (primaries) if necessary
    if (src.primaries != dst.primaries) {
        const struct pl_raw_primaries *csp_src, *csp_dst;
        csp_src = pl_raw_primaries_get(src.primaries),
        csp_dst = pl_raw_primaries_get(dst.primaries);
        struct pl_matrix3x3 cms_mat;
        cms_mat = pl_get_color_mapping_matrix(csp_src, csp_dst, params->intent);
        GLSL("color.rgb = %s * color.rgb;\n", var(sh, (struct pl_shader_var) {
            .var = ra_var_mat3("cms_matrix"),
            .data = PL_TRANSPOSE_3X3(cms_mat.m),
        }));
        // Since this can reduce the gamut, figure out by how much
        for (int c = 0; c < 3; c++)
            ref_peak = fmaxf(ref_peak, cms_mat.m[c][c]);
    }

    // Tone map to prevent clipping when the source signal peak exceeds the
    // encodable range or we've reduced the gamut
    if (ref_peak > 1)
        pl_shader_tone_map(sh, ref_peak, dst_luma, params);

    if (src.light != dst.light)
        pl_shader_inverse_ootf(sh, dst.light, dst_range, dst_luma);

    // Warn for remaining out-of-gamut colors is enabled
    if (params->gamut_warning) {
        GLSL("if (any(greaterThan(color.rgb, vec3(1.01))) ||\n"
             "    any(lessThan(color.rgb, vec3(-0.01))))\n"
             "    color.rgb = vec3(1.0) - color.rgb;) // invert\n");
    }

    if (is_linear)
        pl_shader_delinearize(sh, dst.transfer);

    GLSL("}\n");
}

const struct pl_deband_params pl_deband_default_params = {
    .iterations = 1,
    .threshold  = 4.0,
    .radius     = 16.0,
    .grain      = 6.0,
};

void pl_shader_deband(struct pl_shader *sh, const struct ra_tex *ra_tex,
                      const struct pl_deband_params *params)
{
    if (!require_input(sh, PL_SHADER_SIG_NONE))
        return;

    GLSL("vec4 color;\n");
    GLSL("// pl_shader_deband\n");
    GLSL("{\n");

    ident_t tex, pos, pt;
    tex = bind(sh, ra_tex, "deband", &pl_transform2x2_identity, &pos, NULL, &pt);
    if (!tex)
        return;

    // Initialize the PRNG. This is friendly for wide usage and returns in
    // a very pleasant-looking distribution across frames even if the difference
    // between input coordinates is very small. Shamelessly stolen from some
    // GLSL tricks forum post years from a decade ago.
    ident_t random = fresh(sh, "random"), permute = fresh(sh, "permute");
    GLSLH("float %s(float x) {                          \n"
          "    x = (34.0 * x + 1.0) * x;                \n"
          "    return x - floor(x * 1.0/289.0) * 289.0; \n" // mod 289
          "}                                            \n"
          "float %s(inout float state) {                \n"
          "    state = %s(state);                       \n"
          "    return fract(state * 1.0/41.0);          \n"
          "}\n", permute, random, permute);

    ident_t seed = var(sh, (struct pl_shader_var) {
        .var  = ra_var_float("seed"),
        .data = &params->seed,
    });

    GLSL("vec3 _m = vec3(%s, %s) + vec3(1.0);          \n"
         "float prng = %s(%s(%s(_m.x) + _m.y) + _m.z); \n"
         "vec4 avg, diff;                              \n"
         "color = texture(%s, %s);                     \n",
         pos, seed, permute, permute, permute, tex, pos);

    // Helper function: Compute a stochastic approximation of the avg color
    // around a pixel, given a specified radius
    ident_t average = fresh(sh, "average");
    GLSLH("vec4 %s(float range, inout float prng) {            \n"
          // Compute a random angle and distance
          "    float dist = %s(prng) * range;                  \n"
          "    float dir  = %s(prng) * %f;                     \n"
          "    vec2 o = dist * vec2(cos(dir), sin(dir));       \n"
          // Sample at quarter-turn intervals around the source pixel
          "    vec4 sum = vec4(0.0);                           \n"
          "    sum += texture(%s, %s + %s * vec2( o.x,  o.y)); \n"
          "    sum += texture(%s, %s + %s * vec2(-o.x,  o.y)); \n"
          "    sum += texture(%s, %s + %s * vec2(-o.x, -o.y)); \n"
          "    sum += texture(%s, %s + %s * vec2( o.x, -o.y)); \n"
          // Return the (normalized) average
          "    return 0.25 * sum;                              \n"
          "}\n", average, random, random, M_PI * 2,
          tex, pos, pt, tex, pos, pt,
          tex, pos, pt, tex, pos, pt);

    // For each iteration, compute the average at a given distance and
    // pick it instead of the color if the difference is below the threshold.
    for (int i = 1; i <= params->iterations; i++) {
        GLSL("avg = %s(%f, prng);                                   \n"
             "diff = abs(color - avg);                              \n"
             "color = mix(avg, color, greaterThan(diff, vec4(%f))); \n",
             average, i * params->radius, params->threshold / (1000 * i));
    }

    // Add some random noise to smooth out residual differences
    GLSL("vec3 noise = vec3(%s(prng), %s(prng), %s(prng)); \n"
         "color.rgb += %f * (noise - vec3(0.5));           \n",
         random, random, random, params->grain / 1000.0);

    GLSL("}\n");
}
