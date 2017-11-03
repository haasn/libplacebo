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

/*
 * Some of the filter code originally derives (via mpv) from Glumpy:
 * # Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
 * # Distributed under the (new) BSD License.
 * (https://github.com/glumpy/glumpy/blob/master/glumpy/library/build-spatial-filters.py)
 *
 * The math underlying each filter function was written from scratch, with
 * some algorithms coming from a number of different sources, including:
 * - https://en.wikipedia.org/wiki/Window_function
 * - https://en.wikipedia.org/wiki/Jinc
 * - http://vector-agg.cvs.sourceforge.net/viewvc/vector-agg/agg-2.5/include/agg_image_filters.h
 * - Vapoursynth plugin fmtconv (WTFPL Licensed), which is based on
 *   dither plugin for avisynth from the same author:
 *   https://github.com/vapoursynth/fmtconv/tree/master/src/fmtc
 * - Paul Heckbert's "zoom"
 * - XBMC: ConvolutionKernels.cpp etc.
 * - https://github.com/AviSynth/jinc-resize (only used to verify the math)
 */

#include <math.h>

#include "common.h"
#include "context.h"

bool pl_filter_function_eq(const struct pl_filter_function *a,
                           const struct pl_filter_function *b)
{
    if (!a || !b)
        return a == b;

    bool r = a->resizable == b->resizable &&
             a->weight    == b->weight &&
             a->radius    == b->radius;

    for (int i = 0; i < PL_FILTER_MAX_PARAMS; i++) {
        r &= a->tunable[i] == b->tunable[i];
        if (a->tunable[i])
             r &= a->params[i] == b->params[i];
    }

    return r;
}

bool pl_filter_config_eq(const struct pl_filter_config *a,
                         const struct pl_filter_config *b)
{
    if (!a || !b)
        return a == b;

    return pl_filter_function_eq(a->kernel, b->kernel) &&
           pl_filter_function_eq(a->window, b->window) &&
           a->clamp == b->clamp &&
           a->blur  == b->blur &&
           a->taper == b->taper &&
           a->polar == b->polar;
}

double pl_filter_sample(const struct pl_filter_config *c, double x)
{
    double radius = c->kernel->radius;

    // All filters are symmetric, and in particular only need to be defined
    // for [0, radius].
    x = fabs(x);

    // Apply the blur and taper coefficients as needed
    double kx = c->blur > 0.0 ? x / c->blur : x;
    kx = kx <= c->taper ? 0.0 : (kx - c->taper) / (1.0 - c->taper / radius);

    // Return early for values outside of the kernel radius, since the functions
    // are not necessarily valid outside of this interval. No such check is
    // needed for the window, because it's always stretched to fit.
    if (kx > radius)
        return 0.0;

    double k = c->kernel->weight(c->kernel, kx);

    // Apply the optional windowing function
    if (c->window)
        k *= c->window->weight(c->window, x / radius * c->window->radius);

    return k < 0 ? (1 - c->clamp) * k : k;
}

// Calculate a single filter row of a 1D filter, for a given phase value /
// subpixel offset `offset`. Writes exactly f->row_size values to *out.
static void compute_row(struct pl_filter *f, double offset, float *out)
{
    pl_assert(f->row_size > 0);
    double sum = 0;
    for (int i = 0; i < f->row_size; i++) {
        double x = offset - (i - f->row_size / 2.0 + 1);
        // Readjust the value range to account for a stretched kernel.
        x *= f->params.config.kernel->radius / f->radius;
        double weight = pl_filter_sample(&f->params.config, x);
        out[i] = weight;
        sum += weight;
    }
    // Normalize to preserve energy
    if (sum > 0.0) {
        for (int i = 0; i < f->row_size; i++)
            out[i] /= sum;
    }
}

static struct pl_filter_function *dupfilter(void *tactx,
                                            const struct pl_filter_function *f)
{
    return f ? talloc_memdup(tactx, (void *)f, sizeof(*f)) : NULL;
}

const struct pl_filter *pl_filter_generate(struct pl_context *ctx,
                                       const struct pl_filter_params *params)
{
    pl_assert(params);
    if (params->lut_entries <= 0 || !params->config.kernel) {
        pl_fatal(ctx, "Invalid params: missing lut_entries or config.kernel");
        return NULL;
    }

    struct pl_filter *f = talloc_zero(ctx, struct pl_filter);
    f->params = *params;
    f->params.config.kernel = dupfilter(f, params->config.kernel);
    f->params.config.window = dupfilter(f, params->config.window);

    // Compute the required filter radius
    float radius = f->params.config.kernel->radius;
    f->radius = radius;
    if (params->filter_scale > 1.0)
        f->radius *= params->filter_scale;

    float *weights;
    if (params->config.polar) {
        // Compute a 1D array indexed by radius
        weights = talloc_array(f, float, params->lut_entries);
        f->radius_cutoff = 0.0;
        for (int i = 0; i < params->lut_entries; i++) {
            double x = radius * i / (params->lut_entries - 1);
            weights[i] = pl_filter_sample(&f->params.config, x);
            if (fabs(weights[i]) > params->cutoff)
                f->radius_cutoff = x;
        }
    } else {
        // Pick the most appropriate row size
        f->row_size = ceil(f->radius * 2.0);
        if (params->max_row_size && f->row_size > params->max_row_size) {
            pl_info(ctx, "Required filter size %d exceeds the maximum allowed "
                    "size of %d. This may result in adverse effects (aliasing, "
                    "or moiré artifacts).", f->row_size, params->max_row_size);
            f->row_size = params->max_row_size;
            f->insufficient = true;
        }
        f->row_stride = PL_ALIGN(f->row_size, params->row_stride_align);

        // Compute a 2D array indexed by the subpixel position
        weights = talloc_zero_array(f, float, params->lut_entries * f->row_stride);
        for (int i = 0; i < params->lut_entries; i++) {
            compute_row(f, i / (double)(params->lut_entries - 1),
                        weights + f->row_stride * i);
        }
    }

    f->weights = weights;
    return f;
}

void pl_filter_free(const struct pl_filter **filter)
{
    TA_FREEP((void **) filter);
}

const struct pl_named_filter_function *pl_find_named_filter_function(const char *name)
{
    if (!name)
        return NULL;

    for (int i = 0; pl_named_filter_functions[i].function; i++) {
        if (strcmp(pl_named_filter_functions[i].name, name) == 0)
            return &pl_named_filter_functions[i];
    }

    return NULL;
}

const struct pl_named_filter_config *pl_find_named_filter(const char *name)
{
    if (!name)
        return NULL;

    for (int i = 0; pl_named_filters[i].filter; i++) {
        if (strcmp(pl_named_filters[i].name, name) == 0)
            return &pl_named_filters[i];
    }

    return NULL;
}

// Built-in filter functions

static double box(const struct pl_filter_function *f, double x)
{
    return 1.0;
}

const struct pl_filter_function pl_filter_function_box = {
    .resizable = true,
    .weight    = box,
    .radius    = 0.5,
};

static double triangle(const struct pl_filter_function *f, double x)
{
    return 1.0 - x / f->radius;
}

const struct pl_filter_function pl_filter_function_triangle = {
    .resizable = true,
    .weight    = triangle,
    .radius    = 1.0,
};

static double hann(const struct pl_filter_function *f, double x)
{
    return 0.5 + 0.5 * cos(M_PI * x);
}

const struct pl_filter_function pl_filter_function_hann = {
    .weight = hann,
    .radius = 1.0,
};

static double hamming(const struct pl_filter_function *f, double x)
{
    return 0.54 + 0.46 * cos(M_PI * x);
}

const struct pl_filter_function pl_filter_function_hamming = {
    .weight = hamming,
    .radius = 1.0,
};

static double welch(const struct pl_filter_function *f, double x)
{
    return 1.0 - x * x;
}

const struct pl_filter_function pl_filter_function_welch = {
    .weight = welch,
    .radius = 1.0,
};

static double bessel_i0(double x)
{
    double s = 1.0;
    double y = x * x / 4.0;
    double t = y;
    int i = 2;
    while (t > 1e-12) {
        s += t;
        t *= y / (i * i);
        i += 1;
    }
    return s;
}

static double kaiser(const struct pl_filter_function *f, double x)
{
    double alpha = fmax(f->params[0], 0.0);
    return bessel_i0(alpha * sqrt(1.0 - x * x)) / alpha;
}

const struct pl_filter_function pl_filter_function_kaiser = {
    .tunable = {true},
    .weight  = kaiser,
    .radius  = 1.0,
    .params  = {2.0},
};

static double blackman(const struct pl_filter_function *f, double x)
{
    double a = f->params[0];
    double a0 = (1 - a) / 2.0, a1 = 1 / 2.0, a2 = a / 2.0;
    x *= M_PI;
    return a0 + a1 * cos(x) + a2 * cos(2 * x);
}

const struct pl_filter_function pl_filter_function_blackman = {
    .tunable = {true},
    .weight  = blackman,
    .radius  = 1.0,
    .params  = {0.16},
};

static double gaussian(const struct pl_filter_function *f, double x)
{
    return exp(-2.0 * x * x / f->params[0]);
}

const struct pl_filter_function pl_filter_function_gaussian = {
    .resizable = true,
    .tunable   = {true},
    .weight    = gaussian,
    .radius    = 2.0,
    .params    = {1.0},
};

static double sinc(const struct pl_filter_function *f, double x)
{
    if (x < 1e-8)
        return 1.0;
    x *= M_PI;
    return sin(x) / x;
}

const struct pl_filter_function pl_filter_function_sinc = {
    .resizable = true,
    .weight    = sinc,
    .radius    = 1.0,
};

static double jinc(const struct pl_filter_function *f, double x)
{
    if (x < 1e-8)
        return 1.0;
    x *= M_PI;
    return 2.0 * j1(x) / x;
}

const struct pl_filter_function pl_filter_function_jinc = {
    .resizable = true,
    .weight    = jinc,
    .radius    = 1.2196698912665045, // first zero
};

static double sphinx(const struct pl_filter_function *f, double x)
{
    if (x < 1e-8)
        return 1.0;
    x *= M_PI;
    return 3.0 * (sin(x) - x * cos(x)) / (x * x * x);
}

const struct pl_filter_function pl_filter_function_sphinx = {
    .resizable = true,
    .weight    = sphinx,
    .radius    = 1.4302966531242027, // first zero
};

static double bcspline(const struct pl_filter_function *f, double x)
{
    double b = f->params[0],
           c = f->params[1];
    double p0 = (6.0 - 2.0 * b) / 6.0,
           p2 = (-18.0 + 12.0 * b + 6.0 * c) / 6.0,
           p3 = (12.0 - 9.0 * b - 6.0 * c) / 6.0,
           q0 = (8.0 * b + 24.0 * c) / 6.0,
           q1 = (-12.0 * b - 48.0 * c) / 6.0,
           q2 = (6.0 * b + 30.0 * c) / 6.0,
           q3 = (-b - 6.0 * c) / 6.0;

    // Needed to ensure the kernel is sanely scaled, i.e. bcspline(0.0) = 1.0
    double scale = 1.0 / p0;
    if (x < 1.0) {
        return scale * (p0 + x * x * (p2 + x * p3));
    } else if (x < 2.0) {
        return scale * (q0 + x * (q1 + x * (q2 + x * q3)));
    }
    return 0.0;
}

const struct pl_filter_function pl_filter_function_bcspline = {
    .tunable = {true, true},
    .weight  = bcspline,
    .radius  = 2.0,
    .params  = {0.5, 0.5},
};

const struct pl_filter_function pl_filter_function_catmull_rom = {
    .tunable = {true, true},
    .weight  = bcspline,
    .radius  = 2.0,
    .params  = {0.0, 0.5},
};

const struct pl_filter_function pl_filter_function_mitchell = {
    .tunable = {true, true},
    .weight  = bcspline,
    .radius  = 2.0,
    .params  = {1/3.0, 1/3.0},
};

const struct pl_filter_function pl_filter_function_robidoux = {
    .tunable = {true, true},
    .weight  = bcspline,
    .radius  = 2.0,
    .params  = {12 / (19 + 9 * M_SQRT2), 113 / (58 + 216 * M_SQRT2)},
};

const struct pl_filter_function pl_filter_function_robidouxsharp = {
    .tunable = {true, true},
    .weight  = bcspline,
    .radius  = 2.0,
    .params  = {6 / (13 + 7 * M_SQRT2), 7 / (2 + 12 * M_SQRT2)},
};

#define POW3(x) ((x) <= 0 ? 0 : (x) * (x) * (x))
static double bicubic(const struct pl_filter_function *f, double x)
{
    return (1.0/6.0) * (  1 * POW3(x + 2)
                        - 4 * POW3(x + 1)
                        + 6 * POW3(x + 0)
                        - 4 * POW3(x - 1));
}

const struct pl_filter_function pl_filter_function_bicubic = {
    .weight = bicubic,
    .radius = 2.0,
};

static double spline16(const struct pl_filter_function *f, double x)
{
    if (x < 1.0) {
        return ((x - 9.0/5.0 ) * x - 1.0/5.0 ) * x + 1.0;
    } else {
        return ((-1.0/3.0 * (x-1) + 4.0/5.0) * (x-1) - 7.0/15.0 ) * (x-1);
    }
}

const struct pl_filter_function pl_filter_function_spline16 = {
    .weight = spline16,
    .radius = 2.0,
};

static double spline36(const struct pl_filter_function *f, double x)
{
    if (x < 1.0) {
        return ((13.0/11.0 * x - 453.0/209.0) * x - 3.0/209.0) * x + 1.0;
    } else if (x < 2.0) {
        return ((-6.0/11.0 * (x-1) + 270.0/209.0) * (x-1) - 156.0/ 209.0) * (x-1);
    } else {
        return ((1.0/11.0 * (x-2) - 45.0/209.0) * (x-2) +  26.0/209.0) * (x-2);
    }
}

const struct pl_filter_function pl_filter_function_spline36 = {
    .weight = spline36,
    .radius = 3.0,
};

static double spline64(const struct pl_filter_function *f, double x)
{
    if (x < 1.0) {
        return ((49.0/41.0 * x - 6387.0/2911.0) * x - 3.0/2911.0) * x + 1.0;
    } else if (x < 2.0) {
        return ((-24.0/41.0 * (x-1) + 4032.0/2911.0) * (x-1) - 2328.0/2911.0) * (x-1);
    } else if (x < 3.0) {
        return ((6.0/41.0 * (x-2) - 1008.0/2911.0) * (x-2) + 582.0/2911.0) * (x-2);
    } else {
        return ((-1.0/41.0 * (x-3) + 168.0/2911.0) * (x-3) - 97.0/2911.0) * (x-3);
    }
}

const struct pl_filter_function pl_filter_function_spline64 = {
    .weight = spline64,
    .radius = 4.0,
};

// Named filter functions
const struct pl_named_filter_function pl_named_filter_functions[] = {
    {"box",             &pl_filter_function_box},
    {"dirichlet",       &pl_filter_function_box}, // alias
    {"triangle",        &pl_filter_function_triangle},
    {"hann",            &pl_filter_function_hann},
    {"hanning",         &pl_filter_function_hann}, // alias
    {"hamming",         &pl_filter_function_hamming},
    {"welch",           &pl_filter_function_welch},
    {"kaiser",          &pl_filter_function_kaiser},
    {"blackman",        &pl_filter_function_blackman},
    {"gaussian",        &pl_filter_function_gaussian},
    {"sinc",            &pl_filter_function_sinc},
    {"jinc",            &pl_filter_function_jinc},
    {"sphinx",          &pl_filter_function_sphinx},
    {"bcspline",        &pl_filter_function_bcspline},
    {"hermite",         &pl_filter_function_bcspline}, // alias
    {"catmull_rom",     &pl_filter_function_catmull_rom},
    {"mitchell",        &pl_filter_function_mitchell},
    {"robidoux",        &pl_filter_function_robidoux},
    {"robidouxsharp",   &pl_filter_function_robidouxsharp},
    {"bicubic",         &pl_filter_function_bicubic},
    {"spline16",        &pl_filter_function_spline16},
    {"spline36",        &pl_filter_function_spline36},
    {"spline64",        &pl_filter_function_spline64},
    {0},
};

// Built-in filter function presets
const struct pl_filter_config pl_filter_spline16 = {
    .kernel = &pl_filter_function_spline16,
};

const struct pl_filter_config pl_filter_spline36 = {
    .kernel = &pl_filter_function_spline36,
};

const struct pl_filter_config pl_filter_spline64 = {
    .kernel = &pl_filter_function_spline64,
};

const struct pl_filter_config pl_filter_box = {
    .kernel = &pl_filter_function_box,
};

const struct pl_filter_config pl_filter_triangle = {
    .kernel = &pl_filter_function_triangle,
};

const struct pl_filter_config pl_filter_gaussian = {
    .kernel = &pl_filter_function_gaussian,
};

// Sinc configured to three taps
static const struct pl_filter_function sinc3 = {
    .resizable = true,
    .weight    = sinc,
    .radius    = 3.0,
};

const struct pl_filter_config pl_filter_sinc = {
    .kernel = &sinc3,
};

const struct pl_filter_config pl_filter_lanczos = {
    .kernel = &sinc3,
    .window = &pl_filter_function_sinc,
};

const struct pl_filter_config pl_filter_ginseng = {
    .kernel = &sinc3,
    .window = &pl_filter_function_jinc,
};

// Jinc configured to three taps
static const struct pl_filter_function jinc3 = {
    .resizable = true,
    .weight    = jinc,
    .radius    = 3.2383154841662362, // third zero
};

const struct pl_filter_config pl_filter_ewa_jinc = {
    .kernel = &jinc3,
    .polar = true,
};

const struct pl_filter_config pl_filter_ewa_lanczos = {
    .kernel = &jinc3,
    .window = &pl_filter_function_jinc,
    .polar = true,
};

const struct pl_filter_config pl_filter_ewa_ginseng = {
    .kernel = &jinc3,
    .window = &pl_filter_function_sinc,
    .polar = true,
};

const struct pl_filter_config pl_filter_ewa_hann = {
    .kernel = &jinc3,
    .window = &pl_filter_function_hann,
    .polar = true,
};

const struct pl_filter_config pl_filter_haasnsoft = {
    .kernel = &jinc3,
    .window = &pl_filter_function_hann,
    // The blur is tuned to equal out orthogonal and diagonal contributions
    // on a regular grid. This has the effect of almost completely killing
    // aliasing.
    .blur = 1.11,
    .polar = true,
};

// Spline family
const struct pl_filter_config pl_filter_bicubic = {
    .kernel = &pl_filter_function_bicubic,
};

const struct pl_filter_config pl_filter_catmull_rom = {
    .kernel = &pl_filter_function_catmull_rom,
};

const struct pl_filter_config pl_filter_mitchell = {
    .kernel = &pl_filter_function_mitchell,
};

const struct pl_filter_config pl_filter_robidoux = {
    .kernel = &pl_filter_function_robidoux,
};

const struct pl_filter_config pl_filter_robidouxsharp = {
    .kernel = &pl_filter_function_robidouxsharp,
};

const struct pl_filter_config pl_filter_ewa_robidoux = {
    .kernel = &pl_filter_function_robidoux,
    .polar = true,
};

const struct pl_filter_config pl_filter_ewa_robidouxsharp = {
    .kernel = &pl_filter_function_robidouxsharp,
    .polar = true,
};

// Named filter configs
const struct pl_named_filter_config pl_named_filters[] = {
    {"spline16",            &pl_filter_spline16},
    {"spline36",            &pl_filter_spline36},
    {"spline64",            &pl_filter_spline64},
    {"box",                 &pl_filter_box},
    {"nearest",             &pl_filter_box}, // alias
    {"triangle",            &pl_filter_triangle},
    {"bilinear",            &pl_filter_triangle}, // alias
    {"gaussian",            &pl_filter_gaussian},
    {"sinc",                &pl_filter_sinc},
    {"lanczos",             &pl_filter_lanczos},
    {"ginseng",             &pl_filter_ginseng},
    {"ewa_jinc",            &pl_filter_ewa_jinc},
    {"ewa_lanczos",         &pl_filter_ewa_lanczos},
    {"ewa_ginseng",         &pl_filter_ewa_ginseng},
    {"ewa_hann",            &pl_filter_ewa_hann},
    {"ewa_hanning",         &pl_filter_ewa_hann}, // alias
    {"haasnsoft",           &pl_filter_haasnsoft},
    {"bicubic",             &pl_filter_bicubic},
    {"catmull_rom",         &pl_filter_catmull_rom},
    {"mitchell",            &pl_filter_mitchell},
    {"robidoux",            &pl_filter_robidoux},
    {"robidouxsharp",       &pl_filter_robidouxsharp},
    {"ewa_robidoux",        &pl_filter_ewa_robidoux},
    {"ewa_robidouxsharp",   &pl_filter_ewa_robidouxsharp},
    {0},
};
