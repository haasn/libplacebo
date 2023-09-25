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
#include "filters.h"
#include "log.h"

#ifdef PL_HAVE_WIN32
#define j1 _j1
#endif

bool pl_filter_function_eq(const struct pl_filter_function *a,
                           const struct pl_filter_function *b)
{
    return (a ? a->weight : NULL) == (b ? b->weight : NULL);
}

bool pl_filter_config_eq(const struct pl_filter_config *a,
                         const struct pl_filter_config *b)
{
    if (!a || !b)
        return a == b;

    bool eq = pl_filter_function_eq(a->kernel, b->kernel) &&
              pl_filter_function_eq(a->window, b->window) &&
              a->radius   == b->radius &&
              a->clamp    == b->clamp  &&
              a->blur     == b->blur   &&
              a->taper    == b->taper  &&
              a->polar    == b->polar  &&
              a->antiring == b->antiring;

    for (int i = 0; i < PL_FILTER_MAX_PARAMS; i++) {
        if (a->kernel->tunable[i])
            eq &= a->params[i] == b->params[i];
        if (a->window && a->window->tunable[i])
            eq &= a->wparams[i] == b->wparams[i];
    }

    return eq;
}

double pl_filter_sample(const struct pl_filter_config *c, double x)
{
    const float radius = pl_filter_radius_bound(c);

    // All filters are symmetric, and in particular only need to be defined
    // for [0, radius].
    x = fabs(x);

    // Return early for values outside of the kernel radius, since the functions
    // are not necessarily valid outside of this interval. No such check is
    // needed for the window, because it's always stretched to fit.
    if (x > radius)
        return 0.0;

    // Apply the blur and taper coefficients as needed
    double kx = x <= c->taper ? 0.0 : (x - c->taper) / (1.0 - c->taper / radius);
    if (c->blur > 0.0)
        kx /= c->blur;

    pl_assert(!c->kernel->opaque);
    double k = c->kernel->weight(&(const struct pl_filter_ctx) {
        .radius = radius,
        .params = {
            c->kernel->tunable[0] ? c->params[0] : c->kernel->params[0],
            c->kernel->tunable[1] ? c->params[1] : c->kernel->params[1],
        },
    }, kx);

    // Apply the optional windowing function
    if (c->window) {
        pl_assert(!c->window->opaque);
        double wx = x / radius * c->window->radius;
        k *= c->window->weight(&(struct pl_filter_ctx) {
            .radius = c->window->radius,
            .params = {
                c->window->tunable[0] ? c->wparams[0] : c->window->params[0],
                c->window->tunable[1] ? c->wparams[1] : c->window->params[1],
            },
        }, wx);
    }

    return k < 0 ? (1 - c->clamp) * k : k;
}

static void filter_cutoffs(const struct pl_filter_config *c, float cutoff,
                           float *out_radius, float *out_radius_zero)
{
    const float bound = pl_filter_radius_bound(c);
    float prev = 0.0, fprev = pl_filter_sample(c, prev);
    bool found_root = false;

    const float step = 1e-2f;
    for (float x = 0.0; x < bound + step; x += step) {
        float fx = pl_filter_sample(c, x);
        if ((fprev > cutoff && fx <= cutoff) || (fprev < -cutoff && fx >= -cutoff)) {
            // Found zero crossing
            float root = x - fx * (x - prev) / (fx - fprev); // secant method
            root = fminf(root, bound);
            *out_radius = root;
            if (!found_root) // first root
                *out_radius_zero = root;
            found_root = true;
        }
        prev = x;
        fprev = fx;
    }

    if (!found_root)
        *out_radius_zero = *out_radius = bound;
}

// Compute a single row of weights for a given filter in one dimension, indexed
// by the indicated subpixel offset. Writes `f->row_size` values to `out`.
static void compute_row(struct pl_filter_t *f, double offset, float *out)
{
    double wsum = 0.0;
    for (int i = 0; i < f->row_size; i++) {
        // For the example of a filter with row size 4 and offset 0.3, we have:
        //
        // 0    1 *  2    3
        //
        // * indicates the sampled position. What we want to compute is the
        // distance from each index to that sampled position.
        pl_assert(f->row_size % 2 == 0);
        const int base = f->row_size / 2 - 1; // index to the left of the center
        const double center = base + offset; // offset of center relative to idx 0
        double w = pl_filter_sample(&f->params.config, i - center);
        out[i] = w;
        wsum += w;
    }

    // Readjust weights to preserve energy
    pl_assert(wsum > 0);
    for (int i = 0; i < f->row_size; i++)
        out[i] /= wsum;
}

// Needed for backwards compatibility with v1 configuration API
static struct pl_filter_function *dupfilter(void *alloc,
                                            const struct pl_filter_function *f)
{
    return f ? pl_memdup(alloc, (void *)f, sizeof(*f)) : NULL;
}

pl_filter pl_filter_generate(pl_log log, const struct pl_filter_params *params)
{
    pl_assert(params);
    if (params->lut_entries <= 0 || !params->config.kernel) {
        pl_fatal(log, "Invalid params: missing lut_entries or config.kernel");
        return NULL;
    }

    if (params->config.kernel->opaque) {
        pl_err(log, "Trying to use opaque kernel '%s' in non-opaque context!",
               params->config.kernel->name);
        return NULL;
    }

    if (params->config.window && params->config.window->opaque) {
        pl_err(log, "Trying to use opaque window '%s' in non-opaque context!",
               params->config.window->name);
        return NULL;
    }

    struct pl_filter_t *f = pl_zalloc_ptr(NULL, f);
    f->params = *params;
    f->params.config.kernel = dupfilter(f, params->config.kernel);
    f->params.config.window = dupfilter(f, params->config.window);

    // Compute main lobe and total filter size
    filter_cutoffs(&params->config, params->cutoff, &f->radius, &f->radius_zero);
    f->radius_cutoff = f->radius; // backwards compatibility

    float *weights;
    if (params->config.polar) {
        // Compute a 1D array indexed by radius
        weights = pl_alloc(f, params->lut_entries * sizeof(float));
        for (int i = 0; i < params->lut_entries; i++) {
            double x = f->radius * i / (params->lut_entries - 1);
            weights[i] = pl_filter_sample(&params->config, x);
        }
    } else {
        // Pick the most appropriate row size
        f->row_size = ceilf(f->radius) * 2;
        if (params->max_row_size && f->row_size > params->max_row_size) {
            pl_info(log, "Required filter size %d exceeds the maximum allowed "
                    "size of %d. This may result in adverse effects (aliasing, "
                    "or moiré artifacts).", f->row_size, params->max_row_size);
            f->row_size = params->max_row_size;
            f->insufficient = true;
        }
        f->row_stride = PL_ALIGN(f->row_size, params->row_stride_align);

        // Compute a 2D array indexed by the subpixel position
        weights = pl_calloc(f, params->lut_entries * f->row_stride, sizeof(float));
        for (int i = 0; i < params->lut_entries; i++) {
            compute_row(f, i / (double)(params->lut_entries - 1),
                        weights + f->row_stride * i);
        }
    }

    f->weights = weights;
    return f;
}

void pl_filter_free(pl_filter *filter)
{
    pl_free_ptr((void **) filter);
}

// Built-in filter functions

static double box(const struct pl_filter_ctx *f, double x)
{
    return 1.0;
}

const struct pl_filter_function pl_filter_function_box = {
    .weight    = box,
    .name      = "box",
    .radius    = 1.0,
    .resizable = true,
};

static const struct pl_filter_function filter_function_dirichlet = {
    .name      = "dirichlet", // alias
    .weight    = box,
    .radius    = 1.0,
    .resizable = true,
};

static double triangle(const struct pl_filter_ctx *f, double x)
{
    return 1.0 - x / f->radius;
}

const struct pl_filter_function pl_filter_function_triangle = {
    .name      = "triangle",
    .weight    = triangle,
    .radius    = 1.0,
    .resizable = true,
};

static double cosine(const struct pl_filter_ctx *f, double x)
{
    return cos(x);
}

const struct pl_filter_function pl_filter_function_cosine = {
    .name   = "cosine",
    .weight = cosine,
    .radius = M_PI / 2.0,
};

static double hann(const struct pl_filter_ctx *f, double x)
{
    return 0.5 + 0.5 * cos(M_PI * x);
}

const struct pl_filter_function pl_filter_function_hann = {
    .name   = "hann",
    .weight = hann,
    .radius = 1.0,
};

static const struct pl_filter_function filter_function_hanning = {
    .name   = "hanning", // alias
    .weight = hann,
    .radius = 1.0,
};

static double hamming(const struct pl_filter_ctx *f, double x)
{
    return 0.54 + 0.46 * cos(M_PI * x);
}

const struct pl_filter_function pl_filter_function_hamming = {
    .name   = "hamming",
    .weight = hamming,
    .radius = 1.0,
};

static double welch(const struct pl_filter_ctx *f, double x)
{
    return 1.0 - x * x;
}

const struct pl_filter_function pl_filter_function_welch = {
    .name   = "welch",
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

static double kaiser(const struct pl_filter_ctx *f, double x)
{
    double alpha = fmax(f->params[0], 0.0);
    double scale = bessel_i0(alpha);
    return bessel_i0(alpha * sqrt(1.0 - x * x)) / scale;
}

const struct pl_filter_function pl_filter_function_kaiser = {
    .name    = "kaiser",
    .weight  = kaiser,
    .radius  = 1.0,
    .params  = {2.0},
    .tunable = {true},
};

static double blackman(const struct pl_filter_ctx *f, double x)
{
    double a = f->params[0];
    double a0 = (1 - a) / 2.0, a1 = 1 / 2.0, a2 = a / 2.0;
    x *= M_PI;
    return a0 + a1 * cos(x) + a2 * cos(2 * x);
}

const struct pl_filter_function pl_filter_function_blackman = {
    .name    = "blackman",
    .weight  = blackman,
    .radius  = 1.0,
    .params  = {0.16},
    .tunable = {true},
};

static double bohman(const struct pl_filter_ctx *f, double x)
{
    double pix = M_PI * x;
    return (1.0 - x) * cos(pix) + sin(pix) / M_PI;
}

const struct pl_filter_function pl_filter_function_bohman = {
    .name   = "bohman",
    .weight = bohman,
    .radius = 1.0,
};

static double gaussian(const struct pl_filter_ctx *f, double x)
{
    return exp(-2.0 * x * x / f->params[0]);
}

const struct pl_filter_function pl_filter_function_gaussian = {
    .name      = "gaussian",
    .weight    = gaussian,
    .radius    = 2.0,
    .resizable = true,
    .params    = {1.0},
    .tunable   = {true},
};

static double quadratic(const struct pl_filter_ctx *f, double x)
{
    if (x < 0.5) {
        return 1.0 - 4.0/3.0 * (x * x);
    } else {
        return 2.0 / 3.0 * (x - 1.5) * (x - 1.5);
    }
}

const struct pl_filter_function pl_filter_function_quadratic = {
    .name   = "quadratic",
    .weight = quadratic,
    .radius = 1.5,
};

static const struct pl_filter_function filter_function_quadric = {
    .name   = "quadric", // alias
    .weight = quadratic,
    .radius = 1.5,
};

static double sinc(const struct pl_filter_ctx *f, double x)
{
    if (x < 1e-8)
        return 1.0;
    x *= M_PI;
    return sin(x) / x;
}

const struct pl_filter_function pl_filter_function_sinc = {
    .name      = "sinc",
    .weight    = sinc,
    .radius    = 1.0,
    .resizable = true,
};

static double jinc(const struct pl_filter_ctx *f, double x)
{
    if (x < 1e-8)
        return 1.0;
    x *= M_PI;
    return 2.0 * j1(x) / x;
}

const struct pl_filter_function pl_filter_function_jinc = {
    .name      = "jinc",
    .weight    = jinc,
    .radius    = 1.2196698912665045, // first zero
    .resizable = true,
};

static double sphinx(const struct pl_filter_ctx *f, double x)
{
    if (x < 1e-8)
        return 1.0;
    x *= M_PI;
    return 3.0 * (sin(x) - x * cos(x)) / (x * x * x);
}

const struct pl_filter_function pl_filter_function_sphinx = {
    .name      = "sphinx",
    .weight    = sphinx,
    .radius    = 1.4302966531242027, // first zero
    .resizable = true,
};

static double cubic(const struct pl_filter_ctx *f, double x)
{
    const double b = f->params[0], c = f->params[1];
    double p0 = 6.0 - 2.0 * b,
           p2 = -18.0 + 12.0 * b + 6.0 * c,
           p3 = 12.0 - 9.0 * b - 6.0 * c,
           q0 = 8.0 * b + 24.0 * c,
           q1 = -12.0 * b - 48.0 * c,
           q2 = 6.0 * b + 30.0 * c,
           q3 = -b - 6.0 * c;

    if (x < 1.0) {
        return (p0 + x * x * (p2 + x * p3)) / p0;
    } else {
        return (q0 + x * (q1 + x * (q2 + x * q3))) / p0;
    }
}

const struct pl_filter_function pl_filter_function_cubic = {
    .name    = "cubic",
    .weight  = cubic,
    .radius  = 2.0,
    .params  = {1.0, 0.0},
    .tunable = {true, true},
};

static const struct pl_filter_function filter_function_bicubic = {
    .name    = "bicubic", // alias
    .weight  = cubic,
    .radius  = 2.0,
    .params  = {1.0, 0.0},
    .tunable = {true, true},
};

static const struct pl_filter_function filter_function_bcspline = {
    .name    = "bcspline", // alias
    .weight  = cubic,
    .radius  = 2.0,
    .params  = {1.0, 0.0},
    .tunable = {true, true},
};

const struct pl_filter_function pl_filter_function_hermite = {
    .name    = "hermite",
    .weight  = cubic,
    .radius  = 1.0,
    .params  = {0.0, 0.0},
};

static double spline16(const struct pl_filter_ctx *f, double x)
{
    if (x < 1.0) {
        return ((x - 9.0/5.0 ) * x - 1.0/5.0 ) * x + 1.0;
    } else {
        return ((-1.0/3.0 * (x-1) + 4.0/5.0) * (x-1) - 7.0/15.0 ) * (x-1);
    }
}

const struct pl_filter_function pl_filter_function_spline16 = {
    .name   = "spline16",
    .weight = spline16,
    .radius = 2.0,
};

static double spline36(const struct pl_filter_ctx *f, double x)
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
    .name   = "spline36",
    .weight = spline36,
    .radius = 3.0,
};

static double spline64(const struct pl_filter_ctx *f, double x)
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
    .name   = "spline64",
    .weight = spline64,
    .radius = 4.0,
};

static double oversample(const struct pl_filter_ctx *f, double x)
{
    return 0.0;
}

const struct pl_filter_function pl_filter_function_oversample = {
    .name    = "oversample",
    .weight  = oversample,
    .params  = {0.0},
    .tunable = {true},
    .opaque  = true,
};

const struct pl_filter_function * const pl_filter_functions[] = {
    &pl_filter_function_box,
    &filter_function_dirichlet, // alias
    &pl_filter_function_triangle,
    &pl_filter_function_cosine,
    &pl_filter_function_hann,
    &filter_function_hanning, // alias
    &pl_filter_function_hamming,
    &pl_filter_function_welch,
    &pl_filter_function_kaiser,
    &pl_filter_function_blackman,
    &pl_filter_function_bohman,
    &pl_filter_function_gaussian,
    &pl_filter_function_quadratic,
    &filter_function_quadric, // alias
    &pl_filter_function_sinc,
    &pl_filter_function_jinc,
    &pl_filter_function_sphinx,
    &pl_filter_function_cubic,
    &filter_function_bicubic, // alias
    &filter_function_bcspline, // alias
    &pl_filter_function_hermite,
    &pl_filter_function_spline16,
    &pl_filter_function_spline36,
    &pl_filter_function_spline64,
    &pl_filter_function_oversample,
    NULL,
};

const int pl_num_filter_functions = PL_ARRAY_SIZE(pl_filter_functions) - 1;

const struct pl_filter_function *pl_find_filter_function(const char *name)
{
    if (!name)
        return NULL;

    for (int i = 0; pl_num_filter_functions; i++) {
        if (strcmp(name, pl_filter_functions[i]->name) == 0)
            return pl_filter_functions[i];
    }

    return NULL;
}

// Built-in filter function configs

const struct pl_filter_config pl_filter_spline16 = {
    .name        = "spline16",
    .description = "Spline (2 taps)",
    .kernel      = &pl_filter_function_spline16,
    .allowed     = PL_FILTER_ALL,
};

const struct pl_filter_config pl_filter_spline36 = {
    .name        = "spline36",
    .description = "Spline (3 taps)",
    .kernel      = &pl_filter_function_spline36,
    .allowed     = PL_FILTER_ALL,
};

const struct pl_filter_config pl_filter_spline64 = {
    .name        = "spline64",
    .description = "Spline (4 taps)",
    .kernel      = &pl_filter_function_spline64,
    .allowed     = PL_FILTER_ALL,
};

const struct pl_filter_config pl_filter_nearest = {
    .name        = "nearest",
    .description = "Nearest neighbor",
    .kernel      = &pl_filter_function_box,
    .radius      = 0.5,
    .allowed     = PL_FILTER_SCALING,
    .recommended = PL_FILTER_SCALING,
};

const struct pl_filter_config pl_filter_bilinear = {
    .name        = "bilinear",
    .description = "Bilinear",
    .kernel      = &pl_filter_function_triangle,
    .allowed     = PL_FILTER_ALL,
    .recommended = PL_FILTER_SCALING,
};

const struct pl_filter_config filter_linear = {
    .name        = "linear",
    .description = "Linear mixing",
    .kernel      = &pl_filter_function_triangle,
    .allowed     = PL_FILTER_FRAME_MIXING,
    .recommended = PL_FILTER_FRAME_MIXING,
};

static const struct pl_filter_config filter_triangle = {
    .name        = "triangle",
    .kernel      = &pl_filter_function_triangle,
    .allowed     = PL_FILTER_SCALING,
};

const struct pl_filter_config pl_filter_gaussian = {
    .name        = "gaussian",
    .description = "Gaussian",
    .kernel      = &pl_filter_function_gaussian,
    .params      = {1.0},
    .allowed     = PL_FILTER_ALL,
    .recommended = PL_FILTER_SCALING,
};

const struct pl_filter_config pl_filter_sinc = {
    .name        = "sinc",
    .description = "Sinc (unwindowed)",
    .kernel      = &pl_filter_function_sinc,
    .radius      = 3.0,
    .allowed     = PL_FILTER_ALL,
};

const struct pl_filter_config pl_filter_lanczos = {
    .name        = "lanczos",
    .description = "Lanczos",
    .kernel      = &pl_filter_function_sinc,
    .window      = &pl_filter_function_sinc,
    .radius      = 3.0,
    .allowed     = PL_FILTER_ALL,
    .recommended = PL_FILTER_SCALING,
};

const struct pl_filter_config pl_filter_ginseng = {
    .name        = "ginseng",
    .description = "Ginseng (Jinc-Sinc)",
    .kernel      = &pl_filter_function_sinc,
    .window      = &pl_filter_function_jinc,
    .radius      = 3.0,
    .allowed     = PL_FILTER_ALL,
};

#define JINC_ZERO3 3.2383154841662362076499
#define JINC_ZERO4 4.2410628637960698819573

const struct pl_filter_config pl_filter_ewa_jinc = {
    .name        = "ewa_jinc",
    .description = "EWA Jinc (unwindowed)",
    .kernel      = &pl_filter_function_jinc,
    .radius      = JINC_ZERO3,
    .polar       = true,
    .allowed     = PL_FILTER_SCALING,
};

const struct pl_filter_config pl_filter_ewa_lanczos = {
    .name        = "ewa_lanczos",
    .description = "Jinc (EWA Lanczos)",
    .kernel      = &pl_filter_function_jinc,
    .window      = &pl_filter_function_jinc,
    .radius      = JINC_ZERO3,
    .polar       = true,
    .allowed     = PL_FILTER_SCALING,
    .recommended = PL_FILTER_UPSCALING,
};

const struct pl_filter_config pl_filter_ewa_lanczossharp = {
    .name        = "ewa_lanczossharp",
    .description = "Sharpened Jinc",
    .kernel      = &pl_filter_function_jinc,
    .window      = &pl_filter_function_jinc,
    .radius      = JINC_ZERO3,
    .blur        = 0.98125058372237073562493,
    .polar       = true,
    .allowed     = PL_FILTER_SCALING,
    .recommended = PL_FILTER_UPSCALING,
};

const struct pl_filter_config pl_filter_ewa_lanczos4sharpest = {
    .name        = "ewa_lanczos4sharpest",
    .description = "Sharpened Jinc-AR, 4 taps",
    .kernel      = &pl_filter_function_jinc,
    .window      = &pl_filter_function_jinc,
    .radius      = JINC_ZERO4,
    .blur        = 0.88451209326050047745788,
    .antiring    = 0.8,
    .polar       = true,
    .allowed     = PL_FILTER_SCALING,
    .recommended = PL_FILTER_UPSCALING,
};

const struct pl_filter_config pl_filter_ewa_ginseng = {
    .name        = "ewa_ginseng",
    .description = "EWA Ginseng",
    .kernel      = &pl_filter_function_jinc,
    .window      = &pl_filter_function_sinc,
    .radius      = JINC_ZERO3,
    .polar       = true,
    .allowed     = PL_FILTER_SCALING,
};

const struct pl_filter_config pl_filter_ewa_hann = {
    .name        = "ewa_hann",
    .description = "EWA Hann",
    .kernel      = &pl_filter_function_jinc,
    .window      = &pl_filter_function_hann,
    .radius      = JINC_ZERO3,
    .polar       = true,
    .allowed     = PL_FILTER_SCALING,
};

static const struct pl_filter_config filter_ewa_hanning = {
    .name        = "ewa_hanning",
    .kernel      = &pl_filter_function_jinc,
    .window      = &pl_filter_function_hann,
    .radius      = JINC_ZERO3,
    .polar       = true,
    .allowed     = PL_FILTER_SCALING,
};

// Spline family
const struct pl_filter_config pl_filter_bicubic = {
    .name        = "bicubic",
    .description = "Bicubic",
    .kernel      = &pl_filter_function_cubic,
    .params      = {1.0, 0.0},
    .allowed     = PL_FILTER_SCALING,
    .recommended = PL_FILTER_SCALING,
};

static const struct pl_filter_config filter_cubic = {
    .name        = "cubic",
    .description = "Cubic",
    .kernel      = &pl_filter_function_cubic,
    .params      = {1.0, 0.0},
    .allowed     = PL_FILTER_FRAME_MIXING,
};

const struct pl_filter_config pl_filter_hermite = {
    .name        = "hermite",
    .description = "Hermite",
    .kernel      = &pl_filter_function_hermite,
    .allowed     = PL_FILTER_ALL,
    .recommended = PL_FILTER_DOWNSCALING | PL_FILTER_FRAME_MIXING,
};

const struct pl_filter_config pl_filter_catmull_rom = {
    .name        = "catmull_rom",
    .description = "Catmull-Rom",
    .kernel      = &pl_filter_function_cubic,
    .params      = {0.0, 0.5},
    .allowed     = PL_FILTER_ALL,
    .recommended = PL_FILTER_SCALING,
};

const struct pl_filter_config pl_filter_mitchell = {
    .name        = "mitchell",
    .description = "Mitchell-Netravali",
    .kernel      = &pl_filter_function_cubic,
    .params      = {1/3.0, 1/3.0},
    .allowed     = PL_FILTER_ALL,
    .recommended = PL_FILTER_DOWNSCALING,
};

const struct pl_filter_config pl_filter_mitchell_clamp = {
    .name        = "mitchell_clamp",
    .description = "Mitchell (clamped)",
    .kernel      = &pl_filter_function_cubic,
    .params      = {1/3.0, 1/3.0},
    .clamp       = 1.0,
    .allowed     = PL_FILTER_ALL,
};

const struct pl_filter_config pl_filter_robidoux = {
    .name        = "robidoux",
    .description = "Robidoux",
    .kernel      = &pl_filter_function_cubic,
    .params      = {12 / (19 + 9 * M_SQRT2), 113 / (58 + 216 * M_SQRT2)},
    .allowed     = PL_FILTER_ALL,
};

const struct pl_filter_config pl_filter_robidouxsharp = {
    .name        = "robidouxsharp",
    .description = "RobidouxSharp",
    .kernel      = &pl_filter_function_cubic,
    .params      = {6 / (13 + 7 * M_SQRT2), 7 / (2 + 12 * M_SQRT2)},
    .allowed     = PL_FILTER_ALL,
};

const struct pl_filter_config pl_filter_ewa_robidoux = {
    .name        = "ewa_robidoux",
    .description = "EWA Robidoux",
    .kernel      = &pl_filter_function_cubic,
    .params      = {12 / (19 + 9 * M_SQRT2), 113 / (58 + 216 * M_SQRT2)},
    .polar       = true,
    .allowed     = PL_FILTER_SCALING,
};

const struct pl_filter_config pl_filter_ewa_robidouxsharp = {
    .name        = "ewa_robidouxsharp",
    .description = "EWA RobidouxSharp",
    .kernel      = &pl_filter_function_cubic,
    .params      = {6 / (13 + 7 * M_SQRT2), 7 / (2 + 12 * M_SQRT2)},
    .polar       = true,
    .allowed     = PL_FILTER_SCALING,
};

const struct pl_filter_config pl_filter_oversample = {
    .name        = "oversample",
    .description = "Oversampling",
    .kernel      = &pl_filter_function_oversample,
    .params      = {0.0},
    .allowed     = PL_FILTER_UPSCALING | PL_FILTER_FRAME_MIXING,
    .recommended = PL_FILTER_UPSCALING | PL_FILTER_FRAME_MIXING,
};

const struct pl_filter_config * const pl_filter_configs[] = {
    // Sorted roughly in terms of priority / relevance
    &pl_filter_bilinear,
    &filter_triangle, // alias
    &filter_linear, // pseudo-alias (frame mixing only)
    &pl_filter_nearest,
    &pl_filter_spline16,
    &pl_filter_spline36,
    &pl_filter_spline64,
    &pl_filter_lanczos,
    &pl_filter_ewa_lanczos,
    &pl_filter_ewa_lanczossharp,
    &pl_filter_ewa_lanczos4sharpest,
    &pl_filter_bicubic,
    &filter_cubic, // pseudo-alias (frame mixing only)
    &pl_filter_hermite,
    &pl_filter_gaussian,
    &pl_filter_oversample,
    &pl_filter_mitchell,
    &pl_filter_mitchell_clamp,
    &pl_filter_sinc,
    &pl_filter_ginseng,
    &pl_filter_ewa_jinc,
    &pl_filter_ewa_ginseng,
    &pl_filter_ewa_hann,
    &filter_ewa_hanning, // alias
    &pl_filter_catmull_rom,
    &pl_filter_robidoux,
    &pl_filter_robidouxsharp,
    &pl_filter_ewa_robidoux,
    &pl_filter_ewa_robidouxsharp,

    NULL,
};

const int pl_num_filter_configs = PL_ARRAY_SIZE(pl_filter_configs) - 1;

const struct pl_filter_config *
pl_find_filter_config(const char *name, enum pl_filter_usage usage)
{
    if (!name)
        return NULL;

    for (int i = 0; pl_num_filter_configs; i++) {
        if ((pl_filter_configs[i]->allowed & usage) != usage)
            continue;
        if (strcmp(name, pl_filter_configs[i]->name) == 0)
            return pl_filter_configs[i];
    }

    return NULL;
}

// Backwards compatibility with older API

const struct pl_filter_function_preset pl_filter_function_presets[] = {
    {"none",            NULL},
    {"box",             &pl_filter_function_box},
    {"dirichlet",       &filter_function_dirichlet}, // alias
    {"triangle",        &pl_filter_function_triangle},
    {"cosine",          &pl_filter_function_cosine},
    {"hann",            &pl_filter_function_hann},
    {"hanning",         &filter_function_hanning}, // alias
    {"hamming",         &pl_filter_function_hamming},
    {"welch",           &pl_filter_function_welch},
    {"kaiser",          &pl_filter_function_kaiser},
    {"blackman",        &pl_filter_function_blackman},
    {"bohman",          &pl_filter_function_bohman},
    {"gaussian",        &pl_filter_function_gaussian},
    {"quadratic",       &pl_filter_function_quadratic},
    {"quadric",         &filter_function_quadric}, // alias
    {"sinc",            &pl_filter_function_sinc},
    {"jinc",            &pl_filter_function_jinc},
    {"sphinx",          &pl_filter_function_sphinx},
    {"cubic",           &pl_filter_function_cubic},
    {"bicubic",         &filter_function_bicubic}, // alias
    {"bcspline",        &filter_function_bcspline}, // alias
    {"hermite",         &pl_filter_function_hermite},
    {"spline16",        &pl_filter_function_spline16},
    {"spline36",        &pl_filter_function_spline36},
    {"spline64",        &pl_filter_function_spline64},
    {0},
};

const int pl_num_filter_function_presets = PL_ARRAY_SIZE(pl_filter_function_presets) - 1;

const struct pl_filter_function_preset *pl_find_filter_function_preset(const char *name)
{
    if (!name)
        return NULL;

    for (int i = 0; pl_filter_function_presets[i].name; i++) {
        if (strcmp(pl_filter_function_presets[i].name, name) == 0)
            return &pl_filter_function_presets[i];
    }

    return NULL;
}

const struct pl_filter_preset *pl_find_filter_preset(const char *name)
{
    if (!name)
        return NULL;

    for (int i = 0; pl_filter_presets[i].name; i++) {
        if (strcmp(pl_filter_presets[i].name, name) == 0)
            return &pl_filter_presets[i];
    }

    return NULL;
}

const struct pl_filter_preset pl_filter_presets[] = {
    {"none",                NULL,                   "Built-in sampling"},
    COMMON_FILTER_PRESETS,
    {0}
};

const int pl_num_filter_presets = PL_ARRAY_SIZE(pl_filter_presets) - 1;
