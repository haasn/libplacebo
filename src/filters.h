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

#pragma once

#include <libplacebo/filters.h>

static inline float pl_filter_radius_bound(const struct pl_filter_config *c)
{
    const float r = c->radius && c->kernel->resizable ? c->radius : c->kernel->radius;
    return c->blur > 0.0 ? r * c->blur : r;
}

#define COMMON_FILTER_PRESETS                                                   \
    /* Highest priority / recommended filters */                                \
    {"bilinear",            &pl_filter_bilinear,    "Bilinear"},                \
    {"nearest",             &pl_filter_nearest,     "Nearest neighbour"},       \
    {"bicubic",             &pl_filter_bicubic,     "Bicubic"},                 \
    {"lanczos",             &pl_filter_lanczos,     "Lanczos"},                 \
    {"ewa_lanczos",         &pl_filter_ewa_lanczos, "Jinc (EWA Lanczos)"},      \
    {"ewa_lanczossharp",    &pl_filter_ewa_lanczossharp,     "Sharpened Jinc"}, \
    {"ewa_lanczosradius",   &pl_filter_ewa_lanczosradius,    "Sharpened Jinc, radius 3"}, \
    {"ewa_lanczos4sharpest",&pl_filter_ewa_lanczos4sharpest, "Sharpened Jinc-AR, 4 taps"},\
    {"gaussian",            &pl_filter_gaussian,    "Gaussian"},                \
    {"spline16",            &pl_filter_spline16,    "Spline (2 taps)"},         \
    {"spline36",            &pl_filter_spline36,    "Spline (3 taps)"},         \
    {"spline64",            &pl_filter_spline64,    "Spline (4 taps)"},         \
    {"mitchell",            &pl_filter_mitchell,    "Mitchell-Netravali"},      \
                                                                                \
    /* Remaining filters */                                                     \
    {"sinc",                &pl_filter_sinc,        "Sinc (unwindowed)"},       \
    {"ginseng",             &pl_filter_ginseng,     "Ginseng (Jinc-Sinc)"},     \
    {"ewa_jinc",            &pl_filter_ewa_jinc,    "EWA Jinc (unwindowed)"},   \
    {"ewa_ginseng",         &pl_filter_ewa_ginseng, "EWA Ginseng"},             \
    {"ewa_hann",            &pl_filter_ewa_hann,    "EWA Hann"},                \
    {"hermite",             &pl_filter_hermite,     "Hermite"},                 \
    {"catmull_rom",         &pl_filter_catmull_rom, "Catmull-Rom"},             \
    {"robidoux",            &pl_filter_robidoux,          "Robidoux"},          \
    {"robidouxsharp",       &pl_filter_robidouxsharp,     "RobidouxSharp"},     \
    {"ewa_hermite",         &pl_filter_ewa_hermite,       "EWA Hermite"},       \
    {"ewa_mitchell",        &pl_filter_ewa_mitchell,      "EWA Mitchell"},      \
    {"ewa_catmull_rom",     &pl_filter_ewa_catmull_rom,   "EWA Catmull-Rom"},   \
    {"ewa_robidoux",        &pl_filter_ewa_robidoux,      "EWA Robidoux"},      \
    {"ewa_robidouxsharp",   &pl_filter_ewa_robidouxsharp, "EWA RobidouxSharp"}, \
                                                                                \
    /* Aliases */                                                               \
    {"triangle",            &pl_filter_bilinear},                               \
    {"ewa_hanning",         &pl_filter_ewa_hann}
