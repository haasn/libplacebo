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

#ifndef LIBPLACEBO_OPTIONS_H_
#define LIBPLACEBO_OPTIONS_H_

#include <libplacebo/renderer.h>

PL_API_BEGIN

// High-level heap-managed struct containing storage for all options implied by
// pl_render_params, including a high-level interface for serializing,
// deserializing and interfacing with them in a programmatic way.

typedef const struct pl_opt_t *pl_opt;
typedef struct pl_options_t {
    // Non-NULL `params.*_params` pointers must always point into this struct
    struct pl_render_params params;

    // Backing storage for all of the various rendering parameters. Whether
    // or not these params are active is determined by whether or not
    // `params.*_params` is set to this address or NULL.
    struct pl_deband_params deband_params;
    struct pl_sigmoid_params sigmoid_params;
    struct pl_color_adjustment color_adjustment;
    struct pl_peak_detect_params peak_detect_params;
    struct pl_color_map_params color_map_params;
    struct pl_dither_params dither_params;
    struct pl_icc_params icc_params PL_DEPRECATED_IN(v6.327);
    struct pl_cone_params cone_params;
    struct pl_blend_params blend_params;
    struct pl_deinterlace_params deinterlace_params;
    struct pl_distort_params distort_params;

    // Backing storage for "custom" scalers. `params.upscaler` etc. will
    // always be a pointer either to a built-in pl_filter_config, or one of
    // these structs. `name`, `description` and `allowed` will always be
    // valid for the respective type of filter config.
    struct pl_filter_config upscaler;
    struct pl_filter_config downscaler;
    struct pl_filter_config plane_upscaler;
    struct pl_filter_config plane_downscaler;
    struct pl_filter_config frame_mixer;
} *pl_options;

// Allocate a new set of render params, with internally backed storage for
// all parameters. Initialized to an "empty" config (PL_RENDER_DEFAULTS),
// equivalent to `&pl_render_fast_params`. To initialize the struct instead to
// the recommended default parameters, use `pl_options_reset` with
// `pl_render_default_params`.
//
// If `log` is provided, errors related to parsing etc. will be logged there.
PL_API pl_options pl_options_alloc(pl_log log);
PL_API void pl_options_free(pl_options *opts);

// Resets all options to their default values from a given struct. If `preset`
// is NULL, `opts` is instead reset back to the initial "empty" configuration,
// with all options disabled, as if it was freshly allocated.
//
// Note: This function will also reset structs which were not included in
// `preset`, such as any custom upscalers.
PL_API void pl_options_reset(pl_options opts, const struct pl_render_params *preset);

typedef const struct pl_opt_data_t {
    // Original options struct.
    pl_options opts;

    // Triggering option for this callback invocation.
    pl_opt opt;

    // The raw data associated with this option. Always some pointer into
    // `opts`. Note that only PL_OPT_BOOL, PL_OPT_INT and PL_OPT_FLOAT have
    // a fixed representation, for other fields its usefulness is dubious.
    const void *value;

    // The underlying data, as a formatted, locale-invariant string. Lifetime
    // is limited until the return of this callback.
    const char *text;
} *pl_opt_data;

// Query a single option from `opts` by key, or NULL if none was found.
// The resulting pointer is only valid until the next pl_options_* call.
PL_API pl_opt_data pl_options_get(pl_options opts, const char *key);

// Update an option from a formatted value string (see `pl_opt_data.text`).
// This can be used for all type of options, even non-string ones. In this case,
// `value` will be parsed according to the option type.
//
// Returns whether successful.
PL_API bool pl_options_set_str(pl_options opts, const char *key, const char *value);

// Programmatically iterate over options set in a `pl_options`, running the
// provided callback on each entry.
PL_API void pl_options_iterate(pl_options opts,
                               void (*cb)(void *priv, pl_opt_data data),
                               void *priv);

// Serialize a `pl_options` structs to a comma-separated key/value string. The
// returned string has a lifetime valid until either the next call to
// `pl_options_save`, or until the `pl_options` is freed.
PL_API const char *pl_options_save(pl_options opts);

// Parse a `pl_options` struct from a key/value string, in standard syntax
// "key1=value1,key2=value2,...", and updates `opts` with the new values.
// Valid separators include whitespace, commas (,) and (semi)colons (:;).
//
// Returns true if no errors occurred.
PL_API bool pl_options_load(pl_options opts, const char *str);

// Helpers for interfacing with `opts->params.hooks`. Note that using any of
// these helpers will overwrite the array by an internally managed pointer,
// so care must be taken when combining them with external management of
// this memory. Negative indices are possible and are counted relative to the
// end of the list.
//
// Note: These hooks are *not* included in pl_options_save() and related.
PL_API void pl_options_add_hook(pl_options opts, const struct pl_hook *hook);
PL_API void pl_options_insert_hook(pl_options opts, const struct pl_hook *hook, int idx);
PL_API void pl_options_remove_hook_at(pl_options opts, int idx);

// Underlying options system and list
//
// Note: By necessity, this option list does not cover every single field
// present in `pl_render_params`. In particular, fields like `info_callback`,
// `lut` and `hooks` cannot be configured through the options system, as doing
// so would require interop with C code or I/O. (However, see
// `pl_options_add_hook` and related)

enum pl_option_type {
    // Accepts `yes/no`, `on/off`, `true/false` and variants
    PL_OPT_BOOL,

    // Parsed as human-readable locale-invariant (C) numbers, scientific
    // notation accepted for floats
    PL_OPT_INT,
    PL_OPT_FLOAT,

    // Parsed as a short string containing only alphanumerics and _-,
    // corresponding to some name/identifier. Catch-all bucket for several
    // other types of options, such as presets, struct pointers, and functions
    //
    // Note: These options do not correspond to actual strings in C, the
    // underlying type of option will determine the values of `size` and
    // corresponding interpretation of pointers.
    PL_OPT_STRING,

    PL_OPT_TYPE_COUNT,
};

struct pl_opt_t {
    // Programmatic key uniquely identifying this option.
    const char *key;

    // Longer, human readable friendly name
    const char *name;

    // Data type of option, affects how it is parsed. This field is purely
    // informative for the user, the actual implementation may vary.
    enum pl_option_type type;

    // Minimum/maximum value ranges for numeric options (int / float)
    // If both are 0.0, these limits are disabled/ignored.
    float min, max;

    // If true, this option is considered deprecated and may be removed
    // in the future.
    bool deprecated;

    // If true, this option is considered a 'preset' (read-only), which can
    // be loaded but not saved. (The equivalent underlying options this preset
    // corresponds to will be saved instead)
    bool preset;

    // Internal implementation details (for parsing/saving), opaque to user
    const void *priv;
};

// A list of options, terminated by {0} for convenience
PL_API extern const struct pl_opt_t pl_option_list[];
PL_API extern const int pl_option_count; // excluding terminating {0}

// Returns the `pl_option` associated with a given key, or NULL
PL_API pl_opt pl_find_option(const char *key);

PL_API_END

#endif // LIBPLACEBO_OPTIONS_H_
