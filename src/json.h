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

#pragma once

#include "common.h"

#include "../3rdparty/json.h"

static inline const char *json_errstr(size_t error)
{
    switch (error) {
    case json_parse_error_expected_comma_or_closing_bracket:
        return "expected a comma or a closing bracket";
    case json_parse_error_expected_colon:
        return "expected a colon separating a name and value";
    case json_parse_error_expected_opening_quote:
        return "expected a string, beginning with a quote";
    case json_parse_error_invalid_string_escape_sequence:
        return "invalid escape sequence in string";
    case json_parse_error_invalid_number_format:
        return "invalid number format";
    case json_parse_error_invalid_value:
        return "invalid value";
    case json_parse_error_premature_end_of_buffer:
        return "unexpected end of input";
    case json_parse_error_invalid_string:
        return "malformed string";
    case json_parse_error_allocator_failed:
        return "out of memory";
    case json_parse_error_unexpected_trailing_characters:
        return "unexpected trailing characters";
    case json_parse_error_recursion:
        return "nested too deeply";
    default:
        return "unknown error";
    }
}

static inline pl_str json_str(const struct json_string_s *str)
{
    return (pl_str) { (uint8_t *) str->string, str->string_size };
}
