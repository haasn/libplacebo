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

#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include "pl_string.h"


int pl_str_print_hex(char* buf, size_t len, unsigned short value)
{
    return snprintf(buf, len, "%x", value);
}

int pl_str_print_int(char* buf, size_t len, int value)
{
    return snprintf(buf, len, "%d", value);
}

int pl_str_print_uint(char* buf, size_t len, unsigned int value)
{
    return snprintf(buf, len, "%u", value);
}

int pl_str_print_int64(char* buf, size_t len, int64_t value)
{
    return snprintf(buf, len, "%" PRId64, value);
}

int pl_str_print_uint64(char* buf, size_t len, uint64_t value)
{
    return snprintf(buf, len, "%" PRIu64, value);
}

int pl_str_print_float(char* buf, size_t len, float value)
{
    return snprintf(buf, len, "%g", value);
}

int pl_str_print_double(char* buf, size_t len, double value)
{
    return snprintf(buf, len, "%g", value);
}




bool pl_str_parse_hex(pl_str str, short unsigned int* value)
{
    return pl_str_sscanf(str, "0x%x", value);
}

bool pl_str_parse_int(pl_str str, int* value)
{
    return pl_str_sscanf(str, "%d", value);
}

bool pl_str_parse_uint(pl_str str, unsigned int* value)
{
    return pl_str_sscanf(str, "%u", value);
}

bool pl_str_parse_int64(pl_str str, int64_t* value)
{
    return pl_str_sscanf(str, "%" PRId64, value);
}

bool pl_str_parse_uint64(pl_str str, uint64_t* value)
{
    return pl_str_sscanf(str, "%" PRIu64, value);
}

bool pl_str_parse_float(pl_str str, float* value)
{
    return pl_str_sscanf(str, "%g", value);
}

bool pl_str_parse_double(pl_str str, double* value)
{
    return pl_str_sscanf(str, "%g", value);
}

