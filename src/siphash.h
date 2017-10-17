#pragma once

#include <stddef.h>
#include <stdint.h>

uint64_t siphash64(const uint8_t *in, const size_t inlen);
