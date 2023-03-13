#include "tests.h"

static const pl_str null = {0};
static const pl_str test = PL_STR0("test");
static const pl_str empty = PL_STR0("");

static inline bool is_null(pl_str str)
{
    return !str.len && !str.buf;
}

static inline bool is_empty(pl_str str)
{
    return !str.len;
}

int main()
{
    void *tmp = pl_tmp(NULL);

    REQUIRE(is_null(pl_str0(NULL)));
    REQUIRE(is_null(pl_strdup(tmp, null)));
    char *empty0 = pl_strdup0(tmp, null);
    REQUIRE(empty0 && !empty0[0]);
    REQUIRE(pl_str_equals0(empty, empty0));

    pl_str buf = {0};
    pl_str_append(tmp, &buf, null);
    REQUIRE(is_empty(buf));
    pl_str_append_asprintf(tmp, &buf, "%.*s", PL_STR_FMT(test));
    REQUIRE(pl_str_equals(buf, test));

    pl_str_append_asprintf_c(tmp, &buf, "%d %f %lld %zu %.*sx %hx",
        1, 1.0f, 0xFFll, (size_t) 0, PL_STR_FMT(empty), (unsigned short) 0xCAFE);
    REQUIRE(pl_str_equals0(buf, "test1 1.00000000000000000000 255 0 x cafe"));

    REQUIRE_CMP(pl_strchr(null, ' '), <, 0, "d");
    REQUIRE_CMP((int) pl_strspn(null, " "), ==, 0, "d");
    REQUIRE_CMP((int) pl_strcspn(null, " "), ==, 0, "d");
    REQUIRE(is_null(pl_str_strip(null)));

    REQUIRE_CMP(pl_strchr(test, 's'), ==, 2, "d");
    REQUIRE_CMP((int) pl_strspn(test, "et"), ==, 2, "d");
    REQUIRE_CMP((int) pl_strcspn(test, "xs"), ==, 2, "d");

    REQUIRE(is_null(pl_str_take(null, 10)));
    REQUIRE(is_empty(pl_str_take(test, 0)));
    REQUIRE(is_null(pl_str_drop(null, 10)));
    REQUIRE(is_null(pl_str_drop(test, test.len)));
    REQUIRE(pl_str_equals(pl_str_drop(test, 0), test));

    REQUIRE_CMP(pl_str_find(null, test), <, 0, "d");
    REQUIRE_CMP(pl_str_find(null, null), ==, 0, "d");
    REQUIRE_CMP(pl_str_find(test, null), ==, 0, "d");
    REQUIRE_CMP(pl_str_find(test, test), ==, 0, "d");

    pl_str rest;
    REQUIRE(is_null(pl_str_split_char(null, ' ', &rest)) && is_null(rest));
    REQUIRE(is_null(pl_str_split_str(null, test, &rest)) && is_null(rest));
    REQUIRE(is_empty(pl_str_split_str(test, test, &rest)) && is_empty(rest));
    REQUIRE(is_null(pl_str_getline(null, &rest)) && is_null(rest));

    pl_str right, left = pl_str_split_char(pl_str0("left right"), ' ', &right);
    REQUIRE(pl_str_equals0(left, "left"));
    REQUIRE(pl_str_equals0(right, "right"));

    left = pl_str_split_str0(pl_str0("leftTESTright"), "TEST", &right);
    REQUIRE(pl_str_equals0(left, "left"));
    REQUIRE(pl_str_equals0(right, "right"));

    pl_str out;
    REQUIRE(pl_str_decode_hex(tmp, null, &out) && is_empty(out));
    REQUIRE(!pl_str_decode_hex(tmp, pl_str0("invalid"), &out));

    REQUIRE(pl_str_equals(null, null));
    REQUIRE(pl_str_equals(null, empty));
    REQUIRE(pl_str_startswith(null, null));
    REQUIRE(pl_str_startswith(test, null));
    REQUIRE(pl_str_startswith(test, test));
    REQUIRE(pl_str_endswith(null, null));
    REQUIRE(pl_str_endswith(test, null));
    REQUIRE(pl_str_endswith(test, test));

    float f;
    int i;
    unsigned u;
    int64_t i64;
    uint64_t u64;

    REQUIRE(pl_str_parse_float(pl_str0("1.3984"), &f));     REQUIRE_FEQ(f, 1.3984f, 1e-8);
    REQUIRE(pl_str_parse_float(pl_str0("-8.9100083"), &f)); REQUIRE_FEQ(f, -8.9100083f, 1e-8);
    REQUIRE(pl_str_parse_float(pl_str0("-0"), &f));         REQUIRE_FEQ(f, 0.0f, 1e-8);
    REQUIRE(pl_str_parse_float(pl_str0("-3.14e20"), &f));   REQUIRE_FEQ(f, -3.14e20f, 1e-8);
    REQUIRE(pl_str_parse_float(pl_str0("0.5e-5"), &f));     REQUIRE_FEQ(f, 0.5e-5f, 1e-8);
    REQUIRE(pl_str_parse_float(pl_str0("0.5e+5"), &f));     REQUIRE_FEQ(f, 0.5e+5f, 1e-8);
    REQUIRE(pl_str_parse_int(pl_str0("64239"), &i));        REQUIRE_CMP(i, ==, 64239, "d");
    REQUIRE(pl_str_parse_int(pl_str0("-102"), &i));         REQUIRE_CMP(i, ==, -102, "d");
    REQUIRE(pl_str_parse_int(pl_str0("+1"), &i));           REQUIRE_CMP(i, ==, 1, "d");
    REQUIRE(pl_str_parse_int(pl_str0("-0"), &i));           REQUIRE_CMP(i, ==, 0, "d");
    REQUIRE(pl_str_parse_uint(pl_str0("64239"), &u));       REQUIRE_CMP(u, ==, 64239, "u");
    REQUIRE(pl_str_parse_uint(pl_str0("+1"), &u));          REQUIRE_CMP(u, ==, 1, "u");
    REQUIRE(pl_str_parse_int64(pl_str0("9223372036854775799"), &i64));
    REQUIRE_CMP(i64, ==, 9223372036854775799LL, PRIi64);
    REQUIRE(pl_str_parse_int64(pl_str0("-9223372036854775799"), &i64));
    REQUIRE_CMP(i64, ==, -9223372036854775799LL, PRIi64);
    REQUIRE(pl_str_parse_uint64(pl_str0("18446744073709551609"), &u64));
    REQUIRE_CMP(u64, ==, 18446744073709551609LLU, PRIu64);
    REQUIRE(!pl_str_parse_float(null, &f));
    REQUIRE(!pl_str_parse_float(test, &f));
    REQUIRE(!pl_str_parse_float(empty, &f));
    REQUIRE(!pl_str_parse_int(null, &i));
    REQUIRE(!pl_str_parse_int(test, &i));
    REQUIRE(!pl_str_parse_int(empty, &i));
    REQUIRE(!pl_str_parse_uint(null, &u));
    REQUIRE(!pl_str_parse_uint(test, &u));
    REQUIRE(!pl_str_parse_uint(empty, &u));

    pl_str_builder builder = pl_str_builder_alloc(tmp);
    pl_str_builder_const_str(builder, "hello");
    pl_str_builder_str(builder, pl_str0("world"));
    pl_str res = pl_str_builder_exec(builder);
    REQUIRE(pl_str_equals0(res, "helloworld"));

    pl_str_builder_reset(builder);
    pl_str_builder_printf_c(builder, "foo %d bar %u bat %s baz %lld",
            123, 56u, "quack", 0xDEADBEEFll);
    pl_str_builder_printf_c(builder, " %.*s", PL_STR_FMT(pl_str0("test123")));
    res = pl_str_builder_exec(builder);
    REQUIRE(pl_str_equals0(res, "foo 123 bar 56 bat quack baz 3735928559 test123"));

    pl_free(tmp);
    return 0;
}
