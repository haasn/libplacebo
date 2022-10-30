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
    REQUIRE(is_null(buf));
    pl_str_append_asprintf(tmp, &buf, "%.*s", PL_STR_FMT(test));
    REQUIRE(pl_str_equals(buf, test));

    REQUIRE(pl_strchr(null, ' ') < 0);
    REQUIRE(pl_strspn(null, " ") == 0);
    REQUIRE(pl_strcspn(null, " ") == 0);
    REQUIRE(is_null(pl_str_strip(null)));

    REQUIRE(pl_strchr(test, 's') == 2);
    REQUIRE(pl_strspn(test, "et") == 2);
    REQUIRE(pl_strcspn(test, "xs") == 2);

    REQUIRE(is_null(pl_str_take(null, 10)));
    REQUIRE(is_empty(pl_str_take(test, 0)));
    REQUIRE(is_null(pl_str_drop(null, 10)));
    REQUIRE(is_null(pl_str_drop(test, test.len)));
    REQUIRE(pl_str_equals(pl_str_drop(test, 0), test));

    REQUIRE(pl_str_find(null, test) < 0);
    REQUIRE(pl_str_find(null, null) == 0);
    REQUIRE(pl_str_find(test, null) == 0);
    REQUIRE(pl_str_find(test, test) == 0);

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

    REQUIRE(pl_str_parse_float(pl_str0("1.3984"), &f) && feq(f, 1.3984f, 1e-8));
    REQUIRE(pl_str_parse_float(pl_str0("-8.9100083"), &f) && feq(f, -8.9100083f, 1e-8));
    REQUIRE(pl_str_parse_float(pl_str0("-0"), &f) && feq(f, 0.0f, 1e-8));
    REQUIRE(pl_str_parse_float(pl_str0("-3.14e20"), &f) && feq(f, -3.14e20f, 1e-8));
    REQUIRE(pl_str_parse_float(pl_str0("0.5e-5"), &f) && feq(f, 0.5e-5f, 1e-8));
    REQUIRE(pl_str_parse_float(pl_str0("0.5e+5"), &f) && feq(f, 0.5e+5f, 1e-8));
    REQUIRE(pl_str_parse_int(pl_str0("64239"), &i) && i == 64239);
    REQUIRE(pl_str_parse_int(pl_str0("-102"), &i) && i == -102);
    REQUIRE(pl_str_parse_int(pl_str0("+1"), &i) && i == 1);
    REQUIRE(pl_str_parse_int(pl_str0("-0"), &i) && i == 0);
    REQUIRE(pl_str_parse_uint(pl_str0("64239"), &u) && u == 64239);
    REQUIRE(pl_str_parse_uint(pl_str0("+1"), &u) && u == 1);
    REQUIRE(pl_str_parse_int64(pl_str0("9223372036854775799"), &i64) && i64 == 9223372036854775799LL);
    REQUIRE(pl_str_parse_int64(pl_str0("-9223372036854775799"), &i64) && i64 == -9223372036854775799LL);
    REQUIRE(pl_str_parse_uint64(pl_str0("18446744073709551609"), &u64) && u64 == 18446744073709551609LLU);
    REQUIRE(!pl_str_parse_float(null, &f));
    REQUIRE(!pl_str_parse_float(test, &f));
    REQUIRE(!pl_str_parse_float(empty, &f));
    REQUIRE(!pl_str_parse_int(null, &i));
    REQUIRE(!pl_str_parse_int(test, &i));
    REQUIRE(!pl_str_parse_int(empty, &i));
    REQUIRE(!pl_str_parse_uint(null, &u));
    REQUIRE(!pl_str_parse_uint(test, &u));
    REQUIRE(!pl_str_parse_uint(empty, &u));

    pl_free(tmp);
    return 0;
}
