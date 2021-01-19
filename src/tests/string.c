#include "tests.h"

static const pl_str null = {0};
static const pl_str test = PL_STR0("test");

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
    char *empty = pl_strdup0(tmp, null);
    REQUIRE(empty && !empty[0]);

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
    REQUIRE(pl_str_startswith(null, null));
    REQUIRE(pl_str_startswith(test, null));
    REQUIRE(pl_str_startswith(test, test));
    REQUIRE(pl_str_endswith(null, null));
    REQUIRE(pl_str_endswith(test, null));
    REQUIRE(pl_str_endswith(test, test));

    pl_free(tmp);
    return 0;
}
