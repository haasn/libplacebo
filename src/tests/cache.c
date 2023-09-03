#include "tests.h"

#include <libplacebo/cache.h>

// Returns "foo" for even keys, "bar" for odd
static pl_cache_obj lookup_foobar(void *priv, uint64_t key)
{
    return (pl_cache_obj) {
        .key = 0xFFFF, // test key sanity
        .data = (key & 1) ? "bar" : "foo",
        .size = 3,
    };
}

static void update_count(void *priv, pl_cache_obj obj)
{
    int *count = priv;
    *count += obj.size ? 1 : -1;
}

int main()
{
    pl_log log = pl_test_logger();
    pl_cache test = pl_cache_create(pl_cache_params(
        .log             = log,
        .max_object_size = 16,
        .max_total_size  = 32,
    ));

    pl_cache_obj obj1 = { .key  = 0x1, .data = "abc",  .size = 3 };
    pl_cache_obj obj2 = { .key  = 0x2, .data = "de",   .size = 2 };
    pl_cache_obj obj3 = { .key  = 0x3, .data = "xyzw", .size = 4 };

    REQUIRE(pl_cache_try_set(test, &obj1));
    REQUIRE(pl_cache_try_set(test, &obj2));
    REQUIRE(pl_cache_try_set(test, &obj3));
    REQUIRE_CMP(pl_cache_size(test), ==, 9, "zu");
    REQUIRE_CMP(pl_cache_objects(test), ==, 3, "d");
    REQUIRE(pl_cache_try_set(test, &obj2)); // delete 0x2
    REQUIRE_CMP(pl_cache_size(test), ==, 7, "zu");
    REQUIRE_CMP(pl_cache_objects(test), ==, 2, "d");

    REQUIRE(pl_cache_get(test, &obj1));
    REQUIRE(!pl_cache_get(test, &obj2));
    REQUIRE(pl_cache_get(test, &obj3));
    REQUIRE_CMP(pl_cache_size(test), ==, 0, "zu");
    REQUIRE_CMP(pl_cache_objects(test), ==, 0, "d");
    REQUIRE_MEMEQ(obj1.data, "abc", 3);
    REQUIRE_MEMEQ(obj3.data, "xyzw", 4);

    // Re-insert removed objects (in reversed order)
    REQUIRE(pl_cache_try_set(test, &obj3));
    REQUIRE(pl_cache_try_set(test, &obj1));
    REQUIRE_CMP(pl_cache_size(test), ==, 7, "zu");
    REQUIRE_CMP(pl_cache_objects(test), ==, 2, "d");

    static const uint8_t ref[] = {
        'p', 'l', '_', 'c',         // cache magic
        'a', 'c', 'h', 'e',
        0x1, 0, 0, 0,               // cache version
        0x2, 0, 0, 0,               // number of objects
        // object 3
        0x3, 0, 0, 0, 0, 0, 0, 0,   // key
        0x4, 0, 0, 0, 0, 0, 0, 0,   // size
        0x17, 0x11, 0x47, 0x5e,     // checksum
        0x4e, 0x88, 0x18, 0xec,
        'x', 'y', 'z', 'w',         // data
        // object 1
        0x1, 0, 0, 0, 0, 0, 0, 0,   // key
        0x3, 0, 0, 0, 0, 0, 0, 0,   // size
        0x77, 0x2d, 0x2e, 0x8a,     // checksum
        0x40, 0x4d, 0x20, 0x3a,
        'a', 'b', 'c',              // data
        0,                          // padding
    };

    uint8_t data[100];
    pl_static_assert(sizeof(data) >= sizeof(ref));
    REQUIRE_CMP(pl_cache_save(test, data, sizeof(data)), ==, sizeof(ref), "zu");
    REQUIRE_MEMEQ(data, ref, sizeof(ref));

    pl_cache test2 = pl_cache_create(pl_cache_params( .log = log ));
    REQUIRE_CMP(pl_cache_load(test2, data, sizeof(data)), ==, 2, "d");
    REQUIRE_CMP(pl_cache_size(test2), ==, 7, "zu");
    REQUIRE_CMP(pl_cache_save(test2, NULL, 0), ==, sizeof(ref), "zu");
    REQUIRE_CMP(pl_cache_save(test2, data, sizeof(data)), ==, sizeof(ref), "zu");
    REQUIRE_MEMEQ(data, ref, sizeof(ref));

    // Test loading invalid data
    REQUIRE_CMP(pl_cache_load(test2, ref, 0),   <, 0, "d"); // empty file
    REQUIRE_CMP(pl_cache_load(test2, ref, 5),   <, 0, "d"); // truncated header
    REQUIRE_CMP(pl_cache_load(test2, ref, 64), ==, 1, "d"); // truncated object data
    data[sizeof(ref) - 2] = 'X'; // corrupt data
    REQUIRE_CMP(pl_cache_load(test2, data, sizeof(ref)), ==, 1, "d"); // bad checksum
    pl_cache_destroy(&test2);

    // Inserting too large object should fail
    uint8_t zero[32] = {0};
    pl_cache_obj obj4 = { .key = 0x4, .data = zero, .size = 32 };
    REQUIRE(!pl_cache_try_set(test, &obj4));
    REQUIRE(!pl_cache_get(test, &obj4));
    REQUIRE_CMP(pl_cache_size(test), ==, 7, "zu");
    REQUIRE_CMP(pl_cache_objects(test), ==, 2, "d");

    // Inserting 16-byte object should succeed, and not purge old entries
    obj4 = (pl_cache_obj) { .key = 0x4, .data = zero, .size = 16 };
    REQUIRE(pl_cache_try_set(test, &obj4));
    REQUIRE_CMP(pl_cache_size(test), ==, 23, "zu");
    REQUIRE_CMP(pl_cache_objects(test), ==, 3, "d");
    REQUIRE(pl_cache_get(test, &obj1));
    REQUIRE(pl_cache_get(test, &obj3));
    REQUIRE(pl_cache_get(test, &obj4));
    pl_cache_set(test, &obj1);
    pl_cache_set(test, &obj3);
    pl_cache_set(test, &obj4);
    REQUIRE_CMP(pl_cache_size(test), ==, 23, "zu");
    REQUIRE_CMP(pl_cache_objects(test), ==, 3, "d");

    // Inserting another 10-byte object should purge entry 0x1
    pl_cache_obj obj5 = { .key = 0x5, .data = zero, .size = 10 };
    REQUIRE(pl_cache_try_set(test, &obj5));
    REQUIRE_CMP(pl_cache_size(test), ==, 30, "zu");
    REQUIRE_CMP(pl_cache_objects(test), ==, 3, "d");
    REQUIRE(!pl_cache_get(test, &obj1));
    REQUIRE(pl_cache_get(test, &obj3));
    REQUIRE(pl_cache_get(test, &obj4));
    REQUIRE(pl_cache_get(test, &obj5));
    pl_cache_set(test, &obj3);
    pl_cache_set(test, &obj4);
    pl_cache_set(test, &obj5);
    REQUIRE_CMP(pl_cache_size(test), ==, 30, "zu");
    REQUIRE_CMP(pl_cache_objects(test), ==, 3, "d");

    // Inserting final 6-byte object should purge entry 0x3
    pl_cache_obj obj6 = { .key = 0x6, .data = zero, .size = 6 };
    REQUIRE(pl_cache_try_set(test, &obj6));
    REQUIRE_CMP(pl_cache_size(test), ==, 32, "zu");
    REQUIRE_CMP(pl_cache_objects(test), ==, 3, "d");
    REQUIRE(!pl_cache_get(test, &obj3));
    REQUIRE(pl_cache_get(test, &obj4));
    REQUIRE(pl_cache_get(test, &obj5));
    REQUIRE(pl_cache_get(test, &obj6));
    REQUIRE_CMP(pl_cache_size(test), ==, 0, "zu");
    REQUIRE_CMP(pl_cache_objects(test), ==, 0, "d");
    pl_cache_obj_free(&obj4);
    pl_cache_obj_free(&obj5);
    pl_cache_obj_free(&obj6);

    // Test callback API
    int num_objects = 0;
    test2 = pl_cache_create(pl_cache_params(
        .get  = lookup_foobar,
        .set  = update_count,
        .priv = &num_objects,
    ));

    REQUIRE(pl_cache_get(test2, &obj1));
    REQUIRE_CMP(obj1.key, ==, 0x1, PRIu64);
    REQUIRE_CMP(obj1.size, ==, 3, "zu");
    REQUIRE_MEMEQ(obj1.data, "bar", 3);
    REQUIRE(pl_cache_get(test2, &obj2));
    REQUIRE_CMP(obj2.key, ==, 0x2, PRIu64);
    REQUIRE_CMP(obj2.size, ==, 3, "zu");
    REQUIRE_MEMEQ(obj2.data, "foo", 3);
    REQUIRE_CMP(pl_cache_objects(test2), ==, 0, "d");
    REQUIRE_CMP(num_objects, ==, 0, "d");
    REQUIRE(pl_cache_try_set(test2, &obj1));
    REQUIRE(pl_cache_try_set(test2, &obj2));
    REQUIRE(pl_cache_try_set(test2, &(pl_cache_obj) { .key = 0x789, .data = "abcde", .size = 5 }));
    REQUIRE_CMP(pl_cache_objects(test2), ==, 3, "d");
    REQUIRE_CMP(num_objects, ==, 3, "d");
    REQUIRE(pl_cache_try_set(test2, &obj1));
    REQUIRE(pl_cache_try_set(test2, &obj2));
    REQUIRE_CMP(pl_cache_objects(test2), ==, 1, "d");
    REQUIRE_CMP(num_objects, ==, 1, "d");
    pl_cache_destroy(&test2);

    pl_cache_destroy(&test);
    pl_log_destroy(&log);
    return 0;
}
