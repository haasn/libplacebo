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

enum {
    KEY1 = 0x9c65575f419288f5,
    KEY2 = 0x92da969be9b88086,
    KEY3 = 0x7fcb62540b00bc8b,
    KEY4 = 0x46c60ec11af9dde3,
    KEY5 = 0xcb6760b98ece2477,
    KEY6 = 0xf37dc72b7f9e5c88,
    KEY7 = 0x30c18c962d82e5f5,
};

int main()
{
    pl_log log = pl_test_logger();
    pl_cache test = pl_cache_create(pl_cache_params(
        .log             = log,
        .max_object_size = 16,
        .max_total_size  = 32,
    ));

    pl_cache_obj obj1 = { .key  = KEY1, .data = "abc",  .size = 3 };
    pl_cache_obj obj2 = { .key  = KEY2, .data = "de",   .size = 2 };
    pl_cache_obj obj3 = { .key  = KEY3, .data = "xyzw", .size = 4 };

    REQUIRE(pl_cache_try_set(test, &obj1));
    REQUIRE(pl_cache_try_set(test, &obj2));
    REQUIRE(pl_cache_try_set(test, &obj3));
    REQUIRE_CMP(pl_cache_size(test), ==, 9, "zu");
    REQUIRE_CMP(pl_cache_objects(test), ==, 3, "d");
    REQUIRE(pl_cache_try_set(test, &obj2)); // delete KEY2
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

    uint8_t ref[72];
    memset(ref, 0xbe, sizeof(ref));
    uint8_t *refp = ref;

#define PAD_ALIGN(x) PL_ALIGN2(x, sizeof(uint32_t))
#define W(type, ...)                                    \
    do {                                                \
        size_t sz = sizeof((type){__VA_ARGS__});        \
        pl_assert(ref + sizeof(ref) - refp >= sz);      \
        memcpy(refp, &(type){__VA_ARGS__}, sz);         \
        refp += sz;                                     \
        size_t pad_sz = PAD_ALIGN(sz) - sz;             \
        pl_assert(ref + sizeof(ref) - refp >= pad_sz);  \
        memcpy(refp, &(char[PAD_ALIGN(1)]){0}, pad_sz); \
        refp += pad_sz;                                 \
    } while (0)

    W(char[], 'p', 'l', '_', 'c', 'a', 'c', 'h', 'e');  // cache magic
    W(uint32_t, 1);                                     // cache version
    W(uint32_t, 2);                                     // number of objects

    // object 3
    W(uint64_t, KEY3);                // key
    W(uint64_t, 4);                   // size
#ifdef PL_HAVE_XXHASH
    W(uint64_t, 0xd43612ef3fbee8be);  // hash
#else
    W(uint64_t, 0xec18884e5e471117);  // hash
#endif
    W(char[], 'x', 'y', 'z', 'w');    // data

    // object 1
    W(uint64_t, KEY1);                // key
    W(uint64_t, 3);                   // size
#ifdef PL_HAVE_XXHASH
    W(uint64_t, 0x78af5f94892f3950);  // hash
#else
    W(uint64_t, 0x3a204d408a2e2d77);  // hash
#endif
    W(char[], 'a', 'b', 'c');         // data

#undef W
#undef PAD_ALIGN

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
    pl_cache_obj obj4 = { .key = KEY4, .data = zero, .size = 32 };
    REQUIRE(!pl_cache_try_set(test, &obj4));
    REQUIRE(!pl_cache_get(test, &obj4));
    REQUIRE_CMP(pl_cache_size(test), ==, 7, "zu");
    REQUIRE_CMP(pl_cache_objects(test), ==, 2, "d");

    // Inserting 16-byte object should succeed, and not purge old entries
    obj4 = (pl_cache_obj) { .key = KEY4, .data = zero, .size = 16 };
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

    // Inserting another 10-byte object should purge entry KEY1
    pl_cache_obj obj5 = { .key = KEY5, .data = zero, .size = 10 };
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

    // Inserting final 6-byte object should purge entry KEY3
    pl_cache_obj obj6 = { .key = KEY6, .data = zero, .size = 6 };
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
    REQUIRE_CMP(obj1.key, ==, KEY1, PRIu64);
    REQUIRE_CMP(obj1.size, ==, 3, "zu");
    REQUIRE_MEMEQ(obj1.data, "bar", 3);
    REQUIRE(pl_cache_get(test2, &obj2));
    REQUIRE_CMP(obj2.key, ==, KEY2, PRIu64);
    REQUIRE_CMP(obj2.size, ==, 3, "zu");
    REQUIRE_MEMEQ(obj2.data, "foo", 3);
    REQUIRE_CMP(pl_cache_objects(test2), ==, 0, "d");
    REQUIRE_CMP(num_objects, ==, 0, "d");
    REQUIRE(pl_cache_try_set(test2, &obj1));
    REQUIRE(pl_cache_try_set(test2, &obj2));
    REQUIRE(pl_cache_try_set(test2, &(pl_cache_obj) { .key = KEY7, .data = "abcde", .size = 5 }));
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
