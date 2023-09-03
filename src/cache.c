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

#include <stdio.h>
#include <locale.h>

#include "common.h"
#include "cache.h"
#include "log.h"
#include "pl_thread.h"

const struct pl_cache_params pl_cache_default_params = {0};

struct priv {
    pl_log log;
    pl_mutex lock;
    PL_ARRAY(pl_cache_obj) objects;
    size_t total_size;
    size_t max_total_size;
    size_t max_object_size;
};

int pl_cache_objects(pl_cache cache)
{
    if (!cache)
        return 0;

    struct priv *p = PL_PRIV(cache);
    pl_mutex_lock(&p->lock);
    int num = p->objects.num;
    pl_mutex_unlock(&p->lock);
    return num;
}

size_t pl_cache_size(pl_cache cache)
{
    if (!cache)
        return 0;

    struct priv *p = PL_PRIV(cache);
    pl_mutex_lock(&p->lock);
    size_t size = p->total_size;
    pl_mutex_unlock(&p->lock);
    return size;
}

pl_cache pl_cache_create(const struct pl_cache_params *params)
{
    struct pl_cache_t *cache = pl_zalloc_obj(NULL, cache, struct priv);
    struct priv *p = PL_PRIV(cache);
    pl_mutex_init(&p->lock);
    if (params) {
        cache->params = *params;
        p->log = params->log;
    }

    p->max_object_size = PL_DEF(cache->params.max_object_size, SIZE_MAX);
    p->max_total_size  = PL_DEF(cache->params.max_total_size, SIZE_MAX);
    p->max_object_size = PL_MIN(p->max_object_size, p->max_total_size);
    return cache;
}

static void remove_obj(pl_cache cache, pl_cache_obj obj)
{
    struct priv *p = PL_PRIV(cache);

    p->total_size -= obj.size;
    if (obj.free)
        obj.free(obj.data);
}

void pl_cache_destroy(pl_cache *pcache)
{
    pl_cache cache = *pcache;
    if (!cache)
         return;

    struct priv *p = PL_PRIV(cache);
    for (int i = 0; i < p->objects.num; i++)
        remove_obj(cache, p->objects.elem[i]);

    pl_assert(p->total_size == 0);
    pl_mutex_destroy(&p->lock);
    pl_free((void *) cache);
    *pcache = NULL;
}

void pl_cache_reset(pl_cache cache)
{
    if (!cache)
        return;

    struct priv *p = PL_PRIV(cache);
    pl_mutex_lock(&p->lock);
    for (int i = 0; i < p->objects.num; i++)
        remove_obj(cache, p->objects.elem[i]);
    p->objects.num = 0;
    pl_assert(p->total_size == 0);
    pl_mutex_unlock(&p->lock);
}

static bool try_set(pl_cache cache, pl_cache_obj obj)
{
    struct priv *p = PL_PRIV(cache);

    // Remove any existing entry with this key
    for (int i = p->objects.num - 1; i >= 0; i--) {
        pl_cache_obj prev = p->objects.elem[i];
        if (prev.key == obj.key) {
            remove_obj(cache, prev);
            PL_ARRAY_REMOVE_AT(p->objects, i);
            break;
        }
    }

    if (!obj.size)
        return true;

    if (obj.size > p->max_object_size) {
        PL_DEBUG(p, "Object 0x%"PRIx64" (size %zu) exceeds max size %zu, discarding",
                 obj.key, obj.size, cache->params.max_object_size);
        return false;
    }

    // Make space by deleting old objects
    while (p->total_size + obj.size > p->max_total_size) {
        pl_assert(p->objects.num);
        remove_obj(cache, p->objects.elem[0]);
        PL_ARRAY_REMOVE_AT(p->objects, 0);
    }

    if (!obj.free) {
        obj.data = pl_memdup(NULL, obj.data, obj.size);
        obj.free = pl_free;
    }

    PL_ARRAY_APPEND((void *) cache, p->objects, obj);
    p->total_size += obj.size;
    return true;
}

static pl_cache_obj strip_obj(pl_cache_obj obj)
{
    return (pl_cache_obj) { .key = obj.key };
}

bool pl_cache_try_set(pl_cache cache, pl_cache_obj *pobj)
{
    if (!cache)
        return false;

    pl_cache_obj obj = *pobj;
    struct priv *p = PL_PRIV(cache);
    pl_mutex_lock(&p->lock);
    bool ok = try_set(cache, obj);
    pl_mutex_unlock(&p->lock);
    if (ok) {
        *pobj = strip_obj(obj); // ownership transfers, clear ptr
    } else {
        obj = strip_obj(obj); // ownership remains with caller, clear copy
    }
    if (cache->params.set)
        cache->params.set(cache->params.priv, obj);
    return ok;
}

void pl_cache_set(pl_cache cache, pl_cache_obj *obj)
{
    if (!pl_cache_try_set(cache, obj)) {
        if (obj->free)
            obj->free(obj->data);
        *obj = (pl_cache_obj) { .key = obj->key };
    }
}

static void noop(void *ignored)
{
    (void) ignored;
}

bool pl_cache_get(pl_cache cache, pl_cache_obj *out_obj)
{
    const uint64_t key = out_obj->key;
    if (!cache)
        goto fail;

    struct priv *p = PL_PRIV(cache);
    pl_mutex_lock(&p->lock);

    // Search backwards to prioritize recently added entries
    for (int i = p->objects.num - 1; i >= 0; i--) {
        pl_cache_obj obj = p->objects.elem[i];
        if (obj.key == key) {
            PL_ARRAY_REMOVE_AT(p->objects, i);
            p->total_size -= obj.size;
            pl_mutex_unlock(&p->lock);
            pl_assert(obj.free);
            *out_obj = obj;
            return true;
        }
    }

    pl_mutex_unlock(&p->lock);
    if (!cache->params.get)
        goto fail;

    pl_cache_obj obj = cache->params.get(cache->params.priv, key);
    if (!obj.size)
        goto fail;

    // Sanitize object
    obj.key = key;
    obj.free = PL_DEF(obj.free, noop);
    *out_obj = obj;
    return true;

fail:
    *out_obj = (pl_cache_obj) { .key = key };
    return false;
}

void pl_cache_iterate(pl_cache cache,
                      void (*cb)(void *priv, pl_cache_obj obj),
                      void *priv)
{
    if (!cache)
        return;

    struct priv *p = PL_PRIV(cache);
    pl_mutex_lock(&p->lock);
    for (int i = 0; i < p->objects.num; i++)
        cb(priv, p->objects.elem[i]);
    pl_mutex_unlock(&p->lock);
}

// --- Saving/loading

#define CACHE_MAGIC   "pl_cache"
#define CACHE_VERSION 1
#define PAD_ALIGN(x)  (PL_ALIGN2(x, sizeof(uint32_t)) - (x))

struct __attribute__((__packed__)) cache_header {
    char magic[8];
    uint32_t version;
    uint32_t num_entries;
};

struct __attribute__((__packed__)) cache_entry {
    uint64_t key;
    uint64_t size;
    uint64_t hash;
};

int pl_cache_save_ex(pl_cache cache,
                     void (*write)(void *priv, size_t size, const void *ptr),
                     void *priv)
{
    if (!cache)
        return 0;

    struct priv *p = PL_PRIV(cache);
    pl_mutex_lock(&p->lock);
    pl_clock_t start = pl_clock_now();

    const int num_objects = p->objects.num;
    const size_t saved_bytes = p->total_size;
    write(priv, sizeof(struct cache_header), &(struct cache_header) {
        .magic       = CACHE_MAGIC,
        .version     = CACHE_VERSION,
        .num_entries = num_objects,
    });

    for (int i = 0; i < num_objects; i++) {
        pl_cache_obj obj = p->objects.elem[i];
        write(priv, sizeof(struct cache_entry), &(struct cache_entry) {
            .key  = obj.key,
            .size = obj.size,
            .hash = pl_mem_hash(obj.data, obj.size),
        });
        static const uint8_t padding[PAD_ALIGN(1)] = {0};
        write(priv, obj.size, obj.data);
        write(priv, PAD_ALIGN(obj.size), padding);
    }

    pl_mutex_unlock(&p->lock);
    pl_log_cpu_time(p->log, start, pl_clock_now(), "saving cache");
    if (num_objects)
        PL_DEBUG(p, "Saved %d objects, totalling %zu bytes", num_objects, saved_bytes);

    return num_objects;
}

int pl_cache_load_ex(pl_cache cache,
                     bool (*read)(void *priv, size_t size, void *ptr),
                     void *priv)
{
    if (!cache)
        return 0;

    struct priv *p = PL_PRIV(cache);
    struct cache_header header;
    if (!read(priv, sizeof(header), &header)) {
        PL_ERR(p, "Failed loading cache: file seems empty or truncated");
        return -1;
    }
    if (memcmp(header.magic, CACHE_MAGIC, sizeof(header.magic)) != 0) {
        PL_ERR(p, "Failed loading cache: invalid magic bytes");
        return -1;
    }
    if (header.version != CACHE_VERSION) {
        PL_INFO(p, "Failed loading cache: wrong version... skipping");
        return 0;
    }

    int num_loaded = 0;
    size_t loaded_bytes = 0;
    pl_mutex_lock(&p->lock);
    pl_clock_t start = pl_clock_now();

    for (int i = 0; i < header.num_entries; i++) {
        struct cache_entry entry;
        void *buf = NULL;

        uint8_t padding[PAD_ALIGN(1)];
        if (!read(priv, sizeof(entry), &entry) ||
            !(buf = pl_alloc(NULL, entry.size)) ||
            !read(priv, entry.size, buf) ||
            !read(priv, PAD_ALIGN(entry.size), padding))
        {
            PL_WARN(p, "Cache seems truncated, missing objects.. ignoring rest");
            pl_free(buf);
            return num_loaded;
        }

        uint64_t checksum = pl_mem_hash(buf, entry.size);
        if (checksum != entry.hash) {
            PL_WARN(p, "Cache entry seems corrupt, checksum mismatch.. ignoring rest");
            pl_free(buf);
            return num_loaded;
        }

        bool ok = try_set(cache, (pl_cache_obj) {
            .key  = entry.key,
            .size = entry.size,
            .data = buf,
            .free = pl_free,
        });
        if (ok) {
            num_loaded++;
            loaded_bytes += entry.size;
        } else {
            pl_free(buf);
        }
    }

    pl_mutex_unlock(&p->lock);
    pl_log_cpu_time(p->log, start, pl_clock_now(), "loading cache");
    if (num_loaded)
        PL_DEBUG(p, "Loaded %d objects, totalling %zu bytes", num_loaded, loaded_bytes);

    return num_loaded;
}

// Save/load wrappers

struct ptr_ctx {
    uint8_t *data; // base pointer
    size_t size;   // total size
    size_t pos;    // read/write index
};

static void write_ptr(void *priv, size_t size, const void *ptr)
{
    struct ptr_ctx *ctx = priv;
    size_t end = PL_MIN(ctx->pos + size, ctx->size);
    if (end > ctx->pos)
        memcpy(ctx->data + ctx->pos, ptr, end - ctx->pos);
    ctx->pos += size;
}

static bool read_ptr(void *priv, size_t size, void *ptr)
{
    struct ptr_ctx *ctx = priv;
    if (ctx->pos + size > ctx->size)
        return false;
    memcpy(ptr, ctx->data + ctx->pos, size);
    ctx->pos += size;
    return true;
}

size_t pl_cache_save(pl_cache cache, uint8_t *data, size_t size)
{
    struct ptr_ctx ctx = { data, size };
    pl_cache_save_ex(cache, write_ptr, &ctx);
    return ctx.pos;
}

int pl_cache_load(pl_cache cache, const uint8_t *data, size_t size)
{
    return pl_cache_load_ex(cache, read_ptr, &(struct ptr_ctx) {
        .data = (uint8_t *) data,
        .size = size,
    });
}
