/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015 Mikkel F. Jørgensen, dvide.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */


/*
 * This file implements a generic hash interface such that different
 * instances have the same name, but hidden from each other.
 * The interface maps the local names to a public specific type.
 *
 * This implementations implements a hash table with linear or quadratic
 * probing.
 */

#include <string.h>

#ifdef HASH_TABLE_IMPL
#error "cannot have multiple implementations in same compilation unit"
#endif
#define HASH_TABLE_IMPL
/* Open Addressing */
#define HT_OA

#if defined(_MSC_VER)
#pragma warning(disable: 4127) /* conditional expression is constant */
#endif

#include <stdlib.h>
#include <assert.h>

#ifndef HT_PROBE
#ifdef HT_PROBE_QUADRATIC
#define HT_PROBE(k, i, N) ((k + (i + i * i) / 2) & N)
#else
#define HT_PROBE(k, i, N) ((k + i) & N)
#endif
#endif

static int ht_init(hash_table_t *ht, size_t count)
{
    size_t buckets = 4;

    if ((HT_LOAD_FACTOR_FRAC) > 256 || (HT_LOAD_FACTOR_FRAC) < 1) {
        /*
         * 100% is bad but still the users choice.
         * 101% will never terminate insertion.
         */
        HT_PANIC("hash table failed with impossible load factor");
        return -1;
    }
    while (count > buckets * (HT_LOAD_FACTOR_FRAC) / 256) {
        buckets *= 2;
    }
    ht->table = calloc(buckets, sizeof(ht_item_t));
    if (ht->table == 0) {
        return -1;
    }
    ht->offsets = 0;
    ht->buckets = buckets;
    ht->count = 0;
    return 0;
}

static int ht_resize(hash_table_t *ht, size_t count)
{
    size_t i;
    hash_table_t ht2;
    ht_item_t *T = ht->table;
    void *item;

    if (count < ht->count) {
        count = ht->count;
    }
    if (ht_init(&ht2, count)) {
        return -1;
    }
    for (i = 0; i < ht->buckets; ++i) {
        item = T[i];
        if ((item && item != HT_DELETED)) {
            ht_insert(&ht2, ht_key(item), ht_key_len(item), item, ht_multi);
        }
    }
    ht_clear(ht);
    memcpy(ht, &ht2, sizeof(*ht));
    return 0;
}

static ht_item_t ht_insert(hash_table_t *ht,
        const void *key, size_t len, ht_item_t new_item, int mode)
{
    ht_item_t *T;
    size_t N, i, j, k;
    ht_item_t item, *vacant = 0;

    assert(new_item != HT_MISSING);
    assert(new_item != HT_DELETED);
    assert(new_item != HT_NOMEM);

    if (ht->count >= ht->buckets * (HT_LOAD_FACTOR_FRAC) / 256) {
        if (ht_resize(ht, ht->count * 2)) {
            HT_PANIC("hash table failed to allocate memory during resize");
            return HT_NOMEM;
        }
    }
    T = ht->table;
    N = ht->buckets - 1;
    k = HT_HASH_FUNCTION(key, len);
    i = 0;
    j = HT_PROBE(k, i, N);
    if (mode == ht_unique || mode == ht_multi) {
        ++ht->count;
        while (T[j] && T[j] != HT_DELETED) {
            ++i;
            j = HT_PROBE(k, i, N);
        }
        T[j] = new_item;
        return 0;
    }
    while ((item = T[j])) {
        if (item == HT_DELETED) {
            if (vacant == 0) {
                /*
                 * If a tombstone was found, use the first available,
                 * but continue search for possible match.
                 */
                vacant = &T[j];
            }
        } else if (ht_match(key, len, item)) {
            if (mode == ht_replace) {
                T[j] = new_item;
            }
            return item;
        }
        ++i;
        j = HT_PROBE(k, i, N);
    }
    if (vacant == 0) {
        vacant = &T[j];
    }
    ++ht->count;
    *vacant = new_item;
    return 0;
}

static ht_item_t ht_find(hash_table_t *ht, const void *key, size_t len)
{
    ht_item_t *T = ht->table;
    size_t N, i, j, k;
    ht_item_t item;

    if (T == 0) {
        return 0;
    }
    N = ht->buckets - 1;
    k = HT_HASH_FUNCTION(key, len);
    i = 0;
    j = HT_PROBE(k, i, N);
    while ((item = T[j])) {
        if ((item != HT_DELETED) &&
                ht_match(key, len, item)) {
            return item;
        }
        ++i;
        j = HT_PROBE(k, i, N);
    }
    return 0;
}

static ht_item_t ht_remove(hash_table_t *ht, const void *key, size_t len)
{
    ht_item_t *T = ht->table;
    size_t N, i, j, k;
    ht_item_t item;

    if (T == 0) {
        return 0;
    }
    N = ht->buckets - 1;
    k = HT_HASH_FUNCTION(key, len);
    i = 0;
    j = HT_PROBE(k, i, N);
    while ((item = T[j])) {
        if (item != HT_DELETED &&
                ht_match(key, len, item)) {
            T[j] = HT_DELETED;
            --ht->count;
            return item;
        }
        ++i;
        j = HT_PROBE(k, i, N);
    }
    return 0;
}

static void ht_visit(hash_table_t *ht, ht_visitor_f *visitor, void *context)
{
    size_t i;
    ht_item_t *T = ht->table;
    ht_item_t item;

    for (i = 0; i < ht->buckets; ++i) {
        item = T[i];
        if (item && item != HT_DELETED) {
            visitor(context, item);
        }
    }
}

static void ht_clear(hash_table_t *ht)
{
    if (ht->table) {
        free(ht->table);
    }
    memset(ht, 0, sizeof(*ht));
}
