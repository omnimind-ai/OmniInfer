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

#include <string.h>

/* We use the same define for all implementations */
#ifdef HASH_TABLE_IMPL
#error "cannot have multiple implementations in same compilation unit"
#endif
#define HASH_TABLE_IMPL
/* Robin Hood (with offset table) */
#define HT_RH

#if defined(_MSC_VER)
#pragma warning(disable: 4127) /* conditional expression is constant */
#endif

#include <stdlib.h>
#include <assert.h>

/*
 * A variation of Robin Hashing:
 * We do not calcute distance from buckets, nor do we cache
 * hash keys. Instead we maintain a 7-bit offset that points
 * to where the first entry of a bucket is stored. In Robin Hood hashing
 * all entries conceptually chained to the same bucket are stored
 * immediately after each other in order of insertion. The offset of
 * the next bucket is naturally the end of the previous bucket, off by
 * one. This breaks down when the bucket offset is 0 and the bucket is
 * empty because it suggests there is an element. We cannot distinguish
 * between a single used and unused entry, except by looking at the
 * content or otherwise tag the information on. This is not a problem,
 * just a special case to deal with.
 *
 * The offsets are stored separately which might lead to more cache line
 * traffic, but the alternative is not very elegant - either wasting
 * space or trying to pack offsets on a per cache line basis. We only
 * need 8 bits for offsets. If the offset overflows, bit 7 will be set
 * which we can easily detect. During insertion, offsets are increated
 * on all affected buckets, and likewise decrement on remove. In
 * principle we can use bit parallel increments to update most offsets
 * in a single operation, but it is hardly worthwhile due to setup
 * cost. The approach bears some resemblance to hopscotch hashing which
 * uses local offsets for chaining, but we prefer the simpler Robin
 * Hood approach.
 *
 * If the offset overflows, the table is resized. We expect the packed
 * chains to behave like a special case of a hopscotch layout and
 * consequently have the same bounds, meaning we are unlikely to have
 * neither long offsets nor long chains if we resize below very full
 * so resizing on an offset of 128 should be ok.
 *
 * Our main motivation for this hashing is actually to get rid of
 * tombstones in quadruatic and linear probing. Avoiding tombstones
 * is much simpler when sorting chains Robin Hood style, and we avoid
 * checking for tombstones. We loose this benefit by having to inspect
 * offsets, but then also avoid checking keys before the chain, and
 * after because we can zero in on exactly the entries belonging to
 * bucket.
 *
 * Unlike traditional Robin Hood, we can find a missing key very quickly
 * without any heuristics: we only need to inspect exactly the number
 * of entries in the bucket (or at most 1 if the bucket is empty).
 *
 * Find operations start exactly at an entry with a matching hash key
 * unlike normal Robin Hood which must scan past a few earlier entries
 * on average, or guestimate where to start and seek both ways.
 *
 * We can also very quickly insert a key that is known to be unique
 * because we can add it directly to the end (but possibly requiring
 * a shift of later entries Robin Hood style).
 *
 * Whether these benefits outweighs the cost of a separate offset
 * lookup is unclear, but the reduced memory consumption certainly
 * allows for a lower load factor, which also helps a lot.
 *
 * Traditional Robin Hood Hashing actually permits a chain to become
 * very long. We do not permit this, in line with hopscotch hashing.
 * This is a drawback from a security perspective because worst case
 * this can trigger resizing ad infinitum iff the hash function can
 * be hacked or massive duplicate key insertion can be triggered. By
 * used the provided hash functions and seeding them randomly at
 * startup, and avoiding the multi key feature, it is very unlikely to
 * be a problem with what is known about hash table attacks so far.
 *
 * Values and keys are not stored, only item pointers. Custom macroes
 * or inline functions provide access to key data from the item. We
 * could add a separate value array and treat the item strictly as a
 * key, but we can have a smaller load factor instead, and can more
 * easily avoid copying complex key structures, such as start end
 * pointers to token data for parser.
 *
 * A typical hash table has: key pointer or key value, value pointer
 * or value, a cached hash key or bitmap (for Robin Hood or Hopscotch)
 * which on 64 bit platforms easily amounts to 20 bytes or more per
 * bucket. We use 9 bytes on 64 bit platforms and 5 bytes on 32 bit.
 * This gets us down to a max load of 0.5 and on average about 0.37.
 * This should make it very likely that the first bucket inspected is
 * a direct hit negating the benefit of caching hash keys. In addition,
 * when it is not a direct hit, we get pointers loaded in a cache line
 * to inspect, all known to have the same hash key.
 */

int ht_init(hash_table_t *ht, size_t count)
{
    size_t buckets = 4;

    if ((HT_LOAD_FACTOR_FRAC) > 256 || (HT_LOAD_FACTOR_FRAC) < 1) {
        /*
         * 101% will never terminate insertion.
         * 0% will never terminate resize.
         */
        HT_PANIC("robin hood hash table failed with impossible load factor");
        return -1;
    }
    while (count > buckets * (HT_LOAD_FACTOR_FRAC) / 256) {
        buckets *= 2;
    }
    ht->table = calloc(buckets, sizeof(ht_item_t));
    if (ht->table == 0) {
        return -1;
    }
    ht->offsets = calloc(buckets, sizeof(char));
    if (ht->offsets == 0) {
        free(ht->table);
        ht->table = 0;
        return -1;
    }
    ht->buckets = buckets;
    ht->count = 0;
    return 0;
}

int ht_resize(hash_table_t *ht, size_t count)
{
    size_t i;
    hash_table_t ht2;
    ht_item_t *T = ht->table;
    ht_item_t item;

    if (count < ht->count) {
        count = ht->count;
    }
    if (ht_init(&ht2, count)) {
        return -1;
    }
    for (i = 0; i < ht->buckets; ++i) {
        item = T[i];
        if (item > (ht_item_t)1) {
            ht_insert(&ht2, ht_key(item), ht_key_len(item), item, ht_multi);
        }
    }
    ht_clear(ht);
    memcpy(ht, &ht2, sizeof(*ht));
    return 0;
}

ht_item_t ht_insert(hash_table_t *ht,
        const void *key, size_t len, ht_item_t item, int mode)
{
    ht_item_t *T;
    size_t N, n, j, k, offset;
    ht_item_t new_item;
    char overflow = 0;

    new_item = item;
    if (ht->count >= ht->buckets * (HT_LOAD_FACTOR_FRAC) / 256) {
        if (ht_resize(ht, ht->count * 2)) {
            HT_PANIC("robin hood hash table failed to allocate memory during resize");
            return HT_NOMEM;
        }
    }
    T = ht->table;
    N = ht->buckets - 1;
    k = HT_HASH_FUNCTION(key, len) & N;
    offset = ht->offsets[k];
    j = (k + offset) & N;
    /*
     * T[j] == 0 is a special case because we cannot count
     * zero probe length, and because we should not increment
     * the offset at insertion point in this case.
     *
     * T[j] == 0 implies offset == 0, but this way we avoid
     * hitting memory that we don't need.
     */
    if (offset == 0 && T[j] == 0) {
        ++ht->count;
        T[j] = new_item;
        return 0;
    }
    n = ht->offsets[(k + 1) & N] - offset + 1;
    if (mode == ht_multi) {
        /* Don't search for match before inserting. */
        j = (j + n) & N;
        n = 0;
    }
    while (n--) {
        item = T[j];
        if (ht_match(key, len, item)) {
            if (mode == ht_replace) {
                T[j] = new_item;
            }
            return item;
        }
        j = (j + 1) & N;
    }
    ++ht->count;
    while (k != j) {
        /* Only increment buckets after own bucket. */
        k = (k + 1) & N;
        overflow |= ++ht->offsets[k];
    }
    while ((item = T[j])) {
        T[j] = new_item;
        new_item = item;
        j = (j + 1) & N;
        overflow |= ++ht->offsets[j];
    }
    T[j] = new_item;

    if (overflow < 0) {
        /*
         * At least one offset overflowed, so we need to
         * resize the table.
         */
        if (ht->count * 10 < ht->buckets) {
            HT_PANIC("FATAL: hash table resize on low utilization would explode\n"\
                   "  possible collision DoS or bad hash function");
            return HT_NOMEM;
        }
        if (ht_resize(ht, ht->count * 2)) {
            HT_PANIC("FATAL: hash table resize failed and left hash table inconsistent");\
            /*
             * This renders the hash table in a bad state
             * because we have updated to an inconsistent
             * state.
             */
            return HT_NOMEM;
        }
    }
    return item;
}

ht_item_t ht_find(hash_table_t *ht, const void *key, size_t len)
{
    ht_item_t *T = ht->table;
    size_t N, n, j, k, offset;
    ht_item_t item;

    if (T == 0) {
        return 0;
    }
    N = ht->buckets - 1;
    k = HT_HASH_FUNCTION(key, len) & N;
    offset = ht->offsets[k];
    j = (k + offset) & N;
    if (offset == 0 && T[j] == 0) {
        /* Special case because we cannot count zero probe length. */
        return 0;
    }
    n = ht->offsets[(k + 1) & N] - offset + 1;
    while (n--) {
        item = T[j];
        if (ht_match(key, len, item)) {
            return item;
        }
        j = (j + 1) & N;
    }
    return 0;
}

ht_item_t ht_remove(hash_table_t *ht, const void *key, size_t len)
{
    ht_item_t *T = ht->table;
    size_t N, n, j, k, offset;
    ht_item_t item, *next_item;

    if (T == 0) {
        return 0;
    }
    N = ht->buckets - 1;
    k = HT_HASH_FUNCTION(key, len) & N;
    offset = ht->offsets[k];
    j = (k + offset) & N;
    if (offset == 0 && T[j] == 0) {
        return 0;
    }
    n = ht->offsets[(k + 1) & N] - offset + 1;
    while (n) {
        item = T[j];
        if (ht_match(key, len, item)) {
            break;
        }
        j = (j + 1) & N;
        --n;
    }
    if (n == 0) {
        return 0;
    }
    --ht->count;
    while (k != j) {
        /* Do not update the offset of the bucket that we own. */
        k = (k + 1) & N;
        --ht->offsets[k];
    }
    for (;;) {
        j = (j + 1) & N;
        if (ht->offsets[j] == 0) {
            T[k] = 0;
            return item;
        }
        --ht->offsets[j];
        T[k] = T[j];
        k = j;
    }
}

void ht_visit(hash_table_t *ht, ht_visitor_f *visitor, void *context)
{
    size_t i;
    ht_item_t *T = ht->table;
    ht_item_t item;

    for (i = 0; i < ht->buckets; ++i) {
        item = T[i];
        if (item > (ht_item_t)1) {
            visitor(context, item);
        }
    }
}

void ht_clear(hash_table_t *ht)
{
    if (ht->table) {
        free(ht->table);
    }
    if (ht->offsets) {
        free(ht->offsets);
    }
    memset(ht, 0, sizeof(*ht));
}
