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

#ifndef HASH_TABLE_H
#define HASH_TABLE_H

#include "ht_portable.h"
#include <stddef.h>

/*
 * Define HT_PRIVATE to make all name wrapping interface functions static
 * inline when including hash implementation directly in user code. This
 * can increase performance significantly (3x) on small hash tables with
 * fast hash functions because the compiler can better optimize static
 * functions. Some compiler optimizations will get the same speed
 * with external linkage (clang 4.2 -O4 but not -O3).
 *
 * Can also be used to simple hide the interface from global
 * linkage to avoid name clashes.
 */
#ifndef HT_PRIVATE
#define HT_PRIV
#else
#define HT_PRIV static inline
#endif

/*
 * Generic hash table type. This makes it possible to use hash tables
 * in datastructures and header files that do not have access to
 * the specific hash table implementation. Call to init is optional
 * if the structure is zeroed.
 *
 * Offsets are only used with Robin Hood hashing to segment each chain.
 *
 * Keys and values are both stored in the same item pointer. There are
 * downsides to this over a key / value represention, but since we also
 * use less space we can afford lower the load factor and we can have a
 * more complex key representations. The smaller bucket size also helps
 * when ordering Robin Hood hash chains.
 */
typedef struct hash_table hash_table_t;
struct hash_table {
    void *table;
    char *offsets;
    size_t count;
    /* May be stored as a direct count, or log2. */
    size_t buckets;
};

enum hash_table_insert_mode {
    ht_replace = 0,
    ht_keep = 1,
    ht_unique = 2,
    ht_multi = 3,
};

/*
 * This macro defines the prototypes of the hash table that user code
 * needs for linkage.
 *
 * See also "hash_table_def.h" which builds wrapper functions to a
 * generic hash table implementation so each specialization gets its own
 * set of named functions.
 *
 * The HT_ITEM is normally a pointer to and the hash table does not
 * store any signficant information internally. Customizations map
 * the item value to a key. Certain values can be reserved, for
 * example 0 indicates missing value, and sometimes 1 is reserved for
 * internal tombstones and 2 may be used to return allocation failure.
 */
#define DECLARE_HASH_TABLE(HT_NAME, HT_ITEM)                                \
                                                                            \
typedef hash_table_t HT_NAME##_t;                                           \
typedef HT_ITEM HT_NAME##_item_t;                                           \
                                                                            \
/* Prototype for user supplied callback when visiting all elements. */      \
typedef void HT_NAME##_visitor_f(void *context, HT_ITEM item);              \
                                                                            \
extern const HT_NAME##_item_t HT_NAME##_missing;                            \
extern const HT_NAME##_item_t HT_NAME##_nomem;                              \
extern const HT_NAME##_item_t HT_NAME##_deleted;                            \
                                                                            \
static inline int HT_NAME##_is_valid(HT_ITEM item)                          \
{                                                                           \
    return                                                                  \
        item != HT_NAME##_missing &&                                        \
        item != HT_NAME##_nomem &&                                          \
        item != HT_NAME##_deleted;                                          \
}                                                                           \
                                                                            \
static inline int HT_NAME##_is_missing(HT_ITEM item)                        \
{                                                                           \
    return item == HT_NAME##_missing;                                       \
}                                                                           \
                                                                            \
static inline int HT_NAME##_is_nomem(HT_ITEM item)                          \
{                                                                           \
    return item == HT_NAME##_nomem;                                         \
}                                                                           \
                                                                            \
static inline int HT_NAME##_is_deleted(HT_ITEM item)                        \
{                                                                           \
    return item == HT_NAME##_deleted;                                       \
}                                                                           \
                                                                            \
/*                                                                          \
 * Allocates enough buckets to represent count elements without resizing.   \
 * The actual number of allocated buckets depends on the load factor        \
 * given as a macro argument in the implementation. The bucket number       \
 * rounds up to the nearest power of 2.                                     \
 *                                                                          \
 * `ht` should not be initialized beforehand, otherwise use resize.         \
 * Alternatively, it is also valid to zero initialize the table by          \
 * other means - this will postpone allocation until needed.                \
 *                                                                          \
 * The load factor (template argument) should be positive and at most       \
 * 100%, otherwise insertion and resize cannot succeed. The recommended     \
 * load factor is between 25% and 75%.                                      \
 *                                                                          \
 * Returns 0 on success, -1 on allocation failure or invalid load factor.   \
 */                                                                         \
HT_PRIV int HT_NAME##_init(HT_NAME##_t *ht, size_t count);                  \
                                                                            \
/*                                                                          \
 * Clears the allocated memory. Optionally takes a destructor               \
 * that will visit all items.                                               \
 * The table struct may be reused after being destroyed.                    \
 * May also be called on a zero initialised hash table.                     \
 *                                                                          \
 * Can be called in place of clear for more control.                        \
 */                                                                         \
HT_PRIV void HT_NAME##_destroy(HT_NAME##_t *ht,                             \
        HT_NAME##_visitor_f *destructor, void *context);                    \
                                                                            \
/*                                                                          \
 * Clears the allocated memory, but does manage memory or state of any      \
 * stored items. It is a simpler version of destroy.                        \
 */                                                                         \
HT_PRIV void HT_NAME##_clear(HT_NAME##_t *ht);                              \
                                                                            \
/*                                                                          \
 * Resizes the hash table to hold at least `count` elements.                \
 * The actual number of allocated buckets is a strictly larger power of     \
 * two. If `count` is smaller than the current number of elements,          \
 * that number is used instead of count. Thus, resize(ht, 0) may be         \
 * used to reduce the table size after a spike.                             \
 * The function is called automatically as elements are inserted,           \
 * but shrinking the table should be done manually.                         \
 *                                                                          \
 * If resizing to same size, table is still reallocated but will then       \
 * clean up old tombstones from excessive deletion.                         \
 *                                                                          \
 * Returns 0 on success, -1 on allocation failure.                          \
 */                                                                         \
HT_PRIV int HT_NAME##_resize(HT_NAME##_t *ht, size_t count);                \
                                                                            \
/*                                                                          \
 * Inserts an item pointer in one of the following modes:                   \
 *                                                                          \
 * ht_keep:                                                                 \
 *   If the key exists, the stored item is kept and returned,               \
 *   otherwise it is inserted and null is returned.                         \
 *                                                                          \
 * ht_replace:                                                              \
 *   If the key exists, the stored item is replaced and the old             \
 *   item is returned, otherwise the item is inserted and null              \
 *   is returned.                                                           \
 *                                                                          \
 * ht_unique:                                                               \
 *   Inserts an item without checking if a key exists. Always return        \
 *   null. This is faster when it is known that the key does not exists.    \
 *                                                                          \
 * ht_multi:                                                                \
 *   Same as ht_unique but with the intention that a duplicate key          \
 *   might exist. This should not be abused because not all hash table      \
 *   implementions work well with too many collissions. Robin Hood          \
 *   hashing might reallocate aggressively to keep the chain length         \
 *   down. Linear and Quadratic probing do handle this, albeit slow.        \
 *                                                                          \
 * The inserted item cannot have the value HT_MISSING and depending on      \
 * implementation also not HT_DELETED and HT_NOMEM, but the                 \
 * definitions are type specific.                                           \
 */                                                                         \
HT_PRIV HT_ITEM HT_NAME##_insert(HT_NAME##_t *ht,                           \
        const void *key, size_t len, HT_ITEM item, int mode);               \
                                                                            \
/* Similar to insert, but derives key from item. */                         \
HT_PRIV HT_ITEM HT_NAME##_insert_item(HT_NAME##_t *ht,                      \
        HT_ITEM item, int mode);                                            \
                                                                            \
/*                                                                          \
 * Finds the first matching item if any, or returns null.                   \
 * If there are duplicate keys, the first inserted is returned.             \
 */                                                                         \
HT_PRIV HT_ITEM HT_NAME##_find(HT_NAME##_t *ht,                             \
        const void *key, size_t len);                                       \
                                                                            \
/*                                                                          \
 * Removes first inserted item that match the given key, if any.            \
 * Returns the removed item if any, otherwise null.                         \
 */                                                                         \
HT_PRIV HT_ITEM HT_NAME##_remove(HT_NAME##_t *ht,                           \
        const void *key, size_t len);                                       \
                                                                            \
/*                                                                          \
 * Finds an item that compares the same as the given item but it is         \
 * not necessarily the same item if it either isn't stored, or if           \
 * there are duplicates in the table.                                       \
 */                                                                         \
HT_PRIV HT_ITEM HT_NAME##_find_item(HT_NAME##_t *ht, HT_ITEM item);         \
                                                                            \
/*                                                                          \
 * This removes the first item that matches the given item, not             \
 * necessarily the item itself, and the item need not be present            \
 * in the table. Even if the item is in fact removed, it may still          \
 * be present if stored multiple times through abuse use of the             \
 * insert_unique function.                                                  \
 */                                                                         \
HT_PRIV HT_ITEM HT_NAME##_remove_item(HT_NAME##_t *ht, HT_ITEM item);       \
                                                                            \
/*                                                                          \
 * Calls a function for every item in the hash table. This may be           \
 * used for destructing items, provided the table is not accessed           \
 * subsequently. In fact, the hash_table_clear function takes an            \
 * optional visitor that does exactly that.                                 \
 *                                                                          \
 * The function is linear of the allocated hash table size, so will be      \
 * inefficient if the hash table was resized much larger than the number    \
 * of stored items. In that case it is better to store links in the         \
 * items. For the default resizing, the function is reasonably fast         \
 * because for cache reasons it is very fast to exclude empty elements.     \
 */                                                                         \
HT_PRIV void HT_NAME##_visit(HT_NAME##_t *ht,                               \
        HT_NAME##_visitor_f *visitor, void *context);                       \
                                                                            \
/*                                                                          \
 * Returns number of elements in the table. (Not necessarily the number of  \
 * unique keys.                                                             \
 */                                                                         \
static inline size_t HT_NAME##_count(HT_NAME##_t *ht)                       \
{                                                                           \
    return ht->count;                                                       \
}                                                                           \

#endif /* HASH_TABLE_H */
