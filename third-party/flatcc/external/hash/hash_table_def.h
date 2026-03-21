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

#ifndef HASH_TABLE_DEF_H
#define HASH_TABLE_DEF_H

#include "ht_hash_function.h"
#ifndef HT_HASH_FUNCTION
/*
 * If the default hash function is used, make sure to link with the
 * appropriate hash implementation file.
 */
#define HT_HASH_FUNCTION ht_default_hash_function
#endif

#ifndef HT_LOAD_FACTOR
#define HT_LOAD_FACTOR 0.7
#endif

#define HT_LOAD_FACTOR_FRAC ((size_t)((float)(HT_LOAD_FACTOR)*256))

#ifndef HT_PANIC
#include <stdio.h>
#define HT_PANIC(s) { fprintf(stderr, "aborting on panic: %s\n", s); exit(1); }
#endif

#ifndef HT_MISSING
#define HT_MISSING ((ht_item_t)0)
#endif

#ifndef HT_NOMEM
#define HT_NOMEM ((ht_item_t)1)
#endif

#ifndef HT_DELETED
#define HT_DELETED ((ht_item_t)2)
#endif

#define DEFINE_HASH_TABLE(HT_NAME)                                          \
                                                                            \
typedef HT_NAME##_item_t ht_item_t;                                         \
typedef HT_NAME##_visitor_f ht_visitor_f;                                   \
                                                                            \
/* User supplied. */                                                        \
static inline int ht_match(const void *key, size_t len, ht_item_t item);    \
static inline const void *ht_key(ht_item_t item);                           \
static inline size_t ht_key_len(ht_item_t item);                            \
                                                                            \
/* Implementation supplied. */                                              \
static ht_item_t ht_insert(hash_table_t *ht,                                \
        const void *key, size_t len, ht_item_t new_item, int mode);         \
static ht_item_t ht_find(hash_table_t *ht, const void *key, size_t len);    \
static ht_item_t ht_remove(hash_table_t *ht, const void *key, size_t len);  \
static int ht_init(hash_table_t *ht, size_t count);                         \
static int ht_resize(hash_table_t *ht, size_t count);                       \
static void ht_clear(hash_table_t *ht);                                     \
static void ht_visit(hash_table_t *ht,                                      \
        ht_visitor_f *visitor, void *context);                              \
                                                                            \
const ht_item_t HT_NAME##_missing = HT_MISSING;                             \
const ht_item_t HT_NAME##_nomem = HT_NOMEM;                                 \
const ht_item_t HT_NAME##_deleted = HT_DELETED;                             \
                                                                            \
HT_PRIV void HT_NAME##_clear(HT_NAME##_t *ht)                               \
{                                                                           \
    ht_clear(ht);                                                           \
}                                                                           \
                                                                            \
HT_PRIV void HT_NAME##_destroy(HT_NAME##_t *ht,                             \
        HT_NAME##_visitor_f *destructor, void *context)                     \
{                                                                           \
    if (destructor) {                                                       \
        ht_visit(ht, destructor, context);                                  \
    }                                                                       \
    ht_clear(ht);                                                           \
}                                                                           \
                                                                            \
HT_PRIV int HT_NAME##_init(HT_NAME##_t *ht, size_t count)                   \
{                                                                           \
    return ht_init(ht, count);                                              \
}                                                                           \
                                                                            \
HT_PRIV int HT_NAME##_resize(HT_NAME##_t *ht, size_t count)                 \
{                                                                           \
    return ht_resize(ht, count);                                            \
}                                                                           \
                                                                            \
HT_PRIV ht_item_t HT_NAME##_insert(HT_NAME##_t *ht,                         \
        const void *key, size_t len, ht_item_t new_item, int mode)          \
{                                                                           \
    return ht_insert(ht, key, len, new_item, mode);                         \
}                                                                           \
                                                                            \
HT_PRIV ht_item_t HT_NAME##_insert_item(HT_NAME##_t *ht,                    \
        ht_item_t item, int mode)                                           \
{                                                                           \
    return ht_insert(ht,                                                    \
            ht_key(item),                                                   \
            ht_key_len(item),                                               \
            item, mode);                                                    \
}                                                                           \
                                                                            \
HT_PRIV ht_item_t HT_NAME##_find(HT_NAME##_t *ht,                           \
        const void *key, size_t len)                                        \
{                                                                           \
    return ht_find(ht, key, len);                                           \
}                                                                           \
                                                                            \
HT_PRIV ht_item_t HT_NAME##_find_item(HT_NAME##_t *ht, ht_item_t item)      \
{                                                                           \
    return ht_find(ht,                                                      \
            ht_key(item),                                                   \
            ht_key_len(item));                                              \
}                                                                           \
                                                                            \
HT_PRIV ht_item_t HT_NAME##_remove(HT_NAME##_t *ht,                         \
        const void *key, size_t len)                                        \
{                                                                           \
    return ht_remove(ht, key, len);                                         \
}                                                                           \
                                                                            \
HT_PRIV ht_item_t HT_NAME##_remove_item(HT_NAME##_t *ht, ht_item_t item)    \
{                                                                           \
    return ht_remove(ht, ht_key(item), ht_key_len(item));                   \
}                                                                           \
                                                                            \
HT_PRIV void HT_NAME##_visit(HT_NAME##_t *ht,                               \
        HT_NAME##_visitor_f *visitor, void *context)                        \
{                                                                           \
    ht_visit(ht, visitor, context);                                         \
}                                                                           \

#endif /* HASH_TABLE_DEF_H */
