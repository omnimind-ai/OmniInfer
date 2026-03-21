 /* Note: only one hash table can be implemented a single file. */
#include "../symbols.h"
#include "hash/ht_hash_function.h"

static size_t value_hash_function(const void *key, size_t key_len)
{
    const fb_value_t *value = key;

    (void)key_len;

    switch (value->type) {
    case vt_int:
        return ht_int_hash_function((void *)(size_t)(value->i ^ value->type), sizeof(value->i));
    case vt_uint:
        return ht_int_hash_function((void *)(size_t)(value->u ^ value->type), sizeof(value->u));
    case vt_bool:
        return ht_int_hash_function((void *)(size_t)(value->b ^ value->type), sizeof(value->b));
    default:
        return 0;
    }
}

#define HT_HASH_FUNCTION value_hash_function

#include "hash/hash_table_def.h"
DEFINE_HASH_TABLE(fb_value_set)
#include "hash/hash_table_impl.h"

static inline int ht_match(const void *key, size_t len, fb_value_t *item)
{
    const fb_value_t *value = key;

    (void)len;

    if (value->type != item->type) {
        return 0;
    }
    switch (value->type) {
    case vt_int:
        return value->i == item->i;
    case vt_uint:
        return value->u == item->u;
    case vt_bool:
        return value->b == item->b;
    default:
        return 0;
    }
}

static inline const void *ht_key(fb_value_t *value)
{
    return value;
}

static inline size_t ht_key_len(fb_value_t *value)
{
    (void)value;

    return 0;
}
