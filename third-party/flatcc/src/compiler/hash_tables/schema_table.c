 /* Note: only one hash table can be implemented a single file. */
#include "../symbols.h"
#include "hash/hash_table_def.h"
DEFINE_HASH_TABLE(fb_schema_table)

#include "hash/hash_table_impl.h"

static inline int ht_match(const void *key, size_t len, fb_schema_t *schema)
{
    return len == (size_t)schema->name.name.s.len && memcmp(key, schema->name.name.s.s, len) == 0;
}

static inline const void *ht_key(fb_schema_t *schema)
{
    return schema->name.name.s.s;
}

static inline size_t ht_key_len(fb_schema_t *schema)
{
    return (size_t)schema->name.name.s.len;
}
