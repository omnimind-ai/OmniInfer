 /* Note: only one hash table can be implemented a single file. */
#include "../symbols.h"
#include "hash/hash_table_def.h"
DEFINE_HASH_TABLE(fb_symbol_table)
#include "hash/hash_table_impl.h"

static inline int ht_match(const void *key, size_t len, fb_symbol_t *sym)
{
    return len == ht_key_len(sym) && memcmp(key, ht_key(sym), len) == 0;
}

static inline const void *ht_key(fb_symbol_t *sym)
{
    return sym->ident->text;
}

static inline size_t ht_key_len(fb_symbol_t *sym)
{
    fb_token_t *ident = sym->ident;

    return (size_t)ident->len;
}
