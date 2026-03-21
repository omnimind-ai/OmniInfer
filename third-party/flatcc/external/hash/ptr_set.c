/*
 * Creates a set of stored pointers by using the pointer itself as key.
 *
 * (void *)0 (HT_MISSING) cannot be stored.
 * (void *)1 (HT_DELETED) also cannot be stored.
 *
 * ht_item, ht_key, ht_key_len, and ht_match are required.
 *
 * In this case HT_HASH_FUNCTION is also required because
 * we do not read the content of the key but use the pointer
 * itself as a key. The default behavior would crash.
 *
 * Only one hash table can be defined in a single compilation unit
 * because of static function names in the generic implementation.
 */

#include "ptr_set.h"

static inline size_t ptr_set_hash_function(const void *s, size_t len);
#define HT_HASH_FUNCTION ptr_set_hash_function

#define HT_LOAD_FACTOR 0.7
#include "hash_table_def.h"
DEFINE_HASH_TABLE(ptr_set)

#if defined(PTR_SET_RH)
#include "hash_table_impl_rh.h"
#else
#include "hash_table_impl.h"
#endif

static inline const void *ht_key(ht_item_t x)
{
    return (const void *)x;
}

static inline size_t ht_key_len(ht_item_t x)
{
    return sizeof(x);
}

static inline int ht_match(const void *key, size_t len, ht_item_t x)
{
    (void)len;
    return (size_t)key == (size_t)x;
}

static inline size_t ptr_set_hash_function(const void *s, size_t len)
{
#if defined (PTR_SET_PTR_HASH)
    /* Murmur hash like finalization step. */
    return ht_ptr_hash_function(s, len);
#elif defined (PTR_SET_INT_HASH)
    /* Knuths multiplication. */
    return ht_int_hash_function(s, len);
#else
    (void)len;
    return ht_default_hash_function(&s, sizeof(char *));
#endif
}
