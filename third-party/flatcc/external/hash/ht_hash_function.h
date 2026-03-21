#ifndef HT_HASH_FUNCTION_H
#define HT_HASH_FUNCTION_H

#include <stddef.h>

#ifdef _MSC_VER
/* `inline` only advisory anyway. */
#pragma warning(disable: 4710) /* function not inlined */
#endif

/* Avoid 0 special case in hash functions and allow for configuration with unguessable seed. */
#ifndef HT_HASH_SEED
#define HT_HASH_SEED UINT32_C(0x2f693b52)
#endif

#ifndef HT_HASH_FALLBACK

/*
 * Currently xxhash 0.8.0 with XXH3 class hash functions is the clear leader.
 */
#define XXH_INLINE_ALL
#include "xxhash.h"

static inline size_t ht_default_hash_function(const void *key, size_t len)
{
#ifdef HT_HASH_32
    return (size_t)XXH32(key, len, (unsigned int)(HT_HASH_SEED));
#else
#      ifdef HT_HASH_USES_DEFAULT_SEED
            /* This is slightly faster so do not impose a seed in the default case. */
           return (size_t)XXH3_64bits(key, len);
#      else
           return (size_t)XXH3_64bits_withSeed(key, len, (XXH64_hash_t)(HT_HASH_SEED));
#      endif
#endif
}

#else /* HT_HASH_FALLBACK */

#ifndef HT_HASH_32

/* Note: link with cmetrohash.c */

#include "cmetrohash.h"

static inline size_t ht_default_hash_function(const void *key, size_t len)
{
    uint64_t out;

    cmetrohash64_1((const uint8_t *)key, len, HT_HASH_SEED, (uint8_t *)&out);
    return (unsigned int)out;
}

#else /* HT_HASH_32 */

#include "PMurHash.h"

static inline size_t ht_default_hash_function(const void *key, size_t len)
{
    return (size_t)PMurHash32((HT_HASH_SEED), key, (int)len);
}

#endif /* HT_HASH_32 */

#endif /* HT_HASH_FALLBACK */


#ifndef HT_HASH_32

/* When using the pointer directly as a hash key. */
static inline size_t ht_ptr_hash_function(const void *key, size_t len)
{
    /* MurmurHash3 64-bit finalizer */
    uint64_t x;

    (void)len;

    x = ((uint64_t)(size_t)key) ^ (HT_HASH_SEED);

    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (size_t)x;
}

#else /* HT_HASH_32 */

/* When using the pointer directly as a hash key. */
static inline size_t ht_ptr_hash_function(const void *key, size_t len)
{
    /* http://stackoverflow.com/a/12996028 */
    size_t x;

    x = (size_t)key ^ (HT_HASH_SEED);

    x = ((x >> 16) ^ x) * 0x45d9f3bUL;
    x = ((x >> 16) ^ x) * 0x45d9f3bUL;
    x = ((x >> 16) ^ x);
    return x;
}

#endif /* HT_HASH_32 */

/* This assumes the key points to a 32-bit aligned random value that is its own hash function. */
static inline size_t ht_uint32_identity_hash_function(const void *key, size_t len)
{
    (void)len;
    return (size_t)*(uint32_t *)key;
}

/* This assumes the key points to a 64-bit aligned random value that is its own hash function. */
static inline size_t ht_uint64_identity_hash_function(const void *key, size_t len)
{
    (void)len;
    return (size_t)*(uint64_t *)key;
}

/* This assumes the key points to a 32-bit aligned value. */
static inline size_t ht_uint32_hash_function(const void *key, size_t len)
{
    uint32_t x = *(uint32_t *)key + (uint32_t)(HT_HASH_SEED);

    (void)len;

    /* http://stackoverflow.com/a/12996028 */
    x = ((x >> 16) ^ x) * UINT32_C(0x45d9f3b);
    x = ((x >> 16) ^ x) * UINT32_C(0x45d9f3b);
    x = ((x >> 16) ^ x);
    return x;
}

/* This assumes the key points to a 64-bit aligned value. */
static inline size_t ht_uint64_hash_function(const void *key, size_t len)
{
    uint64_t x = *(uint64_t *)key + UINT64_C(0x9e3779b97f4a7c15) + (uint64_t)(HT_HASH_SEED);

    (void)len;

    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return (size_t)(x ^ (x >> 31));
}

/*
 * Suited for set operations of low-valued integers where the stored
 * hash pointer is the key and the value.
 *
 * This function is especially useful for small hash tables (<1000)
 * where collisions are cheap due to caching but also works for integer
 * sets up to at least 1,000,000.
 *
 * NOTE: The multiplicative hash function by Knuth requires the modulo
 * to table size be done by shifting the upper bits down, since this is
 * where the quality bits reside. This yields significantly fewer
 * collisions which is important for e.g. chained hashing. However, our
 * interface does not provide the required information.
 *
 * When used in open hashing with load factors below 0.7 where the
 * stored pointer is also the key, collision checking is very cheap and
 * this pays off in a large range of table sizes where a more
 * complicated hash simply doesn't pay off.
 * 
 * When used with a pointer set where the pointer is also the key, it is
 * not likely to work as well because the pointer acts as a large
 * integer which works against the design of the hash function. Here a
 * better mix function is probably worthwhile - therefore we also have
 * ht_ptr_hash_function.
 */
static inline size_t ht_int_hash_function(const void *key, size_t len)
{
    (void)len;
    return ((size_t)key ^ (HT_HASH_SEED)) * 2654435761UL;
}

/* Bernsteins hash function, assumes string is zero terminated, len is ignored. */
static inline size_t ht_str_hash_function(const void *key, size_t len)
{
    const unsigned char *str = key;
    size_t hash = 5381 ^ (HT_HASH_SEED);
    size_t c;

    (void)len;

    while ((c = (size_t)*str++))
        hash = ((hash << 5) + hash) ^ c; /* (hash * 33) xor c */

    return hash;
}

/* Hashes at most len characters or until zero termination. */
static inline size_t ht_strn_hash_function(const void *key, size_t len)
{
    const unsigned char *str = key;
    size_t hash = 5381 ^ (HT_HASH_SEED);
    size_t c;

    while (--len && (c = (size_t)*str++))
        hash = ((hash << 5) + hash) ^ c; /* (hash * 33) xor c */

    return hash;
}

static inline uint32_t ht_fnv1a32_hash_function(const void *key, size_t len)
{
#ifndef FNV1A_NOMUL
    const uint32_t prime = UINT32_C(0x1000193);
#endif
    uint32_t hash = UINT32_C(0x811c9dc5);
    const uint8_t *p = key;

    while (len--) {
        hash ^= (uint64_t)*p++;
#ifndef FNV1A_NOMUL
        hash *= prime;
#else
        hash += (hash << 1) + (hash << 4) + (hash << 7) +
            (hash << 8) + (hash << 24);
#endif
    }
    return hash;
}

static inline uint64_t ht_fnv1a64_hash_function(const void *key, size_t len)
{
#ifndef FNV1A_NOMUL
    const uint64_t prime = UINT64_C(0x100000001b3);
#endif
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    const uint8_t *p = key;

    while (len--) {
        hash ^= (uint64_t)*p++;
#ifndef FNV1A_NOMUL
        hash *= prime;
#else
        hash += (hash << 1) + (hash << 4) + (hash << 5) +
		    (hash << 7) + (hash << 8) + (hash << 40);
#endif
    }
    return hash;
}

/* Hashes until string termination and ignores length argument. */
static inline uint32_t ht_fnv1a32_str_hash_function(const void *key, size_t len)
{
#ifndef FNV1A_NOMUL
    const uint32_t prime = UINT32_C(0x1000193);
#endif
    uint32_t hash = UINT32_C(0x811c9dc5);
    const uint8_t *p = key;

    (void)len;

    while (*p) {
        hash ^= (uint64_t)*p++;
#ifndef FNV1A_NOMUL
        hash *= prime;
#else
        hash += (hash << 1) + (hash << 4) + (hash << 7) +
            (hash << 8) + (hash << 24);
#endif
    }
    return hash;
}

/* Hashes until string termination and ignores length argument. */
static inline uint64_t ht_fnv1a64_str_hash_function(const void *key, size_t len)
{
#ifndef FNV1A_NOMUL
    const uint64_t prime = UINT64_C(0x100000001b3);
#endif
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    const uint8_t *p = key;

    (void)len;

    while (*p) {
        hash ^= (uint64_t)*p++;
#ifndef FNV1A_NOMUL
        hash *= prime;
#else
        hash += (hash << 1) + (hash << 4) + (hash << 5) +
		    (hash << 7) + (hash << 8) + (hash << 40);
#endif
    }
    return hash;
}


#endif /* HT_HASH_FUNCTION_H */
