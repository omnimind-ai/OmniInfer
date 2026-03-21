#ifndef HASH_H
#define HASH_H

/* Misc. hash functions that do not comply to a specific interface. */

#include <stdlib.h>

#ifdef _MSC_VER
/* `inline` only advisory anyway. */
#pragma warning(disable: 4710) /* function not inlined */
#endif

static inline uint32_t hash_fnv1a32_update(uint32_t seed, uint8_t *buf, size_t len)
{
    uint8_t *p = buf;
#ifndef FNV1A_NOMUL
    const uint64_t prime = UINT32_C(0x1000193);
#endif
    uint64_t hash = seed;

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

static inline uint32_t hash_fnv1a32(uint8_t *buf, size_t len)
{
    return hash_fnv1a32_update(UINT32_C(0x811c9dc5), buf, len);
}

static inline uint64_t hash_fnv1a64_update(uint64_t v, uint8_t *buf, size_t len)
{
    uint8_t *p = buf;
#ifndef FNV1A_NOMUL
    const uint64_t prime = UINT64_C(0x100000001b3);
#endif
    uint64_t hash = v;

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

static inline uint64_t hash_fnv1a64(uint8_t *buf, size_t len)
{
    return hash_fnv1a64_update(UINT64_C(0xcbf29ce484222325), buf, len);
}

/*
 * MurmurHash 3 final mix with seed to handle 0.
 *
 * Width is number of bits of the value to return.
 * http://stackoverflow.com/a/12996028
 */
static inline uint32_t hash_bucket32(uint32_t v,  size_t width)
{
    uint32_t x = v + UINT32_C(0x2f693b52);

    x = ((x >> 16) ^ x) * UINT32_C(0x45d9f3b);
    x = ((x >> 16) ^ x) * UINT32_C(0x45d9f3b);
    x = ((x >> 16) ^ x);
    return x >> (32 - width);
}

/*
 * SplitMix64 - can be used to disperse fnv1a hash, to hash
 * an integer, or as a simple non-cryptographic prng.
 *
 * Width is number of bits of the value to return.
 * http://stackoverflow.com/a/12996028
 */
static inline uint64_t hash_bucket64(uint64_t v, size_t width)
{
    uint64_t x = v + UINT64_C(0x9e3779b97f4a7c15);
    
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    x = x ^ (x >> 31);
    return x >> (64 - width);
}

static inline uint64_t hash_random64(uint64_t *state)
{
    uint64_t x;

    x = hash_bucket64(*state, 64);
    *state = x;
    return x;
}

/*
 * Faster, less random hash bucket compared to hash_bucket32, but works
 * for smaller integers.
 */
static inline uint32_t hash_mult32(uint32_t v, size_t width)
{
    /* Knuth's multiplicative hash. */
    return (v * UINT32_C(2654435761)) >> (32 - width);
}

#endif /* HASH_H */
