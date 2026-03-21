#ifndef INT_SET_H
#define INT_SET_H

#include "ptr_set.h"

/*
 * The values 0, 1, and 2 are reserved so we map integers
 * before casting them to void *.
 *
 * Instead we disallow the largest positive integers.
 *
 * This is specfic to the implementation of ptr_set, so
 * if it changes, we may have to change here as well.
 */

#define HT_INT_SET_OFFSET ((1 << (8 * sizeof(int) - 1)) - (size_t)2)
#define HT_INT_TO_PTR(x) ((void *)(size_t)((x) - HT_INT_SET_OFFSET))
#define HT_PTR_TO_INT(x) ((int)(size_t)(x) + HT_INT_SET_OFFSET)

/* Return value helpers. */
#define INT_SET_IS_MISSING(x) (HT_PTR_SET_MISSING(HT_INT_TO_PTR(x)))
#define INT_SET_IS_ERROR(x) (HT_PTR_SET_IS_ERROR(HT_INT_TO_PTR(x)))
#define INT_SET_IS_VALID(x) (HT_PTR_SET_IS_VALID(HT_INT_TO_PTR(x)))

typedef ptr_set_t int_set_t;

/* Returns 1 if already present, 0 otherwise. */
static inline int int_set_add(int_set_t *S, int x)
{
    return ptr_set_insert_item(S, HT_INT_TO_PTR(x), ht_keep) != 0;
}

/* Returns 1 if removed, 0 otherwise. */
static inline int int_set_remove(int_set_t *S, int x)
{
    return ptr_set_remove_item(S, HT_INT_TO_PTR(x)) != 0;
}

static inline int int_set_count(int_set_t *S)
{
    return ptr_set_count(S);
}

/* Returns 1 if present, 0 otherwise. */
static inline int int_set_exists(int_set_t *S, int x)
{
    return ptr_set_exists(S, HT_INT_TO_PTR(x));
}

#endif /* INT_SET_H */
