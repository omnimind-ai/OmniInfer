#ifndef HT_PTR_SET_H
#define HT_PTR_SET_H

#include "hash_table.h"

DECLARE_HASH_TABLE(ptr_set, void *)

/* Return value helpers - these are specific to the implementation. */
#define PTR_SET_IS_MISSING(x) ((void *)x == (void *)0)
#define PTR_SET_IS_ERROR(x) ((void *)x == (void *)2)
#define PTR_SET_IS_VALID(x) ((void *)x > (void *)2)

/* Extensions to std. interface. */
static inline int ptr_set_exists(ptr_set_t *S, void *p)
{
    return ptr_set_find_item(S, p) != (void *)0;
}

#endif /* HT_PTR_SET_H */
