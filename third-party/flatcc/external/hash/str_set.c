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

#include <string.h>

#include "str_set.h"
#include "hash_table_def.h"
DEFINE_HASH_TABLE(str_set)
#if defined(STR_SET_RH)
#include "hash_table_impl_rh.h"
#else
#include "hash_table_impl.h"
#endif

/*
 * Simple default implementation of a hash set. The stored items are
 * zero-terminated strings. The hash table does not manage the
 * allocation of the strings, like it doesn't manage any stored items.
 * However, it items are created with, say, strndup, a destructor can be
 * provided to free each item when clearing the table. The remove
 * operation also returns the removed item so it can be deallocated by
 * callee.
 *
 * In general, the key and the item are different, but here they are the
 * same. Normally the key would be referenced by the item.
 */
static inline int ht_match(const void *key, size_t len, str_set_item_t item)
{
    return strncmp(key, item, len) == 0 && item[len] == '\0';
}

static inline const void *ht_key(str_set_item_t item)
{
    return (const void *)item;
}

static inline size_t ht_key_len(str_set_item_t item)
{
    return strlen(item);
}
