/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2017 Mikkel F. Jørgensen, dvide.com
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

#include "ht64.h"
#define HT_HASH_FUNCTION ht_uint64_hash_function

#include "hash_table_def.h"
DEFINE_HASH_TABLE(ht64)

#include "hash_table_impl.h"


static inline int ht_match(const void *key, size_t len, const ht64_item_t item)
{
    return *(const ht64_item_t)key == *item;
}

static inline const void *ht_key(const ht64_item_t item)
{
    return (const void *)item;
}

static inline size_t ht_key_len(const ht64_item_t item)
{
    return sizeof(*item);
}
