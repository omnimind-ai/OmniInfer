Generic hash table implementation with focus on being minimally
invasive on existing items to be indexed.

The key is stored arbitrarily in the referenced item. A custom match
function `HT_MATCH` provides the necessary abstraction. Items are
NOT allocated by the hash table.

Removed items are replaced with a sentinel value (1) to preserve
chaining.

See the example implementations `hash_set.h`,  `hash_item_table.h`,
and `hash_test.c`.

The hash function can also be customized, see the default below.

In all cases the key as assumed to be char string that is not
(necessarily) zero terminated. The length is given separately. Keys
can therefore be arbitrary binary values of arbitrary length.

Instead of initializing the hash table, it may be zeroed. In that
case the count defaults to 4 upon first insert, meaning it can hold
up to 4 items before resizing or less depending on load factor. By
zeroing memory, hash tables use no memory until actually used.

For increased portability we do not rely upon `stdint.h` outside the
default hash function.

Build
-----

There are no special build requirements.

CMakeLists.txt simply links the appropriate hash function with the test
files, but CMake is not required, for example:

    cc load_test.c ptr_set.c cmetrohash64.c -O4 -DNDEBUG -o load_test

There are several significant flags that can be set, but look at
`CMakeLists.txt`, `hash_test.c`, and `load_test.c`.

`initbuild.sh` is an easy way to create a CMake Ninja build for
platforms that support it.

Usage
-----

The hash table is implemented in a generic form with a static (private)
interface. The macros

`HASH_TABLE_HEADER(name, item)` defines the public prototype for the
specialized type, and `HASH_TABLE_API(name)` defines non-static wrapper
functions to access the generic implementation. This avoids creating all
the code as macros which are painful to develop and debug.

See `token_map.h`, `token_map.c` which are used in `hash_test.c`.

If the datatype is only needed in one file, the implementation such as
`token_map.c` can be included after defining `HT_PRIVATE`. This gives
the compiler better optimization opportunities and hides the interface
from other compilation units.

The basic datatype `hash_table_t` is a small struct that can be embedded
anywhere and used as the instance of any hash table derived type.


Note on algorithmic choice
--------------------------

We use linear or quadratic probing hash tables because it allows for
many small hash tables. We overallocate the hash table by a factor 2
(default) but only store a single pointer per item. This probing does
not allow for dense tables by itself, but because the hash table only
stores a single pointer per bucket, we can afford a larger table.
Advanced hashing such as Hopscotch can pack much more densely but
e.g. Hopscotch need to store a bitmask, thus already doubling the
size of the table. Hopscotch is probably good, but more complex and
depends on optimizing bit scan insructions, furthermore, when the use
case is many small tables such as symbol table scopes, cache locality
is less relevant. Chained hashing with 50% load factor is a good
choice, but require intrusive links, and cannot trivially hash string
sets without extra allocation. There is some evidence that linear
probing may be faster than quadratic probing due to cache effects, as
long as we do not pack too densely - however, the tradional quadratic
probing (k + i * i) modulo prime does not cover all buckets. We use
(k + i * (i + 1) / 2) modulo power of 2 which covers all buckets so
without experimentation it is unclear whether linear probing or
quadratic probing is best.

The use of open addressing leads to more key comparisons than chained
hashing. The fact we store the keys indirectly in the stored item is
also not ideal, except when the item is also directly the key. If we
use larger hash tables from the saved space, we suspect this will
still perform well, also considering external factors such as not
having to allocate and copy a key from e.g. a text buffer being
parsed.

It is generally understood that linear probing degrades significantly
with a load factor above 0.7. In this light, it is interesting to note
that Emmanuel Goossaert tested hopscotch hashing and found that bucket
swaps only take place in significance above a load factor of 0.7. A
commenter to Goossaert's blog also found that neighbourhoods rarely
exceed 64 even when allowed to grow on demand. Without deep analysis
it would appear that linear probing and hopscotch is pretty similar
at a load factor of 0.5 especially if tombstones are not present.
Because hopscotch requires extra data (e.g. the hash key or a bitmap
or a linked list) this confirms our intuition that it is better with
lower load factors and smaller buckets, than advanced algorithms.
Furthermore, hopscotch insert degrades badly when it needs to search for
empty buckets at high load factors. Of course, for on disk storage
it is a different matter, and this is why Goossaert is interested
in caching hash keys in buckets.

Robin Hood hashing is mostly interesting when there are many deletions
to clean up and when the load factor increases. In our implementation we
try to keep the per bucket size small: a pointer and a 8 bit offset, or
just a pointer for the linear and quadratic probing implementations.
This makes it affordable with a lower load factor.

This Robin Hood variation stores the offset from the hashed bucket to
where the first entry is stored. This means we can avoiding sampling any
bucket not indexed by the current hash key, and it also means that we
avoid having to store or calculate the hash key when updating.

A sorted Robin Hood hashing implementation was also made, but it prooved
to be error prone with many special cases and slower than regular Robin
Hood hashing. It would conceivably protect against hash collision
attacks through exponential search, but insertions and deletions would
still need to move memory in linear time, making this point mood.
Therefore the sorted Robin Hood variant has been removed.


Section 4.5:
<http://codecapsule.com/2014/05/07/implementing-a-key-value-store-part-6-open-addressing-hash-tables/>

<http://codecapsule.com/2013/08/11/hopscotch-hashing/>

Source file references
----------------------

<http://www.jandrewrogers.com/2015/05/27/metrohash/>

downloaded from

    <https://github.com/rurban/smhasher>
    <https://github.com/rurban/smhasher/commit/00a4e5ab6bfb7b25bd3c7cf915f68984d4910cfd>

    <https://raw.githubusercontent.com/rurban/smhasher/master/cmetrohash64.c>
    <https://raw.githubusercontent.com/rurban/smhasher/master/cmetrohash.h>
    <https://raw.githubusercontent.com/rurban/smhasher/master/PMurHash.c>
    <https://raw.githubusercontent.com/rurban/smhasher/master/PMurHash.h>

As of July 2015, for 64-bit hashes, the C port of the 64 bit metro hash
is a good trade-off between speed and simplicity. The For a 32-bit C hash
function, the ported MurmurHash3 is safe and easy to use in this
environment, but xxHash32 may also be worth considering.

See also <http://www.strchr.com/hash_functions>

xxhash.h was added in 2023 to replace Metrohash and Murmurhash by default.
See file header for credits.
Define the symbol HT_HASH_FALLBACK to revert back to the old hash functions.
The cmetrohash.c file does not need to be linked unless fallback is enabled.
