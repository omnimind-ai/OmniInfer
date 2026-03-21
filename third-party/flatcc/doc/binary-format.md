# FlatBuffers Binary Format


<!-- vim-markdown-toc GFM -->

* [Overview](#overview)
* [Memory Blocks](#memory-blocks)
* [Example](#example)
* [Primitives](#primitives)
  * [Numerics](#numerics)
  * [Boolean](#boolean)
  * [Format Internal Types](#format-internal-types)
  * [Scalars](#scalars)
  * [Structs](#structs)
* [Internals](#internals)
* [Type Hashes](#type-hashes)
  * [Conflicts](#conflicts)
  * [Type Hash Variants](#type-hash-variants)
* [Unions](#unions)
* [Optional Fields](#optional-fields)
* [Alignment](#alignment)
* [Default Values and Deprecated Values](#default-values-and-deprecated-values)
* [Schema Evolution](#schema-evolution)
* [Keys and Sorting](#keys-and-sorting)
* [Size Limits](#size-limits)
* [Verification](#verification)
* [Risks](#risks)
* [Nested FlatBuffers](#nested-flatbuffers)
* [Fixed Length Arrays](#fixed-length-arrays)
* [Big Endian FlatBuffers](#big-endian-flatbuffers)
* [StructBuffers](#structbuffers)
* [StreamBuffers](#streambuffers)
* [Bidirectional Buffers](#bidirectional-buffers)
* [Possible Future Features](#possible-future-features)
  * [Force Align](#force-align)
  * [Mixins](#mixins)

<!-- vim-markdown-toc -->


## Overview

## Memory Blocks

A FlatBuffers layout consists of the following types of memory blocks:

- header
- table
- vector
- string
- vtable
- (structs)

Each of these have a contigous memory layout. The header references the
root table. Every table references a vtable and stores field in a single
block of memory. Some of these fields can be offsets to vectors, strings
and vectors. Vectors are one-dimensional blocks where each element is
self-contained or stores reference to table or a string based on schema
type information. A vtable decides which fields are present in table and
where they are stored. Vtables are the key to support schema evolution.
A table has no special rules about field ordering except fields must be
properly aligned, and if they are ordered by size the will pack more
densely but possibly more complicated to construct and the schema may
provide guidelines on the preferred order. Two vtables might mean
different things but accidentally have the same structure. Such vtables
can be shared between different tables. vtable sharing is important when
vectors store many similar tables. Structs a dense memory regions of
scalar fields and smaller structs. They are mostly found embedded in
tables but they are independent blocks when referenced from a union or
union vector, or when used as a buffer root. Structs hold no references.

Space between the above blocks are zero padded and present in order to
ensure proper alignment. Structs must be packed as densely as possible
according the alignment rules that apply - this ensures that all
implementations will agree on the layout. The blocks must not overlap in
memory but two blocks may be shared if they represent the same data such
as sharing a string.

FlatBuffers are constructed back to front such that lower level objects
such as sub-tables and strings are created first in stored last and such
that the root object is stored last and early in the buffer. See also
[Stream Buffers](#stream-buffers) for a proposed variation over this
theme.

All addressing in FlatBuffers are relative. The reason for this? When
navigating the buffer you don't need to store the buffer start or the
buffer length, you can also find a new reference relative the reference
you already have. vtables require the table location to find a field,
but a table field referencing a string field only needs the table field
location, not the table start in order to find the referenced string.
This results in efficient navigation.

## Example

Uoffsets add their content to their address and are positive while a
tables offset to its vtable is signed and is substracted. A vtables
element is added to the start of the table referring to the vtable.


Schema ([eclectic.fbs]) :

        namespace Eclectic;

        enum Fruit : byte { Banana = -1, Orange = 42 }
        table FooBar {
            meal      : Fruit = Banana;
            density   : long (deprecated);
            say       : string;
            height    : short;
        }
        file_identifier "NOOB";
        root_type FooBar;

JSON :

        { "meal": "Orange", "say": "hello", "height": -8000 }

Buffer :

        header:

            +0x0000 00 01 00 00 ; find root table at offset +0x00000100.
            +0x0004 'N', 'O', 'O', 'B' ; possibly our file identifier

            ...

        table:

            +0x0100 e0 ff ff ff ; 32-bit soffset to vtable location
                                ; two's complement: 2^32 - 0xffffffe0 = -0x20
                                ; effective address: +0x0100 - (-0x20) = +0x0120
            +0x0104 00 01 00 00 ; 32-bit uoffset string field (FooBar.say)
                                ; find string +0x100 = 256 bytes _from_ here
                                ; = +0x0104 + 0x100 = +0x0204.
            +0x0108 42d         ; 8-bit (FooBar.meal)
            +0x0109 0           ; 8-bit padding
            +0x010a -8000d      ; 16-bit (FooBar.height)
            +0x010c  ...        ; (first byte after table end)

            ...

        vtable:

            +0x0120 0c 00       ; vtable length = 12 bytes
            +0x0122 0c 00       ; table length = 12 bytes
            +0x0124 08 00       ; field id 0: +0x08 (meal)
            +0x0126 00 00       ; field id 1: <missing> (density)
            +0x0128 04 00       ; field id 2: +0004 (say)
            +0x012a 0a 00       ; field id 3: +0x0a (height)

            ...

        string:

            +0x0204 05 00 00 00 ; vector element count (5 ubyte elements)
            +0x0208 'h' 'e'     ; vector data
            +0x020a 'l' 'l'     ; vector data
            +0x020c 'o'         ; vector data
            +0x020d  00         ; zero termination
                                ; special case for string vectors

            ...


Actual FlatCC generated buffer :

    Eclectic.FooBar:
      0000  08 00 00 00 4e 4f 4f 42  e8 ff ff ff 08 00 00 00  ....NOOB........
      0010  2a 00 c0 e0 05 00 00 00  68 65 6c 6c 6f 00 00 00  *.......hello...
      0020  0c 00 0c 00 08 00 00 00  04 00 0a 00              ............


Note that FlatCC often places vtables last resulting in `e0 ff ff ff`
style vtable offsets, while Googles flatc builder typically places them
before the table resulting in `20 00 00 00` style vtable offsets which
might help understand why the soffset is subtracted from and not added
to the table start. Both forms are equally valid.


## Primitives

Primitives are data used to build more complex structures such as
strings, vectors, vtables and tables. Although structs are not strictly
a primitive, it helps to view them as self-contained primitives.


### Numerics

FlatBuffers are based on the following primitives that are 8, 16, 32 and
64 bits in size respectively (IEEE-754, two's complement, little endian):

    uint8, uint16, uint32, uint64 (unsigned)
    int8, int16, int32, int64     (two's complement)
    float32, float64              (IEEE-754)


### Boolean

    flatbuffers_bool (uint8)
    flatbuffers_true (flatbuffers_bool assign as 1, read as != 0)
    flatbuffers_false (flatbuffers_bool = 0)

Note that a C99 `bool` type has no defined size or sign so it is not an
exact representation of a flatbuffers boolean encoding.

When a stored value is interpreted as boolean it should not be assumed
to be either 1 or 0 but rather as `not equal to 0`. When storing a
boolean value or when converting a boolean value to integer before
storing, the value should be 1 for true and 0 for false, In C this can
be done using `!!x`.

### Format Internal Types

    flatbuffers_union_type_t (uint8, NONE = 0, 0 <= type <= 255)
    flatbuffers_identifier_t (uint8[4])
    flatbuffers_uoffset_t (uin32)
    flatbuffers_soffset_t (int32),
    flatbuffers_voffset_t (uint16)

These types can change in FlatBuffer derived formats, but when we say
FlatBuffers without further annotations, we mean the above sizes in
little endian encoding. We always refer to specific field type and not a
specific field size because this allows us to derive new formats easily.


### Scalars

To simpify discussion we use the term scalars to mean integers, boolean,
floating point and enumerations. Enumerations always have an underlying
signed or unsigned integer type used for its representation.

Scalars are primitives that has a size between 1 and 8 bytes. Scalars
are stored aligned to the own size.

The format generally supports vectors and structs of any scalar type but
some language bindings might not have support for all combinations such
as arrays of booleans.

An special case is the encoding of a unions type code which internally
is an enumeration but it is not normally permitted in places where we
otherwise allow for scalars.

Another special case is enumerations of type boolean which may not be
widely supported, but possible. The binary format is not concerned with
this distinction because a boolean is just an integer at this level.

### Structs

A struct is a fixed lengthd block of a fixed number of fields in a specific order
defined by a schema. A field is either a scalar, another struct or a fixed length
array of these, or a fixed length char array. A struct cannot contain fields that
contain itself directly or indirectly. A struct is self-contained and has no
references. A struct cannot be empty.

A schema cannot change the layout of a struct without breaking binary
compatibility, unlike tables.

When used as a table field, a struct is embedded within the table block
unless it is union value. A vector of structs are placed in a separate
memory block, similar to vectors of scalars. A vector of unions that has
a struct as member will reference the struct as an offset, and the
struct is then an independent memory block like a table.

FlatCC supports that a struct can be the root object of a FlatBuffer, but
most implementations likely won't support this. Structs as root are very
resource efficient.

Structs cannot have vectors but they can have fixed length array fields. which
are equivalent to stacking multiple non-array fields of the same type after each
other in a compact notation with similar alignment rules. Additionally arrays
can be of char type to have a kind of fixed length string. The char type is not
used outside of char arrays. A fixed length array can contain a struct that
contains one more fixed length arrays. If the char array type is not support, it
can be assumed to be a byte array.


## Internals

All content is little endian and offsets are 4 bytes (`uoffset_t`).
A new buffer location is found by adding a uoffset to the location where
the offset is stored. The first location (offset 0) points to the root
table. `uoffset_t` based references are to tables, vectors, and strings.
References to vtables and to table fields within a table have a
different calculations as discussed below.

A late addition to the format allow for adding a size prefix before the
standard header. When this is done, the builder must know about so it
can align content according to the changed starting position. Receivers
must also know about the size field just as they must know about the
excpected buffer type.

The next 4 bytes (`sizeof(uoffset_t)`) might represent a 4 byte buffer
identifier, or it might be absent but there is no obvious way to know
which.  The file identifier is typically ASCII characters from the
schemas `file_identifier` field padded with 0 bytes but may also contain
any custom binary identifier in little endian encoding. See
[Type-Hashes](#type-hashes). The 0 identifier should be avoided because
the buffer might accidentally contain zero padding when an identifier is
absent and because 0 can be used by API's to speficy that no identifier
should be stored.

When reading a buffer, it should be checked that the length is at least
8 bytes (2 * `sizeof(uoffset_t)`) Otherwise it is not safe to check the
file identifier.

The root table starts with a 4 byte vtable offset (`soffset_t`). The
`soffset_t` has the same size as `uoffset_t` but is signed.

The vtable is found by *subtracting* the signed 4 byte offset to the
location where the vtable offset is stored. Note that the `FlatCC`
builder typically stores vtables at the end of the buffer (clustered
vtables) and therefore the vtable offset is normally negative. Other
builders often store the vtable before the table unless reusing an
existing vtable and this makes the soffset positive.
(Nested FlatBuffers will not store vtables at the end because it would
break compartmentalization).

The vtable is a table of 2 byte offsets (`voffset_t`). The first two
entreis are the vtable size in bytes and the table size in bytes. The
next offset is the vtable entry for table field 0. A vtable will always
have as many entries as the largest stored field in the table, but it
might skip entries that are not needed or known (version independence) -
therefore the vtable length must be checked. The table length is only
needed by verifiers. A vtable lookup of an out of range table id is the
same as a vtable entry that has a zero entry and results in a default
value for the expected type. Multiple tables may shared the same vtable,
even if they have different types. This is done via deduplication during
buffer construction. A table field id maps to a vtable offset using the
formula `vtable-start + sizeof(voffset_t) * (field-id + 2)`, iff the
result is within the vtable size.

The vtable entry stores a byte offset relative to the location where the
`soffset_t` where stored at the start of the table. A table field is
stored immediately after the `soffset_t` or may contain 0 padding. A
table field is always aligned to at least its own size, or more if the
schema demands it. Strings, sub-tables and vectors are stored as a
4-byte `uoffset_t`. Such sub-elements are always found later in the
buffer because `uoffset_t` is unsigned. A struct is stored in-place.
Conceptually a struct is not different from an integer in this respect,
just larger.

If a sub-table or vector is absent and not just without fields, 0
length, the containing table field pointing the sub-table or vector
should be absent. Note that structs cannot contain sub-tables or
vectors.

A struct is always 0 padded up to its alignment. A structs alignment
is given as the largest size of any non-struct member, or the alignment
of a struct member (but not necessarily the size of a struct member), or
the schema specified aligment.

A structs members are stored in-place so the struct fields are known
via the the schema, not via information in the binary format

An enum is represent as its underlying integer type and can be a member
of structs, fields and vectors just like integer and floating point
scalars.

A table is always aligned to `sizeof(uoffset_t)`, but may contain
internal fields with larger alignment. That is, the table start or end
are not affected by alignment requirements of field members, unlike
structs.

A string is a special case of a vector with the extra requirement that a
0 byte must follow the counted content, so the following also applies to
strings.

A vector starts with a `uoffset_t` field the gives the length in element
counts, not in bytes. Table fields points to the length field of a
vector, not the first element. A vector may have 0 length. Note that the
FlatCC C binding will return a pointer to a vectors first element which
is different from the buffer internal reference.

All elements of a vector has the same type which is either a scalar type
including enums, a struct, or a uoffset reference where all the
reference type is to tables all of the same type, all strings, or,
for union vectors, references to members of the same union. Because a
union member can be a struct, it is possible to have vectors that
reference structs instead of embeddding them, but only via unions. It is
not possible to have vectors of vectors other than string vectors,
except indirectly via vectors containing tables.

A vectors first element is aligned to the size of `uoffset_t` or the
alignment of its element type, or the alignment required by the schema,
whichever is larger. Note that the vector itself might be aligned to 4
bytes while the first element is aligned to 8 bytes.

(Currently the schema semantics do no support aligning vectors beyond
their element size, but that might change and can be forced when
building buffers via dedicated api calls).

Strings are stored as vectors of type `ubyte_t`, i.e. 8-bit elements. In
addition, a trailing zero is always stored. The trailing zero is not
counted in the length field of the vector. Technically a string is
supposed to be valid UTF-8, but in praxis it is permitted that strings
contain 0 bytes or other invalid sequences. It is recommended to
explicitly check strings for UTF-8 conformance when this is required
rather than to expect this to alwways be true. However, the ubyte vector
should be preferred for binary data.

A string, vector or table may be referenced by other tables and vectors.
This is known as a directed acyclic graph (DAG). Because uoffsets are
unsigned and because uoffsets are never stored with a zero value (except
for null entries in union vectors), it is not possible for a buffer to
contain cycles which makes them safe to traverse with too much fear of
excessive recursion. This makes it possible to efficiently verify that a
buffer does not contain references or content outside of its expected
boundaries.

A vector can also hold unions, but it is not supported by all
implementations. A union vector is in reality two separate vectors: a
type vector and an offset vector in place of a single unions type and
value fields in table. See unions.


## Type Hashes

A type hash is a 32-bit value defined as the fnv1a32 hash of a tables or
a structs fully qualified name. If the fnv1a32 hash returns 0 it should
instead hash the empty string. 0 is used to indicate that a buffer
should not store an identifier.

Every table and struct has a name and optionally also a namespace of one
or more levels. The fully qualified name is optional namespace followed
by the type name using '.' as a separator. For example
"MyGame.Sample.Monster", or "Eclectic.FooBar".

The type hash can be used as the buffer identifier instead of the schema
provided `file_identifier`. The type hash makes it possible to
distinguish between different root types from the same schema, and even
across schema as long as the namespace is unique.

Type hashes introduce no changes to the binary format but the application
interface must choose to support user defined identifiers or explicitly
support type hashes. Alternatively an application can peak directly into
the buffer at offset 4 (when `uoffset_t` is 4 bytes long).

FlatCC generates the following identifier for the "MyGame.Sample.Monster"
table:

    #define MyGame_Sample_Monster_type_hash ((flatbuffers_thash_t)0xd5be61b)
    #define MyGame_Sample_Monster_type_identifier "\x1b\xe6\x5b\x0d"

But we can also
[compute one online](https://www.tools4noobs.com/online_tools/hash/)
for our example buffer:

        fnv1a32("Eclectic.FooBar") = 0a604f58

Thus we can open a hex editor and locate

            +0x0000 00 01 00 00 ; find root table at offset +0x00000100.
            +0x0004 'N', 'O', 'O', 'B' ; possibly our file identifier

and replace it with

            +0x0000 00 01 00 00 ; find root table at offset +0x00000100.
            +0x0004 58 4f 60 0a ; very likely our file identifier identifier

or generate it with `flatcc`:

        $ bin/flatcc --stdout doc/eclectic.fbs | grep FooBar_type
        #define Eclectic_FooBar_type_hash ((flatbuffers_thash_t)0xa604f58)
        #define Eclectic_FooBar_type_identifier "\x58\x4f\x60\x0a"


The following snippet implements fnv1a32, and returns the empty string
hash if the hash accidentially should return 0:


    static inline flatbuffers_thash_t flatbuffers_type_hash_from_name(const char *name)
    {
        uint32_t hash = 2166136261UL;
        while (*name) {
            hash ^= (uint32_t)*name;
            hash = hash * 16777619UL;
            ++name;
        }
        if (hash == 0) {
            hash = 2166136261UL;
        }
        return hash;
    }

### Conflicts

It is possible to have conficts between two type hashes although the
risk is small. Conflicts are not important as long as an application can
distinguish between all the types it may encouter in actual use. A
switch statement in C will error at compile time for two cases that have
the same value, so the problem is easily detectable and fixable by
modifying a name or a namespace.

For global conflict resolution, a type should be identified by its fully
qualified name using adequate namespaces. This obviously requires it to
be stored separate from the buffer identifier due to size constraints.


### Type Hash Variants

If an alternative buffer format is used, the type hash should be
modified. For example, if `uoffset_t` is defined as a 64-bit value, the
fnv1a64 hash should be used instead. For big endian variants the hash
remains unchanged but is byteswapped. The application will use the same
id while the acces layer will handle the translation.

For buffers using structs as roots, the hash remains unchanged because
the struct is a unique type in schema. In this way a receiver that does
not handle struct roots can avoid trying to read the root as a table.

For futher variations of the format, a format identifier can be inserted
in front of the namespace when generating the hash. There is no formal
approach to this, but as an example, lets say we want to use only 1 byte
per vtable entry and identify these buffers with type hash using the
prefix "ebt:" for example buffer type. We then have the type hash:

    #define type_hash_prefix "ebt:"

    hash = fnv1a32(type_hash_prefix "Eclectic.FooBar");
    hash = hash ? hash : fnv1a32(type_hash_prefix);

If the hash returns 0 we hash the prefix.


## Unions

A union is a contruction on top of the above primitives. It consists of
a type and a value.

In the schema a union type is a set of table types with each table name
assigned a type enumeration starting from 1. 0 is the type NONE meaning
the union has not value assigned. The union type is represented as a
ubyte enum type, or in the binary format as a value of type
`union_type_t` which for standard FlatBuffers is an 8-bit unsigned code
with 0 indicating the union stores not value and a non-zero value
indicating the type of the stored union.

A union is stored in a table as a normal sub-table reference with the
only difference being that the offset does not always point to a table
of the same type. The 8-bit union type is stored as a separate table
field conventially named the same as the union value field except for a
`_type` suffix. The value (storing the table offset) MUST have a field
ID that is exactly one larger than the type field. If value field is
present the type field MUST also be present. If the type is NONE the
value field MUST be absent and the type field MAY be absent because a
union type always defaults to the value NONE.

Vectors of unions is a late addition (mid 2017) to the FlatBuffers
format. FlatCC supports union vectors as of v0.5.0.

Vectors of unions have the same two fields as normal unions but they
both store a vector and both vectors MUST have the same length or both
be absent from the table. The type vector is a vector of 8-bit enums and
the value vector is a vector of table offsets. Obviously each type
vector element represents the type of the table in the corresponding
value element. If an element is of type NONE the value offset must be
stored as 0 which is a circular reference. This is the only offset that
can have the value 0.

A later addition (mid 2017) to the format allows for structs and strings
to also be member of a union. A union value is always an offset to an
independent memory block. For strings this is just the offset to the
string. For tables it is the offset to the table, naturally, and for
structs, it is an offset to an separate aligned memory block that holds
a struct and not an offset to memory inside any other table or struct.
FlatCC supports mixed type unions and vectors of these as of v0.5.0.

## Optional Fields

As of mid 2020 the FlatBuffers format introduced optional scalar table fields.
There is no change to the binary schema, but the semantics have changed slightly
compared to ordinary scalar fields (which remain supported as is): If an
optional field is not stored in a table, it is considered to be a null value. An
optinal scalar field will have null as its default value, so any representable
scalar value will always be stored in the buffer, unlike other scalar fields
which by default do not store the field if the value matches the default numeric
value. This was already possible before by using `force_add` semantics to force
a value to be written even if it was matching the default value, and by
providing an `is_present` test when reading a value so that it would be possible
to distinguish between a value that happened to be a default value, and a value
that was actually absent. However, such techniques were ad-hoc. Optional
fields formalize the semantics of null values for scalars. Other field types
already have meaningful null values. Only table fields can be optional so struct
fields must always assign a value to all members.

## Alignment

All alignments are powers of two between 1 and 256. Large alignments are
only possible via schema specified alignments. The format naturally has
a maximum alignment of the largest scalar it stores, which is a 64-bit
integer or floating point value. Because C malloc typically returns
buffers aligned to a least 8 bytes, it is often safe to place buffers in
heap allocated memory, but if the target system does not permit
unaligned access, or is slow on unaligned access, a buffer should be
placed in sufficiently aligned memory. Typically it is a good idea to
place buffer in cacheline aligned boundary anyway.

A buffers alignment is the same as the largest alignment of any
object or struct it contains.

The buffer size is not guaranteed to be aligned to its own alignment
unlike struct. Googles `flatc` builder does this, at least when size
prefixed. The `FlatCC` tool currently does not, but it might later add
an option to pad up to alignment. This would make it simpler to stack
similar typed buffers in file - but users can retrieve the buffers
alignment and do this manually. Thus, when stacking size prefixed
buffers, each buffer should start aligned to its own size starting at
the size field, and should also be zero padded up to its own alignment.


## Default Values and Deprecated Values

A table can can omit storing any field that is not a required field. For
strings, vectors and tables this result in returning a null value
different from an empty value when reading the buffer. Struct fields not
present in table are also returned as null.

All fields that can return null do not have a default value. Other
values, which are integers, floats and enumerations, can all have
default values. The default value is returned if not found in the table.

If a default value is not specified the default defaults to zero for the
corresponding type. If an enumeration does not have a 0 value and no
explicit default value, this is a schema error.

When building a buffer the builder will compare a stored field to the
known default value. If it matches the field will simple be skipped.
Some builder API's makes it possible to force a default value to be
stored and to check if a field is missing when reading the buffer. This
can be used to handle NULL values differently from default or missing
values.

A deprecated field is a schema construct. The binary format either stores
a table field, or it does not.

A deprecated field should be treated as not available, as in no way to
read the value as opposed to returning a default value regardless of
whether the field is present or not. If they for some reason are made
accessible the verifier must also understand and verify these fields.

A deprecated field also cannot be written to a new buffer, although if
it against guidelines remains possible to do so, it should be done as if
the field was not deprecated.

Structs cannot have default values and cannot have deprecated fields in
stadard FlatBuffers. FlatCC supports marking a struct field as
deprecated. This implies the field will always be zeroed and with no
trivial accessors. A struct can never change size without breaking
support for schema evolution.

FlatCC JSON parsers allow structs to only set some values. Remaining
values will be implicitly zeroed. The C API for writing buffers do not
have this concern because a struct can just be written as a C struct
so there is no control over which fields a user choose to set or zero.
However, structs should be zeroed and padded where data is not otherwise
set. This makes it possible to hash and integrity check structs (though
this is not an integral part of the format).


## Schema Evolution

A table has a known set of low-valued field identifiers. Any unused
field id can be used in a future version. If a field (as is normal) is
implicitly assigned an id, new fields can only be added at the end of
the table. Internally this translates into new versions getting ever
larger vtables. Note that vtables are only stored as large as needed for
the actual content of table, so a rarely used new field will not cause
vtables to grew when unused.

Enumarations may not change values in future versions. Unions may only
added new table names to the end of the union list.

Structs cannot change size nor content. They cannot evolve. FlatCC
permits deprecating fields which means old fields will be zeroed.

Names can be changed to the extend it make sense to the applications already
written around the schema, but it may still break applications relying
on some reflective information. For example, a reader may provide the
string representation of a numeric enum value.

New types can be added. For example adding a new table is always safe
as long as it does not conflict with any existing schemas using the same
namespace.

Required fields cannot stop being required and they cannot be deprecated.

Various attributes and other changes form a gray area that will not make
the binary format unsafe but may still break due to changes in code
generation, serialization to JSON, or similar. For example, a generated
constructor that creates a table from a list of positional arguments
might break if the field order changes or grows or have fields
deprecated. JSON parsers could cease to work on old formats if base64
serialization is added subsequently, and so on.

## Keys and Sorting

Keys and sorting is a meta construct driven by the schema. The binary
format has no special concept of keys and sorting and a vector can be
sorted by one of several keys so it makes no sense to enforce a specific
order.

The basic FlatBuffers format only permit at most one key and generally
sorts vectors by that key during buffer construction. FlatCC does not do
this both because sorting is not practical while building the buffer and
because FlatCC supports sorting by one of several keys. Thus, in general
it is not safe to assume that a vector is sorted, but it can be sorted
if needed.

## Size Limits

A buffer should never be larger than `2^(sizeof(soffset_t) * 8 - 1) - 1`
or `2^31 - 1` i.e. 2GB for standard FlatBuffers. Beyond this safe it is
not safe to represent vtable offsets and implementations can no longer
use signed types to store `uoffset_t` values. This limit also ensures
that all vectors can be represented safely with a signed 32-bit length
type.

The application interface is likely to use a native type for
representing sizes and vector indices. If this type is smaller that
`sizeof(soffset_t)` or equivalently `sizeof(uoffset_t)` there is a risk
of overflow. The simplest way to avoid any issues is to limit the
accepted buffer size of the native size type. For example, on some
embedded microprocessor systems a C compiler might have a 16-bit int and
`size_t` type, even if it supports `uint32_t` as well. In this the safe
assumption is to limit buffers to `2^15 - 1` which is very likely more
than sufficient on such systems.

A builder API might also want to prevent vectors from being created when
they cannot stay within the size type of the platform when the element
size is multipled by the element count. This is deceiving because the
element count might easily be within range. Such issues will be rare in
praxis but they can be the source of magnificent vulnerabilites if not
handled appropriately.


## Verification

Verification as discussed here is not just about implementing a
verifier. It is as much requirements that any builder must fulfill.

The objective of verification is to make it safe to read from an
untrusted buffer, or a trusted buffer that accidentally has an
unexpected type. The verifier does not guarantee that the type is indeed
the expeced type exact it can read the buffer identifier if present
which is still no guarantee. The verifier cannot make it safe to modify
buffers in-place because the cost of doing such a check would be
prohitive in the average case.

A buffer verifier is expected to verify that all objects (strings,
vectors and tables) do not have an end position beyond the externally
specified buffer size and that all offset are aligned relative to offset
zero, and sometimes also relative to actually specified buffer (but
sometimes it is desireable to verify buffers that are not stored
aligned, such as in network buffers).

A verifier primarily checks that:

- the buffer is at least `2 * sizeof(uoffset_t)` such that the root
  root table and buffer identifier can be checked safely.
- any offset being chased is inside the buffer that any data accessed
  from that resuling location is entirely inside the buffer and aligned
  notably the vtables first entry must be valid before the vtable can
  safely validate itself, but this also applies to table, string and
  vector fields.
- any uoffset has size of at least `sizeof(uoffse_t)` to aviod
  self-reference and is no larger than the largest positive soffset_t
  value of same size - this ensures that implementations can safely add
  uoffset_t even if converting to signed first. It also, incidentally,
  ensures compatibility with StreamBuffers - see below.
- vtable size is aligned and does not end outside buffer.
- vtable size is at least the two header fields
  (`2 * `sizeof(voffset_t)`).
- required table fields are present.
- recursively verify all known fields and ignore other fields. Unknown
  fields are vtable entries after the largest known field ID of a table.
  These should be ignored in order to support forward versioning.
- deprecated fields are valid if accessors are available to do so, or
  ignore if the there is no way to access the field by application code.
- vectors end within the buffer.
- strings end within the buffer and has a zero byte after the end which
  is also within the buffer.
- table fields are aligned relative to buffer start - both structs,
  scalars, and offset types.
- table field size is aligned relative to field start.
- any table field does not end outside the tables size as given by the
  vtable.
- table end (without chasing offsets) is not outside buffer.
- all data referenced by offsets are also valid within the buffer
  according the type given by the schema.
- verify that recursion depth is limited to a configurable acceptable
  level for the target system both to protect itself and such that
  general recursive buffer operations need not be concerned with stack
  overflow checks (a depth of 100 or so would normally do).
- verify that if the union type is NONE the value (offset) field is absent and
  if it is not NONE that the value field is present. If the union type
  is known, the table should be verified. If the type is not known
  the table field should be ignored. A reader using the same schema would
  see the union as NONE. An unknown union is not an error in order to
  support forward versioning.
- verifies the union value according to the type just like any other
  field or element.
- verify that a union vector always has type vector if the offset vector
  is present and vice versa.

A verifier may choose to reject unknown fields and union types, but this
should only be an user selectable option, otherwise schema evolution
will not be possible.

A verifier needs to be very careful in how it deals with overflow and
signs. Vector elements multiplied by element size can overflow. Adding
an invalid offset might become valid due to overflow. In C math on
unsigned types yield predictable two's complement overflow while signed
overflow is undefined behavior and can and will result in unpredictable
values with modern optimizing compilers. The upside is that if the
verifier handles all this correctly, the application reader logic can be
much simpler while staying safe.


A verifier does __not__ enforce that:

- structs and other table fields are aligned relative to table start because
  tables are only aligned to their soffset field. This means a table cannot be
  copied naively into a new buffer even if it has no offset fields.
- the order of individual fields within a table. Even if a schema says
  something about ordering this should be considered advisory. A
  verifier may additionally check for ordering for specific
  applications. Table order does not affect safety nor general buffer
  expectations. Ordering might affect size and performance.
- sorting as specified by schema attributes. A table might be sorted in
  different ways and an implementation might avoid sorting for
  performance reasons and other practical reasons. A verifier may,
  however, offer additional verification to ensure specific vectors or
  all vectors are sorted according to schema or other guidelines.  Lack
  of sorting might affect expected behavior but will not make the buffer
  unsafe to access.
- that structures do not overlap. Overlap can result in vulnerabilities
  if a buffer is modified, and no sane builder should create overlaps
  other than proper DAGs except via a separate compression/decopression
  stage of no interest to the verifier.
- strings are UTF-8 compliant. Lack of compliance may affect expections
  or may make strings appear shorter or garbled. Worst case a naive
  UTF-8 reader might reach beyond the end when observing a lead byte
  suggest data after buffer end, but such a read should discover the
  terminal 0 before things get out of hand. Being relaxed permits
  specific use cases where UTF-8 is not suitable without giving up the
  option to verify buffers. Implementations can add additional
  verification for specific usecases for example by providing a
  strict-UTF8 flag to a verifier or by verifying at the application
  level. This also avoids unnecessary duplicate validation, for example
  when an API first verifies the buffer then converts strings to an
  internal heap representation where UTF-8 is validated anyway.
- default values are not stored. It is common to force default values to
  be stored. This may be used to implement NULL values as missing
  values different from default values.
- enum values are only among the enum values of its type. There are many
  use cases where it is convenient to add enum values for example flags
  or enums as units e.g. `2 * kilo + 3 * ounce. More generally ordinary
  integers may have value range restrictions which is also out of scope
  for verifier. An application may provide additional verification when
  requested.

More generally we can say that a verifier is a basic fast assurance that
the buffer is safe to access. Any additional verification is application
specific. The verifier makes it safe to apply secondary validation.
Seconary validation could be automated via schema attributes and may be
very useful as such, but it is a separate problem and out of scope for a
core binary format verifier.

A buffer identifier is optional so the verifier should be informed
whether an identifier must match a given id. It should check both ASCII
text and zero padding not just a string compare. It is non-trivial to
decide if the second buffer field is an identifier, or some other data,
but if the field does not match the expected identifier, it certainly
isn't what is expected.

Note that it is not entirely trivial to check vector lengths because the
element size must be mulplied by the stored element count. For large
elements this can lead to overflows.

NOTE: as of August 2024 it was discovered that a decode long bug in the
C++ builder could result in misaligned vectors when the the length
of the vector was zero and the alignment size larger than the header
fields alignment. In praxis this has not cause any known issues except
that the FlatCC verifier has rejected such buffers. However, it is
problemetatic because it can result in pointers that are not correctly
aligned in C/C++, which is undefined behaviour. Because there are too
presumably too many comprimised buffers in the wild at this point,
combined with the fact that they are currently mostly harmless,
it it recommended to have an option to tolerate misaligned zero-length
vectors as long as the length field is still properly aligned.


## Risks

Because a buffer can contain DAGs constructed to explode exponentiall,
it can be dangerous to print JSON or otherwise copy content blindly
if there is no upper limit on the export size.

In-place modification cannot be trusted because a standard buffer
verifier will detect safe read, but will not detect if two objects are
overlapping. For example, a table could be stored inside another table.
Modifing one table might cause access to the contained table to go out
of bounds, for example by directing the vtable elsewhere.

The platform native integer and size type might not be able to handle
large FlatBuffers - see [Size Limits](#size-limits).

Becaue FlatCC requires buffers to be sorted after builiding, there is
risk due to buffer modifications. It is not sufficient to verify buffers
after sorting because sorting is done inline. Therefore buffers must be
trusted or rewritten before sorting.


## Nested FlatBuffers

FlatBuffers can be nested inside other FlatBuffers. In concept this is
very simple: a nested buffer is just a chunk of binary data stored in a
`ubyte` vector, typically with some convenience methods generated to
access a stored buffer. In praxis it adds a lot of complexity. Either
a nested buffer must be created strictly separately and copied in as
binary data, but often users mix the two builder contexts accidentally
storing strings from one buffer inside another. And when done right, the
containing ubyte vector might not be aligned appropriately making it
invalid to access the buffer without first copying out of the containing
buffer except where unaligned access is permitted. Further, a nested
FlatBuffer necessarily has a length prefix because any ubyte vector has
a length field at its start. Therefore, size prefixed flatbuffers should
not normally be stored as nested buffers, but sometimes it is necessary
in order have the buffer properly aligned after extraction.

The FlatCC builder makes it possible to build nested FlatBuffers while
the containing table of the parent buffer is still open. It is very
careful to ensure alignment and to ensure that vtables are not shared
between the two (or more) buffers, otherwise a buffer could not safely
be copied out. Users can still make mistakes by storing references from
one buffer in another.

Still, this area is so complex that several bugs have been found.
Thus, it is advice to use nested FlatBuffers with some care.

On the other hand, nested FlatBuffers make it possible to trivially
extract parts of FlatBuffer data. Extracting a table would require
chasing pointers and could potentially explode due to shared sub-graphs,
if not handled carefully.

## Fixed Length Arrays

This feature can be seen as equivalent to repeating a field of the same type
multiple times in struct.

Fixed length array struct fields has been introduced mid 2019.

A fixed length array is somewhat like a vector of fixed length containing inline
fixed length elements with no stored size header. The element type can be
scalars, enums and structs but not other fixed length errors (without wrapping
them in a struct).

An array should not be mistaken for vector as vectors are independent objects
while arrays are not. Vectors cannot be fixed length. An array can store fixed
size arrays inline by wrapping them in a struct and the same applies to unions.

The binary format of a fixed length vector of length `n` and type `t` can
be precisely emulated by created a struct that holds exactly `n` fields
of type `t`, `n > 0`. This means that a fixed length array does not
store any length information in a header and that it is stored inline within
a struct. Alignment follows the structs alignment rules with arrays having the
same alignment as their elements and not their entire size.

The maximum length is limited by the maximum struct size and / or an
implementation imposed length limit. Flatcc accepts any array that will fit in
struct with a maximum size of 2^16-1 by default but can be compiled with a
different setting. Googles flatc implementation currently enforces a maximum
element count of 2^16-1.

Assuming the schema compiler computes the correct alignment for the overall
struct, there is no additonal work in verifying a buffer containing a fixed
length array because structs are verified based on the outermost structs size
and alignment without having to inspect its content.

Fixed lenght arrays also support char arrays. The `char` type is similar to the
`ubyte` or `uint8` type but a char can only exist as a char array like
`x:[char:10]`. Chars cannot exist as a standalone struct or table field, and
also not as a vector element. Char arrays are like strings, but they contain no
length information and no zero terminator. They are expected to be endian
neutral and to contain ASCII or UTF-8 encoded text zero padded up the array
size. Text can contain embedded nulls and other control characters. In JSON form
the text is printed with embedded null characters but stripped from trailing
null characters and a parser will padd the missing null characters.


The following example uses fixed length arrays. The example is followed by the
equivalent representation without such arrays.

    struct Basics {
        int a;
        int b;
    }

    struct MyStruct {
        x: int;
        z: [short:3];
        y: float;
        w: [Basics:2];
        name: [char:4];
    }

    // NOT VALID - use a struct wrapper:
    table MyTable {
        t: [ubyte:2];
        m: [MyStruct:2];
    }

Equivalent representation:

    struct MyStructEquivalent {
        x: int;
        z1: short;
        z2: short;
        z3: short;
        y: float;
        wa1: int;
        wa2: int;
        name1: ubyte;
        name2: ubyte;
        name3: ubyte;
        name4: ubyte;
    }

    struct MyStructArrayEquivalent {
        s1: MyStructEquivalent;
        s2: MyStructEquivalent;
    }

    struct tEquivalent {
        t1: ubyte;
        t2: ubyte;
    }

    table MyTableEquivalent {
        t: tEquivalent;
        m: MyStructArrayEquivalent;
    }


Note that forced zero-termination can be obtained by adding a trailing ubyte
field since uninitialized struct fields should be zeroed:

    struct Text {
        str: [char:255];
        zterm: ubyte;
    }

Likewise a length prefix field could be added if the applications involved know
how to interpret such a field:

    struct Text {
        len: ubyte;
        str: [char:254];
        zterm: ubyte;
    }

The above is just an example and not part of the format as such.


## Big Endian FlatBuffers

FlatBuffers are formally always little endian and even on big-endian
platforms they are reasonably efficient to access.

However it is possible to compile FlatBuffers with native big endian
suppport using the FlatCC tool. The procedure is out of scope for this
text, but the binary format is discussed here:

All fields have exactly the same type and size as the little endian
format but all scalars including floating point values are stored
byteswapped within their field size. Offset types are also byteswapped.

The buffer identifier is stored byte swapped if present. For example
the 4-byte "MONS" identifier becomes "SNOM" in big endian format. It is
therefore reasonably easy to avoid accidentially mixing little- and
big-endian buffers. However, it is not trivial to handle identifers that
are not exactly 4-bytes. "HI\0\0" could be "IH\0\0" or "\0\0IH". It is
recommended to always use 4-byte identifies to avoid this problem. See
the FlatCC release 0.4.0 big endian release for details.

When using type hash identifiers the identifier is stored as a big
endian encoded hash value. The user application will the hash in its
native form and accessor code will do the conversion as for other
values.


## StructBuffers

_NOTE: the Google FlatBuffer project originally documented structs as
valid root objects, but never actually implemented it, and has as of mid
2020 changed the specification to disallow root structs as covered in
this section. FlatCC for C has been supporting root structs for a long
time, and they can provide significant speed advantages, so FlatCC will
continue to support these._

Unlike tables, structs are are usually embedded in in a fixed memory
block representing a table, in a vector, or embedded inline in other
structs, but can also be independent when used in a union.

The root object in FlatBuffers is conventionally expected to be a table,
but it can also be struct. FlatCC supports StructBuffers. Since structs
do not contain references, such buffers are truly flat. Most
implementations are not likely to support structs are root but even so
they are very useful:

It is possible to create very compact and very fast buffers this way.
They can be used where one would otherwise consider using manual structs
or memory blocks but with the advantage of a system and language
independent schema.

StructBuffers may be particularly interesting for the Big Endian
variant of FlatBuffers for two reasons: the first being that performance
likely matters when using such buffers and thus Big Endian platforms
might want them. The second reason is the network byte order is
traditionally big endian, and this has proven very difficult to change,
even in new evolving IETF standards. StructBuffers can be used to manage
non-trival big endian encoded structs, especially structs containing
other structs, even when the receiver does not understand FlatBuffers as
concept since the header can just be dropped or trivially documented.


## StreamBuffers

StreamBuffers are so far only a concept although some implementations
may already be able to read them. The format is documented to aid
possible future implementations.

StreamBuffers makes it possible to store partially completed buffers
for example by writing directly to disk or by sending partial buffer
data over a network. Traditional FlatBuffers require an extra copying
step to make this possible, and if the writes are partial, each section
written must also store the segment length to support reassembly.
StreamBuffers avoid this problem.

StreamBuffers treat `uoffset_t` the same as `soffset_t` when the special
limitation that `uoffset_t` is always negative when viewed as two's
complement values.

The implication is that everthing a table or vector references must be
stored earlier in the buffer, except vtables that can be stored freely.
Existing reader implementations that treat `uoffset_t` as signed, such as
Javascript, will be able to read StreamBuffers with no modification.
Other readers can easily be modified by casting the uoffset value to
signed before it.

Verifiers must ensure that any buffer verified always stores all
`uoffset_t` with the same sign. This ensures the DAG structure is
preserved without cycles.

The root table offset remains positive as an exception and points to the
root buffer. A stream buffer root table will be the last table in the
buffer.

The root table offset may be replaced with a root table offset at the
end of the buffer instead of the start. An implementation may also
zero the initial offset and update it later. In either case the buffer
should be aligned accordingly.

Some may prefer the traditional FlatBuffer approach because the root
table is stored and it is somehow easier or faster to access, but modern
CPU cache systems do not care much about the order of access as long as
as their is some aspect of locality of reference and the same applies to
disk access while network access likely will have the end of the buffer
in hot cache as this is the last sent. The average distance between
objects will be the same for both FlatBuffers and StreamBuffers.


## Bidirectional Buffers

Bidirectional Buffers is a generalization of StreamBuffers and FlatBuffers
and so far only exists as an idea.

FlatBuffers and StreamBuffers only allow one direction because it
guarantees easy cycle rejection. Cycles are unwanted because it is
expensive to verify buffers with cycles and because recursive navigation
might never terminate.

As it happens, but we can also easily reject cycles in bidirectional
buffers if we apply certain constraints that are fully backwards
compatible with existing FlatBuffers that have a size below 2GB.

The rule is that when an offset changes direction relative to a parent
objects direction, the object creating the change of direction becomes a
new start or end of buffer for anything reachable via that offset.

The root table R is reached via forward reference from the buffer
header.  Its boundaries are itself and the buffer end. If the this table
has an offset pointing back, for example to new table X, then table X
must see the buffer start and R as two boundaries. X must not directly
or indirectly reach outside this region. Likewise, if the table R points
to new table Y in the forward direction, then Y is bounded by itself and
the buffer end.

A lower bound is the end of a table or a vector althoug we just say the
table or vector is a boundary. An upper bound is until the start of the
table or vector, or the end of the buffer.

When a buffer is verified, the boundaries are initially the buffer start
and buffer end. When the references from the root table are followed,
there new boundaries are either buffer start to root table, or root
table to end, depending on the offsets direction. And so forth
recursively.

We can relax the requirement that simple vectors and vtables cannot
cross these boundaries, except for the hard buffer limits, but it does
compicate things and is likely not necessary.

Nested FlatBuffers already follow similar boundary rules.

The point of Bidirectional buffers is not that it would be interesting
to store things in both directions, but to provide a single coherent
buffer model for both ways of storing buffers without making a special
case. This will allow StreamBuffers to co-exist with FlatBuffers without
any change, and it will allow procedures to build larger buffers to make
their own changes on how to build their subsection of the buffer.

We still need to make allowance for keeping the buffer headers root
pointer implicit by context, at least as an option. Otherwise it is not,
in general, possible to start streaming buffer content before the entire
buffer is written.


## Possible Future Features

This is highly speculative, but documents some ideas that have been
floating in order to avoid incompatible custom extensions on the same
theme. Still these might never be implemented or end up being
implemented differently. Be warned.

### Force Align

`force_align` attribute supported on fields of structs, scalars and
and vectors of fixed length elements.

### Mixins

A mixin is a table or a struct that is apparently expanded into the table
or struct that contains it.

Mixins add new accessors to make it appear as if the fields of the mixed
in type is copied into the containing type although physically this
isn't the case. Mixins can include types that themselves have mixins
and these mixed in fields are also expanded.

A `mixin` is an attribute applied to table or a struct field when that
field has a struct or a table type. The binary format is unchanged and
the table or struct will continue to work as if it wasn't a mixin and
is therefore fully backwards compatible.

Example:

    struct Position {
        spawned: bool(required);
        x: int;
        y: int;
    }

    table Motion {
        place: Position(mixin);
        vx: int = 0;
        vy: int = 0;
    }

    table Status {
        title: string;
        energy: int;
        sleep: int;
    }

    table NPC {
        npcid: int;
        motion: Motion(mixin);
        stat: Status(mixin);
    }

    table Rock {
        here: Position(mixin);
        color: uint32 = 0xa0a0a000;
    }

    table Main {
        npc1: NPC;
        rock1: Rock;
    }

    root_type Main;


Here the table NPC and Rock will appear with read accessors is if they have the fields:

    table NPC {
        npcid: int;
        spawned: bool(required);
        x: int;
        y: int;
        vx: int = 0;
        vy: int = 0;
        title: string;
        energy: int;
        sleep: int;
        place: Position;
        motion: Motion(required);
        stat: Status;
    }

    table Rock {
        spawned: bool(required);
        x: int;
        y: int;
        here: Position(required);
        color: uint32 = 0xa0a0a000;
    }

    table Main {
        npc1: NPC;
        rock1: Rock;
    }

    root_type Main;

or in JSON:

    {
        "npc1": {
            "npcid": 1,
            "spawned": true;
            "x": 2,
            "y": 3,
            "vx": -4,
            "vy": 5,
            "title": "Monster",
            "energy": 6,
            "sleep": 0
        },
        "rock1": {
            "spawned": false;
            "x": 0,
            "y": 0,
            "color": 0xa0a0a000
        }
    }



Note that there is some redundancy here which makes it possible to
access the mixed in fields in different ways and to perform type casts
to a mixed in type.

A cast can happen through a generated function along the lines of
`npc.castToPosition()`, or via field a accessor `npc.getPlace()`.

A JSON serializer excludes the intermediate container fields such as
`place` and `motion` in the example.

A builder may choose to only support the basic interface and required
each mixed in table or struct be created separately. A more advanced
builder would alternatively accept adding fields directly, but would
error if a field is set twice by mixing up the approaches.

If a mixed type has a required field, the required status propagates to
the parent, but non-required siblings are not required as can be seen in
the example above.

Mixins also places some constraints on the involved types. It is not
possible to mix in the same type twice because names would conflict and
it would no longer be possible to do trivially cast a table or struct
to one of its kinds. An empty table could be mixed in to
provide type information but such a type can also only be added once.

Mixing in types introduces the risk of name conflicts. It is not valid
to mix in a type directly or indirectly in a way that would lead to
conflicting field names in a containing type.

Note that in the example it is not possible to mix in the pos struct
twice, otherwise we could have mixed in a Coord class twice to have
position and velocity, but in that case it would be natural to
have two fields of Coord type which are not mixed in.

Schema evolution is fully supported because the vtable and field id's
are kept separate. It is possible to add new fields to any type that
is mixed in. However, adding fields could lead to name conficts
which are then reported by the schema compiler.


[eclectic.fbs]: https://github.com/dvidelabs/flatcc/blob/master/doc/eclectic.fbs
