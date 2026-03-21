# Builder Interface Reference

<!-- vim-markdown-toc GFM -->

* [Introduction](#introduction)
* [Size Prefixed Buffers](#size-prefixed-buffers)
* [Namespaces](#namespaces)
* [Error Codes](#error-codes)
* [Endianess](#endianess)
  * [Deprecated](#deprecated)
* [Buffers](#buffers)
* [Tables](#tables)
  * [Adding Fields](#adding-fields)
  * [Nested Tables](#nested-tables)
* [Packing tables](#packing-tables)
* [Strings](#strings)
* [Structs](#structs)
  * [Fixed Length Arrays in Structs](#fixed-length-arrays-in-structs)
* [Nested Buffers](#nested-buffers)
* [Scalars and Enums](#scalars-and-enums)
* [Vectors](#vectors)
* [Unions](#unions)
  * [Union Vectors](#union-vectors)
  * [Unions of Strings and Structs](#unions-of-strings-and-structs)
* [Error Handling](#error-handling)
* [Type System Overview](#type-system-overview)
* [Cloning](#cloning)
* [Picking](#picking)
* [Sorting Vectors](#sorting-vectors)
  * [Dangers of Sorting](#dangers-of-sorting)
  * [Scanning](#scanning)
* [Example of different interface type users](#example-of-different-interface-type-users)
* [Special Emitters](#special-emitters)

<!-- vim-markdown-toc -->


## Introduction

We assume a separate read-only file and add extensions to this with
support from a builder library and a builder object.

The underlying builder library supports two modes of operation that mix
together: `create` which sends data directly to the target buffer
(emitter object) and a stack driven `start/end` approach which allocates
objects and vectors on the stack. The code generator chooses the most
efficient approach given the circumstances.

Unlike most FlatBuffer language interfaces, tables and vectors are not
created back to front: They are either created completely in one
operation, or they are constructed on a stack front to back until they
can be emitted.  The final buffer is still constructed back to front.
For big-endian platforms this may require temporary stack allocation of
complete vectors where little endian platforms can emit directly.

Tables and vectors stored in other tables or vectors must be completed
before the can be stored, but unlike must language interfaces they can
be constructed while a parent is also being constructed as long as
nesting remains balanced. While this occasionally may require more
stack, it may also avoid external temporary allocation.

A builder object is required to start buffer construction. The builder
must be initialized first and can be reset and reused between buffers,
reusing stack allocation. The builder can have a customized emitter
object but here we use the default. Finalizing the buffer depends
the emitter and we can use a default finalizer only because we use the
default emitter - it allocates and populates a linear buffer from a
paged emitter ring buffer.

Note that in most cases `flatcc_builder_finalize_buffer` is sufficient,
but to be strictly portable, use
`flatcc_builder_finalize_aligned_buffer` and `aligned_free`.
`aligned_free` is often implemented as `free` in `flatcc/portable` but
not on all platforms. As of flatcc version 0.5.0
`flatcc_builder_aligned_free` is provided to add robustness in case the
applications `aligned_free` implementation might differ from the library
version due to changes in compile time flags.

Generally we use the monster example with various extensions, but to
show a simple complete example we use a very simple schema (`myschema.fbs`):

    table mytable { myfield1: int; myfield2: int; }

    #include "myschema_builder.h"

    void testfun() {

        void *buffer;
        size_t size;
        flatcc_builder_t builder, *B;
        mytable_table_t mt;
        B = &builder;
        flatcc_builder_init(B);

        /* Construct a buffer specific to schema. */
        mytable_create_as_root(B, 1, 2);

        /* Retrieve buffer - see also `flatcc_builder_get_direct_buffer`. */
        /* buffer = flatcc_builder_finalize_buffer(B, &size); */
        buffer = flatcc_builder_finalize_aligned_buffer(B, &size);

        /* This is read-only buffer access. */
        mt = mytable_as_root(buffer);
        assert(mytable_myfield1(mt) == 1);
        assert(mytable_myfield2(mt) == 2);

        /* free(buffer); */
        flatcc_builder_aligned_free(buffer);

        /*
         * Reset, but keep allocated stack etc.,
         * or optionally reduce memory using `flatcc_builder_custom_reset`.
         */
        flatcc_builder_reset(B);

        /* ... construct another a buffer */

        /* Reclaim all memory. */
        flatcc_builder_clear(B);
    }

Note that a compiled schema generates a `myschema_reader.h` file and
optionally a `myschema_builder.h` and some common support files. When
building a buffer the `myschema_builder.h` must be used but when only
reading then the `myschema_reader.h` file should be used instead. Here
we are only concerned with building. When building, it is necessary to
link with `libflatccrt.a` runtime library but when reading, all
nesessary code is contained in the generated header files.

The builder object only manages a stack of currently active objects and
does not store an object that is complete. Instead it calls an emitter
object with the partial data ready for emission, similar to a write
function. A default emitter is provided which implements a ring buffer
and the result may be written to a file, copied to a buffer or a
finalized to an allocated buffer. The builder supports these methods
directly for default emitter, and only the default emitter because
emitters are otherwise defined by only one simple emit function - see
`emit_test.c` for a simple example of a custom emitter.
A custom allocator may be useful when working with small buffers in a
constrained environment - the allocator handles temporary stacks,
virtual table caches etc. but not the emitter.

The allocator and emitter interface is documented in the builder library
header pflatcc_builder.h] and the default implementation in
[flatcc_emitter.h]. The default allocator is implemented as part of the
flatcc_builder source.

The builder can be reused between buffers using the `reset` operation.
The default emitter can also be reused and will automaticallhy reset
when the buffer is. For custom emitters, any reset operation must be
called manually. The same applies to clear. The reset operations
maintain allocated memory by also reduce memory consumption across
multiple resets heuristically.


## Size Prefixed Buffers

Buffers can be created with a size prefix of type `uoffset_t`. When
doing this, the buffer is aligned relative to the size prefix such that
buffers can be stacked in a file and for example be accessed via memory
mapping.

The usual `create_as_root` and `start_as_root` has a variant called
`create_as_root_with_size` and `start_as_root_with_size`.

To read a buffer with a size prefix use:

    size_t size;
    buffer = flatbuffers_read_size_prefix(rawbuffer, &size);

The size the size of the buffer excluding the size prefix. When
verifying buffers the buffer and size arguments should be used. See also
[monster_test.c] for an example.

Note that the size prefix ensures internal alignment but does not
guarantee that the next buffer in a file can be appended directly
because the next buffers alignment is unknown and because it potentially
wastes padding bytes.  The buffer size at offset 0 can increased to the
needed alignment as long as endianness is handled and the size of the
size field is subtracted, and zeroes are appended as necesary.

## Namespaces

The generated code is typically wrapped in a custom namespace and
functions and definitions that are library specific are usually mapped
into the namespace. We often use an empty namespace for custom types and
`flatbuffers_` for library names, but usually a `foo_` prefix could also
be used on both cases, where `foo` is a custom namespace.

Note that the name `flatcc_emitter` is only used with the default emitter
and the name [flatcc_builder] is only used for buffer management but not
for constructing content. Once a valid buffer is ready the common and
namespace (`flatbuffers`) and schema specific (or empty) namespace is used
with schema specific operations.

All schema specific content is prefixed with a namespace to avoid
conflicts - although the namespace is empty if the schema doesn't
specify any. Note that the same schema can have multiple
namespaces. An example of a namespace prefixed operation:

    MyGame_Example_Monster_create_as_root(B, ... lots of args);

To simplify this we can use a macro to prefix a namespace. The use
of the name `ns` is arbitrary and we can choose different names for
different namespaces.

    #undef ns
    #define ns(x) MyGame_Example_ ## x

But the above doesn't work with nested calls to ns such as

    ns(Monster_color_add(B, ns(Color_Green));

it would have to be:

    ns(Monster_color_add)(B, ns(Color_Green);

Therefore we have a helper macro the does allow nesting:

    #undef ns
    #define ns(x) FLATBUFFERS_WRAP_NAMESPACE(MyGame_Example, x)

The common namespace can also be wrapped for a more consistent
appearance:

    #undef nsc
    #define nsc(x) FLATBUFFERS_WRAP_NAMESPACE(flatbuffers, x)

    nsc(string_ref_t) s;
    s = nsc(string_create_str(B, "hello, world!"));

instead of

    flatbuffers_string_ref_t s;
    s = flatbuffers_string_create_str(B, "hellow, world!);


## Error Codes

Functions return values can be grouped roughly into 4 groups: functions
returning pointer, references, `size_t` lengths, and `int` status codes.
Pointers and references return 0 on error. Sizes do not return error.
Status codes return 0 on success or an error code that is usually -1.
Status codes may be checked with `flatbuffers_failed(...)`.


## Endianess

The function `flatbuffers_is_native_pe()` provide an efficient runtime
check for endianness. Since FlatBuffers are little endian, the function
returns true when the native endianness matches the protocol endianness
which for FlatBuffers is little endian. We do not hardcode little endian
because it enables us to support other protocols in the future - for
example the struct conversions may be very useful for big endian network
protocols.

> As of flatcc 0.4.0 it is possible to compile flatcc with native
> big-endian support which has been tested on AIX. More details in
> [README Endianness](https://github.com/dvidelabs/flatcc#endianness)


By testing `is_native_pe` dependencies on speficic compile time flags
can be avoided, and these are fragile:

During build, vectors and structs behave differently from tables: A
table updates one field at a time, doing endian conversion along the
way. A struct is either placed in a table, and is converted by the table
specific operation, or it is placed in a vector. A vector only does the
endian conversion when the vector is finished, so when a vector is not
created atomically with a single `create` call, the elements are placed on a
stack. By default this is in native format, but the user may choose to
place buffer encoded structs or scalars in the vector and call
`vec_end_pe`. The same `push` operation can be used to place a
natively encoded struct and a buffer encoded struct in the vector
because it does no conversion at that point. Therefore there is also no
`push_pe` method that would mean to push an unconverted element unto
the stack. Only for tables and entire vectors does the pe command make
sense. If a vector wishes to push a buffer encoded struct when the
vector is otherwise constructed in native encoding or vice versa, the
vector may be extended empty and then assigned using any of the
`assign`, `assign_from_pe` or `assign_to_pe` calls.

We did not mention that a struct can also be a standalone object
as a buffer root, and for that it has a `end_pe` call that essentially
works like a single element vector without a length prefix.

The `clone` operation is a more userfriendly `pe` operation which takes
an object or a vector from an existing buffer and places it in a new
buffer without endian conversion.

### Deprecated

__NOTE: `FLATBUFFERS_LITTLEENDIAN` is deprecated and will be removed in
a future version. It just complicates endina handling.__

The header files tries to define `FLATBUFFERS_LITTLEENDIAN` to 0 or 1
based on system definitions but otherwise leaves the flag undefined.
Simply testing for

    #if FLATBUFFERS_LITTLEENDIAN
    ...
    #endif

will not fail if the endianness is undetected but rather give the
impression that the system is big endian, which is not necessarily true.
The `flatbuffers_is_native_pe()` relates to the detected or system
provided conversion functions if a suitable `endian.h` file after the
header file gave up on its own detection (e.g. `le16toh(1) == 1`).
Therefore, it is better to use `flatbuffers_is_native_pe()` in most
cases. It also avoids making assumptions on whether the protocol is
little or big endian.

## Buffers

A buffer can most simply be created with the `create_as_root` call for
a table or a struct as seen ealier. The `as_root` part is just a thin
wrapper around buffer start and stop calls and using these allows for
more flexibility. the `as_root` also automatically uses the defined file
identifier if any.

The build process begins with starting a buffer. The buffer may contain
a struct or table, so one of these should be constructed subsequently.
Structs are generally created inline in tables, only at the buffer level
is a struct created independently. The api actually permits other
formats, but it will not be valid flatbuffers then.

    flatcc_builder_ref_t root;
    flatcc_builder_init(B);
    /* 0 indicates no file identifier. */
    flatcc_builder_buffer_start(B, 0);
    root = /* ... construct a table or a struct */
    flatcc_builder_buffer_end(B, root);

`buffer_start` takes a file identifier as second argument. If null or a
string with null characters, the identifier is not stored in the buffer.

Regardless of whether a struct or table is declared as root in the schema or
not, there are methods to automatically start both the buffer and struct or buffer
and table such as `Monster_start/end_as_root`. This is also valid for
nested buffers. If the schema has a file identifier, it is used as
identifier for the created object. The alternative
`create_as_root_with_identifier` allows for explicitly setting an id or
explicitly dropping an id by providing a null argument. The
corresponding reader function `Monster_as_root(buffer)` also has a
`Monster_as_root_with_identifier(buffer, id)`. Here the id is ignored if the id
is null, and otherwise the operation returns null if the id does not match.
For the most part ids are handled transparently by these defaults.

The buffer can be started with block alignment and/or a custom
identifier using the `flatcc_builder_buffer_start_aligned`:

    flatcc_builder_buffer_start_aligned(B, "myid", 16);
    ...
    flatcc_builder_buffer_end(B, root);

The alignment can be 0 using the minimum required alignment, which is
derived from the operations between `start/end`. The alignment argument
is called `block_align` and is useful if the emitter operates on blocks
such as encryption, cache line isolation, or compression blocks where
the final buffer should align with the blocks used during construction.
This can lead to significant zero padding just after the block header,
depending on block size.

The schema specified identifier is given as:

    flatbuffers_identifier

and defaults to null. The schema specified extension is given as:

    flatbuffers_extension

and defaults to null. Note that `flatbuffers_` is replaced by whatever
namespace is chosen. Each specific schema type also has a named file
exntension reflection the extension active when the type was defined,
for example:

    MyGame_Example_Monster_file_identifier

This define is used when `create_as_root` automatically sets a file
identifier.

NOTE: before flatcc 0.6.1, the identifier was named

    MyGame_Example_Monster_identifier (DEPRECATED)

but that would conflict with a table field named `identifier` which
happened often enough to be a problem. This naming is now removed on
conflict and will be completely removed in a future version.

When the buffer is ended, nothing special happens but only at this point
does it really makes sense to access the resulting buffer. The default
emitter provides a copy method and a direct buffer access method. These
are made available in the builder interface and will return null for
other emitters. See also [flatcc_builder.h] and the default emitter in
`flatcc_emitter.h`.


## Tables

### Adding Fields

If `Monster` is a table, we can create a Monster buffer (after
builder init) as follows:

    Monster_start(B);
    Monster_Hp_add(B, 80);
    ...
    flatcc_builder_buffer_create(B, Monster_end(B));

All scalar and enums are added similar to the `Monster_add_Hp` call. We
will subsequently see how to deal with other types.

A table can also be created in a single operation using `create`:

    Monster_ref_t m;
    m = Monster_create(B, 80, ...);

The create arguments are those taken by the individual fields `add`
operations which is either an scalar, enum, or a reference returned by
another create or end call. Note that unlike the C++ interface, unions
only take a single argument that is also accepted by the `add` operation
of a union field. Deprecated fields are not included in the argument
list.

As of v0.5.3 the arguments are given in field id order which is usually
the same as the schema listed order, except with id attributes are
given explicitly. Using id order ensures version stability. Note that
since deprecated fields are omitted, deprecated fields can still break
existing code.

BREAKING: Prior to flatcc v0.5.3 the create call would use the schema order
also when fields have id attributes specifying a different order. This
could break code across versions and did not match the C++ behavior.
It was also document that the `original_order` attribute affected create
argument order, but that was incorrect.

NOTE: If the `original_order` attribute is set on a table, the `create`
implementation adds fields to the table in schema listed order,
otherwise it adds fields in order of decreasing size to reduce alignment
overhead. Generally there should be no need to use the `original_order`
attribute. This doesn't affect the call argument order although that
was incorrectly document prior to v 0.5.3.

NOTE: the `create` and `create_as_root` operations are not guaranteed to
be available when the number of fields is sufficiently large because it
might break some compilers. Currently there are no such restrictions.

Scalars and enums do not store the value if it it matches the default
value which is by default 0 and otherwise defined in the schema. To
override this behavior, use `force_add`. In the monster example, health
points default to 100 (percent), so if we wish to force store it in the
buffer we could use:

    Monster_hp_force_add(B, 100);

Only scalar fields and enums have a `force_add` operation since only these
types have a default value, and other types have a meaningful
interpretation of null. (It is not quite clear if empty tables separate
from null/absent are valid in all implementations).

`force_add` may be useful when roundtripping data from a database where it is
relevant to distinguish between any valid value and null. Most readers will not
be able to tell the difference, but it is possible to inspect a flatbuffer to
see if a table field is present, present and default, or absent, meaning null.

NOTE: As of mid 2020, FlatBuffers added optional scalar table fields with support in flatcc 0.6.1. These fields automatically imply `force_add` to represent null values when a field is absent and therefore these fields do not have a `force_add` method and these fields also do not have a default value other than `null`, i.e. null if not added.

If Monster is declared as root, the above may also be called as:

    Monster_start_as_root(B);
    Monster_add_hp(B, 80);
    ...
    Monster_end_as_root(B);

(Calling `Monster_end` instead would require `buffer_end` call
subsequently, and is basically a violation of nesting).

### Nested Tables

Tables can be nested, for example the Mini field may have type
Monster table again (a recursive type):

    buffer_start(B);
    Monster_start(B);
    Monster_add_Hp(B, 80);
    Monster_start(B);
    Monster_hp_add(B, 81);
    ...
    Monster_mini_add(Monster_end(B));
    ...
    flatcc_builder_buffer_end(B, Monster_end(B));

The child Monster table may be created before the parent or as above
between the tables start and end. If created before, reference must be
stored until it can be added. The only requirement is that start and
end are balanced, that the sub-table is ended before the parent, and
that both are created in the same buffer (nested buffers can be created
while the parent buffer is still being created, similar to sub-tables,
so it is possible to mess this up):

    Monster_ref_t root, mini;

    buffer_start(B);
    Monster_start(B);
    Monster_hp_add(B, 81);
    mini = Monster_end(B);

    Monster_start(B);
    Monster_hp_add(B, 80);
    Monster_mini_add(B, mini);
    root = Monster_end(B);

    flatcc_builder_buffer_end(B, root)


Rather than adding a child table explicitly, it can be started and ended
as an operation on the field name, here with `Monster_Mini_start/end`:

    Monster_ref_t root;

    Monster_start(B);
    Monster_add_Hp(B, 80);
    Monster_mini_start(B);
    Monster_hp_add(B, 81);
    Monster_mini_end(B);
    root = Monster_end(B);

    flatcc_builder_buffer_end(B, root);

We can repeat the the table nesting as deep as we like, provided our
builder is willing to allocate enough stack space.

**Warning**: It is possible to use the wrong table type operations
between `start/end` - don't do that. It is a tradeoff between usability
and type safety.

Note that vectors, strings and structs map several standard operations
to a field name, for example `mytable_myfield_push(B, x)`. This is not the
case with table fields which only map `start/end/create` in part because it
would never terminate for recursive types and in part because each table
is different making a generic mapping rather complex and with very long
names.

A table may be created with a constructor, but it requires all
non-scalar objects to be references or pointers. Struct fields must be
pointers to zero padded structs, and strings, vectors and tables must be
references. The constructors are probably most useful for simple tables
with mostly scalar values (here we use the original Monster fields and
leaves out any we have invented for the sake of illustration):

IMPORTANT: objects can generally only be created within a buffer
context, i.e. after `buffer_start`. For example calling
`flatbuffers_uint8_vec_create` before `Monster_create_as_root`
technically violates this rule because the create call also starts the
buffer. It is, however, allowed at the top level. For nested buffers
(see later) this must be avoided because the vector would end up in the
wrong buffer.

    Monster_ref_t m;
    uint8_t invdata[4] = { 1, 2, 3, 4 };
    Vec3_t vec;

    flatbuffers_uint8_vec_ref_t inventory =
        flatbuffers_uint8_vec_create(B, invdata, 4);
    m = Monster_create(B, &vec, 150, 80, name, inventory,
        Color_Red, Any_as_NONE());
    flatcc_builder_buffer_create(m);

or

    Monster_create_as_root(B, &vec, 150, 80, name, inventory,
        Color_Red, Any_as_NONE());

## Packing tables

By reordering the fields, the table may be packed better, or be better
able to reuse an existing vtable. The `create` call already does this
unless the attribute `original_order` has been set. Unions present a
special problem since it is two fields treated as one and the type field
will generally waste padding space if stored in order:

To help pack unions better these can be added with the type
separate from the value reference using `add_type(B, test.type)`,
`add_value(B, test)` where the value is only added if the type is
not `NONE`. The `add_type` should be called last since it is the
smallest type.

The same field should not be added more than at most once. Internal
reservations that track offset fields may overflow otherwise. An
assertion will fail in debug builds.

Required table fields will be asserted in debug builds as part of the
`end/create` call.  Only offset fields can have a required attribute.

The generated `monster_test_reader.h` from [monster_test.fbs] shows how
the default packing takes place in generated `create` calls, see for
example the typealias test. Note that for example vectors are stored
together with integers like `uint32` because references to vectors have
the same size as `uint32`.


## Strings

Strings can be added to tables with zero terminated strings as source

    Monster_start(B);
    ...
    Monster_name_create_str(B, "Mega Monster");
    Monster_end(B);

or strings potententially containing zeroes:

    #define MONSTER "Mega\0Monster"
    Monster_start(B);
    ...
    /* Includes embedded zero. */
    Monster_name_create(B, MONSTER, sizeof(MONSTER));
    Monster_end(B);

or zero terminated source up to at most `max_len` characters.

    #define MONSTER "Mega\0Monster"
    Monster_start(B);
    ...
    /* "Mega" */
    Monster_name_create_strn(B, MONSTER, 12);
    Monster_end(B);

The `create_str` and `create_strn` versions finds the string length via strlen
and strnlen respectively. `append_string` also has `_str/_strn` versions.

A string can also be created from an existing flatbuffer string in which
case the length is expected to be stored 4 bytes before the pointer in
little endian format, and aligned properly:

    Monster_name_clone(B, mybufferstring);

or, create a string at most 4 characters long starting at 0-based index
10, if present:

    Monster_name_slice(B, mybufferstring, 10, 4);

If index or index + len goes beyond the source, the result is truncated
accordingly, possibly resulting in an empty string.

A string can also be create independently. The above is just shortcuts
for that:

    flatbuffers_string_ref_t monster_name;
    monster_name = flatbuffers_string_create_str("Mega Monster");
    Monster_name_add(B, monster_name);

Strings are generally expected to be utf-8, but any binary data will be
stored. Zero termination or embedded control codes are includes as is.
The string gets a final zero temination regardless, not counted in the
string length (in compliance with the FlatBuffers format).

A string can also be constructed from a more elaborate sequence of
operations. A string can be extended, appended to, or truncated and
reappended to, but it cannot be edited after other calls including calls
to update the same string. This may be useful if stripping escape codes
or parsed delimiters, etc., but here we just create the same "Mega
Monster" string in a more convoluted way:

    flatbuffers_string_ref_t name;
    char *s;
    #define N 20
    Monster_start(B);
    ...
    flatbuffers_string_start(B);
    flatbuffers_string_append(B, "Mega", 4);
    flatbuffers_string_append(B, " ", 1);
    s = flatbuffers_string_extend(B, N);
    strncpy(s, "Monster", N);
    flatbuffers_string_truncate(B, N - strlen(s));
    name = flatbuffers_string_end(B);
    Monster_name_add(B, name);
    ...
    Monster_end(B);

`flatbuffers_string_create...` calls are also available when creating
the string separate from adding it to a table, for example:

    flatbuffers_string_h name;
    name = flatbuffers_string_create_str(B, "Mini Monster");

It is guaranteed that any returned the string buffer is zero filled and
has an extra zero after the requested length such that strlen can be
called on the content, but only the requested bytes may be updated.

Every call only returns the substring being added to the string in that
operation. It is also possible to call `flatbuffers_string_edit` to get a
modifiable pointer to the start of the string.

`flatbuffers_string_reserved_len(B)` returns the current string length
including any embedded zeroes, but excluding final zero termination. It
is only valid until `string_end` is called.

See [flatcc_builder.h] for detailed documentation. Essentially `extend`
reserves zeroed space on the stack and returns a buffer to the new
space, and truncate reduces the overall size again, and the string is
then given the final length and a zero termination at the end.

There is no endian conversion (except internally for the string length),
because UTF-8 strings are not sensitive to endianness.

Like tables, the string may be created while a parent container is being
constructed, or before.

Strings can also be used as vector elements, but we will get that when
discussing vectors.

## Structs

Structs in tables can be added as:

    Monster_pos_create(B, 1, 2, 3);

The above essentially does the following:

    Vec3_t *v;
    v = Monster_pos_start(B);
    Vec3_assign(v, 1, 2, -3.2);
    Monster_pos_end(B);

Some versions of the monster schema has extra test fields - these would
break the assign approach above because there would be extra arguments.
Instead we can rely on the zero intialization and assign known fields.

    Vec3_t *v;
    v = Monster_pos_start(B);
    v->x = 1, v->y = 2, v->z = -3.2;
    Monster_pos_end(B);

`Monster_pos_end_pe(B)` can be used when the struct is known to be
little endian (pe for protocol endian, meaning no conversion is necessary),
for example copied from an existing buffer, but then `clone` is a better
choice:

    Monster_pos_clone(B, &v);

When the struct is created alone for use as root:

    Vec3_ref_t root;
    root = Vec3_create(B, 1, 2, 3)
    flatcc_builder_buffer_create(B, root);

An existing struct can be added as:

    Vec3_t v;
    Vec3_assign(&v, 1, 2, 3);
    /* v does not have to be zero padded. */
    Monster_pos_add(B, &v);

When adding a struct that is already little endian, presumably from an
existing buffer, it can be cloned using:

    Monster_pos_clone(B, &v);

Clone assumes the source struct is both little endian and that padding
is already zeroed (example ignores error handling), and `end_pe`
does nothing.

    *Monster_pos_start(B) = v;
    Monster_pos_end_pe(B);

There are several assignment types that convert between host (native)
endianness and buffer endiannes. We use `pe` to indicate
`protocol_endian` rather than just `le` for `little endian` because it
allows us to change endianness to big endian in the the future and it
more clearly states the intention. While big endian is not allowed in
FlatBuffers, big endian structs may be useful in other network
protocols - but it is not currently supported because it would force
little endian platforms to support byte-swapping. The operations are:

`assign_from_pe`, `assign_to_pe`, `copy`, `copy_from_pe`,
`copy_to_pe`, `to_pe` and `from_pe`.

All the copy operations takes a const pointer as source, and
`to/from_pe` is just copy with same source and destination:

    Vec3_t v, v2;
    Vec3_assign_to_pe(&v2, 1, 2, 3);
    Vec3_copy_from_pe(Vec3_clear(&v), &v2);
    Vec3_to_pe(&v);

`from_pe` means from little endian to native endian, end `to_pe`
is the opposite. On little endian platforms all copy operations behave
the same and only move fields, not padding. `to/from_pe` conversion
will leave deprecated fields either as they were, or zero them because
the operation may be skipped entirely on protocol endian native platforms.

While struct fields cannot be deprecated officially, they are supported
if the schema compiler is flagged to accept then. The struct fields are
renamed and assigned 0 when using assign or copy, and assign / create has
no argument for them.

Because padding can carry noise and unintended information, structs
should be cleared before assignment - but if used as a source to copy
the padding is not copied so only the destation need to be zeroed.

If a struct is nested, the assign operation includes all fields as if
the struct was flattened:

    typedef struct Plane Plane_t;
    struct Plane {
        Vec3_t direction;
        Vec3_t normal;
    };
    Plane_t plane;
    Plane_clear(&plane);
    Plane_assign(&plane, 1, 2, 3, 7, 8, 9);

Structs can also be created standalone, similar to tables and vectors,
but FlatBuffers only support this when the struct is used as root.

Assuming Vec3 is declared as root, a buffer only holding a Vec3 struct
can be created using:

    Vec3_create_as_root(B, 1, 2, 3);

Important: do not store the above as a nested buffer - it would be
missing the vector size field. If `Monster_playground` is a ubyte vector
with `nested_flatbuffer` attribute, then
`Monster_playground_start/end_as_root` may be used.

Structs also support `start/end_as_root`. In this case `start` returns
the struct pointer, and `end_pe_as_root` is supported:

    Vec3_t *v;
    v = Vec3_start_as_root(B);
    v->x = 1, v->y = 2, v->z = 3;
    Vec3_end_as_root(B);

(Be careful with the different result codes since a tables `start_as_root`
returns an integer result code where 0 is success while a struct returns
a pointer that is null on failure.)

The following also creates a buffer at top-level, but it may also be
added as a nested buffer because the stack frame detects the nesting:

    Vec3_t *v;
    flatcc_builder_buffer_start(B);
    v = Vec3_start(B);
    v->x = 1, v->y = 2, v->z = 3;
    flatcc_builder_buffer_end(B, Vec3_end(B));

or
    flatcc_builder_buffer_start(B);
    ...
    Monster_start(B);
    flatcc_builder_buffer_start(B);
    v = Vec3_start(B);
    v->x = 1, v->y = 2, v->z = 3;
    Monster_playground_add(B,
        flatcc_builder_buffer_end(B, Vec3_end(B)));
    flatcc_builder_buffer_end(B, Monster_end(B));

or

    flatcc_builder_buffer_ref_t nested_root;
    flatcc_builder_buffer_start(B);
    nested_root = Vec3_create_as_root(B, 1, 2, 3);
    Monster_start(B);
    Monster_playground_add(B, nested_root);
    flatcc_builder_buffer_end(B, Monster_end(B));

A `buffer_ref_t` can be used as `uint8_vec_ref_t` when the
buffer is nested, and otherwise the reference cannot be used
for anything other than testing for failure. The buffer content
should match the type declared in a `nested_flatbuffers` attribute
but it isn't enforced, and a root can be stored in any field of
[ubyte] type.

When `Monster_playground` is declared as nested:

    ...
    Monster_start(B);
    Monster_playground_create_as_root(B, 1, 2, 3);
    flatcc_builder_buffer_end(B, Monster_end(B));
    ...

Be aware that `Vec3_t` is for native updates while `Vec3_struct_t` is a const
pointer to an endian encoded struct used in the reader interface, and actually
also as source type in the clone operation.

### Fixed Length Arrays in Structs

As of flatcc 0.6.0 it is possible to have fixed length arrays as structs
members. A fixed length array is equivalent to having a struct field repeated
one or more times. The schema syntax is `name : [type:count];` similar to an
ordinary struct field `name : type;`. The type is any type that can ba valid
struct field type including enums and nested structs. The size cannot be 0 and
the overall size is limited by the maximum struct size the array is contained
within which is typically 65535 (2^16-1).

For example, given the schema:

    struct MyStruct {
      counters:[int:3];
      // char is only valid as a fixed length array type
      name:[char:6];
    }
    table MyTable {
      mystruct:MyStruct;
    }

The table can be created with:

    ns(MyStruct_t) *x;
    ns(MyTable_start_as_root(B));
    x = ns(MyTable_mystruct_start(B));
    x->counters[0] = 1;
    x->counters[1] = 2;
    x->counters[2] = 3;
    strncpy(x->name, "Kermit", sizeof(x->name));
    ns(MyTable_mystruct_end(B));
    ns(MyTable_end_as_root(B));

Note that char arrays are not zero terminated but they are zero padded, so
strncpy is exactly the right operation to use when assigning to char arrays,
at least when they do not contain embedded nulls which is valid.
Char arrays are expected to be ASCII or UTF-8, but an application may use
other encodings if this is clear to all users.

With assignment:

    int data[3] = { 1, 2, 3 };
    ns(MyStruct_t) *x;
    ns(MyTable_start_as_root(B));
    x = ns(MyTable_mystruct_start(B));
    // Careful: the name argument does not use strncpy internally
    // so the source must be at least the expected length
    // like other array arguments. Strings can have embedded nulls.
    ns(MyStruct_assign(x, data, "Kermit");
    ns(MyTable_mystruct_end(B));
    ns(MyTable_end_as_root(B));

To read a struct the pointer to the struct is retrieved first

    int sum;
    int i;
    const char *name;
    size_t name_len;
    ns(MyTable_table_t) t;
    ns(MyStruct_struct_t) x;

    t = ns(MyTable_as_root(buf));
    x = ns(MyTable_mystruct_get(t));
    for (sum = 0, i = 0; i < ns(MyStruct_counters_get_len()); ++i) {
      sum += ns(MyStruct_counters_get(x, i)) +
      // char arrays are endian neutral, so we can use pointer access.
      name = ns(MyStruct_name_get_ptr(x);
      name_len = strnlen(name, ns(MyStruct_name_get_len()));
      printf("Added counters from %.*s", name_len, name);
      // char arrays can be accessed like other arrays:
      // ns(MyStruct_name_get(x, i);
    }

An alternative to `strnlen` is strip trailing zeroes which will allow for
char arrays embedded zeroes, but there is no direct support for this. The JSON
printer uses this approach to shorten the printed char array string.

The `_get` suffix can be ommitted in the above if the flatcc `-g` has not
supplied to reduce the risk of name conflicts, but not for `_get_len` and
`_get_ptr`.

Note that it is not possible to have fixed length arrays as part of a table but
it is possible to wrap such data in a struct, and it is also possible to have
vectors of structs that contain fixed length arrays.


## Nested Buffers

These are discussed under Structs and Table sections but it is worth
noting that a nested buffers can also be added as pe ubyte vectors
which is probably the original intention with nested buffers. However,
when doing so it can be difficult to ensure the buffer is correctly
aligned. The untyped `flatcc_builder` has various options to deal with
this, but with generated code it is better to create a nested buffer
inline when suitable (with nested `buffer_start/end` or
`mytable_myfield_create_as_root`) - for example a message wrapper with
a union of tables holding buffer for a specific message type. In other
cases the buffer may truly be created independently of the current
buffer and then it can be added with controlled alignment using either
the `flatcc_builder` api for full control, or the `nest` operation on
nested table and struct fields:

To create and add a ubyte vector with a higher alignment than ubytes
single byte alignment, the following operation is available as an
operation on a nested buffer field:

    Monster_playground_nest(B, void *data, size_t size, uint16_t align);

If alignment is unknown, it can be set to 0, and it will default to 8
for nested table types, and to the struct alignment for struct buffers.

Block alignment is inherited from the parent buffer so the child buffer
ends up in its own set of blocks, if block alignment is being used. If
the nested buffer needs a different block alignment, the `flatcc_builder`
api must be used.

All structs and tables have an `start/end/create_as_root` even if they
are not referenced by any `nested_flatbuffers` field and they will
create [ubyte] vectors containing a nested buffer but only [ubyte]
fields with `nested_flatbuffers` attribute will dedicated
`start/end/create_as_root` on the field name. Structs also have
`end_pe_as_root`.


## Scalars and Enums

Scalars keep their original type names `uint8_t`, `double`, etc, but
they get some operations similar to structs. These are contained in a
namespace which by default is `flatbuffers_`, for example:

    uint16_t *flatbuffers_uint16_to_pe(uint16_t *p);
    uint16_t *flatbuffers_uint16_from_pe(uint16_t *p);
    flatbuffers_bool_t *flatbuffers_bool_to_pe(flatbuffers_bool_t *p);
    flatbuffers_bool_t *flatbuffers_bool_from_pe(flatbuffers_bool_t *p);

These may be used freely, but are primarily present as an interface to
the vector operations also defined for structs.

Enums have similar definitions which may be used to convert endianness
without being concerned with the underlying integer type, for example:

    Color_enum_t *Color_to_pe(Color_enum_t *p);

## Vectors

Vectors can be created independently, or directly when updating a table - the
end result is the same. Builder vector operations always reference element
values by pointer, or by reference for offset types like tables and strings.

    uint8_t v;
    Monster_inventory_start(B);
    v = 1;
    flatbuffers_uint8_vec_push(B, &v);
    v = 2;
    flatbuffers_uint8_vec_push(B, &v);
    v = 3;
    flatbuffers_uint8_vec_push(B, &v);
    Monster_inventory_end(B);

or

    flatbuffers_uint8_vec_ref_t inv;
    uint8_t v;
    flatbuffers_uint8_vec_start(B);
    v = 1;
    flatbuffers_uint8_vec_push(B, &v);
    v = 2;
    flatbuffers_uint8_vec_push(B, &v);
    v = 3;
    flatbuffers_uint8_vec_push(B, &v);
    inv = flatbuffers_uint8_vec_end(B);
    Monster_inventory_add(B, inv);

Because it can be tedious and error-prone to recall the exact field
type, and because the operations are not type safe (any kind of push
would be accepted), some vector operations are also mapped to the field
name:

    uint8_t v;
    Monster_inventory_start(B);
    v = 1;
    Monster_inventory_push(B, &v);
    v = 2;
    Monster_inventory_push(B, &v);
    v = 3;
    Monster_inventory_push(B, &v);
    Monster_inventory_end(B);

Note: vector operations on a type uses the `_vec_<operation>` syntax, for
example `uint8_vec_push` or `Monster_vec_push` while operations that are mapped
onto table field names of vector type do not use the `_vec` infix because it is
not a type name, for example `Monster_inventory_push`.

A slightly faster operation preallocates the vector:

    uint8_t *v;
    Monster_inventory_start(B);
    v = Monster_inventory_extend(B, 3);
    v[0] = 1, v[1] = 2, v[2] = 3;
    v = Monster_inventory_extend(B, 2);
    v[0] = 4, v[1] = 5;
    Monster_inventory_end(B);

Push just extends one element at time.  Note that `extend` returns the
pointer to the extended vector segment. The full vector can be accessed
with `edit` and `reserved_len` between `start/end` (recalling that pointers
cannot be reused across buffer calls):

    uint8_t *v, i;
    uint8_t data[] = { 1, 2 };
    Monster_inventory_start(B);
    Monster_inventory_push(B, &data[0]);
    Monster_inventory_push(B, &data[1]);
    v = Monster_inventory_edit(B);
    for (i = 1; i < Monster_inventory_reserved_len(B); ++i) {
        v[i] = v[i - 1] + v[i];
    }
    Monster_inventory_end(B);

Note that the name `reserved_len` is to avoid confusion with
`_vec_len` read operation. It also indicates that it is not the final
size since it may change with `truncate/extend`.

A vector can also contain structs. Let us extend the Monster example
with a vector of positions, so we can have a breadcrumb trail:

    Monster_breadcrumbs_start(B);
    Vec3_vec_push_create(B, 1, 2, 3);
    Vec3_vec_push_create(B, 3, 4, 5);
    Monster_breadcrumbs_end(B);

or

    Monster_breadcrumbs_start(B);
    Monster_breadcrumbs_push_create(B, 1, 2, 3);
    Monster_breadcrumbs_push_create(B, 3, 4, 5);
    Monster_breadcrumbs_end(B);

or

    Vec3_t *trails[2];
    Monster_breadcrumbs_start(B);
    trails = Monster_breadcrumbs_extend(B, 2);
    Vec3_create(&trails[0], 1, 2, 3);
    Vec3_create(&trails[1], 4, 5, 6);
    Monster_breadcrumbs_end(B);

The `vec_start/exttend/end/end_pe/create/create_pe/clone/slice` are
translated into similar calls prefixed with the field name instead of
`vector` and except for `start`, the calls also add the vector to the
table if successful, for example:

    uint8_t data[] = { 1, 2, 3 };
    Monster_inventory_create(B, data, 3);
    Monster_breadcrumbs_slice(B, some_other_breadcrumbs, 0, 10);

Vector operations that are allowed between `vec_start` and
`vec_end(_pe)` are also mapped. These are
`vec_extend/append/truncate/edit/reserved_len`, and `push/push_create/push_copy`.
`push_copy` ensures only valid fields are copied, not zero padding (or
the unofficial deprecated fields).

A struct `push_clone` is the same as a `push_copy` operation
because structs are stored inline in vectors - with the
exception of union vectors which have `push_clone` that does the
right thing.

The `add` call adds a vector created independently from the table field,
and this is what is going on under the surface in the other calls:

    Vec3_t x;
    Vec3_vec_ref_t inv;

    /* Clear any padding in `x` because it is not allocated by builder. */
    Vec3_assign(Vec3_clear(&x), 3, 4, 5);
    Vec3_vec_start(B);
    Vec3_vec_push_create(B, 1, 2, 3);
    Vec3_vec_push(B, &v);
    inv = Vec3_vec_end(B);

    Monster_breadcrumbs_add(B, inv);

As always, a reference such as `inv` may only be used at most once, and
should be used once to avoid garbage.

Note that `Vec3_vec_start` would create an independent struct instead of a
vector of structs. Also note that `vec_ref_t` is a builder specific
temporary type while `vec_t` is intended as a const pointer to the first
element in an existing buffer in little endian encoding with a size
prefix (to be used with clone, for example).

An existing Vec3 struct can also be pushed with `Vec3_push(B, &v)`. The
argument must be zero padded. Because vectors are converted at the end,
there is no `push_pe`, but a struct may be in little endian using push
on all platforms if `vec_end_pe` is used at the end.

A vector may also be created from an existing array:

    uint8_t data[] = { 1, 2, 3 };
    Monster_inventory_add(B, flatbuffers_uint8_vec_create(B, data, 3));

This also applies to arrays of structs as long as they are properly zero
padded. `create_pe` is similar but does not do any endian conversion,
and is similar to `clone` except there are no header prefix.

Likewise an existing vector with proper zero padding may be appended
using the `extend` operation. The format must be native or little endian
depending on whether `vec_end` or `vec_end_pe` is called at the end.

All vectors are converted to little endian when the `end` command is
called. `end_pe` prevents this from happening.

`clone` and `slice` and can be used to copy an entire, or a partial
array from an existing buffer. The pointer must be to the first vector
element in little endian format, and it must have a size prefix and be
aligned (like any flatbuffer vector). `slice` takes a base-0 index and
a vector length where the result is truncated if the source is not
large enough.

    Monster_inventory_clone(B, v);

or

    Monster_inventory_add(flatbuffers_int8_clone(B, v);

or

    Monster_inventory_add(flatbuffers_int8_slice(B, v, 2, 4);

or

    Monster_inventory_slice(B, v, 2, 4);

A vector of strings an be constructed as (`friends` is a string
vector field that we just invented for the occasion):

    flatbuffers_string_ref_t friend, *p;
    Monster_friends_start(B);
      friend = flatbuffer_string_create_str(B, "Peter Pan");
      Monster_friends_push_create_str(B, "Shrek");
      Monster_friends_push_create_str(B, "Pinnochio");
      Monster_friends_push_create_str(B, "Pinnochio");
      Monster_friends_push_create(B, "Hector", 6);
      Monster_friends_push(friend);
      p = Monster_friends_extend(B, 1);
      *p = flatbuffers_string_create_str("Cindarella");
      Monster_friends_push_start(B);
        flatbuffers_string_append("The Little");
        flatbuffers_string_append("Mermaid");
      Monster_friends_push_end(B);
    Monster_friends_end(B);

Vectors and strings have a second argument to start, see also the `spawn` example
below.

Finally, vectors can contain tables. Table vectors are offset
vectors just like string vectors. `push_start` pushes a new table and
allows for updates until `push_end`. If we have a spawn vector of monsters in
the Monster table, we can populate it like this:

    Monster_spawn_start(B);
      Monster_vec_push_start(B);
        Monster_Hp_add(B, 27);
      Monster_vec_push_end(B);
      Monster_vec_push_create(B,
        /* Approximate argument list for illustration only. */
        &vec, 150, 80, name, inventory, Color_Red, Any_as_None());
    Monster_spawn_end(B);

The push operation has constructors `push_start/end/create` for both tables
struct, and string elements. String elements also have
`push_create_str/create_strn/clone/slice`. Structs also have
`push_copy`. Between `push_start` and
`push_end` the operations valid for the given table or string element can be
used (typically `add` for tables, and `append` for strings).

Instead of `Monster_vec_push_start` we can also uses
`Monster_spawn_push_start` etc. - in this case the child type is the
same as the parent, but using the field specific `push_start` ensures we
get the right table element type.

`Monster_spawn_push_start(B)` takes no length argument because it is a
table element, while `Monster_friends_push_start(B)` because it is a
string element (similar to a vector).

`Monster_spawn_start(B)` should just be followed by push operations
rather than following up with `Monster_spawn_extend(B, n)` because we
risk loose references that can lead to crashes. But handled carefully
it is possible:

    Monster_vec_ref_t mvec;
    Monster_spawn_start(B);
    mvec = Monster_spawn_extend(B, 2);
    mvec[0] = Monster_create(B, ...);
    mvec[1] = Monster_create(B, ...);
    Monster_spawn_end(B);

We can also push a reference to an independently create monster table,
all as seen before with strings.

As of flatcc version 0.5.2 it is also possible to clone tables.
Therefore we also have `push_clone` on vectors of tables.

While the use of `extend` and `truncate` is possible with vectors of
strings and tables, they should be used with care because the elements
are references and will just end up as garbage if truncated. On the
other hand, unused elements should be truncated as 0 elements in an
offset vector is not valid.

A vector of tables or strings can be created using an externally built
array of references, for example:

    Monster_ref_t monsters[20];
    Monster_vec_ref_t mvec;
    monsters[0] = Monster_create(B, ...);
    ...
    mvec = Monster_vec_create(B, monsters, 20);

By convention, create calls bypass the internal stack when the endian
format is otherwise compatible, and thus feed the emitter directly.
This is not possible with table and string vectors because the
references in the source vectors must be translated into offsets.
Therefore these create calls are similar to start, append, end calls.
There is an internal, but unexposed `flatcc_builder` version
`create_offset_vector_direct` which destroys the source vector instead
of allocating a stack copy.

## Unions

Unlike the C++ Flatbuffers library, we do not expose a separate union
type field except via a small struct with a union of typed references
and a type field. This struct is given to the create argument, and above
it is zero initialized meaning default None.

Unions can be created with value specific `start/end/create` calls. The add
call is not specialized since it takes a union reference:


    Monster_test_Weapon_start(B);
    Weapon_rounds_add(B, 50);
    Monster_test_Weapon_end(B);

or

    Monster_test_Weapon_create(B, 50);

or

    Monster_test_Weapon_add(B, Weapon_create(B, 50));

or

    Monster_test_Pickup_start(B);
    Pickup_location_create(B, 0, 0, 17);
    Pickup_hint_create_str(B, "Jump High!");
    Monster_test_Pickup_end(B);

or

    Pickup_ref_t test;
    Pickup_start(B);
    Pickup_location_create(B, 0, 0, 17);
    test = Pickup_end(B);
    Monster_test_add(B, Any_as_Pickup(test));

or

    Any_union_ref_t test;
    Pickup_start(B);
    Pickup_location_create(B, 0, 0, 17);
    /* test.Pickup = Pickup_end(B); no longer possible as of v0.5.0 */
    test.value = Pickup_end(B); /* As of v0.5.1. */
    test.type = Any_Pickup;
    Monster_test_add(B, test);

The following is valid and will not return an error, but also has no effect:

    Monster_test_add(B, Any_as_NONE());


_Note: the union structure has been changed for v0.5.0, and v0.5.1.
Both unions and union vectors are now represented by a struct with the
fields { type, value } in the low level interfaces. Before 0.5.0 only
unions of tables were supported._


### Union Vectors

The `monster_test.fbs` schema has a field named manyany in the Monster
table. It is vector of unions of type Any.

We can create a vector using

    Any_union_vec_ref_t anyvec_ref;

    Any_vec_start(B);
    Any_vec_push(TestSimpleTableWithEnum_create(B));
    anyvec_ref = Any_vec_end(B);
    Monster_manyany_add(anyvec_ref);

A union can be constructed with type specific `_push` or `_push_create` operations:

    Monster_manyany_start(B);
    Monster_manyany_push(B, Any_as_TestSimpleTableWithEnum(ref));
    Monster_manyany_end(B);

    Monster_manyany_start(B);
    Monster_manyany_TestSimpleTableWithEnum_push(B, ref);
    Monster_manyany_end(B);

    Monster_manyany_start(B);
    Monster_manyany_TestSimpleTableWithEnum_push_create(B, args);
    Monster_manyany_end(B);

and other similar operations, much like other vectors.

Note that internally `anyvec_ref` is really two references, one to type
vector and one to a table vector. The vector is constructed a single
vector of unions and later split into two before final storage. If it is
necessary to create a union vector from a vector of tables and types,
the low level builder interface has a `direct` call to do this.

Union vectos generally use more temporary stack space because during
construction because each element as a struct of type and reference
which don't back as densely as a two separate tables. In addition the
separated type and table vectors must be constructed temporarily. The
finaly buffer result is resonably compatct since the type vector does
not use much space. Unions will also be somewhat slower to construct,
but not unreasonably so.


### Unions of Strings and Structs

_Note: as of v0.5.0 unions can also contain strings and structs in
addition to tables. Support for these types in other languages may vary,
but C++ does support them too._

All union values are stored by reference. Structs that are not unions
are stored inline in tables and cannot be shared but unions of struct
type are stored by reference and can be shared. A union value is
therefore always a reference. This is mostly transparent because the
generated table field methods has `create/start/end` calls for each union
value type and addition to `add`.

To illustrate the use of these variation we use the Movie table from
[monster_test.fbs]:

    namespace Fantasy;

    table Attacker {
        sword_attack_damage: int;
    }

    struct Rapunzel {
        hair_length: uint16;
    }

    struct BookReader {
        books_read: int;
    }

    union Character {
        MuLan: Attacker = 2,  // Can have name be different from type.
        Rapunzel = 8,         // Or just both the same, as before.
        Belle: Fantasy.BookReader,
        BookFan: BookReader,
        Other: string,
        Unused: string = 255
    }

    table Movie {
        main_character: Character;
        antagonist: Character;
        side_kick: Character;
        cameo: Character;
        characters: [Character];
    }


and the mixed type test case from [monster_test.c]:


    nsf(Character_union_ref_t) ut;
    nsf(Rapunzel_ref_t) cameo_ref;
    nsf(Attacker_ref_t) attacker_ref;
    nsf(BookReader_ref_t) br_ref;
    nsf(BookReader_t *) pbr;
    nsf(Movie_table_t) mov;

    nsf(Movie_start_as_root(B));
    br_ref = nsf(BookReader_create(B, 10));
    cameo_ref = nsf(Rapunzel_create(B, 22));
    ut = nsf(Character_as_Rapunzel(cameo_ref));
    nsf(Movie_main_character_Rapunzel_create(B, 19));
    nsf(Movie_cameo_Rapunzel_add(B, cameo_ref));
    attacker_ref = nsf(Attacker_create(B, 42));
    nsf(Movie_antagonist_MuLan_add(B, attacker_ref));
    nsf(Movie_side_kick_Other_create_str(B, "Nemo"));
    nsf(Movie_characters_start(B));
    nsf(Movie_characters_push(B, ut));
    nsf(Movie_characters_MuLan_push(B, attacker_ref));
    nsf(Movie_characters_MuLan_push_create(B, 1));
    nsf(Character_vec_push(B, nsf(Character_as_Other(nsc(string_create_str(B, "other"))))));
    nsf(Movie_characters_Belle_push(B, br_ref));
    pbr = nsf(Movie_characters_Belle_push_start(B));
    pbr->books_read = 3;
    nsf(Movie_characters_Belle_push_end(B));
    nsf(Movie_characters_Belle_push(B, nsf(BookReader_create(B, 1))));
    nsf(Movie_characters_Belle_push_create(B, 2));
    nsf(Movie_characters_Other_push(B, nsc(string_create_str(B, "another"))));
    nsf(Movie_characters_Other_push_create_str(B, "yet another"));
    nsf(Movie_characters_end(B));
    nsf(Movie_end_as_root(B));

Note that reading a union of string type requires a cast which can be
seen in the full test case in [monster_test.c].
## Error Handling

The API generally expects all error codes to be checked but the
following table and vector operations will accept and return an error:

- `add` null reference to table, vector, or string.
- `push` null reference to table or string.
- `buffer_end/create` null reference to root.

This can simplify pushing or adding atomically created objects, for
example by adding a cloned vector to table field.

It is especially important to check start operations because the builder
will not be in the expected stack frame context after failure and will
not have reserved necessary internal memory, for example when adding a
table field.

On a server with reasonable amount of memory using the default
allocator, and with an emitter that will not return errors, and when it
can be expected that inputs will not exceed the size contraints of the
flatbuffer data types, and if the api is being used correctly, then there
are no reason for failure and error handling may be skipped. However,
it is sometimes desireable for servers to restrict a single clients
memory usage, and then errors are very likely unless the source data is
already limited. As an opposite example, an embedded device sending
small network packages using a fixed but large enough allocation pool,
would be in total control and need not be concerned with any errors.



## Type System Overview

The generated methods for building buffers may look the same but
have different semantics. For example `_clone` on a table field
such as `Monster_enemy_clone` will actually create a table based
on the content of a table in a another buffer, then add that
table to the currently open table. But `Monster_clone` will
create clone and just return a reference without adding the
reference to any table. There is also `push_clone` which adds
an element to an open vector. The same applies to many other
operations.

Basically there are
the following different types of methods:

- Methods on native flatbuffer types, such as
  `flatbuffer_string_start`.
- Methods on generated types types such as `Monster_start`
- Methods on field members such as as `Monster_emeny_start`
- Methods on vectors on vectors of the above such as
  `flatbuffers_string_vec_start`, `Monster_vec_start`.
  `Monster_inventory_vec_start`.
- Slight adaptions for buffer roots and nested buffer roots.

For unions and union vectors the story is more complex - and the
api might need to be cleaned up further, but generally there are
both union type fields, union value fields, and union fields
representing both, and vectors of the same. In additional there
are pseudo fields for each union member because `create` on a
union does not make sense, but
`Monster_myvariant_MyTable_create` does create and `MyTable`
table and assigns it with the correct type to the field
`Monster_myvariant_type` and `Monster_myvariant.


## Cloning

As of flatcc v0.5.2 it is also possible to clone tables, unions,
vectors of tables, vectors of strings, and vectors of unions.
Previously many operations did have a clone or a `push_clone`
operator, but these were all raw byte copies. Table cloning and
union cloning is signficantly more involved as it a simple copy
will not work due to stored references, possible sharing of
references and because the required alignment of table is hard
to reason about without building a new table. Unions and union
vectors are even more difficult.

That said, cloning is now implemented for all relevant data
types.

All clone operations expect the content to originate from
another finalized buffer. For scalars and structs there are
copy operations that are almost the same as clone - they both
avoid endian conversion.

Structs have a special case with clone and copy: Whenever a
struct is stored inline in the desitination buffer, it behaves
like copy. Whenever the destination is a buffer root, or a union
member, the result is a reference to an independent memory
block. When calling clone on a struct type the destination is
unknown and a indendpendent reference is created. If this is not
the intention a `copy` operation can be used. When used field
methods the destination type is known at the right thing will
happen.

Cloning a table will, by default, expand any shared references
in the source into separate copies. This is also true when
cloning string vectors, or any other data that holds references.
Worst case this can blow up memory (which is also true when
printing JSON from a buffer).

It is possible to preserve the exact DAG structure when cloning.
It may not worthwhile for simple use cases but it goes as
follows:

The builder has a pointer to a `flatcc_refmap_t` object. This is
a fairly small stack allocated object that implements a
hashtable. By default this pointer is null, and we have the
above mentioned expansion. If it is not null, each newly cloned
object will have its reference stored in the refmap. The next
time the same object is cloned, the existing reference will be
taken from the refmap instead. See source comments in
`flatcc_refmap.h` and `flatcc_builder.h`, and `monster_test.c`
clone tests.

Note that, for example, it might be relevant to preserve DAG
structure when cloning one object with all its sub-objects, but
if it is cloned a second time, a new copy is desired still while
preseving the inner DAG structure. This can be done by working
with multiple refmaps and simple swapping them out via
`flatcc_builder_set_refmap`. It is also possible to add
references manually to a refmap before cloning.

Warning: the refmap MUST not hold any foreign references when
starting a nested root clone or when cloning inside a nested
buffer that has been started but not ended because it is
invalid to share references between buffers and there are no
safety checks for this.


## Picking

Picking is a method that is related to clone and also introduced
with flatcc 0.5.2. A pick method is only defined on a table
field or a struct field. Instead of taking an a read reference
of same type as the field, it takes a reference to to the same
container type (table or struct). Essentially pick means: find
myself in the other table, clone me, and and me to the new table
which is currently open. So clone takes an entire table where
pick takes a single field. Table cloning is implemented as a
sequence of pick method, one for each field as can be seen in
the generated builder source. A pick operation does nothting if
the field is not set. Pick also works with refmaps because it
does an internal clone operation.  In the generated code, only
clone on types will use the refmap but other clone and pick
operations do depend on these type clone methods.


## Sorting Vectors

Vectors can be sorted, but not by the primary builder interface because:

String and table elements cannot be accessed after they have been
emitted. The emitter can do all sorts of async operations other than
actually building a buffer, for example encrypting blocks and / or send
partial buffers over the network. Scalars could be sorted, but the most
efficient way of emitting vectors does not create a temporary vector but
emits the source directly when endianess allows for it. Less
significant, the buffer producer is likely busy processing content and /
or on a resource constrained device. Altogether, it is much simpler to
not support sorting at this interface level.

To understand how sorting is implemented, lets first look at how an
already sorted vector can be searched:

Every vector of string, scalar and enum element types have a `find`
operation in the reader interface that performs a binary seach. Every
vector of table and struct elements have a `find_by_<field_name>` iff
there is a key attribute on at least one top-level scalar, enum or
string field type. FlatBuffers do not officially allow for multiple key
attributes, but if enabled, there will by a `find_by` operation for
every keyed element field. In addition there is a `find` operation that
maps to the first keyed field.

The read interface returns a vector type, which is a const pointer, when
accessing a table field of vector type. The find operation takes such a
vector as first argument, and a key as second. Strings have variations
to allow for keys with a given length (similar to strcmp vs strncmp).

This leads us to the sort interface:

Every `find` and `find_by` operation has a matching `sort` and `sort_by`
operation table and struct vectors maps `sort` to the first keyed
`sort_by` operation. The sort operation takes a non-const vector which
has the type name suffix `_mutable_vec_t`. These
vectors are not available via the reader interface and must be cast
explicitly from `_vec_t` to `_mutable_vec_t`. When this is done, the
vector can be sorted in-place in the buffer without any memory
allocation and without any recursion.



If the namespace is
`flatbuffers`, a string vector is sorted by:

    flatbuffers_string_vec_t vec;
    vec = ...;
    `flatbuffers_string_vec_sort((flatbuffers_string_mutable_vec_t)vec)`

Scalar and enum vectors have similar inline sort operations, for
example:

    flatbuffers_uint8_vec_sort(flatbuffer_uint8_mutable_vec_t vec);

For vectors of tables or structs the sort function is named by the key
field. Assuming the Monster table has a key attribute on the `Hp` field,
the following sort operation is available:

    MyGame_Example_Monster_vec_t monsters;
    monsters = ...;
    MyGame_Example_Monster_vec_sort_by_Hp(
        (MyGame_Example_Monster_mutable_vec_t)monsters);

Note: this is the reader interface. Any kind of `ref_t` type used by the
builder do not apply here. (Advanced: if an emitter builds a buffer, the
ref type can be used to find the actual vector pointer and then it can
be sorted by casting the pointer to a vector, even if the buffer isn't
finished).

Multiple keys per table or struct is an optional feature. Each key will
have its own sort and find function similar to the above. The first key
also has the shortcut:

    MyGame_Example_Monster_vec_sort(m);

The current implementation uses heap sort which is nearly as fast as
quicksort and has a compact implementation that does not require
recursion or external memory and is robust against DOS attacks by having
worst case O(n log n). It is, however, not a stable sort. The sort
assumes struct have a reasonable size so swap operations can be done
efficiently. For large structs a decicated sort operation building an
external index vector would be better, but this is not supported.

Note that a DAG is valid so there can be multiple vectors referring to
the same table elements, and each can be sorted by a different key.

The find operations are stable meaning they always return the lowest
index of any matching key or `flatbuffers_not_found` which is larger
than any other index.

### Dangers of Sorting

If a buffer was received over, say, an untrusted network the buffer
should be verified before being accessed. But verification only makes it
safe to read a buffer, not to modify a buffer because for example two
vectors can be crafted to overlap each other without breaking any
verification rules.

Thus, sorting is intended to be done shortly after the buffer is
constructed while it can still be trusted.

Using find on a buffer that is supposed to be sorted, but isn't, can
yield unexpected search results, but the result will always be a one
element in the vector being searched, not a buffer overrun.

### Scanning

Some vectors can be sorted by different keys depending on which version
version of `_sort_by` is being used. Obviously `_find_by` must match the
sorted key.

If we need to search for a key that is not sorted, or if we simply do
not want to sort the vector, it is possible to use scanning operations
instead by using `_scan` or `_scan_by`. Scanning is similar to find
except that it does a linear search and it supports scanning from a
given position.

More information on scanning in the
[README](https://github.com/dvidelabs/flatcc#searching-and-sorting)
file, and in the [monster_test.c] test file.


## Example of different interface type users

A resource constrained microcontroller is building flatbuffers from
sensor data using an emitter that sends UDP packages of the flatbuffer
as soon as enough data is ready. A server reassembles the packages or
discards them if any UDP package was lost. One the package is assembled,
the server sorts specific vectors such as temparture levels in the buffer
before it sends the buffer upstream to a storage service through a
TCP/IP connection. The analyzers perform taks such as detecting
abnormal temparature readings based on the sorted vector data.

In the above example, the original sensor devices are not interested in
the reader interface nor the sort interface. While the sort and find
operations may be available, it is dead inline code that does not
inflate the binary codes image size, but the libflatccrt library is
linked in. The collecting server is not interested in the builder
interface and does not link with the `libflatccrt` library but uses
both the inline functions of the reader intrface and the sort interface.
The upstream data storage service uses no interface at all since it
treats the buffers as binary blobs in a database indexed by device and
time. The end users only use the read only interface to visualize and
analyze and has no need for the builder or the sort interface.


## Special Emitters

An emitter only need to implement one function to replace or wrap the
default emitter. See [flatcc_builder.h] on `flatcc_builder_emit_fun` for
details, and also `emit_test.c` for a very simple custom emitter that
just prints debug messages, and [flatcc_emitter.h].

When adding padding `flatcc_builder_padding_base` is used as base in iov
entries and an emitter may detect this pointer and assume the entire
content is just nulls. Usually padding is of limited size by its very
nature so the benefit of handling this is also limited, but it, or a
similar user provided constants can be used for similar purposes:

When creating a vector in a single operation from an external C-array,
no copying takes place on the internal builder stack. Therefore it is
valid to provide a null pointer or a valid array such as
`flatcc_builder_padding_base` that is is too small for the given length,
provided that the emitter is aware of it. This in turn can be used to
allocate space in the emitters internal datastructure so the vector can
be filled after the fact if so desired. Pointer tagging may be another
way to communicate special intent. Be aware that only `create` calls
support this - any `append`, `start/end` or other dynamic operation will
require valid inpout and will stack allocate temporary space.

Emitters always receive a small table of iov entries that together form
a single object including necessary headers and padding, for example a
vector, a string, a nested buffer header, or a vtable. This is
guaranteed by the api, but there is no coordination to provide details
about which call is in order to keep the interface simple and fast. If
this is desired the user must hint the emitter out of band before
calling the relevant build operation. This can also be one indirectly by
setting `user_state` in the emitter and have the emitter inspect this
setting.

When adding vectors piecemeal using `append` or similar as opposed to
zero or less than zero copy approach above, the memory cost is obviously
higher, but unless the individual objects grow large, the stack will
operate in hot cpu cache so the bandwidth from main memory to cpu and
back will not necessarily double. If the stack grows large it may also
be worthwhile trimming the stack with a custom allocator and custom
builder reset between buffers to reduce stack size and initialization
overhead.

[monster_test.c]: https://github.com/dvidelabs/flatcc/blob/master/test/monster_test/monster_test.c
[flatcc_builder.h]: https://github.com/dvidelabs/flatcc/blob/master/include/flatcc/flatcc_builder.h
[flatcc_emitter.h]: https://github.com/dvidelabs/flatcc/blob/master/include/flatcc/flatcc_emitter.h
[monster_test.fbs]: https://github.com/dvidelabs/flatcc/blob/master/test/monster_test/monster_test.fbs
