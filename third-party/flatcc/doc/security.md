# Security Considerations

This sections covers questions such as when is it safe to access a
buffer without risking access violation, buffer overruns, or denial of
service attacks, but cannot possibly cover all security aspects.

## Reading Buffers

When reading a buffer you have to know the schema type before
reading and it is preferable to know the size of the buffer although not
strictly required. If the type is not known, the `file_type`, aka buffer
type, can be checked. This no guarantee due to collisions with other
data formats and because the identifier field may be absent or
misleading. The identifier therefore works best on buffers that can be
trusted.

If a buffer cannot be trusted, such as when receiving it over a public
network, it may be the case that buffer type is known, but it is not
known if someone uses an incorrect implementation of FlatBuffers, or if
the buffer has somehow been corrupted in transit, or someone
intentionally tampered with the buffer. In this case the buffer can be
verified. A verification does not prove that the buffer has the correct
type, but it does prove that it is safe to read (not write) from the
buffer. The buffer size must be known in order to verify the buffer. If
the buffer has a wrong type, but still (unlikey but possible) passes
verification, then unexpected data may be read from the buffer, but it
will not cause any crashes when using the API correctly.

It is preferable to know the required alignment of a buffer which isn't
trivially available unless retrieved from the builder when the buffer is
created. The buffer alignment can be deduced from the schema.

On many systems there is no real problem in accessing a buffer
unaligned, but for systems where it matters, care must be taken because
unaligned access can result in slow performance or access violations.
Even on systems where alignment matters, a standard malloc operation is
often sufficient because it normally aligns to the largest word that
could cause access violations when unaligned. For special use case such
as GPU memory access more alignment may be needed and FlatBuffers
support higher alignments in the schema. Portable `aligned_alloc` and
`aligned_free` support methods are available to help allocate memory with
sufficient alignment. Because compile time flags can change between
compilation of the runtime library and the application,
`flatcc_builder_aligned_free` ensures a consistent deallocation method
for aligned buffers allocated by the runtime library.

A verifier for C requires the buffer to placed in aligned memory and it
will fail if the buffer content is not properly aligned relative to the
buffer or to an absolute memory address regardless of whether the
current systems requires alignment or not. Therefore a buffer verified
on one system is safe to use on all systems. One could use this fact to
sign a buffer, but this is beyond the scope of FlatBuffers itself, and
verifying a signature is likely much slower than re-verifying a buffer
when a verifier is available.

Note: it would be helpful if the verifier allowed verification only
relative to the buffer start instead of requiring the absolute addresses
to be aligned. This would allow verification of buffers before copying
them out of unaligned locations in network buffers and also allow
subsequent reading of such buffers without copying iff the system
supports unaligned access. However, the verifier does not currently
support this.

It is not always safe to verify a buffer. A buffer can be constructed to
trigger deep nesting. The FlatCC verifier has a hard coded non-exact
limit of about 100 levels. This is to protection stack recursion. If the
limit is exceeded, the verifier will safely fail. The limit can be
changed with a compile time flag. If the limit is too permissive a
system may run into stack overflow, but it is unlikely on most systems
today. Typical application code may have similar recursive access
functions. Therefore it is likely that recursion is safe if the verifier
succeeds but it depends on the application.

A buffer can point to the same data from multiple places. This is known
as a DAG. The verifier rejects cycles that could lead to infinite loops
during application traversal but does permit DAGs. For normal use DAGs
are safe but it is possible to maliciously construct a buffer with a
long vector where all elements points to a table that also has a vector
of a similar nature. After a few levels, this can lead to a finite but
exponentially large number of places to visit. The current FlatCC
verifier does not protect against this but Googles flatc compiler has a
verifier the limits the number of visited tables.

When reading a buffer in C no memory allocation takes place after the
buffer has initially been placed in memory. For example, strings can be
read directly as C strings and strings are 0-terminated. A string might
contain embedded 0 bytes which is not strictly correct but permitted.
This will result in a shorter string if used naively as a C string
without reading the string length via the API but it will still be safe.
Other languages might have to allocate memory for objects such as
strings though.

A field can generally be absent. Scalar fields are always safe to
access, even if they are absent, because they have a default value that
will be returned. Tables, Vectors, Strings, and Structs may return null
when a field is absent. This is perfectly valid but if the application
does not check for null, this can lead to an access violation.

A field can marked 'required' in the schema. If it is required, it will
never return null unless the buffer is invalid. A verifier will detect
this. On a practical note some FlatBuffer builders might not enforce the
required field and readers do not always verify buffers before access
(nor should they have to) - therefore an application is advised to check
return values even on required fields unless the buffer is entirely
trusted.

If a buffer is verified, it is safe to access all vector elements up to
its size. Access of elements via API calls do not necessarily check for
out of bounds but some might.

A buffer may also be encoded in big endian format. This is not standard,
but FlatCC supports for systems that are primarily big endian. The
buffer identifier will usually detect the difference because the
identifier will be byte swapped. A reader therefore need to be aware of
this possiblity, but most often this is not a concern since standard
FlatBuffers are always little endian. The verifier will likely fail an
unpexcted endian encoding but at least make it safe to access.


## Thread Safety

There is no thread safety on the FlatBuffers API but read access does
not mutate any state. Every read location is a temporary variable so as
long as the application code is otherwise sane, it is safe read a buffer
from multiple threads and if the buffer is placed on cache line
alignment (typically 64 or 128 bytes) it is also efficient without false
sharing.

A verifier is also safe to use because it it only reads from a buffer.

A builder is inherently NOT safe for multihreaded access. However, with
proper synchronization there is nothing preventing one thread from doing
the grunt work and another putting the high level pieces together as
long as only one thread at a time is access the builder object, or the
associated allocator and emitter objects. From a performance perspective
this doesn't make much sense, but it might from an architectural
perspective.

A builder object can be cleared and reused after a buffer is constructed
or abandoned. The clear operation can optionally reduce the amount of
memory or keep all the memory from the previous operation. In either
case it is safe for new thread to use the builder after it is cleared
but two threads cannot use the builder at the same time.

It is fairly cheap to create a new builder object, but of course cheaper
to reuse existing memory. Often the best option is for each thread to
have its own builder and own memory and defer any sharing to the point
where the buffer is finished and readable.


## Schema Evolution

Accessing a buffer that was created by a more recent of a FlatBuffers
schema is safe iff the new version of the schema was created according
the guidelines for schema evolution - notably no change of field
identifiers or existing enum values and no removal or deprecation of
required fields. Googles flatc tool can check if a new schema version is
safe.

Fields that are not required but deprecated in a new version will still
be safe to access by old version but they will likely read default or
null values and should be prepared for this.


## Copying or Printing Buffer Content

Even if it is safe to read a buffer, it is not safe to copy or even
print a buffer because a DAG can unfold to consume much more output
space than the given input. In data compression this is known as a Zip
Bomb but even without malicious intent, users need to aware of the
potential expansion. This is also a concern when printing JSON.

A table also cannot be trivially copied based on memory content because
it has offsets to other content. This is not an issue when using any
official API but may be if a new buffer is attempted to be constructed
by Frankenstein from parts of existing buffers.

Nested buffers are not allowed to share any content with its parents,
siblings or child buffers for similar reasons.

The verifier should complain if a buffer goes out if bounds.


## Modifying Buffers

It is not safe to modify buffers in-place unless the buffer is
absolutely trusted. Verifying a buffer is not enough. FlatC does not
provide any means to modify a buffer in-place but it is not hard to
achieve this is if so desired. It is especially easy to do this with
structs, so if this is needed this is the way to do it.

Modifying a buffer is unsafe because it is possible to place one table
inside another table, as an example, even if this is not valid. Such
overlaps are too expensive to verify by a standard verifier. As long as
the buffer is not modified this does not pose any real problem, but if a
field is modified in one table, it might cause a field of another table
to point out of bounds. This is so obvious an attack vector that anyone
wanting to hack a system is likely to use this approach. Therefore
in-place modification should be avoided unless on a trusted platform.
For example, a trusted network might bump a time-to-live counter when
passing buffers around.

Even if it is safe to modify buffers, this will not work on platforms
that require endian conversion. This is usually big endian platforms,
but it is possible to compile flatbuffers with native big endian format
as well.
platforms unless extra precautions are taken. FlatCC has a lot of
low-level `to/from_pe` calls that performs the proper


## Building Buffers

A buffer can be constructed incorrectly in a large number of ways that
are not efficient to detect at runtime.

When a buffer is constructed with a debug library then assertions will
sometimes find the most obvious problems such as closing a table after
opening a vector. FlatCC is quite permissive in the order of object
creation but the nesting order must be respected, and it cannot type
check all data. Other FlatBuffer builders typically require that child
objects are completed created before a parent object is started. FlatCC
does not require this but will internally organize objects in a
comptible way. This removes a number of potential mistakes, but not all.

Notably a table from a parent object or any other external reference
should not be used in a nested buffer.

It is a good idea to run a verifier on a constructed buffer at least
until some confidence has been gained in the code building buffers.

If a buffer needs to be constructed with sorted keys this cannot be done
during construction, unlike the C++ API because the builder allocates as
little memory as possible. Instead the reader interface supports a
mutable cast for use with a sort operation. This sort operation must
only be used on absolutely trusted buffers and verification is not
sufficient if malicous overlaps can be expected.

The builder will normally consume very little memory. It operates a few
stacks and a small hash table in additional to a circular buffer to
consum temporary buffer output. It is not possible to access constructed
buffer objects buffer the buffer is complete because data may span
multiple buffers. Once a buffer is complete the content can be
copied out, or a support function can be used allocate new memory with
the final buffer as content.

The internal memory can grow large when the buffer grows large,
naturally. In addition the temporary stacks may grow large if there are
are large tables or notably large vectors that cannot be be copied
directly to the output buffers. This creates a potential for memory
allocation errors, especially on constrained systems.

The builder returns error codes, but it is tedious to check these. It is
not necessary to check return codes if the API is used correctly and if
there are no allocation errors. It is possible to provide a custom
allocator and a custom emitter. These can detect memory failures early
making it potentially safe to use the builder API without any per
operation checks.

The generated JSON parser checks all return codes and can be used to
construct a buffer safely, especially since the buffer is naturally
bounded by the size of the JSON input. JSON printing, on the other hand,
can potentially explode, as discussed earlier.

FlatCC generated create calls such as `MyGame_Example_Monster_create()`
will not be compatible across versions if there are deprecated fields
even if the schema change otherwise respects schema evolutation rules.
This is mostly a concern if new fields are added because compilation
will otherwise break on argument count mismatch. Prior to flatcc-0.5.3
argument order could change if the field (id: x) attribute was used
which could lead to buffers with unexpected content. JSON parsers that
support constructors (objects given as an array of create arguments)
have similar concerns but here trailing arguments can be optional.
