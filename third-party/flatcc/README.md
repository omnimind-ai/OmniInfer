Ubuntu, macOS and Windows: [![Build Status](https://github.com/dvidelabs/flatcc/actions/workflows/ci.yml/badge.svg)](https://github.com/dvidelabs/flatcc/actions/workflows/ci.yml)
Windows: [![Windows Build Status](https://ci.appveyor.com/api/projects/status/github/dvidelabs/flatcc?branch=master&svg=true)](https://ci.appveyor.com/project/dvidelabs/flatcc)
Weekly: [![Build Status](https://github.com/dvidelabs/flatcc/actions/workflows/weekly.yml/badge.svg)](https://github.com/dvidelabs/flatcc/actions/workflows/weekly.yml)


_The JSON parser may change the interface for parsing union vectors in a
future release which requires code generation to match library
versions._

# FlatCC FlatBuffers in C for C

`flatcc` has no external dependencies except for build and compiler
tools, and the C runtime library. With concurrent Ninja builds, a small client
project can build flatcc with libraries, generate schema code, link the project
and execute a test case in a few seconds, produce binaries between 15K and 60K,
read small buffers in 30ns, build FlatBuffers in about 600ns, and with a larger
executable also handle optional json parsing or printing in less than 2 us for a
10 field mixed type message.


<!-- vim-markdown-toc GFM -->

* [Online Forums](#online-forums)
* [Introduction](#introduction)
  * [Integrating FlatCC](#integrating-flatcc)
* [Project Details](#project-details)
* [Reporting Bugs](#reporting-bugs)
* [Status](#status)
  * [Main features supported as of 0.6.1](#main-features-supported-as-of-061)
  * [Supported platforms (CI tested)](#supported-platforms-ci-tested)
  * [Platforms reported to work by users](#platforms-reported-to-work-by-users)
  * [Portability](#portability)
* [Time / Space / Usability Tradeoff](#time--space--usability-tradeoff)
* [Generated Files](#generated-files)
  * [Use of Macros in Generated Code](#use-of-macros-in-generated-code)
  * [Extracting Documentation](#extracting-documentation)
* [Using flatcc](#using-flatcc)
* [Trouble Shooting](#trouble-shooting)
* [Quickstart](#quickstart)
  * [Reading a Buffer](#reading-a-buffer)
  * [Compiling for Read-Only](#compiling-for-read-only)
  * [Building a Buffer](#building-a-buffer)
  * [Verifying a Buffer](#verifying-a-buffer)
  * [Potential Name Conflicts](#potential-name-conflicts)
  * [Debugging a Buffer](#debugging-a-buffer)
* [File and Type Identifiers](#file-and-type-identifiers)
  * [File Identifiers](#file-identifiers)
  * [Type Identifiers](#type-identifiers)
* [JSON Parsing and Printing](#json-parsing-and-printing)
  * [Base64 Encoding](#base64-encoding)
  * [Parsing Fixed Length Arrays](#parsing-fixed-length-arrays)
  * [Runtime Flags](#runtime-flags)
  * [Generic Parsing and Printing.](#generic-parsing-and-printing)
  * [Performance Notes](#performance-notes)
* [Global Scope and Included Schema](#global-scope-and-included-schema)
* [Required Fields and Duplicate Fields](#required-fields-and-duplicate-fields)
* [Fast Buffers](#fast-buffers)
* [Types](#types)
* [Unions](#unions)
  * [Union Scope Resolution](#union-scope-resolution)
* [Fixed Length Arrays](#fixed-length-arrays)
* [Optional Fields](#optional-fields)
* [Endianness](#endianness)
* [Pitfalls in Error Handling](#pitfalls-in-error-handling)
* [Searching and Sorting](#searching-and-sorting)
* [Null Values](#null-values)
* [Portability Layer](#portability-layer)
* [Building](#building)
  * [Unix Build (OS-X, Linux, related)](#unix-build-os-x-linux-related)
  * [Windows Build (MSVC)](#windows-build-msvc)
  * [Docker](#docker)
  * [Cross-compilation](#cross-compilation)
  * [Custom Allocation](#custom-allocation)
  * [Custom Asserts](#custom-asserts)
  * [Shared Libraries](#shared-libraries)
  * [Strict Aliasing](#strict-aliasing)
* [Distribution](#distribution)
  * [Unix Files](#unix-files)
  * [Windows Files](#windows-files)
* [Running Tests on Unix](#running-tests-on-unix)
* [Running Tests on Windows](#running-tests-on-windows)
* [Configuration](#configuration)
* [Using the Compiler and Builder library](#using-the-compiler-and-builder-library)
* [FlatBuffers Binary Format](#flatbuffers-binary-format)
* [Security Considerations](#security-considerations)
* [Style Guide](#style-guide)
* [Benchmarks](#benchmarks)

<!-- vim-markdown-toc -->

## Online Forums

- [Discord - FlatBuffers](https://discord.gg/6qgKs3R)
- [Github - FlatCC Discussions](https://github.com/dvidelabs/flatcc/discussions)

## Introduction

This project builds flatcc, a compiler that generates FlatBuffers code for
C given a FlatBuffer schema file. This introduction also creates a separate test
project with the traditional monster example, here in a C version.

For now assume a Unix like system although that is not a general requirement -
see also [Building](#building). You will need git, cmake, bash, a C compiler,
and either the ninja build system, or make.

    git clone https://github.com/dvidelabs/flatcc.git
    cd flatcc
    # scripts/initbuild.sh ninja
    scripts/initbuild.sh make
    scripts/setup.sh -a ../mymonster
    ls bin
    ls lib
    cd ../mymonster
    ls src
    scripts/build.sh
    ls generated

`scripts/initbuild.sh` is optional and chooses the build backend, which defaults
to ninja.

The setup script builds flatcc using CMake, then creates a test project
directory with the monster example, and a build script which is just a small
shell script. The headers and libraries are symbolically linked into the test
project. You do not need CMake to build your own projects once flatcc is
compiled.

To create another test project named foobar, call `scripts/setup.sh -s -x
../foobar`. This will avoid rebuilding the flatcc project from scratch.

### Integrating FlatCC

To answer the question:

> "How do I integrate FlatCC into my project with minimal dependencies"

You do not need CMake and you do not need a prebuilt runtime library.

You do need the `flatcc` tool to initially generate your own schema files and
then you need to update your project to:

- Include the generated header files.
- Add `include/flatcc` (correct version) to the include path, e.g., using `-I include` if at flatcc root.
- Link with zero or more files from `src/runtime/*.c` (correct version).
- Possibly disable certain warnings (notably disable GCC pedantic).
- Potentially work around platform limitations, e.g., using `-DPORTABLE_HAS_INCLUDE_STDALIGN=0` if `stdalign.h`
  should be, but isn't, available on your platform. See `include/flatcc/portable/*.h` for details.

You should be able to compile a test project with just a single line like:

```
cc -I include src/runtime/build.c myprogram.c
```

If you only read flatbuffer files, and do not build, verify, parse JSON, or print JSON,
you will not need any runtime `.c` files.

If you have a system installed `flatcc` package, it might have the wrong
version of header or library files. You need to somehow override that if you
have generated files from a more recent version of flatcc.

That said, you can also use the main CMake project to build libraries, or
just look at it to see which warnings and flags are important to your compiler.

See also [Building](#building).


## Project Details

NOTE: see
[CHANGELOG](https://github.com/dvidelabs/flatcc/blob/master/CHANGELOG.md).
There are occassionally minor breaking changes as API inconsistencies
are discovered. Unless clearly stated, breaking changes will not affect
the compiled runtime library, only the header files. In case of trouble,
make sure the `flatcc` tool is same version as the `include/flatcc`
path.

The project includes:

- an executable `flatcc` FlatBuffers schema compiler for C and a
  corresponding library `libflatcc.a`. The compiler generates C header
  files or a binary flatbuffers schema.
- a typeless runtime library `libflatccrt.a` for building and verifying
  flatbuffers from C. Generated builder headers depend on this library.
  It may also be useful for other language interfaces. The library
  maintains a stack state to make it easy to build buffers from a parser
  or similar.
- a small `flatcc/portable` header only library for non-C11 compliant
  compilers, and small helpers for all compilers including endian
  handling and numeric printing and parsing.


See also:

- [Reporting Bugs](https://github.com/dvidelabs/flatcc#reporting-bugs)

- [Google FlatBuffers](http://google.github.io/flatbuffers/)

- [Build Instructions](https://github.com/dvidelabs/flatcc#building)

- [Quickstart](https://github.com/dvidelabs/flatcc#quickstart)

- [Builder Interface Reference]

- [Benchmarks]

The `flatcc` compiler is implemented as a standalone tool instead of
extending Googles `flatc` compiler in order to have a pure portable C
library implementation of the schema compiler that is designed to fail
graciously on abusive input in long running processes. It is also
believed a C version may help provide schema parsing to other language
interfaces that find interfacing with C easier than C++. The FlatBuffers
team at Googles FPL lab has been very helpful in providing feedback and
answering many questions to help ensure the best possible compatibility.
Notice the name `flatcc` (FlatBuffers C Compiler) vs Googles `flatc`.

The JSON format is compatible with Googles `flatc` tool. The `flatc`
tool converts JSON from the command line using a schema and a buffer as
input. `flatcc` generates schema specific code to read and write JSON
at runtime. While the `flatcc` approach is likely much faster and also
easier to deploy, the `flatc` approach is likely more convenient when
manually working with JSON such as editing game scenes. Both tools have
their place.

**NOTE: Big-endian platforms are only supported as of release 0.4.0.**


## Reporting Bugs

If possible, please provide a short reproducible schema and source file
with a main program the returns 1 on error and 0 on success and a small
build script. Preferably generate a hexdump and call the buffer verifier
to ensure the input is valid and link with the debug library
`flatccrt_d`.

See also [Debugging a Buffer](#debugging-a-buffer), and [readfile.h]
useful for reading an existing buffer for verification.

Example:

[samples/bugreport](samples/bugreport)

eclectic.fbs :

```c
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
```

myissue.c :

```c
/* Minimal test with all headers generated into a single file. */
#include "build/myissue_generated.h"
#include "flatcc/support/hexdump.h"

int main(int argc, char *argv[])
{
    int ret;
    void *buf;
    size_t size;
    flatcc_builder_t builder, *B;

    (void)argc;
    (void)argv;

    B = &builder;
    flatcc_builder_init(B);

    Eclectic_FooBar_start_as_root(B);
    Eclectic_FooBar_say_create_str(B, "hello");
    Eclectic_FooBar_meal_add(B, Eclectic_Fruit_Orange);
    Eclectic_FooBar_height_add(B, -8000);
    Eclectic_FooBar_end_as_root(B);
    buf = flatcc_builder_get_direct_buffer(B, &size);
#if defined(PROVOKE_ERROR) || 0
    /* Provoke error for testing. */
    ((char*)buf)[0] = 42;
#endif
    ret = Eclectic_FooBar_verify_as_root(buf, size);
    if (ret) {
        hexdump("Eclectic.FooBar buffer for myissue", buf, size, stdout);
        printf("could not verify Electic.FooBar table, got %s\n", flatcc_verify_error_string(ret));
    }
    flatcc_builder_clear(B);
    return ret;
}
```
build.sh :
```sh
#!/bin/sh
cd $(dirname $0)

FLATBUFFERS_DIR=../..
NAME=myissue
SCHEMA=eclectic.fbs
OUT=build

FLATCC_EXE=$FLATBUFFERS_DIR/bin/flatcc
FLATCC_INCLUDE=$FLATBUFFERS_DIR/include
FLATCC_LIB=$FLATBUFFERS_DIR/lib

mkdir -p $OUT
$FLATCC_EXE --outfile $OUT/${NAME}_generated.h -a $SCHEMA || exit 1
cc -I$FLATCC_INCLUDE -g -o $OUT/$NAME $NAME.c -L$FLATCC_LIB -lflatccrt_d || exit 1
echo "running $OUT/$NAME"
if $OUT/$NAME; then
    echo "success"
else
    echo "failed"
    exit 1
fi
```

## Status

Release 0.6.2 (in development) is primarily a bug fix release, refer to
CHANGELOG for details. A long standing bug has been fixed where where objects
created before a call to _create_as_root would not be properly aligned, and
buffer end is now also padded to largest object seen within the buffer. Note
that for clang debug builds, -fsanitize=undefined has been added and this may
require dependent source code to also use that flag to avoid missing linker
symbols. The feature can be disabled in CMakeLists.txt. CMake has been bumped to
version 3.16 which is the latest version where Appveyor will build MVSC 2015.
Warnings added for potential name conflicts in enum names and table fields
that could result in confusing C compiler errors. Not all warnigs are errors
to the option `-s` was also added to silence warnings. The existing `-g`
option can resolve some conflicts.

Release 0.6.1 contains primarily bug fixes and numerous contributions from the
community to handle platform edge cases. Additionally, pendantic GCC warnings
are disabled, relying instead on clang, since GCC is too aggressive, breaks
builds frequently and works against portability. An existing C++ test case
ensures that C code also works with common C++ compilers, but it can break some
environments, so there is now a flag to disable that test without disabling
all tests. Support for Optional Scalar Values in the FlatBuffer format has
been added. There is also improved support for abstracting memory allocation
on various platforms. `<table>_identifier` has been deprecated in favor
`<table>_file_identifier` in generated code due to `identifier` easily leading
to name conflicts. `file_extension` constant in generated code is now without
prefixed dot (.).

Release 0.6.0 introduces a "primary" attribute to be used together with a key
attribute to chose default key for finding and sorting. If primary is absent,
the key with the lowest id becomes primary. Tables and vectors can now be sorted
recursively on primary keys. BREAKING: previously the first listed, not the
lowest id, would be the primary key. Also introduces fixed length scalar arrays
in struct fields (struct and enum elements are not supported). Structs support
fixed length array fields, including char arrays. Empty structs never fully
worked and are no longer supported, they are also no longer supported by flatc.
NOTE: char arrays are not currently part of Googles flatc compiler - int8 arrays
may be used instead. BREAKING: empty structs are no longer supported - they are
also not valid in Googles flatc compiler. See CHANGELOG for additional changes.
DEPRECATED: low-level `cast_to/from` from functions in `flatcc_accessors.h` will
be removed in favor of `read/write_from/to` because the cast interface breaks
float conversion on some uncommon platforms. This should not affect normal use
but remains valid in this release.

Release 0.5.3 inlcudes various bug fixes (see changelog) and one
breaking but likely low impact change: BREAKING: 0.5.3 changes behavour
of builder create calls so arguments are always ordered by field id when
id attributes are being used, for example
`MyGame_Example_Monster_create()` in `monster_test.fbs`
([#81](https://github.com/dvidelabs/flatcc/issues/81)). Fixes undefined
behavior when sorting tables by a numeric key field.

Release 0.5.2 introduces optional `_get` suffix to reader methods. By
using `flatcc -g` only `_get` methods are valid. This removes potential
name conficts for some field names. 0.5.2 also introduces the long
awaited clone operation for tables and vectors. A C++ smoketest was
added to reduce the number void pointer assignment errors that kept
sneaking in. The runtime library now needs an extra file `refmap.c`.

Release 0.5.1 fixes a buffer overrun in the JSON printer and improves
the portable libraries <stdalign.h> compatibility with C++ and the
embedded `newlib` standard library. JSON printing and parsing has been
made more consistent to help parse and print tables other than the
schema root as seen in the test driver in [test_json.c]. The
[monster_test.fbs] file has been reorganized to keep the Monster table
more consistent with Googles flatc version and a minor schema namespace
inconsistency has been resolved as a result. Explicit references to
portable headers have been moved out of generated source. extern "C" C++
guards added around generated headers. 0.5.1 also cleaned up the
low-level union interface so the terms { type, value } are used
consistently over { type, member } and { types, members }.


### Main features supported as of 0.6.1

- generated FlatBuffers reader and builder headers for C
- generated FlatBuffers verifier headers for C
- generated FlatBuffers JSON parser and printer for C
- ability to concatenate all output into one file, or to stdout
- robust dependency file generation for build systems
- binary schema (.bfbs) generation
- pre-generated reflection headers for handling .bfbs files
- cli schema compiler and library for compiling schema
- runtime library for builder, verifier and JSON support
- thorough test cases
- monster sample project
- fast build times
- support for big endian platforms (as of 0.4.0)
- support for big endian encoded flatbuffers on both le and be platforms. Enabled on `be` branch.
- size prefixed buffers - see also [Builder Interface Reference]
- flexible configuration of malloc alternatives and runtime
  aligned_alloc/free support in builder library.
- feature parity with C++ FlatBuffers schema features added in 2017
  adding support for union vectors and mixed type unions of strings,
  structs, and tables, and type aliases for uint8, ..., float64.
- base64(url) encoded binary data in JSON.
- sort fields by primary key (as of 0.6.0)
- char arrays (as of 0.6.0)
- optional scalar values (as of 0.6.1)

There are no plans to make frequent updates once the project becomes
stable, but input from the community will always be welcome and included
in releases where relevant, especially with respect to testing on
different target platforms.


### Supported platforms (CI tested)

As of flatcc v.0.6.2 the weekly build on github actions tests a range of
gcc and clang versions on Ubuntu and macOS but the lists changes slightly
as the underlying CI runners sometimes drop a compiler version. Windows
is tested on Appveyor with MSVC 2025 and a later version on Github actions.
See the appveyour and github actions workflow files for details.

In the past (ca. < flatcc-0.6.2) the following have been tested, and chances are
that an older version of flatcc is still useful if the latest version does not
compile on a given platform.

- Ubuntu Trusty gcc 4.4, 4.6-4.9, 5, 6, 7 and clang 3.6, 3.8
- OS-X current clang / gcc
- Windows MSVC 2010, 2013, 2015, 2015 Win64, 2017, 2017 Win64
- C++11/C++14 user code on the above platforms.

C11/C++11 is the reference that is expected to always work.

The GCC `--pedantic` compiler option is not supported as of GCC-8+
because it forces non-portable code changes and because it tends to
break the code base with each new GCC release.

Older/non-standard versions of C++ compilers cause problems because
`static_assert` and `alignas` behave in strange ways where they are
neither absent nor fully working as expected. There are often
workarounds, but it is more reliable to use `-std=c++11` or
`-std=c++14`.

The portably library does not support GCC C++ pre 4.7 because the
portable library does not work around C++ limitations in stdalign.h and
stdint.h before GCC 4.7. This could be fixed but is not a priority.

The monster sample does not work with MSVC 2010 because it intentionally
uses C99 style code to better follow the C++ version.

The build option `FLATCC_TEST` can be used to disable all tests which
might make flatcc compile on platforms that are otherwise problematic.
The buld option `FLATCC_CXX_TEST` can be disabled specifically for C++
tests (a simple C++ file that includes generated C code).

### Platforms reported to work by users

- ESP32 SoC SDK with FreeRTOS and newlib has been reported to compile
  cleanly with C++ 14 using flatcc generated JSON parsers, as of flatcc
  0.5.1.
- FreeRTOS when using custom memory allocation methods.
- Arduino
- IBM XLC on AIX big endian Power PC has been tested for release 0.4.0
  but is not part of regular release tests.
- IBM s390x big endian via QEMU.

### Portability

There is no reason why other or older compilers cannot be supported, but
it may require some work in the build configuration and possibly
updates to the portable library. The above is simply what has been
tested and configured.

The portability layer has some features that are generally important for
things like endian handling, and others to provide compatibility for
optional and missing C11 features. Together this should support most C
compilers around, but relies on community feedback for maturity.

The necessary size of the runtime include files can be reduced
significantly by using -std=c11 and avoiding JSON (which needs a lot of
numeric parsing support), and by removing `include/flatcc/reflection`
which is present to support handling of binary schema files and can be
generated from `reflection/reflection.fbs`, and removing
`include/flatcc/support` which is only used for tests and samples. The
exact set of required files may change from release to release, and it
doesn't really matter with respect to the compiled code size.


## Time / Space / Usability Tradeoff

The priority has been to design an easy to use C builder interface that
is reasonably fast, suitable for both servers and embedded devices, but
with usability over absolute performance - still the small buffer output
rate is measured in millons per second and read access 10-100 millon
buffers per second from a rough estimate. Reading FlatBuffers is more
than an order of magnitude faster than building them.

For 100MB buffers with 1000 monsters, dynamically extended monster
names, monster vector, and inventory vector, the bandwidth reaches about
2.2GB/s and 45ms/buffer on 2.2GHz Haswell Core i7 CPU. This includes
reading back and validating all data. Reading only a few key fields
increases bandwidth to 2.7GB/s and 37ms/op. For 10MB buffers bandwidth
may be higher but eventually smaller buffers will be hit by call
overhead and thus we get down to 300MB/s at about 150ns/op encoding
small buffers. These numbers are just a rough guideline - they obviously
depend on hardware, compiler, and data encoded. Measurements are
excluding an initial warmup step.

The generated JSON parsers are roughly 4 times slower than building a
FlatBuffer directly in C or C++, or about 2200ns vs 600ns for a 700 byte
JSON message. JSON parsing is thus roughly two orders of magnitude faster
than reading the equivalent Protocol Buffer, as reported on the [Google
FlatBuffers
Benchmarks](http://google.github.io/flatbuffers/flatbuffers_benchmarks.html)
page. LZ4 compression would estimated double the overall processing time
of JSON parsing. JSON printing is faster than parsing but not very
significantly so. JSON compresses to roughly half the size of compressed
FlatBuffers on large buffers, but compresses worse on small buffers (not
to mention when not compressing at all).

It should be noted that FlatBuffer read performance exclude verification
which JSON parsers and Protocol Buffers inherently include by their
nature. Verification has not been benchmarked, but would presumably add
less than 50% read overhead unless only a fraction of a large buffer is to
be read.

See also [Benchmarks].

The client C code can avoid almost any kind of allocation to build
buffers as a builder stack provides an extensible arena before
committing objects - for example appending strings or vectors piecemeal.
The stack is mostly bypassed when a complete object can be constructed
directly such as a vector from integer array on little endian platforms.

The reader interface should be pretty fast as is with less room for
improvement performance wise. It is also much simpler than the builder.

Usability has also been prioritized over smallest possible generated
source code and compile time. It shouldn't affect the compiled size
by much.

The compiled binary output should be reasonably small for everything but
the most restrictive microcontrollers. A 33K monster source test file
(in addition to the generated headers and the builder library) results
in a less than 50K optimized binary executable file including overhead
for printf statements and other support logic, or a 30K object file
excluding the builder library.

Read-only binaries are smaller but not necessarily much smaller than
builders considering they do less work: The compatibility test reads a
pre-generated binary `monsterdata_test.golden` monster file and verifies
that all content is as expected. This results in a 13K optimized binary
executable or a 6K object file. The source for this check is 5K
excluding header files. Readers do not need to link with a library.

JSON parsers bloat the compiled C binary compared to pure Flatbuffer
usage because they inline the parser decision tree. A JSON parser for
monster.fbs may add 100K +/- optimization settings to the executable
binary.


## Generated Files

The generated code for building flatbuffers,
and for parsing and printing flatbuffers, all need access to
`include/flatcc`. The reader does no rely on any library but all other
generated files rely on the `libflatccrt.a` runtime library. Note that
`libflatcc.a` is only required if the flatcc compiler itself is required
as a library.

The reader and builder rely on generated common reader and builder
header files. These common file makes it possible to change the global
namespace and redefine basic types (`uoffset_t` etc.). In the future
this _might_ move into library code and use macros for these
abstractions and eventually have a set of predefined files for types
beyond the standard 32-bit unsigned offset (`uoffset_t`). The runtime
library is specific to one set of type definitions.

Refer to [monster_test.c] and the generated files for detailed guidance
on use. The monster schema used in this project is a slight adaptation
to the original to test some additional edge cases.

For building flatbuffers a separate builder header file is generated per
schema. It requires a `flatbuffers_common_builder.h` file also generated
by the compiler and a small runtime library `libflatccrt.a`. It is
because of this requirement that the reader and builder generated code
is kept separate. Typical uses can be seen in the [monster_test.c] file.
The builder allows for repeated pushing of content to a vector or a
string while a containing table is being updated which simplifies
parsing of external formats. It is also possible to build nested buffers
in-line - at first this may sound excessive but it is useful when
wrapping a union of buffers in a network interface and it ensures proper
alignment of all buffer levels.

For verifying flatbuffers, a `myschema_verifier.h` is generated. It
depends on the runtime library and the reader header.

Json parsers and printers generate one file per schema file and included
schema will have their own parsers and printers which including parsers
and printers will depend upon, rather similar to how builders work.

Low level note: the builder generates all vtables at the end of the
buffer instead of ad-hoc in front of each table but otherwise does the
same deduplication of vtables. This makes it possible to cluster vtables
in hot cache or to make sure all vtables are available when partially
transmitting a buffer. This behavior can be disabled by a runtime flag.

Because some use cases may include very constrained embedded devices,
the builder library can be customized with an allocator object and a
buffer emitter object. The separate emitter ensures a buffer can be
constructed without requiring a full buffer to be present in memory at
once, if so desired.

The typeless builder library is documented in [flatcc_builder.h] and
[flatcc_emitter.h] while the generated typed builder api for C is
documented in [Builder Interface Reference].


### Use of Macros in Generated Code

Occasionally a concern is raised about the dense nature of the macros
used in the generated code. These macros make it difficult to understand
which functions are actually available. The [Builder Interface Reference]
attempts to document the operations in general fashion. To get more
detailed information, generated function prototypes can be extracted
with the `scripts/flatcc-doc.sh` script.

Some are also concerned with macros being "unsafe". Macros are not
unsafe when used with FlatCC because they generate static or static
inline functions. These will trigger compile time errors if used
incorrectly to the same extend that they would in direct C code.

The expansion compresses the generated output by more than a factor 10
ensuring that code under source control does not explode and making it
possible to compare versions of generated code in a meaningful manner
and see if it matches the intended schema. The macros are also important
for dealing with platform abstractions via the portable headers.

Still, it is possible to see the generated output although not supported
directly by the build system. As an example,
`include/flatcc/reflection` contains pre-generated header files for the
reflection schema. To see the expanded output using the `clang` compiler
tool chain, run:

	clang -E -DNDEBUG -I include \
			include/flatcc/reflection/reflection_reader.h | \
	clang-format

Other similar commands are likely available on platforms not supporting
clang.

Note that the compiler will optimize out nearly all of the generated
code and only use the logic actually referenced by end-user code because
the functions are static or static inline. The remaining parts generally
inline efficiently into the application code resulting in a reasonably
small binary code size.

More details can be found in
[#88](https://github.com/dvidelabs/flatcc/issues/88)


### Extracting Documentation

The expansion of generated code can be used to get documentation for
a specific object type.

The following script automates this process:

    scripts/flatcc-doc.sh <schema-file> <name-prefix> [<outdir>]

writing function prototypes to `<outdir>/<name-prefix>.doc`.

Note that the script requires the clang compiler and the clang-format
tool, but the script could likely be adapted for other tool chains as well.

The principle behind the script can be illustrated using the reflection
schema as an example, where documentation for the Object table is
extracted:

    bin/flatcc reflection/reflection.fbs -a --json --stdout | \
        clang - -E -DNDEBUG -I include | \
        clang-format -style="WebKit" | \
    	grep "^static.* reflection_Object_\w*(" | \
        cut -f 1 -d '{' | \
        grep -v deprecated | \
        grep -v ");" | \
        sed 's/__tmp//g' | \
        sed 's/)/);/g'

The WebKit style of clang-format ensures that parameters and the return
type are all placed on the same line. Grep extracts the function headers
and cut strips function bodies starting on the same line. Sed strips
`__tmp` suffix from parameter names used to avoid macro name conflicts.
Grep strips `);` to remove redundant forward declarations and sed then
adds ; to make each line a valid C prototype.

The above is not guaranteed to always work as output may change, but it
should go a long way.

A small extract of the output, as of flatcc-v0.5.2

	static inline size_t reflection_Object_vec_len(reflection_Object_vec_t vec);
	static inline reflection_Object_table_t reflection_Object_vec_at(reflection_Object_vec_t vec, size_t i);
	static inline reflection_Object_table_t reflection_Object_as_root_with_identifier(const void* buffer, const char* fid);
	static inline reflection_Object_table_t reflection_Object_as_root_with_type_hash(const void* buffer, flatbuffers_thash_t thash);
	static inline reflection_Object_table_t reflection_Object_as_root(const void* buffer);
	static inline reflection_Object_table_t reflection_Object_as_typed_root(const void* buffer);
	static inline flatbuffers_string_t reflection_Object_name_get(reflection_Object_table_t t);
	static inline flatbuffers_string_t reflection_Object_name(reflection_Object_table_t t);
	static inline int reflection_Object_name_is_present(reflection_Object_table_t t);
	static inline size_t reflection_Object_vec_scan_by_name(reflection_Object_vec_t vec, const char* s);
	static inline size_t reflection_Object_vec_scan_n_by_name(reflection_Object_vec_t vec, const char* s, int n);
	...


Examples are provided in following script using the reflection and monster schema:

    scripts/reflection-doc-example.sh
    scripts/monster-doc-example.sh

The monster doc example essentially calls:

	scripts/flatcc-doc.sh samples/monster/monster.fbs MyGame_Sample_Monster_

resulting in the file `MyGame_Sample_Monster_.doc`:

	static inline size_t MyGame_Sample_Monster_vec_len(MyGame_Sample_Monster_vec_t vec);
	static inline MyGame_Sample_Monster_table_t MyGame_Sample_Monster_vec_at(MyGame_Sample_Monster_vec_t vec, size_t i);
	static inline MyGame_Sample_Monster_table_t MyGame_Sample_Monster_as_root_with_identifier(const void* buffer, const char* fid);
	static inline MyGame_Sample_Monster_table_t MyGame_Sample_Monster_as_root_with_type_hash(const void* buffer, flatbuffers_thash_t thash);
	static inline MyGame_Sample_Monster_table_t MyGame_Sample_Monster_as_root(const void* buffer);
	static inline MyGame_Sample_Monster_table_t MyGame_Sample_Monster_as_typed_root(const void* buffer);
	static inline MyGame_Sample_Vec3_struct_t MyGame_Sample_Monster_pos_get(MyGame_Sample_Monster_table_t t);
	static inline MyGame_Sample_Vec3_struct_t MyGame_Sample_Monster_pos(MyGame_Sample_Monster_table_t t);
	static inline int MyGame_Sample_Monster_pos_is_present(MyGame_Sample_Monster_table_t t);
	static inline int16_t MyGame_Sample_Monster_mana_get(MyGame_Sample_Monster_table_t t);
	static inline int16_t MyGame_Sample_Monster_mana(MyGame_Sample_Monster_table_t t);
	static inline const int16_t* MyGame_Sample_Monster_mana_get_ptr(MyGame_Sample_Monster_table_t t);
	static inline int MyGame_Sample_Monster_mana_is_present(MyGame_Sample_Monster_table_t t);
	static inline size_t MyGame_Sample_Monster_vec_scan_by_mana(MyGame_Sample_Monster_vec_t vec, int16_t key);
	static inline size_t MyGame_Sample_Monster_vec_scan_ex_by_mana(MyGame_Sample_Monster_vec_t vec, size_t begin, size_t end, int16_t key);
	...


FlatBuffer native types can also be extracted, for example string operations:

	scripts/flatcc-doc.sh samples/monster/monster.fbs flatbuffers_string_

resulting in `flatbuffers_string_.doc`:

	static inline size_t flatbuffers_string_len(flatbuffers_string_t s);
	static inline size_t flatbuffers_string_vec_len(flatbuffers_string_vec_t vec);
	static inline flatbuffers_string_t flatbuffers_string_vec_at(flatbuffers_string_vec_t vec, size_t i);
	static inline flatbuffers_string_t flatbuffers_string_cast_from_generic(const flatbuffers_generic_t p);
	static inline flatbuffers_string_t flatbuffers_string_cast_from_union(const flatbuffers_union_t u);
	static inline size_t flatbuffers_string_vec_find(flatbuffers_string_vec_t vec, const char* s);
	static inline size_t flatbuffers_string_vec_find_n(flatbuffers_string_vec_t vec, const char* s, size_t n);
	static inline size_t flatbuffers_string_vec_scan(flatbuffers_string_vec_t vec, const char* s);
	static inline size_t flatbuffers_string_vec_scan_n(flatbuffers_string_vec_t vec, const char* s, size_t n);
	static inline size_t flatbuffers_string_vec_scan_ex(flatbuffers_string_vec_t vec, size_t begin, size_t end, const char* s);
	...

## Using flatcc

Refer to `flatcc -h` for details.

An online version listed here: [flatcc-help.md] but please use `flatcc
-h` for an up to date reference.


The compiler can either generate a single header file or headers for all
included schema and a common file and with or without support for both
reading (default) and writing (-w) flatbuffers. The simplest option is
to use (-a) for all and include the `myschema_builder.h` file.

The (-a) or (-v) also generates a verifier file.

Make sure `flatcc` under the `include` folder is visible in the C
compilers include path when compiling flatbuffer builders.

The `flatcc` (-I) include path will assume all schema files with same
base name (case insentive) are identical and will only include the
first. All generated files use the input basename and will land in
working directory or the path set by (-o).

Files can be generated to stdout using (--stdout). C headers will be
ordered and concatenated, but are otherwise identical to the separate
file output. Each include statement is guarded so this will not lead to
missing include files.

The generated code, especially with all combined with --stdout, may
appear large, but only the parts actually used will take space up the
the final executable or object file. Modern compilers inline and include
only necessary parts of the statically linked builder library.

JSON printer and parser can be generated using the --json flag or
--json-printer or json-parser if only one of them is required. There are
some certain runtime library compile time flags that can optimize out
printing symbolic enums, but these can also be disabled at runtime.

## Trouble Shooting

Make sure to link with `libflatccrt` (rt for runtime) and not `libflatcc` (the schema compiler), otherwise the builder will not be available. Also make sure to have the 'include' of the flatcc project root in the include path.

Flatcc will by default expect a `file_identifier` in the buffer when reading or
verifying a buffer.

A buffer can have an unexpected 4-byte identifier at offset 4, or the identifier
might be absent.

Not all language interfaces support file identifiers in buffers, and if they do, they might not do so in an older version. Users have reported problems with both Python and Lua interfaces but this is easily resolved.

Check the return value of the verifier:

    int ret;
    char *s;

    ret = MyTable_verify_as_root(buf, size);
    if (ret) {
        s = flatcc_verify_error_string(ret);
        printf("buffer failed: %s\n", s);
    }

To verify a buffer with no identifier, or to ignore a different identifier,
use the `_with_identifier` version of the verifier with a null identifier:

    char *identifier = 0;

    MyTable_verify_as_root_with_identifier(buf, size, identifier);

To read a buffer use:

    MyTable_as_root_with_identifier(buf, 0);

And to build a buffer without an identifier use:

    MyTable_start_as_root_with_identifier(builder, 0);
    ...
    MyTable_end_as_root_with_identifier(builder, 0);

Several other `as_root` calls have an `as_root_with_identifier` version,
including JSON printing.

## Quickstart

After [building](https://github.com/dvidelabs/flatcc#building) the `flatcc tool`,
binaries are located in the `bin` and `lib` directories under the
`flatcc` source tree.

You can either jump directly to the [monster
example](https://github.com/dvidelabs/flatcc/tree/master/samples/monster)
that follows
[Googles FlatBuffers Tutorial](https://google.github.io/flatbuffers/flatbuffers_guide_tutorial.html), or you can read along the quickstart guide below. If you follow
the monster tutorial, you may want to clone and build flatcc and copy
the source to a separate project directory as follows:

    git clone https://github.com/dvidelabs/flatcc.git
    flatcc/scripts/setup.sh -a mymonster
    cd mymonster
    scripts/build.sh
    build/mymonster

`scripts/setup.sh` will as a minimum link the library and tool into a
custom directory, here `mymonster`. With (-a) it also adds a simple
build script, copies the example, and updates `.gitignore` - see
`scripts/setup.sh -h`. Setup can also build flatcc, but you still have
to ensure the build environment is configured for your system.

To write your own schema files please follow the main FlatBuffers
project documentation on [writing schema
files](https://google.github.io/flatbuffers/flatbuffers_guide_writing_schema.html).

The [Builder Interface Reference] may be useful after studying the
monster sample and quickstart below.

When looking for advanced examples such as sorting vectors and finding
elements by a key, you should find these in the
[`test/monster_test`](https://github.com/dvidelabs/flatcc/tree/master/test/monster_test) project.

The following quickstart guide is a broad simplification of the
`test/monster_test` project - note that the schema is slightly different
from the tutorial. Focus is on the C specific framework rather
than general FlatBuffers concepts.

You can still use the setup tool to create an empty project and
follow along, but there are no assumptions about that in the text below.

### Reading a Buffer

Here we provide a quick example of read-only access to Monster flatbuffer -
it is an adapted extract of the [monster_test.c] file.

First we compile the schema read-only with common (-c) support header and we
add the recursion because [monster_test.fbs] includes other files.

    flatcc -cr --reader test/monster_test/monster_test.fbs

For simplicity we assume you build an example project in the project
root folder, but in praxis you would want to change some paths, for
example:

    mkdir -p build/example
    flatcc -cr --reader -o build/example test/monster_test/monster_test.fbs
    cd build/example

We get:

    flatbuffers_common_reader.h
    include_test1_reader.h
    include_test2_reader.h
    monster_test_reader.h

(There is also the simpler `samples/monster/monster.fbs` but then you won't get
included schema files).

Namespaces can be long so we optionally use a macro to manage this.

    #include "monster_test_reader.h"

    #undef ns
    #define ns(x) FLATBUFFERS_WRAP_NAMESPACE(MyGame_Example, x)

    int verify_monster(void *buffer)
    {
        ns(Monster_table_t) monster;
        /* This is a read-only reference to a flatbuffer encoded struct. */
        ns(Vec3_struct_t) vec;
        flatbuffers_string_t name;
        size_t offset;

        if (!(monster = ns(Monster_as_root(buffer)))) {
            printf("Monster not available\n");
            return -1;
        }
        if (ns(Monster_hp(monster)) != 80) {
            printf("Health points are not as expected\n");
            return -1;
        }
        if (!(vec = ns(Monster_pos(monster)))) {
            printf("Position is absent\n");
            return -1;
        }

        /* -3.2f is actually -3.20000005 and not -3.2 due to representation loss. */
        if (ns(Vec3_z(vec)) != -3.2f) {
            printf("Position failing on z coordinate\n");
            return -1;
        }

        /* Verify force_align relative to buffer start. */
        offset = (char *)vec - (char *)buffer;
        if (offset & 15) {
            printf("Force align of Vec3 struct not correct\n");
            return -1;
        }

        /*
         * If we retrieved the buffer using `flatcc_builder_finalize_aligned_buffer` or
         * `flatcc_builder_get_direct_buffer` the struct should also
         * be aligned without subtracting the buffer.
         */
        if (vec & 15) {
            printf("warning: buffer not aligned in memory\n");
        }

        /* ... */
        return 0;
    }
    /* main() {...} */


### Compiling for Read-Only

Assuming our above file is `monster_example.c` the following are a few
ways to compile the project for read-only - compilation with runtime
library is shown later on.

    cc -I include monster_example.c -o monster_example

    cc -std=c11 -I include monster_example.c -o monster_example

    cc -D FLATCC_PORTABLE -I include monster_example.c -o monster_example

The include path or source path is likely different. Some files in
`include/flatcc/portable` are always used, but the `-D FLATCC_PORTABLE`
flag includes additional files to support compilers lacking c11
features.

NOTE: on some clang/gcc platforms it may be necessary to use -std=gnu99 or
-std=gnu11 if the linker is unable find `posix_memalign`, see also comments in
[paligned_alloc.h].


### Building a Buffer

Here we provide a very limited example of how to build a buffer - only a few
fields are updated. Pleaser refer to [monster_test.c] and the doc directory
for more information.

First we must generate the files:

    flatcc -a monster_test.fbs

This produces:

    flatbuffers_common_reader.h
    flatbuffers_common_builder.h
    include_test1_reader.h
    include_test1_builder.h
    include_test1_verifier.h
    include_test2_reader.h
    include_test2_builder.h
    include_test2_verifier.h
    monster_test_reader.h
    monster_test_builder.h
    monster_test_verifier.h

Note: we wouldn't actually do the readonly generation shown earlier
unless we only intend to read buffers - the builder generation always
generates read acces too.

By including `"monster_test_builder.h"` all other files are included
automatically. The C compiler needs the `-I include` directive to access
`flatcc/flatcc_builder.h`, `flatcc/flatcc_verifier.h`, and other files
depending on specifics, assuming the project root is the current
directory.

The verifiers are not required and just created because we lazily chose
the -a option.

The builder must be initialized first to set up the runtime environment
we need for building buffers efficiently - the builder depends on an
emitter object to construct the actual buffer - here we implicitly use
the default. Once we have that, we can just consider the builder a
handle and focus on the FlatBuffers generated API until we finalize the
buffer (i.e. access the result).  For non-trivial uses it is recommended
to provide a custom emitter and for example emit pages over the network
as soon as they complete rather than merging all pages into a single
buffer using `flatcc_builder_finalize_buffer`, or the simplistic
`flatcc_builder_get_direct_buffer` which returns null if the buffer is
too large. See also documentation comments in [flatcc_builder.h] and
[flatcc_emitter.h]. See also `flatc_builder_finalize_aligned_buffer` in
`builder.h` and the [Builder Interface Reference] when malloc aligned
buffers are insufficent.


    #include "monster_test_builder.h"

    /* See [monster_test.c] for more advanced examples. */
    void build_monster(flatcc_builder_t *B)
    {
        ns(Vec3_t *vec);

        /* Here we use a table, but structs can also be roots. */
        ns(Monster_start_as_root(B));

        ns(Monster_hp_add(B, 80));
        /* The vec struct is zero-initalized. */
        vec = ns(Monster_pos_start(B));
        /* Native endian. */
        vec->x = 1, vec->y = 2, vec->z = -3.2f;
        /* _end call converts to protocol endian format - for LE it is a nop. */
        ns(Monster_pos_end(B));

        /* Name is required, or we get an assertion in debug builds. */
        ns(Monster_name_create_str(B, "MyMonster"));

        ns(Monster_end_as_root(B));
    }

    #include "flatcc/support/hexdump.h"

    int main(int argc, char *argv[])
    {
        flatcc_builder_t builder;
        void *buffer;
        size_t size;

        flatcc_builder_init(&builder);

        build_monster(&builder);
        /* We could also use `flatcc_builder_finalize_buffer` and free the buffer later. */
        buffer = flatcc_builder_get_direct_buffer(&builder, &size);
        assert(buffer);
        verify_monster(buffer);

        /* Visualize what we got ... */
        hexdump("monster example", buffer, size, stdout);

        /*
         * Here we can call `flatcc_builder_reset(&builder) if
         * we wish to build more buffers before deallocating
         * internal memory with `flatcc_builder_clear`.
         */

        flatcc_builder_clear(&builder);
        return 0;
    }

Compile the example project:

    cc -std=c11 -I include monster_example.c lib/libflatccrt.a -o monster_example

Note that the runtime library is required for building buffers, but not
for reading them. If it is incovenient to distribute the runtime library
for a given target, source files may be used instead. Each feature has
its own source file, so not all runtime files are needed for building a
buffer:

    cc -std=c11 -I include monster_example.c \
        src/runtime/emitter.c src/runtime/builder.c \
        -o monster_example

Other features such as the verifier and the JSON printer and parser
would each need a different file in src/runtime. Which file should be
obvious from the filenames except that JSON parsing also requires the
builder and emitter source files.


### Verifying a Buffer

A buffer can be verified to ensure it does not contain any ranges that
point outside the the given buffer size, that all data structures are
aligned according to the flatbuffer principles, that strings are zero
terminated, and that required fields are present.

In the builder example above, we can apply a verifier to the output:

    #include "monster_test_builder.h"
    #include "monster_test_verifier.h"
    int ret;
    ...
    ... finalize
    if ((ret = ns(Monster_verify_as_root_with_identifier(buffer, size,
            "MONS")))) {
        printf("Monster buffer is invalid: %s\n",
        flatcc_verify_error_string(ret));
    }

The [readfile.h] utility may also be helpful in reading an existing
buffer for verification.

Flatbuffers can optionally leave out the identifier, here "MONS". Use a
null pointer as identifier argument to ignore any existing identifiers
and allow for missing identifiers.

Nested flatbuffers are always verified with a null identifier, but it
may be checked later when accessing the buffer.

The verifier does NOT verify that two datastructures are not
overlapping. Sometimes this is indeed valid, such as a DAG (directed
acyclic graph) where for example two string references refer to the same
string in the buffer. In other cases an attacker may maliciously
construct overlapping datastructures such that in-place updates may
cause subsequent invalid buffers. Therefore an untrusted buffer should
never be updated in-place without first rewriting it to a new buffer.

The CMake build system has build option to enable assertions in the
verifier. This will break debug builds and not usually what is desired,
but it can be very useful when debugging why a buffer is invalid. Traces
can also be enabled so table offset and field id can be reported.

See also `include/flatcc/flatcc_verifier.h`.

When verifying buffers returned directly from the builder, it may be
necessary to use the `flatcc_builder_finalize_aligned_buffer` to ensure
proper alignment and use `aligned_free` to free the buffer (or as of
v0.5.0 also `flatcc_builder_aligned_free`), see also the
[Builder Interface Reference]. Buffers may also be copied into aligned
memory via mmap or using the portable layers `paligned_alloc.h` feature
which is available when including generated headers.
`test/flatc_compat/flatc_compat.c` is an example of how this can be
done. For the majority of use cases, standard allocation would be
sufficient, but for example standard 32-bit Windows only allocates on an
8-byte boundary and can break the monster schema because it has 16-byte
aligned fields.

NOTE: as of August 2024 it has been discovered that C++ writer code has been
aligning empty vectors to the size field only, even if elements require
greater alignment like the double type which requires 8. This would cause the
FlatCC verifier to (correctly) reject these vectors because it would result
in an invalid C pointer type on some architectures. However, because this has
been in effect for over 10 years, the consensus is to have verifiers tolerate
this behaviour even if C++ will eventually fix this issue. The FlatCC
verifier has been updated to accept such buffers by default with an optional
compile time flag to enforce the strict behaviour as well
(`FLATCC_ENFORCE_ALIGNED_EMPTY_VECTORS`). In principle the misaligned vectors
can potentially lead to undefined behaviour in agressively optimized C
compilers. As of now it appears to be safe to read such buffers on common
platforms and it is preferable to avoid additional runtime reader overhead to
deal with this. For more, see [FlatCC #287], [Google Flatbuffers #8374],
[FlatCC #289].


### Potential Name Conflicts

If unfortunate, it is possible to have a read accessor method conflict
with other generated methods and typenames. Usually a small change in
the schema will resolve this issue.

As of flatcc 0.6.2 conservative warnings will detect most likely
potential conflicts in enum member names and table field names. The
`-s` option silences warnings.

As of flatcc 0.5.2 read accors are generated with and without a `_get`
suffix so it is also possible to use `Monster_pos_get(monster)` instead
of `Monster_pos(monster)`. When calling flatcc with option `-g` the
read accesors will only be generated with `_get` suffix. This avoids
potential name conflicts. An example of a conflict is a field name
like `pos_add` when there is also a `pos` field because the builder
interface generates the `add` suffix. Using the -g option avoids this
problem, but it is preferable to choose another name such as `added_pos`
when the schema can be modified.

The `-g` option only changes the content of the
`flatbuffers_common_reader.h` file, so it is technically  possible to
use different versions of this file if they are not mixed.

If an external code generator depends on flatcc output, it should use
the `_get` suffix because it will work with and without the -g option,
but only as of version 0.5.2 or later. For human readable code it is
probaly simpler to stick to the orignal naming convention without the
`_get` suffix.

Even with the above, it is still possible to have a conflict with the
union type field. If a union field is named `foo`, an additional field
is automatically - this field is named `foo_type` and holds,
unsurprisingly, the type of the union.

Namespaces can also cause conflicts. If a schema has the namespace
Foo.Bar and table named MyTable with a field name hello, then a
read accessor will be named: `Foo_Bar_MyTable_hello_get`. It
is also possible to have a table named `Bar_MyTable` because `_` are
allowed in FlatBuffers schema names, but in this case we have name
conflict in the generated the C code. FlatCC does not attempt to avoid
such conflicts so such schema are considered invalid.

Notably several users have experienced conflicts with a table or struct
field named 'identifier' because `<table-name>_identifier` has been
defined to be the file identifier to be used when creating a buffer with
that table (or struct) as root. As of 0.6.1, the name is
`<table-name>_file_identifier` to reduce the risk of conflicts. The old
form is deprecated but still generated for tables without a field named
'identifier' for backwards compatibility. Mostly this macro is used for
higher level functions such as `mytable_create_as_root` which need to
know what identifier to use.


### Debugging a Buffer

When reading a FlatBuffer does not provide the expected results, the
first line of defense is to ensure that the code being tested is linked
against `flatccrt_d`, the debug build of the runtime library. This will
raise an assertion if calls to the builder are not properly balanced or
if required fields are not being set.

To dig further into a buffer, call the buffer verifier and see if the
buffer is actually valid with respect to the expected buffer type.

Strings and tables will be returned as null pointers when their
corresponding field is not set in the buffer. User code should test for
this but it might also be helpful to temporarily or permanently set the
`required` attribute in the schema. The builder will then detect missing fields
when cerating buffers and the verifier can will detect their absence in
an existing buffer.

If the verifier rejects a buffer, the error can be printed (see
[Verifying a Buffer](#verifying-a-buffer)), but it will not say exactly
where the problem was found. To go further, the verifier can be made to
assert where the problem is encountered so the buffer content can be
analyzed. This is enabled with:

    -DFLATCC_DEBUG_VERIFY=1

Note that this will break test cases where a buffer is expected to fail
verification.

To dump detailed contents of a valid buffer, or the valid contents up to
the point of failure, use:

    -DFLATCC_TRACE_VERIFY=1

Both of these options can be set as CMake options, or in the
[flatcc_rtconfig.h] file.

When reporting bugs, output from the above might also prove helpful.

The JSON parser and printer can also be used to create and display
buffers. The parser will use the builder API correctly or issue a syntax
error or an error on required field missing. This can rule out some
uncertainty about using the api correctly. The [test_json.c] file and
[test_json_parser.c] have
test functions that can be adapted for custom tests.

For advanced debugging the [hexdump.h] file can be used to dump the buffer
contents. It is used in [test_json.c] and also in [monster_test.c].
See also [FlatBuffers Binary Format].

As of April 2022, Googles flatc tool has implemented an `--annotate` feature.
This provides an annotated hex dump given a binary buffer and a schema. The
output can be used to troubleshoot and rule out or confirm suspected encoding
bugs in the buffer at hand. The eclectic example in the [FlatBuffers Binary
Format] document contains a hand written annotated example which inspired the
`--annotate` feature, but it is not the exact same output format. Note also that
`flatc` generated buffers tend to have vtables before the table it is referenced
by, while flatcc normally packs all vtables at the end of the buffer for
better padding and cache efficiency.

See also [flatc --annotate].

Note: There is experimental support for text editor that supports
clangd language server or similar. You can edit `CMakeList.txt`
to generate `build/Debug/compile_comands.json`, at least when
using clang as a compiler, and copy or symlink it from root. Or
come with a better suggestion. There are `.gitignore` entries for
`compile_flags.txt` and `compile_commands.json` in project root.


## File and Type Identifiers

There are two ways to identify the content of a FlatBuffer. The first is
to use file identifiers which are defined in the schema. The second is
to use `type identifiers` which are calculated hashes based on each
tables name prefixed with its namespace, if any. In either case the
identifier is stored at offset 4 in binary FlatBuffers, when present.
Type identifiers are not to be confused with union types.

### File Identifiers

The FlatBuffers schema language has the optional `file_identifier`
declaration which accepts a 4 characer ASCII string. It is intended to be
human readable. When absent, the buffer potentially becomes 4 bytes
shorter (depending on padding).

The `file_identifier` is intended to match the `root_type` schema
declaration, but this does not take into account that it is convenient
to create FlatBuffers for other types as well. `flatcc` makes no special
destinction for the `root_type` while Googles `flatc` JSON parser uses
it to determine the JSON root object type.

As a consequence, the file identifier is ambigous. Included schema may
have separate `file_identifier` declarations. To at least make sure each
type is associated with its own schemas `file_identifier`, a symbol is
defined for each type. If the schema has such identifier, it will be
defined as the null identifier.

The generated code defines the identifiers for a given table:

    #ifndef MyGame_Example_Monster_file_identifier
    #define MyGame_Example_Monster_file_identifier "MONS"
    #endif

The user can now override the identifier for a given type, for example:

    #define MyGame_Example_Vec3_file_identifier "VEC3"
    #include "monster_test_builder.h"

    ...
    MyGame_Example_Vec3_create_as_root(B, ...);

The `create_as_root` method uses the identifier for the type in question,
and so does other `_as_root` methods.

The `file_extension` is handled in a similar manner:

    #ifndef MyGame_Example_Monster_file_extension
    #define MyGame_Example_Monster_file_extension "mon"
    #endif

### Type Identifiers

To better deal with the ambigouties of file identifiers, type
identifiers have been introduced as an alternative 4 byte buffer
identifier. The hash is standardized on FNV-1a for interoperability.

The type identifier use a type hash which maps a fully qualified type
name into a 4 byte hash. The type hash is a 32-bit native value and the
type identifier is a 4 character little endian encoded string of the
same value.

In this example the type hash is derived from the string
"MyGame.Example.Monster" and is the same for all FlatBuffer code
generators that supports type hashes.

The value 0 is used to indicate that one does not care about the
identifier in the buffer.

    ...
    MyGame_Example_Monster_create_as_typed_root(B, ...);
    buffer = flatcc_builder_get_direct_buffer(B);
    MyGame_Example_Monster_verify_as_typed_root(buffer, size);
    // read back
    monster = MyGame_Example_Monster_as_typed_root(buffer);

    switch (flatbuffers_get_type_hash(buffer)) {
    case MyGame_Example_Monster_type_hash:
        ...

    }
    ...
    if (flatbuffers_get_type_hash(buffer) ==
        flatbuffers_type_hash_from_name("Some.Old.Buffer")) {
        printf("Buffer is the old version, not supported.\n");
    }

More API calls are available to naturally extend the existing API. See
[monster_test.c] for more.

The type identifiers are defined like:

    #define MyGame_Example_Monster_type_hash ((flatbuffers_thash_t)0x330ef481)
    #define MyGame_Example_Monster_type_identifier "\x81\xf4\x0e\x33"

The `type_identifier` can be used anywhere the original 4 character
file identifier would be used, but a buffer must choose which system, if any,
to use. This will not affect the `file_extension`.

NOTE: The generated `_type_identifier` strings should not normally be
used when an identifier string is expected in the generated API because
it may contain null bytes which will be zero padded after the first null
before comparison. Use the API calls that take a type hash instead. The
`type_identifier` can be used in low level [flatcc_builder.h] calls
because it handles identifiers as a fixed byte array and handles type
hashes and strings the same.

NOTE: it is possible to compile the flatcc runtime to encode buffers in
big endian format rather than the standard little endian format
regardless of the host platforms endianness. If this is done, the
identifier field in the buffer is always byte swapped regardless of the
identifier method chosen. The API calls make this transparent, so "MONS"
will be stored as "SNOM" but should still be verified as "MONS" in API
calls. This safeguards against mixing little- and big-endian buffers.
Likewise, type hashes are always tested in native (host) endian format.


The
[`flatcc/flatcc_identifier.h`](https://github.com/dvidelabs/flatcc/blob/master/include/flatcc/flatcc_identifier.h)
file contains an implementation of the FNV-1a hash used. The hash was
chosen for simplicity, availability, and collision resistance. For
better distribution, and for internal use only, a dispersion function is
also provided, mostly to discourage use of alternative hashes in
transmission since the type hash is normally good enough as is.

_Note: there is a potential for collisions in the type hash values
because the hash is only 4 bytes._


## JSON Parsing and Printing

JSON support files are generated with `flatcc --json`.

This section is not a tutorial on JSON printing and parsing, it merely
covers some non-obvious aspects. The best source to get started quickly
is the test file:

    test/json_test/json_test.c

For detailed usage, please refer to:

    test/json_test/test_json_printer.c
    test/json_test/test_json_parser.c
    test/json_test/json_test.c
    test/benchmark/benchflatccjson

See also JSON parsing section in the Googles FlatBuffers [schema
documentation](https://google.github.io/flatbuffers/flatbuffers_guide_writing_schema.html).

By using the flatbuffer schema it is possible to generate schema
specific JSON printers and parsers. This differs for better and worse
from Googles `flatc` tool which takes a binary schema as input and
processes JSON input and output. Here that parser and printer only rely
on the `flatcc` runtime library, is faster (probably significantly so),
but requires recompilition when new JSON formats are to be supported -
this is not as bad as it sounds - it would for example not be difficult
to create a Docker container to process a specific schema in a web
server context.

The parser always takes a text buffer as input and produces output
according to how the builder object is initialized. The printer has
different init functions: one for printing to a file pointer, including
stdout, one for printing to a fixed length external buffer, and one for
printing to a dynamically growing buffer. The dynamic buffer may be
reused between prints via the reset function. See `flatcc_json_parser.h`
for details.

The parser will accept unquoted names (not strings) and trailing commas,
i.e. non-strict JSON and also allows for hex `\x03` in strings. Strict
mode must be enabled by a compile time flag. In addition the parser
schema specific symbolic enum values that can optionally be unquoted
where a numeric value is expected:

    color: Green
    color: Color.Green
    color: MyGame.Example.Color.Green
    color: 2

The symbolic values do not have to be quoted (unless required by runtime
or compile time configuration), but can be while numeric values cannot
be quoted. If no namespace is provided, like `color: Green`, the symbol
must match the receiving enum type. Any scalar value may receive a
symbolic value either in a relative namespace like `hp: Color.Green`, or
an absolute namespace like `hp: MyGame.Example.Color.Green`, but not
`hp: Green` (since `hp` in the monster example schema) is not an enum
type with a `Green` value). A namespace is relative to the namespace of
the receiving object.

It is also possible to have multiple values, but these always have to be
quoted in order to be compatible with Googles flatc tool for Flatbuffers
1.1:

    color: "Green Red"

_Unquoted multi-valued enums can be enabled at compile time but this is
deprecated because it is incompatible with both Googles flatc JSON and
also with other possible future extensions: `color: Green Red`_

These value-valued expressions were originally intended for enums that
have the bit flag attribute defined (which Color does have), but this is
tricky to process, so therefore any symblic value can be listed in a
sequence with or without namespace as appropriate. Because this further
causes problems with signed symbols the exact definition is that all
symbols are first coerced to the target type (or fail), then added to
the target type if not the first this results in:

    color: "Green Blue Red Blue"
    color: 19

Because Green is 2, Red is 1, Blue is 8 and repeated.

__NOTE__: Duplicate values should be considered implementation dependent
as it cannot be guaranteed that all flatbuffer JSON parsers will handle
this the same. It may also be that this implementation will change in
the future, for example to use bitwise or when all members and target
are of bit flag type.

It is not valid to specify an empty set like:

    color: ""

because it might be understood as 0 or the default value, and it does
not unquote very well.

The printer will by default print valid json without any spaces and
everything quoted. Use the non-strict formatting option (see headers and
test examples) to produce pretty printing. It is possibly to disable
symbolic enum values using the `noenum` option.

Only enums will print symbolic values are there is no history of any
parsed symbolic values at all. Furthermore, symbolic values are only
printed if the stored value maps cleanly to one value, or in the case of
bit-flags, cleanly to multiple values. For exmaple if parsing `color: Green Red`
it will print as `"color":"Red Green"` by default, while `color: Green
Blue Red Blue` will print as `color:19`.

Both printer and parser are limited to roughly 100 table nesting levels
and an additional 100 nested struct depths. This can be changed by
configuration flags but must fit in the runtime stack since the
operation is recursive descent. Exceedning the limits will result in an
error.

Numeric values are coerced to the receiving type. Integer types will
fail if the assignment does not fit the target while floating point
values may loose precision silently. Integer types never accepts
floating point values. Strings only accept strings.

Nested flatbuffers may either by arrays of byte sized integers, or a
table or a struct of the target type. See test cases for details.

The parser will by default fail on unknown fields, but these can also be
skipped silently with a runtime option.

Unions are difficult to parse. A union is two json fields: a table as
usual, and an enum to indicate the type which has the same name with a
`_type` suffix and accepts a numeric or symbolic type code:

    {
      name: "Container Monster",
      test_type: Monster,
      test: { name: "Contained Monster" }
    }

based on the schema is defined in [monster_test.fbs].

Because other json processors may sort fields, it is possible to receive
the type field after the test field. The parser does not store temporary
datastructures. It constructs a flatbuffer directly. This is not
possible when the type is late. This is handled by parsing the field as
a skipped field on a first pass, followed by a typed back-tracking
second pass once the type is known (only the table is parsed twice, but
for nested unions this can still expand). Needless to say this slows down
parsing. It is an error to provide only the table field or the type
field alone, except if the type is `NONE` or `0` in which case the table
is not allowed to be present.

Union vectors are supported as of v0.5.0. A union vector is represented
as two vectors, one with a vector of tables and one with a vector of
types, similar to ordinary unions. It is more efficient to place the
type vector first because it avoids backtracking. Because a union of
type NONE cannot be represented by absence of table field when dealing
with vectors of unions, a table must have the value `null` if its type
is NONE in the corresponding type vector. In other cases a table should
be absent, and not null.

Here is an example of JSON containing Monster root table with a union
vector field named `manyany` which is a vector of `Any` unions in the
[monster_test.fbs] schema:

    {
        "name": "Monster",
        "manyany_type": [ "Monster", "NONE" ],
        "manyany": [{"name": "Joe"}, null]
    }

### Base64 Encoding

As of v0.5.0 it is possible to encode and decode a vector of type
`[uint8]` (aka `[ubyte]`) as a base64 encoded string or a base64url
encoded string as documented in RFC 4648. Any other type, notably the
string type, do not handle base64 encoding.

Limiting the support to `[uint8]` avoids introducing binary data into
strings and also avoids dealing with sign and endian encoding of binary
data of other types. Furthermore, array encoding of values larger than 8
bits are not necessarily less efficient than base64.

Base64 padding is always printed and is optional when parsed. Spaces,
linebreaks, JSON string escape character '\\', or any other character
not in the base64(url) alphabet are rejected as a parse error.

The schema must add the attribute `(base64)` or `(base64url)` to the
field holding the vector, for example:

    table Monster {
        name: string;
        sprite: [uint8] (base64);
        token: [uint8] (base64url);
    }

If more complex data needs to be encoded as base64 such as vectors of
structs, this can be done via nested FlatBuffers which are also of type
`[uint8]`.

Note that for some use cases it might be desireable to read binary data as
base64 into memory aligned to more than 8 bits. This is not currently
possible, but it is recognized that a `(force_align: n)` attribute on
`[ubyte]` vectors could be useful, but it can also be handled via nested
flatbuffers which also align data.

### Parsing Fixed Length Arrays

Fixed length arrays introduced in 0.6.0 allow for structs containing arrays
of fixed length scalars, structs and chars. Arrays are parsed like vectors
for of similar type but are zero padded if shorter than expected and fails
if longer than expected. The flag `reject_array_underflow` will error if an
array is shorter than expected instead of zero padding. The flag
`skip_array_overflow` will allow overlong arrays and simply drop extra elements.

Char arrays are parsed like strings and zero padded if short than expected, but
they are not zero terminated. A string like "hello" will exactly fit into a
field of type `[char:5]`. Trailing zero characters are not printed, but embedded
zero characters are. This allows for loss-less roundtrips without having to zero
pad strings. Note that other arrays are always printed in full. If the flag
`skip_array_overflow` is set, a string might be truncated in the middle of a
multi-byte character. This is not checked nor enforced by the verifier.

### Runtime Flags

Both the printer and the parser have the ability to accept runtime flags that
modifies their behavior. Please refer to header file comments for documentation
and test cases for examples. Notably it is possible to print unquoted symbols
and to ignore unknown fields when parsing instead of generating an error.

Note that deprecated fields are considered unknown fields during parsing so in
order to process JSON from an old schema version with deprecated fields present,
unknown symbols must be skipped.

### Generic Parsing and Printing.

As of v0.5.1 [test_json.c] demonstrates how a single parser driver can be used
to parse different table types without changes to the driver or to the schema.

For example, the following layout can be used to configure a generic parser or printer.

	struct json_scope {
		const char *identifier;
		flatcc_json_parser_table_f *parser;
		flatcc_json_printer_table_f *printer;
		flatcc_table_verifier_f *verifier;
	};

	static const struct json_scope Monster = {
		/* The is the schema global file identifier. */
		ns(Monster_identifier),
		ns(Monster_parse_json_table),
		ns(Monster_print_json_table),
		ns(Monster_verify_table)
	};

The `Monster` scope can now be used by a driver or replaced with a new scope as needed:

	/* Abbreviated ... */
	struct json_scope = Monster;
    flatcc_json_parser_table_as_root(B, &parser_ctx, json, strlen(json), parse_flags,
            scope->identifier, scope->parser);
	/* Printing and verifying works roughly the same. */

The generated table `MyGame_Example_Monster_parse_json_as_root` is a thin
convenience wrapper roughly implementing the above.

The generated `monster_test_parse_json` is a higher level convenience wrapper named
of the schema file itself, not any specific table. It parses the `root_type` configured
in the schema. This is how the `test_json.c` test driver operated prior to v0.5.1 but
it made it hard to test parsing and printing distinct table types.

Note that verification is not really needed for JSON parsing because a
generated JSON parser is supposed to build buffers that always verify (except
for binary encoded nested buffers), but it is useful for testing.


### Performance Notes

Note that json parsing and printing is very fast reaching 500MB/s for
printing and about 300 MB/s for parsing. Floating point parsing can
significantly skew these numbers. The integer and floating point parsing
and printing are handled via support functions in the portable library.
In addition the floating point `include/flatcc/portable/grisu3_*` library
is used unless explicitly disable by a compile time flag. Disabling
`grisu3` will revert to `sprintf` and `strtod`. Grisu3 will fall back to
`strtod` and `grisu3` in some rare special cases. Due to the reliance on
`strtod` and because `strtod` cannot efficiently handle
non-zero-terminated buffers, it is recommended to zero terminate
buffers. Alternatively, grisu3 can be compiled with a flag that allows
errors in conversion. These errors are very small and still correct, but
may break some checksums. Allowing for these errors can significantly
improve parsing speed and moves the benchmark from below half a million
parses to above half a million parses per second on 700 byte json
string, on a 2.2 GHz core-i7.

While unquoted strings may sound more efficient due to the compact size,
it is actually slower to process. Furthermore, large flatbuffer
generated JSON files may compress by a factor 8 using gzip or a factor
4 using LZ4 so this is probably the better place to optimize. For small
buffers it may be more efficient to compress flatbuffer binaries, but
for large files, json may actually compress significantly better due to
the absence of pointers in the format.

SSE 4.2 has been experimentally added, but it the gains are limited
because it works best when parsing space, and the space parsing is
already fast without SSE 4.2 and because one might just leave out the
spaces if in a hurry. For parsing strings, trivial use of SSE 4.2 string
scanning doesn't work well becasuse all the escape codes below ASCII 32
must be detected rather than just searching for `\` and `"`. That is not
to say there are not gains, they just don't seem worthwhile.

The parser is heavily optimized for 64-bit because it implements an
8-byte wide trie directly in code. It might work well for 32-bit
compilers too, but this hasn't been tested. The large trie does put some
strain on compile time. Optimizing beyond -O2 leads to too large
binaries which offsets any speed gains.


## Global Scope and Included Schema

Attributes included in the schema are viewed in a global namespace and
each include file adds to this namespace so a schema file can use
included attributes without namespace prefixes.

Each included schema will also add types to a global scope until it sees
a `namespace` declaration. An included schema does not inherit the
namespace of an including file or an earlier included file, so all
schema files starts in the global scope. An included file can, however,
see other types previously defined in the global scope. Because include
statements always appear first in a schema, this can only be earlier
included files, not types from a containing schema.

The generated output for any included schema is indendent of how it was
included, but it might not compile without the earlier included files
being present and included first. By including the toplevel `myschema.h`
or `myschema_builder.h` all these dependencies are handled correctly.

Note: `libflatcc.a` can only parse a single schema when the schema is
given as a memory buffer, but can handle the above when given a
filename. It is possible to concatenate schema files, but a `namespace;`
declaration must be inserted as a separator to revert to global
namespace at the start of each included file. This can lead to subtle
errors because if one parent schema includes two child schema `a.fbs`
and `b.fbs`, then `b.fbs` should not be able to see anything in `a.fbs`
even if they share namespaces. This would rarely be a problem in praxis,
but it means that schema compilation from memory buffers cannot
authoratively validate a schema. The reason the schema must be isolated
is that otherwise code generation for a given schema could change with
how it is being used leading to very strange errors in user code.


## Required Fields and Duplicate Fields

If a field is required such as Monster.name, the table end call will
assert in debug mode and create incorrect tables in non-debug builds.
The assertion may not be easy to decipher as it happens in library code
and it will not tell which field is missing.

When reading the name, debug mode will again assert and non-debug builds
will return a default value.

Writing the same field twice will also trigger an assertion in debug
builds.


## Fast Buffers

Buffers can be used for high speed communication by using the ability to
create buffers with structs as root. In addition the default emitter
supports `flatcc_emitter_direct_buffer` for small buffers so no extra copy
step is required to get a linear buffer in memory. Preliminary
measurements suggests there is a limit to how fast this can go (about
6-7 mill. buffers/sec) because the builder object must be reset between
buffers which involves zeroing allocated buffers. Small tables with a
simple vector achieve roughly half that speed. For really high speed a
dedicated builder for structs would be needed. See also
[monster_test.c].


## Types

All types stored in a buffer has a type suffix such as `Monster_table_t`
or `Vec3_struct_t` (and namespace prefix which we leave out here). These
types are read-only pointers into endian encoded data. Enum types are
just constants easily grasped from the generated code. Tables are dense so
they are never accessed directly.

Enums support schema evolution meaning that more names can be added to
the enumeration in a future schema version. As of v0.5.0 the function
`_is_known_value` can be used ot check if an enum value is known to the
current schema version.

Structs have a dual purpose because they are also valid types in native
format, yet the native reprsention has a slightly different purpose.
Thus the convention is that a const pointer to a struct encoded in a
flatbuffer has the type `Vec3_struct_t` where as a writeable pointer to
a native struct has the type `Vec3_t *` or `struct Vec3 *`.

All types have a `_vec_t` suffix which is a const pointer to the
underlying type. For example `Monster_table_t` has the vector type
`Monster_vec_t`. There is also a non-const variant with suffix
`_mutable_vec_t` which is rarely used. However, it is possible to sort
vectors in-place in a buffer, and for this to work, the vector must be
cast to mutable first. A vector (or string) type points to the element
with index 0 in the buffer, just after the length field, and it may be
cast to a native type for direct access with attention to endian
encoding. (Note that `table_t` types do point to the header field unlike
vectors.) These types are all for the reader interface. Corresponding
types with a `_ref_t` suffix such as `_vec_ref_t` are used during
the construction of buffers.

Native scalar types are mapped from the FlatBuffers schema type names
such as ubyte to `uint8_t` and so forth. These types also have vector
types provided in the common namespace (default `flatbuffers_`) so
a `[ubyte]` vector has type `flatbuffers_uint8_vec_t` which is defined
as `const uint8_t *`.

The FlatBuffers boolean type is strictly 8 bits wide so we cannot use or
emulate `<stdbool.h>` where `sizeof(bool)` is implementation dependent.
Therefore `flatbuffers_bool_t` is defined as `uint8_t` and used to
represent FlatBuffers boolean values and the constants of same type:
`flatbuffers_true = 1` and `flatbuffers_false = 0`. Even so,
`pstdbool.h` is available in the `include/flatcc/portable` directory if
`bool`, `true`, and `false` are desired in user code and `<stdbool.h>`
is unavailable.

`flatbuffers_string_t` is `const char *` but imply the returned pointer
has a length prefix just before the pointer. `flatbuffers_string_vec_t`
is a vector of strings. The `flatbufers_string_t` type guarantees that a
length field is present using `flatbuffers_string_len(s)` and that the
string is zero terminated. It also suggests that it is in utf-8 format
according to the FlatBuffers specification, but not checks are done and
the `flatbuffers_create_string(B, s, n)` call explicitly allows for
storing embedded null characters and other binary data.

All vector types have operations defined as the typename with `_vec_t`
replaced by `_vec_at` and `_vec_len`. For example
`flatbuffers_uint8_vec_at(inv, 1)` or `Monster_vec_len(inv)`. The length
or `_vec_len` will be 0 if the vector is missing whereas `_vec_at` will
assert in debug or behave undefined in release builds following out of
bounds access. This also applies to related string operations.

The FlatBuffers schema uses the following scalar types: `ubyte`, `byte`,
`ushort`, `short, uint`, `int`, `ulong`, and `long` to represent
unsigned and signed integer types of length 8, 16, 32, and 64
respectively. The schema syntax has been updated to also support the
type aliases `uint8`, `int8`, `uint16`, `int16`, `uint32`, `int32`,
`uint64`, `int64` to represent the same basic types. Likewise, the
schema uses the types `float` and `double` to represent IEEE-754
binary32 and binary64 floating point formats where the updated syntax
also supports the type aliases `float32` and `float64`.

The C interface uses the standard C types such as uint8 and double to
represent scalar types and this is unaffected by the schema type name
used, so the schema vector type `[float64]` is represented as
`flatbuffers_double_vec_t` the same as `[double]` would be.

Note that the C standard does not guarantee that the C types `float` and
`double` are represented by the IEEE-754 binary32 single precision
format and the binary64 double precision format respectively, although
they usually are. If this is not the case FlatCC cannot work correctly
with FlatBuffers floating point values. (If someone really has this
problem, it would be possible to fix).

Unions are represented with a two table fields, one with a table field
and one with a type field. See separate section on Unions. As of flatcc
v0.5.0 union vectors are also supported.

## Unions

A union represents one of several possible tables. A table with a union
field such as `Monster.equipped` in the samples schema will have two
accessors: `MyGame_Sample_Monster_equipped(t)` of type
`flatbuffers_generic_t` and `MyGame_Sample_Monster_equipped_type(t)` of
type `MyGame_Sample_Equipment_union_type_t`. A generic type is is just a
const void pointer that can be assigned to the expected table type,
struct type, or string type. The enumeration has a type code for member
of the union and also `MyGame_Sample_Equipment_NONE` which has the value
0.

The union interface were changed in 0.5.0 and 0.5.1 to use a consistent
{ type, value } naming convention for both unions and union vectors
in all interfaces and to support unions and union vectors of multiple
types.

A union can be accessed by its field name, like Monster
`MyGame_Sample_Monster_equipped(t)` and its type is given by
`MyGame_Sample_Monster_type(t)`, or a `flatbuffers_union_t` struct
can be returned with `MyGame_Sample_monster_union(t)` with the fields
{ type, value }. A union vector is accessed in the same way but {
type, value } represents a type vector and a vector of the given type,
e.g. a vector Monster tables or a vector of strings.

There is a test in [monster_test.c] covering union vectors and a
separate test focusing on mixed type unions that also has union vectors.


### Union Scope Resolution

Googles `monster_test.fbs` schema has the union (details left out):

	namespace MyGame.Example2;
	table Monster{}

	namespace MyGame.Example;
	table Monster{}

	union Any { Monster, MyGame.Example2.Monster }

where the two Monster tables are defined in separate namespaces.

`flatcc` rejects this schema due to a name conflict because it uses the
basename of a union type, here `Monster` to generate the union member names
which are also used in JSON parsing.  This can be resolved by adding an
explicit name such as `Monster2` to resolve the conflict:

	union Any { Monster, Monster2: MyGame.Example2.Monster }

This syntax is accepted by both `flatc` and `flatcc`.

Both versions will implement the same union with the same type codes in the
binary format but generated code will differ in how the types are referred to.

In JSON the monster type values are now identified by
`MyGame.Example.Any.Monster`, or just `Monster`, when assigning the first
monster type to an Any union field, and `MyGame.Example.Any.Monster2`, or just
`Monster2` when assigning the second monster type. C uses the usual enum
namespace prefixed symbols like `MyGame_Example_Any_Monster2`.

## Fixed Length Arrays

Fixed Length Arrays is a late feature to the FlatBuffers format introduced in
flatc and flatcc mid 2019. Currently only scalars arrays are supported, and only
as struct fields. To use fixed length arrays as a table field wrap it in a
struct first. It would make sense to support struct elements and enum elements,
but that has not been implemented. Char arrays are more controversial due to
verification and zero termination and are also not supported. Arrays are aligned
to the size of the first field and are equivalent to repeating elements within
the struct.

The schema syntax is:

```
struct MyStruct {
    my_array : [float:10];
}
```

See `test_fixed_array` in [monster_test.c] for an example of how to work with
these arrays.

Flatcc opts to allow arbitrary length fixed length arrays but limit the entire
struct to 2^16-1 bytes. Tables cannot hold larger structs, and the C language
does not guarantee support for larger structs. Other implementations might have
different limits on maximum array size. Arrays of 0 length are not permitted.


## Optional Fields

Optional scalar table fields were introduced to FlatBuffers mid 2020 in order to
better handle null values also for scalar data types, as is common in SQL
databases. Before describing optional values, first understand how ordinary
scalar values work in FlatBuffers:

Imagine a FlatBuffer table with a `mana` field from the monster sample schema.
Ordinarily a scalar table field has implicit default value of 0 like `mana :
uint8;`, or an explicit default value specified in the schema like `mana : uint8
= 100;`. When a value is absent from a table field, the default value is
returned, and when a value is added during buffer construction, it will not
actually be stored if the value matches the default value, unless the
`force_add` option is used to write a value even if it matches the default
value. Likewise the `is_present` method can be used to test if a field was
actually stored in the buffer when reading it.

When a table has many fields, most of which just hold default settings,
significant space can be saved using default values, but it also means that an
absent value does not indicate null. Field absence is essentially just a data
compression technique, not a semantic change to the data. However, it is
possible to use `force_add` and `is_present` to interpret values as null when
not present, except that this is not a standardized technique. Optional fields
represents a standardized way to achieve this.

Scalar fields can be marked as optional by assigning `null` as a default
value. For example, some objects might not have a meaningful `mana`
value, so it could be represented as `lifeforce : uint8 = null`. Now the
`lifeforce` field has become an optional field. In the FlatCC implementation
this means that the field is written, it will always be written also if the
value is 0 or any other representable value. It also means that the `force_add`
method is not available for the field because `force_add` is essentially always
in effect for the field. On the read side, optional scalar fields behave exactly is ordinary scalar fields that have not specified a default value, that is, if the field is absent, 0 will be returned and `is_present` will return false. Instead optional scalar fields get a new accessor method with the suffix `_option()` which returns a struct with two fiels: `{ is_null, value }` where `_option().is_null == !is_present()` and `_option().value` is the same value is the `_get()` method, which will be 0 if `is_null` is true. The option struct is named after the type similar to unions, for example `flatbuffers_uint8_option_t` or `MyGame_Example_Color_option_t`, and the option accessor method also works similar to unions. Note that `_get()` will also return 0 for optional enum values that are null (i.e. absent), even if the enum value does not have an enumerated element with the value 0. Normally enums without a 0 element is not allowed in the schema unless a default value is specified, but in this case it is null, and `_get()` needs some value to return in this case.

By keeping the original accessors, read logic can be made simpler and faster when it is not important whether a value is null or 0 and at the same time the option value can be returned and stored.

Note that struct fields cannot be optional. Also note that, non-scalar table fields are not declared optional because these types can already represent null via a null pointer or a NONE union type.

JSON parsing and printing change behavior for scalar fields by treating absent
fields differently according the optional semantics. For example parsing a
missing field will not store a default value even if the parser is executed with
a flag to force default values to be stored and the printer will not print
absent optional fields even if otherwise flagged to print default values.
Currenlty the JSON printers and parsers do not print or parse JSON null and can
only represent null be absence of a field.

For an example of reading and writing, as well as printing and parsing JSON,
optional scalar fields, please refer to [optional_scalars_test.fbs] and [optional_scalars_test.c].


## Endianness

The [pendian_detect.h]` file detects endianness
for popular compilers and provides a runtime fallback detection for
others. In most cases even the runtime detection will be optimized out
at compile time in release builds.

The `FLATBUFFERS_LITTLEENDIAN` flag is respected for compatibility with
Googles `flatc` compiler, but it is recommended to avoid its use and
work with the mostly standard flags defined and/or used in
`pendian_detect.h`, or to provide for additional compiler support.

As of flatcc 0.4.0 there is support for flatbuffers running natively on
big endian hosts. This has been tested on IBM AIX. However, always run
tests against the system of interest - the release process does not cover
automated tests on any BE platform.

As of flatcc 0.4.0 there is also support for compiling the flatbuffers
runtime library with flatbuffers encoded in big endian format regardless
of the host platforms endianness. Longer term this should probably be
placed in a separate library with separate name prefixes or suffixes,
but it is usable as is. Redefine `FLATBUFFERS_PROTOCOL_IS_LE/BE`
accordingly in [flatcc_types.h]. This is already done in
the `be` branch. This branch is not maintained but the master branch can
be merged into it as needed.

Note that standard flatbuffers are always encoded in little endian but
in situations where all buffer producers and consumers are big endian,
the non standard big endian encoding may be faster, depending on
intrinsic byteswap support. As a curiosity, the `load_test` actually
runs faster with big endian buffers on a little endian MacOS platform
for reasons only the optimizer will know, but read performance of small
buffers drop to 40% while writing buffers generally drops to 80-90%
performance. For platforms without compiler intrinsics for byteswapping,
this can be much worse.

Flatbuffers encoded in big endian will have the optional file identifier
byteswapped. The interface should make this transparent, but details
are still being worked out. For example, a buffer should always verify
the monster buffer has the identifier "MONS", but internally the buffer
will store the identifier as "SNOM" on big endian encoded buffers.

Because buffers can be encode in two ways, `flatcc` uses the term
`native` endianness and `protocol` endianess. `_pe` is a suffix used in
various low level API calls to convert between native and protocol
endianness without caring about whether host or buffer is little or big
endian.

If it is necessary to write application code that behaves differently if
the native encoding differs from protocol encoding, use
`flatbuffers_is_pe_native()`. This is a function, not a define, but for
all practical purposes it will have same efficience while also
supporting runtime endian detection where necessary.

The flatbuffer environment only supports reading either big or little
endian for the time being. To test which is supported, use the define
`FLATBUFFERS_PROTOCOL_IS_LE` or `FLATBUFFERS_PROTOCOL_IS_BE`. They are
defines as 1 and 0 respectively.


## Pitfalls in Error Handling

The builder API often returns a reference or a pointer where null is
considered an error or at least a missing object default. However, some
operations do not have a meaningful object or value to return. These
follow the convention of 0 for success and non-zero for failure.
Also, if anything fails, it is not safe to proceed with building a
buffer.  However, to avoid overheads, there is no hand holding here. On
the upside, failures only happen with incorrect use or allocation
failure and since the allocator can be customized, it is possible to
provide a central error state there or to guarantee no failure will
happen depending on use case, assuming the API is otherwise used
correctly.  By not checking error codes, this logic also optimizes out
for better performance.


## Searching and Sorting

The builder API does not support sorting due to the complexity of
customizable emitters, but the reader API does support sorting so a
buffer can be sorted at a later stage. This requires casting a vector to
mutable and calling the sort method available for fields with keys.

The sort uses heap sort and can sort a vector in-place without using
external memory or recursion. Due to the lack of external memory, the
sort is not stable. The corresponding find operation returns the lowest
index of any matching key, or `flatbuffers_not_found`.

When configured in `config.h` (the default), the `flatcc` compiler
allows multiple keyed fields unlike Googles `flatc` compiler. This works
transparently by providing `<table_name>_vec_sort_by_<field_name>` and
`<table_name>_vec_find_by_<field_name>` methods for all keyed fields.
The first field maps to `<table_name>_vec_sort` and
`<table_name>_vec_find`. Obviously the chosen find method must match
the chosen sort method. The find operation is O(logN).

As of v0.6.0 the default key used for find and and sort without the `by_name`
suffix is the field with the smaller id instead of the first listed in the
schema which is often but not always the same thing.

v0.6.0 also introduces the `primary_key` attribute that can be used instead of
the `key` attribute on at most one field. The two attributes are mutually
exclusive. This can be used if a key field with a higher id should be the
default key. There is no difference when only one field has a `key` or
`primary_key` attribute, so in that case choose `key` for compatiblity.
Googles flatc compiler does not recognize the `primary_key` attribute.

As of v0.6.0 a 'sorted' attribute has been introduced together with the sort
operations `<table_name>_sort` and `<union_name>_sort`. If a table or a union,
directly or indirectly, contains a vector with the 'sorted' attribute, then the
sort operation is made available. The sort will recursively visit all children
with vectors marked sorted. The sort operatoin will use the default (primary)
key. A table or union must first be cast to mutable, for example
`ns(Monster_sort((ns(Monster_mutable_table_t))monster)`. The actual vector
sort operations are the same as before, they are just called automatically.
The `sorted` attribute can only be set on vectors that are not unions. The
vector can be of scalar, string, struct, or table type. `sorted` is only valid
for a struct or table vector if the struct or table has a field with a `key`
or `primary_key` attribute. NOTE: A FlatBuffer can reference the same object
multiple times. The sort operation will be repeated if this is the case.
Sometimes that is OK, but if it is a concern, remove the `sorted` attribute
and sort the vector manually. Note that sharing can also happen via a shared
containing object. The sort operations are generated in `_reader.h` files
and only for objects directly or indirectly affected by the `sorted` attribute.
Unions have a new mutable case operator for use with sorting unions:
`ns(Any_sort(ns(Any_mutable_cast)(my_any_union))`. Usually unions will be
sorted via a containing table which performs this cast automatically. See also
`test_recursive_sort` in [monster_test.c].

As of v0.4.1 `<table_name>_vec_scan_by_<field_name>` and the default
`<table_name>_vec_scan` are also provided, similar to `find`, but as a
linear search that does not require the vector to be sorted. This is
especially useful for searching by a secondary key (multiple keys is a
non-standard flatcc feature). `_scan_ex` searches a sub-range [a, b)
where b is an exclusive index. `b = flatbuffers_end == flatbuffers_not_found
== (size_t)-1` may be used when searching from a position to the end,
and `b` can also conveniently be the result of a previous search.

`rscan` searches in the opposite direction starting from the last
element. `rscan_ex` accepts the same range arguments as `scan_ex`. If
`a >= b or a >= len` the range is considered empty and
`flatbuffers_not_found` is returned. `[r]scan[_ex]_n[_by_name]` is for
length terminated string keys. See [monster_test.c] for examples.

Note that `find` requires `key` attribute in the schema. `scan` is also
available on keyed fields. By default `flatcc` will also enable scan by
any other field but this can be disabled by a compile time flag.

Basic types such as `uint8_vec` also have search operations.

See also [Builder Interface Reference] and [monster_test.c].


## Null Values

The FlatBuffers format does not fully distinguish between default values
and missing or null values but it is possible to force values to be
written to the buffer. This is discussed further in the
[Builder Interface Reference]. For SQL data roundtrips this may be more
important that having compact data.

The `_is_present` suffix on table access methods can be used to detect if
value is present in a vtable, for example `Monster_hp_present`. Unions
return true of the type field is present, even if it holds the value
None.

The `add` methods have corresponding `force_add` methods for scalar and enum
values to force storing the value even if it is default and thus making
it detectable by `is_present`.


## Portability Layer

The portable library is placed under `include/flatcc/portable` and is
required by flatcc, but isn't strictly part of the `flatcc` project. It
is intended as an independent light-weight header-only library to deal
with compiler and platform variations. It is placed under the flatcc
include path to simplify flatcc runtime distribution and to avoid
name and versioning conflicts if used by other projects.

The license of portable is different from `flatcc`. It is mostly MIT or
Apache depending on the original source of the various parts.

A larger set of portable files is included if `FLATCC_PORTABLE` is
defined by the user when building.

    cc -D FLATCC_PORTABLE -I include monster_test.c -o monster_test

Otherwise a targeted subset is
included by `flatcc_flatbuffers.h` in order to deal with non-standard
behavior of some C11 compilers.

`pwarnings.h` is also always included so compiler specific warnings can
be disabled where necessary.

The portable library includes the essential parts of the grisu3 library
found in `external/grisu3`, but excludes the test cases. The JSON
printer and parser relies on fast portable numeric print and parse
operations based mostly on grisu3.

If a specific platform has been tested, it would be good with feedback
and possibly patches to the portability layer so these can be made
available to other users.

## Building

Note: if a test fails, see [Strict Aliasing](#strict-aliasing)
for a possible resolution.

### Unix Build (OS-X, Linux, related)

To initialize and run the build (see required build tools below):

    scripts/build.sh

The `bin` and `lib` folders will be created with debug and release
build products.

The build depends on `CMake`. By default the `Ninja` build tool is also required,
but alternatively `make` can be used.

Optionally switch to a different build tool by choosing one of:

    scripts/initbuild.sh make
    scripts/initbuild.sh make-concurrent
    scripts/initbuild.sh ninja

where `ninja` is the default and `make-concurrent` is `make` with the `-j` flag.

To enforce a 32-bit build on a 64-bit machine the following configuration
can be used:

    scripts/initbuild.sh make-32bit

which uses `make` and provides the `-m32` flag to the compiler.
A custom build configuration `X` can be added by adding a
`scripts/build.cfg.X` file.

`scripts/initbuild.sh` cleans the build if a specific build
configuration is given as argument. Without arguments it only ensures
that CMake is initialized and is therefore fast to run on subsequent
calls. This is used by all test scripts.

To install build tools on OS-X, and build:

    brew update
    brew install cmake ninja
    git clone https://github.com/dvidelabs/flatcc.git
    cd flatcc
    scripts/build.sh

To install build tools on Ubuntu, and build:

    sudo apt-get update
    sudo apt-get install cmake ninja-build
    git clone https://github.com/dvidelabs/flatcc.git
    cd flatcc
    scripts/build.sh

To install build tools on Centos, and build:

    sudo yum group install "Development Tools"
    sudo yum install cmake
    git clone https://github.com/dvidelabs/flatcc.git
    cd flatcc
    scripts/initbuild.sh make # there is no ninja build tool
    scripts/build.sh


OS-X also has a HomeBrew package:

    brew update
    brew install flatcc

or for the bleeding edge:

    brew update
    brew install flatcc --HEAD


### Windows Build (MSVC)

Install CMake, MSVC, and git (tested with MSVC 14 2015).

In PowerShell:

    git clone https://github.com/dvidelabs/flatcc.git
    cd flatcc
    mkdir build\MSVC
    cd build\MSVC
    cmake -G "Visual Studio 14 2015" ..\..

Optionally also build from the command line (in build\MSVC):

    cmake --build . --target --config Debug
    cmake --build . --target --config Release

In Visual Studio:

    open flatcc\build\MSVC\FlatCC.sln
    build solution
    choose Release build configuration menu
    rebuild solution

*Note that `flatcc\CMakeList.txt` sets the `-DFLATCC_PORTABLE` flag and
that `include\flatcc\portable\pwarnings.h` disable certain warnings for
warning level -W3.*

### Docker

Docker image:

- <https://github.com/neomantra/docker-flatbuffers>


### Cross-compilation

Users have been reporting some degree of success using cross compiles
from Linux x86 host to embedded ARM Linux devices.

For this to work, `FLATCC_TEST` option should be disabled in part
because cross-compilation cannot run the cross-compiled flatcc tool, and
in part because there appears to be some issues with CMake custom build
steps needed when building test and sample projects.

2024-03-08: WARNING:
-O2 -mcpu=cortex-m7 targets using the arm-none-eabi 13.2.Rel1 toolchain
can result in uninitialized stack access when not compiled with
`-fno-strict-aliasing`
-mcpu=cortex-m0 and -mcpu=cortex-m1 appears to be unaffected.
See also [issue #274](https://github.com/dvidelabs/flatcc/issues/274).
2024-10-03: Fix available on flatcc master branch when you read this.
See also CHANGELOG comments for release 0.6.2.


The option `FLATCC_RTONLY` will disable tests and only build the runtime
library.

The following is not well tested, but may be a starting point:

    mkdir -p build/xbuild
    cd build/xbuild
    cmake ../.. -DBUILD_SHARED_LIBS=on -DFLATCC_RTONLY=on \
      -DCMAKE_BUILD_TYPE=Release

Overall, it may be simpler to create a separate Makefile and just
compile the few `src/runtime/*.c` into a library and distribute the
headers as for other platforms, unless `flatcc` is also required for the
target. Or to simply include the runtime source and header files in the user
project.

Note that no tests will be built nor run with `FLATCC_RTONLY` enabled.
It is highly recommended to at least run the `tests/monster_test`
project on a new platform.


### Custom Allocation

Some target systems will not work with Posix `malloc`, `realloc`, `free`
and C11 `aligned_alloc`. Or they might, but more allocation control is
desired. The best approach is to use `flatcc_builder_custom_init` to
provide a custom allocator and emitter object, but for simpler case or
while piloting a new platform
[flatcc_alloc.h](include/flatcc/flatcc_alloc.h) can be used to override
runtime allocation functions. _Carefully_ read the comments in this file
if doing so. There is a test case implementing a new emitter, and a
custom allocator can be copied from the one embedded in the builder
library source.


### Custom Asserts

On systems where the default POSIX `assert` call is unavailable, or when
a different assert behaviour is desirable, it is possible to override
the default behaviour in runtime part of flatcc library via logic defined
in [flatcc_assert.h](include/flatcc/flatcc_assert.h).

By default Posix `assert` is beeing used. It can be changed by preprocessor definition:

    -DFLATCC_ASSERT=own_assert

but it will not override assertions used in the portable library, notably the
Grisu3 fast numerical conversion library used with JSON parsing.

Runtime assertions can be disabled using:

    -DFLATCC_NO_ASSERT

This will also disable Grisu3 assertions. See
[flatcc_assert.h](include/flatcc/flatcc_assert.h) for details.

The `<assert.h>` file will in all cases remain a dependency for C11 style static
assertions. Static assertions are needed to ensure the generated structs have
the correct physical layout on all compilers. The portable library has a generic
static assert implementation for older compilers.


### Shared Libraries

By default libraries are built statically.

Occasionally there are requests
[#42](https://github.com/dvidelabs/flatcc/pull/42) for also building shared
libraries. It is not clear how to build both static and shared libraries
at the same time without choosing some unconvential naming scheme that
might affect install targets unexpectedly.

CMake supports building shared libraries out of the box using the
standard library name using the following option:

    CMAKE ... -DBUILD_SHARED_LIBS=ON ...

See also [CMake Gold: Static + shared](http://cgold.readthedocs.io/en/latest/tutorials/libraries/static-shared.html).


## Strict Aliasing

The Flatcc build files should take care of strict aliasing issues on
common platforms, but it is not a solved problem, so here is some background
information.

In most cases this is a non-issue with the current flatcc code
base, but that does not help in the cases where it is an issue.

Compilers have become increasingly aggressive with applying, and defaulting
to, strict aliasing rules.

FlatCC does not guarantee that strict aliasing rules are followed, but
the code base is updated as issues are detected. If a test fails or segfaults
the first thing to check is `-fno-strict-aliasing`, or the platform equivalent,
or to disable pointer casts, as discussed below.

Strict aliasing means that a cast like `p2 = *(T *)p1` is not valid because the
compiler thinks that p2 does not depend on data pointed to by p1. In most cases
compilers are sensible enough to handle this, but not always. It can, and
will, lead to reading from uninitialized memory or segfaults. There are two ways
around this, one is to use unions to convert from integer to float, which is
valid in C, but not in C++, and the other is to use `memcpy` for small constant
sizes, which is guaranteed safe, but can be slow if not optimized, and it is
not always optimized. (Not strictly `memcpy` but access via cast to `char *` or
other "narrow" type).

FlatCC manages this in [flatcc_accessors.h] which forwards to platform dependent
code in [pmemaccess.h]. Note that is applies to the runtime code base only. For
compile time the only issue should be hash tables and these should also be safe.

FlatCC either uses optimized `memcpy` or non-compliant pointer casts depending on
the platform. Essentially, buffer memory is first copied, or pointer cast, into
an unsigned integer of a given size. This integer is then endian converted into
another unsigned integer. Then that integer is converted into a final integer
type or floating point type using union casts. This generally optimizes out to
very few assembly instructions, but when it does not, code size and execution
time can grow significantly.

It has been observed that targets both default to strict aliasing with
-O2 optimization, and at the same to uses a function call for
`memcpy(dest, src, sizeof(uint32_t))`, but where `__builtin_memcpy`
does optimize well, hence requiring detection of a fast `memcpy` operation.

This is a game between being reasonably performant and compliant.

`-DPORTABLE_MEM_PTR_ACCESS=0` will force the runtime code to not use
pointer casts but it can potentially generate suboptimal code and
can be set 1 if the compiler and build configuration is known to not
have issues with strict aliasing. It is set to 1 for most x86/64 targets
since this has been working for a long time in FlatCC builds and tests,
while memcpy might not work efficient.


## Distribution

Install targes may be built with:

    mkdir -p build/install
    cd build/install
    cmake ../.. -DBUILD_SHARED_LIBS=on -DFLATCC_RTONLY=on \
      -DCMAKE_BUILD_TYPE=Release -DFLATCC_INSTALL=on
    make install

However, this is not well tested and should be seen as a starting point.
The normal scripts/build.sh places files in bin and lib of the source tree.

By default lib files a built into the `lib` subdirectory of the project. This
can be changed, for example like `-DFLATCC_INSTALL_LIB=lib64`.


### Unix Files

To distribute the compiled binaries the following files are
required:

Compiler:

    bin/flatcc                 (command line interface to schema compiler)
    lib/libflatcc.a            (optional, for linking with schema compiler)
    include/flatcc/flatcc.h    (optional, header and doc for libflatcc.a)

Runtime:

    include/flatcc/**          (runtime header files)
    include/flatcc/reflection  (optional)
    include/flatcc/support     (optional, only used for test and samples)
    lib/libflatccrt.a          (runtime library)

In addition the runtime library source files may be used instead of
`libflatccrt.a`. This may be handy when packaging the runtime library
along with schema specific generated files for a foreign target that is
not binary compatible with the host system:

    src/runtime/*.c

### Windows Files

The build products from MSVC are placed in the bin and lib subdirectories:

    flatcc\bin\Debug\flatcc.exe
    flatcc\lib\Debug\flatcc_d.lib
    flatcc\lib\Debug\flatccrt_d.lib
    flatcc\bin\Release\flatcc.exe
    flatcc\lib\Release\flatcc.lib
    flatcc\lib\Release\flatccrt.lib

Runtime `include\flatcc` directory is distributed like other platforms.


## Running Tests on Unix

Run

    scripts/test.sh [--no-clean]

**NOTE:** The test script will clean everything in the build directy before
initializing CMake with the chosen or default build configuration, then
build Debug and Release builds, and run tests for both.

The script must end with `TEST PASSED`, or it didn't pass.

To make sure everything works, also run the benchmarks:

    scripts/benchmark.sh


## Running Tests on Windows

In Visual Studio the test can be run as follows: first build the main
project, the right click the `RUN_TESTS` target and chose build. See
the output window for test results.

It is also possible to run tests from the command line after the project has
been built:

    cd build\MSVC
    ctest

Note that the monster example is disabled for MSVC 2010.

Be aware that tests copy and generate certain files which are not
automatically cleaned by Visual Studio. Close the solution and wipe the
`MSVC` directory, and start over to get a guaranteed clean build.

Please also observe that the file `.gitattributes` is used to prevent
certain files from getting CRLF line endings. Using another source
control systems might break tests, notably
`test/flatc_compat/monsterdata_test.golden`.


*Note: Benchmarks have not been ported to Windows.*


## Configuration

The configuration

    config/config.h

drives the permitted syntax and semantics of the schema compiler and
code generator. These generally default to be compatible with
Googles `flatc` compiler. It also sets things like permitted nesting
depth of structs and tables.

The runtime library has a separate configuration file

    include/flatcc/flatcc_rtconfig.h

This file can modify certain aspects of JSON parsing and printing such
as disabling the Grisu3 library or requiring that all names in JSON are
quoted.

For most users, it should not be relevant to modify these configuration
settings. If changes are required, they can be given in the build
system - it is not necessary to edit the config files, for example
to disable trailing comma in the JSON parser:

    cc -DFLATCC_JSON_PARSE_ALLOW_TRAILING_COMMA=0 ...


## Using the Compiler and Builder library

The compiler library `libflatcc.a` can compile schemas provided
in a memory buffer or as a filename. When given as a buffer, the schema
cannot contain include statements - these will cause a compile error.

When given a filename the behavior is similar to the commandline
`flatcc` interface, but with more options - see `flatcc.h` and
`config/config.h`.

`libflatcc.a` supports functions named `flatcc_...`. `reflection...` may
also be available which are simple the C generated interface for the
binary schema. The builder library is also included. These last two
interfaces are only present because the library supports binary schema
generation.

The standalone runtime library `libflatccrt.a` is a collection of the
`src/runtime/*.c` files. This supports the generated C headers for
various features. It is also possible to distribute and compile with the
source files directly.  For debugging, it is useful to use the
`libflatccrt_d.a` version because it catches a lot of incorrect API use
in assertions.

The runtime library may also be used by other languages. See comments
in [flatcc_builder.h]. JSON parsing is on example of an
alternative use of the builder library so it may help to inspect the
generated JSON parser source and runtime source.

## FlatBuffers Binary Format

Mostly for implementers: [FlatBuffers Binary Format]


## Security Considerations

See [Security Considerations].


## Style Guide

FlatCC coding style is largely similar to the [WebKit Style], with the following notable exceptions:

* Syntax requiring C99 or later is avoided, except `<stdint.h>` types are made available.
* If conditions always use curly brackets, or single line statements without linebreak: `if (err) return -1;`.
* NULL and nullptr are generally just represented as `0`.
* Comments are old-school C-style (pre C99). Text is generally cased with punctuation: `/* A comment. */`
* `true` and `false` keywords are not used (pre C99).
* In code generation there is essentially no formatting to avoid excessive bloat.
* Struct names and other types is lower case since this is C, not C++.
* `snake_case` is used over `camelCase`.
* Header guards are used over `#pragma once` because it is non-standard and not always reliable in filesystems with ambigious paths.
* Comma is not placed first in multi-line calls (but maybe that would be a good idea for diff stability).
* `config.h` inclusion might be handled differently in that `flatbuffers.h` includes the config file.
* `unsigned` is not used without `int` for historical reasons. Generally a type like `uint32_t` is preferred.
* Use `TODO:` instead of `FIXME:` in comments for historical reasons.

All the main source code in compiler and runtime aim to be C11 compatible and
uses many C11 constructs. This is made possible through the included portable
library such that older compilers can also function. Therefore any platform specific adaptations will be provided by updating
the portable library rather than introducing compile time flags in the main
source code.


## Benchmarks

See [Benchmarks]

[Builder Interface Reference]: https://github.com/dvidelabs/flatcc/blob/master/doc/builder.md
[FlatBuffers Binary Format]: https://github.com/dvidelabs/flatcc/blob/master/doc/binary-format.md
[Benchmarks]: https://github.com/dvidelabs/flatcc/blob/master/doc/benchmarks.md
[monster_test.c]: https://github.com/dvidelabs/flatcc/blob/master/test/monster_test/monster_test.c
[monster_test.fbs]: https://github.com/dvidelabs/flatcc/blob/master/test/monster_test/monster_test.fbs
[optional_scalars_test.fbs]: https://github.com/dvidelabs/flatcc/blob/optional/test/optional_scalars_test/optional_scalars_test.fbs
[optional_scalars_test.c]: https://github.com/dvidelabs/flatcc/blob/optional/test/optional_scalars_test/optional_scalars_test.c
[paligned_alloc.h]: https://github.com/dvidelabs/flatcc/blob/master/include/flatcc/portable/paligned_alloc.h
[test_json.c]: https://github.com/dvidelabs/flatcc/blob/master/test/json_test/test_json.c
[test_json_parser.c]: https://github.com/dvidelabs/flatcc/blob/master/test/json_test/test_json_parser.c
[flatcc_builder.h]: https://github.com/dvidelabs/flatcc/blob/master/include/flatcc/flatcc_builder.h
[flatcc_emitter.h]: https://github.com/dvidelabs/flatcc/blob/master/include/flatcc/flatcc_emitter.h
[flatcc-help.md]: https://github.com/dvidelabs/flatcc/blob/master/doc/flatcc-help.md
[flatcc_rtconfig.h]: https://github.com/dvidelabs/flatcc/blob/master/include/flatcc/flatcc_rtconfig.h
[hexdump.h]: https://github.com/dvidelabs/flatcc/blob/master/include/flatcc/support/hexdump.h
[readfile.h]: include/flatcc/support/readfile.h
[Security Considerations]: https://github.com/dvidelabs/flatcc/blob/master/doc/security.md
[flatc --annotate]: https://github.com/google/flatbuffers/tree/master/tests/annotated_binary
[flatcc_accessors.h]: https://github.com/dvidelabs/flatcc/blob/master/include/flatcc/flatcc_accessors.h
[pmemaccess.h]: https://github.com/dvidelabs/flatcc/blob/master/include/flatcc/portable/pmemaccess.h
[pendian_detect.h]: include/flatcc/portable/pendian_detect.h`
[flatcc_types.h]: include/flatcc/flatcc_types.h
[WebKit Style]: https://webkit.org/code-style-guidelines/
[FlatCC #287]: https://github.com/dvidelabs/flatcc/issues/287
[Google Flatbuffers #8374]: https://github.com/google/flatbuffers/issues/8374
[FlatCC #289]: https://github.com/dvidelabs/flatcc/issues/289
