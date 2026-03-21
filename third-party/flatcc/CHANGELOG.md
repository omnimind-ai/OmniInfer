# Change Log

## [0.6.2-pre]

- CMake: avoid assuming location of build dir during configuration.
- Use untyped integer constants in place of enums for public interface flags to
  allow for safe bit masking operations (PR #248).
- Added experimental support for generating `compile_commands.json` via
  `CMakeList.txt` for use with clangd.
- Remove `fallthrough` macro for improved portability (#247, #252).
- Added `parse_float/double_compare`, `parse_float/double_is_equal` to
  portable library, and added `parse_float/double_isnan` to mirror isinf.
  This should help with GCC 32-bit double precision conversion issue.
- Add Github Actions builds to replace stale Travis CI build. This also
  includes source code fixes for some build variants. Although
  Windows build is included it only covers recent 64-bit Windows. More
  work is need for older Windows variants. (#250).
- Increase maximum allowed schema file size from 64 KiB to 1 MB (#256).
- Fix seg fault in json parser while adding null characters to a too
  short input string for a fixed length char array struct field (#257).
- Fix regression where empty namespace in schema does not reset root scope
  correctly in parser (#265).
- Fix lexer checks that breaks with UTF-8, notably UTF-8 schema comments (#267).
- Add sanitizer flag for clang debug and related warnings (input from several
  PRs incl. #237).
- Fix missing runtime check for building too large tables (#235).
- Fix alignment of large objects created outside root buffer (#127).
- Pad top level buffer end to largest object in buffer
- Add `_with_size` verifiers to enable verification of size prefixed buffers
  that are aligned above 4 bytes (sizeof(uoffset_t)) (#210, PR #213).
  Also fixes a (uoffset_t) cast that would truncate very large verifier `bufsiz`
  arguments above 4GB before the verifier checked for argument overflow
  (not expected to be a practical problem, but could be abused).
- Fix `__[portable_aligned_alloc,` the fallback implementation of `aligned_alloc`
  so returns null proper and not a small offset on alloc failure. Most
  platforms do not depend on this function (#269).
  Also fix equivalent FLATCC_ALIGNED_ALLOC that does not necessarily use malloc.
- Fix flatcc compiler error message when schema has a union as first table field
  with explicit id attribute 1. Explicit id must leave space for the hidden type
  field, but id 1 is valid since id 0 is valid for the type field id. (#271).
- Add C23 support for "pstdbool.h" header.
- Fix strict aliasing UB. In the vast majority of cases `*(T *)p` style memory
  access is supported by compilers, but some both enable strict aliasing by
  default while being very aggressive with strict aliasing. This can lead to
  reading uninitialized memory and/or segfaults in the monster_test test case,
  but is less likely problematic in dynamically allocated memory buffers.
  For performance reasons this fix is not universally applied when deemed
  unnecessary. Added README section "Strict Aliasing". (#274).
- Fix flatcc cli flags for --schema --outfile. (#216).
- Replace cmetrohash and murmur hash with xxhash. (#238).
- Update verifier to accept buffers with misaligned empty vectors because C++
  has been discovered to be doing this for 10 years by now. The original
  stricter verifier can be activated with a compile time flag (#287, #289).
- Fix low impact incorrect use of strncmp in str_set hash table comp:arison.
  In flatcc this only affects duplicate header file inclusion detection during
  C code generation. Unlikely worst case a file might not be included which
  would cause a compile time error in user code. However, do take note if
  using the str_set table for other purposes. (#308).
- Fix some GCC indentation warnings in generated code - warnings are
  disabled in build already, but this may help some users (#313).
- Fix %d vs uint32 type in some printformatting (#315).
- Fix UB in builder, memsetting 0 bytes of null pointer (#344).
- Fix GCC 15 warning in grisu3 (#338).
- Bump CMake to >= 3.26.1 (Debian Bookworm).
- Fix CMAKE warnings on custom commands with TARGET and DEPENDS (CMP0175).
- Issue warnings on some potential name conflicts that can break C source
  and add `-s` option to silence warnings (#354).
- Fix potential UB in pprintint due to integer promotion, and silence warnings.

## [0.6.1]

- Add `flatcc_builder_alloc` and `flatcc_builder_free` to handle situations
  where standard allocation has been redefined via macros so `free` is no longer
  safe to use. These are similar to the existing `aligned_alloc/free` functions.
- Fix a potential, but never seen, low level race condition in the builder when
  writing a union field because the builder might reallocate between type
  field and value field. Affects `flacc_common_builder.h` but not `builder.c`.
- Fix GCC 8.3 reversal on `__alignas_is_defined` for -std=c++11, affecting
  pstdalign.h (#130).
- Make C++ test optional via build flag and disable for pre GCC-4.7 (#134).
- Avoid conflicts with table or struct fields named `identifier` (#99, #135)
  by not generating `<table-name>_identifier` for the table (or structs) file
  identifier, instead `<table-name>_file_identifier` is generated and the old
  form is generated for backwards compatibility when there is no conflict.
- DEPRECATED: `<table-name>_identifier` should now be
  `<table-name>_file_identifer`.
- `FLATCC_ASSERT` and `FLATCC_NO_ASSERT` provided for custom runtime assert
  handling (#139).
- Add guards in case bswap macroes are already defined and use POSIX compliant
  Linux detection (#129).
- Fix precision loss of floating point default values in generated code (#140).
- Fix anon union pedantic warning in internal builder struct (PR #143).
- Remove warnings triggered by -Wsign-conversion.
- BREAKING: some functions taking a string or buffer argument, and a separate
  size argument, have changed the size type from `int` to `size_t` in an effort
  to standardize and to remove sign conversion warnings.
- Fixed size arrays renamed to fixed length arrays for consistency with Googles
  project.
- Support `aligned_alloc` for MingW (#155).
- Disable config flag `FLATCC_ASCENDING_ENUM` now that Googles flatc compiler
  also has support for reordered enums.
- Fix cli options so common files can be generated without schema files
  (PR #156).
- Make build.sh work for non-bash shells (PR #159).
- Fix parsing json struct as roort (#157).
- Add support for optional scalar values (#162).
- Fix enum out of range (#176).
- Add stdalign support for TCC compiler (#174).
- Add RPC data to bfbs binary schema (#181).
- Fix support for printing JSON enum vectors (#182).
- Silence GCC 11 indentation warning (#183).
- Fix type of code field on json test (#184).
- Add `end_loc` to parser state of root json parsers (#186).
- BREAKING: Add per table `<name>_file_extension` and fixed wrong per table
  `<name>_file_identifier` assignment when multiple files are included due to
  macro expansion conflict. Extensions are now specified without extra dot
  prefix unless specified explicitly in the schema file. The default extension
  is now 'bin' instead of '.bin' (#187).
- Fix buffer overrun when parser reports error on large symbols (#188).
- BREAKING: Print --version to stdout, not stderr.
- Fix schema parser returning on success on some failure modes (#193).
- Fix larger integer literal types in JSON parser and printer (#194).
- Add pattributes.h to portable library and replace GCC fallthrough comments
  with fallthrough attribute to also silence clang warnings (#203).
- Remove misguided include guards from `portable/pdiagnostic_push/pop.h` and fix
  related and expected warnings in other code. NOTE: End user code might be
  affected because warnings were disabled more broadly than intended. Also note
  that warnings will still be disabled after pop if the compiler does not
  support push/pop diagnostics (#205).
- Fix verifier crash on malicious string length input (#221).
- Fix potential crash parsing unterminated JSON (#223).
- Allow 0 (and other unknown values) as schema default value for enums with
  `bit_flags` attribute.
- Disable -pedantic flag for GCC >= 8, it just keeps breaking perfectly valid
  code (#227).

## [0.6.0]

- BREAKING: if there are multiple table fields with a key attribute, the
  default key is now the field with the smaller id rather than the first
  listed in case these differ (which is not often the case).
- The attribute `primary_key` has been added to choose a specific keyed
  field as the default key field for finding and sorting instead of the
  field with the smallest id.
- Add mutable types: `mutable_table_t`, `mutable_union_t`, `mutable_union_vec_t`
  and casts: `mutable_union_cast`, `mutable_union_vec_cast`.
- Disallow key attribute on deprecated fields.
- Add 'sorted' attribute for scalar, string, table and struct vectors.
  Tables and structs must have a key field defined.
- Add recursive table and union `_sort` operation in `_reader.h` for
  types that contain a sorted vector, either directly or indirectly.
  NOTE: shared vectors in a DAG will be sorted multiple times.
- Allow attributes to be declared multiple times, including known attributes.
- Allow table and struct field names to be reserved names such as 'namespace'
  or 'table'. This can be disabled in `config/config.h`.
  Separately do the same for enum member. This change was motivated by
  JSON fields that can have names such as "table" or "namespace".
- Use `FLATCC_INSTALL_LIB` for install targets in addition to ordinary builds
  (#109).
- Add missing case break in flatcc compiler - no impact on behavior (#110).
- Make `flatbuffers_not_found` and `flatbuffers_end` constant values both
  because it is more correct, and to silence warnings on some systems.
- Fix flatcc endless loop with our of order field id's (#112).
- Add support for fixed length arrays as struct member fields, including
  fixed length char arrays. NOTE: bfbs schema uses byte arrays instead of
  char arrays since Googles flatc tool does not have char arrays.
- Fix `aligned_free` when used with `FLATCC_USE_GENERIC_ALIGNED_ALLOC` (#118).
- Fix potential buffer overrun when parsing JSON containing surrogate pairs
  with a resulting UTF-8 code point of length 4 (bug introduced in 0.5.3).
- BREAKING: empty structs are no longer supported. They were fully supported
  and Googles flatc compiler also no longer support them.
- Fix incorrect sprintf arg when printing NaN with the grisu3 algorithm (#119).
- Fix gcc 9 regression on `-fallthrough` comment syntax (#120).
- Silence false positive gcc 9 warning on strncpy truncation
  (-Wstringop-truncation) (#120).
- Silence false positive gcc 9 warning on printf null string
  (-Wno-format-overflow) (#120).
- Silence GCC `-Wstrict-prototypes` warnings that incorrectly complain about
  function definitions with empty instead of void arguments. Only for runtime
  source (#122).

## [0.5.3]
- BREAKING: 0.5.3 changes behavour of builder create calls so arguments
  are always ordered by field id when id attributes are being used, for
  example `MyGame_Example_Monster_create()` in `monster_test_fbs` (#81).
- Fix `refmap_clear` so that the refmap is always ready for reuse.
- Remove C++ from CMake toolchain when not building tests (#94).
- Add missing `_get` suffix in `flatbuffers_common_reader.h` needed when
  used with flatcc -g option (#96).
- Remove stray break statement that prevented generation of search
  methods for scalar table fields with the key attribute set (#97).
- Fix return value when creating a struct field fails which is unlikely
  to happen, and hence has low impact (#98).
- Fix file identifier string to type hash cast as implicit cast was not safe on
  all platforms.
- Fix reallocation bug when cloning vectors of non-scalar content (#102).
- JSON parser: combine valid UTF-16 surrogate pairs such as "\uD834\uDD1E"
  (U+1D11E) into single 4 byte UTF-8 sequence as per spec. Unmatched half
  pairs are intentionally decoded into invalid UTF-8 as before.
- Fix sorting tables by scalar keys. Sorting by integer key could lead to
  undefined behavior (#104).
- Add `FLATCC_INSTALL_LIB` configuration to CMake to change the
  default <project>/lib path, for example `cmake -DFLATCC_INSTALL_LIB=lib64`.
- Fix return code in `flatcc_parse_file` library function.

## [0.5.2]

- Handle union vectors in binary schema generation (.bfbs).
- Handle mixed union types in binary schema (.bfbs).
- Fix .bfbs bug failing to export fields of string type correctly.
- Fix how vectors are printed in samples/reflection project.
- Add support for KeyValue attributes in binary schema (.bfbs).
- Added `__tmp` suffix to macro variables in `flatbuffers_common_reader.h`
  to avoid potential name conflicts (#82).
- Added `_get` suffix to all table and struct read accessors in
  addition to existing accesors (`Monster_name()` vs `Monster_name_get()`
  (#82).
- Added `-g` option flatcc commandline to only generate read accessors
  with the `_get` suffix in order to avoid potential name conficts (#82).
- Fix stdalign.h not available in MSVC C++ in any known version.
- Added test case for building flatcc project with C++ compiler (#79, #80).
- Fix `flatbuffers_int8_vec_t` type which was incorrrectly unsigned.
- Added table, union, vector clone, and union vector operations. Table
  fields now also have a `_pick` method taking a source table of same
  type as argument which is roughly a combined get, clone and add
  operation for a single field. `_pick` will pick a field even if it is
  a default value and it will succedd as a no-operation if the source is
  absent. `_clone` discards deprecated fields. Unknown union types are
  also discarded along with unions of type NONE, even if present in
  source.  Warning: `_clone` will expand DAGs.
- Added `_get_ptr` reader method on scalar struct and table member
  fields which returns the equivalent of a single field struct `_get`
  method. This is used to clone scalar values without endian
  conversion. NOTE: scalars do not have assertions on `required`
  attribute, so be careful with null return values. For structs it
  is valid to apply `_get_ptr` to a null container struct such that
  navigation can be done without additional checks.
- Added `_push_clone` synonym for scalar and struct vector `_push_copy`.
- Add `include/flatcc/flatcc_refmap_h` and `src/runtime/refmap.c` to
  runtime library. The runtime library now depends of refmap.c but has
  very low overhead when not expeclity enabled for use with cloning.
- Add `flatbuffers_type_hash_from_string` to avoid gcc-8 strncpy warning
  (#86).
- Add long form flatcc options --common, --common-reader, --common-builder,
  --reader, --builder, --verifier.
- Remove flatcc deprecated option for schema namespace.
- Add scripts/flatcc-doc.sh to help document generated code.
- Remove unnecessary memory.h includes (#92)

## [0.5.1]

- Fix parent namespace lookup in the schema parser when the namespace
  prefix is omitted.
- Fix buffer overrun in JSON printer when exhausting flush buffer (#70).
- More consistent name generation across verifier and json parsers
  allowing for namespace wrapped parse/verify/print table functions.
- Fix unhelpful error on precision loss from float/double conversion
  in schema and JSON parser.
- Align `monster_test.fbs` Monster table more closely with Googles flatc
  version after they diverged a bit. (Subtables may differ).
- Some C++ compatiblity fixes so `include/{support/flatcc}` headers can
  be included into C++ without `extern "C"`.
- Fix missing null ptr check in fall-back `aligned_free`.
- Enable `posix_memalign` as a default build option on recent GNU
  systems because -std=c11 blocks automatic detection. This avoids
  using a less efficient fallback implementation.
- Add portable/include wrappers so build systems can add include paths
  to ensure that <stdint.h>, <stdbool.h> etc. is available. Flatcc does
  not currently rely on these.
- Replace `flatcc/portable/pdiagnostic_push/pop.h` includes in generated
  code with `flatcc/flatcc_pro/epilogue.h` and add `__cplusplus extern
  "C"` guards in those. This removes explicit references to the portable
  headers in generated code and improves C++ compatibility (#72).
- Change inconsistent `const void *` to `const char *` in JSON buffer
  argument to generated `_as_root` parsers (#73).
- Simplify use of json printers by auto-flushing and terminating buffers
  when a root object has been printed (#74).
- BREAKING: in extension of the changes in 0.5.0 for unions and union
  vectors, the low-level methods and structs now consistently use the
  terminology { type, value } for union types and values rather than {
  type, member } or { types, members }. The binary builder interface
  remains unchanged.
- Silence (unjustified) uninitialized gcc warnings (#75).
- Fix C++14 missing `__alignas_is_defined`.
- Remove newlib stdalign conflict (#77).
- Add `flatcc_json_printer_total`.
- Add `flatcc_builder_table_add_union_vector`.

## [0.5.0]
- New schema type aliases: int8, uint8, int16, uint16, int32, uint32,
  int64, uint64, float32, float64.
- Low-level: access multiple user frames in builder via handles.
- Support for `_is_known_type` and `_is_known_value` on union and enum
  types.
- More casts for C++ compatiblity (#59).
- Fix regressions in verifier fix in 0.4.3 that might report out of
  bounds in rare cases (#60).
- Silence gcc 7.x warnings about implicit fallthrough (#61).
- Fix rare special in JSON parser causing spurious unknown symbol.
- Reading and writing union vectors. The C++ interface also supports
  these types, but other languages likely won't for a while.
- New `_union(t)` method for accessing a union fields type and member
  table in a single call. The method also supports union vectors to
  retrieve the type vector and member vector as a single object.
- BREAKING: In generated builder code for union references of the form
  `<union-name>_union_ref_t` the union members and the hidden `_member`
  field has been replaced with a single `member` field. Union
  constructors work as before: `Any_union_ref_t uref =
  Any_as_weapon(weapon_ref)` Otherwise use `.type` and `.member` fields
  directly. This change was necessary to support the builder API's new
  union vectors without hitting strict aliasing rules, for example as
  argument to `flatcc_builder_union_vector_push`. Expected impact: low
  or none. The physical struct layout remains unchanged.
- BREAKING: `flatbuffers_generic_table_[vec_]t` has been renamed to
  `flatbuffers_generic_[vec_]t`.
- BREAKING: The verifiers runtime library interface has changed argument
  order from `align, size` to `size, align` in order to be consistent
  with the builders interface so generated code must match library
  version. No impact on user code calling generated verifier functions.
- BREAKING: generated json table parser now calls `table_end` and
  returns the reference in a new `pref` argument. Generated json struct
  parsers now renamed with an `_inline` suffix and the orignal name now
  parses a non-inline struct similar to the table parsers. No impact to
  user code that only calls the generated root parser.
- Fix off-by-one indexing in `flatbuffers_generic_vec_at`. Impact
  low since it was hardly relevant before union vectors were introduced
  in this release.
- Add document on security considerations (#63).
- Add support for base64 and base64url attributes in JSON printing and
  parsing of [ubyte] table fields.
- Added `flatcc_builder_aligned_free` and `flatcc_builder_aligned_alloc`
  to ensure `aligned_free` implementation matches allocation compiled
  into the runtime library. Note that alignment and size arguments are
  ordered opposite to most runtime library calls for consistency with
  the C11 `aligned_alloc` prototype.
- Support for struct and string types in unions.
- Add missing `_create` method on table union member fields.
- Add `_clone` and `_clone_as_[typed_]root[_with_size]` methods on structs.
  `_clone` was already supported on structs inlined in table fields.
- Fix harmless but space consuming overalignment of union types.
- Add `flatbuffers_union_type_t` with `flatbuffers_union_type_vec` operations.
- Fix scoping bug on union types in JSON parser: symbolic names of the form
  `MyUnion.MyUnionMember` were not accepted on a union type field but
  `MyNamespace.MyUnion.MyMember` and `MyMember` was supported. This has been
  fixed so all forms are valid. Plain enums did not have this issue.
- Place type identifiers early in generated `_reader.h` file to avoid
  circular reference issue with nested buffers when nested buffer type
  is placed after referencing table in schema.
- Fix verify bug on struct buffers - and in test case - not affecting
  ordinary buffers with table as root.

## [0.4.3]
- Fix issue with initbuild.sh for custom builds (#43)
- Add casts to aid clean C++ builds (#47)
- Add missing const specifier in generated `buffer_start` methods - removes C++
  warnings (#48)
- Update external/hash, removed buggy Sorted Robin Hood Hash that wasn't
  faster anyway - no impact on flatcc.
- Fix JSON parsing bug where some names are prefixes of others (#50).
- A Table of Contents in documentation :-)
- Move repetitive generated JSON string parsing into library.
- Add tests for JSON runtime compiled with different flags such as
  permitting unquoted keys.
- Fix building nested buffers when the parent buffer has not yet emitted
  any data (#51).
- Fix building nested buffers using the _nest() call (#52).
- Add `FLATCC_TRACE_VERIFY` as build option.
- Allow more costumization of allocation functions (#55).
- Avoid dependency on PORTABLE_H include guard which is too common (#55).
- (possibly breaking) Fix duplicate field check in flatcc_builder_table_add call.
- Fix incorrect infinity result in grisu3 parser and double to float
  overflow handling in parse_float in portable library (affects JSON
  of abnormal numeric values).
- Fix return value handling of parse_float, parse_double in JSON parser.
- Fix verifier vector alignment check - affects vectors with element size 8+.
- Fix missing static in generated enum and union functions in JSON
  printer (#57).

## [0.4.2]
- Fix SIGNIFICANT bug miscalculating the number of builder frames in
  use. Nesting 8 levels would cause memory corruption (#41).
- Fix minor memory leak in flatcc compiler.
- Reduce collisions in builders vtable hash.
- Remove broken dependency on `<mm_malloc.h>` for some GCC versions in
  `paligned_alloc.h` (#40).
- Allow C++ files to include `pstdalign.h` and `paligned_alloc.h` (#39).

## [0.4.1]
- Test for `posix_memalign` on GCC platforms and fix fallback
  `aligned_alloc`.
- Fix JSON parser handling of empty objects and tables.
- Fix JSON parser - some fields would not be accepted as valid (#17).
- Fix rare uncompilable doc comment in schema (#21).
- Avoid crash on certain table parser error cases (#30).
- Add support for scan similar to find in reader API, but for O(N)
  unsorted search, or search by a secondary key, and in sub-ranges.
- Optionally, and by default, allow scan by any field (#29), not just keys.
- More compact code generation for reader (hiding scan methods).
- Use __flatbuffers_utype_t for union type in reader instead of uint8_t.
- Add unaligned write to punaligned for completeness.
- Promote use of `flatcc_builder_finalize_aligned_buffer` in doc and
  samples over `flatcc_builder_finalize_buffer`.
- Add scope counter to pstatic_assert.h to avoid line number conflicts.
- Fix compiler error/warning for negative enums in generated JSON parser (#35).
- Fix potential compiler error/warnings for large enum/defaults in
  generated reader/builder (#35).
- Fix tab character in C++ style comments (#34)
- Fix incorrect api usage in binary schema builder (#32)
- Support hex constants in fbs schema (flatc also supports these now) (#33).


## [0.4.0]
- Fix Windows detection in flatcc/support/elapsed.h used by benchmark.
- Fix #8 surplus integer literal suffix in portable byteswap fallback.
- Fix `pstatic_assert.h` missing fallback case.
- Fix #9 return values from allocation can be zero without being an error.
- Fix #11 by avoiding dependency on -lm (libmath) by providing a cleaner
  over/underflow function in `include/flatcc/portable/pparsefp.h`.
- Fix #12 infinite loop during flatbuffer build operations caused by
  rare vtable dedupe hash table collision chains.
- Added `include/flatcc/support/cdump.h` tool for encoding buffers in C.
- JSON code generators no longer use non-portable PRIszu print
  modifiers. Fixes issue on IBM XLC AIX.
- Deprecated support for PRIsz? print modifiers in
  `include/flatcc/portable/pinttypes.h`, they would require much more
  work to be portable.
- Fix and clean up `__STDC__` version checks in portable library.
- Improve IBM XLC support in `pstdalign.h`.
- Always include `pstdalign.h` in `flatcc_flatbuffers.h` because some
  C11 compilers fail to provide `stdalign.h`.
- Buffer verifier used to mostly, but not always, verify buffer
  alignment relative to buffer start. With size prefixed buffers it is
  necessary to verify relative to the allocated buffer, which is also
  safer and more consistent, but adds requirements to aligned allocation.
- `monster_test` and `flatc_compat` test now uses aligned alloc.
- Add `aligned_alloc` and `aligned_free` to `pstdalign.h`.
- `flatcc_builder_finalize_aligned_buffer` now requires `aligned_free`
  to be fully portable and no longer use unaligned malloc as fallback,
  but still works with `free` on most platforms (not Windows).

- BREAKING: Size prefixed buffers added requires a minor change
  to the low-level flatcc builder library with a flag argument to create
  and start buffer calls. This should not affect user code.


Changes related to big endian support which do not affect little endian
platforms with little endian wire format.

- Support for big endian platforms, tested on IBM AIX Power PC.
- Support for big endian encoded flatbuffers on both little and big
  endian host platforms via `FLATBUFFERS_PROTOCOL_IS_LE/BE` in
  `include/flatcc/flatcc_types.h`. Use `flatbuffers_is_native_pe()` to
  see if the host native endian format matches the buffer protocol.
  NOTE: file identifier at buffer offset 4 is always byteswapped.

In more detail:

- Fix vtable conversion to protocol endian format. This keeps cached
  vtables entirely in native format and reduces hash collisions and only
  converts when emitting the vtable to a buffer location.
- Fix structs created with parameter list resulting in double endian
  conversion back to native.
- Fix string swap used in sort due to endian sensitive diff math.
- Disable direct vector access test case when running on non-native
  endian platform.
- Update JSON printer test to handle `FLATBUFFERS_PROTOCOL_IS_BE`.
- Fix emit test case. Incorrect assumption on acceptable null pointer
  breaks with null pointer conversion. Also add binary check when
  `FLATBUFFERS_PROTOCOL_IS_BE`.
- Add binary test case to `json_test` when `FLATBUFFERS_PROTOCOL_IS_BE`.
- Fix endian sensitive voffset access in json printer.
- Update `flatc_compat` to reverse acceptance of 'golden' little endian
  reference buffer when `FLATBUFFERS_PROTOCOL_IS_BE`.

## [0.3.5a]
- Fix regression introduced in 0.3.5 that caused double memory free on
  input file buffer. See issue #7.

## [0.3.5]

- Allow flatcc cli options anywhere in the argument list.
- Add --outfile option similar to --stdout, but to a file.
- Add --depfile and --deptarget options for build dependencies.
- Allow some test cases to accept arguments to avoid hardcoded paths.
- Deprecate --schema-namespace=no option to disable namespace prefixes
  in binary schema as Google flatbuffers now also includes namespaces
  according to https://github.com/google/flatbuffers/pull/4025

## [0.3.4]

- Add `FLATCC_RTONLY` and `FLATCC_INSTALL` build options.
- Fix issue4: when building a buffer and the first thing created is an
  empty table, the builder wrongly assumed allocation failure. Affects
  runtime library.
- `scripts/setup.sh` now also links to debug libraries useful for bug
  reporting.
- Add ULL suffix to large printed constants in generated code which
  would otherwise required --std=c99 to silence warnings.

## [0.3.3]

- BREAKING: `verify_as_root` no longer takes an identifier argument, use
  `verify_as_root_with_identifier`. `myschema_verifier.h` now
  includes `myschema_reader.h` to access identifier.
  identifer argument, and variants for type identifiers;.
- Added scripts/setup.sh to quickly get started on small user projects.
- Support `namespace ;` for reverting to global namespace in schema.
- Enable block comments now that they are supported in flatc.
- Parse and validate new `rpc_service` schema syntax, but with no
  support for code generation.
- Add type hash support (`create/verify_as_typed_root` etc.) to
  optionally store and verify file identifiers based on hashed fully
  qualified type names.
- Fix potential issue with detection of valid file identifiers in
  buffer.
- Moved `include/support` into `include/flatcc/support`, renamed
  `include/support/readfile.h` function `read_file` to `readfile`.
- Make `FLATCC_TEST` build option skip building samples and test
  files, as opposed to just skip running the tests.
- `vec_at`, `vec_find`, etc. now use index type `size_t` instead of
  `flatbuffers_uoffset_t`.
- Removed `size_t` conversion warnings on Win64.

## [0.3.2]

- Move compiler warning handling from generated headers to portable
  library.
- Clean up warnings and errors for older gcc, clang and MSVC compilers.
- CI builds.
- Fix and improve portable version of `static_assert`.
- Add integer parsing to portable library.

## [0.3.1]

- Add support for MSVC on Windows.
- Allow FlatBuffer enums to be used in switch statements without warnings.
- Remove warnings for 32-bit builds.
- Fix runtime detection of endianness and add support for more
  platforms with compile time detection of endianness.
- Fix scope bug where global namespace symbols from included schema
  would be invisible in parent schema.
- Add missing `static` for generated union verifiers.
- Fix bug in json printer unicode escape and hash table bug in
  compiler.
- Fix json parser under allocation bug.
- Fix `emit_test` early dealloc bug.

## [0.3.0]

- Rename examples folder to samples folder.
- Add samples/monster example.
- BREAKING: added missing `_vec` infix on some operations related to
  building vectors. For example `Weapon_push` -> `Weapon_vec_push`.
- BREAKING: vector and string start functions no longer takes a
  count/len argument as it proved tedious and not very useful.
  The return value is now 0 on success rather than a buffer pointer.
  Use `_extend` call after start when the length argument is non-zero.

## [0.2.1]

- Disallow unquoted symbolic list in JSON parser by default for Google
  flatc compatibility.
- Remove PRIVATE flags from CMake build files to support older CMake
  versions.
- Simplify switching between ninja and make build tools.
- Fix incorrectly named unaligned read macros - impacts non-x86 targets.
- Mirror grisu3 headers in portable library to avoid dependency on
  `external/grisu3`.
- Use correct grisu3 header for parsing, improving json parsing times.
- Move `include/portable` to `include/flatcc/portable` to simplify
  runtime distribution and to prevent potential name and versioning
  conflicts.
- Fix `is_union` in bfbs2json.c example.

## [0.2.0]

- BREAKING: flatcc verify functions now return a distinct error code.
  This breaks existing code. Before non-zero was success, now `flatcc_verify_ok` == 0.
  The error code can be converted to a string using `flatcc_verify_error_string(ret)`.
- BREAKING, minor: Remove user state from builder interface - now
  providing a user stack instead.
- Fix verification of nested flatbuffers.
- Fix certain header fields that was not endian encoded in builder.
- MAJOR: Generate json printer and parser.
- Added high performance integer printinger to portable library
  and fast floating point priting to runtime library (grisu3) for JSON.
- Comparison against default value now prints float to generated source
  with full precision ("%.17g").
- MAJOR: read-only generated files no longer attempt to be independent
  of files in the flatcc include dir. Instead they will use one
  well-defined source of information for flatbuffer types and endian detection.
- Always depend `portable/pendian.h` and `portable/pendian_detect.h`.
  (The `include/portable` dir can be copied under `include/flatcc` if so desired).
- Updates to set of include files in include/flatcc.
- Upgrade to pstdint.h 0.1.15 to fix 64-bit printf issue on OS-X.
- Support for portable unaligned reads.
- Hide symbols that leak into namespace via parent include.
- Suppress unused function and variable warnings for GCC (in addition to clang).

## [0.1.1]

- Rename libflatccbuilder.a to libflatccrt.a (flatcc runtime).
- Add suffix to all generated files (`monster_test.h -> monster_test_reader.h`)
- Add buffer verifiers (`monster_test_verifier.h).
- Assert on error in flatcc builder by default.
- Fix -I include path regression in `flatcc` command.

## [0.1.0]

- Initial public release.
