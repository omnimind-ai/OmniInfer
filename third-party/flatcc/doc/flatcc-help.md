```
flatcc FlatBuffers schema compiler for C by dvide.com
version: 0.6.2
usage: flatcc [options] file [...]
options:
  --reader                   (default) Generate reader
  -c, --common               Generate common include header(s)
  --common_reader            Generate common reader include header(s)
  --common_builder           Generate common builder include header(s)
  -w, --builder              Generate builders (writable buffers)
  -v, --verifier             Generate verifier
  -r, --recursive            Recursively generate included schema files
  -a                         Generate all (like -cwvr)
  -g                         Use _get suffix only to avoid conflicts
  -s                         Silence warnings
  -d                         Dependency file like gcc -MMD
  -I<inpath>                 Search path for include files (multiple allowed)
  -o<outpath>                Write files relative to this path (dir must exist)
  --stdout                   Concatenate all output to stdout
  --outfile=<file>           Like --stdout, but to a file.
  --depfile=<file>           Dependency file like gcc -MF.
  --deptarget=<file>         Override --depfile target like gcc -MT.
  --prefix=<prefix>          Add prefix to all generated names (no _ added)
  --common-prefix=<prefix>   Replace 'flatbuffers' prefix in common files
  --schema                   Generate binary schema (.bfbs)
  --schema-length=no         Add length prefix to binary schema
  --verifier                 Generate verifier for schema
  --json-parser              Generate json parser for schema
  --json-printer             Generate json printer for schema
  --json                     Generate both json parser and printer for schema
  --version                  Show version
  -h | --help                Help message

This is a flatbuffer compatible compiler implemented in C generating C
source. It is largely compatible with the flatc compiler provided by
Google Fun Propulsion Lab but does not support JSON objects or binary
schema.

By example 'flatcc monster.fbs' generates a 'monster.h' file which
provides functions to read a flatbuffer. A common include header is also
required. The common file is generated with the -c option. The reader
has no external dependencies.

The -w (--builder) option enables code generation to build buffers:
`flatbuffers -w monster.fbs` will generate `monster.h` and
`monster_builder.h`, and also a builder specific common file with the
-cw option. The builder must link with the extern `flatbuilder` library.

-v (--verifier) generates a verifier file per schema. It depends on the
runtime library but not on other generated files, except other included
verifiers.

-r (--recursive) generates all schema included recursively.

--reader is the default option to generate reader output but can be used
explicitly together with other options that would otherwise disable it.

All C output can be concated to a single file using --stdout or
--outfile with content produced in dependency order. The outfile is
relative to cwd.

-g Only add '_get' suffix to read accessors such that, for example,
only 'Monster_name_get(monster)` will be generated and not also
'Monster_name(monster)'. This avoids potential conflicts with
other generated symbols when a schema change is impractical.

-s Silence warnings, notably on potential conflicts that might cause
C source code to generate errors. These warnings are not perfect and can
be ignored if the C code compiles, though it is recommended to change
names in the schema if possible.

-d generates a dependency file, e.g. 'monster.fbs.d' in the output dir.

--depfile implies -d but accepts an explicit filename with a path
relative to cwd. The dependency files content is a gnu make rule with a
target followed by the included schema files The target must match how
it is seen by the rest of the build system and defaults to e.g.
'monster_reader.h' or 'monster.bfbs' paths relative to the working
directory.

--deptarget overrides the default target for --depfile, simiar to gcc -MT.

--schema will generate a binary .bfbs file for each top-level schema file.
Can be used with --stdout if no C output is specified. When used with multiple
files --schema-length=yes is recommend.

--schema-length adds a length prefix of type uoffset_t to binary schema so
they can be concatenated - the aligned buffer starts after the prefix.

--json-parser generates a file that implements a fast typed json parser for
the schema. It depends on some flatcc headers and the runtime library but
not on other generated files except other parsers from included schema.

--json-printer generates a file that implements json printers for the schema
and has dependencies similar to --json-parser.

--json is generates both printer and parser.

The generated source can redefine offset sizes by including a modified
`flatcc_types.h` file. The flatbuilder library must then be compiled with the
same `flatcc_types.h` file. In this case --prefix and --common-prefix options
may be helpful to avoid conflict with standard offset sizes.

The output size may seem bulky, but most content is rarely used inline
functions and macros. The compiled binary need not be large.

The generated source assumes C11 functionality for alignment, compile
time assertions and inline functions but an optional set of portability
headers can be included to work with most any compiler. The portability
layer is not throughly tested so a platform specific test is required
before production use. Upstream patches are welcome.
```
