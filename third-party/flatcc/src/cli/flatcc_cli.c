#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "flatcc/flatcc.h"
#include "config.h"

#define VERSION FLATCC_VERSION_TEXT
#define TITLE FLATCC_TITLE_TEXT

void usage(FILE *fp)
{
    fprintf(fp, "%s\n", TITLE);
    fprintf(fp, "version: %s\n", VERSION);
    fprintf(fp, "usage: flatcc [options] file [...]\n");
    fprintf(fp, "options:\n"
            "  --reader                   (default) Generate reader\n"
            "  -c, --common               Generate common include header(s)\n"
            "  --common_reader            Generate common reader include header(s)\n"
            "  --common_builder           Generate common builder include header(s)\n"
            "  -w, --builder              Generate builders (writable buffers)\n"
            "  -v, --verifier             Generate verifier\n"
            "  -r, --recursive            Recursively generate included schema files\n"
            "  -a                         Generate all (like -cwvr)\n"
            "  -g                         Use _get suffix only to avoid conflicts\n"
            "  -s                         Silence warnings\n"
            "  -d                         Dependency file like gcc -MMD\n"
            "  -I<inpath>                 Search path for include files (multiple allowed)\n"
            "  -o<outpath>                Write files relative to this path (dir must exist)\n"
            "  --stdout                   Concatenate all output to stdout\n"
            "  --outfile=<file>           Like --stdout, but to a file.\n"
            "  --depfile=<file>           Dependency file like gcc -MF.\n"
            "  --deptarget=<file>         Override --depfile target like gcc -MT.\n"
            "  --prefix=<prefix>          Add prefix to all generated names (no _ added)\n"
            "  --common-prefix=<prefix>   Replace 'flatbuffers' prefix in common files\n"
#if FLATCC_REFLECTION
            "  --schema                   Generate binary schema (.bfbs)\n"
            "  --schema-length=no         Add length prefix to binary schema\n"
#endif
            "  --verifier                 Generate verifier for schema\n"
            "  --json-parser              Generate json parser for schema\n"
            "  --json-printer             Generate json printer for schema\n"
            "  --json                     Generate both json parser and printer for schema\n"
            "  --version                  Show version\n"
            "  -h | --help                Help message\n"
    );
}

void help(FILE *fp)
{
    usage(fp);
    fprintf(fp,
        "\n"
        "This is a flatbuffer compatible compiler implemented in C generating C\n"
        "source. It is largely compatible with the flatc compiler provided by\n"
        "Google Fun Propulsion Lab but does not support JSON objects or binary\n"
        "schema.\n"
        "\n"
        "By example 'flatcc monster.fbs' generates a 'monster.h' file which\n"
        "provides functions to read a flatbuffer. A common include header is also\n"
        "required. The common file is generated with the -c option. The reader\n"
        "has no external dependencies.\n"
        "\n"
        "The -w (--builder) option enables code generation to build buffers:\n"
        "`flatbuffers -w monster.fbs` will generate `monster.h` and\n"
        "`monster_builder.h`, and also a builder specific common file with the\n"
        "-cw option. The builder must link with the extern `flatbuilder` library.\n"
        "\n"
        "-v (--verifier) generates a verifier file per schema. It depends on the\n"
        "runtime library but not on other generated files, except other included\n"
        "verifiers.\n"
        "\n"
        "-r (--recursive) generates all schema included recursively.\n"
        "\n"
        "--reader is the default option to generate reader output but can be used\n"
        "explicitly together with other options that would otherwise disable it.\n"
        "\n"
        "All C output can be concated to a single file using --stdout or\n"
        "--outfile with content produced in dependency order. The outfile is\n"
        "relative to cwd.\n"
        "\n"
        "-g Only add '_get' suffix to read accessors such that, for example,\n"
        "only 'Monster_name_get(monster)` will be generated and not also\n"
        "'Monster_name(monster)'. This avoids potential conflicts with\n"
        "other generated symbols when a schema change is impractical.\n"
        "\n"
        "-s Silence warnings, notably on potential conflicts that might cause\n"
        "C source code to generate errors. These warnings are not perfect and can\n"
        "be ignored if the C code compiles, though it is recommended to change\n"
        "names in the schema if possible.\n"
        "\n"
        "-d generates a dependency file, e.g. 'monster.fbs.d' in the output dir.\n"
        "\n"
        "--depfile implies -d but accepts an explicit filename with a path\n"
        "relative to cwd. The dependency files content is a gnu make rule with a\n"
        "target followed by the included schema files The target must match how\n"
        "it is seen by the rest of the build system and defaults to e.g.\n"
        "'monster_reader.h' or 'monster.bfbs' paths relative to the working\n"
        "directory.\n"
        "\n"
        "--deptarget overrides the default target for --depfile, simiar to gcc -MT.\n"
        "\n"

#if FLATCC_REFLECTION
        "--schema will generate a binary .bfbs file for each top-level schema file.\n"
        "Can be used with --stdout if no C output is specified. When used with multiple\n"
        "files --schema-length=yes is recommend.\n"
        "\n"
        "--schema-length adds a length prefix of type uoffset_t to binary schema so\n"
        "they can be concatenated - the aligned buffer starts after the prefix.\n"
        "\n"
#else
        "Flatbuffers binary schema support (--schema) has been disabled."
        "\n"
#endif
        "--json-parser generates a file that implements a fast typed json parser for\n"
        "the schema. It depends on some flatcc headers and the runtime library but\n"
        "not on other generated files except other parsers from included schema.\n"
        "\n"
        "--json-printer generates a file that implements json printers for the schema\n"
        "and has dependencies similar to --json-parser.\n"
        "\n"
        "--json is generates both printer and parser.\n"
        "\n"
#if FLATCC_REFLECTION
#if 0 /* Disable deprecated features. */
        "DEPRECATED:\n"
        "  --schema-namespace controls if typenames in schema are prefixed a namespace.\n"
        "  namespaces should always be present.\n"
        "\n"
#endif
#endif
        "The generated source can redefine offset sizes by including a modified\n"
        "`flatcc_types.h` file. The flatbuilder library must then be compiled with the\n"
        "same `flatcc_types.h` file. In this case --prefix and --common-prefix options\n"
        "may be helpful to avoid conflict with standard offset sizes.\n"
        "\n"
        "The output size may seem bulky, but most content is rarely used inline\n"
        "functions and macros. The compiled binary need not be large.\n"
        "\n"
        "The generated source assumes C11 functionality for alignment, compile\n"
        "time assertions and inline functions but an optional set of portability\n"
        "headers can be included to work with most any compiler. The portability\n"
        "layer is not throughly tested so a platform specific test is required\n"
        "before production use. Upstream patches are welcome.\n");
}

enum { noarg, suffixarg, nextarg };

int parse_bool_arg(const char *a)
{
    if (strcmp(a, "0") == 0 || strcmp(a, "no") == 0) {
        return 0;
    }
    if (strcmp(a, "1") == 0 || strcmp(a, "yes") == 0) {
        return 1;
    }
    fprintf(stderr, "invalid boolean argument: '%s', must be '0', '1', 'yes' or 'no'\n", a);
    return -1;
}

int match_long_arg(const char *option, const char *s, size_t n)
{
    return strncmp(option, s, n) == 0 && strlen(option) == n;
}

int set_opt(flatcc_options_t *opts, const char *s, const char *a)
{
    int ret = noarg;
    size_t n = strlen(s);
    const char *v = strchr(s, '=');
    if (v) {
        a = v + 1;
        n = (size_t)(v - s);
    }
    if (*s == 'h' || 0 == strcmp("-help", s)) {
        /* stdout so less and more works. */
        help(stdout);
        exit(0);
    }
    if (0 == strcmp("-version", s)) {
        fprintf(stdout, "%s\n", TITLE);
        fprintf(stdout, "version: %s\n", VERSION);
        exit(0);
    }
    if (0 == strcmp("-stdout", s)) {
        opts->gen_stdout = 1;
        return noarg;
    }
    if (0 == strcmp("-common", s)) {
        opts->cgen_common_reader = 1;
        opts->cgen_common_builder = 1;
        return noarg;
    }
    if (0 == strcmp("-common_reader", s)) {
        opts->cgen_common_reader = 1;
        return noarg;
    }
    if (0 == strcmp("-common_builder", s)) {
        opts->cgen_common_builder = 1;
        return noarg;
    }
    if (0 == strcmp("-reader", s)) {
        opts->cgen_reader = 1;
        return noarg;
    }
    if (0 == strcmp("-builder", s)) {
        opts->cgen_builder = 1;
        return noarg;
    }
    if (0 == strcmp("-verifier", s)) {
        opts->cgen_verifier = 1;
        return noarg;
    }
    if (0 == strcmp("-recursive", s)) {
        opts->cgen_recursive = 1;
        return noarg;
    }
#if FLATCC_REFLECTION
    if (0 == strcmp("-schema", s)) {
        opts->bgen_bfbs = 1;
        return noarg;
    }
#endif
    if (0 == strcmp("-json-parser", s)) {
        opts->cgen_json_parser = 1;
        return noarg;
    }
    if (0 == strcmp("-json-printer", s)) {
        opts->cgen_json_printer = 1;
        return noarg;
    }
    if (0 == strcmp("-json", s)) {
        opts->cgen_json_parser = 1;
        opts->cgen_json_printer = 1;
        return noarg;
    }
#if FLATCC_REFLECTION
#if 0 /* Disable deprecated features. */
    if (match_long_arg("-schema-namespace", s, n)) {
        fprintf(stderr, "warning: --schema-namespace is deprecated\n"
                        "         a namespace is added by default and should always be present\n");
        if (!a) {
            fprintf(stderr, "--schema-namespace option needs an argument\n");
            exit(-1);
        }
        if(0 > (opts->bgen_qualify_names = parse_bool_arg(a))) {
            exit(-1);
        }
        return v ? noarg : nextarg;
    }
    if (match_long_arg("-schema-length", s, n)) {
        if (!a) {
            fprintf(stderr, "--schema-length option needs an argument\n");
            exit(-1);
        }
        if(0 > (opts->bgen_length_prefix = parse_bool_arg(a))) {
            exit(-1);
        }
        return v ? noarg : nextarg;
    }
#endif
#endif
    if (match_long_arg("-depfile", s, n)) {
        if (!a) {
            fprintf(stderr, "--depfile option needs an argument\n");
            exit(-1);
        }
        opts->gen_depfile = a;
        opts->gen_dep = 1;
        return v ? noarg : nextarg;
    }
    if (match_long_arg("-deptarget", s, n)) {
        if (!a) {
            fprintf(stderr, "--deptarget option needs an argument\n");
            exit(-1);
        }
        opts->gen_deptarget = a;
        return v ? noarg : nextarg;
    }
    if (match_long_arg("-outfile", s, n)) {
        if (!a) {
            fprintf(stderr, "--outfile option needs an argument\n");
            exit(-1);
        }
        opts->gen_outfile= a;
        return v ? noarg : nextarg;
    }
    if (match_long_arg("-common-prefix", s, n)) {
        if (!a) {
            fprintf(stderr, "--common-prefix option needs an argument\n");
            exit(-1);
        }
        opts->nsc = a;
        return v ? noarg : nextarg;
    }
    if (match_long_arg("-prefix", s, n)) {
        if (!a) {
            fprintf(stderr, "-n option needs an argument\n");
            exit(-1);
        }
        opts->ns = a;
        return v ? noarg : nextarg;
    }
    switch (*s) {
    case '-':
        fprintf(stderr, "invalid option: -%s\n", s);
        exit(-1);
    case 'I':
        if (s[1]) {
            ret = suffixarg;
            a = s + 1;
        } else if (!a) {
            fprintf(stderr, "-I option needs an argument\n");
            exit(-1);
        } else {
            ret = nextarg;
        }
        opts->inpaths[opts->inpath_count++] = a;
        return ret;
    case 'o':
        if (opts->outpath) {
            fprintf(stderr, "-o option can only be specified once\n");
            exit(-1);
        }
        if (s[1]) {
            ret = suffixarg;
            a = s + 1;
        } else if (!a) {
            fprintf(stderr, "-o option needs an argument\n");
            exit(-1);
        } else {
            ret = nextarg;
        }
        opts->outpath = a;
        return ret;
    case 'w':
        opts->cgen_builder = 1;
        return noarg;
    case 'v':
        opts->cgen_verifier = 1;
        return noarg;
    case 'c':
        opts->cgen_common_reader = 1;
        opts->cgen_common_builder = 1;
        return noarg;
    case 'r':
        opts->cgen_recursive = 1;
        return noarg;
    case 'g':
        opts->cgen_no_conflicts = 1;
        return noarg;
    case 's':
        opts->silence = 1;
        return noarg;
    case 'd':
        opts->gen_dep = 1;
        return noarg;
    case 'a':
        opts->cgen_reader = 1;
        opts->cgen_builder = 1;
        opts->cgen_verifier = 1;
        opts->cgen_common_reader = 1;
        opts->cgen_common_builder = 1;
        opts->cgen_recursive = 1;
        return noarg;
    default:
        fprintf(stderr, "invalid option: -%c\n", *s);
        exit(-1);
    }
    return noarg;
}

int get_opt(flatcc_options_t *opts, const char *s, const char *a)
{
    if (s[1] == '-') {
        return nextarg == set_opt(opts, s + 1, a);
    }
    ++s;
    if (*s == 0) {
        fprintf(stderr, "- is not a valid option\n");
        exit(-1);
    }
    while (*s) {
        switch (set_opt(opts, s, a)) {
        case noarg:
            ++s;
            continue;
        case suffixarg:
            return 0;
        case nextarg:
            return 1;
        }
    }
    return noarg;
}

void parse_opts(int argc, const char *argv[], flatcc_options_t *opts)
{
    int i;
    const char *s, *a;

    for (i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            s = argv[i];
            a = i + 1 < argc ? argv[i + 1] : 0;
            i += get_opt(opts, s, a);
        } else {
            opts->srcpaths[opts->srcpath_count++] = argv[i];
        }
    }
}

int main(int argc, const char *argv[])
{
    flatcc_options_t opts;
    flatcc_context_t ctx = 0;
    int i, ret;
    const char **src;

    ctx = 0;
    ret = 0;
    if (argc < 2) {
        usage(stderr);
        exit(-1);
    }
    flatcc_init_options(&opts);
    if (!(opts.inpaths = malloc((size_t)argc * sizeof(char *)))) {
        fprintf(stderr, "memory allocation failure\n");
        exit(-1);
    }
    if (!(opts.srcpaths = malloc((size_t)argc * sizeof(char *)))) {
        fprintf(stderr, "memory allocation failure\n");
        free((void *)opts.inpaths);
        exit(-1);
    }

    parse_opts(argc, argv, &opts);
    if (opts.cgen_builder && opts.cgen_common_reader) {
        opts.cgen_common_builder = 1;
    }
    if (opts.srcpath_count == 0) {
        /* No input files, so only generate header(s). */
        if (!(opts.cgen_common_reader || opts.cgen_common_builder) || opts.bgen_bfbs) {
            fprintf(stderr, "filename missing\n");
            goto fail;
        }
        if (!(ctx = flatcc_create_context(&opts, 0, 0, 0))) {
            fprintf(stderr, "internal error: failed to create parsing context\n");
            goto fail;
        }
        if (flatcc_generate_files(ctx)) {
            goto fail;
        }
        flatcc_destroy_context(ctx);
        ctx = 0;
        goto done;
    }
    opts.cgen = opts.cgen_reader || opts.cgen_builder || opts.cgen_verifier
        || opts.cgen_common_reader || opts.cgen_common_builder
        || opts.cgen_json_parser || opts.cgen_json_printer;
    if (!opts.bgen_bfbs && (!opts.cgen || opts.cgen_builder || opts.cgen_verifier)) {
        /* Assume default if no other output specified when deps required it. */
        opts.cgen_reader = 1;
        opts.cgen = 1;
    }
    if (opts.bgen_bfbs && opts.cgen) {
        if (opts.gen_stdout) {
            fprintf(stderr, "--stdout cannot be used with mixed text and binary output\n");
            goto fail;
        }
        if (opts.gen_outfile) {
            fprintf(stderr, "--outfile cannot be used with mixed text and binary output\n");
            goto fail;
        }
    }
    if (opts.gen_deptarget && !opts.gen_depfile) {
        fprintf(stderr, "--deptarget cannot be used without --depfile\n");
        goto fail;
    }
    if (opts.gen_stdout && opts.gen_outfile) {
        fprintf(stderr, "--outfile cannot be used with --stdout\n");
        goto fail;
    }
    for (i = 0, src = opts.srcpaths; i < opts.srcpath_count; ++i, ++src) {
        if (!(ctx = flatcc_create_context(&opts, *src, 0, 0))) {
            fprintf(stderr, "internal error: failed to create parsing context\n");
            goto fail;
        }
        if (flatcc_parse_file(ctx, *src)) {
            goto fail;
        }
        if (flatcc_generate_files(ctx)) {
            goto fail;
        }
        flatcc_destroy_context(ctx);
        ctx = 0;
        /* for --stdout and --outfile options: append to file and skip generating common headers. */
        opts.gen_append = 1;
    }
    goto done;
fail:
    ret = -1;
done:
    if (ctx) {
        flatcc_destroy_context(ctx);
        ctx = 0;
    }
    if (ret) {
        fprintf(stderr, "output failed\n");
    }
    free((void *)opts.inpaths);
    free((void *)opts.srcpaths);
    return ret;
}
