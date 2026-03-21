#include <assert.h>
#include "config.h"
#include "parser.h"
#include "semantics.h"
#include "fileio.h"
#include "codegen.h"
#include "flatcc/flatcc.h"

#define checkfree(s) if (s) { free(s); s = 0; }

void flatcc_init_options(flatcc_options_t *opts)
{
    memset(opts, 0, sizeof(*opts));

    opts->max_schema_size = FLATCC_MAX_SCHEMA_SIZE;
    opts->max_include_depth = FLATCC_MAX_INCLUDE_DEPTH;
    opts->max_include_count = FLATCC_MAX_INCLUDE_COUNT;
    opts->allow_boolean_conversion = FLATCC_ALLOW_BOOLEAN_CONVERSION;
    opts->allow_enum_key = FLATCC_ALLOW_ENUM_KEY;
    opts->allow_enum_struct_field = FLATCC_ALLOW_ENUM_STRUCT_FIELD;
    opts->allow_multiple_key_fields = FLATCC_ALLOW_MULTIPLE_KEY_FIELDS;
    opts->allow_primary_key = FLATCC_ALLOW_PRIMARY_KEY;
    opts->allow_scan_for_all_fields = FLATCC_ALLOW_SCAN_FOR_ALL_FIELDS;
    opts->allow_string_key = FLATCC_ALLOW_STRING_KEY;
    opts->allow_struct_field_deprecate = FLATCC_ALLOW_STRUCT_FIELD_DEPRECATE;
    opts->allow_struct_field_key = FLATCC_ALLOW_STRUCT_FIELD_KEY;
    opts->allow_struct_root = FLATCC_ALLOW_STRUCT_ROOT;
    opts->ascending_enum = FLATCC_ASCENDING_ENUM;
    opts->hide_later_enum = FLATCC_HIDE_LATER_ENUM;
    opts->hide_later_struct = FLATCC_HIDE_LATER_STRUCT;
    opts->offset_size = FLATCC_OFFSET_SIZE;
    opts->voffset_size = FLATCC_VOFFSET_SIZE;
    opts->utype_size = FLATCC_UTYPE_SIZE;
    opts->bool_size = FLATCC_BOOL_SIZE;

    opts->require_root_type = FLATCC_REQUIRE_ROOT_TYPE;
    opts->strict_enum_init = FLATCC_STRICT_ENUM_INIT;
    /*
     * Index 0 is table elem count, and index 1 is table size
     * so max count is reduced by 2, meaning field id's
     * must be between 0 and vt_max_count - 1.
     * Usually, the table is 16-bit, so FLATCC_VOFFSET_SIZE = 2.
     * Strange expression to avoid shift overflow on 64 bit size.
     */
    opts->vt_max_count = ((1LL << (FLATCC_VOFFSET_SIZE * 8 - 1)) - 1) * 2;

    opts->default_schema_ext = FLATCC_DEFAULT_SCHEMA_EXT;
    opts->default_bin_schema_ext = FLATCC_DEFAULT_BIN_SCHEMA_EXT;
    opts->default_bin_ext = FLATCC_DEFAULT_BIN_EXT;

    opts->cgen_no_conflicts = FLATCC_CGEN_NO_CONFLICTS;

    opts->cgen_pad = FLATCC_CGEN_PAD;
    opts->cgen_sort = FLATCC_CGEN_SORT;
    opts->cgen_pragmas = FLATCC_CGEN_PRAGMAS;

    opts->cgen_common_reader = 0;
    opts->cgen_common_builder = 0;
    opts->cgen_reader = 0;
    opts->cgen_builder = 0;
    opts->cgen_json_parser = 0;
    opts->cgen_spacing = FLATCC_CGEN_SPACING;

    opts->bgen_bfbs = FLATCC_BGEN_BFBS;
    opts->bgen_qualify_names = FLATCC_BGEN_QUALIFY_NAMES;
    opts->bgen_length_prefix = FLATCC_BGEN_LENGTH_PREFIX;
}

flatcc_context_t flatcc_create_context(flatcc_options_t *opts, const char *name,
        flatcc_error_fun error_out, void *error_ctx)
{
    fb_parser_t *P;

    if (!(P = malloc(sizeof(*P)))) {
        return 0;
    }
    if (fb_init_parser(P, opts, name, error_out, error_ctx, 0)) {
        free(P);
        return 0;
    }
    return P;
}

static flatcc_context_t __flatcc_create_child_context(flatcc_options_t *opts, const char *name,
        fb_parser_t *P_parent)
{
    fb_parser_t *P;

    if (!(P = malloc(sizeof(*P)))) {
        return 0;
    }
    if (fb_init_parser(P, opts, name, P_parent->error_out, P_parent->error_ctx, P_parent->schema.root_schema)) {
        free(P);
        return 0;
    }
    return P;
}

/* TODO: handle include files via some sort of buffer read callback
 * and possible transfer file based parser to this logic. */
int flatcc_parse_buffer(flatcc_context_t ctx, const char *buf, size_t buflen)
{
    fb_parser_t *P = ctx;

    /* Currently includes cannot be handled by buffers, so they should done. */
    P->opts.disable_includes = 1;
    if ((size_t)buflen > P->opts.max_schema_size && P->opts.max_schema_size > 0) {
        fb_print_error(P, "input exceeds maximum allowed size\n");
        return -1;
    }
    /* Add self to set of visible schema. */
    ptr_set_insert_item(&P->schema.visible_schema, &P->schema, ht_keep);
    return fb_parse(P, buf, buflen, 0) || fb_build_schema(P) ? -1 : 0;
}

static void visit_dep(void *context, void *ptr)
{
    fb_schema_t *parent = context;
    fb_schema_t *dep = ptr;

    ptr_set_insert_item(&parent->visible_schema, dep, ht_keep);
}

static void add_visible_schema(fb_schema_t *parent, fb_schema_t *dep)
{
    ptr_set_visit(&dep->visible_schema, visit_dep, parent);
}

static int __parse_include_file(fb_parser_t *P_parent, const char *filename)
{
    flatcc_context_t *ctx = 0;
    fb_parser_t *P = 0;
    fb_root_schema_t *rs;
    flatcc_options_t *opts = &P_parent->opts;
    fb_schema_t *dep;

    rs = P_parent->schema.root_schema;
    if (rs->include_depth >= opts->max_include_depth && opts->max_include_depth > 0) {
        fb_print_error(P_parent, "include nesting level too deep\n");
        return -1;
    }
    if (rs->include_count >= opts->max_include_count && opts->max_include_count > 0) {
        fb_print_error(P_parent, "include count limit exceeded\n");
        return -1;
    }
    if (!(ctx = __flatcc_create_child_context(opts, filename, P_parent))) {
        return -1;
    }
    P = (fb_parser_t *)ctx;
    /* Don't parse the same file twice, or any other file with same name. */
    if ((dep = fb_schema_table_find_item(&rs->include_index, &P->schema))) {
        add_visible_schema(&P_parent->schema, dep);
        flatcc_destroy_context(ctx);
        return 0;
    }
    P->dependencies = P_parent->dependencies;
    P_parent->dependencies = P;
    P->referer_path = P_parent->path;
    /* Each parser has a root schema instance, but only the root parsers instance is used. */
    rs->include_depth++;
    rs->include_count++;
    if (flatcc_parse_file(ctx, filename)) {
        return -1;
    }
    add_visible_schema(&P_parent->schema, &P->schema);
    return 0;
}

/*
 * The depends file format is a make rule:
 *
 * <outputfile> : <dep1-file> <dep2-file> ...
 *
 * like -MMD option for gcc/clang:
 * lib.o.d generated with content:
 *
 * lib.o : header1.h header2.h
 *
 * We use a file name <basename>.depends for schema <basename>.fbs with content:
 *
 * <basename>_reader.h : <included-schema-1> ...
 *
 * The .d extension could mean the D language and we don't have sensible
 * .o.d name because of multiple outputs, so .depends is better.
 *
 * (the above above is subject to the configuration of extensions).
 *
 * TODO:
 * perhaps we should optionally add a dependency to the common reader
 * and builder files when they are generated separately as they should in
 * concurrent builds.
 *
 * TODO:
 * 1. we should have a file for every output we produce (_builder.h * etc.)
 * 2. reader might not even be in the output, e.g. verifier only.
 * 3. multiple outputs doesn't work with ninja build 1.7.1, so just
 *    use reader for now, and possible add an option for multiple
 *    outputs later.
 *
 *  http://stackoverflow.com/questions/11855386/using-g-with-mmd-in-makefile-to-automatically-generate-dependencies
 *  https://groups.google.com/forum/#!topic/ninja-build/j-2RfBIOd_8
 *  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=47485
 *
 *  Spaces in gnu make:
 *  https://www.cmcrossroads.com/article/gnu-make-meets-file-names-spaces-them
 *  See comments on gnu make handling of spaces.
 *  http://clang.llvm.org/doxygen/DependencyFile_8cpp_source.html
 */
static int __flatcc_gen_depends_file(fb_parser_t *P)
{
    FILE *fp = 0;
    const char *outpath, *basename; 
    const char *depfile, *deproot, *depext;
    const char *targetfile, *targetsuffix, *targetroot;
    char *path = 0, *deppath = 0, *tmppath = 0, *targetpath = 0; 
    int ret = -1;

    /*
     * The dependencies list is only correct for root files as it is a
     * linear list. To deal with children, we would have to filter via
     * the visible schema hash table, but we don't really need that.
     */
    assert(P->referer_path == 0);

    outpath = P->opts.outpath ? P->opts.outpath : "";
    basename = P->schema.basename;
    targetfile = P->opts.gen_deptarget;


    /* The following is mostly considering build tools generating
     * a depfile as Ninja build would use it. It is a bit strict
     * on path variations and currenlty doesn't accept multiple
     * build products in a build rule (Ninja 1.7.1).
     *
     * Make depfile relative to cwd so the user can add output if
     * needed, otherwise it is not possible, or difficult, to use a path given
     * by a build tool, relative the cwd. If --depfile is not given,
     * then -d is given or we would not be here. In that case we add an
     * extension "<basename>.fbs.d" in the outpath.
     *
     * A general problem is that the outpath may be a build root dir or
     * a current subdir for a custom build rule while the dep file
     * content needs the same path every time, not just an equivalent
     * path. For dependencies, we can rely on the input schema path.
     * The input search paths may because confusion but we choose the
     * discovered path relative to cwd consistently for each schema file
     * encountered.
     *
     * The target file (<target>: <include1.fbs> <include2.fbs> ...)
     * is tricky because it is not unique - but we can chose <schema>_reader.h
     * or <schema>.bfbs prefixed with outpath. The user should choose an
     * outpath relative to cwd or an absolute path depending on what the
     * build system prefers. This may not be so easy in praxis, but what
     * can we do?
     *
     * It is important to note the default target and the default
     * depfile name is not just a convenience. Sometimes it is much
     * simpler to use this version over an explicit path, sometimes
     * perhaps not so much.
     */

    if (P->opts.gen_depfile) {
        depfile = P->opts.gen_depfile;
        deproot = "";
        depext = "";
    } else {
        depfile = basename;
        deproot = outpath;
        depext = FLATCC_DEFAULT_DEP_EXT;
    }
    if (targetfile) {
        targetsuffix = "";
        targetroot = "";
    } else {
        targetsuffix = P->opts.bgen_bfbs 
                ? FLATCC_DEFAULT_BIN_SCHEMA_EXT 
                : FLATCC_DEFAULT_DEP_TARGET_SUFFIX;
        targetfile = basename;
        targetroot = outpath;
    }

    checkmem(path = fb_create_join_path(deproot, depfile, depext, 1));

    checkmem(tmppath = fb_create_join_path(targetroot, targetfile, targetsuffix, 1));
    /* Handle spaces in dependency file. */
    checkmem((targetpath = fb_create_make_path(tmppath)));
    checkfree(tmppath);

    fp = fopen(path, "wb");
    if (!fp) {
        fb_print_error(P, "could not open dependency file for output: %s\n", path);
        goto done;
    }
    fprintf(fp, "%s:", targetpath);

    /* Don't depend on root schema. */
    P = P->dependencies;
    while (P) {
        checkmem((deppath = fb_create_make_path(P->path)));
        fprintf(fp, " %s", deppath);
        P = P->dependencies;
        checkfree(deppath);
    }
    fprintf(fp, "\n");
    ret = 0;

done:
    checkfree(path);
    checkfree(tmppath);
    checkfree(targetpath);
    checkfree(deppath);
    if (fp) {
        fclose(fp);
    }
    return ret;
}

int flatcc_parse_file(flatcc_context_t ctx, const char *filename)
{
    fb_parser_t *P = ctx;
    size_t inpath_len, filename_len;
    char *buf, *path, *include_file;
    const char *inpath;
    size_t size;
    fb_name_t *inc;
    int i, ret, is_root;

    filename_len = strlen(filename);
    /* Don't parse the same file twice, or any other file with same basename. */
    if (fb_schema_table_insert_item(&P->schema.root_schema->include_index, &P->schema, ht_keep)) {
        return 0;
    }
    buf = 0;
    path = 0;
    include_file = 0;
    ret = -1;
    is_root = !P->referer_path;

    /*
     * For root files, read file relative to working dir first. For
     * included files (`referer_path` set), first try include paths
     * in order, then path relative to including file.
     */
    if (is_root) {
        if (!(buf = fb_read_file(filename, P->opts.max_schema_size, &size))) {
            if (size + P->schema.root_schema->total_source_size > P->opts.max_schema_size && P->opts.max_schema_size > 0) {
                fb_print_error(P, "input exceeds maximum allowed size\n");
                goto done;
            }
        } else {
            checkmem((path = fb_copy_path(filename)));
        }
    }
    for (i = 0; !buf && i < P->opts.inpath_count; ++i) {
        inpath = P->opts.inpaths[i];
        inpath_len = strlen(inpath);
        checkmem((path = fb_create_join_path_n(inpath, inpath_len, filename, filename_len, "", 1)));
        if (!(buf = fb_read_file(path, P->opts.max_schema_size, &size))) {
            free(path);
            path = 0;
            if (size > P->opts.max_schema_size && P->opts.max_schema_size > 0) {
                fb_print_error(P, "input exceeds maximum allowed size\n");
                goto done;
            }
        }
    }
    if (!buf && !is_root) {
        inpath = P->referer_path;
        inpath_len = fb_find_basename(inpath, strlen(inpath));
        checkmem((path = fb_create_join_path_n(inpath, inpath_len, filename, filename_len, "", 1)));
        if (!(buf = fb_read_file(path, P->opts.max_schema_size, &size))) {
            free(path);
            path = 0;
            if (size > P->opts.max_schema_size && P->opts.max_schema_size > 0) {
                fb_print_error(P, "input exceeds maximum allowed size\n");
                goto done;
            }
        }
    }
    if (!buf) {
        fb_print_error(P, "error reading included schema file: %s\n", filename);
        goto done;
    }
    P->schema.root_schema->total_source_size += size;
    P->path = path;
    /* Parser owns path. */
    path = 0;
    /*
     * Even if we do not have the recursive option set, we still
     * need to parse all include files to make sense of the current
     * file.
     */
    if (!fb_parse(P, buf, size, 1)) {
        /* Parser owns buffer. */
        buf = 0;
        inc = P->schema.includes;
        while (inc) {
            checkmem((include_file = fb_copy_path_n(inc->name.s.s, (size_t)inc->name.s.len)));
            if (__parse_include_file(P, include_file)) {
                goto done;
            }
            free(include_file);
            include_file = 0;
            inc = inc->link;
        }
        /* Add self to set of visible schema. */
        ptr_set_insert_item(&P->schema.visible_schema, &P->schema, ht_keep);
        if (fb_build_schema(P)) {
            goto done;
        }
        /*
        * We choose to only generate optional .depends files for root level
        * files. These will contain all nested files regardless of
        * recursive file generation flags.
        */
        if (P->opts.gen_dep && is_root) {
            if (__flatcc_gen_depends_file(P)) {
                goto done;
            }
        }
        ret = 0;
    }

done:
    /* Parser owns buffer so don't free it here. */
    checkfree(path);
    checkfree(include_file);
    return ret;
}

#if FLATCC_REFLECTION
int flatcc_generate_binary_schema_to_buffer(flatcc_context_t ctx, void *buf, size_t bufsiz)
{
    fb_parser_t *P = ctx;

    if (fb_codegen_bfbs_to_buffer(&P->opts, &P->schema, buf, &bufsiz)) {
        return (int)bufsiz;
    }
    return -1;
}

void *flatcc_generate_binary_schema(flatcc_context_t ctx, size_t *size)
{
    fb_parser_t *P = ctx;

    return fb_codegen_bfbs_alloc_buffer(&P->opts, &P->schema, size);
}
#endif

int flatcc_generate_files(flatcc_context_t ctx)
{
    fb_parser_t *P = ctx, *P_leaf;
    fb_output_t *out, output;
    int ret = 0;
    out = &output;

    if (!P || P->failed) {
        return -1;
    }
    P_leaf = 0;
    while (P) {
        P->inverse_dependencies = P_leaf;
        P_leaf = P;
        P = P->dependencies;
    }
    P = ctx;
#if FLATCC_REFLECTION
    if (P->opts.bgen_bfbs) {
        if (fb_codegen_bfbs_to_file(&P->opts, &P->schema)) {
            return -1;
        }
    }
#endif
#ifdef PR216
    /* Note: P->opts.cgen now represents the below expression, and bfbs_gen is mutually exclusive with cgen */
    if (!(P->opts.cgen_reader | P->opts.cgen_builder | P->opts.cgen_verifier
        | P->opts.cgen_common_reader | P->opts.cgen_common_builder
        | P->opts.cgen_json_parser | P->opts.cgen_json_printer)) {
        return 0;
    }
#else
    /* Experimental alternative to PR216 */
    if (P->opts.bgen_bfbs) {
        return 0;
    }
#endif
    if (fb_init_output_c(out, &P->opts)) {
        return -1;
    }
    /* This does not require a parse first. */
    if (!P->opts.gen_append && (ret = fb_codegen_common_c(out))) {
        goto done;
    }
    /* If no file parsed - just common files if at all. */
    if (!P->has_schema) {
        goto done;
    }
    if (!P->opts.cgen_recursive) {
        ret = fb_codegen_c(out, &P->schema);
        goto done;
    }
    /* Make sure stdout and outfile output is generated in the right order. */
    P = P_leaf;
    while (!ret && P) {
        ret = P->failed || fb_codegen_c(out, &P->schema);
        P = P->inverse_dependencies;
    }
done:
    fb_end_output_c(out);
    return ret;
}

void flatcc_destroy_context(flatcc_context_t ctx)
{
    fb_parser_t *P = ctx, *dep = 0;

    while (P) {
        dep = P->dependencies;
        fb_clear_parser(P);
        free(P);
        P = dep;
    }
}
