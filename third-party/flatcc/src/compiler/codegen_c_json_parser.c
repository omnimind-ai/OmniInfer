#include <stdlib.h>
#include "codegen_c.h"
#include "flatcc/flatcc_types.h"
#include "catalog.h"

/* -DFLATCC_PORTABLE may help if inttypes.h is missing. */
#ifndef PRId64
#include <inttypes.h>
#endif

#define PRINTLN_SPMAX 64
static char println_spaces[PRINTLN_SPMAX];

static void println(fb_output_t *out, const char * format, ...)
{
    int i = out->indent * out->opts->cgen_spacing;
    va_list ap;

    if (println_spaces[0] == 0) {
        memset(println_spaces, 0x20, PRINTLN_SPMAX);
    }
    /* Don't indent on blank lines. */
    if (*format) {
        while (i > PRINTLN_SPMAX) {
            fprintf(out->fp, "%.*s", (int)PRINTLN_SPMAX, println_spaces);
            i -= PRINTLN_SPMAX;
        }
        /* Use modulo to reset margin if we go too far. */
        fprintf(out->fp, "%.*s", i, println_spaces);
        va_start (ap, format);
        vfprintf (out->fp, format, ap);
        va_end (ap);
    }
    fprintf(out->fp, "\n");
}

/*
 * Unknown fields and unknown union members can be failed
 * rather than ignored with a config flag.
 *
 * Default values an be forced with a config flat.
 *
 * Forward schema isn't perfect: Unknown symbolic constants
 * cannot be used with known fields but will be ignored
 * in ignored fields.
 */

static int gen_json_parser_pretext(fb_output_t *out)
{
    println(out, "#ifndef %s_JSON_PARSER_H", out->S->basenameup);
    println(out, "#define %s_JSON_PARSER_H", out->S->basenameup);
    println(out, "");
    println(out, "/* " FLATCC_GENERATED_BY " */");
    println(out, "");
    println(out, "#include \"flatcc/flatcc_json_parser.h\"");
    fb_gen_c_includes(out, "_json_parser.h", "_JSON_PARSER_H");
    gen_prologue(out);
    println(out, "");
    return 0;
}

static int gen_json_parser_footer(fb_output_t *out)
{
    gen_epilogue(out);
    println(out, "#endif /* %s_JSON_PARSER_H */", out->S->basenameup);
    return 0;
}

typedef struct dict_entry dict_entry_t;
struct dict_entry {
    const char *text;
    int len;
    void *data;
    int hint;
};

/* Returns length of name that reminds after tag at current position. */
static int get_dict_suffix_len(dict_entry_t *de, int pos)
{
    int n;

    n = de->len;
    if (pos + 8 > n) {
        return 0;
    }
    return n - pos - 8;
}

/*
 * Returns the length name that reminds if it terminates at the tag
 * and 0 if it has a suffix.
 */
static int get_dict_tag_len(dict_entry_t *de, int pos)
{
    int n;

    n = de->len;
    if (pos + 8 >= n) {
        return n - pos;
    }
    return 0;
}

/*
 * 8 byte word part of the name starting at characert `pos` in big
 * endian encoding with first char always at msb, zero padded at lsb.
 * Returns length of tag [0;8].
 */
static int get_dict_tag(dict_entry_t *de, int pos, uint64_t *tag, uint64_t *mask,
        const char **tag_name, int *tag_len)
{
    int i, n = 0;
    const char *a = 0;
    uint64_t w = 0;

    if (pos > de->len) {
        goto done;
    }
    a = de->text + pos;
    n = de->len - pos;
    if (n > 8) {
        n = 8;
    }
    i = n;
    while (i--) {
        w |= ((uint64_t)a[i]) << (56 - (i * 8));
    }
    *tag = w;
    *mask = ~(((uint64_t)(1) << (8 - n) * 8) - 1);
done:
    if (tag_name) {
        *tag_name = a;
    }
    if (tag_len) {
        *tag_len = n;
    }
    return n;
}


/*
 * Find the median, but move earlier if the previous entry
 * is a strict prefix within the range.
 *
 * `b` is inclusive.
 *
 * The `pos` is a window into the key at an 8 byte multiple.
 *
 * Only consider the range `[pos;pos+8)` and move the median
 * up if an earlier key is a prefix or match within this
 * window. This is needed to handle trailing data in
 * a compared external key, and also to handle sub-tree
 * branching when two keys has same tag at pos.
 *
 * Worst case we get a linear search of length 8 if all
 * keys are perfect prefixes of their successor key:
 *   `a, ab, abc, ..., abcdefgh`
 * While the midpoint stills seeks towards 'a' for longer
 * such sequences, the branch logic will pool those
 * squences the share prefix groups of length 8.
 */
static int split_dict_left(dict_entry_t *dict, int a, int b, int pos)
{
    int m = a + (b - a) / 2;
    uint64_t wf = 0, wg = 0, wmf = 0, wmg = 0;

    while (m > a) {
        get_dict_tag(&dict[m - 1], pos, &wf, &wmf, 0, 0);
        get_dict_tag(&dict[m], pos, &wg, &wmg, 0, 0);
        if (((wf ^ wg) & wmf) != 0) {
            return m;
        }
        --m;
    }
    return m;
}

/*
 * When multiple tags are identical after split_dict_left has moved
 * intersection up so a == m, we need to split in the opposite direction
 * to ensure progress untill all tags in the range are identical
 * at which point the trie must descend.
 *
 * If all tags are the same from intersection to end, b + 1 is returned
 * which is not a valid element.
 */
static int split_dict_right(dict_entry_t *dict, int a, int b, int pos)
{
    int m = a + (b - a) / 2;
    uint64_t wf = 0, wg = 0, wmf = 0, wmg = 0;

    while (m < b) {
        get_dict_tag(&dict[m], pos, &wf, &wmf, 0, 0);
        get_dict_tag(&dict[m + 1], pos, &wg, &wmg, 0, 0);
        if (((wf ^ wg) & wmf) != 0) {
            return m + 1;
        }
        ++m;
    }
    return m + 1;
}

/*
 * Returns the first index where the tag does not terminate at
 * [pos..pos+7], or b + 1 if none exists.
 */
static int split_dict_descend(dict_entry_t *dict, int a, int b, int pos)
{
    while (a <= b) {
        if (0 < get_dict_suffix_len(&dict[a], pos)) {
            break;
        }
        ++a;
    }
    return a;
}


static int dict_cmp(const void *x, const void *y)
{
    const dict_entry_t *a = x, *b = y;
    int k, n = a->len > b->len ? b->len : a->len;

    k = memcmp(a->text, b->text, (size_t)n);
    return k ? k : a->len - b->len;
}

/* Includes union vectors. */
static inline int is_union_member(fb_member_t *member)
{
    return (member->type.type == vt_compound_type_ref || member->type.type == vt_vector_compound_type_ref)
            && member->type.ct->symbol.kind == fb_is_union;
}

static dict_entry_t *build_compound_dict(fb_compound_type_t *ct, int *count_out)
{
    fb_symbol_t *sym;
    fb_member_t *member;
    size_t n;
    dict_entry_t *dict, *de;
    char *strbuf = 0;
    size_t strbufsiz = 0;
    int is_union;
    size_t union_index = 0;

    n = 0;
    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        if (member->metadata_flags & fb_f_deprecated) {
            continue;
        }
        is_union = is_union_member(member);
        if (is_union) {
            ++n;
            strbufsiz += (size_t)member->symbol.ident->len + 6;
        }
        ++n;
    }
    *count_out = (int)n;
    if (n == 0) {
        return 0;
    }
    dict = malloc(n * sizeof(dict_entry_t) + strbufsiz);
    if (!dict) {
        return 0;
    }
    strbuf = (char *)dict + n * sizeof(dict_entry_t);
    de = dict;
    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        if (member->metadata_flags & fb_f_deprecated) {
            continue;
        }
        de->text = member->symbol.ident->text;
        de->len = (int)member->symbol.ident->len;
        de->data = member;
        de->hint = 0;
        ++de;
        is_union = is_union_member(member);
        if (is_union) {
            member->export_index = union_index++;
            de->len = (int)member->symbol.ident->len + 5;
            de->text = strbuf;
            memcpy(strbuf, member->symbol.ident->text, (size_t)member->symbol.ident->len);
            strbuf += member->symbol.ident->len;
            strcpy(strbuf, "_type");
            strbuf += 6;
            de->data = member;
            de->hint = 1;
            ++de;
        }
    }
    qsort(dict, n, sizeof(dict[0]), dict_cmp);
    return dict;
}

typedef struct {
    int count;
    fb_schema_t *schema;
    dict_entry_t *de;
} install_enum_context_t;

static void count_visible_enum_symbol(void *context, fb_symbol_t *sym)
{
    install_enum_context_t *p = context;

    if (get_enum_if_visible(p->schema, sym)) {
        p->count++;
    }
}

static void install_visible_enum_symbol(void *context, fb_symbol_t *sym)
{
    install_enum_context_t *p = context;

    if (get_enum_if_visible(p->schema, sym)) {
        p->de->text = sym->ident->text;
        p->de->len = (int)sym->ident->len;
        p->de->data = sym;
        p->de++;
    }
}

/*
 * A scope dictionary contains all the enum types defined under the given
 * namespace of the scope. The actually namespace is not contained in
 * the name - it is an implicit prefix. It is used when looking up a
 * symbolic constant assigned to a field such that the constant is first
 * searched for in the same scope (namespace) as the one that defined
 * the table owning the field assigned to. If that fails, a global
 * namespace prefixed lookup is needed, but this is separate from this
 * dictionary. In case of conflicts the local scope takes precedence
 * and must be searched first. Because each table parsed can a have a
 * unique local scope, we cannot install the the unprefixed lookup in
 * the same dictionary as the global lookup.
 *
 * NOTE: the scope may have been contanimated by being expanded by a
 * parent schema so we check that each symbol is visible to the current
 * schema. If we didn't do this, we would risk referring to enum parsers
 * that are not included in the generated source. The default empty
 * namespace (i.e. scope) is an example where this easily could happen.
 */
static dict_entry_t *build_local_scope_dict(fb_schema_t *schema, fb_scope_t *scope, int *count_out)
{
    dict_entry_t *dict;
    install_enum_context_t iec;

    fb_clear(iec);

    iec.schema = schema;

    fb_symbol_table_visit(&scope->symbol_index, count_visible_enum_symbol, &iec);
    *count_out = iec.count;

    if (iec.count == 0) {
        return 0;
    }
    dict = malloc((size_t)iec.count * sizeof(dict[0]));
    if (!dict) {
        return 0;
    }
    iec.de = dict;
    fb_symbol_table_visit(&scope->symbol_index, install_visible_enum_symbol, &iec);
    qsort(dict, (size_t)iec.count, sizeof(dict[0]), dict_cmp);
    return dict;
}

static dict_entry_t *build_global_scope_dict(catalog_t *catalog, int *count_out)
{
    size_t i, n = (size_t)catalog->nenums;
    dict_entry_t *dict;

    *count_out = (int)n;
    if (n == 0) {
        return 0;
    }
    dict = malloc(n * sizeof(dict[0]));
    if (!dict) {
        return 0;
    }
    for (i = 0; i < (size_t)catalog->nenums; ++i) {
        dict[i].text = catalog->enums[i].name;
        dict[i].len = (int)strlen(catalog->enums[i].name);
        dict[i].data = catalog->enums[i].ct;
        dict[i].hint = 0;
    }
    qsort(dict, (size_t)catalog->nenums, sizeof(dict[0]), dict_cmp);
    *count_out = catalog->nenums;
    return dict;
}

static void clear_dict(dict_entry_t *dict)
{
    if (dict) {
        free(dict);
    }
}

static int gen_field_match_handler(fb_output_t *out, fb_compound_type_t *ct, void *data, int is_union_type)
{
    fb_member_t *member = data;
    fb_scoped_name_t snref;
    fb_symbol_text_t scope_name;

    int is_struct_container;
    int is_string = 0;
    int is_enum = 0;
    int is_vector = 0;
    int is_offset = 0;
    int is_scalar = 0;
    int is_optional = 0;
    int is_table = 0;
    int is_struct = 0;
    int is_union = 0;
    int is_union_vector = 0;
    int is_union_type_vector = 0;
    int is_base64 = 0;
    int is_base64url = 0;
    int is_nested = 0;
    int is_array = 0;
    int is_char_array = 0;
    size_t array_len = 0;
    fb_scalar_type_t st = 0;
    const char *tname_prefix = "n/a", *tname = "n/a"; /* suppress compiler warnigns */
    fb_literal_t literal;

    fb_clear(snref);

    fb_copy_scope(ct->scope, scope_name);
    is_struct_container = ct->symbol.kind == fb_is_struct;
    is_optional = !!(member->flags & fb_fm_optional);

    switch (member->type.type) {
    case vt_vector_type:
    case vt_vector_compound_type_ref:
    case vt_vector_string_type:
        is_vector = 1;
        break;
    }

    switch (member->type.type) {
    case vt_fixed_array_compound_type_ref:
    case vt_vector_compound_type_ref:
    case vt_compound_type_ref:
        fb_compound_name(member->type.ct, &snref);
        is_enum = member->type.ct->symbol.kind == fb_is_enum;
        is_struct = member->type.ct->symbol.kind == fb_is_struct;
        is_table = member->type.ct->symbol.kind == fb_is_table;
        is_union = member->type.ct->symbol.kind == fb_is_union && !is_union_type;
        if (is_enum) {
            st = member->type.ct->type.st;
            is_scalar = 1;
        }
        break;
    case vt_vector_string_type:
    case vt_string_type:
        is_string = 1;
        break;
    case vt_vector_type:
        /* Nested types are processed twice, once as an array, once as an object. */
        is_nested = member->nest != 0;
        is_base64 = member->metadata_flags & fb_f_base64;
        is_base64url = member->metadata_flags & fb_f_base64url;
        is_scalar = 1;
        st = member->type.st;
        break;
    case vt_fixed_array_type:
        is_scalar = 1;
        is_array = 1;
        array_len = member->type.len;
        st = member->type.st;
        break;
    case vt_scalar_type:
        is_scalar = 1;
        st = member->type.st;
        break;
    }
    if (member->type.type == vt_fixed_array_compound_type_ref) {
        assert(is_struct_container);
        is_array = 1;
        array_len = member->type.len;
    }
    if (is_base64 || is_base64url) {
        /* Even if it is nested, parse it as a regular base64 or base64url encoded vector. */
        if (st != fb_ubyte || !is_vector) {
            gen_panic(out, "internal error: unexpected base64 or base64url field type\n");
            return -1;
        }
        is_nested = 0;
        is_vector = 0;
        is_scalar = 0;
    }
    if (is_union_type) {
        is_scalar = 0;
    }
    if (is_vector && is_union_type) {
        is_union_type_vector = 1;
        is_vector = 0;
    }
    if (is_vector && is_union) {
        is_union_vector = 1;
        is_vector = 0;
    }
    if (is_array && is_scalar && st == fb_char) {
        is_array = 0;
        is_scalar = 0;
        is_char_array = 1;
    }
    if (is_nested == 1) {
        println(out, "if (buf != end && *buf == '[') { /* begin nested */"); indent();
    }
repeat_nested:
    if (is_nested == 2) {
        unindent(); println(out, "} else { /* nested */"); indent();
        fb_compound_name((fb_compound_type_t *)&member->nest->symbol, &snref);
        if (member->nest->symbol.kind == fb_is_table) {
            is_table = 1;
        } else {
            is_struct = 1;
        }
        is_vector = 0;
        is_scalar = 0;
        println(out, "if (flatcc_builder_start_buffer(ctx->ctx, 0, 0, 0)) goto failed;");
    }
    is_offset = !is_scalar && !is_struct && !is_union_type;

    if (is_scalar) {
        tname_prefix = scalar_type_prefix(st);
        tname = st == fb_bool ? "uint8_t" : scalar_type_name(st);
    }

    /* Other types can also be vector, so we wrap. */
    if (is_vector) {
        if (is_offset) {
            println(out, "if (flatcc_builder_start_offset_vector(ctx->ctx)) goto failed;");
        } else {
            println(out,
                "if (flatcc_builder_start_vector(ctx->ctx, %"PRIu64", %hu, UINT64_C(%"PRIu64"))) goto failed;",
                (uint64_t)member->size, (short)member->align,
                (uint64_t)FLATBUFFERS_COUNT_MAX(member->size));
        }
    }
    if (is_array) {
        if (is_scalar) {
            println(out, "size_t count = %d;", array_len);
            println(out, "%s *base = (%s *)((size_t)struct_base + %"PRIu64");",
                    tname, tname, (uint64_t)member->offset);
        }
        else {
            println(out, "size_t count = %d;", array_len);
            println(out, "void *base = (void *)((size_t)struct_base + %"PRIu64");",
                    (uint64_t)member->offset);
        }
    }
    if (is_char_array) {
        println(out, "char *base = (char *)((size_t)struct_base + %"PRIu64");",
                    (uint64_t)member->offset);
        println(out, "buf = flatcc_json_parser_char_array(ctx, buf, end, base, %d);", array_len);
    }
    if (is_array || is_vector) {
        println(out, "buf = flatcc_json_parser_array_start(ctx, buf, end, &more);");
        /* Note that we reuse `more` which is safe because it is updated at the end of the main loop. */
        println(out, "while (more) {"); indent();
    }
    if (is_scalar) {
        println(out, "%s val = 0;", tname);
        println(out, "static flatcc_json_parser_integral_symbol_f *symbolic_parsers[] = {");
        indent(); indent();
        /*
         * The scope name may be empty when no namespace is used. In that
         * case the global scope is the same, but performance the
         * duplicate doesn't matter.
         */
        if (is_enum) {
            println(out, "%s_parse_json_enum,", snref.text);
            println(out, "%s_local_%sjson_parser_enum,", out->S->basename, scope_name);
            println(out, "%s_global_json_parser_enum, 0 };", out->S->basename);
        } else {
            println(out, "%s_local_%sjson_parser_enum,", out->S->basename, scope_name);
            println(out, "%s_global_json_parser_enum, 0 };", out->S->basename);
        }
        unindent(); unindent();
    }
    /* It is not safe to acquire the pointer before building element table or string. */
    if (is_vector && !is_offset) {
        println(out, "if (!(pval = flatcc_builder_extend_vector(ctx->ctx, 1))) goto failed;");
    }
    if (is_struct_container) {
        if (!is_array && !is_char_array) {
            /* `struct_base` is given as argument to struct parsers. */
            println(out, "pval = (void *)((size_t)struct_base + %"PRIu64");", (uint64_t)member->offset);
        }
    } else if (is_struct && !is_vector) {
        /* Same logic as scalars in tables, but scalars must be tested for default. */
        println(out,
            "if (!(pval = flatcc_builder_table_add(ctx->ctx, %"PRIu64", %"PRIu64", %"PRIu16"))) goto failed;",
            (uint64_t)member->id, (uint64_t)member->size, (uint16_t)member->align);
    }
    if (is_scalar) {
        println(out, "buf = flatcc_json_parser_%s(ctx, (mark = buf), end, &val);", tname_prefix);
        println(out, "if (mark == buf) {"); indent();
        println(out, "buf = flatcc_json_parser_symbolic_%s(ctx, (mark = buf), end, symbolic_parsers, &val);", tname_prefix);
        println(out, "if (buf == mark || buf == end) goto failed;");
        unindent(); println(out, "}");
        if (!is_struct_container && !is_vector && !is_base64 && !is_base64url) {
#if !FLATCC_JSON_PARSE_FORCE_DEFAULTS
            /* We need to create a check for the default value and create a table field if not the default. */
            if (!is_optional) {
                if (!print_literal(st, &member->value, literal)) return -1;
                println(out, "if (val != %s || (ctx->flags & flatcc_json_parser_f_force_add)) {", literal); indent();
            }
#endif
            println(out, "if (!(pval = flatcc_builder_table_add(ctx->ctx, %"PRIu64", %"PRIu64", %hu))) goto failed;",
                    (uint64_t)member->id, (uint64_t)member->size, (short)member->align);
#if !FLATCC_JSON_PARSE_FORCE_DEFAULTS
#endif
        }
        /* For scalars in table field, and in struct container. */
        if (is_array) {
            println(out, "if (count) {"); indent();
            println(out, "%s%s_write_to_pe(base, val);", out->nsc, tname_prefix);
            println(out, "--count;");
            println(out, "++base;");
            unindent(); println(out, "} else if (!(ctx->flags & flatcc_json_parser_f_skip_array_overflow)) {"); indent();
            println(out, "return flatcc_json_parser_set_error(ctx, buf, end, flatcc_json_parser_error_array_overflow);");
            unindent(); println(out, "}");
        } else {
            println(out, "%s%s_write_to_pe(pval, val);", out->nsc, tname_prefix);
        }
        if (!is_struct_container && !is_vector && !(is_scalar && is_optional)) {
            unindent(); println(out, "}");
        }
    } else if (is_struct) {
        if (is_array) {
            println(out, "if (count) {"); indent();
            println(out, "buf = %s_parse_json_struct_inline(ctx, buf, end, base);", snref.text);
            println(out, "--count;");
            println(out, "base = (void *)((size_t)base + %"PRIu64");", member->type.ct->size);
            unindent(); println(out, "} else if (!(ctx->flags & flatcc_json_parser_f_skip_array_overflow)) {"); indent();
            println(out, "return flatcc_json_parser_set_error(ctx, buf, end, flatcc_json_parser_error_array_overflow);");
            unindent(); println(out, "}");
        } else {
            println(out, "buf = %s_parse_json_struct_inline(ctx, buf, end, pval);", snref.text);
        }
    } else if (is_string) {
        println(out, "buf = flatcc_json_parser_build_string(ctx, buf, end, &ref);");
    } else if (is_base64 || is_base64url) {
        println(out, "buf = flatcc_json_parser_build_uint8_vector_base64(ctx, buf, end, &ref, %u);",
                !is_base64);
    } else if (is_table) {
        println(out, "buf = %s_parse_json_table(ctx, buf, end, &ref);", snref.text);
    } else if (is_union) {
        if (is_union_vector) {
            println(out, "buf = flatcc_json_parser_union_vector(ctx, buf, end, %"PRIu64", %"PRIu64", h_unions, %s_parse_json_union);",
                (uint64_t)member->export_index, member->id, snref.text);
        } else {
            println(out, "buf = flatcc_json_parser_union(ctx, buf, end, %"PRIu64", %"PRIu64", h_unions, %s_parse_json_union);",
                (uint64_t)member->export_index, member->id, snref.text);
        }
    } else if (is_union_type) {
        println(out, "static flatcc_json_parser_integral_symbol_f *symbolic_parsers[] = {");
        indent(); indent();
        println(out, "%s_parse_json_enum,", snref.text);
        println(out, "%s_local_%sjson_parser_enum,", out->S->basename, scope_name);
        println(out, "%s_global_json_parser_enum, 0 };", out->S->basename);
        unindent(); unindent();
        if (is_union_type_vector) {
        println(out, "buf = flatcc_json_parser_union_type_vector(ctx, buf, end, %"PRIu64", %"PRIu64", h_unions, symbolic_parsers, %s_parse_json_union, %s_json_union_accept_type);",
                (uint64_t)member->export_index, member->id, snref.text, snref.text);
        } else {
            println(out, "buf = flatcc_json_parser_union_type(ctx, buf, end, %"PRIu64", %"PRIu64", h_unions, symbolic_parsers, %s_parse_json_union);",
                (uint64_t)member->export_index, member->id, snref.text);
        }
    } else if (!is_vector && !is_char_array) {
        gen_panic(out, "internal error: unexpected type for trie member\n");
        return -1;
    }
    if (is_vector) {
        if (is_offset) {
            /* Deal with table and string vector elements - unions cannot be elements. */
            println(out, "if (!ref || !(pref = flatcc_builder_extend_offset_vector(ctx->ctx, 1))) goto failed;");
            /* We don't need to worry about endian conversion - offsets vectors fix this automatically. */
            println(out, "*pref = ref;");
        }
        println(out, "buf = flatcc_json_parser_array_end(ctx, buf, end, &more);");
        unindent(); println(out, "}");
        if (is_offset) {
            println(out, "ref = flatcc_builder_end_offset_vector(ctx->ctx);");
        } else {
            println(out, "ref = flatcc_builder_end_vector(ctx->ctx);");
        }
    }
    if (is_array) {
        println(out, "buf = flatcc_json_parser_array_end(ctx, buf, end, &more);");
        unindent(); println(out, "}");
        println(out, "if (count) {"); indent();
        println(out, "if (ctx->flags & flatcc_json_parser_f_reject_array_underflow) {"); indent();
        println(out, "return flatcc_json_parser_set_error(ctx, buf, end, flatcc_json_parser_error_array_underflow);");
        unindent(); println(out, "}");
        if (is_scalar) {
            println(out, "memset(base, 0, count * sizeof(*base));");
        } else {
            println(out, "memset(base, 0, count * %"PRIu64");", (uint64_t)member->type.ct->size);
        }
        unindent(); println(out, "}");
    }
    if (is_nested == 1) {
        is_nested = 2;
        goto repeat_nested;
    }
    if (is_nested == 2) {
        println(out, "if (!ref) goto failed;");
        println(out, "ref = flatcc_builder_end_buffer(ctx->ctx, ref);");
        unindent(); println(out, "} /* end nested */");
    }
    if (is_nested || is_vector || is_table || is_string || is_base64 || is_base64url) {
        println(out, "if (!ref || !(pref = flatcc_builder_table_add_offset(ctx->ctx, %"PRIu64"))) goto failed;", member->id);
        println(out, "*pref = ref;");
    }
    return 0;
}

static void gen_field_match(fb_output_t *out, fb_compound_type_t *ct, void *data, int hint, int n)
{
    println(out, "buf = flatcc_json_parser_match_symbol(ctx, (mark = buf), end, %d);", n);
    println(out, "if (mark != buf) {"); indent();
    gen_field_match_handler(out, ct, data, hint);
    unindent(); println(out, "} else {"); indent();
}

/* This also handles union type enumerations. */
static void gen_enum_match_handler(fb_output_t *out, fb_compound_type_t *ct, void *data, int unused_hint)
{
    fb_member_t *member = data;

    (void)unused_hint;

    /*
     * This is rather unrelated to the rest, we just use the same
     * trie generation logic. Here we simply need to assign a known
     * value to the enum parsers output arguments.
     */
    switch (ct->type.st) {
    case fb_bool:
    case fb_ubyte:
    case fb_ushort:
    case fb_uint:
    case fb_ulong:
        println(out, "*value = UINT64_C(%"PRIu64"), *value_sign = 0;",
                member->value.u);
        break;
    case fb_byte:
    case fb_short:
    case fb_int:
    case fb_long:
        if (member->value.i < 0) {
            println(out, "*value = UINT64_C(%"PRIu64"), *value_sign = 1;", member->value.i);
        } else {
            println(out, "*value = UINT64_C(%"PRIu64"), *value_sign = 0;", member->value.i);
        }
        break;
    default:
        gen_panic(out, "internal error: invalid enum type\n");
    }
}

static void gen_enum_match(fb_output_t *out, fb_compound_type_t *ct, void *data, int hint, int n)
{
    println(out, "buf = flatcc_json_parser_match_constant(ctx, (mark = buf), end, %d, aggregate);", n);
    println(out, "if (buf != mark) {"); indent();
    gen_enum_match_handler(out, ct, data, hint);
    unindent(); println(out, "} else {"); indent();
}

static void gen_scope_match_handler(fb_output_t *out, fb_compound_type_t *unused_ct, void *data, int unused_hint)
{
    fb_compound_type_t *ct = data;
    fb_scoped_name_t snt;

    (void)unused_ct;
    (void)unused_hint;
    assert(ct->symbol.kind == fb_is_enum || ct->symbol.kind == fb_is_union);

    fb_clear(snt);
    fb_compound_name(ct, &snt);
    /* May be included from another file. Unions also have _enum parsers. */
    println(out, "buf = %s_parse_json_enum(ctx, buf, end, value_type, value, aggregate);", snt.text);
}

static void gen_scope_match(fb_output_t *out, fb_compound_type_t *ct, void *data, int hint, int n)
{
    println(out, "buf = flatcc_json_parser_match_scope(ctx, (mark = buf), end, %d);", n);
    println(out, "if (buf != mark) {"); indent();
    gen_scope_match_handler(out, ct, data, hint);
    unindent(); println(out, "} else {"); indent();
}

static void gen_field_unmatched(fb_output_t *out)
{
    println(out, "buf = flatcc_json_parser_unmatched_symbol(ctx, buf, end);");
}

static void gen_enum_unmatched(fb_output_t *out)
{
    println(out, "return unmatched;");
}

static void gen_scope_unmatched(fb_output_t *out)
{
    println(out, "return unmatched;");
}

/*
 * Generate a trie for all members or a compound type.
 * This may be a struct or a table.
 *
 * We have a ternary trie where a search word w compares:
 * w < wx_tag is one branch [a;x), iff a < x.
 * w > wx_tag is another branch (y;b], iff b > y
 * and w == wx_tag is a third branch [x;y].
 *
 * The sets [a;x) and (y;b] may be empty in which case a non-match
 * action is triggered.
 *
 * [x..y] is a set of one or more fields that share the same tag at the
 * current position. The first (and only the first) field name in this
 * set may terminate withint the current tag (when suffix length k ==
 * 0).  There is therefore potentially both a direct field action and a
 * sub-tree action. Once there is only one field in the set and the
 * field name terminates within the current tag, the search word is
 * masked and tested against the field tag and the search word is also
 * tested for termination in the buffer at the first position after the
 * field match. If the termination was not found a non-match action is
 * triggered.
 *
 * A non-match action may be to silently consume the rest of the
 * search identifier and then the json value, or to report and
 * error.
 *
 * A match action triggers a json value parse of a known type
 * which updates into a flatcc builder object. If the type is
 * basic (string or scalar) the update simple, otherwise if
 * the type is within the same schema, we push context
 * and switch to parse the nested type, otherwise we call
 * a parser in another schema. When a trie is done, we
 * switch back context if in the same schema. The context
 * lives on a stack. This avoids deep recursion because
 * schema parsers are not mutually recursive.
 *
 * The trie is also used to parse enums and scopes (namespace prefixes)
 * with a slight modification.
 */

enum trie_type { table_trie, struct_trie, enum_trie, local_scope_trie, global_scope_trie };
typedef struct trie trie_t;

typedef void gen_match_f(fb_output_t *out, fb_compound_type_t *ct, void *data, int hint, int n);
typedef void gen_unmatched_f(fb_output_t *out);

struct trie {
    dict_entry_t *dict;
    gen_match_f *gen_match;
    gen_unmatched_f *gen_unmatched;
    /* Not used with scopes. */
    fb_compound_type_t *ct;
    int type;
    int union_total;
    int label;
};

/*
 * This function is a final handler of the `gen_trie` function. Often
 * just to handle a single match, but also to handle a prefix range
 * special case like keys in `{ a, alpha, alpha2 }`.
 *
 * (See also special case of two non-prefix keys below).
 *
 * We know that all keys [a..b] have length in the range [pos..pos+8)
 * and also that key x is proper prefix of key x + 1, x in [a..b).
 *
 * It is possible that `a == b`.
 *
 * We conduct a binary search by testing the middle for masked match and
 * gradually refine until we do not have a match or have a single
 * element match.
 *
 * (An alternative algorithm xors 8 byte tag with longest prefix and
 * finds ceiling of log 2 using a few bit logic operations or intrinsic
 * zero count and creates a jump table of at most 8 elements, but is
 * hardly worthwhile vs 3 comparisons and 3 AND operations and often
 * less than that.)
 *
 * Once we have a single element match we need to confirm the successor
 * symbol is not any valid key - this differs among trie types and is
 * therefore the polymorph match logic handles the final confirmed match
 * or mismatch.
 *
 * Each trie type has special operation for implementing a matched and
 * a failed match. Our job is to call these for each key in the range.
 *
 * While not the original intention, the `gen_prefix_trie` also handles the
 * special case where the set has two keys where one is not a prefix of
 * the other, but both terminate in the same tag. In this case we can
 * immediately do an exact match test and skip the less than
 * comparision. We need no special code for this, assuming the function
 * is called correctly. This significantly reduces the branching in a
 * case like "Red, Green, Blue".
 *
 * If `label` is positive, it is used to jump to additional match logic
 * when a prefix was not matched. If 0 there is no additional logic and
 * the symbol is considered unmatched immediately.
 */
static void gen_prefix_trie(fb_output_t *out, trie_t *trie, int a, int b, int pos, int label)
{
    int m, n;
    uint64_t tag = 00, mask = 0;
    const char *name;
    int len;

    /*
     * Weigh the intersection towards the longer prefix. Notably if we
     * have two keys it makes no sense to check the shorter key first.
     */
    m = a + (b - a + 1) / 2;

    n = get_dict_tag(&trie->dict[m], pos, &tag, &mask, &name, &len);
    if (n == 8) {
        println(out, "if (w == 0x%"PRIx64") { /* \"%.*s\" */", tag, len, name); indent();
    } else {
        println(out, "if ((w & 0x%"PRIx64") == 0x%"PRIx64") { /* \"%.*s\" */",
                mask, tag, len, name); indent();
    }
    if (m == a) {
        /* There can be only one. */
        trie->gen_match(out, trie->ct, trie->dict[m].data, trie->dict[m].hint, n);
        if (label > 0) {
            println(out, "goto pfguard%d;", label);
        } else {
            trie->gen_unmatched(out);
        }
        unindent(); println(out, "}");
        unindent(); println(out, "} else { /* \"%.*s\" */", len, name); indent();
        if (label > 0) {
            println(out, "goto pfguard%d;", label);
        } else {
            trie->gen_unmatched(out);
        }
    } else {
        if (m == b) {
            trie->gen_match(out, trie->ct, trie->dict[m].data, trie->dict[m].hint, n);
            if (label > 0) {
                println(out, "goto pfguard%d;", label);
            } else {
                trie->gen_unmatched(out);
            }
            unindent(); println(out, "}");
        } else {
            gen_prefix_trie(out, trie, m, b, pos, label);
        }
        unindent(); println(out, "} else { /* \"%.*s\" */", len, name); indent();
        gen_prefix_trie(out, trie, a, m - 1, pos, label);
    }
    unindent(); println(out, "} /* \"%.*s\" */", len, name);
}

static void gen_trie(fb_output_t *out, trie_t *trie, int a, int b, int pos)
{
    int x, k;
    uint64_t tag = 0, mask = 0;
    const char *name = "";
    int len = 0, has_prefix_key = 0, prefix_guard = 0, has_descend;
    int label = 0;

    /*
     * Process a trie at the level given by pos. A single level covers
     * one tag.
     *
     * A tag is a range of 8 characters [pos..pos+7] that is read as a
     * single big endian word and tested as against a ternary trie
     * generated in code. In generated code the tag is stored in "w".
     *
     * Normally trailing data in a tag is not a problem
     * because the difference between two keys happen in the middle and
     * trailing data is not valid key material. When the difference is
     * at the end, we get a lot of special cases to handle.
     *
     * Regardless, when we believe we have a match, a final check is
     * made to ensure that the next character after the match is not a
     * valid key character - for quoted keys a valid termiantot is a
     * quote, for unquoted keys it can be one of several characters -
     * therefore quoted keys are faster to parse, even if they consume
     * more space. The trie does not care about these details, the
     * gen_match function handles this transparently for different
     * symbol types.
     */


    /*
    * If we have one or two keys that terminate in this tag, there is no
    * need to do a branch test before matching exactly.
    *
    * We observe that `gen_prefix_trie` actually handles this
    * case well, even though it was not designed for it.
    */
    if ((get_dict_suffix_len(&trie->dict[a], pos) == 0) &&
        (b == a || (b == a + 1 && get_dict_suffix_len(&trie->dict[b], pos) == 0))) {
        gen_prefix_trie(out, trie, a, b, pos, 0);
        return;
    }

    /*
     * Due trie nature, we have a left, middle, and right range where
     * the middle range all compare the same at the current trie level
     * when masked against shortest (and first) key in middle range.
     */
    x = split_dict_left(trie->dict, a, b, pos);

    if (x > a) {
        /*
         * This is normal early branch with a key `a < x < b` such that
         * any shared prefix ranges do not span x.
         */
        get_dict_tag(&trie->dict[x], pos, &tag, &mask, &name, &len);
        println(out, "if (w < 0x%"PRIx64") { /* branch \"%.*s\" */", tag, len, name); indent();
        gen_trie(out, trie, a, x - 1, pos);
        unindent(); println(out, "} else { /* branch \"%.*s\" */", len, name); indent();
        gen_trie(out, trie, x, b, pos);
        unindent(); println(out, "} /* branch \"%.*s\" */", len, name);
        return;
    }
    x = split_dict_right(trie->dict, a, b, pos);

    /*
     * [a .. x-1] is a non-empty sequence of prefixes,
     * for example { a123, a1234, a12345 }.
     * The keys might not terminate in the current tag. To find those
     * that do, we will evaluate k such that:
     * [a .. k-1] are prefixes that terminate in the current tag if any
     * such exists.
     * [x..b] are keys that are prefixes up to at least pos + 7 but
     * do not terminate in the current tag.
     * [k..x-1] are prefixes that do not termiante in the current tag.
     * Note that they might not be prefixes when considering more than the
     * current tag.
     * The range [a .. x-1] can ge generated with `gen_prefix_trie`.
     *
     * We generally have the form
     *
     * [a..b] =
     * (a)<prefixes>, (k-1)<descend-prefix>, (k)<descend>, (x)<reminder>
     *
     * Where <prefixes> are keys that terminate at the current tag.
     * <descend> are keys that have the prefixes as prefix but do not
     * terminate at the current tag.
     * <descend-prerfix> is a single key that terminates exactly
     * where the tag ends. If there are no descend keys it is part of
     * prefixes, otherwise it is tested as a special case.
     * <reminder> are any keys larger than the prefixes.
     *
     * The reminder keys cannot be tested before we are sure that no
     * prefix is matching at least no prefixes that is not a
     * descend-prefix. This is because less than comparisons are
     * affected by trailing data within the tag caused by prefixes
     * terminating early. Trailing data is not a problem if two keys are
     * longer than the point where they differ even if they terminate
     * within the current tag.
     *
     * Thus, if we have non-empty <descend> and non-empty <reminder>,
     * the reminder must guard against any matches in prefix but not
     * against any matches in <descend>. If <descend> is empty and
     * <prefixes> == <descend-prefix> a guard is also not needed.
     */

    /* Find first prefix that does not terminate at the current level, or x if absent */
    k = split_dict_descend(trie->dict, a, x - 1, pos);
    has_descend = k < x;

    /* If we have a descend, process that in isolation. */
    if (has_descend) {
        has_prefix_key = k > a && get_dict_tag_len(&trie->dict[k - 1], pos) == 8;
        get_dict_tag(&trie->dict[k], pos, &tag, &mask, &name, &len);
        println(out, "if (w == 0x%"PRIx64") { /* descend \"%.*s\" */", tag, len, name); indent();
        if (has_prefix_key) {
            /* We have a key that terminates at the descend prefix. */
            println(out, "/* descend prefix key \"%.*s\" */", len, name);
            trie->gen_match(out, trie->ct, trie->dict[k - 1].data, trie->dict[k - 1].hint, 8);
            println(out, "/* descend suffix \"%.*s\" */", len, name);
        }
        println(out, "buf += 8;");
        println(out, "w = flatcc_json_parser_symbol_part(buf, end);");
        gen_trie(out, trie, k, x - 1, pos + 8);
        if (has_prefix_key) {
            unindent(); println(out, "} /* desend suffix \"%.*s\" */", len, name);
            /* Here we move the <descend-prefix> key out of the <descend> range. */
            --k;
        }
        unindent(); println(out, "} else { /* descend \"%.*s\" */", len, name); indent();
    }
    prefix_guard = a < k && x <= b;
    if (prefix_guard) {
        label = ++trie->label;
    }
    if (a < k) {
        gen_prefix_trie(out, trie, a, k - 1, pos, label);
    }
    if (prefix_guard) {
        /* All prefixes tested, but none matched. */
        println(out, "goto endpfguard%d;", label);
        margin();
        println(out, "pfguard%d:", label);
        unmargin();
    }
    if (x <= b) {
        gen_trie(out, trie, x, b, pos);
    } else if (a >= k) {
        trie->gen_unmatched(out);
    }
    if (prefix_guard) {
        margin();
        println(out, "endpfguard%d:", label);
        unmargin();
        println(out, "(void)0;");
    }
    if (has_descend) {
        unindent(); println(out, "} /* descend \"%.*s\" */", len, name);
    }
}


/*
 * Parsing symbolic constants:
 *
 * An enum parser parses the local symbols and translate them into
 * numeric values.
 *
 * If a symbol wasn't matched, e.g. "Red", it might be matched with
 * "Color.Red" but the enum parser does not handle this.
 *
 * Instead a scope parser maps each type in the scope to a call
 * to an enum parser, e.g. "Color." maps to a color enum parser
 * that understands "Red". If this also fails, a call is made
 * to a global scope parser that maps a namespace to a local
 * scope parser, for example "Graphics.Color.Red" first
 * recognizes the namespace "Graphics." which may or may not
 * be the same as the local scope tried earlier, then "Color."
 * is matched and finally "Red".
 *
 * The scope and namespace parsers may cover extend namespaces from
 * include files so each file calls into dependencies as necessary.
 * This means the same scope can have multiple parsers and must
 * therefore be name prefixed by the basename of the include file.
 *
 * The enums can only exist in a single file.
 *
 * The local scope is defined as the scope in which the consuming
 * fields container is defined, so if Pen is a table in Graphics
 * with a field named "ink" and the pen is parsed as
 *   { "ink": "Color.Red" }, then Color would be parsed in the
 * Graphics scope. If ink was and enum of type Color, the enum
 * parser would be tried first. If ink was, say, an integer
 * type, it would not try an enum parse first but try the local
 * scope, then the namespace scope.
 *
 * It is permitted to have multiple symbols in a string when
 * the enum type has flag attribute so values can be or'ed together.
 * The parser does not attempt to validate this and will simple
 * 'or' together multiple values after coercing each to the
 * receiving field type: "Has.ink Has.shape Has.brush".
 */


/*
 * Used by scalar/enum/union_type table fields to look up symbolic
 * constants in same scope as the table was defined, thus avoiding
 * namespace prefix.
 *
 * Theh matched name then calls into the type specific parser which
 * may be in a dependent file.
 *
 * Because each scope may be extended in dependent schema files
 * we recreate the scope in full in each file.
 */
static void gen_local_scope_parser(void *context, fb_scope_t *scope)
{
    fb_output_t *out = context;
    int n = 0;
    trie_t trie;
    fb_symbol_text_t scope_name;

    fb_clear(trie);
    fb_copy_scope(scope, scope_name);
    if (((trie.dict = build_local_scope_dict(out->S, scope, &n)) == 0) && n > 0) {
        gen_panic(out, "internal error: could not build dictionary for json parser\n");
        return;
    }
    /* Not used for scopes. */
    trie.ct = 0;
    trie.type = local_scope_trie;
    trie.gen_match = gen_scope_match;
    trie.gen_unmatched = gen_scope_unmatched;
    println(out, "static const char *%s_local_%sjson_parser_enum(flatcc_json_parser_t *ctx, const char *buf, const char *end,",
        out->S->basename, scope_name);
    indent(); indent();
    println(out, "int *value_type, uint64_t *value, int *aggregate)");
    unindent(); unindent();
    println(out, "{"); indent();
    if (n == 0) {
        println(out, "/* Scope has no enum / union types to look up. */");
        println(out, "return buf; /* unmatched; */");
        unindent(); println(out, "}");
    } else {
        println(out, "const char *unmatched = buf;");
        println(out, "const char *mark;");
        println(out, "uint64_t w;");
        println(out, "");
        println(out, "w = flatcc_json_parser_symbol_part(buf, end);");
        gen_trie(out, &trie, 0, n - 1, 0);
        println(out, "return buf;");
        unindent(); println(out, "}");
    }
    println(out, "");
    clear_dict(trie.dict);
}

/*
 * This parses namespace prefixed types. Because scopes can be extended
 * in dependent schema files, each file has its own global scope parser.
 * The matched types call into type specific parsers that may be in
 * a dependent file.
 *
 * When a local scope is also parsed, it should be tried before the
 * global scope.
 */
static int gen_global_scope_parser(fb_output_t *out)
{
    int n = 0;
    trie_t trie;
    catalog_t catalog;

    fb_clear(trie);
    if (build_catalog(&catalog, out->S, 1, &out->S->root_schema->scope_index)) {
        return -1;
    }

    if ((trie.dict = build_global_scope_dict(&catalog, &n)) == 0 && n > 0) {
        clear_catalog(&catalog);
        gen_panic(out, "internal error: could not build dictionary for json parser\n");
        return -1;
    }
    /* Not used for scopes. */
    trie.ct = 0;
    trie.type = global_scope_trie;
    trie.gen_match = gen_scope_match;
    trie.gen_unmatched = gen_scope_unmatched;
    println(out, "static const char *%s_global_json_parser_enum(flatcc_json_parser_t *ctx, const char *buf, const char *end,", out->S->basename);
    indent(); indent();
    println(out, "int *value_type, uint64_t *value, int *aggregate)");
    unindent(); unindent();
    println(out, "{"); indent();
    if (n == 0) {
        println(out, "/* Global scope has no enum / union types to look up. */");
        println(out, "return buf; /* unmatched; */");
        unindent(); println(out, "}");
    } else {
        println(out, "const char *unmatched = buf;");
        println(out, "const char *mark;");
        println(out, "uint64_t w;");
        println(out, "");
        println(out, "w = flatcc_json_parser_symbol_part(buf, end);");
        gen_trie(out, &trie, 0, n - 1, 0);
        println(out, "return buf;");
        unindent(); println(out, "}");
    }
    println(out, "");
    clear_dict(trie.dict);
    clear_catalog(&catalog);
    return 0;
}

/*
 * Constants have the form `"Red"` or `Red` but may also be part
 * of a list of flags: `"Normal High Average"` or `Normal High
 * Average`. `more` indicates more symbols follow.
 *
 * Returns input argument if there was no valid match,
 * `end` on syntax error, and `more=1` if matched and
 * there are more constants to parse.
 * Applies the mached and coerced constant to `pval`
 * with a binary `or` operation so `pval` must be initialized
 * to 0 before teh first constant in a list.
 */
static int gen_enum_parser(fb_output_t *out, fb_compound_type_t *ct)
{
    fb_scoped_name_t snt;
    int n = 0;
    trie_t trie;

    fb_clear(trie);
    assert(ct->symbol.kind == fb_is_enum || ct->symbol.kind == fb_is_union);

    if ((trie.dict = build_compound_dict(ct, &n)) == 0 && n > 0) {
        gen_panic(out, "internal error: could not build dictionary for json parser\n");
        return -1;
    }
    trie.ct = ct;
    trie.type = enum_trie;
    trie.gen_match = gen_enum_match;
    trie.gen_unmatched = gen_enum_unmatched;

    fb_clear(snt);
    fb_compound_name(ct, &snt);

    println(out, "static const char *%s_parse_json_enum(flatcc_json_parser_t *ctx, const char *buf, const char *end,", snt.text);
    indent(); indent();
    println(out, "int *value_sign, uint64_t *value, int *aggregate)");
    unindent(); unindent();
    println(out, "{"); indent();
    if (n == 0) {
        println(out, "/* Enum has no fields. */");
        println(out, "*aggregate = 0;");
        println(out, "return buf; /* unmatched; */");
        unindent(); println(out, "}");
    } else {
        println(out, "const char *unmatched = buf;");
        println(out, "const char *mark;");
        println(out, "uint64_t w;");
        println(out, "");
        println(out, "w = flatcc_json_parser_symbol_part(buf, end);");
        gen_trie(out, &trie, 0, n - 1, 0);
        println(out, "return buf;");
        unindent(); println(out, "}");
    }
    println(out, "");
    clear_dict(trie.dict);
    return 0;
}

/*
 * We do not check for duplicate settings or missing struct fields.
 * Missing fields are zeroed.
 *
 * TODO: we should track nesting level because nested structs do not
 * interact with the builder so the builders level limit will not kick
 * in. As long as we get input from our own parser we should, however,
 * be reasonable safe as nesting is bounded.
 */
static int gen_struct_parser_inline(fb_output_t *out, fb_compound_type_t *ct)
{
    fb_scoped_name_t snt;
    int n;
    trie_t trie;

    fb_clear(trie);
    assert(ct->symbol.kind == fb_is_struct);
    if ((trie.dict = build_compound_dict(ct, &n)) == 0 && n > 0) {
        gen_panic(out, "internal error: could not build dictionary for json parser\n");
        return -1;
    }
    trie.ct = ct;
    trie.type = struct_trie;
    trie.gen_match = gen_field_match;
    trie.gen_unmatched = gen_field_unmatched;

    fb_clear(snt);
    fb_compound_name(ct, &snt);
    println(out, "static const char *%s_parse_json_struct_inline(flatcc_json_parser_t *ctx, const char *buf, const char *end, void *struct_base)", snt.text);
    println(out, "{"); indent();
    println(out, "int more;");
    if (n > 0) {
        println(out, "flatcc_builder_ref_t ref;");
        println(out, "void *pval;");
        println(out, "const char *mark;");
        println(out, "uint64_t w;");
    }
    println(out, "");
    println(out, "buf = flatcc_json_parser_object_start(ctx, buf, end, &more);");
    println(out, "while (more) {"); indent();
    if (n == 0) {
        println(out, "/* Empty struct. */");
        println(out, "buf = flatcc_json_parser_unmatched_symbol(ctx, buf, end);");
    } else {
        println(out, "buf = flatcc_json_parser_symbol_start(ctx, buf, end);");
        println(out, "w = flatcc_json_parser_symbol_part(buf, end);");
        gen_trie(out, &trie, 0, n - 1, 0);
    }
    println(out, "buf = flatcc_json_parser_object_end(ctx, buf, end , &more);");
    unindent(); println(out, "}");
    println(out, "return buf;");
    if (n > 0) {
        /* Set runtime error if no other error was set already. */
        margin();
        println(out, "failed:");
        unmargin();
        println(out, "return flatcc_json_parser_set_error(ctx, buf, end, flatcc_json_parser_error_runtime);");
    }
    unindent(); println(out, "}");
    println(out, "");
    clear_dict(trie.dict);
    return 0;
}

static int gen_struct_parser(fb_output_t *out, fb_compound_type_t *ct)
{
    fb_scoped_name_t snt;

    assert(ct->symbol.kind == fb_is_struct);
    fb_clear(snt);
    fb_compound_name(ct, &snt);
    println(out, "static const char *%s_parse_json_struct(flatcc_json_parser_t *ctx, const char *buf, const char *end, flatcc_builder_ref_t *result)", snt.text);
    println(out, "{"); indent();
    println(out, "void *pval;");
    println(out, "");
    println(out, "*result = 0;");
    println(out, "if (!(pval = flatcc_builder_start_struct(ctx->ctx, %"PRIu64", %"PRIu16"))) goto failed;",
            (uint64_t)ct->size, (uint16_t)ct->align);
    println(out, "buf = %s_parse_json_struct_inline(ctx, buf, end, pval);", snt.text);
    println(out, "if (ctx->error || !(*result = flatcc_builder_end_struct(ctx->ctx))) goto failed;");
    println(out, "return buf;");
    margin();
    println(out, "failed:");
    unmargin();
    println(out, "return flatcc_json_parser_set_error(ctx, buf, end, flatcc_json_parser_error_runtime);");
    unindent(); println(out, "}");
    println(out, "");
    println(out, "static inline int %s_parse_json_as_root(flatcc_builder_t *B, flatcc_json_parser_t *ctx, const char *buf, size_t bufsiz, flatcc_json_parser_flags_t flags, const char *fid)", snt.text);
    println(out, "{"); indent();
    println(out, "return flatcc_json_parser_struct_as_root(B, ctx, buf, bufsiz, flags, fid, %s_parse_json_struct);",
            snt.text);
    unindent(); println(out, "}");
    println(out, "");
    return 0;
}

static int gen_table_parser(fb_output_t *out, fb_compound_type_t *ct)
{
    fb_scoped_name_t snt;
    fb_member_t *member;
    int first, i, n;
    int is_union, is_required;
    trie_t trie;

    fb_clear(trie);
    assert(ct->symbol.kind == fb_is_table);
    if ((trie.dict = build_compound_dict(ct, &n)) == 0 && n > 0) {
        gen_panic(out, "internal error: could not build dictionary for json parser\n");
        return -1;
    }
    trie.ct = ct;
    trie.type = table_trie;
    trie.gen_match = gen_field_match;
    trie.gen_unmatched = gen_field_unmatched;

    trie.union_total = 0;
    for (i = 0; i < n; ++i) {
        trie.union_total += !!trie.dict[i].hint;
    }

    fb_clear(snt);
    fb_compound_name(ct, &snt);
    println(out, "static const char *%s_parse_json_table(flatcc_json_parser_t *ctx, const char *buf, const char *end, flatcc_builder_ref_t *result)", snt.text);
    println(out, "{"); indent();
    println(out, "int more;");

    if (n > 0) {
        println(out, "void *pval;");
        println(out, "flatcc_builder_ref_t ref, *pref;");
        println(out, "const char *mark;");
        println(out, "uint64_t w;");
    }
    if (trie.union_total) {
        println(out, "size_t h_unions;");
    }
    println(out, "");
    println(out, "*result = 0;");
    println(out, "if (flatcc_builder_start_table(ctx->ctx, %"PRIu64")) goto failed;",
        ct->count);
    if (trie.union_total) {
        println(out, "if (end == flatcc_json_parser_prepare_unions(ctx, buf, end, %"PRIu64", &h_unions)) goto failed;", (uint64_t)trie.union_total);
    }
    println(out, "buf = flatcc_json_parser_object_start(ctx, buf, end, &more);");
    println(out, "while (more) {"); indent();
    println(out, "buf = flatcc_json_parser_symbol_start(ctx, buf, end);");
    if (n > 0) {
        println(out, "w = flatcc_json_parser_symbol_part(buf, end);");
        gen_trie(out, &trie, 0, n - 1, 0);
    } else {
        println(out, "/* Table has no fields. */");
        println(out, "buf = flatcc_json_parser_unmatched_symbol(ctx, buf, end);");
    }
    println(out, "buf = flatcc_json_parser_object_end(ctx, buf, end, &more);");
    unindent(); println(out, "}");
    println(out, "if (ctx->error) goto failed;");
    for (first = 1, i = 0; i < n; ++i) {
        member = trie.dict[i].data;
        if (member->metadata_flags & fb_f_deprecated) {
            continue;
        }
        is_union = is_union_member(member);
        is_required = member->metadata_flags & fb_f_required;
        if (is_required) {
            if (first) {
                println(out, "if (!flatcc_builder_check_required_field(ctx->ctx, %"PRIu64")", member->id - !!is_union);
                indent();
            } else {
                println(out, "||  !flatcc_builder_check_required_field(ctx->ctx, %"PRIu64")", member->id - !!is_union);
            }
            first = 0;
        }
    }
    if (!first) {
        unindent(); println(out, ") {"); indent();
        println(out, "buf = flatcc_json_parser_set_error(ctx, buf, end, flatcc_json_parser_error_required);");
        println(out, "goto failed;");
        unindent(); println(out, "}");
    }
    if (trie.union_total) {
        println(out, "buf = flatcc_json_parser_finalize_unions(ctx, buf, end, h_unions);");
    }
    println(out, "if (!(*result = flatcc_builder_end_table(ctx->ctx))) goto failed;");
    println(out, "return buf;");
    /* Set runtime error if no other error was set already. */
    margin();
    println(out, "failed:");
    unmargin();
    println(out, "return flatcc_json_parser_set_error(ctx, buf, end, flatcc_json_parser_error_runtime);");
    unindent(); println(out, "}");
    println(out, "");
    println(out, "static inline int %s_parse_json_as_root(flatcc_builder_t *B, flatcc_json_parser_t *ctx, const char *buf, size_t bufsiz, flatcc_json_parser_flags_t flags, const char *fid)", snt.text);
    println(out, "{"); indent();
    println(out, "return flatcc_json_parser_table_as_root(B, ctx, buf, bufsiz, flags, fid, %s_parse_json_table);",
            snt.text);
    unindent(); println(out, "}");
    println(out, "");
    clear_dict(trie.dict);
    return 0;
}

static int gen_union_parser(fb_output_t *out, fb_compound_type_t *ct)
{
    fb_scoped_name_t snt, snref;
    fb_symbol_t *sym;
    fb_member_t *member;
    int n;
    const char *s;

    fb_clear(snt);
    fb_clear(snref);
    fb_compound_name(ct, &snt);
    println(out, "static const char *%s_parse_json_union(flatcc_json_parser_t *ctx, const char *buf, const char *end, uint8_t type, flatcc_builder_ref_t *result)", snt.text);
    println(out, "{"); indent();
    println(out, "");
    println(out, "*result = 0;");
    println(out, "switch (type) {");
    println(out, "case 0: /* NONE */"); indent();
    println(out, "return flatcc_json_parser_none(ctx, buf, end);");
    unindent();
    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        symbol_name(sym, &n, &s);
        switch (member->type.type) {
        case vt_missing:
            /* NONE is of type vt_missing and already handled. */
            continue;
        case vt_compound_type_ref:
            fb_compound_name(member->type.ct, &snref);
            println(out, "case %u: /* %.*s */", (unsigned)member->value.u, n, s); indent();
            switch (member->type.ct->symbol.kind) {
            case fb_is_table:
                println(out, "buf = %s_parse_json_table(ctx, buf, end, result);", snref.text);
                break;
            case fb_is_struct:
                println(out, "buf = %s_parse_json_struct(ctx, buf, end, result);", snref.text);
                break;
            default:
                gen_panic(out, "internal error: unexpected compound union member type\n");
                return -1;
            }
            println(out, "break;");
            unindent();
            continue;
        case vt_string_type:
            println(out, "case %u: /* %.*s */", (unsigned)member->value.u, n, s); indent();
            println(out, "buf = flatcc_json_parser_build_string(ctx, buf, end, result);");
            println(out, "break;");
            unindent();
            continue;
        default:
            gen_panic(out, "internal error: unexpected union member type\n");
            return -1;
        }
    }
    /* Unknown union, but not an error if we allow schema forwarding. */
    println(out, "default:"); indent();
    println(out, "if (!(ctx->flags & flatcc_json_parser_f_skip_unknown)) {"); indent();
    println(out, "return flatcc_json_parser_set_error(ctx, buf, end, flatcc_json_parser_error_unknown_union);");
    unindent(); println(out, "} else {"); indent();
    println(out, "return flatcc_json_parser_generic_json(ctx, buf, end);");
    unindent(); println(out, "}");
    unindent(); println(out, "}");
    println(out, "if (ctx->error) return buf;");
    println(out, "if (!*result) {");
    indent(); println(out, "return flatcc_json_parser_set_error(ctx, buf, end, flatcc_json_parser_error_runtime);");
    unindent(); println(out, "}");
    println(out, "return buf;");
    unindent(); println(out, "}");
    println(out, "");
    return 0;
}

static int gen_union_accept_type(fb_output_t *out, fb_compound_type_t *ct)
{
    fb_scoped_name_t snt, snref;
    fb_symbol_t *sym;
    fb_member_t *member;
    int n;
    const char *s;

    fb_clear(snt);
    fb_clear(snref);
    fb_compound_name(ct, &snt);
    println(out, "static int %s_json_union_accept_type(uint8_t type)", snt.text);
    println(out, "{"); indent();
    println(out, "switch (type) {");
    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        symbol_name(sym, &n, &s);
        if (member->type.type == vt_missing) {
            println(out, "case 0: return 1; /* NONE */");
            continue;
        }
        println(out, "case %u: return 1; /* %.*s */", (unsigned)member->value.u, n, s);
    }
    /* Unknown union, but not an error if we allow schema forwarding. */
    println(out, "default: return 0;"); indent();
    unindent(); println(out, "}");
    unindent(); println(out, "}");
    println(out, "");
    return 0;
}

static void gen_local_scope_prototype(void *context, fb_scope_t *scope)
{
    fb_output_t *out = context;
    fb_symbol_text_t scope_name;

    fb_copy_scope(scope, scope_name);

    println(out, "static const char *%s_local_%sjson_parser_enum(flatcc_json_parser_t *ctx, const char *buf, const char *end,",
        out->S->basename, scope_name);
    println(out, "int *value_type, uint64_t *value, int *aggregate);");
}

static int gen_root_table_parser(fb_output_t *out, fb_compound_type_t *ct)
{
    fb_scoped_name_t snt;

    fb_clear(snt);
    fb_compound_name(ct, &snt);

    println(out, "static int %s_parse_json(flatcc_builder_t *B, flatcc_json_parser_t *ctx,", out->S->basename);
    indent(); indent();
    println(out, "const char *buf, size_t bufsiz, flatcc_json_parser_flags_t flags)");
    unindent(); unindent();
    println(out, "{"); indent();
    println(out, "flatcc_json_parser_t parser;");
    println(out, "flatcc_builder_ref_t root;");
    println(out, "");
    println(out, "ctx = ctx ? ctx : &parser;");
    println(out, "flatcc_json_parser_init(ctx, B, buf, buf + bufsiz, flags);");
    if (out->S->file_identifier.type == vt_string) {
        println(out, "if (flatcc_builder_start_buffer(B, \"%.*s\", 0, 0)) return -1;",
        out->S->file_identifier.s.len, out->S->file_identifier.s.s);
    } else {
        println(out, "if (flatcc_builder_start_buffer(B, 0, 0, 0)) return -1;");
    }
    println(out, "%s_parse_json_table(ctx, buf, buf + bufsiz, &root);", snt.text);
    println(out, "if (ctx->error) {"); indent();
    println(out, "return ctx->error;");
    unindent(); println(out, "}");
    println(out, "if (!flatcc_builder_end_buffer(B, root)) return -1;");
    println(out, "ctx->end_loc = buf;");
    println(out, "return 0;");
    unindent(); println(out, "}");
    println(out, "");
    return 0;
}

static int gen_root_struct_parser(fb_output_t *out, fb_compound_type_t *ct)
{
    fb_scoped_name_t snt;

    fb_clear(snt);
    fb_compound_name(ct, &snt);

    println(out, "static int %s_parse_json(flatcc_builder_t *B, flatcc_json_parser_t *ctx,", out->S->basename);
    indent(); indent();
    println(out, "const char *buf, size_t bufsiz, int flags)");
    unindent(); unindent();
    println(out, "{"); indent();
    println(out, "flatcc_json_parser_t ctx_;");
    println(out, "flatcc_builder_ref_t root;");
    println(out, "");
    println(out, "ctx = ctx ? ctx : &ctx_;");
    println(out, "flatcc_json_parser_init(ctx, B, buf, buf + bufsiz, flags);");
    if (out->S->file_identifier.type == vt_string) {
        println(out, "if (flatcc_builder_start_buffer(B, \"%.*s\", 0, 0)) return -1;",
        out->S->file_identifier.s.len, out->S->file_identifier.s.s);
    } else {
        println(out, "if (flatcc_builder_start_buffer(B, 0, 0, 0)) return -1;");
    }
    println(out, "buf = %s_parse_json_struct(ctx, buf, buf + bufsiz, &root);", snt.text);
    println(out, "if (ctx->error) {"); indent();
    println(out, "return ctx->error;");
    unindent(); println(out, "}");
    println(out, "if (!flatcc_builder_end_buffer(B, root)) return -1;");
    println(out, "ctx->end_loc = buf;");
    println(out, "return 0;");
    unindent(); println(out, "}");
    println(out, "");
    return 0;
}


static int gen_root_parser(fb_output_t *out)
{
    fb_symbol_t *root_type = out->S->root_type.type;

    if (!root_type) {
        return 0;
    }
    if (root_type) {
        switch (root_type->kind) {
        case fb_is_table:
            return gen_root_table_parser(out, (fb_compound_type_t *)root_type);
        case fb_is_struct:
            return gen_root_struct_parser(out, (fb_compound_type_t *)root_type);
        default:
            break;
        }
    }
    return 0;
}

static int gen_json_parser_prototypes(fb_output_t *out)
{
    fb_symbol_t *sym;
    fb_scoped_name_t snt;
    fb_symbol_t *root_type = out->S->root_type.type;

    fb_clear(snt);

    if (root_type)
    switch (root_type->kind) {
    case fb_is_table:
    case fb_is_struct:
        println(out, "/*");
        println(out, " * Parses the default root table or struct of the schema and constructs a FlatBuffer.");
        println(out, " *");
        println(out, " * Builder `B` must be initialized. `ctx` can be null but will hold");
        println(out, " * hold detailed error info on return when available.");
        println(out, " * Returns 0 on success, or error code.");
        println(out, " * `flags` : 0 by default, `flatcc_json_parser_f_skip_unknown` silently");
        println(out, " * ignores unknown table and structs fields, and union types.");
        println(out, " */");
    println(out, "static int %s_parse_json(flatcc_builder_t *B, flatcc_json_parser_t *ctx,",
            out->S->basename);
    indent(); indent();
    println(out, "const char *buf, size_t bufsiz, flatcc_json_parser_flags_t flags);");
    unindent(); unindent();
        println(out, "");
        break;
    default:
        break;
    }
    for (sym = out->S->symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_union:
            fb_compound_name((fb_compound_type_t *)sym, &snt);
            println(out, "static const char *%s_parse_json_union(flatcc_json_parser_t *ctx, const char *buf, const char *end, uint8_t type, flatcc_builder_ref_t *pref);", snt.text);
            println(out, "static int %s_json_union_accept_type(uint8_t type);", snt.text);
            /* A union also has an enum parser to get the type. */
            println(out, "static const char *%s_parse_json_enum(flatcc_json_parser_t *ctx, const char *buf, const char *end,", snt.text);
            indent(); indent();
            println(out, "int *value_type, uint64_t *value, int *aggregate);");
            unindent(); unindent();
            break;
        case fb_is_struct:
            fb_compound_name((fb_compound_type_t *)sym, &snt);
            println(out, "static const char *%s_parse_json_struct_inline(flatcc_json_parser_t *ctx, const char *buf, const char *end, void *struct_base);", snt.text);
            println(out, "static const char *%s_parse_json_struct(flatcc_json_parser_t *ctx, const char *buf, const char *end, flatcc_builder_ref_t *result);", snt.text);
            break;
        case fb_is_table:
            fb_compound_name((fb_compound_type_t *)sym, &snt);
            println(out, "static const char *%s_parse_json_table(flatcc_json_parser_t *ctx, const char *buf, const char *end, flatcc_builder_ref_t *result);", snt.text);
            break;
        case fb_is_enum:
            fb_compound_name((fb_compound_type_t *)sym, &snt);
            println(out, "static const char *%s_parse_json_enum(flatcc_json_parser_t *ctx, const char *buf, const char *end,", snt.text);
            indent(); indent();
            println(out, "int *value_type, uint64_t *value, int *aggregate);", snt.text);
            unindent(); unindent();
            break;
        }
    }
    fb_scope_table_visit(&out->S->root_schema->scope_index, gen_local_scope_prototype, out);
    println(out, "static const char *%s_global_json_parser_enum(flatcc_json_parser_t *ctx, const char *buf, const char *end,", out->S->basename);
    indent(); indent();
    println(out, "int *value_type, uint64_t *value, int *aggregate);");
    unindent(); unindent();
    println(out, "");
    return 0;
}

static int gen_json_parsers(fb_output_t *out)
{
    fb_symbol_t *sym;

    for (sym = out->S->symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_union:
            gen_union_parser(out, (fb_compound_type_t *)sym);
            gen_union_accept_type(out, (fb_compound_type_t *)sym);
            gen_enum_parser(out, (fb_compound_type_t *)sym);
            break;
        case fb_is_struct:
            gen_struct_parser_inline(out, (fb_compound_type_t *)sym);
            gen_struct_parser(out, (fb_compound_type_t *)sym);
            break;
        case fb_is_table:
            gen_table_parser(out, (fb_compound_type_t *)sym);
            break;
        case fb_is_enum:
            gen_enum_parser(out, (fb_compound_type_t *)sym);
            break;
        }
    }
    fb_scope_table_visit(&out->S->root_schema->scope_index, gen_local_scope_parser, out);
    gen_global_scope_parser(out);
    gen_root_parser(out);
    return 0;
}

int fb_gen_c_json_parser(fb_output_t *out)
{
    gen_json_parser_pretext(out);
    gen_json_parser_prototypes(out);
    gen_json_parsers(out);
    gen_json_parser_footer(out);
    return 0;
}
