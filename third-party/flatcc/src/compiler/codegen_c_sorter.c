#include "codegen_c.h"

#include "flatcc/flatcc_types.h"

/* -DFLATCC_PORTABLE may help if inttypes.h is missing. */
#ifndef PRId64
#include <inttypes.h>
#endif

/* Used internally to identify sortable objects. */
enum {
    /* object contains at least one direct vector that needs sorting */
    direct_sortable = 1,
    /* object contains at least one indirect vector that needs sorting */
    indirect_sortable = 1,
    /* object contains at least one direct or indirect vector that needs sorting */
    sortable = 3,
};

static int gen_union_sorter(fb_output_t *out, fb_compound_type_t *ct)
{
    fb_symbol_t *sym;
    fb_member_t *member;
    fb_scoped_name_t snt, snref;
    int n;
    const char *s;

    fb_clear(snt);
    fb_clear(snref);
    fb_compound_name(ct, &snt);

    fprintf(out->fp,
            "static void %s_sort(%s_mutable_union_t u)\n{\n    switch (u.type) {\n",
            snt.text, snt.text);
    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        symbol_name(sym, &n, &s);
        switch (member->type.type) {
        case vt_compound_type_ref:
            fb_compound_name(member->type.ct, &snref);
            switch (member->type.ct->symbol.kind) {
            case fb_is_table:
                if (member->type.ct->export_index & sortable) {
                    fprintf(out->fp,
                            "    case %s_%.*s: %s_sort(u.value); break;\n",
                            snt.text, n, s, snref.text);
                }
                break;
            default:
                break;
            }
            break;
        default:
            break;
        }
    }
    fprintf(out->fp,
            "    default: break;\n    }\n}\n\n");
    return 0;
}

static int gen_table_sorter(fb_output_t *out, fb_compound_type_t *ct)
{
    fb_symbol_t *sym;
    fb_member_t *member;
    fb_scoped_name_t snt, snref;
    const char *tname_prefix;
    const char *nsc = out->nsc;
    const char *s;
    int n;

    fb_clear(snt);
    fb_clear(snref);
    fb_compound_name(ct, &snt);

    fprintf(out->fp,
            "static void %s_sort(%s_mutable_table_t t)\n{\n",
            snt.text, snt.text);

    fprintf(out->fp, "    if (!t) return;\n");
    /* sort all children before sorting current table */
    if (ct->export_index & indirect_sortable)
    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        if (member->metadata_flags & fb_f_deprecated) {
            continue;
        }
        symbol_name(sym, &n, &s);
        switch (member->type.type) {
        case vt_compound_type_ref:
            fb_compound_name(member->type.ct, &snref);
            switch (member->type.ct->symbol.kind) {
            case fb_is_table:
                if (!(member->type.ct->export_index & sortable)) continue;
                fprintf(out->fp,
                        "    __%ssort_table_field(%s, %.*s, %s, t);\n",
                        nsc, snt.text, n, s, snref.text);
                break;
            case fb_is_union:
                if (!(member->type.ct->export_index & sortable)) continue;
                fprintf(out->fp,
                        "    __%ssort_union_field(%s, %.*s, %s, t);\n",
                        nsc, snt.text, n, s, snref.text);
                break;
            default:
                continue;
            }
            break;
        case vt_vector_compound_type_ref:
            fb_compound_name(member->type.ct, &snref);
            switch (member->type.ct->symbol.kind) {
            case fb_is_table:
                if (!(member->type.ct->export_index & sortable)) continue;
                fprintf(out->fp,
                        "    __%ssort_table_vector_field_elements(%s, %.*s, %s, t);\n",
                        nsc, snt.text, n, s, snref.text);
                break;
            case fb_is_union:
                /* Although union vectors cannot be sorted, their content can be. */
                if (!(member->type.ct->export_index & sortable)) continue;
                fprintf(out->fp,
                        "    __%ssort_union_vector_field_elements(%s, %.*s, %s, t);\n",
                        nsc, snt.text, n, s, snref.text);
                break;
            default:
                continue;
            }
            break;
        }
    }
    if (ct->export_index & direct_sortable)
    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        symbol_name(sym, &n, &s);
        if (!(member->metadata_flags & fb_f_sorted)) continue;
        switch (member->type.type) {
        case vt_vector_type:
            tname_prefix = scalar_type_prefix(member->type.st);
            fprintf(out->fp,
                    "    __%ssort_vector_field(%s, %.*s, %s%s, t)\n",
                    nsc, snt.text, n, s, nsc, tname_prefix);
            break;
        case vt_vector_string_type:
            fprintf(out->fp,
                    "    __%ssort_vector_field(%s, %.*s, %s%s, t)\n",
                    nsc, snt.text, n, s, nsc, "string");
            break;
        case vt_vector_compound_type_ref:
            if (!member->type.ct->primary_key) {
                gen_panic(out, "internal error: unexpected type during code generation");
                return -1;
            }
            fb_compound_name(member->type.ct, &snref);
            switch (member->type.ct->symbol.kind) {
            case fb_is_table:
            case fb_is_struct:
                fprintf(out->fp,
                        "    __%ssort_vector_field(%s, %.*s, %s, t)\n",
                        nsc, snt.text, n, s, snref.text);
                break;
            /* Union vectors cannot be sorted. */
            default:
                break;
            }
            break;
        }
    }
    fprintf(out->fp, "}\n\n");
    return 0;
}

static int gen_table_sorter_prototypes(fb_output_t *out)
{
    fb_symbol_t *sym;
    fb_scoped_name_t snt;
    fb_compound_type_t *ct;

    fb_clear(snt);

    for (sym = out->S->symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_table:
            ct = (fb_compound_type_t *)sym;
            if (ct->export_index & sortable) {
                fb_compound_name(ct, &snt);
                fprintf(out->fp,
                        "static void %s_sort(%s_mutable_table_t t);\n",
                        snt.text, snt.text);
            }
        }
    }
    fprintf(out->fp, "\n");
    return 0;
}

static int gen_union_sorters(fb_output_t *out)
{
    fb_symbol_t *sym;
    fb_compound_type_t *ct;

    for (sym = out->S->symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_union:
            ct = (fb_compound_type_t *)sym;
            if (ct->export_index & sortable) {
                gen_union_sorter(out, ct);
            }
            break;
        default:
            break;
        }
    }
    return 0;
}

static int gen_table_sorters(fb_output_t *out)
{
    fb_symbol_t *sym;
    fb_compound_type_t *ct;

    for (sym = out->S->symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_table:
            ct = (fb_compound_type_t *)sym;
            if (ct->export_index & sortable) {
                gen_table_sorter(out, ct);
            }
            break;
        default:
            break;
        }
    }
    return 0;
}

/*
 * Return 1 if the table or union is known to be sortable,
 * and 0 if that information is not available.
 *
 * Note that if neither a table nor its direct children have
 * sortable vectors, the table might still be sortable via a
 * union member or via deeper nested tables. By iterating
 * repeatedly over all objects, the indirect_sortable
 * property eventually propagetes to all affected objects.
 * At that point no object will change its return value
 * on repeated calls.
 */
static int mark_member_sortable(fb_compound_type_t *ct)
{
    fb_symbol_t *sym;
    fb_member_t *member;

    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        if (member->metadata_flags & fb_f_deprecated) {
            continue;
        }
        if (member->metadata_flags & fb_f_sorted) {
            ct->export_index |= direct_sortable;
        }
        switch (member->type.type) {
        case vt_compound_type_ref:
            switch (member->type.ct->symbol.kind) {
            case fb_is_table:
            case fb_is_union:
                break;
            default:
                continue;
            }
            break;
        case vt_vector_compound_type_ref:
            switch (member->type.ct->symbol.kind) {
            case fb_is_table:
            case fb_is_union:
                break;
            default:
                continue;
            }
            break;
        default:
            continue;
        }
        if (member->type.ct->export_index & (sortable | indirect_sortable)) {
            ct->export_index |= indirect_sortable;
        }
    }
    return !!(ct->export_index & sortable);
}

static void init_sortable(fb_compound_type_t *ct)
{
    fb_symbol_t *sym;
    fb_member_t *member;

    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        switch (member->type.type) {
        case vt_compound_type_ref:
        case vt_vector_compound_type_ref:
            member->type.ct->export_index = 0;
            break;
        default:
            continue;
        }
    }
    ct->export_index = 0;
}

/*
 * Use fix-point iteration to implement a breadth first
 * search for tables and unions that can be sorted. The
 * problem is slightly tricky due to self-referential types:
 * a graph colored depth-first search might terminate before
 * it is known whether any non-direct descendants are
 * sortable.
 */
static int mark_sortable(fb_output_t *out)
{
    fb_symbol_t *sym;
    int old_count = -1, count = 0;

    /* Initialize state kept in the custom export_index symbol table field. */
    for (sym = out->S->symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_table:
        case fb_is_union:
            init_sortable((fb_compound_type_t *)sym);
            break;
        }
    }
    /* Perform fix-point iteration search. */
    while (old_count != count) {
        old_count = count;
        count = 0;
        for (sym = out->S->symbols; sym; sym = sym->link) {
            switch (sym->kind) {
            case fb_is_table:
            case fb_is_union:
                count += mark_member_sortable((fb_compound_type_t *)sym);
                break;
            }
        }
    }
    return 0;
}

/* To be generated towards the end of _reader.h when sort option is active. */
int fb_gen_c_sorter(fb_output_t *out)
{
    mark_sortable(out);
    gen_table_sorter_prototypes(out);
    gen_union_sorters(out);
    gen_table_sorters(out);
    return 0;
}
