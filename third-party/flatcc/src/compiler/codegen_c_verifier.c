#include "codegen_c.h"

#include "flatcc/flatcc_types.h"

/* -DFLATCC_PORTABLE may help if inttypes.h is missing. */
#ifndef PRId64
#include <inttypes.h>
#endif

static int gen_verifier_pretext(fb_output_t *out)
{
    fprintf(out->fp,
        "#ifndef %s_VERIFIER_H\n"
        "#define %s_VERIFIER_H\n",
        out->S->basenameup, out->S->basenameup);

    fprintf(out->fp, "\n/* " FLATCC_GENERATED_BY " */\n\n");
    /* Needed to get the file identifiers */
    fprintf(out->fp, "#ifndef %s_READER_H\n", out->S->basenameup);
    fprintf(out->fp, "#include \"%s_reader.h\"\n", out->S->basename);
    fprintf(out->fp, "#endif\n");
    fprintf(out->fp, "#include \"flatcc/flatcc_verifier.h\"\n");
    fb_gen_c_includes(out, "_verifier.h", "_VERIFIER_H");
    gen_prologue(out);
    fprintf(out->fp, "\n");
    return 0;
}

static int gen_verifier_footer(fb_output_t *out)
{
    gen_epilogue(out);
    fprintf(out->fp,
        "#endif /* %s_VERIFIER_H */\n",
        out->S->basenameup);
    return 0;
}

static int gen_union_verifier(fb_output_t *out, fb_compound_type_t *ct)
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
            "static int %s_union_verifier(flatcc_union_verifier_descriptor_t *ud)\n{\n    switch (ud->type) {\n",
            snt.text);
    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        symbol_name(sym, &n, &s);
        switch (member->type.type) {
        case vt_missing:
            /* NONE is of type vt_missing and already handled. */
            continue;
        case vt_compound_type_ref:
            fb_compound_name(member->type.ct, &snref);
            switch (member->type.ct->symbol.kind) {
            case fb_is_table:
                fprintf(out->fp,
                        "    case %u: return flatcc_verify_union_table(ud, %s_verify_table); /* %.*s */\n",
                        (unsigned)member->value.u, snref.text, n, s);
                continue;
            case fb_is_struct:
                fprintf(out->fp,
                        "    case %u: return flatcc_verify_union_struct(ud, %"PRIu64", %"PRIu16"); /* %.*s */\n",
                        (unsigned)member->value.u, member->type.ct->size, member->type.ct->align, n, s);
                continue;
            default:
                gen_panic(out, "internal error: unexpected compound type for union verifier");
                return -1;
            }
        case vt_string_type:
            fprintf(out->fp,
                    "    case %u: return flatcc_verify_union_string(ud); /* %.*s */\n",
                    (unsigned)member->value.u, n, s);
            continue;
        default:
            gen_panic(out, "internal error: unexpected type for union verifier");
            return -1;
        }
    }
    fprintf(out->fp,
            "    default: return flatcc_verify_ok;\n    }\n}\n\n");
    return 0;
}

static int gen_table_verifier(fb_output_t *out, fb_compound_type_t *ct)
{
    fb_symbol_t *sym;
    fb_member_t *member;
    fb_scoped_name_t snt, snref;
    int required, first = 1;
    const char *nsc = out->nsc;

    fb_clear(snt);
    fb_clear(snref);
    fb_compound_name(ct, &snt);

    fprintf(out->fp,
            "static int %s_verify_table(flatcc_table_verifier_descriptor_t *td)\n{\n",
            snt.text);

    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        if (member->metadata_flags & fb_f_deprecated) {
            continue;
        }

        if (first) {
            fprintf(out->fp, "    int ret;\n    if ((ret = ");
        } else {
            fprintf(out->fp, ")) return ret;\n    if ((ret = ");
        }
        first = 0;
        required = (member->metadata_flags & fb_f_required) != 0;
        switch (member->type.type) {
        case vt_scalar_type:
            fprintf(
                    out->fp,
                    "flatcc_verify_field(td, %"PRIu64", %"PRIu64", %"PRIu16")",
                    member->id, member->size, member->align);
            break;
        case vt_vector_type:
            if (member->nest) {
                fb_compound_name((fb_compound_type_t *)&member->nest->symbol, &snref);
                if (member->nest->symbol.kind == fb_is_table) {
                    fprintf(out->fp,
                        "flatcc_verify_table_as_nested_root(td, %"PRIu64", "
                        "%u, 0, %"PRIu16", %s_verify_table)",
                        member->id, required, member->align, snref.text);
                } else {
                    fprintf(out->fp,
                        "flatcc_verify_struct_as_nested_root(td, %"PRIu64", "
                        "%u, 0, %"PRIu64",  %"PRIu16")",
                        member->id, required, member->size, member->align);
                }
            } else {
                fprintf(out->fp,
                        "flatcc_verify_vector_field(td, %"PRIu64", %d, %"PRIu64", %"PRIu16", INT64_C(%"PRIu64"))",
                        member->id, required, member->size, member->align, (uint64_t)FLATBUFFERS_COUNT_MAX(member->size));
            };
            break;
        case vt_string_type:
            fprintf(out->fp,
                    "flatcc_verify_string_field(td, %"PRIu64", %d)",
                    member->id, required);
            break;
        case vt_vector_string_type:
            fprintf(out->fp,
                    "flatcc_verify_string_vector_field(td, %"PRIu64", %d)",
                    member->id, required);
            break;
        case vt_compound_type_ref:
            fb_compound_name(member->type.ct, &snref);
            switch (member->type.ct->symbol.kind) {
            case fb_is_enum:
            case fb_is_struct:
                fprintf(out->fp,
                        "flatcc_verify_field(td, %"PRIu64", %"PRIu64", %"PRIu16")",
                        member->id, member->size, member->align);
                break;
            case fb_is_table:
                fprintf(out->fp,
                        "flatcc_verify_table_field(td, %"PRIu64", %d, &%s_verify_table)",
                        member->id, required, snref.text);
                break;
            case fb_is_union:
                fprintf(out->fp,
                        "flatcc_verify_union_field(td, %"PRIu64", %d, &%s_union_verifier)",
                        member->id, required, snref.text);
                break;
            default:
                gen_panic(out, "internal error: unexpected compound type for table verifier");
                return -1;
            }
            break;
        case vt_vector_compound_type_ref:
            fb_compound_name(member->type.ct, &snref);
            switch (member->type.ct->symbol.kind) {
            case fb_is_table:
                fprintf(out->fp,
                        "flatcc_verify_table_vector_field(td, %"PRIu64", %d, &%s_verify_table)",
                        member->id, required, snref.text);
                break;
            case fb_is_enum:
            case fb_is_struct:
                fprintf(out->fp,
                        "flatcc_verify_vector_field(td, %"PRIu64", %d, %"PRIu64", %"PRIu16", INT64_C(%"PRIu64"))",
                        member->id, required, member->size, member->align, (uint64_t)FLATBUFFERS_COUNT_MAX(member->size));
                break;
            case fb_is_union:
                fprintf(out->fp,
                        "flatcc_verify_union_vector_field(td, %"PRIu64", %d, &%s_union_verifier)",
                        member->id, required, snref.text);
                break;
            default:
                gen_panic(out, "internal error: unexpected vector compound type for table verifier");
                return -1;
            }
            break;
        }
        fprintf(out->fp, " /* %.*s */", (int)sym->ident->len, sym->ident->text);
    }
    if (!first) {
        fprintf(out->fp, ")) return ret;\n");
    }
    fprintf(out->fp, "    return flatcc_verify_ok;\n");
    fprintf(out->fp, "}\n\n");
    fprintf(out->fp,
            "static inline int %s_verify_as_root(const void *buf, size_t bufsiz)\n"
            "{\n    return flatcc_verify_table_as_root(buf, bufsiz, %s_identifier, &%s_verify_table);\n}\n\n",
            snt.text, snt.text, snt.text);
    fprintf(out->fp,
            "static inline int %s_verify_as_root_with_size(const void *buf, size_t bufsiz)\n"
            "{\n    return flatcc_verify_table_as_root_with_size(buf, bufsiz, %s_identifier, &%s_verify_table);\n}\n\n",
            snt.text, snt.text, snt.text);
    fprintf(out->fp,
            "static inline int %s_verify_as_typed_root(const void *buf, size_t bufsiz)\n"
            "{\n    return flatcc_verify_table_as_root(buf, bufsiz, %s_type_identifier, &%s_verify_table);\n}\n\n",
            snt.text, snt.text, snt.text);
    fprintf(out->fp,
            "static inline int %s_verify_as_typed_root_with_size(const void *buf, size_t bufsiz)\n"
            "{\n    return flatcc_verify_table_as_root_with_size(buf, bufsiz, %s_type_identifier, &%s_verify_table);\n}\n\n",
            snt.text, snt.text, snt.text);
    fprintf(out->fp,
            "static inline int %s_verify_as_root_with_identifier(const void *buf, size_t bufsiz, const char *fid)\n"
            "{\n    return flatcc_verify_table_as_root(buf, bufsiz, fid, &%s_verify_table);\n}\n\n",
            snt.text, snt.text);
    fprintf(out->fp,
            "static inline int %s_verify_as_root_with_identifier_and_size(const void *buf, size_t bufsiz, const char *fid)\n"
            "{\n    return flatcc_verify_table_as_root_with_size(buf, bufsiz, fid, &%s_verify_table);\n}\n\n",
            snt.text, snt.text);
    fprintf(out->fp,
            "static inline int %s_verify_as_root_with_type_hash(const void *buf, size_t bufsiz, %sthash_t thash)\n"
            "{\n    return flatcc_verify_table_as_typed_root(buf, bufsiz, thash, &%s_verify_table);\n}\n\n",
            snt.text, nsc, snt.text);
    fprintf(out->fp,
            "static inline int %s_verify_as_root_with_type_hash_and_size(const void *buf, size_t bufsiz, %sthash_t thash)\n"
            "{\n    return flatcc_verify_table_as_typed_root_with_size(buf, bufsiz, thash, &%s_verify_table);\n}\n\n",
            snt.text, nsc, snt.text);
    return 0;
}

static int gen_struct_verifier(fb_output_t *out, fb_compound_type_t *ct)
{
    fb_scoped_name_t snt;

    fb_clear(snt);
    fb_compound_name(ct, &snt);

    fprintf(out->fp,
            "static inline int %s_verify_as_root(const void *buf, size_t bufsiz)\n"
            "{\n    return flatcc_verify_struct_as_root(buf, bufsiz, %s_identifier, %"PRIu64", %"PRIu16");\n}\n\n",
            snt.text, snt.text, ct->size, ct->align);
    fprintf(out->fp,
            "static inline int %s_verify_as_root_with_size(const void *buf, size_t bufsiz)\n"
            "{\n    return flatcc_verify_struct_as_root_with_size(buf, bufsiz, %s_identifier, %"PRIu64", %"PRIu16");\n}\n\n",
            snt.text, snt.text, ct->size, ct->align);
    fprintf(out->fp,
            "static inline int %s_verify_as_typed_root(const void *buf, size_t bufsiz)\n"
            "{\n    return flatcc_verify_struct_as_typed_root(buf, bufsiz, %s_type_hash, %"PRIu64", %"PRIu16");\n}\n\n",
            snt.text, snt.text, ct->size, ct->align);
    fprintf(out->fp,
            "static inline int %s_verify_as_typed_root_with_size(const void *buf, size_t bufsiz)\n"
            "{\n    return flatcc_verify_struct_as_typed_root_with_size(buf, bufsiz, %s_type_hash, %"PRIu64", %"PRIu16");\n}\n\n",
            snt.text, snt.text, ct->size, ct->align);
    fprintf(out->fp,
            "static inline int %s_verify_as_root_with_type_hash(const void *buf, size_t bufsiz, %sthash_t thash)\n"
            "{\n    return flatcc_verify_struct_as_typed_root(buf, bufsiz, thash, %"PRIu64", %"PRIu16");\n}\n\n",
            snt.text, out->nsc, ct->size, ct->align);
    fprintf(out->fp,
            "static inline int %s_verify_as_root_with_type_hash_and_size(const void *buf, size_t bufsiz, %sthash_t thash)\n"
            "{\n    return flatcc_verify_struct_as_typed_root_with_size(buf, bufsiz, thash, %"PRIu64", %"PRIu16");\n}\n\n",
            snt.text, out->nsc, ct->size, ct->align);
    fprintf(out->fp,
            "static inline int %s_verify_as_root_with_identifier(const void *buf, size_t bufsiz, const char *fid)\n"
            "{\n    return flatcc_verify_struct_as_root(buf, bufsiz, fid, %"PRIu64", %"PRIu16");\n}\n\n",
            snt.text, ct->size, ct->align);
    fprintf(out->fp,
            "static inline int %s_verify_as_root_with_identifier_and_size(const void *buf, size_t bufsiz, const char *fid)\n"
            "{\n    return flatcc_verify_struct_as_root_with_size(buf, bufsiz, fid, %"PRIu64", %"PRIu16");\n}\n\n",
            snt.text, ct->size, ct->align);
    return 0;
}

static int gen_verifier_prototypes(fb_output_t *out)
{
    fb_symbol_t *sym;
    fb_scoped_name_t snt;

    fb_clear(snt);

    for (sym = out->S->symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_table:
            fb_compound_name((fb_compound_type_t *)sym, &snt);
            fprintf(out->fp,
                    "static int %s_verify_table(flatcc_table_verifier_descriptor_t *td);\n",
                    snt.text);
        }
    }
    fprintf(out->fp, "\n");
    return 0;
}

static int gen_union_verifiers(fb_output_t *out)
{
    fb_symbol_t *sym;

    for (sym = out->S->symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_union:
            gen_union_verifier(out, (fb_compound_type_t *)sym);
        }
    }
    return 0;
}

static int gen_struct_verifiers(fb_output_t *out)
{
    fb_symbol_t *sym;

    for (sym = out->S->symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_struct:
            gen_struct_verifier(out, (fb_compound_type_t *)sym);
        }
    }
    return 0;
}

static int gen_table_verifiers(fb_output_t *out)
{
    fb_symbol_t *sym;

    for (sym = out->S->symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_table:
            gen_table_verifier(out, (fb_compound_type_t *)sym);
        }
    }
    return 0;
}

int fb_gen_c_verifier(fb_output_t *out)
{
    gen_verifier_pretext(out);
    gen_verifier_prototypes(out);
    gen_union_verifiers(out);
    gen_struct_verifiers(out);
    gen_table_verifiers(out);
    gen_verifier_footer(out);
    return 0;
}
