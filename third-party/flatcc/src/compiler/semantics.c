#include <string.h>
#include <assert.h>

#include "semantics.h"
#include "parser.h"
#include "coerce.h"
#include "stdio.h"

/* -DFLATCC_PORTABLE may help if inttypes.h is missing. */
#ifndef PRId64
#include <inttypes.h>
#endif

/* Same order as enum! */
static const char *fb_known_attribute_names[] = {
    "",
    "id",
    "deprecated",
    "original_order",
    "force_align",
    "bit_flags",
    "nested_flatbuffer",
    "key",
    "required",
    "hash",
    "base64",
    "base64url",
    "primary_key",
    "sorted",
};

static const char *fb_reserved_kw_enum_members[] = {
    "_const_ptr_add",
    "_ptr_add",
    "_size",
    "array_copy",
    "array_copy_from_pe",
    "array_copy_to_pe",
    "assign",
    "assign_from_pe",
    "assign_to_pe",
    "cast_from_be",
    "cast_from_le",
    "cast_from_pe",
    "cast_to_be",
    "cast_to_le",
    "cast_to_pe",
    "copy",
    "copy_from_pe",
    "copy_to_pe",
    "enum_t",
    "from_pe",
    "is_known_value",
    "name",
    "parse_json_enum",
    "print_json_enum",
    "read",
    "read_from_pe",
    "read_to_pe",
    "to_pe",
    // "vec_*",
    "write",
    "write_from_pe",
    "write_to_pe",
};

static const char *fb_reserved_kw_table_fields[] = {
      "add_type",
      "add_value",
      "begin",
      "clone",
      "create",
      "end",
      "end_pe",
      "force_add",
      "table_t",
      "verity_table",
};

static const char *fb_reserved_kw_table_field_prefixes[] = {
      "as_",
      "clone_as_",
      "create_as_",
      "end_as_",
      "parse_json_",
      "print_json_",
      "start_as_",
      // "vec_  ",
      "verify_as_*",
};

static const char *fb_reserved_kw_table_field_suffixes[] = {
      "_add",
      "_clone",
      "_create",
      "_end",
      "_get",
      "_get_ptr",
      "_force_add",
      "_is_present",
      "_pick",
      "_push_start",
      "_push_end",
      "_push_create",
      "_start",
};

/* For table and enum members. */
static const char *fb_reserved_kw_vec_prefixes[] = {
      "vec_",
};

static const int fb_known_attribute_types[] = {
    vt_invalid, /* Unknowns have arbitrary types. */
    vt_uint,
    vt_missing,
    vt_missing,
    vt_uint,
    vt_missing,
    vt_string,
    vt_missing,
    vt_missing,
    vt_string,
    vt_missing,
    vt_missing,
    vt_missing,
    vt_missing,
};

static fb_scalar_type_t map_scalar_token_type(fb_token_t *t)
{
    switch (t->id) {
    case tok_kw_uint64:
    case tok_kw_ulong:
        return fb_ulong;
    case tok_kw_uint32:
    case tok_kw_uint:
        return fb_uint;
    case tok_kw_uint16:
    case tok_kw_ushort:
        return fb_ushort;
    case tok_kw_uint8:
    case tok_kw_ubyte:
        return fb_ubyte;
    case tok_kw_char:
        return fb_char;
    case tok_kw_bool:
        return fb_bool;
    case tok_kw_int64:
    case tok_kw_long:
        return fb_long;
    case tok_kw_int32:
    case tok_kw_int:
        return fb_int;
    case tok_kw_int16:
    case tok_kw_short:
        return fb_short;
    case tok_kw_int8:
    case tok_kw_byte:
        return fb_byte;
    case tok_kw_float64:
    case tok_kw_double:
        return fb_double;
    case tok_kw_float32:
    case tok_kw_float:
        return fb_float;
    default:
        return fb_missing_type;
    }
}

/*
 * The flatc compiler currently has a 256 limit.
 *
 * Some target C compilers might respect anything above
 * 16 and may reguire that PAD option of the C code generator.
 */
static inline int is_valid_align(uint64_t align)
{
    uint64_t n = 1;
    if (align == 0 || align > FLATCC_FORCE_ALIGN_MAX) {
        return 0;
    }
    while (n <= align) {
        if (n == align) {
            return 1;
        }
        n *= 2;
    }
    return 0;
}

static inline uint64_t fb_align(uint64_t size, uint64_t align)
{
    assert(is_valid_align(align));

    return (size + align - 1) & ~(align - 1);
}

/*
 * The FNV-1a 32-bit little endian hash is a FlatBuffers standard for
 * transmission of type identifiers in a compact form, in particular as
 * alternative file identifiers. Note that if hash becomes 0, we map it
 * to hash("").
 */
static inline void set_type_hash(fb_compound_type_t *ct)
{
    fb_ref_t *name;
    fb_symbol_t *sym;
    uint32_t hash;

    hash = fb_hash_fnv1a_32_init();
    if (ct->scope) {
        for (name = ct->scope->name; name; name = name->link) {
            hash = fb_hash_fnv1a_32_append(hash, name->ident->text, (size_t)name->ident->len);
            hash = fb_hash_fnv1a_32_append(hash, ".", 1);
        }
    }
    sym = &ct->symbol;
    hash = fb_hash_fnv1a_32_append(hash, sym->ident->text, (size_t)sym->ident->len);
    if (hash == 0) {
        hash = fb_hash_fnv1a_32_init();
    }
    ct->type_hash = hash;
}

static inline fb_scope_t *fb_find_scope_by_string(fb_schema_t *S, const char *name, size_t len)
{
    if (!S || !S->root_schema) {
        return 0;
    }
    if (len == 0) {
        /* Global scope. */
        name = 0;
    }
    return fb_scope_table_find(&S->root_schema->scope_index, name, len);
}

/* count = 0 indicates zero-terminated ref list, name = 0 indicates global scope. */
static inline fb_scope_t *fb_find_scope_by_ref(fb_schema_t *S, const fb_ref_t *name, int count)
{
    if (!S || !S->root_schema) {
        return 0;
    }
    return fb_scope_table_find(&S->root_schema->scope_index, name, (size_t)(-count));
}

static inline fb_symbol_t *define_fb_symbol(fb_symbol_table_t *si, fb_symbol_t *sym)
{
    return fb_symbol_table_insert_item(si, sym, ht_keep);
}

static inline fb_symbol_t *find_fb_symbol_by_token(fb_symbol_table_t *si, fb_token_t *token)
{
    return fb_symbol_table_find(si, token->text, (size_t)token->len);
}

static inline fb_name_t *define_fb_name(fb_name_table_t *ni, fb_name_t *name)
{
    return fb_name_table_insert_item(ni, name, ht_keep);
}

static inline fb_name_t *find_fb_name_by_token(fb_name_table_t *ni, fb_token_t *token)
{
    return fb_name_table_find(ni, token->text, (size_t)token->len);
}

/* Returns 1 if value exists, 0 otherwise, */
static inline int add_to_value_set(fb_value_set_t *vs, fb_value_t *value)
{
    return fb_value_set_insert_item(vs, value, ht_keep) != 0;
}

static inline int is_in_value_set(fb_value_set_t *vs, fb_value_t *value)
{
    return 0 != fb_value_set_find_item(vs, value);
}

/*
 * An immediate parent scope does not necessarily exist and it might
 * appear in a later search, so we return the nearest existing parent
 * and do not cache the parent.
 */
static inline fb_scope_t *find_parent_scope(fb_parser_t *P, fb_scope_t *scope)
{
    fb_ref_t *p;
    int count;
    fb_scope_t *parent;

    parent = 0;
    count = 0;
    if (scope == 0) {
        return 0;
    }
    p = scope->name;
    while (p) {
        ++count;
        p = p->link;
    }
    if (count == 0) {
        return 0;
    }
    while (count-- > 1) {
        if ((parent = fb_find_scope_by_ref(&P->schema, scope->name, count))) {
            return parent;
        }
    }
    /* Root scope. */
    return fb_find_scope_by_ref(&P->schema, 0, 0);
}

static inline fb_symbol_t *lookup_string_reference(fb_parser_t *P, fb_scope_t *local, const char *s, size_t len)
{
    fb_symbol_t *sym;
    fb_scope_t *scope;
    const char *name, *basename;
    size_t k;

    name = s;
    basename = s;
    k = len;
    while (k > 0) {
        if (s[--k] == '.') {
            basename = s + k + 1;
            --len;
            break;
        }
    }
    len -= k;
    if (local && k == 0) {
        do {
            if ((sym = fb_symbol_table_find(&local->symbol_index, basename, len))) {
                if (get_compound_if_visible(&P->schema, sym)) {
                    return sym;
                }
            }
            local = find_parent_scope(P, local);
        } while (local);
        return 0;
    }
    if (!(scope = fb_find_scope_by_string(&P->schema, name, k))) {
        return 0;
    }
    return fb_symbol_table_find(&scope->symbol_index, basename, len);
}

/*
 * First search the optional local scope, then the scope of the namespace prefix if any.
 * If `enumval` is non-zero, the last namepart is stored in that
 * pointer and the lookup stops before that part.
 *
 * If the reference is prefixed with a namespace then the scope is
 * looked up relative to root then the basename is searched in that
 * scope.
 *
 * If the refernce is not prefixed with a namespace then the name is
 * search in the local symbol table (which may be the root if null) and
 * if that fails, the nearest existing parent scope is used as the new
 * local scope and the process is repeated until local is root.
 *
 * This means that namespace prefixes cannot be relative to a parent
 * namespace or to the current scope, but simple names can be found in a
 * parent namespace.
 */
static inline fb_symbol_t *lookup_reference(fb_parser_t *P, fb_scope_t *local, fb_ref_t *name, fb_ref_t **enumval)
{
    fb_ref_t *basename, *last, *p;
    fb_scope_t *scope;
    fb_symbol_t *sym;
    int count;

    count = 0;
    scope = 0;
    p = name;
    last = 0;
    basename = 0;
    while (p) {
        basename = last;
        last = p;
        p = p->link;
        ++count;
    }
    if (enumval) {
        --count;
        *enumval = last;
    } else {
        basename = last;
    }
    if (!basename) {
        return 0;
    }
    if (local && count == 1) {
        do {
            if ((sym = find_fb_symbol_by_token(&local->symbol_index, basename->ident))) {
                if (get_compound_if_visible(&P->schema, sym)) {
                    return sym;
                }
            }
            local = find_parent_scope(P, local);
        } while (local);
        return 0;
    }
    /* Null name is valid in scope lookup, indicating global scope. */
    if (count == 1) {
        name = 0;
    }
    if (!(scope = fb_find_scope_by_ref(&P->schema, name, count - 1))) {
        return 0;
    }
    sym = find_fb_symbol_by_token(&scope->symbol_index, basename->ident);
    if (sym && get_compound_if_visible(&P->schema, sym)) {
        return sym;
    }
    return 0;
}

static inline fb_symbol_t *lookup_type_reference(fb_parser_t *P, fb_scope_t *local, fb_ref_t *name)
{
    return lookup_reference(P, local, name, 0);
}

/*
 * `ct` is null when looking up names for scalar types and otherwise it is
 * the enum type being assigned. The provided reference may reference
 * an enum value in the `ct` type, or another enum if a scope/type is
 * given.
 */
static inline int lookup_enum_name(fb_parser_t *P, fb_scope_t *local, fb_compound_type_t *ct, fb_ref_t *ref, fb_value_t *value)
{
    fb_symbol_t *sym;
    fb_ref_t *enumval;
    fb_member_t *member;

    enumval = 0;
    assert(ref);
    assert(ct == 0 || ct->symbol.kind == fb_is_enum);
    sym = lookup_reference(P, local, ref, &enumval);
    if (sym && sym->kind == fb_is_enum) {
        ct = (fb_compound_type_t *)sym;
    } else if (ref->link) {
        /* If there was a scope / type prefix, it was not found, or it was not an enum type. */
        return -1;
    }
    if (!ct) {
        return -1;
    }
    sym = find_fb_symbol_by_token(&ct->index, enumval->ident);
    if (!sym) {
        return -1;
    }
    member = (fb_member_t *)sym;
    *value = member->value;
    return 0;
}

/* This is repeated for every include file, but this pose no problem. */
static void install_known_attributes(fb_parser_t *P)
{
    unsigned int i;
    fb_attribute_t *a;

    for (i = 0; i < KNOWN_ATTR_COUNT; ++i) {
        /* Don't put it in the parsed list, just the index. */
        a = new_elem(P, sizeof(*a));
        a->known = i;
        a->name.name.s.s = (char *)fb_known_attribute_names[i];
        a->name.name.s.len = (int)strlen(fb_known_attribute_names[i]);
        a->name.name.type = vt_string;
        a->name.link = 0;
        if ((a = (fb_attribute_t *)define_fb_name(&P->schema.root_schema->attribute_index, &a->name))) {
            /*
             * If the user alredy defined the attribute, keep that instead.
             * (Memory leak is ok here.)
             */
            a->known = i;
        }
    }
}

static void install_reserved_keyword_table(fb_parser_t *P, const char *table[], size_t count, int kind)
{
    fb_reserved_kw_t *kw;
    size_t i;

    for (i = 0; i < count; ++i) {
        kw = new_elem(P, sizeof(*kw));
        kw->name.name.s.s = (char *)table[i];
        kw->name.name.s.len = (int)strlen(table[i]);
        kw->name.link = 0;
        kw->kind = kind;
        define_fb_name(&P->schema.root_schema->keyword_index, &kw->name);
    }
}

#define KW_TABLE_LEN(T) (sizeof(T)/sizeof(T[0]))

static void install_reserved_keywords(fb_parser_t *P)
{
    install_reserved_keyword_table(P, fb_reserved_kw_enum_members,
                                   KW_TABLE_LEN(fb_reserved_kw_enum_members),
                                   fb_reserved_kw_kind_enum_member);
    install_reserved_keyword_table(P, fb_reserved_kw_table_fields,
                                   KW_TABLE_LEN(fb_reserved_kw_table_fields),
                                   fb_reserved_kw_kind_table_field);
    install_reserved_keyword_table(P, fb_reserved_kw_table_field_prefixes,
                                   KW_TABLE_LEN(fb_reserved_kw_table_field_prefixes),
                                   fb_reserved_kw_kind_table_field_prefix);
    install_reserved_keyword_table(P, fb_reserved_kw_table_field_suffixes,
                                   KW_TABLE_LEN(fb_reserved_kw_table_field_suffixes),
                                   fb_reserved_kw_kind_table_field_suffix);
    install_reserved_keyword_table(P, fb_reserved_kw_vec_prefixes,
                                   KW_TABLE_LEN(fb_reserved_kw_vec_prefixes),
                                   fb_reserved_kw_kind_vec_prefix);
}

static void check_reserved_enum_member(fb_parser_t *P, fb_symbol_t *sym)
{
    const char *text = sym->ident->text;
    size_t len = (size_t)sym->ident->len;
    fb_reserved_kw_t *kw = 0;
    size_t i;

    kw = (fb_reserved_kw_t *)fb_name_table_find(&P->schema.root_schema->keyword_index,
                                                         text, len);
    if (kw && kw->kind == fb_reserved_kw_kind_enum_member) {
        warn_tok(P, sym->ident, "potential enum member name conflict in C source");
        warn_tok_as_string(P, sym->ident, "name is reserved", kw->name.name.s.s, (size_t)kw->name.name.s.len);
        return;
    }
    for (i = 0; i < len; ++i) {
        if (text[i] =='_') {
            if (i + 1 < len) {
                kw = (fb_reserved_kw_t *)fb_name_table_find(&P->schema.root_schema->keyword_index,
                                                         text, i + 1);
                if (kw && (kw->kind == fb_reserved_kw_kind_vec_prefix)) {
                    warn_tok(P, sym->ident, "potential enum member name conflict in C source");
                    warn_tok_as_string(P, sym->ident, "prefix is reserved", kw->name.name.s.s, (size_t)kw->name.name.s.len);
                    return;
                }
            }
        }
    }
}

static void check_reserved_table_field(fb_parser_t *P, fb_symbol_t *sym)
{
    const char *text = sym->ident->text;
    size_t len = (size_t)sym->ident->len;
    fb_reserved_kw_t *kw = 0;
    size_t i;

    kw = (fb_reserved_kw_t *)fb_name_table_find(&P->schema.root_schema->keyword_index,
                                                         text, len);
    if (kw && kw->kind == fb_reserved_kw_kind_table_field) {
        warn_tok(P, sym->ident, "potential table field conflict in C source (option -g might help)");
        warn_tok_as_string(P, sym->ident, "name is reserved", kw->name.name.s.s, (size_t)kw->name.name.s.len);
        return;
    }
    for (i = 0; i < len; ++i) {
        if (text[i] =='_') {
            if (i + 1 < len) {
                kw = (fb_reserved_kw_t *)fb_name_table_find(&P->schema.root_schema->keyword_index,
                                                         text, i + 1);
                if (kw && (kw->kind == fb_reserved_kw_kind_table_field_prefix
                    || kw->kind == fb_reserved_kw_kind_vec_prefix)) {
                    warn_tok(P, sym->ident, "potential table field conflict in C source (option -g might help)");
                    warn_tok_as_string(P, sym->ident, "prefix is reserved", kw->name.name.s.s, (size_t)kw->name.name.s.len);
                    return;
                }
            }
            if (i > 0) {
                kw = (fb_reserved_kw_t *)fb_name_table_find(&P->schema.root_schema->keyword_index,
                                                         text + i, len - i);
                if (kw && kw->kind == fb_reserved_kw_kind_table_field_suffix) {
                    warn_tok(P, sym->ident, "potential table field conflict in C source (option -g might help)");
                    warn_tok_as_string(P, sym->ident, "suffix is reserved", kw->name.name.s.s, (size_t)kw->name.name.s.len);
                    return;
                }
            }
        }
    }
}

static void revert_order(fb_compound_type_t **list) {
    fb_compound_type_t *next, *prev = 0, *link = *list;

    while (link) {
        next = link->order;
        link->order = prev;
        prev = link;
        link = next;
    }
    *list = prev;
}

static inline unsigned short process_metadata(fb_parser_t *P, fb_metadata_t *m,
        uint16_t expect, fb_metadata_t *out[KNOWN_ATTR_COUNT])
{
    uint16_t flags;
    unsigned int i, n = FLATCC_ATTR_MAX;
    int type;
    fb_attribute_t *a;

    memset(out, 0, sizeof(out[0]) * KNOWN_ATTR_COUNT);
    for (flags = 0; m && n; --n, m = m->link) {
        a = (fb_attribute_t *)find_fb_name_by_token(&P->schema.root_schema->attribute_index, m->ident);
        if (!a) {
            error_tok(P, m->ident, "unknown attribute not declared");
            continue;
        }
        if (!(i = a->known)) {
            continue;
        }
        if (!((1 << i) & expect)) {
            error_tok(P, m->ident, "known attribute not expected in this context");
            continue;
        }
        flags |= 1 << i;
        if (out[i]) {
            error_tok(P, m->ident, "known attribute listed multiple times");
            continue;
        }
        out[i] = m;
        type = fb_known_attribute_types[i];
        if (type == vt_missing && m->value.type != vt_missing) {
            error_tok(P, m->ident, "known attribute does not expect a value");
            continue;
        }
        if (type == vt_string && m->value.type != vt_string) {
            error_tok(P, m->ident, "known attribute expects a string");
            continue;
        }
        if (type == vt_uint && m->value.type != vt_uint) {
            error_tok(P, m->ident, "known attribute expects an unsigned integer");
            continue;
        }
        if (type == vt_int && m->value.type != vt_uint && m->value.type != vt_int) {
            error_tok(P, m->ident, "known attribute expects an integer");
            continue;
        }
        if (type == vt_bool && m->value.type != vt_bool) {
            error_tok(P, m->ident, "known attribute expects 'true' or 'false'");
            continue;
        }
    }
    if (m) {
        error_tok(P, m->ident, "too many attributes");
    }
    return flags;
}

/*
 * Recursive types are allowed, according to FlatBuffers Internals doc.,
 * but this cannot be possible for structs because they have no default
 * value or null option, and can only hold scalars and other structs, so
 * recursion would never terminate. Enums are simple types and cannot be
 * recursive either. Unions reference tables which may reference unions,
 * and recursion works well here. Tables allow any other table, union,
 * or scalar value to be optional or default, so recursion is possible.
 * In conclusion, Unions and Table members may reference all other
 * types, and self. Enums are trivially checked because the only allow
 * scalars, which leaves structs that can build illegal forms.
 *
 * Object instances cannot be recursive meaning the object graph is
 * always a tree, but this isn't really a concern for the schema
 * compiler, and for the builder it happens naturally as it only adds to
 * the buffer (though a compressor might reuse old data without
 * violating the tree?).
 *
 * Conclusion: check structs for circular references and allow
 * everything else to unfold, provided they otherwise pass type checks.
 *
 * Algorithm:
 *
 * Depth first search of the struct reference tree. We maintain flags to
 * find back-links. We prune sub-trees already fully analyzed by using
 * the closed flag. This operation is O(N) since each struct member is
 * visited once.
 *
 * Recursion is limited to prevent blowing the stack and to protect
 * against abuse.
 */
static int analyze_struct(fb_parser_t *P, fb_compound_type_t *ct)
{
    fb_symbol_t *sym;
    fb_compound_type_t *type;
    fb_member_t *member;
    int ret = 0;
    uint64_t size;
    uint16_t align;
    fb_token_t *t;

    assert(ct->symbol.kind == fb_is_struct);

    assert(!(ct->symbol.flags & fb_circular_open));
    if (ct->symbol.flags & fb_circular_closed) {
        return 0;
    }
    assert(!ct->order);
    ct->symbol.flags |= fb_circular_open;
    align = 1;
    for (sym = ct->members; sym; sym = sym->link) {
        type = 0;
        if (P->nesting_level >= FLATCC_NESTING_MAX) {
            error(P, "maximum allowed nesting level exceeded while processing struct hierarchy");
            return -1;
        }
        member = (fb_member_t *)sym;
        switch (member->type.type) {
        case vt_fixed_array_type:
        case vt_scalar_type:
            t = member->type.t;
            member->type.st = map_scalar_token_type(t);
            size = sizeof_scalar_type(member->type.st);
            if (size < 1) {
                error_sym_tok(P, sym, "unexpected type", t);
                return -1;
            }
            member->align = (uint16_t)size;
            member->size = size * member->type.len;
            break;
        case vt_fixed_array_compound_type_ref:
        case vt_compound_type_ref:
            /* Enums might not be valid, but then it would be detected earlier. */
            if (member->type.ct->symbol.kind == fb_is_enum) {
                type = member->type.ct;
                size = type->size;
                member->align = (uint16_t)size;
                member->size = member->type.len * type->size;
                break;
            } else if (member->type.ct->symbol.kind == fb_is_struct) {
                type = member->type.ct;
                if (type->symbol.flags & fb_circular_open) {
                    error_sym_2(P, sym, "circular reference to struct at", &type->symbol);
                    return -1;
                }
                if (!(type->symbol.flags & fb_circular_closed)) {
                    if (P->opts.hide_later_struct) {
                        error_sym_2(P, sym, "dependency on later defined struct not permitted with current settings", &type->symbol);
                    }
                    ++P->nesting_level;
                    ret = analyze_struct(P, type);
                    --P->nesting_level;
                    if (ret) {
                        return ret;
                    }
                }
                member->align = type->align;
                member->size = member->type.len * type->size;
                break;
            } else {
                error_sym(P, sym, "unexpected compound type for field");
                return -1;
            }
            break;
        case vt_invalid:
            /* Old error. */
            return -1;
        default:
            error_sym(P, sym, "unexpected type");
            return -1;
        }
        member->offset = fb_align(ct->size, member->align);
        if (member->offset < ct->size || member->offset + member->size < member->offset) {
            error_sym(P, sym, "struct size overflow");
            return -1;
        }
        size = member->offset + member->size;
        if (size < ct->size || size > FLATCC_STRUCT_MAX_SIZE) {
            error_sym(P, sym, "struct field overflows maximum allowed struct size");
        };
        ct->size = size;
        /*
         * FB spec is not very clear on this - but experimentally a
         * force aligned member struct will force that alignment upon a
         * containing struct if the alignment would otherwise be
         * smaller. In otherwise, a struct is aligned to the alignment
         * of the largest member, not just the largest scalar member
         * (directly or indirectly).
         */
        if (align < member->align) {
            align = member->align;
        }
    }
    if (ct->align > 0) {
        if (align > ct->align) {
            error_sym(P, &ct->symbol, "'force_align' cannot be smaller than natural alignment for");
            ct->align = align;
        }
    } else {
        ct->align = align;
    }
    /* Add trailing padding if necessary. */
    ct->size = fb_align(ct->size, ct->align);

    if (ct->size == 0) {
        error_sym(P, &ct->symbol, "struct cannot be empty");
        return -1;
    }

    ct->symbol.flags |= fb_circular_closed;
    ct->symbol.flags &= (uint16_t)~fb_circular_open;
    ct->order = P->schema.ordered_structs;
    P->schema.ordered_structs = ct;
    return ret;
}

static int define_nested_table(fb_parser_t *P, fb_scope_t *local, fb_member_t *member, fb_metadata_t *m)
{
    fb_symbol_t *type_sym;

    if (member->type.type != vt_vector_type || member->type.st != fb_ubyte) {
        error_tok(P, m->ident, "'nested_flatbuffer' attribute requires a [ubyte] vector type");
        return -1;
    }
    if (m->value.type != vt_string) {
        /* All known attributes get automatically type checked, so just ignore. */
        return -1;
    }
    type_sym = lookup_string_reference(P, local, m->value.s.s, (size_t)m->value.s.len);
    if (!type_sym) {
        error_tok_as_string(P, m->ident, "nested reference not found", m->value.s.s, (size_t)m->value.s.len);
        return -1;
    }
    if (type_sym->kind != fb_is_table) {
        if (!P->opts.allow_struct_root) {
            error_tok_2(P, m->ident, "nested reference does not refer to a table", type_sym->ident);
            return -1;
        }
        if (type_sym->kind != fb_is_struct) {
            error_tok_2(P, m->ident, "nested reference does not refer to a table or a struct", type_sym->ident);
            return -1;
        }
    }
    member->nest = (fb_compound_type_t *)type_sym;
    return 0;
}

static int process_struct(fb_parser_t *P, fb_compound_type_t *ct)
{
    fb_symbol_t *sym, *old, *type_sym;
    fb_member_t *member;
    fb_metadata_t *knowns[KNOWN_ATTR_COUNT], *m;
    uint16_t allow_flags;
    int key_count = 0, primary_count = 0, key_ok = 0;

    if (ct->type.type) {
        error_sym(P, &ct->symbol, "internal error: struct cannot have a type");
        return -1;
    }
    ct->metadata_flags = process_metadata(P, ct->metadata, fb_f_force_align, knowns);
    if ((m = knowns[fb_attr_force_align])) {
        if (!is_valid_align(m->value.u)) {
            error_sym(P, &ct->symbol, "'force_align' exceeds maximum permitted alignment or is not a power of 2");
        } else {
            /* This may still fail on natural alignment size. */
            ct->align = (uint16_t)m->value.u;
        }
    }
    for (sym = ct->members; sym; sym = sym->link) {
        if ((old = define_fb_symbol(&ct->index, sym))) {
            error_sym_2(P, sym, "struct field already defined here", old);
            continue;
        }
        if (sym->kind != fb_is_member) {
            error_sym(P, sym, "internal error: field type expected");
            return -1;
        }
        key_ok = 1;
        member = (fb_member_t *)sym;
        allow_flags = 0;
        /* Notice the difference between fb_f_ and fb_attr_ (flag vs index). */
        if (P->opts.allow_struct_field_key) {
            allow_flags |= fb_f_key;
            if (P->opts.allow_primary_key) {
                allow_flags |= fb_f_primary_key;
            }
        }
        if (P->opts.allow_struct_field_deprecate) {
            allow_flags |= fb_f_deprecated;
        }
        member->metadata_flags = process_metadata(P, member->metadata, allow_flags, knowns);
        switch (member->type.type) {
        case vt_fixed_array_type_ref:
            key_ok = 0;
            goto lbl_type_ref;
        case vt_type_ref:
lbl_type_ref:
            type_sym = lookup_type_reference(P, ct->scope, member->type.ref);
            if (!type_sym) {
                error_ref_sym(P, member->type.ref, "unknown type reference used with struct field", sym);
                member->type.type = vt_invalid;
                continue;
            }
            member->type.ct = (fb_compound_type_t*)type_sym;
            member->type.type = member->type.type == vt_fixed_array_type_ref ?
                vt_fixed_array_compound_type_ref : vt_compound_type_ref;
            if (type_sym->kind != fb_is_struct) {
                if (P->opts.allow_enum_struct_field) {
                    if (type_sym->kind != fb_is_enum) {
                        error_sym_2(P, sym, "struct fields can only be scalars, structs, and enums, or arrays of these, but has type", type_sym);
                        member->type.type = vt_invalid;
                        return -1;
                    }
                    if (!P->opts.allow_enum_key) {
                        key_ok = 0;
                        break;
                    }
                } else {
                    error_sym_2(P, sym, "struct fields can only be scalars and structs, or arrays of these, but has type", type_sym);
                    member->type.type = vt_invalid;
                    return -1;
                }
            } else {
                key_ok = 0;
            }
            break;
        case vt_fixed_array_string_type:
            error_sym(P, sym, "fixed length arrays cannot have string elements");
            member->type.type = vt_invalid;
            return -1;
        case vt_fixed_array_type:
            key_ok = 0;
            break;
        case vt_scalar_type:
            break;
        default:
            error_sym(P, sym, "struct member member can only be of struct scalar, or fixed length scalar array type");
            return -1;
        }
        if (!key_ok) {
            if (member->metadata_flags & fb_f_key) {
                member->type.type = vt_invalid;
                error_sym(P, sym, "key attribute now allowed for this kind of field");
                return -1;
            }
            if (member->metadata_flags & fb_f_primary_key) {
                member->type.type = vt_invalid;
                error_sym(P, sym, "primary_key attribute now allowed for this kind of field");
                return -1;
            }
        }
        if (member->metadata_flags & fb_f_deprecated) {
            if (member->metadata_flags & fb_f_key) {
                error_sym(P, sym, "key attribute not allowed for deprecated struct member");
                return -1;
            } else if (member->metadata_flags & fb_f_primary_key) {
                error_sym(P, sym, "primary_key attribute not allowed for deprecated struct member");
                return -1;
            }
        }
        if (member->metadata_flags & fb_f_key) {
            if (member->metadata_flags & fb_f_primary_key) {
                error_sym(P, sym, "primary_key attribute conflicts with key attribute");
                member->type.type = vt_invalid;
                return -1;
            }
            key_count++;
            if (!ct->primary_key) {
                /* First key is primary key if no primary key is given explicitly. */
                ct->primary_key = member;
            }
        } else if (member->metadata_flags & fb_f_primary_key) {
            if (primary_count++) {
                error_sym(P, sym, "at most one struct member can have a primary_key attribute");
                member->type.type = vt_invalid;
                return -1;
            }
            key_count++;
            /* Allow backends to treat primary key as an ordinary key. */
            member->metadata_flags |= fb_f_key;
            ct->primary_key = member;
        }
        if (member->value.type) {
            error_sym(P, sym, "struct member member cannot have a default value");
            continue;
        }
    }
    if (key_count) {
        ct->symbol.flags |= fb_indexed;
    }
    /* Set primary key flag for backends even if chosen by default. */
    if (ct->primary_key) {
        ct->primary_key->metadata_flags |= fb_f_primary_key;
    }
    if (key_count > 1 && !P->opts.allow_multiple_key_fields) {
        error_sym(P, &ct->symbol, "table has multiple key fields, but at most one is permitted");
        return -1;
    }
    return 0;
}

static fb_member_t *original_order_members(fb_parser_t *P, fb_member_t *next)
{
    fb_member_t *head = 0;
    fb_member_t **tail = &head;

    /* Not used for now, but in case we need error messages etc. */
    (void)P;

    while (next) {
        *tail = next;
        tail = &next->order;
        next = (fb_member_t *)(((fb_symbol_t *)next)->link);
    }
    *tail = 0;
    return head;
}

/*
 * Alignment of table offset fields are generally not stored, and
 * vectors store the element alignment for scalar types, so we
 * detect alignment based on type also. Unions are tricky since they
 * use a single byte type followed by an offset, but it is impractical
 * to store these separately so we sort by the type field.
 */
static fb_member_t *align_order_members(fb_parser_t *P, fb_member_t *members)
{
    uint16_t i, j, k;
    fb_member_t *heads[9] = {0};
    fb_member_t **tails[9] = {0};
    fb_member_t *next = members;

    while (next) {
        k = next->align;
        switch (next->type.type) {
        case vt_compound_type_ref:
            switch (next->type.ct->symbol.kind) {
            case fb_is_struct:
            case fb_is_enum:
                k = next->type.ct->align;
                break;
            case fb_is_union:
                /*
                 * Unions align to their offsets because the type can
                 * always be added last in a second pass since it is 1
                 * byte.
                 */
                k = (uint16_t)P->opts.offset_size;
                break;
            default:
                k = (uint16_t)P->opts.offset_size;
                break;
            }
            break;
        case vt_vector_compound_type_ref:
        case vt_string_type:
        case vt_vector_type:
        case vt_vector_string_type:
            k = (uint16_t)P->opts.offset_size;
            break;
        case vt_invalid:
            /* Just to have some sane behavior. */
            return original_order_members(P, members);
        default:
            k = next->align;
            break;
        }
        assert(k > 0);
        i = 0;
        while (k >>= 1) {
            ++i;
        }
        /* Normally the largest alignment is 256, but otherwise we group them together. */
        if (i > 7) {
            i = 7;
        }
        if (!heads[i]) {
            heads[i] = next;
        } else {
            *tails[i] = next;
        }
        tails[i] = &next->order;
        next = (fb_member_t *)(((fb_symbol_t *)next)->link);
    }
    i = j = 8;
    tails[8] = &heads[8];
    while (j) {
        while (i && !heads[--i]) {
        }
        *tails[j] = heads[i];
        j = i;
    }
    return heads[8];
}

/* Temporary markers only used during assignment of field identifiers. */
enum { unused_field = 0, normal_field, type_field };

static int process_table(fb_parser_t *P, fb_compound_type_t *ct)
{
    char msg_buf [100];
    fb_symbol_t *sym, *old, *type_sym;
    fb_member_t *member;
    fb_metadata_t *knowns[KNOWN_ATTR_COUNT], *m;
    int ret = 0;
    uint64_t count = 0;
    int need_id = 0, id_failed = 0;
    uint64_t max_id = 0;
    int key_ok, key_count = 0, primary_count = 0;
    int is_union_vector, is_vector;
    uint64_t i, j;
    int max_id_errors = 10;
    int allow_flags = 0;

    /*
     * This just tracks the presence of a `normal_field` or a hidden
     * `type_field`.  The size is litmited to 16-bit unsigned offsets.
     * It is only of relevance for ithe optional `id` table field
     * attribute.
     */
    uint8_t *field_marker = 0;
    fb_symbol_t **field_index = 0;

    assert(ct->symbol.kind == fb_is_table);
    assert(!ct->type.type);

    ct->metadata_flags = process_metadata(P, ct->metadata, fb_f_original_order, knowns);
    /*
     * `original_order` now lives as a flag, we need not consider it
     * further until code generation.
     */
    for (sym = ct->members; sym; sym = sym->link) {
        key_ok = 0;
        type_sym = 0;
        is_vector = 0;
        is_union_vector = 0;
        if ((old = define_fb_symbol(&ct->index, sym))) {
            error_sym_2(P, sym, "table member already defined here", old);
            continue;
        }
        if (sym->kind != fb_is_member) {
            error_sym(P, sym, "internal error: member type expected");
            return -1;
        }
        check_reserved_table_field(P, sym);
        member = (fb_member_t *)sym;
        if (member->type.type == vt_invalid) {
            continue;
        }
        if (member->type.type == vt_scalar_type || member->type.type == vt_vector_type) {
            member->type.st = map_scalar_token_type(member->type.t);
        }
        allow_flags =
                fb_f_id | fb_f_nested_flatbuffer | fb_f_deprecated | fb_f_key |
                fb_f_required | fb_f_hash | fb_f_base64 | fb_f_base64url | fb_f_sorted;

        if (P->opts.allow_primary_key) {
            allow_flags |= fb_f_primary_key;
        }
        member->metadata_flags = process_metadata(P, member->metadata, (uint16_t)allow_flags, knowns);
        if ((m = knowns[fb_attr_nested_flatbuffer])) {
            define_nested_table(P, ct->scope, member, m);
        }
        /* Note: we allow base64 and base64url with nested attribute. */
        if ((member->metadata_flags & fb_f_base64) &&
                (member->type.type != vt_vector_type || member->type.st != fb_ubyte)) {
            error_sym(P, sym, "'base64' attribute is only allowed on vectors of type ubyte");
        }
        if ((member->metadata_flags & fb_f_base64url) &&
                (member->type.type != vt_vector_type || member->type.st != fb_ubyte)) {
            error_sym(P, sym, "'base64url' attribute is only allowed on vectors of type ubyte");
        }
        if ((member->metadata_flags & (fb_f_base64 | fb_f_base64url)) ==
                (fb_f_base64 | fb_f_base64url)) {
            error_sym(P, sym, "'base64' and 'base64url' attributes cannot both be set");
        }
        m = knowns[fb_attr_id];
        if (m && count == 0) {
            need_id = 1;
            field_marker = P->tmp_field_marker;
            memset(field_marker, 0, (size_t)P->opts.vt_max_count);
        }
        if (!id_failed) {
            if (count >= P->opts.vt_max_count) {
                error_sym(P, sym, "too many fields for vtable size");
                id_failed = 1;
            } else if (!need_id) {
                member->id = (unsigned short)count;
            }
            ++count;
        }
        switch (member->type.type) {
        case vt_scalar_type:
            if (member->value.type == vt_null) {
                member->value.type = vt_uint;
                member->value.u = 0;
                member->flags |= fb_fm_optional;
            }
            if (member->metadata_flags & fb_f_required) {
                if (member->flags & fb_fm_optional) {
                    error_sym(P, sym, "'required' attribute is incompatible with optional table field (= null)");
                } else {
                    error_sym(P, sym, "'required' attribute is redundant on scalar table field");
                }
            }
            key_ok = 1;
            if (member->value.type == vt_name_ref) {
                if (lookup_enum_name(P, ct->scope, 0, member->value.ref, &member->value)) {
                    error_ref_sym(P, member->value.ref, "unknown name used as initializer for scalar field", sym);
                    member->type.type = vt_invalid;
                    continue;
                }
            }
            if (!member->value.type) {
                /*
                 * Simplifying by ensuring we always have a default
                 * value where an initializer is possible (also goes for enum).
                 */
                member->value.type = vt_uint;
                member->value.u = 0;
            }
            if (fb_coerce_scalar_type(P, sym, member->type.st, &member->value)) {
                member->type.type = vt_invalid;
                continue;
            }
            member->size = sizeof_scalar_type(member->type.st);
            member->align = (uint16_t)member->size;
            break;
        case vt_vector_type:
            is_vector = 1;
            member->size = sizeof_scalar_type(member->type.st);
            member->align =(uint16_t) member->size;
            if (member->value.type) {
                error_sym(P, sym, "scalar vectors cannot have an initializer");
                member->type.type = vt_invalid;
                continue;
            }
            break;
        case vt_fixed_array_type_ref:
        case vt_fixed_array_string_type:
        case vt_fixed_array_type:
            error_sym(P, sym, "fixed length arrays can only be used with structs");
            member->type.type = vt_invalid;
            return -1;
        case vt_string_type:
            /* `size` or `align` not defined - these are implicit uoffset types. */
            key_ok = P->opts.allow_string_key;
            if (member->value.type) {
                error_sym(P, sym, "strings cannot have an initializer");
                member->type.type = vt_invalid;
                continue;
            }
            break;
        case vt_vector_string_type:
            is_vector = 1;
            /* `size` or `align` not defined - these are implicit uoffset types. */
            if (member->value.type) {
                error_sym(P, sym, "string vectors cannot have an initializer");
                member->type.type = vt_invalid;
                continue;
            }
            break;
        case vt_type_ref:
            type_sym = lookup_type_reference(P, ct->scope, member->type.ref);
            if (!type_sym) {
                error_ref_sym(P, member->type.ref, "unknown type reference used with table field", sym);
                member->type.type = vt_invalid;
                /* We cannot count id's without knowing if it is a union. */
                id_failed = 1;
                continue;
            }
            switch (type_sym->kind) {
            case fb_is_enum:
                /*
                 * Note the enums without a 0 element requires an
                 * initializer in the schema, but that cannot happen
                 * with a null value, so in this case the value is force
                 * to 0. This is only relevant when using the `_get()`
                 * accessor instead of the `_option()`.
                 */
                if (member->value.type == vt_null) {
                    member->value.type = vt_uint;
                    member->value.u = 0;
                    member->flags |= fb_fm_optional;
                }
                if (member->metadata_flags & fb_f_required) {
                    if (member->flags & fb_fm_optional) {
                        error_sym(P, sym, "'required' attribute is incompatible with optional enum table field (= null)");
                    } else {
                        error_sym(P, sym, "'required' attribute is redundant on enum table field");
                    }
                }
                key_ok = P->opts.allow_enum_key;
                break;
            case fb_is_table:
            case fb_is_struct:
            case fb_is_union:
                break;
            case fb_is_rpc_service:
                error_sym_2(P, sym, "rpc service is not a valid table field type", type_sym);
                member->type.type = vt_invalid;
                continue;
            default:
                error_sym_2(P, sym, "internal error: unexpected field type", type_sym);
                member->type.type = vt_invalid;
                continue;
            }
            member->type.type = vt_compound_type_ref;
            member->type.ct = (fb_compound_type_t*)type_sym;
            /*
             * Note: this information transfer won't always work because
             * structs do not know their full size at this point so
             * codegen must use the member->type.ct values.
             */
            member->size = member->type.ct->size;
            member->align = member->type.ct->align;

            if (type_sym->kind == fb_is_union && !id_failed) {
                /* Hidden union type field. */
                if (!need_id) {
                    member->id = (unsigned short)count;
                }
                ++count;
            }
            if (member->value.type) {
                if (type_sym->kind != fb_is_enum) {
                    error_sym(P, sym, "non-scalar field cannot have an initializer");
                    member->type.type = vt_invalid;
                    continue;
                }
                if (member->value.type == vt_name_ref) {
                    if (lookup_enum_name(P, ct->scope, member->type.ct, member->value.ref, &member->value)) {
                        error_ref_sym(P, member->value.ref, "unknown name used as initializer for enum field", sym);
                        member->type.type = vt_invalid;
                        continue;
                    }
                } else {
                    if (fb_coerce_scalar_type(P, sym, ((fb_compound_type_t *)type_sym)->type.st, &member->value)) {
                        member->type.type = vt_invalid;
                        continue;
                    }
                    /* Bitflags can have complex combinations of values, and do not nativele have a 0 value. */
                    if (P->opts.strict_enum_init && !(member->type.ct->metadata_flags & fb_f_bit_flags)
                            && !(member->flags & fb_fm_optional)) {
                        if (!is_in_value_set(&member->type.ct->value_set, &member->value)) {
                            error_sym(P, sym, "initializer does not match a defined enum value");
                            member->type.type = vt_invalid;
                            continue;
                        }
                    }
                }
            } else {
                /* Enum is the only type that cannot always default to 0. */
                if (type_sym->kind == fb_is_enum) {
                    member->value.type = vt_uint;
                    member->value.u = 0;
                    if (fb_coerce_scalar_type(P, type_sym, ((fb_compound_type_t *)type_sym)->type.st, &member->value)) {
                        member->type.type = vt_invalid;
                        continue;
                    }
                    if (P->opts.strict_enum_init) {
                        /* TODO: consider if this error is necessary for bit_flags - flatc 2.0.0 errors on this. */
                        if (!is_in_value_set(&member->type.ct->value_set, &member->value)) {
                            error_sym_2(P, sym,
                                    "enum type requires an explicit initializer because it has no 0 value", type_sym);
                            member->type.type = vt_invalid;
                            continue;
                        }
                    }
                }
            }
            break;
        case vt_vector_type_ref:
            type_sym = lookup_type_reference(P, ct->scope, member->type.ref);
            if (!type_sym) {
                error_ref_sym(P, member->type.ref, "unknown vector type reference used with table field", sym);
                member->type.type = vt_invalid;
                continue;
            }
            switch (type_sym->kind) {
            case fb_is_enum:
            case fb_is_table:
            case fb_is_struct:
            case fb_is_union:
                break;
            default:
                /* Vectors of strings are handled separately but this is irrelevant to the user. */
                error_sym_tok(P, sym, "vectors can only hold scalars, structs, enums, strings, tables, and unions", member->type.t);
                member->type.type = vt_invalid;
                continue;
            }
            is_vector = 1;
            is_union_vector = type_sym->kind == fb_is_union;
            if (member->value.type) {
                error_sym(P, sym, "non-scalar field cannot have an initializer");
                member->type.type = vt_invalid;
                continue;
            }
            /* Size of the vector element, not of the vector itself. */
            member->type.type = vt_vector_compound_type_ref;
            member->type.ct = (fb_compound_type_t*)type_sym;
            member->size = member->type.ct->size;
            member->align = member->type.ct->align;
            if (type_sym->kind == fb_is_union && !id_failed) {
                /* Hidden union type field. */
                if (!need_id) {
                    member->id = (unsigned short)count;
                }
                ++count;
            }
            break;
        default:
            error_sym(P, sym, "unexpected table field type encountered");
            member->type.type = vt_invalid;
            continue;
        }
        if (!id_failed) {
            if (m && !need_id && !id_failed) {
                error_tok(P, m->ident, "unexpected id attribute, must be used on all fields, or none");
                id_failed = 1;
            } else if (!m && need_id && !id_failed) {
                error_sym(P, sym, "id attribute missing, must be used on all fields, or none");
                id_failed = 1;
            } else if (m) {
                if (m->value.type == vt_uint) {
                    if (m->value.u >= P->opts.vt_max_count) {
                        error_sym(P, sym, "id too large to fit in vtable");
                        id_failed = 1;
                    } else {
                        member->id = (unsigned short)m->value.u;
                        if (member->id > max_id) {
                            max_id = member->id;
                        }
                    }
                } else if (m->value.type == vt_int) {
                    error_tok(P, m->ident, "id attribute cannot be negative");
                    id_failed = 1;
                } else {
                    error_tok(P, m->ident, "unexpecte id attribute type");
                    id_failed = 1;
                }
            }
        }
        if (need_id && !id_failed) {
            if (field_marker[member->id] == type_field) {
                error_tok(P, m->ident, "id attribute value conflicts with a hidden type field");
                id_failed = normal_field;
            } else if (field_marker[member->id]) {
                error_tok(P, m->ident, "id attribute value conflicts with another field");
            } else {
                field_marker[member->id] = normal_field;
            }
            if (!id_failed && type_sym && type_sym->kind == fb_is_union) {
                if (member->id <= 0) {
                    error_tok(P, m->ident, is_union_vector ?
                            "id attribute value should be larger to accommodate hidden union vector type field" :
                            "id attribute value should be larger to accommodate hidden union type field");
                    id_failed = 1;
                } else if (field_marker[member->id - 1] == type_field) {
                    error_tok(P, m->ident, is_union_vector ?
                            "hidden union vector type field below attribute id value conflicts with another hidden type field" :
                            "hidden union type field below attribute id value conflicts with another hidden type field");
                    id_failed = 1;
                } else if (field_marker[member->id - 1]) {
                    error_tok(P, m->ident, is_union_vector ?
                            "hidden union vector type field below attribute id value conflicts with another field" :
                            "hidden union type field below attribute id value conflicts with another field");
                    id_failed = 1;
                } else {
                    field_marker[member->id - 1] = type_field;
                }
            }
        }
        if (member->metadata_flags & fb_f_deprecated) {
            if (member->metadata_flags & fb_f_key) {
                error_sym(P, sym, "key attribute not allowed for deprecated field");
                return -1;
            } else if (member->metadata_flags & fb_f_primary_key) {
                error_sym(P, sym, "primary_key attribute not allowed for deprecated field");
                return -1;
            }
        }
        if (member->metadata_flags & fb_f_key) {
            ++key_count;
            if (!key_ok) {
                error_sym(P, sym, "key attribute not allowed for this kind of field");
                member->type.type = vt_invalid;
            } else if (member->metadata_flags & fb_f_primary_key) {
                error_sym(P, sym, "primary_key attribute conflicts with key attribute");
                member->type.type = vt_invalid;
            } else if (!ct->primary_key ||
                    (primary_count == 0 && ct->primary_key->id > member->id)) {
                /*
                 * Set key field with lowest id as default primary key
                 * unless a key field also has a primary attribute.
                 */
                ct->primary_key = member;
            }
        } else if (member->metadata_flags & fb_f_primary_key) {
            if (member->metadata_flags & fb_f_primary_key) {
                if (primary_count++) {
                    error_sym(P, sym, "at most one field can have a primary_key attribute");
                    member->type.type = vt_invalid;
                    continue;
                } else {
                    ct->primary_key = member;
                }
            }
            key_count++;
            /* Allow backends to treat primary key as an ordinary key. */
            member->metadata_flags |= fb_f_key;
        }
        if (member->metadata_flags & fb_f_sorted) {
            if (is_union_vector) {
                error_sym(P, sym, "sorted attribute not allowed for union vectors");
                member->type.type = vt_invalid;
                return -1;
            } else if (!is_vector) {
                error_sym(P, sym, "sorted attribute only allowed for vectors");
                member->type.type = vt_invalid;
                return -1;
            }
            /*
             * A subsequent call to validate_table_attr will verify that a
             * sorted vector of tables or structs have a defined field
             * key. This cannot be done before all types have been
             * processed.
             */
        }
    }
    if (!id_failed) {
        ct->count = count;
    }
    if (!id_failed && need_id) {
        if (count && max_id >= count) {
            for (i = 0; i < max_id; ++i) {
                if (field_marker[i] == 0) {
                    if (!max_id_errors--) {
                        error_sym(P, &ct->symbol, "... more id's missing");
                        break;
                    }
                    sprintf(msg_buf,  "id range not consequtive from 0, missing id: %"PRIu64"", i);
                    error_sym(P, &ct->symbol, msg_buf);
                }
            }
            id_failed = 1;
        }
    }
    /* Order in which data is ordered in binary buffer. */
    if (ct->metadata_flags & fb_f_original_order) {
        ct->ordered_members = original_order_members(P, (fb_member_t *)ct->members);
    } else {
        /* Size efficient ordering. */
        ct->ordered_members = align_order_members(P, (fb_member_t *)ct->members);
    }
    if (!id_failed && need_id && count > 0) {
        field_index = P->tmp_field_index;
        memset(field_index, 0, sizeof(field_index[0]) * (size_t)P->opts.vt_max_count);
        /*
         * Reorder by id so table constructor arguments in code
         * generators always use same ordering across versions.
         */
        for (sym = ct->members; sym; sym = sym->link) {
            member = (fb_member_t *)sym;
            field_index[member->id] = sym;
        }
        j = 0;
        if (field_index[0] == 0) {
            j = 1; /* Adjust for union type. */
        }
        ct->members = field_index[j];
        for (i = j + 1; i <= max_id; ++i) {
            if (field_index[i] == 0) ++i; /* Adjust for union type. */
            field_index[j]->link = field_index[i];
            j = i;
        }
        field_index[max_id]->link = 0;
    }
    if (key_count) {
        ct->symbol.flags |= fb_indexed;
    }
    /* Set primary key flag for backends even if chosen by default. */
    if (ct->primary_key) {
        ct->primary_key->metadata_flags |= fb_f_primary_key;
    }
    if (key_count > 1 && !P->opts.allow_multiple_key_fields) {
        error_sym(P, &ct->symbol, "table has multiple key fields, but at most one is permitted");
        ret = -1;
    }
    if (id_failed) {
        ret = -1;
    }
    return ret;
}

/*
 * Post processing of process_table because some information is only
 * available when all types have been processed.
 */
static int validate_table_attr(fb_parser_t *P, fb_compound_type_t *ct)
{
    fb_symbol_t *sym;
    fb_member_t *member;

    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        if (member->metadata_flags & fb_f_deprecated) {
            continue;
        }

        if (member->type.type == vt_vector_compound_type_ref &&
                member->metadata_flags & fb_f_sorted) {
            switch (member->type.ct->symbol.kind) {
            case fb_is_table:
                if (!member->type.ct->primary_key) {
                    error_sym(P, sym, "sorted table vector only valid when table has a key field");
                    return -1;
                }
                break;
            case fb_is_struct:
                if (!member->type.ct->primary_key) {
                    error_sym(P, sym, "sorted struct vector only valid when struct has a key field");
                    return -1;
                }
                break;
            /* Other cases already handled in process_table. */
            default:
                continue;
            }
        }
    }
    return 0;
}

/*
 * The parser already makes sure we have exactly one request type,
 * one response type, and no initializer.
 *
 * We are a bit heavy on flagging attributes because their behavior
 * isn't really specified at this point.
 */
static int process_rpc_service(fb_parser_t *P, fb_compound_type_t *ct)
{
    fb_symbol_t *sym, *old, *type_sym;
    fb_member_t *member;
#if FLATCC_ALLOW_RPC_SERVICE_ATTRIBUTES || FLATCC_ALLOW_RPC_METHOD_ATTRIBUTES
    fb_metadata_t *knowns[KNOWN_ATTR_COUNT];
#endif

    assert(ct->symbol.kind == fb_is_rpc_service);
    assert(!ct->type.type);

    /*
     * Deprecated is defined for fields - but it is unclear if this also
     * covers rpc services. Anyway, we accept it since it may be useful,
     * and does not harm.
     */
#if FLATCC_ALLOW_RPC_SERVICE_ATTRIBUTES
    /* But we have no known attributes to support. */
    ct->metadata_flags = process_metadata(P, ct->metadata, 0, knowns);
#else
    if (ct->metadata) {
        error_sym(P, &ct->symbol, "rpc services cannot have attributes");
        /* Non-fatal. */
    }
#endif

    /*
     * `original_order` now lives as a flag, we need not consider it
     * further until code generation.
     */
    for (sym = ct->members; sym; sym = sym->link) {
        type_sym = 0;
        if ((old = define_fb_symbol(&ct->index, sym))) {
            error_sym_2(P, sym, "rpc method already defined here", old);
            continue;
        }
        if (sym->kind != fb_is_member) {
            error_sym(P, sym, "internal error: member type expected");
            return -1;
        }
        member = (fb_member_t *)sym;
        if (member->value.type) {
            error_sym(P, sym, "internal error: initializer should have been rejected by parser");
        }
        if (member->type.type == vt_invalid) {
            continue;
        }
        if (member->type.type != vt_type_ref) {
            error_sym(P, sym, "internal error: request type expected to be a type reference");
        }
        type_sym = lookup_type_reference(P, ct->scope, member->req_type.ref);
        if (!type_sym) {
            error_ref_sym(P, member->req_type.ref, "unknown type reference used with rpc request type", sym);
            member->type.type = vt_invalid;
            continue;
        } else {
            if (type_sym->kind != fb_is_table) {
                error_sym_2(P, sym, "rpc request type must reference a table, defined here", type_sym);
                member->type.type = vt_invalid;
                continue;
            }
            member->req_type.type = vt_compound_type_ref;
            member->req_type.ct = (fb_compound_type_t*)type_sym;
        }
        type_sym = lookup_type_reference(P, ct->scope, member->type.ref);
        if (!type_sym) {
            error_ref_sym(P, member->type.ref, "unknown type reference used with rpc response type", sym);
            member->type.type = vt_invalid;
            continue;
        } else {
            if (type_sym->kind != fb_is_table) {
                error_sym_2(P, sym, "rpc response type must reference a table, defined here", type_sym);
                member->type.type = vt_invalid;
                continue;
            }
            member->type.type = vt_compound_type_ref;
            member->type.ct = (fb_compound_type_t*)type_sym;
            /* Symbols have no size. */
            member->size = 0;
        }
#if FLATCC_ALLOW_RPC_METHOD_ATTRIBUTES
#if FLATCC_ALLOW_DEPRECATED_RPC_METHOD
        member->metadata_flags = process_metadata(P, member->metadata, fb_f_deprecated, knowns);
#else
        member->metadata_flags = process_metadata(P, member->metadata, 0, knowns);
#endif
#else
        if (member->metadata) {
            error_sym(P, sym, "rpc methods cannot have attributes");
            /* Non-fatal. */
            continue;
        }
#endif
    }
    return 0;
}

static int process_enum(fb_parser_t *P, fb_compound_type_t *ct)
{
    fb_symbol_t *sym, *old, *type_sym;
    fb_member_t *member;
    fb_metadata_t *knowns[KNOWN_ATTR_COUNT];
    fb_value_t index = { { { 0 } }, 0, 0 };
    fb_value_t old_index;
    int first = 1;
    int bit_flags = 0;
    int is_union = ct->symbol.kind == fb_is_union;

    if (!is_union) {
        assert(ct->symbol.kind == fb_is_enum);
        if (!ct->type.type) {
            ct->type.type = vt_invalid;
            error_sym(P, &ct->symbol, "enum must have a type");
            return -1;
        }
        if (ct->type.type == vt_missing) {
            /*
             * Enums normally require a type, but the parser may have an
             * option to allow missing type, and then we provide a
             * sensible default.
             */
            ct->type.st = fb_int;
            ct->type.type = vt_scalar_type;
        } else if (ct->type.type == vt_scalar_type) {
            ct->type.st = map_scalar_token_type(ct->type.t);
        } else {
            /* Spec does not mention boolean type in enum, but we allow it. */
            error_sym(P, &ct->symbol, "enum type must be a scalar integral type or bool");
            return -1;
        }
        ct->size = sizeof_scalar_type(ct->type.st);
        ct->align = (uint16_t)ct->size;
    } else {
        if (ct->type.type) {
            error_sym(P, &ct->symbol, "unions cannot have a type, they are always enumerated as ubyte");
            return -1;
        }
        /*
         * We preprocess unions as enums to get the value assignments.
         * The type field is not documented, but generated output from
         * flatc suggests ubyte. We use a an injected token to represent
         * the ubyte type so we do not have to hardcode elsewhere.
         */
        ct->type.type = vt_scalar_type;
        ct->type.st = fb_ubyte;
        /*
         * The union field use the type field and not the offset field
         * to define its size because type.type is scalar.
         */
        ct->size = sizeof_scalar_type(fb_ubyte);
        ct->align = (uint16_t)ct->size;
    }

    ct->metadata_flags = process_metadata(P, ct->metadata, fb_f_bit_flags, knowns);
    if (ct->metadata_flags & fb_f_bit_flags) {
        bit_flags = 1;
        index.type = vt_uint;
        index.u = 0;
    }

    if (ct->type.st == fb_bool) {
        index.b = 0;
        index.type = vt_bool;
    } else {
        index.i = 0;
        index.type = vt_int;
        if (fb_coerce_scalar_type(P, (fb_symbol_t *)ct, ct->type.st, &index)) {
            error(P, "internal error: unexpected conversion failure on enum 0 index");
            return -1;
        }
    }
    old_index = index;

    for (sym = ct->members; sym; sym = sym->link, first = 0) {
        member = (fb_member_t *)sym;
        if (!is_union) {
            check_reserved_enum_member(P, sym);
        }
        if ((old = define_fb_symbol(&ct->index, sym))) {
            if (old->ident == &P->t_none) {
                /*
                 * Parser injects `NONE` as the first union member and
                 * it therefore gets index 0. Additional use of NONE
                 * will fail.
                 */
                error_sym(P, sym, "'NONE' is a predefined value");
                member->type.type = vt_invalid;
                continue;
            }
            error_sym_2(P, sym, "value already defined here", old);
            member->type.type = vt_invalid;
            continue;
        }
        if (sym->kind != fb_is_member) {
            error_sym(P, sym, "internal error: enum value type expected");
            return -1;
        }
        /* Enum / union values cannot have metadata. */
        assert(member->metadata == 0);
        if (is_union) {
            if (member->symbol.ident == &P->t_none) {
                /* Handle implicit NONE specially. */
                member->type.type = vt_missing;
            } else if (member->type.type == vt_string_type) {
                member->size = 0;
            } else if (member->type.type != vt_type_ref) {
                if (member->type.type != vt_invalid) {
                    error_sym(P, sym, "union member type must be string or a reference to a table or a struct");
                    member->type.type = vt_invalid;
                }
                continue;
            } else {
                type_sym = lookup_type_reference(P, ct->scope, member->type.ref);
                if (!type_sym) {
                    error_ref_sym(P, member->type.ref, "unknown type reference used with union member", sym);
                    member->type.type = vt_invalid;
                    continue;
                } else {
                    if (type_sym->kind != fb_is_table && type_sym->kind != fb_is_struct) {
                        error_sym_2(P, sym, "union member type reference must be a table or a struct, defined here", type_sym);
                        member->type.type = vt_invalid;
                        continue;
                    }
                    member->type.type = vt_compound_type_ref;
                    member->type.ct = (fb_compound_type_t*)type_sym;
                    /* Symbols have no size. */
                    member->size = 0;
                }
            }
        }
        if (!member->value.type && !first) {
            if (index.type == vt_uint) {
                if (ct->type.st == fb_long && index.u == UINT64_MAX) {
                    /* Not captured by range check. */
                    error_sym(P, sym, "64-bit unsigned int overflow");
                }
                index.u = index.u + 1;
            } else if (index.type == vt_int && !first) {
                if (ct->type.st == fb_long && index.i == INT64_MAX) {
                    /* Not captured by range check. */
                    error_sym(P, sym, "64-bit signed int overflow");
                }
                index.i = index.i + 1;
            } else if (index.type == vt_bool && !first) {
                if (index.b == 1) {
                    error_sym(P, sym, "boolean overflow: cannot enumerate past true");
                }
                index.b = 1;
            }
        }
        if (bit_flags) {
            if (member->value.type) {
                if (member->value.type != vt_uint) {
                    error_sym(P, sym, "enum value must be unsigned int when used with 'bit_flags'");
                    return -1;
                } else {
                    index = member->value;
                }
            }
            if (index.u >= sizeof_scalar_type(ct->type.st) * 8) {
                error_sym(P, sym, "enum value out of range when used with 'bit_flags'");
                return -1;
            }
            member->value.u = UINT64_C(1) << index.u;
            member->value.type = vt_uint;
            if (fb_coerce_scalar_type(P, sym, ct->type.st, &member->value)) {
                 /* E.g. enumval = 15 causes signed overflow with short. */
                error_sym(P, sym, "enum value out of range when used with 'bit_flags'");
                return -1;
            }
        } else {
            if (member->value.type) {
                index = member->value;
            }
            /*
             * Captures errors in user assigned values. Also captures
             * overflow on auto-increment on all types except maximum
             * representation size, i.e. long or ulong which we handled
             * above.
             */
            if (fb_coerce_scalar_type(P, sym, ct->type.st, &index)) {
                return -1;
            }
            member->value = index;
        }
        if (!first && P->opts.ascending_enum) {
            /* Without ascending enum we also allow duplicate values (but not names). */
            if ((index.type == vt_uint && index.u <= old_index.u) ||
                    (index.type == vt_int && index.i <= old_index.i)) {
                if (is_union && old_index.u == 0) {
                    /*
                     * The user explicitly assigned zero, or less, to the first
                     * element (here second element after parser injecting
                     * the NONE element).
                     */
                    error_sym(P, sym, "union values must be positive, 0 is reserved for implicit 'NONE'");
                    member->value.type = vt_invalid;
                    return -1;
                }
                error_sym(P, sym, "enum values must be in ascending order");
                member->value.type = vt_invalid;
                return -1;
            }
            if (index.type == vt_bool && index.b <= old_index.b) {
                error_sym(P, sym, "enum of type bool can only enumerate from false (0) to true (1)");
                member->value.type = vt_invalid;
                return -1;
            }
        }
        old_index = index;
        if (add_to_value_set(&ct->value_set, &member->value)) {
            if (is_union) {
                error_sym(P, sym, "union members require unique positive values (0 being reserved for 'NONE'");
                member->value.type = vt_invalid;
                return -1;
            } else {
                /*
                 * With ascending enums this won't happen, but
                 * otherwise flag secondary values so we can remove them
                 * from inverse name mappings in code gen.
                 */
                member->symbol.flags |= fb_duplicate;
            }
        }
        if (member->metadata) {
            error_sym(P, sym, "enum values cannot have attributes");
            /* Non-fatal. */
            continue;
        }
    }
    return 0;
}

static inline int process_union(fb_parser_t *P, fb_compound_type_t *ct)
{
    return process_enum(P, ct);
}

int fb_build_schema(fb_parser_t *P)
{
    fb_schema_t *S = &P->schema;
    fb_symbol_t *sym, *old_sym;
    fb_name_t *old_name;
    fb_compound_type_t *ct;
    fb_attribute_t *a;

    /* Make sure self is visible at this point in time. */
    assert(ptr_set_exists(&P->schema.visible_schema, &P->schema));
    for (sym = S->symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_table:
        case fb_is_enum:
        case fb_is_union:
        case fb_is_struct:
        case fb_is_rpc_service:
            ct = (fb_compound_type_t *)sym;
            set_type_hash(ct);
            ct->schema = &P->schema;
            if (ct->scope && (old_sym = define_fb_symbol(&ct->scope->symbol_index, sym))) {
                error_sym_2(P, sym, "symbol already defined here", old_sym);
            }
        }
    }

    /*
     * Known attributes will be pre-defined if not provided by the
     * user. After that point, all attribute references must be
     * defined.
     */
    for (a = (fb_attribute_t *)S->attributes; a; a = (fb_attribute_t *)a->name.link) {
        if ((old_name = define_fb_name(&S->root_schema->attribute_index, &a->name))) {
            /*
             * Allow attributes to be defined multiple times, including
             * known attributes.
             */
#if 0
            error_name(P, &a->name, "attribute already defined");
#endif
        }
    }
    install_known_attributes(P);
    install_reserved_keywords(P);

    if (!P->opts.hide_later_enum) {
        for (sym = S->symbols; sym; sym = sym->link) {
            switch (sym->kind) {
            case fb_is_enum:
                ct = (fb_compound_type_t *)sym;
                if (process_enum(P, ct)) {
                    ct->type.type = vt_invalid;
                    continue;
                }
                break;
        default:
            continue;
            }
        }
    }

    /*
     * Resolve type references both earlier and later than point of
     * reference. This also supports recursion for tables and unions.
     */
    for (sym = S->symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_struct:
            ct = (fb_compound_type_t *)sym;
            if (process_struct(P, ct)) {
                ct->type.type = vt_invalid;
                continue;
            }
            break;
        case fb_is_table:
            /* Handle after structs and enums. */
            continue;
        case fb_is_rpc_service:
            /*
             * Also handle rpc_service later like tables, just in case
             * we allow non-table types in request/response type.
             */
            continue;
        case fb_is_enum:
            if (P->opts.hide_later_enum) {
                ct = (fb_compound_type_t *)sym;
                if (process_enum(P, ct)) {
                    ct->type.type = vt_invalid;
                    continue;
                }
            }
            break;
        case fb_is_union:
            ct = (fb_compound_type_t *)sym;
            if (process_union(P, ct)) {
                ct->type.type = vt_invalid;
                continue;
            }
            break;
        default:
            error_sym(P, sym, "internal error: unexpected symbol at schema level");
            return -1;
        }
    }
    for (sym = P->schema.symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_struct:
            /*
             * Structs need two stages, first process symbols, then
             * analyze for size, alignment, and circular references.
             */
            ct = (fb_compound_type_t *)sym;
            if (ct->type.type != vt_invalid && analyze_struct(P, ct)) {
                ct->type.type = vt_invalid;
                continue;
            }
            break;
        default:
            continue;
        }
    }
    for (sym = P->schema.symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_table:
            ct = (fb_compound_type_t *)sym;
            /* Only now is the full struct size available. */
            if (ct->type.type != vt_invalid && process_table(P, ct)) {
                ct->type.type = vt_invalid;
                continue;
            }
            break;
        case fb_is_rpc_service:
            ct = (fb_compound_type_t *)sym;
            /* Only now is the full struct size available. */
            if (ct->type.type != vt_invalid && process_rpc_service(P, ct)) {
                ct->type.type = vt_invalid;
                continue;
            }
        }
    }
    for (sym = P->schema.symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_table:
            ct = (fb_compound_type_t *)sym;
            /*
             * Some table attributes depend on attributes on members and
             * therefore can only be validated after procesing.
             */
            if (ct->type.type != vt_invalid && validate_table_attr(P, ct)) {
                ct->type.type = vt_invalid;
                continue;
            }
        }
    }
    revert_order(&P->schema.ordered_structs);
    if (!S->root_type.name) {
        if (P->opts.require_root_type) {
            error(P, "root type not declared");
        }
    } else {
        sym = S->root_type.type = lookup_type_reference(P,
                S->root_type.scope, S->root_type.name);
        if (!sym) {
            error_ref(P, S->root_type.name, "root type not found");
        } else if (P->opts.allow_struct_root) {
            if (sym->kind != fb_is_struct && sym->kind != fb_is_table) {
                error_ref(P, S->root_type.name, "root type must be a struct or a table");
            } else {
                S->root_type.type = sym;
            }
        } else {
            if (sym->kind != fb_is_table) {
                error_ref(P, S->root_type.name, "root type must be a table");
            } else {
                S->root_type.type = sym;
            }
        }
        S->root_type.name = 0;
    }
    P->has_schema = !P->failed;
    return P->failed;
}
