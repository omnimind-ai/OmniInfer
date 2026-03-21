/* Flatbuffers parser attributes and symbols. */

#ifndef SYMBOLS_H
#define SYMBOLS_H

#include <stdint.h>

#include "config.h"
#include "lex/tokens.h"
#include "hash/hash_table.h"
#include "hash/ptr_set.h"

typedef struct fb_token fb_token_t;
typedef struct fb_string fb_string_t;
typedef struct fb_value fb_value_t;
typedef struct fb_symbol fb_symbol_t;

typedef struct fb_metadata fb_metadata_t;

typedef struct fb_name fb_name_t;
typedef fb_symbol_t fb_namespace_t;
typedef fb_symbol_t fb_ref_t;
/* Doc is not strictly a symbol, just a chained token list, but close enough. */
typedef fb_symbol_t fb_doc_t;
typedef fb_name_t fb_include_t;
typedef struct fb_attribute fb_attribute_t;
typedef struct fb_reserved_kw fb_reserved_kw_t;

typedef struct fb_member fb_member_t;
typedef struct fb_compound_type fb_compound_type_t;

typedef struct fb_scope fb_scope_t;
typedef struct fb_root_schema fb_root_schema_t;
typedef struct fb_root_type fb_root_type_t;
typedef struct fb_schema fb_schema_t;

enum {
    tok_kw_base = LEX_TOK_KW_BASE,
    tok_kw_bool,
    tok_kw_byte,
    tok_kw_char,
    tok_kw_enum,
    tok_kw_float32,
    tok_kw_float64,
    tok_kw_int,
    tok_kw_int8,
    tok_kw_int16,
    tok_kw_int32,
    tok_kw_int64,
    tok_kw_long,
    tok_kw_null,
    tok_kw_true,
    tok_kw_uint,
    tok_kw_false,
    tok_kw_float,
    tok_kw_short,
    tok_kw_table,
    tok_kw_ubyte,
    tok_kw_uint8,
    tok_kw_uint16,
    tok_kw_uint32,
    tok_kw_uint64,
    tok_kw_ulong,
    tok_kw_union,
    tok_kw_double,
    tok_kw_string,
    tok_kw_struct,
    tok_kw_ushort,
    tok_kw_include,
    tok_kw_attribute,
    tok_kw_namespace,
    tok_kw_root_type,
    tok_kw_rpc_service,
    tok_kw_file_extension,
    tok_kw_file_identifier,
    LEX_TOK_KW_END,
    /* Pseudo keywords. */
    tok_kw_doc_comment
};

struct fb_token {
    const char *text;
    long len;
    long linenum;
    long pos;
    long id;
};

enum fb_scalar_type {
    fb_missing_type = 0,
    fb_ulong,
    fb_uint,
    fb_ushort,
    fb_ubyte,
    fb_bool,
    fb_long,
    fb_int,
    fb_short,
    fb_byte,
    fb_double,
    fb_float,
    fb_char,
};

typedef enum fb_scalar_type fb_scalar_type_t;

static inline size_t sizeof_scalar_type(fb_scalar_type_t st)
{
    static const size_t scalar_type_size[] = {
        0, 8, 4, 2, 1, 1, 8, 4, 2, 1, 8, 4, 1
    };

    return scalar_type_size[st];
}

enum fb_value_type {
    vt_missing = 0,
    vt_invalid = 1,
    vt_null,
    vt_string,
    vt_float,
    vt_int,
    vt_uint,
    vt_bool,
    vt_vector_type,
    vt_scalar_type,
    vt_vector_string_type,
    vt_string_type,
    vt_vector_type_ref,
    vt_type_ref,
    vt_name_ref,
    vt_compound_type_ref,
    vt_vector_compound_type_ref,
    vt_fixed_array_type,
    vt_fixed_array_type_ref,
    vt_fixed_array_string_type,
    vt_fixed_array_compound_type_ref
};

struct fb_string {
    char *s;
    /* printf statements relies on this being int. */
    int len;
};

struct fb_value {
    union {
        fb_string_t s;
        double f;
        int64_t i;
        uint64_t u;
        uint8_t b;
        fb_token_t *t;
        fb_compound_type_t *ct;
        fb_scalar_type_t st;
        fb_ref_t *ref;
    };
    unsigned short type;
    uint32_t len;
};

enum fb_kind {
    fb_is_table,
    fb_is_struct,
    fb_is_rpc_service,
    fb_is_enum,
    fb_is_union,
    fb_is_member
};

/*
 * Used for white, gray, black graph coloring while detecting circular
 * references.
 */
enum fb_symbol_flags {
    fb_circular_open = 1,
    fb_circular_closed = 2,
    fb_duplicate = 4,
    fb_indexed = 8,
};

enum fb_member_flags {
    fb_fm_optional = 1
};

/*
 * We keep the link first in all structs so that we can use a
 * generic list reverse function after all symbols have been pushed
 * within a scope.
 */
struct fb_symbol {
    fb_symbol_t *link;
    fb_token_t *ident;
    uint16_t kind;
    uint16_t flags;
};

struct fb_name {
    fb_name_t *link;
    fb_value_t name;
};

#define fb_name_table __flatcc_fb_name_table
#define fb_value_set __flatcc_fb_value_set
#define fb_symbol_table __flatcc_fb_symbol_table
#define fb_scope_table __flatcc_fb_scope_table

DECLARE_HASH_TABLE(fb_name_table, fb_name_t *)
DECLARE_HASH_TABLE(fb_schema_table, fb_schema_t *)
DECLARE_HASH_TABLE(fb_value_set, fb_value_t *)
DECLARE_HASH_TABLE(fb_symbol_table, fb_symbol_t *)
DECLARE_HASH_TABLE(fb_scope_table, fb_scope_t *)

struct fb_member {
    fb_symbol_t symbol;
    /* Struct or table field type, or method response type. */
    fb_value_t type;
    /* Method request type only used for methods. */
    fb_value_t req_type;
    fb_value_t value;
    fb_metadata_t *metadata;
    fb_doc_t *doc;
    uint16_t metadata_flags;
    /*
     * `align`, `offset` are for structs only.  64-bit allows for
     * dynamically configured 64-bit file offsets. Align is restricted to
     * at most 256 and must be a power of 2.
     */
    uint16_t align;
    uint16_t flags;
    uint64_t offset;
    uint64_t size;

    /* `id` is for table fields only. */
    uint64_t id;
    /*
     * Resolved `nested_flatbuffer` attribute type. Always a table if
     * set, and only on struct and table fields.
     */
    fb_compound_type_t *nest;
    /* Used to generate table fields in sorted order. */
    fb_member_t *order;

    /*
     * Use by code generators. Only valid during export and may hold
     * garbage from a prevous export.
     */
    size_t export_index;
};

struct fb_metadata {
    fb_metadata_t *link;
    fb_token_t *ident;
    fb_value_t value;
};

struct fb_compound_type {
    fb_symbol_t symbol;
    /* `scope` may span multiple input files, but has a unique namespace. */
    fb_scope_t *scope;
    /* Identifies the the schema the symbol belongs. */
    fb_schema_t *schema;
    fb_symbol_t *members;
    fb_member_t *ordered_members;
    fb_member_t *primary_key;
    fb_metadata_t *metadata;
    fb_doc_t *doc;
    fb_value_t type;
    fb_symbol_table_t index;
    /* Only for enums. */
    fb_value_set_t value_set;
    /* FNV-1a 32 bit hash of fully qualified name, accidental 0 maps to hash(""). */
    uint32_t type_hash;
    uint16_t metadata_flags;
    /* `count` is for tables only. */
    uint64_t count;
    /* `align`, `size` is for structs only. */
    uint16_t align;
    uint64_t size;
    /* Sort structs with forward references. */
    fb_compound_type_t *order;
    /*
     * Use by code generators. Only valid during export and may hold
     * garbage from a previous export.
     */
    size_t export_index;
};

enum fb_known_attributes {
    fb_attr_unknown = 0,
    fb_attr_id = 1,
    fb_attr_deprecated = 2,
    fb_attr_original_order = 3,
    fb_attr_force_align = 4,
    fb_attr_bit_flags = 5,
    fb_attr_nested_flatbuffer = 6,
    fb_attr_key = 7,
    fb_attr_required = 8,
    fb_attr_hash = 9,
    fb_attr_base64 = 10,
    fb_attr_base64url = 11,
    fb_attr_primary_key = 12,
    fb_attr_sorted = 13,
    KNOWN_ATTR_COUNT
};

enum fb_known_attribute_flags {
    fb_f_unknown = 1 << fb_attr_unknown,
    fb_f_id = 1 << fb_attr_id,
    fb_f_deprecated = 1 << fb_attr_deprecated,
    fb_f_original_order = 1 << fb_attr_original_order,
    fb_f_force_align = 1 << fb_attr_force_align,
    fb_f_bit_flags = 1 << fb_attr_bit_flags,
    fb_f_nested_flatbuffer = 1 << fb_attr_nested_flatbuffer,
    fb_f_key = 1 << fb_attr_key,
    fb_f_required = 1 << fb_attr_required,
    fb_f_hash = 1 << fb_attr_hash,
    fb_f_base64 = 1 << fb_attr_base64,
    fb_f_base64url = 1 << fb_attr_base64url,
    fb_f_primary_key = 1 << fb_attr_primary_key,
    fb_f_sorted = 1 << fb_attr_sorted,
};

enum fb_reserved_kw_kind {
    fb_reserved_kw_kind_table_field = 1,
    fb_reserved_kw_kind_table_field_prefix = 2,
    fb_reserved_kw_kind_table_field_suffix = 3,
    fb_reserved_kw_kind_enum_member = 4,
    fb_reserved_kw_kind_vec_prefix = 5,
};

struct fb_attribute {
    fb_name_t name;
    unsigned int known;
};

struct fb_reserved_kw {
    fb_name_t name;
    int kind;
};

struct fb_scope {
    fb_ref_t *name;
    fb_symbol_table_t symbol_index;
    fb_string_t prefix;
};

struct fb_root_schema {
    fb_scope_table_t scope_index;
    fb_name_table_t attribute_index;
    fb_name_table_t keyword_index;
    fb_schema_table_t include_index;
    int include_count;
    int include_depth;
    size_t total_source_size;
};

struct fb_root_type {
    /* Root decl. before symbol is visible. */
    fb_ref_t *name;
    /* Resolved symbol. */
    fb_symbol_t *type;
    fb_scope_t *scope;
};

/*
 * We store the parsed structure as token references. Tokens are stored
 * in a single array pointing into the source buffer.
 *
 * Strings may contain multiple tokens when holding control characters
 * and line breaks, but for our purposes the first string part is
 * sufficient.
 */

struct fb_schema {
    fb_include_t *includes;
    fb_name_t *attributes;
    fb_value_t file_identifier;
    fb_value_t file_extension;
    fb_symbol_t *symbols;
    /* Topologically sorted structs. */
    fb_compound_type_t *ordered_structs;
    fb_root_type_t root_type;
    fb_root_schema_t *root_schema;
    /* Only used if schema is root. */
    fb_root_schema_t root_schema_instance;

    /* An optional scope prefix for generated code. */
    fb_string_t prefix;

    /* The basenameup in a format that can be index. */
    fb_name_t name;

    /* These are allocated strings that must be freed. */

    /* Name of schema being processed without path or default extension. */
    char *basename;
    /* Uppercase basename for codegen and for case insenstive file inclusion check. */
    char *basenameup;
    /* Basename with extension. */
    char *errorname;

    /*
     * The dependency schemas visible to this schema (includes self).
     * Compound symbols have a link to schema which can be checked
     * against this set to see if the symbol is visible in this
     * conctext.
     */
    ptr_set_t visible_schema;
};

/*
 * Helpers to ensure a symbol is actually visible because a scope
 * (namespace) may be extended when a parent inlcudes another file
 * first.
 */
static inline fb_compound_type_t *get_enum_if_visible(fb_schema_t *schema, fb_symbol_t *sym)
{
    fb_compound_type_t *ct = 0;

    switch (sym->kind) {
    case fb_is_union:
        /* Fall through. */
    case fb_is_enum:
        ct = (fb_compound_type_t *)sym;
        if (!ptr_set_exists(&schema->visible_schema, ct->schema)) {
            ct = 0;
        }
        break;
    default:
        break;
    }
    return ct;
}

static inline fb_compound_type_t *get_compound_if_visible(fb_schema_t *schema, fb_symbol_t *sym)
{
    fb_compound_type_t *ct = 0;

    switch (sym->kind) {
    case fb_is_struct:
    case fb_is_table:
    case fb_is_rpc_service:
    case fb_is_union:
    case fb_is_enum:
        ct = (fb_compound_type_t *)sym;
        if (!ptr_set_exists(&schema->visible_schema, ct->schema)) {
            ct = 0;
        }
        break;
    default:
        break;
    }
    return ct;
}

/* Constants are specific to 32-bit FNV-1a hash. It is important to use unsigned integers. */
static inline uint32_t fb_hash_fnv1a_32_init(void)
{
    return 2166136261UL;
}

static inline uint32_t fb_hash_fnv1a_32_append(uint32_t hash, const char *data, size_t len)
{
    while (len--) {
        hash ^= *(uint8_t *)data++;
        hash = hash * 16777619UL;
    }
    return hash;
}

#endif /* SYMBOLS_H */
