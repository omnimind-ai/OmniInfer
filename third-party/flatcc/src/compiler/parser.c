/*
 * FlatBuffers IDL parser.
 *
 * Originally based on the numeric parser in the Luthor lexer project.
 *
 * We are moving away from TDOP approach because the grammer doesn't
 * really benefit from it. We use the same overall framework.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdarg.h>

#include "semantics.h"
#include "codegen.h"
#include "fileio.h"
#include "pstrutil.h"
#include "flatcc/portable/pparseint.h"

void fb_default_error_out(void *err_ctx, const char *buf, size_t len)
{
    (void)err_ctx;

    fwrite(buf, 1, len, stderr);
}

int fb_print_error(fb_parser_t *P, const char * format, ...)
{
    int n;
    va_list ap;
    char buf[ERROR_BUFSIZ];

    va_start (ap, format);
    n = vsnprintf (buf, ERROR_BUFSIZ, format, ap);
    va_end (ap);
    if (n >= ERROR_BUFSIZ) {
        strcpy(buf + ERROR_BUFSIZ - 5, "...\n");
        n = ERROR_BUFSIZ - 1;
    }
    P->error_out(P->error_ctx, buf, (size_t)n);
    return n;
}

const char *error_find_file_of_token(fb_parser_t *P, fb_token_t *t)
{
    /*
     * Search token in dependent buffers if not in current token
     * buffer. We can do this as a linear search because we limit the
     * number of output errors.
     */
    while (P) {
        if (P->ts <= t && P->te > t) {
            return P->schema.errorname;
        }
        P = P->dependencies;
    }
    return "";
}

void error_report_ext(fb_parser_t *P, fb_token_t *t, const char *msg, fb_token_t *peer, const char *s, size_t len, int is_warning)
{
    const char *file, *peer_file;
    const char *kind = is_warning ? "warning" : "error";

    if (is_warning && P->opts.silence) {
        return;
    }
    if (t && !s) {
        s = t->text;
        len = (size_t)t->len;
    }
    if (!msg) {
        msg = "";
    }
    if (!s) {
        s = "";
        len = 0;
    }
    if (t && !peer) {
        file = error_find_file_of_token(P, t);
        fb_print_error(P, "%s:%ld:%ld: %s: '%.*s': %s\n",
                file, (long)t->linenum, (long)t->pos, kind, len, s, msg);
    } else if (t && peer) {
        file = error_find_file_of_token(P, t);
        peer_file = error_find_file_of_token(P, peer);
        fb_print_error(P, "%s:%ld:%ld: %s: '%.*s': %s: %s:%ld:%ld: '%.*s'\n",
                file, (long)t->linenum, (long)t->pos, kind, len, s, msg,
                peer_file, (long)peer->linenum, (long)peer->pos, (int)peer->len, peer->text);
    } else if (!t && !peer) {
        fb_print_error(P, "error: %s\n", msg);
    } else if (peer) {
        peer_file = error_find_file_of_token(P, peer);
        fb_print_error(P, "%s: %s: %s:%ld:%ld: '%.*s'\n",
                kind, msg,
                peer_file, (long)peer->linenum, (long)peer->pos, (int)peer->len, peer->text);
    } else {
        fb_print_error(P, "internal error: unexpected state\n");
    }
    if (!is_warning) {
        ++P->failed;
    }
}

void error_report(fb_parser_t *P, fb_token_t *t, const char *msg, fb_token_t *peer, const char *s, size_t len)
{
    error_report_ext(P, t, msg, peer, s, len, 0);
}

void warn_report(fb_parser_t *P, fb_token_t *t, const char *msg, fb_token_t *peer, const char *s, size_t len)
{
    error_report_ext(P, t, msg, peer, s, len, 1);
}

void error_ref_sym(fb_parser_t *P, fb_ref_t *ref, const char *msg, fb_symbol_t *s2)
{
    fb_ref_t *p;
    char buf[FLATCC_MAX_IDENT_SHOW + 1];
    size_t k = FLATCC_MAX_IDENT_SHOW;
    size_t n = 0;
    size_t n0 = 0;
    int truncated = 0;

    p = ref;
    while (p && k > 0) {
        if (n0 > 0) {
            buf[n0] = '.';
            --k;
            ++n0;
        }
        n = (size_t)p->ident->len;
        if (k < n) {
            n = k;
            truncated = 1;
        }
        memcpy(buf + n0, p->ident->text, n);
        k -= n;
        n0 += n;
        p = p->link;
    }
    if (p) truncated = 1;
    buf[n0] = '\0';
    if (n0 > 0) {
        --n0;
    }
    if (truncated) {
        memcpy(buf + FLATCC_MAX_IDENT_SHOW + 1 - 4, "...\0", 4);
        n0 = FLATCC_MAX_IDENT_SHOW;
    }
    error_report(P, ref->ident, msg, s2 ? s2->ident : 0, buf, n0);
}

//#define LEX_DEBUG

/* Flatbuffers reserve keywords. */
#define LEX_KEYWORDS

#define LEX_C_BLOCK_COMMENT
/*
 * Flatbuffers also support /// on a single line for documentation but
 * we can handle that within the normal line comment parsing logic.
 */
#define LEX_C99_LINE_COMMENT
/*
 * String escapes are not defined in fb schema but it only uses strings
 * for attribute, namespace, file ext, and file id. For JSON objects we
 * use C string escapes but control characters must be detected.
 */
#define LEX_C_STRING

/* Accept numbers like -0x42 as integer literals. */
#define LEX_HEX_NUMERIC

#include "parser.h"

#ifdef LEX_DEBUG

static void print_token(fb_token_t *t)
{
    lex_fprint_token(stderr, t->id, t->text, t->text + t->len, t->linenum, t->pos);
}

static void debug_token(const char *info, fb_token_t *t)
{
    fprintf(stderr, "%s\n    ", info);
    print_token(t);
}
#else
#define debug_token(info, t) ((void)0)
#endif

static void revert_metadata(fb_metadata_t **list)
{
    REVERT_LIST(fb_metadata_t, link, list);
}

static void revert_symbols(fb_symbol_t **list)
{
    REVERT_LIST(fb_symbol_t, link, list);
}

static void revert_names(fb_name_t **list)
{
    REVERT_LIST(fb_name_t, link, list);
}

static inline fb_doc_t *fb_add_doc(fb_parser_t *P, fb_token_t *t)
{
    fb_doc_t *p;

    p = new_elem(P, sizeof(*p));
    p->ident = t;
    p->link = P->doc;
    P->doc = p;
    return p;
}

#define fb_assign_doc(P, p) {\
    revert_symbols(&P->doc); p->doc = P->doc; P->doc = 0; }

static inline fb_compound_type_t *fb_add_table(fb_parser_t *P)
{
    fb_compound_type_t *p;

    p = new_elem(P, sizeof(*p));
    p->symbol.link = P->schema.symbols;
    p->symbol.kind = fb_is_table;
    P->schema.symbols = &p->symbol;
    p->scope = P->current_scope;
    fb_assign_doc(P, p);
    return p;
}

static inline fb_compound_type_t *fb_add_struct(fb_parser_t *P)
{
    fb_compound_type_t *p;

    p = new_elem(P, sizeof(*p));
    p->symbol.link = P->schema.symbols;
    p->symbol.kind = fb_is_struct;
    P->schema.symbols = &p->symbol;
    p->scope = P->current_scope;
    fb_assign_doc(P, p);
    return p;
}

static inline fb_compound_type_t *fb_add_rpc_service(fb_parser_t *P)
{
    fb_compound_type_t *p;

    p = new_elem(P, sizeof(*p));
    p->symbol.link = P->schema.symbols;
    p->symbol.kind = fb_is_rpc_service;
    P->schema.symbols = &p->symbol;
    p->scope = P->current_scope;
    fb_assign_doc(P, p);
    return p;
}

static inline fb_compound_type_t *fb_add_enum(fb_parser_t *P)
{
    fb_compound_type_t *p;

    p = new_elem(P, sizeof(*p));
    p->symbol.link = P->schema.symbols;
    p->symbol.kind = fb_is_enum;
    P->schema.symbols = &p->symbol;
    p->scope = P->current_scope;
    fb_assign_doc(P, p);
    return p;
}

static inline fb_compound_type_t *fb_add_union(fb_parser_t *P)
{
    fb_compound_type_t *p;

    p = new_elem(P, sizeof(*p));
    p->symbol.link = P->schema.symbols;
    p->symbol.kind = fb_is_union;
    P->schema.symbols = &p->symbol;
    p->scope = P->current_scope;
    fb_assign_doc(P, p);
    return p;
}

static inline fb_ref_t *fb_add_ref(fb_parser_t *P, fb_token_t *t)
{
    fb_ref_t *p;

    p = new_elem(P, sizeof(*p));
    p->ident = t;
    return p;
}

static inline fb_attribute_t *fb_add_attribute(fb_parser_t *P)
{
    fb_attribute_t *p;

    p = new_elem(P, sizeof(*p));
    p->name.link = P->schema.attributes;
    P->schema.attributes = &p->name;
    return p;
}

static inline fb_include_t *fb_add_include(fb_parser_t *P)
{
    fb_include_t *p;
    p = new_elem(P, sizeof(*p));
    p->link = P->schema.includes;
    return P->schema.includes = p;
}

static inline fb_scope_t *fb_add_scope(fb_parser_t *P, fb_ref_t *name)
{
    fb_scope_t *p;

    p = fb_scope_table_find(&P->schema.root_schema->scope_index, name, 0);
    if (p) {
        return p;
    }
    p = new_elem(P, sizeof(*p));
    p->name = name;
    p->prefix = P->schema.prefix;

    fb_scope_table_insert_item(&P->schema.root_schema->scope_index, p, ht_keep);
    return p;
}

static inline fb_metadata_t *fb_add_metadata(fb_parser_t *P, fb_metadata_t **metadata)
{
    fb_metadata_t *p;
    p = new_elem(P, sizeof(*p));
    p->link = *metadata;
    return *metadata = p;
}

static inline fb_member_t *fb_add_member(fb_parser_t *P, fb_symbol_t **members)
{
    fb_member_t *p;
    p = new_elem(P, sizeof(*p));
    p->symbol.link = *members;
    p->symbol.kind = fb_is_member;
    *members = (fb_symbol_t *)p;
    fb_assign_doc(P, p);
    return p;
}

static inline int is_end(fb_token_t *t)
{
    return t->id == LEX_TOK_EOF;
}

static fb_token_t *next(fb_parser_t *P)
{
again:
    ++P->token;
    if (P->token == P->te) {
        /* We keep returning end of token to help binary operators etc., if any. */
        --P->token;
        assert(0);
        switch (P->token->id) {
        case LEX_TOK_EOS: case LEX_TOK_EOB: case LEX_TOK_EOF:
            P->token->id = LEX_TOK_EOF;
            return P->token;
        }
        error_tok(P, P->token, "unexpected end of input");
    }
    if (P->token->id == tok_kw_doc_comment) {
        /* Note: we can have blanks that are control characters here, such as \t. */
        fb_add_doc(P, P->token);
        goto again;
    }
    debug_token("next", P->token);
    return P->token;
}

static void recover(fb_parser_t *P, long token_id, int consume)
{
    while (!is_end(P->token)) {
        if (P->token->id == token_id) {
            if (consume) {
                next(P);
            }
            P->doc = 0;
            return;
        }
        next(P);
    }
}

static void recover2(fb_parser_t *P, long token_id, int consume, long token_id_2, int consume_2)
{
    while (!is_end(P->token)) {
        if (P->token->id == token_id) {
            if (consume) {
                next(P);
            }
            P->doc = 0;
            return;
        }
        if (P->token->id == token_id_2) {
            if (consume_2) {
                next(P);
            }
            P->doc = 0;
            return;
        }
        next(P);
    }
}

static inline fb_token_t *optional(fb_parser_t *P, long id) {
    fb_token_t *t = 0;
    if (P->token->id == id) {
        t = P->token;
        next(P);
    }
    return t;
}

static inline fb_token_t *match(fb_parser_t *P, long id, char *msg) {
    fb_token_t *t = 0;
    if (P->token->id == id) {
        t = P->token;
        next(P);
    } else {
        error_tok(P, P->token, msg);
    }
    return t;
}

/*
 * When a keyword should also be accepted as an identifier.
 * This is useful for JSON where field naems are visible.
 * Since field names are not referenced within the schema,
 * this is generally safe. Enums can also be resererved but
 * they can then not be used as default values. Table names
 * and other type names should not be remapped as they can then
 * not by used as a type name for other fields.
 */
#if FLATCC_ALLOW_KW_FIELDS
static inline void remap_field_ident(fb_parser_t *P)
{
    if (P->token->id >= LEX_TOK_KW_BASE && P->token->id < LEX_TOK_KW_END) {
        P->token->id = LEX_TOK_ID;
    }
}
#else
static inline void remap_field_ident(fb_parser_t *P) { (void)P; }
#endif

#if FLATCC_ALLOW_KW_ENUMS
static inline void remap_enum_ident(fb_parser_t *P)
{
    if (P->token->id >= LEX_TOK_KW_BASE && P->token->id < LEX_TOK_KW_END) {
        P->token->id = LEX_TOK_ID;
    }
}
#else
static inline void remap_enum_ident(fb_parser_t *P) { (void)P; }
#endif

static fb_token_t *advance(fb_parser_t *P, long id, const char *msg, fb_token_t *peer)
{
    /*
     * `advance` is generally used at end of statements so it is a
     * convenient place to get rid of rogue doc comments we can't attach
     * to anything meaningful.
     */
    P->doc = 0;
    if (P->token->id != id) {
        error_tok_2(P, P->token, msg, peer);
        return P->token;
    }
    return next(P);
}

static void read_integer_value(fb_parser_t *P, fb_token_t *t, fb_value_t *v, int sign)
{
    int status;

    v->type = vt_uint;
    /* The token does not store the sign internally. */
    parse_integer(t->text, (size_t)t->len, &v->u, &status);
    if (status != PARSE_INTEGER_UNSIGNED) {
        v->type = vt_invalid;
        error_tok(P, t, "invalid integer format");
    }
    if (sign) {
        v->i = -(int64_t)v->u;
        v->type = vt_int;
#ifdef FLATCC_FAIL_ON_INT_SIGN_OVERFLOW
        /* Sometimes we might want this, so don't fail by default. */
        if (v->i > 0) {
            v->type = vt_invalid;
            error_tok(P, t, "sign overflow in integer format");
        }
#endif
    }
}

static void read_hex_value(fb_parser_t *P, fb_token_t *t, fb_value_t *v, int sign)
{
    int status;

    v->type = vt_uint;
    /* The token does not store the sign internally. */
    parse_hex_integer(t->text, (size_t)t->len, &v->u, &status);
    if (status != PARSE_INTEGER_UNSIGNED) {
        v->type = vt_invalid;
        error_tok(P, t, "invalid hex integer format");
    }
    if (sign) {
        v->i = -(int64_t)v->u;
        v->type = vt_int;
#ifdef FLATCC_FAIL_ON_INT_SIGN_OVERFLOW
        /* Sometimes we might want this, so don't fail by default. */
        if (v->i > 0) {
            v->type = vt_invalid;
            error_tok(P, t, "sign overflow in hex integer format");
        }
#endif
    }
}

static void read_float_value(fb_parser_t *P, fb_token_t *t, fb_value_t *v, int sign)
{
    char *end;

    v->type = vt_float;
    v->f = strtod(t->text, &end);
    if (end != t->text + t->len) {
        v->type = vt_invalid;
        error_tok(P, t, "invalid float format");
    } else if (t->text[0] == '.') {
        v->type = vt_invalid;
        /* The FB spec requires this, in line with the JSON format. */
        error_tok(P, t, "numeric values must start with a digit");
    } else if (sign) {
        v->f = -v->f;
    }
}

/*
 * We disallow escape characters, newlines and other control characters,
 * but especially escape characters because they would require us to
 * reallocate the string and convert the escaped characters. We also
 * disallow non-utf8 characters, but we do not check for it. The tab
 * character could meaningfully be accepted, but we don't.
 *
 * String literals are only used to name attributes, namespaces,
 * file identifiers and file externsions, so we really have no need
 * for these extra featuresescape .
 *
 * JSON strings should be handled separately, if or when supported -
 * either by converting escapes and reallocating the string, or
 * simply by ignoring the escape errors and use the string unmodified.
 */
static void parse_string_literal(fb_parser_t *P, fb_value_t *v)
{
    fb_token_t *t;

    v->type = vt_string;
    v->s.s = 0;
    v->s.len = 0;

    for (;;) {
        t = P->token;
        switch (t->id) {
        case LEX_TOK_STRING_PART:
            if (v->s.s == 0) {
                v->s.s = (char *)t->text;
            }
            break;
        case LEX_TOK_STRING_ESCAPE:
            v->type = vt_invalid;
            error_tok(P, t, "escape not allowed in strings");
            break;
        case LEX_TOK_STRING_CTRL:
            v->type = vt_invalid;
            error_tok_as_string(P, t, "control characters not allowed in strings", "?", 1);
            break;
        case LEX_TOK_STRING_NEWLINE:
            v->type = vt_invalid;
            error_tok(P, t, "newline not allowed in strings");
            break;
        case LEX_TOK_STRING_UNTERMINATED:
        case LEX_TOK_STRING_END:
            goto done;

        default:
            error_tok(P, t, "internal error: unexpected token in string");
            v->type = vt_invalid;
            goto done;
        }
        next(P);
    }
done:
    /*
     * If we were to ignore all errors, we would get the full
     * string as is excluding delimiting quotes.
     */
    if (v->s.s) {
        v->s.len = (int)(P->token->text - v->s.s);
    }
    if (!match(P, LEX_TOK_STRING_END, "unterminated string")) {
        v->type = vt_invalid;
    }
}

/* Current token must be an identifier. */
static void parse_ref(fb_parser_t *P, fb_ref_t **ref)
{
    *ref = fb_add_ref(P, P->token);
    next(P);
    ref = &((*ref)->link);
    while (optional(P, '.')) {
        if (P->token->id != LEX_TOK_ID) {
            error_tok(P, P->token, "namespace prefix expected identifier");
            break;
        }
        *ref = fb_add_ref(P, P->token);
        ref = &((*ref)->link);
        next(P);
    }
}

/* `flags` */
enum { allow_string_value = 1, allow_id_value = 2, allow_null_value = 4 };
static void parse_value(fb_parser_t *P, fb_value_t *v, int flags, const char *error_msg)
{
    fb_token_t *t;
    fb_token_t *sign;

    sign = optional(P, '-');
    t = P->token;

    switch (t->id) {
    case LEX_TOK_HEX:
        read_hex_value(P, t, v, sign != 0);
        break;
    case LEX_TOK_INT:
        read_integer_value(P, t, v, sign != 0);
        break;
    case LEX_TOK_FLOAT:
        read_float_value(P, t, v, sign != 0);
        break;
    case tok_kw_true:
        v->b = 1;
        v->type = vt_bool;
        break;
    case tok_kw_false:
        v->b = 0;
        v->type = vt_bool;
        break;
    case tok_kw_null:
        if (!(flags & allow_null_value)) {
            v->type = vt_invalid;
            error_tok(P, t, error_msg);
            return;
        }
        v->type = vt_null;
        break;
    case LEX_TOK_STRING_BEGIN:
        next(P);
        parse_string_literal(P, v);
        if (!(flags & allow_string_value)) {
            v->type = vt_invalid;
            error_tok(P, t, error_msg);
            return;
        }
        if (sign) {
            v->type = vt_invalid;
            error_tok(P, t, "string constants cannot be signed");
            return;
        }
        return;
    case LEX_TOK_ID:
        parse_ref(P, &v->ref);
        v->type = vt_name_ref;
        if (sign) {
            v->type = vt_invalid;
            /* Technically they could, but we do not allow it. */
            error_tok(P, t, "named values cannot be signed");
        }
        return;
    default:
        /* We might have consumed a sign, but never mind that. */
        error_tok(P, t, error_msg);
        return;
    }
    if (sign && v->type == vt_bool) {
        v->type = vt_invalid;
        error_tok(P, t, "boolean constants cannot be signed");
    }
    next(P);
}

static void parse_fixed_array_size(fb_parser_t *P, fb_token_t *ttype, fb_value_t *v)
{
    const char *error_msg = "fixed length array length expected to be an unsigned integer";
    fb_value_t vsize;
    fb_token_t *tlen = P->token;

    parse_value(P, &vsize, 0, error_msg);
    if (vsize.type != vt_uint) {
        error_tok(P, tlen, error_msg);
        v->type = vt_invalid;
        return;
    }
    if (v->type == vt_invalid) return;
    switch (v->type) {
    case vt_vector_type:
        v->type = vt_fixed_array_type;
        break;
    case vt_vector_type_ref:
        v->type = vt_fixed_array_type_ref;
        break;
    case vt_vector_string_type:
        v->type = vt_fixed_array_string_type;
        break;
    case vt_invalid:
        return;
    default:
        error_tok(P, ttype, "invalid fixed length array type");
        v->type = vt_invalid;
        return;
    }
    if (vsize.u == 0) {
        error_tok(P, tlen, "fixed length array length cannot be 0");
        v->type = vt_invalid;
        return;
    }
    /*
     * This allows for safe 64-bit multiplication by elements no
     * larger than 2^32-1 and also fits into the value len field.
     * without extra size cost.
     */
    if (vsize.u > UINT32_MAX) {
        error_tok(P, tlen, "fixed length array length overflow");
        v->type = vt_invalid;
        return;
    }
    v->len = (uint32_t)vsize.u;
}

/* ':' must already be matched */
static void parse_type(fb_parser_t *P, fb_value_t *v)
{
    fb_token_t *t = 0;
    fb_token_t *ttype = 0;
    fb_token_t *t0 = P->token;
    int vector = 0;

    v->len = 1;
    v->type = vt_invalid;
    while ((t = optional(P, '['))) {
        ++vector;
    }
    if (vector > 1) {
        error_tok(P, t0, "vector type can only be one-dimensional");
    }
    ttype = P->token;
    switch (ttype->id) {
    case tok_kw_int:
    case tok_kw_bool:
    case tok_kw_byte:
    case tok_kw_long:
    case tok_kw_uint:
    case tok_kw_float:
    case tok_kw_short:
    case tok_kw_char:
    case tok_kw_ubyte:
    case tok_kw_ulong:
    case tok_kw_ushort:
    case tok_kw_double:
    case tok_kw_int8:
    case tok_kw_int16:
    case tok_kw_int32:
    case tok_kw_int64:
    case tok_kw_uint8:
    case tok_kw_uint16:
    case tok_kw_uint32:
    case tok_kw_uint64:
    case tok_kw_float32:
    case tok_kw_float64:
        v->t = P->token;
        v->type = vector ? vt_vector_type : vt_scalar_type;
        next(P);
        break;
    case tok_kw_string:
        v->t = P->token;
        v->type = vector ? vt_vector_string_type : vt_string_type;
        next(P);
        break;
    case LEX_TOK_ID:
        parse_ref(P, &v->ref);
        v->type = vector ? vt_vector_type_ref : vt_type_ref;
        break;
    case ']':
        error_tok(P, t, "vector type cannot be empty");
        break;
    default:
        error_tok(P, ttype, "invalid type specifier");
        break;
    }
    if (vector && optional(P, ':')) {
        parse_fixed_array_size(P, ttype, v);
    }
    while (optional(P, ']') && vector--) {
    }
    if (vector) {
        error_tok_2(P, t, "vector type missing ']' to match", t0);
    }
    if ((t = optional(P, ']'))) {
        error_tok_2(P, t, "extra ']' not matching", t0);
        while (optional(P, ']')) {
        }
    }
    if (ttype->id == tok_kw_char && v->type != vt_invalid) {
        if (v->type != vt_fixed_array_type) {
            error_tok(P, ttype, "char can only be used as a fixed length array type [char:<n>]");
            v->type = vt_invalid;
        }
    }
}

static fb_metadata_t *parse_metadata(fb_parser_t *P)
{
    fb_token_t *t, *t0;
    fb_metadata_t *md = 0;

    if (!(t0 = optional(P, '('))) {
        return 0;
    }
    if ((t = optional(P, LEX_TOK_ID)))
    for (;;) {
        fb_add_metadata(P, &md);
        md->ident = t;
        if (optional(P, ':')) {
            parse_value(P, &md->value, allow_string_value, "scalar or string value expected");
        }
        if (P->failed >= FLATCC_MAX_ERRORS) {
            return md;
        }
        if (!optional(P, ',')) {
            break;
        }
        if (!(t = match(P, LEX_TOK_ID, "attribute name expected identifier after ','"))) {
            break;
        }
    }
    advance(P, ')', "metadata expected ')' to match", t0);
    revert_metadata(&md);
    return md;
}

static void parse_field(fb_parser_t *P, fb_member_t *fld)
{
    fb_token_t *t;

    remap_field_ident(P);
    if (!(t = match(P, LEX_TOK_ID, "field expected identifier"))) {
        goto fail;
    }
    fld->symbol.ident = t;
    if (!match(P, ':', "field expected ':' before mandatory type")) {
        goto fail;
    }
    parse_type(P, &fld->type);
    if (optional(P, '=')) {
        /*
         * Because types can be named references, we do not check the
         * default assignment before the schema is fully parsed.
         * We allow the initializer to be a name in case it is an enum
         * name.
         */
        parse_value(P, &fld->value, allow_id_value | allow_null_value, "initializer must be of scalar type or null");
    }
    fld->metadata = parse_metadata(P);
    advance(P, ';', "field must be terminated with ';'", 0);
    return;
fail:
    recover2(P, ';', 1, '}', 0);
}

static void parse_method(fb_parser_t *P, fb_member_t *fld)
{
    fb_token_t *t;
    if (!(t = match(P, LEX_TOK_ID, "method expected identifier"))) {
        goto fail;
    }
    fld->symbol.ident = t;
    if (!match(P, '(', "method expected '(' after identifier")) {
        goto fail;
    }
    parse_type(P, &fld->req_type);
    if (!match(P, ')', "method expected ')' after request type")) {
        goto fail;
    }
    if (!match(P, ':', "method expected ':' before mandatory response type")) {
        goto fail;
    }
    parse_type(P, &fld->type);
    if ((t = optional(P, '='))) {
        error_tok(P, t, "method does not accept an initializer");
        goto fail;
    }
    fld->metadata = parse_metadata(P);
    advance(P, ';', "method must be terminated with ';'", 0);
    return;
fail:
    recover2(P, ';', 1, '}', 0);
}

/* `enum` must already be matched. */
static void parse_enum_decl(fb_parser_t *P, fb_compound_type_t *ct)
{
    fb_token_t *t, *t0;
    fb_member_t *member;

    if (!(ct->symbol.ident = match(P, LEX_TOK_ID, "enum declaration expected identifier"))) {
        goto fail;
    }
    if (optional(P, ':')) {
        parse_type(P, &ct->type);
        if (ct->type.type != vt_scalar_type) {
            error_tok(P, ct->type.t, "integral type expected");
        } else {
            switch (ct->type.t->id) {
            case tok_kw_float:
            case tok_kw_double:
            case tok_kw_float32:
            case tok_kw_float64:
                error_tok(P, ct->type.t, "integral type expected");
                break;
            default:
                break;
            }
        }
    }
    ct->metadata = parse_metadata(P);
    if (!((t0 = match(P, '{', "enum declaration expected '{'")))) {
        goto fail;
    }
    for (;;) {
        remap_enum_ident(P);
        if (!(t = match(P, LEX_TOK_ID,
                "member identifier expected"))) {
            goto fail;
        }
        if (P->failed >= FLATCC_MAX_ERRORS) {
            goto fail;
        }
        member = fb_add_member(P, &ct->members);
        member->symbol.ident = t;
        if (optional(P, '=')) {
            t = P->token;
            parse_value(P, &member->value, 0, "integral constant expected");
            /* Leave detailed type (e.g. no floats) and range checking to a later stage. */
        }
        /*
         * Trailing comma is optional in flatc but not in grammar, we
         * follow flatc.
         */
        if (!optional(P, ',') || P->token->id == '}') {
            break;
        }
        P->doc = 0;
    }
    if (t0) {
        advance(P, '}', "enum missing closing '}' to match", t0);
    }
    revert_symbols(&ct->members);
    return;
fail:
    recover(P, '}', 1);
}

/* `union` must already be matched. */
static void parse_union_decl(fb_parser_t *P, fb_compound_type_t *ct)
{
    fb_token_t *t0;
    fb_member_t *member;
    fb_ref_t *ref;
    fb_token_t *t;

    if (!(ct->symbol.ident = match(P, LEX_TOK_ID, "union declaration expected identifier"))) {
        goto fail;
    }
    ct->metadata = parse_metadata(P);
    if (!((t0 = match(P, '{', "union declaration expected '{'")))) {
        goto fail;
    }
    for (;;) {
        if (P->token->id != LEX_TOK_ID) {
            error_tok(P, P->token, "union expects an identifier");
            goto fail;
        }
        if (P->failed >= FLATCC_MAX_ERRORS) {
            goto fail;
        }
        t = P->token;
        member = fb_add_member(P, &ct->members);
        parse_ref(P, &ref);
        member->type.ref = ref;
        member->type.type = vt_type_ref;
        while (ref->link) {
            ref = ref->link;
        }
        /* The union member name is the unqualified reference. */
        member->symbol.ident = ref->ident;
        if (optional(P, ':')) {
            if (member->type.ref->link) {
                error_tok(P, t, "qualified union member name cannot have an explicit type");
            }
            parse_type(P, &member->type);
            /* Leave type checking to later stage. */
        }
        if (optional(P, '=')) {
            parse_value(P, &member->value, 0, "integral constant expected");
            /* Leave detailed type (e.g. no floats) and range checking to a later stage. */
        }
        if (!optional(P, ',') || P->token->id == '}') {
            break;
        }
        P->doc = 0;
    }
    advance(P, '}', "union missing closing '}' to match", t0);
    revert_symbols(&ct->members);
    /* Add implicit `NONE` member first in the list. */
    member = fb_add_member(P, &ct->members);
    member->symbol.ident = &P->t_none;
    return;
fail:
    recover2(P, ';', 1, '}', 0);
}

/* `struct` , `table`, or 'rpc_service' must already be matched. */
static void parse_compound_type(fb_parser_t *P, fb_compound_type_t *ct, long token)
{
    fb_token_t *t = 0;

    if (!(t = match(P, LEX_TOK_ID, "Declaration expected an identifier"))) {
        goto fail;
    }
    ct->symbol.ident = t;
    ct->metadata = parse_metadata(P);
    if (!(match(P, '{', "Declaration expected '{'"))) {
        goto fail;
    }
    t = P->token;

/* Allow empty tables and structs. */
#if 0
    if (P->token->id == '}') {
        error_tok(P, t, "table / struct declaration cannot be empty");
    }
#endif
    while (P->token->id != '}') {
        if (token == tok_kw_rpc_service) {
            parse_method(P, fb_add_member(P, &ct->members));
        } else {
            parse_field(P, fb_add_member(P, &ct->members));
        }
        if (P->failed >= FLATCC_MAX_ERRORS) {
            goto fail;
        }
    }
    if (!optional(P, '}') && t) {
        error_tok_2(P, P->token, "Declaration missing closing '}' to match", t);
    }
    revert_symbols(&ct->members);
    return;
fail:
    recover(P, '}', 1);
}

static void parse_namespace(fb_parser_t *P)
{
    fb_ref_t *ref = 0;
    fb_token_t *t = P->token;

    if (optional(P, ';') && t) {
        /* Revert to global namespace. */
        P->current_scope = P->root_scope;
        return;
    }
    if (P->token->id != LEX_TOK_ID) {
        error_tok(P, P->token, "namespace expects an identifier");
        recover(P, ';', 1);
        return;
    }
    parse_ref(P, &ref);
    advance(P, ';', "missing ';' expected by namespace at", t);
    P->current_scope = fb_add_scope(P, ref);
}

static void parse_root_type(fb_parser_t *P, fb_root_type_t *rt)
{
    fb_token_t *t = P->token;

    if (rt->name) {
        error_tok(P, P->token, "root_type already set");
    }
    parse_ref(P, &rt->name);
    rt->scope = P->current_scope;
    advance(P, ';', "missing ';' expected by root_type at", t);
}

static void parse_include(fb_parser_t *P)
{
    fb_token_t *t = P->token;

    while (optional(P, tok_kw_include)) {
        if (P->opts.disable_includes) {
            error_tok(P, t, "include statements not supported by current environment");
        }
        if (P->failed >= FLATCC_MAX_ERRORS) {
            return;
        }
        if (!match(P, LEX_TOK_STRING_BEGIN,
                    "include expected a string literal as filename")) {
            recover(P, ';', 1);
        }
        parse_string_literal(P, &fb_add_include(P)->name);
        match(P, ';', "include statement expected ';'");
    }
}

static void parse_attribute(fb_parser_t *P, fb_attribute_t *a)
{
    fb_token_t *t = P->token;

    if (match(P, LEX_TOK_STRING_BEGIN, "attribute expected string literal")) {
        parse_string_literal(P, &a->name.name);
        if (a->name.name.s.len == 0) {
            error_tok_as_string(P, t, "attribute name cannot be empty", 0, 0);
        }
    }
    match(P, ';', "attribute expected ';'");
}

static void parse_file_extension(fb_parser_t *P, fb_value_t *v)
{
    if (v->type == vt_string) {
        error_tok_as_string(P, P->token, "file extension already set", v->s.s, (size_t)v->s.len);
    }
    if (!match(P, LEX_TOK_STRING_BEGIN, "file_extension expected string literal")) {
        goto fail;
    }
    parse_string_literal(P, v);
    match(P, ';', "file_extension expected ';'");
    return;
fail:
    recover(P, ';', 1);
}

static void parse_file_identifier(fb_parser_t *P, fb_value_t *v)
{
    fb_token_t *t;
    if (v->type != vt_missing) {
        error_tok_as_string(P, P->token, "file identifier already set", v->s.s, (size_t)v->s.len);
    }
    if (!match(P, LEX_TOK_STRING_BEGIN, "file_identifier expected string literal")) {
        goto fail;
    }
    t = P->token;
    parse_string_literal(P, v);
    if (v->s.s && v->s.len != 4) {
        v->type = vt_invalid;
        error_tok(P, t, "file_identifier must be 4 characters");
    }
    match(P, ';', "file_identifier expected ';'");
    return;
fail:
    recover(P, ';', 1);
}

static void parse_schema_decl(fb_parser_t *P)
{
    switch(P->token->id) {
    case tok_kw_namespace:
        next(P);
        parse_namespace(P);
        break;
    case tok_kw_file_extension:
        next(P);
        parse_file_extension(P, &P->schema.file_extension);
        break;
    case tok_kw_file_identifier:
        next(P);
        parse_file_identifier(P, &P->schema.file_identifier);
        break;
    case tok_kw_root_type:
        next(P);
        parse_root_type(P, &P->schema.root_type);
        break;
    case tok_kw_attribute:
        next(P);
        parse_attribute(P, fb_add_attribute(P));
        break;
    case tok_kw_struct:
        next(P);
        parse_compound_type(P, fb_add_struct(P), tok_kw_struct);
        break;
    case tok_kw_table:
        next(P);
        parse_compound_type(P, fb_add_table(P), tok_kw_table);
        break;
    case tok_kw_rpc_service:
        next(P);
        parse_compound_type(P, fb_add_rpc_service(P), tok_kw_rpc_service);
        break;
    case tok_kw_enum:
        next(P);
        parse_enum_decl(P, fb_add_enum(P));
        break;
    case tok_kw_union:
        next(P);
        parse_union_decl(P, fb_add_union(P));
        break;
    case tok_kw_include:
        error_tok(P, P->token, "include statements must be placed first in the schema");
        break;
    case '{':
        error_tok(P, P->token, "JSON objects in schema file is not supported - but a schema specific JSON parser can be generated");
        break;
    case LEX_TOK_CTRL:
        error_tok_as_string(P, P->token, "unexpected control character in schema definition", "?", 1);
        break;
    case LEX_TOK_COMMENT_CTRL:
        error_tok_as_string(P, P->token, "unexpected control character in comment", "?", 1);
        break;
    case LEX_TOK_COMMENT_UNTERMINATED:
        error_tok_as_string(P, P->token, "unterminated comment", "<eof>", 5);
        break;
    default:
        error_tok(P, P->token, "unexpected token in schema definition");
        break;
    }
}

static int parse_schema(fb_parser_t *P)
{
    fb_token_t *t, *t0;
    parse_include(P);
    t = P->token;
    for (;;) {
        if (is_end(t)) {
            break;
        }
        if (P->failed >= FLATCC_MAX_ERRORS) {
            return -1;
        }
        t0 = t;
        parse_schema_decl(P);
        t = P->token;
        if (t == t0) {
            if (P->failed) {
                return -1;
            }
            error_tok(P, t, "extra tokens in input");
            return -1;
        }
    }
    revert_names(&P->schema.attributes);
    revert_symbols(&P->schema.symbols);
    return 0;
}

static inline void clear_elem_buffers(fb_parser_t *P)
{
    void **p, **p2;

    p = P->elem_buffers;
    while (p) {
        p2 = *((void**)p);
        free(p);
        p = p2;
    };
}

static void push_token(fb_parser_t *P, long id, const char *first, const char *last)
{
    size_t offset;
    fb_token_t *t;

    if (P->token == P->te) {
        offset = (size_t)(P->token - P->ts);
        P->tcapacity = P->tcapacity ? 2 * P->tcapacity : 1024;
        P->ts = realloc(P->ts, (size_t)P->tcapacity * sizeof(fb_token_t));
        checkmem(P->ts);
        P->te = P->ts + P->tcapacity;
        P->token = P->ts + offset;
    }
    t = P->token;
    t->id = id;
    t->text = first;
    t->len = (long)(last - first);
    t->linenum = P->linenum;
    t->pos = (long)(first - P->line + 1);
    ++P->token;
}

/*
 * If the file contains a control character, we can get multiple
 * comments per line.
 */
static inline void push_comment(fb_parser_t *P, const char *first, const char *last)
{
    if (P->doc_mode) {
        push_token(P, tok_kw_doc_comment, first, last);
    }
}

static void inject_token(fb_token_t *t, const char *lex, long id)
{
    t->id = id;
    t->text = lex;
    t->len = (long)strlen(lex);
    t->pos = 0;
    t->linenum = 0;
}

/* --- Customize lexer --- */

/* Depends on the `context` argument given to the lex function. */
#define ctx(name) (((fb_parser_t *)context)->name)

#define lex_emit_newline(first, last) (ctx(linenum)++, ctx(line) = last)

#define lex_emit_string_newline(first, last)                            \
    (ctx(linenum)++, ctx(line) = last,                                  \
    push_token((fb_parser_t*)context, LEX_TOK_STRING_NEWLINE, first, last))

/*
 * Add emtpy comment on comment start - otherwise we miss empty lines.
 * Save is_doc becuase comment_part does not remember.
 */
#define lex_emit_comment_begin(first, last, is_doc)                     \
    { ctx(doc_mode) = is_doc; push_comment((fb_parser_t*)context, last, last); }
#define lex_emit_comment_part(first, last) push_comment((fb_parser_t*)context, first, last)
#define lex_emit_comment_end(first, last) (ctx(doc_mode) = 0)

/* By default emitted as lex_emit_other which would be ignored. */
#define lex_emit_comment_unterminated(pos)                                  \
    push_token((fb_parser_t*)context, LEX_TOK_COMMENT_UNTERMINATED, pos, pos)

#define lex_emit_comment_ctrl(pos)                                          \
    if (lex_isblank(*pos) || !lex_isctrl(*pos)) {                           \
        push_comment((fb_parser_t*)context, pos, pos + 1);                  \
    } else {                                                                \
        push_token((fb_parser_t*)context, LEX_TOK_CTRL,                     \
                pos, pos + 1);                                              \
    }

/*
 * Provide hook to lexer for emitting tokens. We can override many
 * things, but most default to calling lex_emit, so that is all we need
 * to handle.
 *
 * `context` is a magic name available to macros in the lexer.
 */
#define lex_emit(token, first, last)                                    \
    push_token((fb_parser_t*)context, token, first, last)

/*
 * We could just eos directly as it defaults to emit, but formally we
 * should use the eof marker which is always zero, so parser can check
 * for it easily, if needed.
 */
#define lex_emit_eos(first, last)                                       \
    push_token((fb_parser_t*)context, LEX_TOK_EOF, first, last)

/*
 * This event happens in place of eos if we exhaust the input buffer.
 * In this case we treat this as end of input, but this choice prevents
 * us from parsing across multiple buffers.
 */
#define lex_emit_eob(pos)                                       \
    push_token((fb_parser_t*)context, LEX_TOK_EOF, pos, pos)

/*
 * Luthor is our speedy generic lexer - it knows most common operators
 * and therefore allows us to fail meaningfully on those that we don't
 * support here, which is most.
 */
#include "lex/luthor.c"

#include "keywords.h"

/* Root schema `rs` is null for top level parser. */
int fb_init_parser(fb_parser_t *P, fb_options_t *opts, const char *name,
        fb_error_fun error_out, void *error_ctx, fb_root_schema_t *rs)
{
    size_t n, name_len;
    char *s;

    memset(P, 0, sizeof(*P));

    if (error_out) {
        P->error_out = error_out;
        P->error_ctx = error_ctx;
    } else {
        P->error_out = fb_default_error_out;
    }
    if (opts) {
        memcpy(&P->opts, opts, sizeof(*opts));
    } else {
        flatcc_init_options(&P->opts);
    }
    P->schema.root_schema = rs ? rs : &P->schema.root_schema_instance;
    switch (P->opts.offset_size) {
    case 2:
    case 4:
    case 8:
        break;
    default:
        error(P, "invalid offset configured, must be 2, 4 (default), or 8");
        return -1;
    }
    switch (P->opts.voffset_size) {
    case 2:
    case 4:
    case 8:
        break;
    default:
        error(P, "invalid voffset configured, must be 2 (default), 4, or 8");
        return -1;
    }
    if (!name) {
        /* Mostly for testing, just so we always have a name. */
        name = FLATCC_DEFAULT_FILENAME;
    }
    if (name == 0) {
        name = "";
    }
    name_len = strlen(name);
    checkmem((P->schema.basename = fb_create_basename(name, name_len, opts->default_schema_ext)));
    n = strlen(P->schema.basename);
    checkmem(s = fb_copy_path_n(P->schema.basename, n));
    pstrntoupper(s, n);
    P->schema.basenameup = s;
    P->schema.name.name.s.s = s;
    P->schema.name.name.s.len = (int)n;
    checkmem((P->schema.errorname = fb_create_basename(name, name_len, "")));
    P->schema.prefix.s = "";
    P->schema.prefix.len = 0;
    if (opts->ns) {
        P->schema.prefix.s = (char *)opts->ns;
        P->schema.prefix.len = (int)strlen(opts->ns);
    }
    P->root_scope = fb_add_scope(P, 0);
    P->current_scope = P->root_scope;
    assert(P->current_scope == fb_scope_table_find(&P->schema.root_schema->scope_index, 0, 0));
    return 0;
}

/*
 * Main entry function for this specific parser type.
 * We expect a zero terminated string.
 *
 * The parser structure is uninitialized upon entry, and should be
 * cleared with `clear_flatbuffer_parser` subsequently.
 *
 * Datastructures point into the token buffer and into the input
 * buffer, so the parser and input should not be cleared prematurely.
 *
 * The input buffer must remain valid until the parser is cleared
 * because the internal represenation stores pointers into the buffer.
 *
 * `own_buffer` indicates that the the buffer should be deallocated when
 * the parser is cleaned up.
 */
int fb_parse(fb_parser_t *P, const char *input, size_t len, int own_buffer)
{
    static const char *id_none = "NONE";
    static const char *id_ubyte = "ubyte";

    P->line = input;
    P->linenum = 1;

    /* Used with union defaults. */
    inject_token(&P->t_none, id_none, LEX_TOK_ID);
    inject_token(&P->t_ubyte, id_ubyte, tok_kw_ubyte);

    if (own_buffer) {
        P->managed_input = input;
    }
    lex(input, len, 0, P);

    P->te = P->token;
    P->token = P->ts;
    /* Only used while processing table id's. */
    checkmem((P->tmp_field_marker = malloc(sizeof(P->tmp_field_marker[0]) * (size_t)P->opts.vt_max_count)));
    checkmem((P->tmp_field_index = malloc(sizeof(P->tmp_field_index[0]) * (size_t)P->opts.vt_max_count)));
    if (P->token->id == tok_kw_doc_comment) {
        next(P);
    }
    parse_schema(P);
    return P->failed;
}

static void __destroy_scope_item(void *item, fb_scope_t *scope)
{
    /* Each scope points into table that is cleared separately. */
    (void)item;

    fb_symbol_table_clear(&scope->symbol_index);
}

void fb_clear_parser(fb_parser_t *P)
{
    fb_symbol_t *sym;
    fb_compound_type_t *ct;

    for (sym = P->schema.symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_struct:
        case fb_is_table:
        case fb_is_rpc_service:
        case fb_is_enum:
        case fb_is_union:
            ct = (fb_compound_type_t *)sym;
            fb_symbol_table_clear(&ct->index);
            fb_value_set_clear(&ct->value_set);
        }
    }
    fb_schema_table_clear(&P->schema.root_schema_instance.include_index);
    fb_name_table_clear(&P->schema.root_schema_instance.attribute_index);
    ptr_set_clear(&P->schema.visible_schema);
    if (P->tmp_field_marker) {
        free(P->tmp_field_marker);
    }
    if (P->tmp_field_index) {
        free(P->tmp_field_index);
    }
    if (P->ts) {
        free(P->ts);
    }
    if (P->schema.basename) {
        free((void *)P->schema.basename);
    }
    if (P->schema.basenameup) {
        free((void *)P->schema.basenameup);
    }
    if (P->schema.errorname) {
        free((void *)P->schema.errorname);
    }
    /*
     * P->referer_path in included files points to parent P->path, so
     * don't free it, and don't access it after this point.
     */
    if (P->path) {
        free((void *)P->path);
    }
    fb_scope_table_destroy(&P->schema.root_schema_instance.scope_index,
            __destroy_scope_item, 0);
    /* Destroy last since destructor has references into elem buffer. */
    clear_elem_buffers(P);
    if (P->managed_input) {
        free((void *)P->managed_input);
    }
    memset(P, 0, sizeof(*P));
}
