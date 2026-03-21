#ifndef CODEGEN_C_H
#define CODEGEN_C_H

#include <assert.h>
#include <stdarg.h>

#include "symbols.h"
#include "parser.h"
#include "codegen.h"

/* -DFLATCC_PORTABLE may help if inttypes.h is missing. */
#ifndef PRId64
#include <inttypes.h>
#endif

#define __FLATCC_ERROR_TYPE "INTERNAL_ERROR_UNEXPECTED_TYPE"

#ifndef gen_panic
#define gen_panic(context, msg) fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, msg), assert(0), exit(-1)
#endif


static inline void token_name(fb_token_t *t, int *n, const char **s) {
    *n = (int)t->len;
    *s = t->text;
}

typedef char fb_symbol_text_t[FLATCC_NAME_BUFSIZ];
typedef struct fb_scoped_name fb_scoped_name_t;

/* Should be zeroed because scope is cached across updates. */
struct fb_scoped_name {
    fb_symbol_text_t text;
    fb_scope_t *scope;
    int scope_len, len, total_len;
};

#define fb_clear(x) (memset(&(x), 0, sizeof(x)))

/* Returns length or -1 if length exceeds namespace max. */
int __flatcc_fb_copy_scope(fb_scope_t *scope, char *buf);
#define fb_copy_scope __flatcc_fb_copy_scope

void __flatcc_fb_scoped_symbol_name(fb_scope_t *scope, fb_symbol_t *sym, fb_scoped_name_t *sn);
#define fb_scoped_symbol_name __flatcc_fb_scoped_symbol_name

static inline void fb_compound_name(fb_compound_type_t *ct, fb_scoped_name_t *sn)
{
    fb_scoped_symbol_name(ct->scope, &ct->symbol, sn);
}

static inline void symbol_name(fb_symbol_t *sym, int *n, const char **s) {
    token_name(sym->ident, n, s);
}

static inline const char *scalar_type_ns(fb_scalar_type_t scalar_type, const char *ns)
{
    return scalar_type == fb_bool ? ns : "";
}

static inline const char *scalar_type_prefix(fb_scalar_type_t scalar_type)
{
    const char *tname;
    switch (scalar_type) {
    case fb_ulong:
        tname = "uint64";
        break;
    case fb_uint:
        tname = "uint32";
        break;
    case fb_ushort:
        tname = "uint16";
        break;
    case fb_char:
        tname = "char";
        break;
    case fb_ubyte:
        tname = "uint8";
        break;
    case fb_bool:
        tname = "bool";
        break;
    case fb_long:
        tname = "int64";
        break;
    case fb_int:
        tname = "int32";
        break;
    case fb_short:
        tname = "int16";
        break;
    case fb_byte:
        tname = "int8";
        break;
    case fb_float:
        tname = "float";
        break;
    case fb_double:
        tname = "double";
        break;
    default:
        gen_panic(0, "internal error: unexpected type during code generation");
        tname = __FLATCC_ERROR_TYPE;
        break;
    }
    return tname;
}

static inline const char *scalar_type_name(fb_scalar_type_t scalar_type)
{
    const char *tname;
    switch (scalar_type) {
    case fb_ulong:
        tname = "uint64_t";
        break;
    case fb_uint:
        tname = "uint32_t";
        break;
    case fb_ushort:
        tname = "uint16_t";
        break;
    case fb_char:
        tname = "char";
        break;
    case fb_ubyte:
        tname = "uint8_t";
        break;
    case fb_bool:
        tname = "bool_t";
        break;
    case fb_long:
        tname = "int64_t";
        break;
    case fb_int:
        tname = "int32_t";
        break;
    case fb_short:
        tname = "int16_t";
        break;
    case fb_byte:
        tname = "int8_t";
        break;
    case fb_float:
        tname = "float";
        break;
    case fb_double:
        tname = "double";
        break;
    default:
        gen_panic(0, "internal error: unexpected type during code generation");
        tname = __FLATCC_ERROR_TYPE;
        break;
    }
    return tname;
}

static inline const char *scalar_vector_type_name(fb_scalar_type_t scalar_type)
{
    const char *tname;
    switch (scalar_type) {
    case fb_ulong:
        tname = "uint64_vec_t";
        break;
    case fb_uint:
        tname = "uint32_vec_t";
        break;
    case fb_ushort:
        tname = "uint16_vec_t";
        break;
    case fb_char:
        tname = "char_vec_t";
        break;
    case fb_ubyte:
        tname = "uint8_vec_t";
        break;
    case fb_bool:
        tname = "uint8_vec_t";
        break;
    case fb_long:
        tname = "int64_vec_t";
        break;
    case fb_int:
        tname = "int32_vec_t";
        break;
    case fb_short:
        tname = "int16_vec_t";
        break;
    case fb_byte:
        tname = "int8_vec_t";
        break;
    case fb_float:
        tname = "float_vec_t";
        break;
    case fb_double:
        tname = "double_vec_t";
        break;
    default:
        gen_panic(0, "internal error: unexpected type during code generation");
        tname = __FLATCC_ERROR_TYPE;
        break;
    }
    return tname;
}

/* Only for integers. */
static inline const char *scalar_cast(fb_scalar_type_t scalar_type)
{
    const char *cast;
    switch (scalar_type) {
    case fb_ulong:
        cast = "UINT64_C";
        break;
    case fb_uint:
        cast = "UINT32_C";
        break;
    case fb_ushort:
        cast = "UINT16_C";
        break;
    case fb_char:
        cast = "char";
        break;
    case fb_ubyte:
        cast = "UINT8_C";
        break;
    case fb_bool:
        cast = "UINT8_C";
        break;
    case fb_long:
        cast = "INT64_C";
        break;
    case fb_int:
        cast = "INT32_C";
        break;
    case fb_short:
        cast = "INT16_C";
        break;
    case fb_byte:
        cast = "INT8_C";
        break;
    default:
        gen_panic(0, "internal error: unexpected type during code generation");
        cast = "";
        break;
    }
    return cast;
}

typedef char fb_literal_t[100];

static inline size_t print_literal(fb_scalar_type_t scalar_type, const fb_value_t *value, fb_literal_t literal)
{
    const char *cast;

    switch (value->type) {
    case vt_uint:
        cast = scalar_cast(scalar_type);
        return (size_t)sprintf(literal, "%s(%"PRIu64")", cast, (uint64_t)value->u);
        break;
    case vt_int:
        cast = scalar_cast(scalar_type);
        return (size_t)sprintf(literal, "%s(%"PRId64")", cast, (int64_t)value->i);
        break;
    case vt_bool:
        cast = scalar_cast(scalar_type);
        return (size_t)sprintf(literal, "%s(%u)", cast, (unsigned)value->b);
        break;
    case vt_float:
        /*
         * .9g ensures sufficient precision in 32-bit floats and
         * .17g ensures sufficient precision for 64-bit floats (double).
         * The '#' forces a decimal point that would not be printed
         * for integers which would result in the wrong type in C
         * source.
         */
        if (scalar_type == fb_float) {
            return (size_t)sprintf(literal, "%#.9gf", (float)value->f);
        } else {
            return (size_t)sprintf(literal, "%#.17g", (double)value->f);
        }
        break;
    default:
        gen_panic(0, "internal error: unexpected type during code generation");
        *literal = 0;
        return 0;
    }
}

static inline const char *scalar_suffix(fb_scalar_type_t scalar_type)
{
    const char *suffix;
    switch (scalar_type) {
    case fb_ulong:
        suffix = "ULL";
        break;
    case fb_uint:
        suffix = "UL";
        break;
    case fb_ushort:
        suffix = "U";
        break;
    case fb_char:
        suffix = "";
        break;
    case fb_ubyte:
        suffix = "U";
        break;
    case fb_bool:
        suffix = "U";
        break;
    case fb_long:
        suffix = "LL";
        break;
    case fb_int:
        suffix = "L";
        break;
    case fb_short:
        suffix = "";
        break;
    case fb_byte:
        suffix = "";
        break;
    case fb_double:
        suffix = "";
        break;
    case fb_float:
        suffix = "F";
        break;
    default:
        gen_panic(0, "internal error: unexpected type during code generation");
        suffix = "";
        break;
    }
    return suffix;
}

/* See also: https://github.com/philsquared/Catch/issues/376 */
static inline int gen_prologue(fb_output_t *out)
{
    if (out->opts->cgen_pragmas) {
        fprintf(out->fp, "#include \"flatcc/flatcc_prologue.h\"\n");
    }
    return 0;
}

static inline int gen_epilogue(fb_output_t *out)
{
    if (out->opts->cgen_pragmas) {
        fprintf(out->fp, "#include \"flatcc/flatcc_epilogue.h\"\n");
    }
    return 0;
}

/* This assumes the output context is named out which it is by convention. */
#define indent() (out->indent++)
#define unindent() { assert(out->indent); out->indent--; }
#define margin() { out->tmp_indent = out->indent; out->indent = 0; }
#define unmargin() { out->indent = out->tmp_indent; }

/* Redefine names to avoid polluting library namespace. */

int __flatcc_fb_init_output_c(fb_output_t *out, fb_options_t *opts);
#define fb_init_output_c __flatcc_fb_init_output_c

int __flatcc_fb_open_output_file(fb_output_t *out, const char *name, size_t len, const char *ext);
#define fb_open_output_file __flatcc_fb_open_output_file

void __flatcc_fb_close_output_file(fb_output_t *out);
#define fb_close_output_file __flatcc_fb_close_output_file

void __flatcc_fb_gen_c_includes(fb_output_t *out, const char *ext, const char *extup);
#define fb_gen_c_includes __flatcc_fb_gen_c_includes

int __flatcc_fb_gen_common_c_header(fb_output_t *out);
#define fb_gen_common_c_header __flatcc_fb_gen_common_c_header

int __flatcc_fb_gen_common_c_builder_header(fb_output_t *out);
#define fb_gen_common_c_builder_header __flatcc_fb_gen_common_c_builder_header

int __flatcc_fb_gen_c_reader(fb_output_t *out);
#define fb_gen_c_reader __flatcc_fb_gen_c_reader

int __flatcc_fb_gen_c_builder(fb_output_t *out);
#define fb_gen_c_builder __flatcc_fb_gen_c_builder

int __flatcc_fb_gen_c_verifier(fb_output_t *out);
#define fb_gen_c_verifier __flatcc_fb_gen_c_verifier

int __flatcc_fb_gen_c_sorter(fb_output_t *out);
#define fb_gen_c_sorter __flatcc_fb_gen_c_sorter

int __flatcc_fb_gen_c_json_parser(fb_output_t *out);
#define fb_gen_c_json_parser __flatcc_fb_gen_c_json_parser

int __flatcc_fb_gen_c_json_printer(fb_output_t *out);
#define fb_gen_c_json_printer __flatcc_fb_gen_c_json_printer

#endif /* CODEGEN_C_H */
