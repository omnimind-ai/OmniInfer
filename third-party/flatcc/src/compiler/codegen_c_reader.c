#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "codegen_c.h"
#include "codegen_c_sort.h"

static inline int match_kw_identifier(fb_symbol_t *sym)
{
    return (sym->ident->len == 10 &&
            memcmp(sym->ident->text, "identifier", 10) == 0);
}

/*
 * Use of file identifiers for undeclared roots is fuzzy, but we need an
 * identifer for all, so we use the one defined for the current schema
 * file and allow the user to override. This avoids tedious runtime file
 * id arguments to all create calls.
 *
 * As later addition to FlatBuffers, type hashes may replace file
 * identifiers when explicitly stated. These are FNV-1a hashes of the
 * fully qualified type name (dot separated).
 *
 * We generate the type hash both as a native integer constants for use
 * in switch statements, and encoded as a little endian C string for use
 * as a file identifier.
 */
static void print_type_identifier(fb_output_t *out, fb_compound_type_t *ct)
{
    uint8_t buf[17];
    uint8_t *p;
    uint8_t x;
    int i;
    const char *nsc = out->nsc;
    fb_scoped_name_t snt;
    const char *name;
    uint32_t type_hash;
    int conflict = 0;
    fb_symbol_t *sym;
    const char *file_identifier;
    int file_identifier_len;
    const char *quote;

    fb_clear(snt);

    fb_compound_name(ct, &snt);
    name = snt.text;
    type_hash = ct->type_hash;

    /*
     * It's not practical to detect all possible name conflicts, but
     * 'identifier' is common enough to require special handling.
     */
    for (sym = ct->members; sym; sym = sym->link) {
        if (match_kw_identifier(sym)) {
            conflict = 1;
            break;
        }
    }
    if (out->S->file_identifier.type == vt_string) {
        quote = "\"";
        file_identifier = out->S->file_identifier.s.s;
        file_identifier_len = out->S->file_identifier.s.len;
    } else {
        quote = "";
        file_identifier = "0";
        file_identifier_len = 1;
    }
    fprintf(out->fp,
            "#ifndef %s_file_identifier\n"
            "#define %s_file_identifier %s%.*s%s\n"
            "#endif\n",
            name, name, quote, file_identifier_len, file_identifier, quote);
    if (!conflict) {
        /* For backwards compatibility. */
        fprintf(out->fp,
                "/* deprecated, use %s_file_identifier */\n"
                "#ifndef %s_identifier\n"
                "#define %s_identifier %s%.*s%s\n"
                "#endif\n",
                name, name, name, quote, file_identifier_len, file_identifier, quote);
    }
    fprintf(out->fp,
        "#define %s_type_hash ((%sthash_t)0x%lx)\n",
        name, nsc, (unsigned long)(type_hash));
    p = buf;
    i = 4;
    while (i--) {
        *p++ = '\\';
        *p++ = 'x';
        x = type_hash & 0x0f;
        x += x > 9 ? 'a' - 10 : '0';
        type_hash >>= 4;
        p[1] = x;
        x = type_hash & 0x0f;
        x += x > 9 ? 'a' - 10 : '0';
        type_hash >>= 4;
        p[0] = x;
        p += 2;
    }
    *p = '\0';
    fprintf(out->fp,
        "#define %s_type_identifier \"%s\"\n",
        name, buf);
}

static void print_file_extension(fb_output_t *out, fb_compound_type_t *ct)
{
    fb_scoped_name_t snt;
    const char *name;

    fb_clear(snt);
    fb_compound_name(ct, &snt);
    name = snt.text;

    if (out->S->file_extension.type == vt_string) {
        fprintf(out->fp,
                "#ifndef %s_file_extension\n"
                "#define %s_file_extension \"%.*s\"\n"
                "#endif\n",
                name, name, out->S->file_extension.s.len, out->S->file_extension.s.s);
    } else {
        fprintf(out->fp,
                "#ifndef %s_file_extension\n"
                "#define %s_file_extension \"%s\"\n"
                "#endif\n",
                name, name, out->opts->default_bin_ext);
    }
}

/* Finds first occurrence of matching key when vector is sorted on the named field. */
static void gen_find(fb_output_t *out)
{
    const char *nsc = out->nsc;

    /*
     * E: Element accessor (elem = E(vector, index)).
     * L: Length accessor (length = L(vector)).
     * A: Field accessor (or the identity function), result must match the diff function D's first arg.
     * V: The vector to search (assuming sorted).
     * T: The scalar, enum or string key type, (either the element, or a field of the element).
     * K: The search key.
     * Kn: optional key length so external strings do not have to be zero terminated.
     * D: the diff function D(v, K, Kn) :: v - <K, Kn>
     *
     * returns index (0..len - 1), or not_found (-1).
     */
    fprintf(out->fp,
        "#include <string.h>\n"
        "static const size_t %snot_found = (size_t)-1;\n"
        "static const size_t %send = (size_t)-1;\n"
        "#define __%sidentity(n) (n)\n"
        "#define __%smin(a, b) ((a) < (b) ? (a) : (b))\n",
        nsc, nsc, nsc, nsc);
    fprintf(out->fp,
        "/* Subtraction doesn't work for unsigned types. */\n"
        "#define __%sscalar_cmp(x, y, n) ((x) < (y) ? -1 : (x) > (y))\n"
        "static inline int __%sstring_n_cmp(%sstring_t v, const char *s, size_t n)\n"
        "{ size_t nv = %sstring_len(v); int x = strncmp(v, s, nv < n ? nv : n);\n"
        "  return x != 0 ? x : nv < n ? -1 : nv > n; }\n"
        "/* `n` arg unused, but needed by string find macro expansion. */\n"
        "static inline int __%sstring_cmp(%sstring_t v, const char *s, size_t n) { (void)n; return strcmp(v, s); }\n",
        nsc, nsc, nsc, nsc, nsc, nsc);
    fprintf(out->fp,
        "/* A = identity if searching scalar vectors rather than key fields. */\n"
        "/* Returns lowest matching index or not_found. */\n"
        "#define __%sfind_by_field(A, V, E, L, K, Kn, T, D)\\\n"
        "{ T v__tmp; size_t a__tmp = 0, b__tmp, m__tmp; if (!(b__tmp = L(V))) { return %snot_found; }\\\n"
        "  --b__tmp;\\\n"
        "  while (a__tmp < b__tmp) {\\\n"
        "    m__tmp = a__tmp + ((b__tmp - a__tmp) >> 1);\\\n"
        "    v__tmp = A(E(V, m__tmp));\\\n"
        "    if ((D(v__tmp, (K), (Kn))) < 0) {\\\n"
        "      a__tmp = m__tmp + 1;\\\n"
        "    } else {\\\n"
        "      b__tmp = m__tmp;\\\n"
        "    }\\\n"
        "  }\\\n"
        "  if (a__tmp == b__tmp) {\\\n"
        "    v__tmp = A(E(V, a__tmp));\\\n"
        "    if (D(v__tmp, (K), (Kn)) == 0) {\\\n"
        "       return a__tmp;\\\n"
        "    }\\\n"
        "  }\\\n"
        "  return %snot_found;\\\n"
        "}\n",
        nsc, nsc, nsc);
    fprintf(out->fp,
        "#define __%sfind_by_scalar_field(A, V, E, L, K, T)\\\n"
        "__%sfind_by_field(A, V, E, L, K, 0, T, __%sscalar_cmp)\n"
        "#define __%sfind_by_string_field(A, V, E, L, K)\\\n"
        "__%sfind_by_field(A, V, E, L, K, 0, %sstring_t, __%sstring_cmp)\n"
        "#define __%sfind_by_string_n_field(A, V, E, L, K, Kn)\\\n"
        "__%sfind_by_field(A, V, E, L, K, Kn, %sstring_t, __%sstring_n_cmp)\n",
        nsc, nsc, nsc, nsc, nsc,
        nsc, nsc, nsc, nsc, nsc, nsc);
    fprintf(out->fp,
        "#define __%sdefine_find_by_scalar_field(N, NK, TK)\\\n"
        "static inline size_t N ## _vec_find_by_ ## NK(N ## _vec_t vec__tmp, TK key__tmp)\\\n"
        "__%sfind_by_scalar_field(N ## _ ## NK, vec__tmp, N ## _vec_at, N ## _vec_len, key__tmp, TK)\n",
        nsc, nsc);
    fprintf(out->fp,
        "#define __%sdefine_scalar_find(N, T)\\\n"
        "static inline size_t N ## _vec_find(N ## _vec_t vec__tmp, T key__tmp)\\\n"
        "__%sfind_by_scalar_field(__%sidentity, vec__tmp, N ## _vec_at, N ## _vec_len, key__tmp, T)\n",
        nsc, nsc, nsc);
    fprintf(out->fp,
        "#define __%sdefine_find_by_string_field(N, NK) \\\n"
        "/* Note: find only works on vectors sorted by this field. */\\\n"
        "static inline size_t N ## _vec_find_by_ ## NK(N ## _vec_t vec__tmp, const char *s__tmp)\\\n"
        "__%sfind_by_string_field(N ## _ ## NK, vec__tmp, N ## _vec_at, N ## _vec_len, s__tmp)\\\n"
        "static inline size_t N ## _vec_find_n_by_ ## NK(N ## _vec_t vec__tmp, const char *s__tmp, size_t n__tmp)\\\n"
        "__%sfind_by_string_n_field(N ## _ ## NK, vec__tmp, N ## _vec_at, N ## _vec_len, s__tmp, n__tmp)\n",
        nsc, nsc, nsc);
    fprintf(out->fp,
        "#define __%sdefine_default_find_by_scalar_field(N, NK, TK)\\\n"
        "static inline size_t N ## _vec_find(N ## _vec_t vec__tmp, TK key__tmp)\\\n"
        "{ return N ## _vec_find_by_ ## NK(vec__tmp, key__tmp); }\n",
        nsc);
    fprintf(out->fp,
        "#define __%sdefine_default_find_by_string_field(N, NK) \\\n"
        "static inline size_t N ## _vec_find(N ## _vec_t vec__tmp, const char *s__tmp)\\\n"
        "{ return N ## _vec_find_by_ ## NK(vec__tmp, s__tmp); }\\\n"
        "static inline size_t N ## _vec_find_n(N ## _vec_t vec__tmp, const char *s__tmp, size_t n__tmp)\\\n"
        "{ return N ## _vec_find_n_by_ ## NK(vec__tmp, s__tmp, n__tmp); }\n",
        nsc);
}

static void gen_union(fb_output_t *out)
{
    const char *nsc = out->nsc;

    fprintf(out->fp,
        "typedef struct %sunion {\n"
        "    %sunion_type_t type;\n"
        "    %sgeneric_t value;\n"
        "} %sunion_t;\n"
        "typedef struct %sunion_vec {\n"
        "    const %sunion_type_t *type;\n"
        "    const %suoffset_t *value;\n"
        "} %sunion_vec_t;\n",
        nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc);
    fprintf(out->fp,
        "typedef struct %smutable_union {\n"
        "    %sunion_type_t type;\n"
        "    %smutable_generic_t value;\n"
        "} %smutable_union_t;\n"
        "typedef struct %smutable_union_vec {\n"
        "    %sunion_type_t *type;\n"
        "    %suoffset_t *value;\n"
        "} %smutable_union_vec_t;\n",
        nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc);
    fprintf(out->fp,
        "static inline %smutable_union_t %smutable_union_cast(%sunion_t u__tmp)\\\n"
        "{ %smutable_union_t mu = { u__tmp.type, (%smutable_generic_t)u__tmp.value };\\\n"
        "  return mu; }\n",
        nsc, nsc, nsc, nsc, nsc);
    fprintf(out->fp,
        "static inline %smutable_union_vec_t %smutable_union_vec_cast(%sunion_vec_t uv__tmp)\\\n"
        "{ %smutable_union_vec_t muv =\\\n"
        "  { (%sunion_type_t *)uv__tmp.type, (%suoffset_t *)uv__tmp.value }; return muv; }\n",
        nsc, nsc, nsc, nsc, nsc, nsc);
    fprintf(out->fp,
        "#define __%sunion_type_field(ID, t)\\\n"
        "{\\\n"
        "    __%sread_vt(ID, offset__tmp, t)\\\n"
        "    return offset__tmp ? __%sread_scalar_at_byteoffset(__%sutype, t, offset__tmp) : 0;\\\n"
        "}\n",
        nsc, nsc, nsc, nsc);
    fprintf(out->fp,
        "static inline %sstring_t %sstring_cast_from_union(const %sunion_t u__tmp)\\\n"
        "{ return %sstring_cast_from_generic(u__tmp.value); }\n",
        nsc, nsc, nsc, nsc);
    fprintf(out->fp,
        "#define __%sdefine_union_field(NS, ID, N, NK, T, r)\\\n"
        "static inline T ## _union_type_t N ## _ ## NK ## _type_get(N ## _table_t t__tmp)\\\n"
        "__## NS ## union_type_field(((ID) - 1), t__tmp)\\\n"
        "static inline NS ## generic_t N ## _ ## NK ## _get(N ## _table_t t__tmp)\\\n"
        "__## NS ## table_field(NS ## generic_t, ID, t__tmp, r)\\\n", nsc);
    if (!out->opts->cgen_no_conflicts) {
        fprintf(out->fp,
            "static inline T ## _union_type_t N ## _ ## NK ## _type(N ## _table_t t__tmp)\\\n"
            "__## NS ## union_type_field(((ID) - 1), t__tmp)\\\n"
            "static inline NS ## generic_t N ## _ ## NK(N ## _table_t t__tmp)\\\n"
            "__## NS ## table_field(NS ## generic_t, ID, t__tmp, r)\\\n");
    }
    fprintf(out->fp,
        "static inline int N ## _ ## NK ## _is_present(N ## _table_t t__tmp)\\\n"
        "__## NS ## field_present(ID, t__tmp)\\\n"
        "static inline T ## _union_t N ## _ ## NK ## _union(N ## _table_t t__tmp)\\\n"
        "{ T ## _union_t u__tmp = { 0, 0 }; u__tmp.type = N ## _ ## NK ## _type_get(t__tmp);\\\n"
        "  if (u__tmp.type == 0) { return u__tmp; } u__tmp.value = N ## _ ## NK ## _get(t__tmp); return u__tmp; }\\\n"
        "static inline NS ## string_t N ## _ ## NK ## _as_string(N ## _table_t t__tmp)\\\n"
        "{ return NS ## string_cast_from_generic(N ## _ ## NK ## _get(t__tmp)); }\\\n"
        "\n");
    fprintf(out->fp,
        "#define __%sdefine_union_vector_ops(NS, T)\\\n"
        "static inline size_t T ## _union_vec_len(T ## _union_vec_t uv__tmp)\\\n"
        "{ return NS ## vec_len(uv__tmp.type); }\\\n"
        "static inline T ## _union_t T ## _union_vec_at(T ## _union_vec_t uv__tmp, size_t i__tmp)\\\n"
        "{ T ## _union_t u__tmp = { 0, 0 }; size_t n__tmp = NS ## vec_len(uv__tmp.type);\\\n"
        "  FLATCC_ASSERT(n__tmp > (i__tmp) && \"index out of range\"); u__tmp.type = uv__tmp.type[i__tmp];\\\n"
        "  /* Unknown type is treated as NONE for schema evolution. */\\\n"
        "  if (u__tmp.type == 0) return u__tmp;\\\n"
        "  u__tmp.value = NS ## generic_vec_at(uv__tmp.value, i__tmp); return u__tmp; }\\\n"
        "static inline NS ## string_t T ## _union_vec_at_as_string(T ## _union_vec_t uv__tmp, size_t i__tmp)\\\n"
        "{ return (NS ## string_t) NS ## generic_vec_at_as_string(uv__tmp.value, i__tmp); }\\\n"
        "\n",
        nsc);
    fprintf(out->fp,
        "#define __%sdefine_union_vector(NS, T)\\\n"
        "typedef NS ## union_vec_t T ## _union_vec_t;\\\n"
        "typedef NS ## mutable_union_vec_t T ## _mutable_union_vec_t;\\\n"
        "static inline T ## _mutable_union_vec_t T ## _mutable_union_vec_cast(T ## _union_vec_t u__tmp)\\\n"
        "{ return NS ## mutable_union_vec_cast(u__tmp); }\\\n"
        "__## NS ## define_union_vector_ops(NS, T)\n",
        nsc);
    fprintf(out->fp,
        "#define __%sdefine_union(NS, T)\\\n"
        "typedef NS ## union_t T ## _union_t;\\\n"
        "typedef NS ## mutable_union_t T ## _mutable_union_t;\\\n"
        "static inline T ## _mutable_union_t T ## _mutable_union_cast(T ## _union_t u__tmp)\\\n"
        "{ return NS ## mutable_union_cast(u__tmp); }\\\n"
        "__## NS ## define_union_vector(NS, T)\n",
        nsc);
    fprintf(out->fp,
        "#define __%sdefine_union_vector_field(NS, ID, N, NK, T, r)\\\n"
        "__## NS ## define_vector_field(ID - 1, N, NK ## _type, T ## _vec_t, r)\\\n"
        "__## NS ## define_vector_field(ID, N, NK, flatbuffers_generic_vec_t, r)\\\n"
        "static inline T ## _union_vec_t N ## _ ## NK ## _union(N ## _table_t t__tmp)\\\n"
        "{ T ## _union_vec_t uv__tmp; uv__tmp.type = N ## _ ## NK ## _type_get(t__tmp);\\\n"
        "  uv__tmp.value = N ## _ ## NK(t__tmp);\\\n"
        "  FLATCC_ASSERT(NS ## vec_len(uv__tmp.type) == NS ## vec_len(uv__tmp.value)\\\n"
        "  && \"union vector type length mismatch\"); return uv__tmp; }\n",
        nsc);
}

/* Linearly finds first occurrence of matching key, doesn't require vector to be sorted. */
static void gen_scan(fb_output_t *out)
{
    const char *nsc = out->nsc;

    /*
     * E: Element accessor (elem = E(vector, index)).
     * L: Length accessor (length = L(vector)).
     * A: Field accessor (or the identity function), result must match the diff function D's first arg.
     * V: The vector to search (assuming sorted).
     * T: The scalar, enum or string key type, (either the element, or a field of the element).
     * K: The search key.
     * Kn: optional key length so external strings do not have to be zero terminated.
     * D: the diff function D(v, K, Kn) :: v - <K, Kn>
     *
     * returns index (0..len - 1), or not_found (-1).
     */
    fprintf(out->fp,
        "/* A = identity if searching scalar vectors rather than key fields. */\n"
        "/* Returns lowest matching index or not_found. */\n"
        "#define __%sscan_by_field(b, e, A, V, E, L, K, Kn, T, D)\\\n"
        "{ T v__tmp; size_t i__tmp;\\\n"
        "  for (i__tmp = b; i__tmp < e; ++i__tmp) {\\\n"
        "    v__tmp = A(E(V, i__tmp));\\\n"
        "    if (D(v__tmp, (K), (Kn)) == 0) {\\\n"
        "       return i__tmp;\\\n"
        "    }\\\n"
        "  }\\\n"
        "  return %snot_found;\\\n"
        "}\n",
        nsc, nsc);
    fprintf(out->fp,
        "#define __%srscan_by_field(b, e, A, V, E, L, K, Kn, T, D)\\\n"
        "{ T v__tmp; size_t i__tmp = e;\\\n"
        "  while (i__tmp-- > b) {\\\n"
        "    v__tmp = A(E(V, i__tmp));\\\n"
        "    if (D(v__tmp, (K), (Kn)) == 0) {\\\n"
        "       return i__tmp;\\\n"
        "    }\\\n"
        "  }\\\n"
        "  return %snot_found;\\\n"
        "}\n",
        nsc, nsc);
    fprintf(out->fp,
        "#define __%sscan_by_scalar_field(b, e, A, V, E, L, K, T)\\\n"
        "__%sscan_by_field(b, e, A, V, E, L, K, 0, T, __%sscalar_cmp)\n"
        "#define __%sscan_by_string_field(b, e, A, V, E, L, K)\\\n"
        "__%sscan_by_field(b, e, A, V, E, L, K, 0, %sstring_t, __%sstring_cmp)\n"
        "#define __%sscan_by_string_n_field(b, e, A, V, E, L, K, Kn)\\\n"
        "__%sscan_by_field(b, e, A, V, E, L, K, Kn, %sstring_t, __%sstring_n_cmp)\n",
        nsc, nsc, nsc, nsc, nsc,
        nsc, nsc, nsc, nsc, nsc, nsc);
    fprintf(out->fp,
        "#define __%srscan_by_scalar_field(b, e, A, V, E, L, K, T)\\\n"
        "__%srscan_by_field(b, e, A, V, E, L, K, 0, T, __%sscalar_cmp)\n"
        "#define __%srscan_by_string_field(b, e, A, V, E, L, K)\\\n"
        "__%srscan_by_field(b, e, A, V, E, L, K, 0, %sstring_t, __%sstring_cmp)\n"
        "#define __%srscan_by_string_n_field(b, e, A, V, E, L, K, Kn)\\\n"
        "__%srscan_by_field(b, e, A, V, E, L, K, Kn, %sstring_t, __%sstring_n_cmp)\n",
        nsc, nsc, nsc, nsc, nsc,
        nsc, nsc, nsc, nsc, nsc, nsc);
    fprintf(out->fp,
        "#define __%sdefine_scan_by_scalar_field(N, NK, T)\\\n"
        "static inline size_t N ## _vec_scan_by_ ## NK(N ## _vec_t vec__tmp, T key__tmp)\\\n"
        "__%sscan_by_scalar_field(0, N ## _vec_len(vec__tmp), N ## _ ## NK ## _get, vec__tmp, N ## _vec_at, N ## _vec_len, key__tmp, T)\\\n"
        "static inline size_t N ## _vec_scan_ex_by_ ## NK(N ## _vec_t vec__tmp, size_t begin__tmp, size_t end__tmp, T key__tmp)\\\n"
        "__%sscan_by_scalar_field(begin__tmp, __%smin(end__tmp, N ## _vec_len(vec__tmp)), N ## _ ## NK ## _get, vec__tmp, N ## _vec_at, N ## _vec_len, key__tmp, T)\\\n"
        "static inline size_t N ## _vec_rscan_by_ ## NK(N ## _vec_t vec__tmp, T key__tmp)\\\n"
        "__%srscan_by_scalar_field(0, N ## _vec_len(vec__tmp), N ## _ ## NK ## _get, vec__tmp, N ## _vec_at, N ## _vec_len, key__tmp, T)\\\n"
        "static inline size_t N ## _vec_rscan_ex_by_ ## NK(N ## _vec_t vec__tmp, size_t begin__tmp, size_t end__tmp, T key__tmp)\\\n"
        "__%srscan_by_scalar_field(begin__tmp, __%smin(end__tmp, N ## _vec_len(vec__tmp)), N ## _ ## NK ## _get, vec__tmp, N ## _vec_at, N ## _vec_len, key__tmp, T)\n",
        nsc, nsc, nsc, nsc, nsc, nsc, nsc);
    fprintf(out->fp,
        "#define __%sdefine_scalar_scan(N, T)\\\n"
        "static inline size_t N ## _vec_scan(N ## _vec_t vec__tmp, T key__tmp)\\\n"
        "__%sscan_by_scalar_field(0, N ## _vec_len(vec__tmp), __%sidentity, vec__tmp, N ## _vec_at, N ## _vec_len, key__tmp, T)\\\n"
        "static inline size_t N ## _vec_scan_ex(N ## _vec_t vec__tmp, size_t begin__tmp, size_t end__tmp, T key__tmp)\\\n"
        "__%sscan_by_scalar_field(begin__tmp, __%smin(end__tmp, N ## _vec_len(vec__tmp)), __%sidentity, vec__tmp, N ## _vec_at, N ## _vec_len, key__tmp, T)\\\n"
        "static inline size_t N ## _vec_rscan(N ## _vec_t vec__tmp, T key__tmp)\\\n"
        "__%srscan_by_scalar_field(0, N ## _vec_len(vec__tmp), __%sidentity, vec__tmp, N ## _vec_at, N ## _vec_len, key__tmp, T)\\\n"
        "static inline size_t N ## _vec_rscan_ex(N ## _vec_t vec__tmp, size_t begin__tmp, size_t end__tmp, T key__tmp)\\\n"
        "__%srscan_by_scalar_field(begin__tmp, __%smin(end__tmp, N ## _vec_len(vec__tmp)), __%sidentity, vec__tmp, N ## _vec_at, N ## _vec_len, key__tmp, T)\n",
        nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc);
    fprintf(out->fp,
        "#define __%sdefine_scan_by_string_field(N, NK) \\\n"
        "static inline size_t N ## _vec_scan_by_ ## NK(N ## _vec_t vec__tmp, const char *s__tmp)\\\n"
        "__%sscan_by_string_field(0, N ## _vec_len(vec__tmp), N ## _ ## NK ## _get, vec__tmp, N ## _vec_at, N ## _vec_len, s__tmp)\\\n"
        "static inline size_t N ## _vec_scan_n_by_ ## NK(N ## _vec_t vec__tmp, const char *s__tmp, size_t n__tmp)\\\n"
        "__%sscan_by_string_n_field(0, N ## _vec_len(vec__tmp), N ## _ ## NK ## _get, vec__tmp, N ## _vec_at, N ## _vec_len, s__tmp, n__tmp)\\\n"
        "static inline size_t N ## _vec_scan_ex_by_ ## NK(N ## _vec_t vec__tmp, size_t begin__tmp, size_t end__tmp, const char *s__tmp)\\\n"
        "__%sscan_by_string_field(begin__tmp, __%smin(end__tmp, N ## _vec_len(vec__tmp)), N ## _ ## NK ## _get, vec__tmp, N ## _vec_at, N ## _vec_len, s__tmp)\\\n"
        "static inline size_t N ## _vec_scan_ex_n_by_ ## NK(N ## _vec_t vec__tmp, size_t begin__tmp, size_t end__tmp, const char *s__tmp, size_t n__tmp)\\\n"
        "__%sscan_by_string_n_field(begin__tmp, __%smin( end__tmp, N ## _vec_len(vec__tmp)), N ## _ ## NK ## _get, vec__tmp, N ## _vec_at, N ## _vec_len, s__tmp, n__tmp)\\\n"
        "static inline size_t N ## _vec_rscan_by_ ## NK(N ## _vec_t vec__tmp, const char *s__tmp)\\\n"
        "__%srscan_by_string_field(0, N ## _vec_len(vec__tmp), N ## _ ## NK ## _get, vec__tmp, N ## _vec_at, N ## _vec_len, s__tmp)\\\n"
        "static inline size_t N ## _vec_rscan_n_by_ ## NK(N ## _vec_t vec__tmp, const char *s__tmp, size_t n__tmp)\\\n"
        "__%srscan_by_string_n_field(0, N ## _vec_len(vec__tmp), N ## _ ## NK ## _get, vec__tmp, N ## _vec_at, N ## _vec_len, s__tmp, n__tmp)\\\n"
        "static inline size_t N ## _vec_rscan_ex_by_ ## NK(N ## _vec_t vec__tmp, size_t begin__tmp, size_t end__tmp, const char *s__tmp)\\\n"
        "__%srscan_by_string_field(begin__tmp, __%smin(end__tmp, N ## _vec_len(vec__tmp)), N ## _ ## NK ## _get, vec__tmp, N ## _vec_at, N ## _vec_len, s__tmp)\\\n"
        "static inline size_t N ## _vec_rscan_ex_n_by_ ## NK(N ## _vec_t vec__tmp, size_t begin__tmp, size_t end__tmp, const char *s__tmp, size_t n__tmp)\\\n"
        "__%srscan_by_string_n_field(begin__tmp, __%smin( end__tmp, N ## _vec_len(vec__tmp)), N ## _ ## NK ## _get, vec__tmp, N ## _vec_at, N ## _vec_len, s__tmp, n__tmp)\n",
        nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc);
    fprintf(out->fp,
        "#define __%sdefine_default_scan_by_scalar_field(N, NK, TK)\\\n"
        "static inline size_t N ## _vec_scan(N ## _vec_t vec__tmp, TK key__tmp)\\\n"
        "{ return N ## _vec_scan_by_ ## NK(vec__tmp, key__tmp); }\\\n"
        "static inline size_t N ## _vec_scan_ex(N ## _vec_t vec__tmp, size_t begin__tmp, size_t end__tmp, TK key__tmp)\\\n"
        "{ return N ## _vec_scan_ex_by_ ## NK(vec__tmp, begin__tmp, end__tmp, key__tmp); }\\\n"
        "static inline size_t N ## _vec_rscan(N ## _vec_t vec__tmp, TK key__tmp)\\\n"
        "{ return N ## _vec_rscan_by_ ## NK(vec__tmp, key__tmp); }\\\n"
        "static inline size_t N ## _vec_rscan_ex(N ## _vec_t vec__tmp, size_t begin__tmp, size_t end__tmp, TK key__tmp)\\\n"
        "{ return N ## _vec_rscan_ex_by_ ## NK(vec__tmp, begin__tmp, end__tmp, key__tmp); }\n",
        nsc);
    fprintf(out->fp,
        "#define __%sdefine_default_scan_by_string_field(N, NK) \\\n"
        "static inline size_t N ## _vec_scan(N ## _vec_t vec__tmp, const char *s__tmp)\\\n"
        "{ return N ## _vec_scan_by_ ## NK(vec__tmp, s__tmp); }\\\n"
        "static inline size_t N ## _vec_scan_n(N ## _vec_t vec__tmp, const char *s__tmp, size_t n__tmp)\\\n"
        "{ return N ## _vec_scan_n_by_ ## NK(vec__tmp, s__tmp, n__tmp); }\\\n"
        "static inline size_t N ## _vec_scan_ex(N ## _vec_t vec__tmp, size_t begin__tmp, size_t end__tmp, const char *s__tmp)\\\n"
        "{ return N ## _vec_scan_ex_by_ ## NK(vec__tmp, begin__tmp, end__tmp, s__tmp); }\\\n"
        "static inline size_t N ## _vec_scan_ex_n(N ## _vec_t vec__tmp, size_t begin__tmp, size_t end__tmp, const char *s__tmp, size_t n__tmp)\\\n"
        "{ return N ## _vec_scan_ex_n_by_ ## NK(vec__tmp, begin__tmp, end__tmp, s__tmp, n__tmp); }\\\n"
        "static inline size_t N ## _vec_rscan(N ## _vec_t vec__tmp, const char *s__tmp)\\\n"
        "{ return N ## _vec_rscan_by_ ## NK(vec__tmp, s__tmp); }\\\n"
        "static inline size_t N ## _vec_rscan_n(N ## _vec_t vec__tmp, const char *s__tmp, size_t n__tmp)\\\n"
        "{ return N ## _vec_rscan_n_by_ ## NK(vec__tmp, s__tmp, n__tmp); }\\\n"
        "static inline size_t N ## _vec_rscan_ex(N ## _vec_t vec__tmp, size_t begin__tmp, size_t end__tmp, const char *s__tmp)\\\n"
        "{ return N ## _vec_rscan_ex_by_ ## NK(vec__tmp, begin__tmp, end__tmp, s__tmp); }\\\n"
        "static inline size_t N ## _vec_rscan_ex_n(N ## _vec_t vec__tmp, size_t begin__tmp, size_t end__tmp, const char *s__tmp, size_t n__tmp)\\\n"
        "{ return N ## _vec_rscan_ex_n_by_ ## NK(vec__tmp, begin__tmp, end__tmp, s__tmp, n__tmp); }\n",
        nsc);
}

static void gen_helpers(fb_output_t *out)
{
    const char *nsc = out->nsc;

    fprintf(out->fp,
        /*
         * Include the basic primitives for accessing flatbuffer data types independent
         * of endianness.
         *
         * The included file must define the basic types and accessors
         * prefixed with the common namespace which by default is
         * "flatbuffers_".
         */
        "#include \"flatcc/flatcc_flatbuffers.h\"\n"
        "\n\n");
    /*
     * The remapping of basic types to the common namespace makes it
     * possible to have different definitions. The generic
     * `flatbuffers_uoffset_t` etc. cannot be trusted to have one specific
     * size since it depends on the included `flatcc/flatcc_types.h`
     * filer, but the namespace prefixed types can be trusted if used carefully.
     * For example the common namespace could be `flatbuffers_large_`
     * when allowing for 64 bit offsets.
     */
    if (strcmp(nsc, "flatbuffers_")) {
        fprintf(out->fp,
                "typedef flatbuffers_uoffset_t %suoffset_t;\n"
                "typedef flatbuffers_soffset_t %ssoffset_t;\n"
                "typedef flatbuffers_voffset_t %svoffset_t;\n"
                "typedef flatbuffers_utype_t %sutype_t;\n"
                "typedef flatbuffers_bool_t %sbool_t;\n"
                "\n",
                nsc, nsc, nsc, nsc, nsc);
        fprintf(out->fp,
                "#define %sendian flatbuffers_endian\n"
                "__flatcc_define_basic_scalar_accessors(%s, flatbuffers_endian)"
                "__flatcc_define_integer_accessors(%sbool, flatbuffers_bool_t,\\\n"
                "        FLATBUFFERS_BOOL_WIDTH, flatbuffers_endian)\\\n"
                "__flatcc_define_integer_accessors(%sunion_type, flatbuffers_union_type_t,\n"
                "        FLATBUFFERS_UTYPE_WIDTH, flatbuffers_endian)\\\n",
                "\n",
                nsc, nsc, nsc);
        fprintf(out->fp,
                "__flatcc_define_integer_accessors(__%suoffset, flatbuffers_uoffset_t,\n"
                "        FLATBUFFERS_UOFFSET_WIDTH, flatbuffers_endian)\n"
                "__flatcc_define_integer_accessors(__%ssoffset, flatbuffers_soffset_t,\n"
                "        FLATBUFFERS_SOFFSET_WIDTH, flatbuffers_endian)\n"
                "__flatcc_define_integer_accessors(__%svoffset, flatbuffers_voffset_t,\n"
                "        FLATBUFFERS_VOFFSET_WIDTH, flatbuffers_endian)\n"
                "__flatcc_define_integer_accessors(__%sutype, flatbuffers_utype_t,\n"
                "        FLATBUFFERS_UTYPE_WIDTH, flatbuffers_endian)\n"
                "__flatcc_define_integer_accessors(__%sthash, flatbuffers_thash_t,\n"
                "        FLATBUFFERS_THASH_WIDTH, flatbuffers_endian)\n",
                nsc, nsc, nsc, nsc, nsc);
        fprintf(out->fp,
                "#ifndef %s_WRAP_NAMESPACE\n"
                "#define %s_WRAP_NAMESPACE(ns, x) ns ## _ ## x\n"
                "#endif\n",
                out->nscup, out->nscup);
    }
    /* Build out a more elaborate type system based in the primitives included. */
    fprintf(out->fp,
        "#define __%sread_scalar_at_byteoffset(N, p, o) N ## _read_from_pe((uint8_t *)(p) + (o))\n"
        "#define __%sread_scalar(N, p) N ## _read_from_pe(p)\n",
        nsc, nsc);
    fprintf(out->fp,
        "#define __%sread_vt(ID, offset, t)\\\n"
        "%svoffset_t offset = 0;\\\n"
        "{   %svoffset_t id__tmp, *vt__tmp;\\\n"
        "    FLATCC_ASSERT(t != 0 && \"null pointer table access\");\\\n"
        "    id__tmp = ID;\\\n"
        "    vt__tmp = (%svoffset_t *)((uint8_t *)(t) -\\\n"
        "        __%ssoffset_read_from_pe(t));\\\n"
        "    if (__%svoffset_read_from_pe(vt__tmp) >= sizeof(vt__tmp[0]) * (id__tmp + 3u)) {\\\n"
        "        offset = __%svoffset_read_from_pe(vt__tmp + id__tmp + 2);\\\n"
        "    }\\\n"
        "}\n",
        nsc, nsc, nsc, nsc, nsc, nsc, nsc);
    fprintf(out->fp,
            "#define __%sfield_present(ID, t) { __%sread_vt(ID, offset__tmp, t) return offset__tmp != 0; }\n",
            nsc, nsc);
    fprintf(out->fp,
        "#define __%sscalar_field(T, ID, t)\\\n"
        "{\\\n"
        "    __%sread_vt(ID, offset__tmp, t)\\\n"
        "    if (offset__tmp) {\\\n"
        "        return (const T *)((uint8_t *)(t) + offset__tmp);\\\n"
        "    }\\\n"
        "    return 0;\\\n"
        "}\n",
        nsc, nsc);
    fprintf(out->fp,
        "#define __%sdefine_scalar_field(ID, N, NK, TK, T, V)\\\n"
        "static inline T N ## _ ## NK ## _get(N ## _table_t t__tmp)\\\n"
        "{ __%sread_vt(ID, offset__tmp, t__tmp)\\\n"
        "  return offset__tmp ? __%sread_scalar_at_byteoffset(TK, t__tmp, offset__tmp) : V;\\\n"
        "}\\\n", nsc, nsc, nsc);
    if (!out->opts->cgen_no_conflicts) {
        fprintf(out->fp,
            "static inline T N ## _ ## NK(N ## _table_t t__tmp)\\\n"
            "{ __%sread_vt(ID, offset__tmp, t__tmp)\\\n"
            "  return offset__tmp ? __%sread_scalar_at_byteoffset(TK, t__tmp, offset__tmp) : V;\\\n"
            "}\\\n", nsc, nsc);
    }
    fprintf(out->fp,
        "static inline const T *N ## _ ## NK ## _get_ptr(N ## _table_t t__tmp)\\\n"
        "__%sscalar_field(T, ID, t__tmp)\\\n", nsc);
    fprintf(out->fp,
        "static inline int N ## _ ## NK ## _is_present(N ## _table_t t__tmp)\\\n"
        "__%sfield_present(ID, t__tmp)",nsc);
    if (out->opts->allow_scan_for_all_fields) {
        fprintf(out->fp, "\\\n__%sdefine_scan_by_scalar_field(N, NK, T)\n", nsc);
    } else {
        fprintf(out->fp, "\n");
    }
    fprintf(out->fp,
        "#define __%sdefine_scalar_optional_field(ID, N, NK, TK, T, V)\\\n"
        "__%sdefine_scalar_field(ID, N, NK, TK, T, V)\\\n"
        "static inline TK ## _option_t N ## _ ## NK ## _option(N ## _table_t t__tmp)\\\n"
        "{ TK ## _option_t ret; __%sread_vt(ID, offset__tmp, t__tmp)\\\n"
        "  ret.is_null = offset__tmp == 0; ret.value = offset__tmp ?\\\n"
        "  __%sread_scalar_at_byteoffset(TK, t__tmp, offset__tmp) : V;\\\n"
        "  return ret; }\n", nsc, nsc, nsc, nsc);
    fprintf(out->fp,
        "#define __%sstruct_field(T, ID, t, r)\\\n"
        "{\\\n"
        "    __%sread_vt(ID, offset__tmp, t)\\\n"
        "    if (offset__tmp) {\\\n"
        "        return (T)((uint8_t *)(t) + offset__tmp);\\\n"
        "    }\\\n"
        "    FLATCC_ASSERT(!(r) && \"required field missing\");\\\n"
        "    return 0;\\\n"
        "}\n",
        nsc, nsc);
    fprintf(out->fp,
        "#define __%soffset_field(T, ID, t, r, adjust)\\\n"
        "{\\\n"
        "    %suoffset_t *elem__tmp;\\\n"
        "    __%sread_vt(ID, offset__tmp, t)\\\n"
        "    if (offset__tmp) {\\\n"
        "        elem__tmp = (%suoffset_t *)((uint8_t *)(t) + offset__tmp);\\\n"
        "        /* Add sizeof so C api can have raw access past header field. */\\\n"
        "        return (T)((uint8_t *)(elem__tmp) + adjust +\\\n"
        "              __%suoffset_read_from_pe(elem__tmp));\\\n"
        "    }\\\n"
        "    FLATCC_ASSERT(!(r) && \"required field missing\");\\\n"
        "    return 0;\\\n"
        "}\n",
        nsc, nsc, nsc, nsc, nsc);
    fprintf(out->fp,
        "#define __%svector_field(T, ID, t, r) __%soffset_field(T, ID, t, r, sizeof(%suoffset_t))\n"
        "#define __%stable_field(T, ID, t, r) __%soffset_field(T, ID, t, r, 0)\n",
        nsc, nsc, nsc, nsc, nsc);
    fprintf(out->fp,
        "#define __%sdefine_struct_field(ID, N, NK, T, r)\\\n"
        "static inline T N ## _ ## NK ## _get(N ## _table_t t__tmp)\\\n"
        "__%sstruct_field(T, ID, t__tmp, r)", nsc, nsc);
    if (!out->opts->cgen_no_conflicts) {
        fprintf(out->fp,
            "\\\nstatic inline T N ## _ ## NK(N ## _table_t t__tmp)\\\n"
            "__%sstruct_field(T, ID, t__tmp, r)", nsc);
    }
    fprintf(out->fp,
        "\\\nstatic inline int N ## _ ## NK ## _is_present(N ## _table_t t__tmp)\\\n"
        "__%sfield_present(ID, t__tmp)\n", nsc);
    fprintf(out->fp,
        "#define __%sdefine_vector_field(ID, N, NK, T, r)\\\n"
        "static inline T N ## _ ## NK ## _get(N ## _table_t t__tmp)\\\n"
        "__%svector_field(T, ID, t__tmp, r)", nsc, nsc);
    if (!out->opts->cgen_no_conflicts) {
        fprintf(out->fp,
            "\\\nstatic inline T N ## _ ## NK(N ## _table_t t__tmp)\\\n"
            "__%svector_field(T, ID, t__tmp, r)", nsc);
    }
    fprintf(out->fp,
        "\\\nstatic inline int N ## _ ## NK ## _is_present(N ## _table_t t__tmp)\\\n"
        "__%sfield_present(ID, t__tmp)\n", nsc);
    fprintf(out->fp,
        "#define __%sdefine_table_field(ID, N, NK, T, r)\\\n"
        "static inline T N ## _ ## NK ## _get(N ## _table_t t__tmp)\\\n"
        "__%stable_field(T, ID, t__tmp, r)", nsc, nsc);
    if (!out->opts->cgen_no_conflicts) {
        fprintf(out->fp,
            "\\\nstatic inline T N ## _ ## NK(N ## _table_t t__tmp)\\\n"
            "__%stable_field(T, ID, t__tmp, r)", nsc);
    }
    fprintf(out->fp,
        "\\\nstatic inline int N ## _ ## NK ## _is_present(N ## _table_t t__tmp)\\\n"
        "__%sfield_present(ID, t__tmp)\n", nsc);
    fprintf(out->fp,
        "#define __%sdefine_string_field(ID, N, NK, r)\\\n"
        "static inline %sstring_t N ## _ ## NK ## _get(N ## _table_t t__tmp)\\\n"
        "__%svector_field(%sstring_t, ID, t__tmp, r)", nsc, nsc, nsc, nsc);
    if (!out->opts->cgen_no_conflicts) {
        fprintf(out->fp,
        "\\\nstatic inline %sstring_t N ## _ ## NK(N ## _table_t t__tmp)\\\n"
        "__%svector_field(%sstring_t, ID, t__tmp, r)", nsc, nsc, nsc);
    }
    fprintf(out->fp,
        "\\\nstatic inline int N ## _ ## NK ## _is_present(N ## _table_t t__tmp)\\\n"
        "__%sfield_present(ID, t__tmp)", nsc);
        if (out->opts->allow_scan_for_all_fields) {
            fprintf(out->fp, "\\\n__%sdefine_scan_by_string_field(N, NK)\n", nsc);
        } else {
            fprintf(out->fp, "\n");
        }
    fprintf(out->fp,
        "#define __%svec_len(vec)\\\n"
        "{ return (vec) ? (size_t)__%suoffset_read_from_pe((flatbuffers_uoffset_t *)vec - 1) : 0; }\n"
        "#define __%sstring_len(s) __%svec_len(s)\n",
        nsc, nsc, nsc, nsc);
    fprintf(out->fp,
        "static inline size_t %svec_len(const void *vec)\n"
        "__%svec_len(vec)\n",
        nsc, nsc);
    fprintf(out->fp,
        /* Tb is the base type for loads. */
        "#define __%sscalar_vec_at(N, vec, i)\\\n"
        "{ FLATCC_ASSERT(%svec_len(vec) > (i) && \"index out of range\");\\\n"
        "  return __%sread_scalar(N, &(vec)[i]); }\n",
        nsc, nsc, nsc);
    fprintf(out->fp,
        "#define __%sstruct_vec_at(vec, i)\\\n"
        "{ FLATCC_ASSERT(%svec_len(vec) > (i) && \"index out of range\"); return (vec) + (i); }\n",
        nsc, nsc);
    fprintf(out->fp,
        "/* `adjust` skips past the header for string vectors. */\n"
        "#define __%soffset_vec_at(T, vec, i, adjust)\\\n"
        "{ const %suoffset_t *elem__tmp = (vec) + (i);\\\n"
        "  FLATCC_ASSERT(%svec_len(vec) > (i) && \"index out of range\");\\\n"
        "  return (T)((uint8_t *)(elem__tmp) + (size_t)__%suoffset_read_from_pe(elem__tmp) + (adjust)); }\n",
        nsc, nsc, nsc, nsc);
    fprintf(out->fp,
            "#define __%sdefine_scalar_vec_len(N)\\\n"
            "static inline size_t N ## _vec_len(N ##_vec_t vec__tmp)\\\n"
            "{ return %svec_len(vec__tmp); }\n",
            nsc, nsc);
    fprintf(out->fp,
            "#define __%sdefine_scalar_vec_at(N, T) \\\n"
            "static inline T N ## _vec_at(N ## _vec_t vec__tmp, size_t i__tmp)\\\n"
            "__%sscalar_vec_at(N, vec__tmp, i__tmp)\n",
            nsc, nsc);
    fprintf(out->fp,
            "typedef const char *%sstring_t;\n"
            "static inline size_t %sstring_len(%sstring_t s)\n"
            "__%sstring_len(s)\n",
            nsc, nsc, nsc, nsc);
    fprintf(out->fp,
            "typedef const %suoffset_t *%sstring_vec_t;\n"
            "typedef %suoffset_t *%sstring_mutable_vec_t;\n"
            "static inline size_t %sstring_vec_len(%sstring_vec_t vec)\n"
            "__%svec_len(vec)\n"
            "static inline %sstring_t %sstring_vec_at(%sstring_vec_t vec, size_t i)\n"
            "__%soffset_vec_at(%sstring_t, vec, i, sizeof(vec[0]))\n",
            nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc);
    fprintf(out->fp, "typedef const void *%sgeneric_t;\n", nsc);
    fprintf(out->fp, "typedef void *%smutable_generic_t;\n", nsc);
    fprintf(out->fp,
        "static inline %sstring_t %sstring_cast_from_generic(const %sgeneric_t p)\n"
        "{ return p ? ((const char *)p) + __%suoffset__size() : 0; }\n",
        nsc, nsc, nsc, nsc);
    fprintf(out->fp,
            "typedef const %suoffset_t *%sgeneric_vec_t;\n"
            "typedef %suoffset_t *%sgeneric_table_mutable_vec_t;\n"
            "static inline size_t %sgeneric_vec_len(%sgeneric_vec_t vec)\n"
            "__%svec_len(vec)\n"
            "static inline %sgeneric_t %sgeneric_vec_at(%sgeneric_vec_t vec, size_t i)\n"
            "__%soffset_vec_at(%sgeneric_t, vec, i, 0)\n"
            "static inline %sgeneric_t %sgeneric_vec_at_as_string(%sgeneric_vec_t vec, size_t i)\n"
            "__%soffset_vec_at(%sgeneric_t, vec, i, sizeof(vec[0]))\n",
            nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc);
    gen_union(out);
    gen_find(out);
    gen_scan(out);
    if (out->opts->cgen_sort) {
        gen_sort(out);
        fprintf(out->fp,
            "#define __%ssort_vector_field(N, NK, T, t)\\\n"
            "{ T ## _mutable_vec_t v__tmp = (T ## _mutable_vec_t) N ## _ ## NK ## _get(t);\\\n"
            "  if (v__tmp) T ## _vec_sort(v__tmp); }\n",
            nsc);
        fprintf(out->fp,
            "#define __%ssort_table_field(N, NK, T, t)\\\n"
            "{ T ## _sort((T ## _mutable_table_t)N ## _ ## NK ## _get(t)); }\n",
            nsc);
        fprintf(out->fp,
            "#define __%ssort_union_field(N, NK, T, t)\\\n"
            "{ T ## _sort(T ## _mutable_union_cast(N ## _ ## NK ## _union(t))); }\n",
            nsc);
        fprintf(out->fp,
            "#define __%ssort_table_vector_field_elements(N, NK, T, t)\\\n"
            "{ T ## _vec_t v__tmp = N ## _ ## NK ## _get(t); size_t i__tmp, n__tmp;\\\n"
            "  n__tmp = T ## _vec_len(v__tmp); for (i__tmp = 0; i__tmp < n__tmp; ++i__tmp) {\\\n"
            "  T ## _sort((T ## _mutable_table_t)T ## _vec_at(v__tmp, i__tmp)); }}\n",
            nsc);
        fprintf(out->fp,
            "#define __%ssort_union_vector_field_elements(N, NK, T, t)\\\n"
            "{ T ## _union_vec_t v__tmp = N ## _ ## NK ## _union(t); size_t i__tmp, n__tmp;\\\n"
            "  n__tmp = T ## _union_vec_len(v__tmp); for (i__tmp = 0; i__tmp < n__tmp; ++i__tmp) {\\\n"
            "  T ## _sort(T ## _mutable_union_cast(T ## _union_vec_at(v__tmp, i__tmp))); }}\n",
            nsc);
    } else {
        fprintf(out->fp, "/* sort disabled */\n");
    }
    fprintf(out->fp,
            "#define __%sdefine_scalar_vector(N, T)\\\n"
            "typedef const T *N ## _vec_t;\\\n"
            "typedef T *N ## _mutable_vec_t;\\\n"
            "__%sdefine_scalar_vec_len(N)\\\n"
            "__%sdefine_scalar_vec_at(N, T)\\\n"
            "__%sdefine_scalar_find(N, T)\\\n"
            "__%sdefine_scalar_scan(N, T)",
            nsc, nsc, nsc, nsc, nsc);
    if (out->opts->cgen_sort) {
        fprintf(out->fp, "\\\n__%sdefine_scalar_sort(N, T)\n", nsc);
    } else {
        fprintf(out->fp, "\n");
    }
    fprintf(out->fp, "\n");
    /* Elaborate on the included basic type system. */
    fprintf(out->fp,
            "#define __%sdefine_integer_type(N, T, W)\\\n"
            "__flatcc_define_integer_accessors(N, T, W, %sendian)\\\n"
            "__%sdefine_scalar_vector(N, T)\n",
            nsc, nsc, nsc);
    fprintf(out->fp,
            "__%sdefine_scalar_vector(%sbool, %sbool_t)\n"
            "__%sdefine_scalar_vector(%schar, char)\n"
            "__%sdefine_scalar_vector(%suint8, uint8_t)\n"
            "__%sdefine_scalar_vector(%sint8, int8_t)\n"
            "__%sdefine_scalar_vector(%suint16, uint16_t)\n"
            "__%sdefine_scalar_vector(%sint16, int16_t)\n"
            "__%sdefine_scalar_vector(%suint32, uint32_t)\n"
            "__%sdefine_scalar_vector(%sint32, int32_t)\n"
            "__%sdefine_scalar_vector(%suint64, uint64_t)\n"
            "__%sdefine_scalar_vector(%sint64, int64_t)\n"
            "__%sdefine_scalar_vector(%sfloat, float)\n"
            "__%sdefine_scalar_vector(%sdouble, double)\n"
            "__%sdefine_scalar_vector(%sunion_type, %sunion_type_t)\n",
            nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc,
            nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc);
    fprintf(out->fp,
            "static inline size_t %sstring_vec_find(%sstring_vec_t vec, const char *s)\n"
            "__%sfind_by_string_field(__%sidentity, vec, %sstring_vec_at, %sstring_vec_len, s)\n"
            "static inline size_t %sstring_vec_find_n(%sstring_vec_t vec, const char *s, size_t n)\n"
            "__%sfind_by_string_n_field(__%sidentity, vec, %sstring_vec_at, %sstring_vec_len, s, n)\n",
            nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc);
    fprintf(out->fp,
            "static inline size_t %sstring_vec_scan(%sstring_vec_t vec, const char *s)\n"
            "__%sscan_by_string_field(0, %sstring_vec_len(vec), __%sidentity, vec, %sstring_vec_at, %sstring_vec_len, s)\n"
            "static inline size_t %sstring_vec_scan_n(%sstring_vec_t vec, const char *s, size_t n)\n"
            "__%sscan_by_string_n_field(0, %sstring_vec_len(vec), __%sidentity, vec, %sstring_vec_at, %sstring_vec_len, s, n)\n"
            "static inline size_t %sstring_vec_scan_ex(%sstring_vec_t vec, size_t begin, size_t end, const char *s)\n"
            "__%sscan_by_string_field(begin, __%smin(end, %sstring_vec_len(vec)), __%sidentity, vec, %sstring_vec_at, %sstring_vec_len, s)\n"
            "static inline size_t %sstring_vec_scan_ex_n(%sstring_vec_t vec, size_t begin, size_t end, const char *s, size_t n)\n"
            "__%sscan_by_string_n_field(begin, __%smin(end, %sstring_vec_len(vec)), __%sidentity, vec, %sstring_vec_at, %sstring_vec_len, s, n)\n"
            "static inline size_t %sstring_vec_rscan(%sstring_vec_t vec, const char *s)\n"
            "__%srscan_by_string_field(0, %sstring_vec_len(vec), __%sidentity, vec, %sstring_vec_at, %sstring_vec_len, s)\n"
            "static inline size_t %sstring_vec_rscan_n(%sstring_vec_t vec, const char *s, size_t n)\n"
            "__%srscan_by_string_n_field(0, %sstring_vec_len(vec), __%sidentity, vec, %sstring_vec_at, %sstring_vec_len, s, n)\n"
            "static inline size_t %sstring_vec_rscan_ex(%sstring_vec_t vec, size_t begin, size_t end, const char *s)\n"
            "__%srscan_by_string_field(begin, __%smin(end, %sstring_vec_len(vec)), __%sidentity, vec, %sstring_vec_at, %sstring_vec_len, s)\n"
            "static inline size_t %sstring_vec_rscan_ex_n(%sstring_vec_t vec, size_t begin, size_t end, const char *s, size_t n)\n"
            "__%srscan_by_string_n_field(begin, __%smin(end, %sstring_vec_len(vec)), __%sidentity, vec, %sstring_vec_at, %sstring_vec_len, s, n)\n",
            nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc,
            nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc,
            nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc,
            nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc,
            nsc, nsc, nsc, nsc);
    if (out->opts->cgen_sort) {
        fprintf(out->fp, "__%sdefine_string_sort()\n", nsc);
    }
    fprintf(out->fp,
        "#define __%sdefine_struct_scalar_fixed_array_field(N, NK, TK, T, L)\\\n"
        "static inline T N ## _ ## NK ## _get(N ## _struct_t t__tmp, size_t i__tmp)\\\n"
        "{ if (!t__tmp || i__tmp >= L) return 0;\\\n"
        "  return __%sread_scalar(TK, &(t__tmp->NK[i__tmp])); }\\\n"
        "static inline const T *N ## _ ## NK ## _get_ptr(N ## _struct_t t__tmp)\\\n"
        "{ return t__tmp ? t__tmp->NK : 0; }\\\n"
        "static inline size_t N ## _ ## NK ## _get_len(void) { return L; }",
        nsc, nsc);
    if (!out->opts->cgen_no_conflicts) {
        fprintf(out->fp,
            "\\\nstatic inline T N ## _ ## NK (N ## _struct_t t__tmp, size_t i__tmp)\\\n"
            "{ return N ## _ ## NK ## _get(t__tmp, i__tmp); }");
    }
    fprintf(out->fp, "\n");;
    fprintf(out->fp,
        "#define __%sdefine_struct_struct_fixed_array_field(N, NK, T, L)\\\n"
        "static inline T N ## _ ## NK ## _get(N ## _struct_t t__tmp, size_t i__tmp)\\\n"
        "{ if (!t__tmp || i__tmp >= L) return 0; return t__tmp->NK + i__tmp; }"
        "static inline T N ## _ ## NK ## _get_ptr(N ## _struct_t t__tmp)\\\n"
        "{ return t__tmp ? t__tmp->NK : 0; }\\\n"
        "static inline size_t N ## _ ## NK ## _get_len(void) { return L; }",
        nsc);
    if (!out->opts->cgen_no_conflicts) {
        fprintf(out->fp,
            "\\\nstatic inline T N ## _ ## NK(N ## _struct_t t__tmp, size_t i__tmp)\\\n"
            "{ if (!t__tmp || i__tmp >= L) return 0; return t__tmp->NK + i__tmp; }");
    }
    fprintf(out->fp, "\n");
    fprintf(out->fp,
        "#define __%sdefine_struct_scalar_field(N, NK, TK, T)\\\n"
        "static inline T N ## _ ## NK ## _get(N ## _struct_t t__tmp)\\\n"
        "{ return t__tmp ? __%sread_scalar(TK, &(t__tmp->NK)) : 0; }\\\n"
        "static inline const T *N ## _ ## NK ## _get_ptr(N ## _struct_t t__tmp)\\\n"
        "{ return t__tmp ? &(t__tmp->NK) : 0; }",
        nsc, nsc);
    if (!out->opts->cgen_no_conflicts) {
        fprintf(out->fp,
            "\\\nstatic inline T N ## _ ## NK (N ## _struct_t t__tmp)\\\n"
            "{ return t__tmp ? __%sread_scalar(TK, &(t__tmp->NK)) : 0; }",
            nsc);
    }
    if (out->opts->allow_scan_for_all_fields) {
        fprintf(out->fp, "\\\n__%sdefine_scan_by_scalar_field(N, NK, T)\n", nsc);
    } else {
        fprintf(out->fp, "\n");
    }
    fprintf(out->fp,
            "#define __%sdefine_struct_struct_field(N, NK, T)\\\n"
            "static inline T N ## _ ## NK ## _get(N ## _struct_t t__tmp) { return t__tmp ? &(t__tmp->NK) : 0; }",
            nsc);
    if (!out->opts->cgen_no_conflicts) {
    fprintf(out->fp,
            "\\\nstatic inline T N ## _ ## NK (N ## _struct_t t__tmp) { return t__tmp ? &(t__tmp->NK) : 0; }\n");
    } else {
        fprintf(out->fp, "\n");
    }
    fprintf(out->fp,
            "/* If fid is null, the function returns true without testing as buffer is not expected to have any id. */\n"
            "static inline int %shas_identifier(const void *buffer, const char *fid)\n"
            "{ %sthash_t id, id2 = 0; if (fid == 0) { return 1; };\n"
            "  id2 = %stype_hash_from_string(fid);\n"
            "  id = __%sthash_read_from_pe(((%suoffset_t *)buffer) + 1);\n"
            "  return id2 == 0 || id == id2; }\n"
            "static inline int %shas_type_hash(const void *buffer, %sthash_t thash)\n"
            "{ return thash == 0 || (__%sthash_read_from_pe((%suoffset_t *)buffer + 1) == thash); }\n\n"
            "static inline %sthash_t %sget_type_hash(const void *buffer)\n"
            "{ return __%sthash_read_from_pe((flatbuffers_uoffset_t *)buffer + 1); }\n\n"
            "#define %sverify_endian() %shas_identifier(\"\\x00\\x00\\x00\\x00\" \"1234\", \"1234\")\n",
            nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc);
    fprintf(out->fp,
            "static inline void *%sread_size_prefix(void *b, size_t *size_out)\n"
            "{ if (size_out) { *size_out = (size_t)__%suoffset_read_from_pe(b); }\n"
            "  return (uint8_t *)b + sizeof(%suoffset_t); }\n", nsc, nsc, nsc);
    fprintf(out->fp,
            "/* Null file identifier accepts anything, otherwise fid should be 4 characters. */\n"
            "#define __%sread_root(T, K, buffer, fid)\\\n"
            "  ((!buffer || !%shas_identifier(buffer, fid)) ? 0 :\\\n"
            "  ((T ## _ ## K ## t)(((uint8_t *)buffer) +\\\n"
            "    __%suoffset_read_from_pe(buffer))))\n"
            "#define __%sread_typed_root(T, K, buffer, thash)\\\n"
            "  ((!buffer || !%shas_type_hash(buffer, thash)) ? 0 :\\\n"
            "  ((T ## _ ## K ## t)(((uint8_t *)buffer) +\\\n"
            "    __%suoffset_read_from_pe(buffer))))\n",
            nsc, nsc, nsc, nsc, nsc, nsc);
    fprintf(out->fp,
            "#define __%snested_buffer_as_root(C, N, T, K)\\\n"
            "static inline T ## _ ## K ## t C ## _ ## N ## _as_root_with_identifier(C ## _ ## table_t t__tmp, const char *fid__tmp)\\\n"
            "{ const uint8_t *buffer__tmp = C ## _ ## N(t__tmp); return __%sread_root(T, K, buffer__tmp, fid__tmp); }\\\n"
            "static inline T ## _ ## K ## t C ## _ ## N ## _as_typed_root(C ## _ ## table_t t__tmp)\\\n"
            "{ const uint8_t *buffer__tmp = C ## _ ## N(t__tmp); return __%sread_root(T, K, buffer__tmp, C ## _ ## type_identifier); }\\\n"
            "static inline T ## _ ## K ## t C ## _ ## N ## _as_root(C ## _ ## table_t t__tmp)\\\n"
            "{ const char *fid__tmp = T ## _file_identifier;\\\n"
            "  const uint8_t *buffer__tmp = C ## _ ## N(t__tmp); return __%sread_root(T, K, buffer__tmp, fid__tmp); }\n",
            nsc, nsc, nsc, nsc);
    fprintf(out->fp,
            "#define __%sbuffer_as_root(N, K)\\\n"
            "static inline N ## _ ## K ## t N ## _as_root_with_identifier(const void *buffer__tmp, const char *fid__tmp)\\\n"
            "{ return __%sread_root(N, K, buffer__tmp, fid__tmp); }\\\n"
            "static inline N ## _ ## K ## t N ## _as_root_with_type_hash(const void *buffer__tmp, %sthash_t thash__tmp)\\\n"
            "{ return __%sread_typed_root(N, K, buffer__tmp, thash__tmp); }\\\n"
            "static inline N ## _ ## K ## t N ## _as_root(const void *buffer__tmp)\\\n"
            "{ const char *fid__tmp = N ## _file_identifier;\\\n"
            "  return __%sread_root(N, K, buffer__tmp, fid__tmp); }\\\n"
            "static inline N ## _ ## K ## t N ## _as_typed_root(const void *buffer__tmp)\\\n"
            "{ return __%sread_typed_root(N, K, buffer__tmp, N ## _type_hash); }\n"
            "#define __%sstruct_as_root(N) __%sbuffer_as_root(N, struct_)\n"
            "#define __%stable_as_root(N) __%sbuffer_as_root(N, table_)\n",
            nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc, nsc);
    fprintf(out->fp, "\n");
}

int fb_gen_common_c_header(fb_output_t *out)
{
    const char *nscup = out->nscup;

    fprintf(out->fp,
        "#ifndef %s_COMMON_READER_H\n"
        "#define %s_COMMON_READER_H\n",
        nscup, nscup);
    fprintf(out->fp, "\n/* " FLATCC_GENERATED_BY " */\n\n");
    fprintf(out->fp, "/* Common FlatBuffers read functionality for C. */\n\n");
    if (!out->opts->cgen_sort) {
        fprintf(out->fp,
                "/*"
                " * This code is generated without support for vector sort operations\n"
                " * but find operations are supported on pre-sorted vectors.\n"
                " */\n");
    }
    gen_prologue(out);
    gen_helpers(out);
    gen_epilogue(out);
    fprintf(out->fp,
        "#endif /* %s_COMMON_H */\n",
        nscup);
    return 0;
}

static void gen_pretext(fb_output_t *out)
{
    const char *nsc = out->nsc;
    const char *nscup = out->nscup;

    int do_pad = out->opts->cgen_pad;

    fprintf(out->fp,
        "#ifndef %s_READER_H\n"
        "#define %s_READER_H\n",
        out->S->basenameup, out->S->basenameup);

    fprintf(out->fp, "\n/* " FLATCC_GENERATED_BY " */\n\n");
    if (do_pad) {
        fprintf(out->fp,
        "/*\n"
        " * Generated with 'pad' option which expects #pragma pack(1) and\n"
        " * #pragma pack() to be supported, and which adds extra padding\n"
        " * fields to structs.\n"
        " *\n"
        " * This is mostly relevant for some micro controller platforms, but\n"
        " * may also be needed with 'force_align' attributes > 16.\n"
        " *\n"
        " * The default output uses C11 <stdalign.h> alignas(n) which can be\n"
        " * defined as `__attribute__((aligned (n)))` or similar on many\n"
        " * older platforms.\n"
        " */\n"
        "\n");
    }

    fprintf(out->fp,
            "#ifndef %s_COMMON_READER_H\n"
            "#include \"%scommon_reader.h\"\n"
            "#endif\n",
            nscup, nsc);
    fb_gen_c_includes(out, "_reader.h", "_READER_H");

    /*
     * Must be in included in every file using static_assert to ensure
     * static_assert_scope.h counter can avoid conflicts.
     */
    fprintf(out->fp,
                "#include \"flatcc/flatcc_flatbuffers.h\"\n");
    if (!do_pad) {
        fprintf(out->fp,
                "#ifndef __alignas_is_defined\n"
                "#include <stdalign.h>\n"
                "#endif\n");
    }
    gen_prologue(out);
    if (out->S->file_identifier.type == vt_string) {
        fprintf(out->fp,
            "#undef %sidentifier\n"
            "#define %sidentifier \"%.*s\"\n",
            nsc,
            nsc, out->S->file_identifier.s.len, out->S->file_identifier.s.s);
    } else {
        fprintf(out->fp,
            "#ifndef %sidentifier\n"
            "#define %sidentifier 0\n"
            "#endif\n",
            nsc, nsc);
    }
    if (out->S->file_extension.type == vt_string) {
        fprintf(out->fp,
            "#undef %sextension\n"
            "#define %sextension \"%.*s\"\n",
            nsc,
            nsc, out->S->file_extension.s.len, out->S->file_extension.s.s);
    } else {
        fprintf(out->fp,
            "#ifndef %sextension\n"
            "#define %sextension \"%s\"\n"
            "#endif\n",
            nsc, nsc, out->opts->default_bin_ext);
    }
    fprintf(out->fp, "\n");
}

static void gen_footer(fb_output_t *out)
{
    gen_epilogue(out);
    fprintf(out->fp, "#endif /* %s_READER_H */\n", out->S->basenameup);
}

static void gen_forward_decl(fb_output_t *out, fb_compound_type_t *ct)
{
    fb_scoped_name_t snt;
    const char *nsc = out->nsc;

    fb_clear(snt);

    assert(ct->symbol.kind == fb_is_struct || ct->symbol.kind == fb_is_table);

    fb_compound_name(ct, &snt);
    if (ct->symbol.kind == fb_is_struct) {
        if (ct->size == 0) {
            gen_panic(out, "internal error: unexpected empty struct");
            return;
        } else {
            fprintf(out->fp, "typedef struct %s %s_t;\n",
                    snt.text, snt.text);
        }
        fprintf(out->fp, "typedef const %s_t *%s_struct_t;\n",
                snt.text, snt.text);
        fprintf(out->fp, "typedef %s_t *%s_mutable_struct_t;\n",
                snt.text, snt.text);
        fprintf(out->fp, "typedef const %s_t *%s_vec_t;\n",
                snt.text, snt.text);
        fprintf(out->fp, "typedef %s_t *%s_mutable_vec_t;\n",
                snt.text, snt.text);
    } else {
        fprintf(out->fp, "typedef const struct %s_table *%s_table_t;\n",
                snt.text, snt.text);
        fprintf(out->fp, "typedef struct %s_table *%s_mutable_table_t;\n",
                snt.text, snt.text);
        fprintf(out->fp, "typedef const %suoffset_t *%s_vec_t;\n", nsc, snt.text);
        fprintf(out->fp, "typedef %suoffset_t *%s_mutable_vec_t;\n", nsc, snt.text);
    }
}

static inline void print_doc(fb_output_t *out, const char *indent, fb_doc_t *doc)
{
    long ln = 0;
    int first = 1;
    if (doc == 0) {
        return;
    }
    while (doc) {
        if (ln != doc->ident->linenum) {
            if (first) {
                /* Not all C compilers understand // comments. */
                fprintf(out->fp, "%s/** ", indent);
                ln = doc->ident->linenum;
            } else {
                fprintf(out->fp, "\n%s * ", indent);
            }
        }
        first = 0;
        fprintf(out->fp, "%.*s", (int)doc->ident->len, doc->ident->text);
        ln = doc->ident->linenum;
        doc = doc->link;
    }
    fprintf(out->fp, " */\n");
}

static void gen_struct(fb_output_t *out, fb_compound_type_t *ct)
{
    fb_member_t *member;
    fb_symbol_t *sym;
    unsigned align;
    size_t offset = 0;
    const char *tname, *tname_ns, *tname_prefix;
    int n, len;
    const char *s;
    unsigned pad_index = 0, deprecated_index = 0, pad;
    const char *kind;
    int do_pad = out->opts->cgen_pad;
    int is_primary_key, current_key_processed;
    const char *nsc = out->nsc;

    fb_scoped_name_t snt;
    fb_scoped_name_t snref;

    fb_clear(snt);
    fb_clear(snref);

    assert(ct->symbol.kind == fb_is_struct);
    assert(ct->align > 0 || ct->count == 0);
    assert(ct->size > 0 || ct->count == 0);

    fb_compound_name(ct, &snt);
    print_doc(out, "", ct->doc);
    if (ct->size == 0) {
        gen_panic(out, "internal error: unexpected empty struct");
    } else {
        if (do_pad) {
            fprintf(out->fp, "#pragma pack(1)\n");
        }
        /*
         * Unfortunately the following is not valid in C11:
         *
         *      struct alignas(4) mystruct { ... };
         *
         * we can only use alignas on members (unlike C++, and unlike
         * non-portable C compiler variants).
         *
         * By padding the first element to the struct size we get around
         * this problem. It shouldn't strictly be necessary to add padding
         * fields, but compilers might not support padding above 16 bytes,
         * so we do that as a precaution with an optional compiler flag.
         */
        fprintf(out->fp, "struct %s {\n", snt.text);
        for (sym = ct->members; sym; sym = sym->link) {
            current_key_processed = 0;
            member = (fb_member_t *)sym;
            is_primary_key = ct->primary_key == member;
            print_doc(out, "    ", member->doc);
            symbol_name(sym, &n, &s);
            align = offset == 0 ? ct->align : member->align;
            if (do_pad && (pad = (unsigned)(member->offset - offset))) {
                fprintf(out->fp, "    uint8_t __padding%u[%u];\n",
                        pad_index++, pad);
            }
            if (member->metadata_flags & fb_f_deprecated) {
                pad = (unsigned)member->size;
                if (do_pad) {
                    fprintf(out->fp, "    uint8_t __deprecated%u[%u]; /* was: '%.*s' */\n",
                            deprecated_index++, pad, n, s);
                } else {
                    fprintf(out->fp, "    alignas(%u) uint8_t __deprecated%u[%u]; /* was: '%.*s' */\n",
                            align, deprecated_index++, pad, n, s);
                }
                offset = (unsigned)(member->offset + member->size);
                continue;
            }
            switch (member->type.type) {
            case vt_fixed_array_type:
                tname_ns = scalar_type_ns(member->type.st, nsc);
                tname = scalar_type_name(member->type.st);
                len = (int)member->type.len;
                if (do_pad) {
                    fprintf(out->fp, "    %s%s ", tname_ns, tname);
                } else {
                    fprintf(out->fp, "    alignas(%u) %s%s ", align, tname_ns, tname);
                }
                fprintf(out->fp, "%.*s[%d];\n", n, s, len);
                break;
            case vt_scalar_type:
                tname_ns = scalar_type_ns(member->type.st, nsc);
                tname = scalar_type_name(member->type.st);
                if (do_pad) {
                    fprintf(out->fp, "    %s%s ", tname_ns, tname);
                } else {
                    fprintf(out->fp, "    alignas(%u) %s%s ", align, tname_ns, tname);
                }
                fprintf(out->fp, "%.*s;\n", n, s);
                break;
            case vt_fixed_array_compound_type_ref:
                assert(member->type.ct->symbol.kind == fb_is_struct || member->type.ct->symbol.kind == fb_is_enum);
                kind = member->type.ct->symbol.kind == fb_is_struct ? "" : "enum_";
                fb_compound_name(member->type.ct, &snref);
                len = (int)member->type.len;
                if (do_pad) {
                    fprintf(out->fp, "    %s_%st ", snref.text, kind);
                } else {
                    fprintf(out->fp, "    alignas(%u) %s_%st ", align, snref.text, kind);
                }
                fprintf(out->fp, "%.*s[%d];\n", n, s, len);
                break;
            case vt_compound_type_ref:
                assert(member->type.ct->symbol.kind == fb_is_struct || member->type.ct->symbol.kind == fb_is_enum);
                kind = member->type.ct->symbol.kind == fb_is_struct ? "" : "enum_";
                fb_compound_name(member->type.ct, &snref);
                if (do_pad) {
                    fprintf(out->fp, "    %s_%st ", snref.text, kind);
                } else {
                    fprintf(out->fp, "    alignas(%u) %s_%st ", align, snref.text, kind);
                }
                fprintf(out->fp, "%.*s;\n", n, s);
                break;
            default:
                fprintf(out->fp, "    %s ", __FLATCC_ERROR_TYPE);
                fprintf(out->fp, "%.*s;\n", n, s);
                gen_panic(out, "internal error: unexpected type during code generation");
                break;
            }
            offset = (unsigned)(member->offset + member->size);
        }
        if (do_pad && (pad = (unsigned)(ct->size - offset))) {
            fprintf(out->fp, "    uint8_t __padding%u[%u];\n",
                    pad_index, pad);
        }
        fprintf(out->fp, "};\n");
        if (do_pad) {
            fprintf(out->fp, "#pragma pack()\n");
        }
        fprintf(out->fp,
                "static_assert(sizeof(%s_t) == %"PRIu64", \"struct size mismatch\");\n\n",
                snt.text, (uint64_t)ct->size);
        fprintf(out->fp,
                "static inline const %s_t *%s__const_ptr_add(const %s_t *p, size_t i) { return p + i; }\n", snt.text, snt.text, snt.text);
        fprintf(out->fp,
                "static inline %s_t *%s__ptr_add(%s_t *p, size_t i) { return p + i; }\n", snt.text, snt.text, snt.text);
        fprintf(out->fp,
                "static inline %s_struct_t %s_vec_at(%s_vec_t vec, size_t i)\n"
                "__%sstruct_vec_at(vec, i)\n",
                snt.text, snt.text, snt.text,
                nsc);
    }
    fprintf(out->fp, "static inline size_t %s__size(void) { return %"PRIu64"; }\n",
            snt.text, (uint64_t)ct->size);
    fprintf(out->fp,
            "static inline size_t %s_vec_len(%s_vec_t vec)\n"
            "__%svec_len(vec)\n",
            snt.text, snt.text, nsc);
    fprintf(out->fp,
            "__%sstruct_as_root(%s)\n",
            nsc, snt.text);
    fprintf(out->fp, "\n");

    /* Create accessors which respect endianness and which return 0 on null struct access. */
    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        is_primary_key = ct->primary_key == member;
        if (member->metadata_flags & fb_f_deprecated) {
            continue;
        }
        symbol_name(&member->symbol, &n, &s);
        switch (member->type.type) {
        case vt_fixed_array_type:
            tname_ns = scalar_type_ns(member->type.st, nsc);
            tname = scalar_type_name(member->type.st);
            tname_prefix = scalar_type_prefix(member->type.st);
            fprintf(out->fp,
                "__%sdefine_struct_scalar_fixed_array_field(%s, %.*s, %s%s, %s%s, %"PRIu64")\n",
                nsc, snt.text, n, s, nsc, tname_prefix, tname_ns, tname, (uint64_t)member->type.len);
            /* TODO: if member->type.st == fb_char add string specific methods. */
            break;
        case vt_scalar_type:
            tname_ns = scalar_type_ns(member->type.st, nsc);
            tname = scalar_type_name(member->type.st);
            tname_prefix = scalar_type_prefix(member->type.st);
            fprintf(out->fp,
                "__%sdefine_struct_scalar_field(%s, %.*s, %s%s, %s%s)\n",
                nsc, snt.text, n, s, nsc, tname_prefix, tname_ns, tname);
            if (!out->opts->allow_scan_for_all_fields && (member->metadata_flags & fb_f_key)) {
                fprintf(out->fp,
                        "__%sdefine_scan_by_scalar_field(%s, %.*s, %s%s)\n",
                        nsc, snt.text, n, s, tname_ns, tname);
            }
            if (member->metadata_flags & fb_f_key) {
                if (!is_primary_key) {
                    fprintf(out->fp, "/* Note: this is not the primary key field on this struct. */\n");
                }
                fprintf(out->fp,     "/* Note: find only works on vectors sorted by this field. */\n");
                fprintf(out->fp,
                        "__%sdefine_find_by_scalar_field(%s, %.*s, %s%s)\n",
                        nsc, snt.text, n, s, tname_ns, tname);
                if (out->opts->cgen_sort) {
                    fprintf(out->fp,
                        "__%sdefine_struct_sort_by_scalar_field(%s, %.*s, %s%s, %s_t)\n",
                        nsc, snt.text, n, s, tname_ns, tname, snt.text);
                }
                if (is_primary_key) {
                    fprintf(out->fp,
                        "__%sdefine_default_find_by_scalar_field(%s, %.*s, %s%s)\n",
                        nsc, snt.text, n, s, tname_ns, tname);
                    fprintf(out->fp,
                        "__%sdefine_default_scan_by_scalar_field(%s, %.*s, %s%s)\n",
                        nsc, snt.text, n, s, tname_ns, tname);
                    if (out->opts->cgen_sort) {
                        fprintf(out->fp,
                            "#define %s_vec_sort %s_vec_sort_by_%.*s\n",
                            snt.text, snt.text, n, s);
                    }
                }
                current_key_processed = 1;
            }
            break;
        case vt_fixed_array_compound_type_ref:
            fb_compound_name(member->type.ct, &snref);
            switch (member->type.ct->symbol.kind) {
            case fb_is_enum:
                fprintf(out->fp,
                    "__%sdefine_struct_scalar_fixed_array_field(%s, %.*s, %s, %s_enum_t, %"PRIu64")\n",
                    nsc, snt.text, n, s, snref.text, snref.text, (uint64_t)member->type.len);
                break;
            case fb_is_struct:
                fprintf(out->fp,
                    "__%sdefine_struct_struct_fixed_array_field(%s, %.*s, %s_struct_t, %"PRIu64")\n",
                    nsc, snt.text, n, s, snref.text, (uint64_t)member->type.len);
                break;
            }
            break;

        case vt_compound_type_ref:
            fb_compound_name(member->type.ct, &snref);
            switch (member->type.ct->symbol.kind) {
            case fb_is_enum:
                fprintf(out->fp,
                    "__%sdefine_struct_scalar_field(%s, %.*s, %s, %s_enum_t)\n",
                    nsc, snt.text, n, s, snref.text, snref.text);
                if (!out->opts->allow_scan_for_all_fields && (member->metadata_flags & fb_f_key)) {
                    fprintf(out->fp,
                            "__%sdefine_scan_by_scalar_field(%s, %.*s, %s_enum_t)\n",
                            nsc, snt.text, n, s, snref.text);
                }
                if (member->metadata_flags & fb_f_key) {
                    if (!is_primary_key) {
                        fprintf(out->fp, "/* Note: this is not the primary key of this table. */\n");
                    }
                    fprintf(out->fp,     "/* Note: find only works on vectors sorted by this field. */\n");
                    fprintf(out->fp,
                            "__%sdefine_find_by_scalar_field(%s, %.*s, %s_enum_t)\n",
                            nsc, snt.text, n, s, snref.text);
                    if (out->opts->cgen_sort) {
                        fprintf(out->fp,
                            "__%sdefine_struct_sort_by_scalar_field(%s, %.*s, %s_enum_t, %s_t)\n",
                            nsc, snt.text, n, s, snref.text, snt.text);
                    }
                    if (is_primary_key) {
                        fprintf(out->fp,
                            "__%sdefine_default_find_by_scalar_field(%s, %.*s, %s_enum_t)\n",
                            nsc, snt.text, n, s, snref.text);
                        fprintf(out->fp,
                            "__%sdefine_default_scan_by_scalar_field(%s, %.*s, %s_enum_t)\n",
                            nsc, snt.text, n, s, snref.text);
                        if (out->opts->cgen_sort) {
                            fprintf(out->fp,
                                "#define %s_vec_sort %s_vec_sort_by_%.*s\n",
                                snt.text, snt.text, n, s);
                        }
                    }
                    current_key_processed = 1;
                }
                break;
            case fb_is_struct:
                /*
                 * For completeness provide an accessor which returns member pointer
                 * or null if container struct is null.
                 */
                fprintf(out->fp,
                    "__%sdefine_struct_struct_field(%s, %.*s, %s_struct_t)\n",
                    nsc, snt.text, n, s, snref.text);
                break;
            }

        }
        if ((member->metadata_flags & fb_f_key) && !current_key_processed) {
            fprintf(out->fp,
                "/* Note: field has key, but there is no support for find by fields of this type. */\n");
            /*
             * If the first key already exists, but was for an unsupported
             * type, we do not map the next possible key to generic find.
             */
        }
    }
    fprintf(out->fp, "\n");
}

/*
 * Enums are integers, but we cannot control the size.
 * To produce a typesafe and portable result, we generate constants
 * instead.
 */
static void gen_enum(fb_output_t *out, fb_compound_type_t *ct)
{
    fb_member_t *member;
    fb_symbol_t *sym;
    const char *tname, *tname_ns, *s, *kind;
    fb_literal_t literal;
    int n, w;
    int is_union;
    fb_scoped_name_t snt;
    const char *nsc = out->nsc;

    fb_clear(snt);

    assert(ct->symbol.kind == fb_is_enum || ct->symbol.kind == fb_is_union);
    assert(ct->type.type == vt_scalar_type);

    tname_ns = scalar_type_ns(ct->type.st, nsc);
    tname = scalar_type_name(ct->type.st);

    w = (int)ct->size * 8;

    is_union = ct->symbol.kind != fb_is_enum;
    kind = is_union ? "union_type" : "enum";
    fb_compound_name(ct, &snt);
    print_doc(out, "", ct->doc);
    fprintf(out->fp,
            "typedef %s%s %s_%s_t;\n",
            tname_ns, tname, snt.text, kind);
    fprintf(out->fp,
            "__%sdefine_integer_type(%s, %s_%s_t, %u)\n",
            nsc, snt.text, snt.text, kind, w);
    if (is_union) {
        fprintf(out->fp,
            "__%sdefine_union(%s, %s)\n",
            nsc, nsc, snt.text);
    }
    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        print_doc(out, "", member->doc);
        symbol_name(&member->symbol, &n, &s);
        print_literal(ct->type.st, &member->value, literal);
        /*
         * This must be a define, not a static const integer, otherwise it
         * won't work in switch statements - except with GNU extensions.
         */
        fprintf(out->fp,
                "#define %s_%.*s ((%s_%s_t)%s)\n",
                snt.text, n, s, snt.text, kind, literal);
    }
    fprintf(out->fp, "\n");

    if (is_union) {
        fprintf(out->fp, "static inline const char *%s_type_name(%s_union_type_t type)\n"
                "{\n",
                snt.text, snt.text);
    } else {
        fprintf(out->fp, "static inline const char *%s_name(%s_enum_t value)\n"
                "{\n",
                snt.text, snt.text);
    }


    if (is_union) {
        fprintf(out->fp, "    switch (type) {\n");
    } else {
        fprintf(out->fp, "    switch (value) {\n");
    }
    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        symbol_name(&member->symbol, &n, &s);
        if (sym->flags & fb_duplicate) {
            fprintf(out->fp,
                    "    /* case %s_%.*s: return \"%.*s\"; (duplicate) */\n",
                    snt.text, n, s, n, s);
        } else {
            fprintf(out->fp,
                    "    case %s_%.*s: return \"%.*s\";\n",
                    snt.text, n, s, n, s);
        }
    }
    fprintf(out->fp,
            "    default: return \"\";\n"
            "    }\n"
            "}\n");
    fprintf(out->fp, "\n");

    if (is_union) {
        fprintf(out->fp, "static inline int %s_is_known_type(%s_union_type_t type)\n"
                "{\n",
                snt.text, snt.text);
    } else {
        fprintf(out->fp, "static inline int %s_is_known_value(%s_enum_t value)\n"
                "{\n",
                snt.text, snt.text);
    }
    if (is_union) {
        fprintf(out->fp, "    switch (type) {\n");
    } else {
        fprintf(out->fp, "    switch (value) {\n");
    }
    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        symbol_name(&member->symbol, &n, &s);
        if (sym->flags & fb_duplicate) {
            fprintf(out->fp,
                    "    /* case %s_%.*s: return 1; (duplicate) */\n",
                    snt.text, n, s);
        } else {
            fprintf(out->fp,
                    "    case %s_%.*s: return 1;\n",
                    snt.text, n, s);
        }
    }
    fprintf(out->fp,
            "    default: return 0;\n"
            "    }\n"
            "}\n");
    fprintf(out->fp, "\n");

}

static void gen_nested_root(fb_output_t *out, fb_symbol_t *root_type, fb_symbol_t *container, fb_symbol_t *member)
{
    const char *s;
    int n;
    const char *kind;
    const char *nsc = out->nsc;
    fb_scoped_name_t snt;
    fb_scoped_name_t snc;

    fb_clear(snt);
    fb_clear(snc);
    if (!root_type) {
        return;
    }
    /*
     * Current flatc compiler only accepts tables, but here we support
     * both tables and structs in so far the parser and analyzer
     * allows for it.
     */
    switch (root_type->kind) {
    case fb_is_table:
        kind = "table_";
        break;
    case fb_is_struct:
        kind = "struct_";
        break;
    default:
        gen_panic(out, "internal error: roots can only be structs or tables");
        return;
    }
    fb_compound_name((fb_compound_type_t *)root_type, &snt);
    assert(container->kind == fb_is_table);
    fb_compound_name((fb_compound_type_t *)container, &snc);
    symbol_name(member, &n, &s);
    fprintf(out->fp, "__%snested_buffer_as_root(%s, %.*s, %s, %s)\n", nsc, snc.text, n, s, snt.text, kind);
}

static void gen_table(fb_output_t *out, fb_compound_type_t *ct)
{
    fb_member_t *member;
    fb_symbol_t *sym;
    const char *s, *tname, *tname_ns, *tname_prefix;
    int n, r;
    int is_primary_key, current_key_processed;
    const char *nsc = out->nsc;
    fb_scoped_name_t snt;
    fb_scoped_name_t snref;
    fb_literal_t literal;
    int is_optional;

    assert(ct->symbol.kind == fb_is_table);

    fb_clear(snt);
    fb_clear(snref);

    fprintf(out->fp, "\n");
    fb_compound_name(ct, &snt);
    print_doc(out, "", ct->doc);
    fprintf(out->fp,
            /*
             * We don't really need the struct, but it provides better
             * type safety than a typedef void *.
             */
            "struct %s_table { uint8_t unused__; };\n"
            "\n",
            snt.text);
    fprintf(out->fp,
            "static inline size_t %s_vec_len(%s_vec_t vec)\n"
            "__%svec_len(vec)\n",
            snt.text, snt.text, nsc);
    fprintf(out->fp,
            "static inline %s_table_t %s_vec_at(%s_vec_t vec, size_t i)\n"
            "__%soffset_vec_at(%s_table_t, vec, i, 0)\n",
            snt.text, snt.text, snt.text, nsc, snt.text);
    fprintf(out->fp,
            "__%stable_as_root(%s)\n",
            nsc, snt.text);
    fprintf(out->fp, "\n");

    for (sym = ct->members; sym; sym = sym->link) {
        current_key_processed = 0;
        member = (fb_member_t *)sym;
        is_primary_key = ct->primary_key == member;
        is_optional = !!(member->flags & fb_fm_optional);
        print_doc(out, "", member->doc);
        /*
         * In flatc, there can at most one key field, and it should be
         * scalar or string. Here we export all keys using the
         * <table>_vec_find_by_<fieldname> convention and let the parser deal with
         * semantics. Keys on unsupported fields are ignored. The first
         * valid find operation is also mapped to just <table>_vec_find.
         */
        symbol_name(&member->symbol, &n, &s);
        if (member->metadata_flags & fb_f_deprecated) {
            fprintf(out->fp, "/* Skipping deprecated field: '%s_%.*s' */\n\n", snt.text, n, s);
            continue;
        }
        r = (member->metadata_flags & fb_f_required) != 0;
        switch (member->type.type) {
        case vt_scalar_type:
            tname_ns = scalar_type_ns(member->type.st, nsc);
            tname = scalar_type_name(member->type.st);
            tname_prefix = scalar_type_prefix(member->type.st);
            print_literal(member->type.st, &member->value, literal);
            if (is_optional) {
                fprintf(out->fp,
                    "__%sdefine_scalar_optional_field(%"PRIu64", %s, %.*s, %s%s, %s%s, %s)\n",
                    nsc, (uint64_t)member->id, snt.text, n, s, nsc, tname_prefix, tname_ns, tname, literal);
            } else {
                fprintf(out->fp,
                    "__%sdefine_scalar_field(%"PRIu64", %s, %.*s, %s%s, %s%s, %s)\n",
                    nsc, (uint64_t)member->id, snt.text, n, s, nsc, tname_prefix, tname_ns, tname, literal);
            }
            if (!out->opts->allow_scan_for_all_fields && (member->metadata_flags & fb_f_key)) {
                fprintf(out->fp,
                        "__%sdefine_scan_by_scalar_field(%s, %.*s, %s%s)\n",
                        nsc, snt.text, n, s, tname_ns, tname);
            }
            if (member->metadata_flags & fb_f_key) {
                if (!is_primary_key) {
                    fprintf(out->fp, "/* Note: this is not the primary key of this table. */\n");
                }
                fprintf(out->fp,     "/* Note: find only works on vectors sorted by this field. */\n");
                fprintf(out->fp,
                        "__%sdefine_find_by_scalar_field(%s, %.*s, %s%s)\n",
                        nsc, snt.text, n, s, tname_ns, tname);
                if (out->opts->cgen_sort) {
                    fprintf(out->fp,
                        "__%sdefine_table_sort_by_scalar_field(%s, %.*s, %s%s)\n",
                        nsc, snt.text, n, s, tname_ns, tname);
                }
                if (is_primary_key) {
                    fprintf(out->fp,
                        "__%sdefine_default_find_by_scalar_field(%s, %.*s, %s%s)\n",
                        nsc, snt.text, n, s, tname_ns, tname);
                    fprintf(out->fp,
                        "__%sdefine_default_scan_by_scalar_field(%s, %.*s, %s%s)\n",
                        nsc, snt.text, n, s, tname_ns, tname);
                    if (out->opts->cgen_sort) {
                        fprintf(out->fp,
                            "#define %s_vec_sort %s_vec_sort_by_%.*s\n",
                            snt.text, snt.text, n, s);
                    }
                }
                current_key_processed = 1;
            }
            break;
        case vt_vector_type:
            /* They all use a namespace. */
            tname = scalar_vector_type_name(member->type.st);
            tname_ns = nsc;
            fprintf(out->fp,
                "__%sdefine_vector_field(%"PRIu64", %s, %.*s, %s%s, %u)\n",
                nsc, (uint64_t)member->id, snt.text, n, s, tname_ns, tname, r);
            if (member->nest) {
                gen_nested_root(out, &member->nest->symbol, &ct->symbol, &member->symbol);
            }
            break;
        case vt_string_type:
            fprintf(out->fp,
                "__%sdefine_string_field(%"PRIu64", %s, %.*s, %u)\n",
                nsc, (uint64_t)member->id, snt.text, n, s, r);
            if (!out->opts->allow_scan_for_all_fields && (member->metadata_flags & fb_f_key)) {
                fprintf(out->fp,
                    "__%sdefine_scan_by_string_field(%s, %.*s)\n",
                    nsc, snt.text, n, s);
            }
            if (member->metadata_flags & fb_f_key) {
                if (!is_primary_key) {
                    fprintf(out->fp, "/* Note: this is not the primary key of this table. */\n");
                }
                fprintf(out->fp,
                    "__%sdefine_find_by_string_field(%s, %.*s)\n",
                    nsc, snt.text, n, s);
                if (out->opts->cgen_sort) {
                    fprintf(out->fp,
                        "__%sdefine_table_sort_by_string_field(%s, %.*s)\n",
                        nsc, snt.text, n, s);
                }
                if (is_primary_key) {
                    fprintf(out->fp,
                        "__%sdefine_default_find_by_string_field(%s, %.*s)\n",
                        nsc, snt.text, n, s);
                    fprintf(out->fp,
                        "__%sdefine_default_scan_by_string_field(%s, %.*s)\n",
                        nsc, snt.text, n, s);
                    if (out->opts->cgen_sort) {
                        fprintf(out->fp,
                            "#define %s_vec_sort %s_vec_sort_by_%.*s\n",
                            snt.text, snt.text, n, s);
                    }
                }
                current_key_processed = 1;
            }
            break;
        case vt_vector_string_type:
            fprintf(out->fp,
                "__%sdefine_vector_field(%"PRIu64", %s, %.*s, %sstring_vec_t, %u)\n",
                nsc, (uint64_t)member->id, snt.text, n, s, nsc, r);
            break;
        case vt_compound_type_ref:
            fb_compound_name(member->type.ct, &snref);
            switch (member->type.ct->symbol.kind) {
            case fb_is_struct:
                fprintf(out->fp,
                    "__%sdefine_struct_field(%"PRIu64", %s, %.*s, %s_struct_t, %u)\n",
                    nsc, (uint64_t)member->id, snt.text, n, s, snref.text, r);
                break;
            case fb_is_table:
                fprintf(out->fp,
                    "__%sdefine_table_field(%"PRIu64", %s, %.*s, %s_table_t, %u)\n",
                    nsc, (uint64_t)member->id, snt.text, n, s, snref.text, r);
                break;
            case fb_is_enum:
                print_literal(member->type.ct->type.st, &member->value, literal);
                if (is_optional) {
                    fprintf(out->fp,
                        "__%sdefine_scalar_optional_field(%"PRIu64", %s, %.*s, %s, %s_enum_t, %s)\n",
                        nsc, (uint64_t)member->id, snt.text, n, s, snref.text, snref.text, literal);
                } else {
                    fprintf(out->fp,
                        "__%sdefine_scalar_field(%"PRIu64", %s, %.*s, %s, %s_enum_t, %s)\n",
                        nsc, (uint64_t)member->id, snt.text, n, s, snref.text, snref.text, literal);
                }
                if (!out->opts->allow_scan_for_all_fields && (member->metadata_flags & fb_f_key)) {
                    fprintf(out->fp,
                            "__%sdefine_scan_by_scalar_field(%s, %.*s, %s_enum_t)\n",
                            nsc, snt.text, n, s, snref.text);
                }
                if (member->metadata_flags & fb_f_key) {
                    if (!is_primary_key) {
                        fprintf(out->fp, "/* Note: this is not the primary key of this table. */\n");
                    }
                    fprintf(out->fp,     "/* Note: find only works on vectors sorted by this field. */\n");
                    fprintf(out->fp,
                            "__%sdefine_find_by_scalar_field(%s, %.*s, %s_enum_t)\n",
                            nsc, snt.text, n, s, snref.text);
                    if (out->opts->cgen_sort) {
                        fprintf(out->fp,
                                "__%sdefine_table_sort_by_scalar_field(%s, %.*s, %s_enum_t)\n",
                                nsc, snt.text, n, s, snref.text);
                    }
                    if (is_primary_key) {
                        fprintf(out->fp,
                                "__%sdefine_default_find_by_scalar_field(%s, %.*s, %s_enum_t)\n",
                                nsc, snt.text, n, s, snref.text);
                        fprintf(out->fp,
                                "__%sdefine_default_scan_by_scalar_field(%s, %.*s, %s_enum_t)\n",
                                nsc, snt.text, n, s, snref.text);
                        if (out->opts->cgen_sort) {
                            fprintf(out->fp,
                                    "#define %s_vec_sort %s_vec_sort_by_%.*s\n",
                                    snt.text, snt.text, n, s);
                        }
                    }
                    current_key_processed = 1;
                }
                break;
            case fb_is_union:
                fprintf(out->fp,
                    "__%sdefine_union_field(%s, %"PRIu64", %s, %.*s, %s, %u)\n",
                    nsc, nsc, (uint64_t)member->id, snt.text, n, s, snref.text, r);
                break;
            default:
                gen_panic(out, "internal error: unexpected compound type in table during code generation");
                break;
            }
            break;
        case vt_vector_compound_type_ref:
            fb_compound_name(member->type.ct, &snref);
            switch (member->type.ct->symbol.kind) {
            case fb_is_struct:
                break;
            case fb_is_table:
                break;
            case fb_is_enum:
                break;
            case fb_is_union:
                break;
            default:
                gen_panic(out, "internal error: unexpected vector compound type in table during code generation");
                break;
            }
            if (member->type.ct->symbol.kind == fb_is_union) {
                fprintf(out->fp,
                    "__%sdefine_union_vector_field(%s, %"PRIu64", %s, %.*s, %s, %u)\n",
                    nsc, nsc, (uint64_t)member->id, snt.text, n, s, snref.text, r);
            } else {
                fprintf(out->fp,
                    "__%sdefine_vector_field(%"PRIu64", %s, %.*s, %s_vec_t, %u)\n",
                    nsc, (uint64_t)member->id, snt.text, n, s, snref.text, r);
            }
            break;
        default:
            gen_panic(out, "internal error: unexpected table member type during code generation");
            break;
        }
        if ((member->metadata_flags & fb_f_key) && !current_key_processed) {
            fprintf(out->fp,
                "/* Note: field has key, but there is no support for find by fields of this type. */\n");
            /*
             * If the first key already exists, but was for an unsupported
             * type, we do not map the next possible key to generic find.
             */
        }
    }
}

int fb_gen_c_reader(fb_output_t *out)
{
    fb_symbol_t *sym;
    fb_compound_type_t *ct;

    gen_pretext(out);

    for (ct = out->S->ordered_structs; ct; ct = ct->order) {
        gen_forward_decl(out, ct);
    }
    fprintf(out->fp, "\n");
    for (sym = out->S->symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_table:
            gen_forward_decl(out, (fb_compound_type_t *)sym);
            break;
        }
    }
    /* Must be placed early due to nested buffer circular references. */
    for (sym = out->S->symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_struct:
            /* Fall through. */
        case fb_is_table:
            print_type_identifier(out, (fb_compound_type_t *)sym);
            print_file_extension(out, (fb_compound_type_t *)sym);
            break;
        }
    }
    fprintf(out->fp, "\n");
    for (sym = out->S->symbols; sym; sym = sym->link) {
        switch (sym->kind) {
         /* Enums must come before structs in case they are referenced. */
        case fb_is_enum:
            gen_enum(out, (fb_compound_type_t *)sym);
            break;
        }
    }
    fprintf(out->fp, "\n");
    /* Generate structs in topologically sorted order. */
    for (ct = out->S->ordered_structs; ct; ct = ct->order) {
            gen_struct(out, ct);
    }
    for (sym = out->S->symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_enum:
        case fb_is_struct:
            /* Already generated. */
            break;
        case fb_is_union:
            gen_enum(out, (fb_compound_type_t *)sym);
            break;
        case fb_is_table:
            gen_table(out, (fb_compound_type_t *)sym);
            break;
        case fb_is_rpc_service:
            /* Ignore. */
            break;
        default:
            gen_panic(out, "internal error: unexpected schema component");
            break;
        }
    }
    fprintf(out->fp, "\n");

    if (out->opts->cgen_sort) {
        fb_gen_c_sorter(out);
    }

    gen_footer(out);
    return 0;
}
