#include <string.h>

#include "codegen_c.h"

int fb_gen_common_c_builder_header(fb_output_t *out)
{
    const char *nsc = out->nsc;
    const char *nscup = out->nscup;

    fprintf(out->fp, "#ifndef %s_COMMON_BUILDER_H\n", nscup);
    fprintf(out->fp, "#define %s_COMMON_BUILDER_H\n", nscup);
    fprintf(out->fp, "\n/* " FLATCC_GENERATED_BY " */\n\n");
    fprintf(out->fp, "/* Common FlatBuffers build functionality for C. */\n\n");
    gen_prologue(out);

    fprintf(out->fp, "#ifndef FLATBUILDER_H\n");
    fprintf(out->fp, "#include \"flatcc/flatcc_builder.h\"\n");
    fprintf(out->fp, "#endif\n");
    if (strcmp(nsc, "flatcc_builder_")) {
        fprintf(out->fp, "typedef flatcc_builder_t %sbuilder_t;\n", nsc);
        fprintf(out->fp, "typedef flatcc_builder_ref_t %sref_t;\n", nsc);
        fprintf(out->fp, "typedef flatcc_builder_ref_t %svec_ref_t;\n", nsc);
        fprintf(out->fp, "typedef flatcc_builder_union_ref_t %sunion_ref_t;\n", nsc);
        fprintf(out->fp, "typedef flatcc_builder_union_vec_ref_t %sunion_vec_ref_t;\n", nsc);
        fprintf(out->fp, "/* integer return code (ref and ptr always fail on 0) */\n"
                "#define %sfailed(x) ((x) < 0)\n", nsc);
    }
    fprintf(out->fp, "typedef %sref_t %sroot_t;\n", nsc, nsc);
    fprintf(out->fp, "#define %sroot(ref) ((%sroot_t)(ref))\n", nsc, nsc);
    if (strcmp(nsc, "flatbuffers_")) {
        fprintf(out->fp, "#define %sis_native_pe flatbuffers_is_native_pe\n", nsc);
        fprintf(out->fp, "typedef flatbuffers_fid_t %sfid_t;\n", nsc);
    }
    fprintf(out->fp, "\n");

    fprintf(out->fp,
        "#define __%smemoize_begin(B, src)\\\n"
        "do { flatcc_builder_ref_t _ref; if ((_ref = flatcc_builder_refmap_find((B), (src)))) return _ref; } while (0)\n"
        "#define __%smemoize_end(B, src, op) do { return flatcc_builder_refmap_insert((B), (src), (op)); } while (0)\n"
        "#define __%smemoize(B, src, op) do { __%smemoize_begin(B, src); __%smemoize_end(B, src, op); } while (0)\n"
        "\n",
        nsc, nsc, nsc, nsc, nsc);

    fprintf(out->fp,
        "#define __%sbuild_buffer(NS)\\\n"
        "typedef NS ## ref_t NS ## buffer_ref_t;\\\n"
        "static inline int NS ## buffer_start(NS ## builder_t *B, const NS ##fid_t fid)\\\n"
        "{ return flatcc_builder_start_buffer(B, fid, 0, 0); }\\\n"
        "static inline int NS ## buffer_start_with_size(NS ## builder_t *B, const NS ##fid_t fid)\\\n"
        "{ return flatcc_builder_start_buffer(B, fid, 0, flatcc_builder_with_size); }\\\n"
        "static inline int NS ## buffer_start_aligned(NS ## builder_t *B, NS ##fid_t fid, uint16_t block_align)\\\n"
        "{ return flatcc_builder_start_buffer(B, fid, block_align, 0); }\\\n"
        "static inline int NS ## buffer_start_aligned_with_size(NS ## builder_t *B, NS ##fid_t fid, uint16_t block_align)\\\n"
        "{ return flatcc_builder_start_buffer(B, fid, block_align, flatcc_builder_with_size); }\\\n"
        "static inline NS ## buffer_ref_t NS ## buffer_end(NS ## builder_t *B, NS ## ref_t root)\\\n"
        "{ return flatcc_builder_end_buffer(B, root); }\n"
        "\n",
        nsc);

    fprintf(out->fp,
        "#define __%sbuild_table_root(NS, N, FID, TFID)\\\n"
        "static inline int N ## _start_as_root(NS ## builder_t *B)\\\n"
        "{ return NS ## buffer_start(B, FID) ? -1 : N ## _start(B); }\\\n"
        "static inline int N ## _start_as_root_with_size(NS ## builder_t *B)\\\n"
        "{ return NS ## buffer_start_with_size(B, FID) ? -1 : N ## _start(B); }\\\n"
        "static inline int N ## _start_as_typed_root(NS ## builder_t *B)\\\n"
        "{ return NS ## buffer_start(B, TFID) ? -1 : N ## _start(B); }\\\n"
        "static inline int N ## _start_as_typed_root_with_size(NS ## builder_t *B)\\\n"
        "{ return NS ## buffer_start_with_size(B, TFID) ? -1 : N ## _start(B); }\\\n"
        "static inline NS ## buffer_ref_t N ## _end_as_root(NS ## builder_t *B)\\\n"
        "{ return NS ## buffer_end(B, N ## _end(B)); }\\\n"
        "static inline NS ## buffer_ref_t N ## _end_as_typed_root(NS ## builder_t *B)\\\n"
        "{ return NS ## buffer_end(B, N ## _end(B)); }\\\n"
        /*
         * Unlike structs, we do no use flatcc_builder_create_buffer
         * because we would have to manage alignment, and we save very
         * little because tables require stack allocations in any case.
         */
        "static inline NS ## buffer_ref_t N ## _create_as_root(NS ## builder_t *B __ ## N ## _formal_args)\\\n"
        "{ if (NS ## buffer_start(B, FID)) return 0; return NS ## buffer_end(B, N ## _create(B __ ## N ## _call_args)); }\\\n"
        "static inline NS ## buffer_ref_t N ## _create_as_root_with_size(NS ## builder_t *B __ ## N ## _formal_args)\\\n"
        "{ if (NS ## buffer_start_with_size(B, FID)) return 0; return NS ## buffer_end(B, N ## _create(B __ ## N ## _call_args)); }\\\n"
        "static inline NS ## buffer_ref_t N ## _create_as_typed_root(NS ## builder_t *B __ ## N ## _formal_args)\\\n"
        "{ if (NS ## buffer_start(B, TFID)) return 0; return NS ## buffer_end(B, N ## _create(B __ ## N ## _call_args)); }\\\n"
        "static inline NS ## buffer_ref_t N ## _create_as_typed_root_with_size(NS ## builder_t *B __ ## N ## _formal_args)\\\n"
        "{ if (NS ## buffer_start_with_size(B, TFID)) return 0; return NS ## buffer_end(B, N ## _create(B __ ## N ## _call_args)); }\\\n"
        "static inline NS ## buffer_ref_t N ## _clone_as_root(NS ## builder_t *B, N ## _table_t t)\\\n"
        "{ if (NS ## buffer_start(B, FID)) return 0; return NS ## buffer_end(B, N ## _clone(B, t)); }\\\n"
        "static inline NS ## buffer_ref_t N ## _clone_as_root_with_size(NS ## builder_t *B, N ## _table_t t)\\\n"
        "{ if (NS ## buffer_start_with_size(B, FID)) return 0; return NS ## buffer_end(B, N ## _clone(B, t)); }\\\n"
        "static inline NS ## buffer_ref_t N ## _clone_as_typed_root(NS ## builder_t *B, N ## _table_t t)\\\n"
        "{ if (NS ## buffer_start(B, TFID)) return 0;return NS ## buffer_end(B, N ## _clone(B, t)); }\\\n"
        "static inline NS ## buffer_ref_t N ## _clone_as_typed_root_with_size(NS ## builder_t *B, N ## _table_t t)\\\n"
        "{ if (NS ## buffer_start_with_size(B, TFID)) return 0; return NS ## buffer_end(B, N ## _clone(B, t)); }\n"
        "\n",
        nsc);

    fprintf(out->fp,
        "#define __%sbuild_table_prolog(NS, N, FID, TFID)\\\n"
        "__%sbuild_table_vector_ops(NS, N ## _vec, N)\\\n"
        "__%sbuild_table_root(NS, N, FID, TFID)\n"
        "\n",
        nsc, nsc, nsc);


    fprintf(out->fp,
        "#define __%sbuild_struct_root(NS, N, A, FID, TFID)\\\n"
        "static inline N ## _t *N ## _start_as_root(NS ## builder_t *B)\\\n"
        "{ return NS ## buffer_start(B, FID) ? 0 : N ## _start(B); }\\\n"
        "static inline N ## _t *N ## _start_as_root_with_size(NS ## builder_t *B)\\\n"
        "{ return NS ## buffer_start_with_size(B, FID) ? 0 : N ## _start(B); }\\\n"
        "static inline N ## _t *N ## _start_as_typed_root(NS ## builder_t *B)\\\n"
        "{ return NS ## buffer_start(B, TFID) ? 0 : N ## _start(B); }\\\n"
        "static inline N ## _t *N ## _start_as_typed_root_with_size(NS ## builder_t *B)\\\n"
        "{ return NS ## buffer_start_with_size(B, TFID) ? 0 : N ## _start(B); }\\\n"
        "static inline NS ## buffer_ref_t N ## _end_as_root(NS ## builder_t *B)\\\n"
        "{ return NS ## buffer_end(B, N ## _end(B)); }\\\n"
        "static inline NS ## buffer_ref_t N ## _end_as_typed_root(NS ## builder_t *B)\\\n"
        "{ return NS ## buffer_end(B, N ## _end(B)); }\\\n"
        "static inline NS ## buffer_ref_t N ## _end_pe_as_root(NS ## builder_t *B)\\\n"
        "{ return NS ## buffer_end(B, N ## _end_pe(B)); }\\\n"
        "static inline NS ## buffer_ref_t N ## _end_pe_as_typed_root(NS ## builder_t *B)\\\n"
        "{ return NS ## buffer_end(B, N ## _end_pe(B)); }\\\n"
        "static inline NS ## buffer_ref_t N ## _create_as_root(NS ## builder_t *B __ ## N ## _formal_args)\\\n"
        "{ return flatcc_builder_create_buffer(B, FID, 0,\\\n"
        "  N ## _create(B __ ## N ## _call_args), A, 0); }\\\n"
        "static inline NS ## buffer_ref_t N ## _create_as_root_with_size(NS ## builder_t *B __ ## N ## _formal_args)\\\n"
        "{ return flatcc_builder_create_buffer(B, FID, 0,\\\n"
        "  N ## _create(B __ ## N ## _call_args), A, flatcc_builder_with_size); }\\\n"
        "static inline NS ## buffer_ref_t N ## _create_as_typed_root(NS ## builder_t *B __ ## N ## _formal_args)\\\n"
        "{ return flatcc_builder_create_buffer(B, TFID, 0,\\\n"
        "  N ## _create(B __ ## N ## _call_args), A, 0); }\\\n"
        "static inline NS ## buffer_ref_t N ## _create_as_typed_root_with_size(NS ## builder_t *B __ ## N ## _formal_args)\\\n"
        "{ return flatcc_builder_create_buffer(B, TFID, 0,\\\n"
        "  N ## _create(B __ ## N ## _call_args), A, flatcc_builder_with_size); }\\\n"
        "static inline NS ## buffer_ref_t N ## _clone_as_root(NS ## builder_t *B, N ## _struct_t p)\\\n"
        "{ return flatcc_builder_create_buffer(B, FID, 0, N ## _clone(B, p), A, 0); }\\\n"
        "static inline NS ## buffer_ref_t N ## _clone_as_root_with_size(NS ## builder_t *B, N ## _struct_t p)\\\n"
        "{ return flatcc_builder_create_buffer(B, FID, 0, N ## _clone(B, p), A, flatcc_builder_with_size); }\\\n"
        "static inline NS ## buffer_ref_t N ## _clone_as_typed_root(NS ## builder_t *B, N ## _struct_t p)\\\n"
        "{ return flatcc_builder_create_buffer(B, TFID, 0, N ## _clone(B, p), A, 0); }\\\n"
        "static inline NS ## buffer_ref_t N ## _clone_as_typed_root_with_size(NS ## builder_t *B, N ## _struct_t p)\\\n"
        "{ return flatcc_builder_create_buffer(B, TFID, 0, N ## _clone(B, p), A, flatcc_builder_with_size); }\n"
        "\n",
        nsc);

    fprintf(out->fp,
        "#define __%sbuild_nested_table_root(NS, N, TN, FID, TFID)\\\n"
        "static inline int N ## _start_as_root(NS ## builder_t *B)\\\n"
        "{ return NS ## buffer_start(B, FID) ? -1 : TN ## _start(B); }\\\n"
        "static inline int N ## _start_as_typed_root(NS ## builder_t *B)\\\n"
        "{ return NS ## buffer_start(B, TFID) ? -1 : TN ## _start(B); }\\\n"
        "static inline int N ## _end_as_root(NS ## builder_t *B)\\\n"
        "{ return N ## _add(B, NS ## buffer_end(B, TN ## _end(B))); }\\\n"
        "static inline int N ## _end_as_typed_root(NS ## builder_t *B)\\\n"
        "{ return N ## _add(B, NS ## buffer_end(B, TN ## _end(B))); }\\\n"
        "static inline int N ## _nest(NS ## builder_t *B, void *data, size_t size, uint16_t align)\\\n"
        "{ return N ## _add(B, flatcc_builder_create_vector(B, data, size, 1,\\\n"
        "  align ? align : 8, FLATBUFFERS_COUNT_MAX(1))); }\\\n"
        "static inline int N ## _typed_nest(NS ## builder_t *B, void *data, size_t size, uint16_t align)\\\n"
        "{ return N ## _add(B, flatcc_builder_create_vector(B, data, size, 1,\\\n"
        "  align ? align : 8, FLATBUFFERS_COUNT_MAX(1))); }\\\n"
        "static inline int N ## _clone_as_root(NS ## builder_t *B, TN ## _table_t t)\\\n"
        "{ return N ## _add(B, TN ## _clone_as_root(B, t)); }\\\n"
        "static inline int N ## _clone_as_typed_root(NS ## builder_t *B, TN ## _table_t t)\\\n"
        "{ return N ## _add(B, TN ## _clone_as_typed_root(B, t)); }\n"
        "\n",
        nsc);

    fprintf(out->fp,
        "#define __%sbuild_nested_struct_root(NS, N, TN, A, FID, TFID)\\\n"
        "static inline TN ## _t *N ## _start_as_root(NS ## builder_t *B)\\\n"
        "{ return NS ## buffer_start(B, FID) ? 0 : TN ## _start(B); }\\\n"
        "static inline TN ## _t *N ## _start_as_typed_root(NS ## builder_t *B)\\\n"
        "{ return NS ## buffer_start(B, FID) ? 0 : TN ## _start(B); }\\\n"
        "static inline int N ## _end_as_root(NS ## builder_t *B)\\\n"
        "{ return N ## _add(B, NS ## buffer_end(B, TN ## _end(B))); }\\\n"
        "static inline int N ## _end_as_typed_root(NS ## builder_t *B)\\\n"
        "{ return N ## _add(B, NS ## buffer_end(B, TN ## _end(B))); }\\\n"
        "static inline int N ## _end_pe_as_root(NS ## builder_t *B)\\\n"
        "{ return N ## _add(B, NS ## buffer_end(B, TN ## _end_pe(B))); }\\\n"
        "static inline int N ## _create_as_root(NS ## builder_t *B __ ## TN ## _formal_args)\\\n"
        "{ return N ## _add(B, flatcc_builder_create_buffer(B, FID, 0,\\\n"
        "  TN ## _create(B __ ## TN ## _call_args), A, flatcc_builder_is_nested)); }\\\n"
        "static inline int N ## _create_as_typed_root(NS ## builder_t *B __ ## TN ## _formal_args)\\\n"
        "{ return N ## _add(B, flatcc_builder_create_buffer(B, TFID, 0,\\\n"
        "  TN ## _create(B __ ## TN ## _call_args), A, flatcc_builder_is_nested)); }\\\n"
        "static inline int N ## _nest(NS ## builder_t *B, void *data, size_t size, uint16_t align)\\\n"
        "{ return N ## _add(B, flatcc_builder_create_vector(B, data, size, 1,\\\n"
        "  align < A ? A : align, FLATBUFFERS_COUNT_MAX(1))); }\\\n"
        "static inline int N ## _typed_nest(NS ## builder_t *B, void *data, size_t size, uint16_t align)\\\n"
        "{ return N ## _add(B, flatcc_builder_create_vector(B, data, size, 1,\\\n"
        "  align < A ? A : align, FLATBUFFERS_COUNT_MAX(1))); }\\\n"
        "static inline int N ## _clone_as_root(NS ## builder_t *B, TN ## _struct_t p)\\\n"
        "{ return N ## _add(B, TN ## _clone_as_root(B, p)); }\\\n"
        "static inline int N ## _clone_as_typed_root(NS ## builder_t *B, TN ## _struct_t p)\\\n"
        "{ return N ## _add(B, TN ## _clone_as_typed_root(B, p)); }\n"
        "\n",
        nsc);

    fprintf(out->fp,
        "#define __%sbuild_vector_ops(NS, V, N, TN, T)\\\n"
        "static inline T *V ## _extend(NS ## builder_t *B, size_t len)\\\n"
        "{ return (T *)flatcc_builder_extend_vector(B, len); }\\\n"
        "static inline T *V ## _append(NS ## builder_t *B, const T *data, size_t len)\\\n"
        "{ return (T *)flatcc_builder_append_vector(B, data, len); }\\\n"
        "static inline int V ## _truncate(NS ## builder_t *B, size_t len)\\\n"
        "{ return flatcc_builder_truncate_vector(B, len); }\\\n"
        "static inline T *V ## _edit(NS ## builder_t *B)\\\n"
        "{ return (T *)flatcc_builder_vector_edit(B); }\\\n"
        "static inline size_t V ## _reserved_len(NS ## builder_t *B)\\\n"
        "{ return flatcc_builder_vector_count(B); }\\\n"
        "static inline T *V ## _push(NS ## builder_t *B, const T *p)\\\n"
        "{ T *_p; return (_p = (T *)flatcc_builder_extend_vector(B, 1)) ? (memcpy(_p, p, TN ## __size()), _p) : 0; }\\\n"
        "static inline T *V ## _push_copy(NS ## builder_t *B, const T *p)\\\n"
        "{ T *_p; return (_p = (T *)flatcc_builder_extend_vector(B, 1)) ? TN ## _copy(_p, p) : 0; }\\\n"
        /* push_clone is the same as a for push_copy for scalar and struct vectors
         * but copy has different semantics as a standalone operation so we can't use
         * clone to implement push_clone - it would create a reference to a struct. */
        "static inline T *V ## _push_clone(NS ## builder_t *B, const T *p)\\\n"
        "{ T *_p; return (_p = (T *)flatcc_builder_extend_vector(B, 1)) ? TN ## _copy(_p, p) : 0; }\\\n"
        "static inline T *V ## _push_create(NS ## builder_t *B __ ## TN ## _formal_args)\\\n"
        "{ T *_p; return (_p = (T *)flatcc_builder_extend_vector(B, 1)) ? TN ## _assign(_p __ ## TN ## _call_args) : 0; }\n"
        "\n",
        nsc);

    fprintf(out->fp,
        /* NS: common namespace, N: typename, T: element type, S: elem size, A: alignment */
        "#define __%sbuild_vector(NS, N, T, S, A)\\\n"
        "typedef NS ## ref_t N ## _vec_ref_t;\\\n"
        "static inline int N ## _vec_start(NS ## builder_t *B)\\\n"
        "{ return flatcc_builder_start_vector(B, S, A, FLATBUFFERS_COUNT_MAX(S)); }\\\n"
        "static inline N ## _vec_ref_t N ## _vec_end_pe(NS ## builder_t *B)\\\n"
        "{ return flatcc_builder_end_vector(B); }\\\n"
        "static inline N ## _vec_ref_t N ## _vec_end(NS ## builder_t *B)\\\n"
        "{ if (!NS ## is_native_pe()) { size_t i, n; T *p = (T *)flatcc_builder_vector_edit(B);\\\n"
        "    for (i = 0, n = flatcc_builder_vector_count(B); i < n; ++i)\\\n"
        "    { N ## _to_pe(N ## __ptr_add(p, i)); }} return flatcc_builder_end_vector(B); }\\\n"
        "static inline N ## _vec_ref_t N ## _vec_create_pe(NS ## builder_t *B, const T *data, size_t len)\\\n"
        "{ return flatcc_builder_create_vector(B, data, len, S, A, FLATBUFFERS_COUNT_MAX(S)); }\\\n"
        "static inline N ## _vec_ref_t N ## _vec_create(NS ## builder_t *B, const T *data, size_t len)\\\n"
        "{ if (!NS ## is_native_pe()) { size_t i; T *p; int ret = flatcc_builder_start_vector(B, S, A, FLATBUFFERS_COUNT_MAX(S)); if (ret) { return ret; }\\\n"
        "  p = (T *)flatcc_builder_extend_vector(B, len); if (!p) return 0;\\\n"
        "  for (i = 0; i < len; ++i) { N ## _copy_to_pe(N ## __ptr_add(p, i), N ## __const_ptr_add(data, i)); }\\\n"
        "  return flatcc_builder_end_vector(B); } else return flatcc_builder_create_vector(B, data, len, S, A, FLATBUFFERS_COUNT_MAX(S)); }\\\n"
        "static inline N ## _vec_ref_t N ## _vec_clone(NS ## builder_t *B, N ##_vec_t vec)\\\n"
        "{ __%smemoize(B, vec, flatcc_builder_create_vector(B, vec, N ## _vec_len(vec), S, A, FLATBUFFERS_COUNT_MAX(S))); }\\\n"
        "static inline N ## _vec_ref_t N ## _vec_slice(NS ## builder_t *B, N ##_vec_t vec, size_t index, size_t len)\\\n"
        "{ size_t n = N ## _vec_len(vec); if (index >= n) index = n; n -= index; if (len > n) len = n;\\\n"
        "  return flatcc_builder_create_vector(B, N ## __const_ptr_add(vec, index), len, S, A, FLATBUFFERS_COUNT_MAX(S)); }\\\n"
        "__%sbuild_vector_ops(NS, N ## _vec, N, N, T)\n"
        "\n",
        nsc, nsc, nsc);

    fprintf(out->fp,
        "#define __%sbuild_union_vector_ops(NS, V, N, TN)\\\n"
        "static inline TN ## _union_ref_t *V ## _extend(NS ## builder_t *B, size_t len)\\\n"
        "{ return flatcc_builder_extend_union_vector(B, len); }\\\n"
        "static inline TN ## _union_ref_t *V ## _append(NS ## builder_t *B, const TN ## _union_ref_t *data, size_t len)\\\n"
        "{ return flatcc_builder_append_union_vector(B, data, len); }\\\n"
        "static inline int V ## _truncate(NS ## builder_t *B, size_t len)\\\n"
        "{ return flatcc_builder_truncate_union_vector(B, len); }\\\n"
        "static inline TN ## _union_ref_t *V ## _edit(NS ## builder_t *B)\\\n"
        "{ return (TN ## _union_ref_t *) flatcc_builder_union_vector_edit(B); }\\\n"
        "static inline size_t V ## _reserved_len(NS ## builder_t *B)\\\n"
        "{ return flatcc_builder_union_vector_count(B); }\\\n"
        "static inline TN ## _union_ref_t *V ## _push(NS ## builder_t *B, const TN ## _union_ref_t ref)\\\n"
        "{ return flatcc_builder_union_vector_push(B, ref); }\\\n"
        "static inline TN ## _union_ref_t *V ## _push_clone(NS ## builder_t *B, TN ## _union_t u)\\\n"
        "{ return TN ## _vec_push(B, TN ## _clone(B, u)); }\n"
        "\n",
        nsc);

    fprintf(out->fp,
        "#define __%sbuild_union_vector(NS, N)\\\n"
        "static inline int N ## _vec_start(NS ## builder_t *B)\\\n"
        "{ return flatcc_builder_start_union_vector(B); }\\\n"
        "static inline N ## _union_vec_ref_t N ## _vec_end(NS ## builder_t *B)\\\n"
        "{ return flatcc_builder_end_union_vector(B); }\\\n"
        "static inline N ## _union_vec_ref_t N ## _vec_create(NS ## builder_t *B, const N ## _union_ref_t *data, size_t len)\\\n"
        "{ return flatcc_builder_create_union_vector(B, data, len); }\\\n"
        "__%sbuild_union_vector_ops(NS, N ## _vec, N, N)\\\n"
        "/* Preserves DAG structure separately for type and value vector, so a type vector could be shared for many value vectors. */\\\n"
        "static inline N ## _union_vec_ref_t N ## _vec_clone(NS ## builder_t *B, N ##_union_vec_t vec)\\\n"
        "{ N ## _union_vec_ref_t _uvref, _ret = { 0, 0 }; NS ## union_ref_t _uref; size_t _i, _len;\\\n"
        "  if (vec.type == 0) return _ret;\\\n"
        "  _uvref.type = flatcc_builder_refmap_find(B, vec.type); _uvref.value = flatcc_builder_refmap_find(B, vec.value);\\\n"
        "  _len = N ## _union_vec_len(vec); if (_uvref.type == 0) {\\\n"
        "  _uvref.type = flatcc_builder_refmap_insert(B, vec.type, (flatcc_builder_create_type_vector(B, vec.type, _len))); }\\\n"
        "  if (_uvref.type == 0) { return _ret; } if (_uvref.value == 0) {\\\n"
        "  if (flatcc_builder_start_offset_vector(B)) return _ret;\\\n"
        "  for (_i = 0; _i < _len; ++_i) { _uref = N ## _clone(B, N ## _union_vec_at(vec, _i));\\\n"
        "    if (!_uref.value || !(flatcc_builder_offset_vector_push(B, _uref.value))) return _ret; }\\\n"
        "  _uvref.value = flatcc_builder_refmap_insert(B, vec.value, flatcc_builder_end_offset_vector(B));\\\n"
        "  if (_uvref.value == 0) return _ret; } return _uvref; }\n"
        "\n",
        nsc, nsc);

    /* In addtion to offset_vector_ops... */
    fprintf(out->fp,
        "#define __%sbuild_string_vector_ops(NS, N)\\\n"
        "static inline int N ## _push_start(NS ## builder_t *B)\\\n"
        "{ return NS ## string_start(B); }\\\n"
        "static inline NS ## string_ref_t *N ## _push_end(NS ## builder_t *B)\\\n"
        "{ return NS ## string_vec_push(B, NS ## string_end(B)); }\\\n"
        "static inline NS ## string_ref_t *N ## _push_create(NS ## builder_t *B, const char *s, size_t len)\\\n"
        "{ return NS ## string_vec_push(B, NS ## string_create(B, s, len)); }\\\n"
        "static inline NS ## string_ref_t *N ## _push_create_str(NS ## builder_t *B, const char *s)\\\n"
        "{ return NS ## string_vec_push(B, NS ## string_create_str(B, s)); }\\\n"
        "static inline NS ## string_ref_t *N ## _push_create_strn(NS ## builder_t *B, const char *s, size_t max_len)\\\n"
        "{ return NS ## string_vec_push(B, NS ## string_create_strn(B, s, max_len)); }\\\n"
        "static inline NS ## string_ref_t *N ## _push_clone(NS ## builder_t *B, NS ## string_t string)\\\n"
        "{ return NS ## string_vec_push(B, NS ## string_clone(B, string)); }\\\n"
        "static inline NS ## string_ref_t *N ## _push_slice(NS ## builder_t *B, NS ## string_t string, size_t index, size_t len)\\\n"
        "{ return NS ## string_vec_push(B, NS ## string_slice(B, string, index, len)); }\n"
        "\n",
        nsc);

    /* In addtion to offset_vector_ops... */
    fprintf(out->fp,
        "#define __%sbuild_table_vector_ops(NS, N, TN)\\\n"
        "static inline int N ## _push_start(NS ## builder_t *B)\\\n"
        "{ return TN ## _start(B); }\\\n"
        "static inline TN ## _ref_t *N ## _push_end(NS ## builder_t *B)\\\n"
        "{ return N ## _push(B, TN ## _end(B)); }\\\n"
        "static inline TN ## _ref_t *N ## _push_create(NS ## builder_t *B __ ## TN ##_formal_args)\\\n"
        "{ return N ## _push(B, TN ## _create(B __ ## TN ## _call_args)); }\n"
        "\n",
        nsc);

    fprintf(out->fp,
        "#define __%sbuild_offset_vector_ops(NS, V, N, TN)\\\n"
        "static inline TN ## _ref_t *V ## _extend(NS ## builder_t *B, size_t len)\\\n"
        "{ return flatcc_builder_extend_offset_vector(B, len); }\\\n"
        "static inline TN ## _ref_t *V ## _append(NS ## builder_t *B, const TN ## _ref_t *data, size_t len)\\\n"
        "{ return flatcc_builder_append_offset_vector(B, data, len); }\\\n"
        "static inline int V ## _truncate(NS ## builder_t *B, size_t len)\\\n"
        "{ return flatcc_builder_truncate_offset_vector(B, len); }\\\n"
        "static inline TN ## _ref_t *V ## _edit(NS ## builder_t *B)\\\n"
        "{ return (TN ## _ref_t *)flatcc_builder_offset_vector_edit(B); }\\\n"
        "static inline size_t V ## _reserved_len(NS ## builder_t *B)\\\n"
        "{ return flatcc_builder_offset_vector_count(B); }\\\n"
        "static inline TN ## _ref_t *V ## _push(NS ## builder_t *B, const TN ## _ref_t ref)\\\n"
        "{ return ref ? flatcc_builder_offset_vector_push(B, ref) : 0; }\n"
        "\n",
        nsc);

    fprintf(out->fp,
        "#define __%sbuild_offset_vector(NS, N)\\\n"
        "typedef NS ## ref_t N ## _vec_ref_t;\\\n"
        "static inline int N ## _vec_start(NS ## builder_t *B)\\\n"
        "{ return flatcc_builder_start_offset_vector(B); }\\\n"
        "static inline N ## _vec_ref_t N ## _vec_end(NS ## builder_t *B)\\\n"
        "{ return flatcc_builder_end_offset_vector(B); }\\\n"
        "static inline N ## _vec_ref_t N ## _vec_create(NS ## builder_t *B, const N ## _ref_t *data, size_t len)\\\n"
        "{ return flatcc_builder_create_offset_vector(B, data, len); }\\\n"
        "__%sbuild_offset_vector_ops(NS, N ## _vec, N, N)\\\n"
        "static inline N ## _vec_ref_t N ## _vec_clone(NS ## builder_t *B, N ##_vec_t vec)\\\n"
        "{ int _ret; N ## _ref_t _e; size_t _i, _len; __%smemoize_begin(B, vec);\\\n"
        " _len = N ## _vec_len(vec); if (flatcc_builder_start_offset_vector(B)) return 0;\\\n"
        "  for (_i = 0; _i < _len; ++_i) { if (!(_e = N ## _clone(B, N ## _vec_at(vec, _i)))) return 0;\\\n"
        "    if (!flatcc_builder_offset_vector_push(B, _e)) return 0; }\\\n"
        "  __%smemoize_end(B, vec, flatcc_builder_end_offset_vector(B)); }\\\n"
        "\n",
        nsc, nsc, nsc, nsc);

    fprintf(out->fp,
        "#define __%sbuild_string_ops(NS, N)\\\n"
        "static inline char *N ## _append(NS ## builder_t *B, const char *s, size_t len)\\\n"
        "{ return flatcc_builder_append_string(B, s, len); }\\\n"
        "static inline char *N ## _append_str(NS ## builder_t *B, const char *s)\\\n"
        "{ return flatcc_builder_append_string_str(B, s); }\\\n"
        "static inline char *N ## _append_strn(NS ## builder_t *B, const char *s, size_t len)\\\n"
        "{ return flatcc_builder_append_string_strn(B, s, len); }\\\n"
        "static inline size_t N ## _reserved_len(NS ## builder_t *B)\\\n"
        "{ return flatcc_builder_string_len(B); }\\\n"
        "static inline char *N ## _extend(NS ## builder_t *B, size_t len)\\\n"
        "{ return flatcc_builder_extend_string(B, len); }\\\n"
        "static inline char *N ## _edit(NS ## builder_t *B)\\\n"
        "{ return flatcc_builder_string_edit(B); }\\\n"
        "static inline int N ## _truncate(NS ## builder_t *B, size_t len)\\\n"
        "{ return flatcc_builder_truncate_string(B, len); }\n"
        "\n",
        nsc);

    fprintf(out->fp,
        "#define __%sbuild_string(NS)\\\n"
        "typedef NS ## ref_t NS ## string_ref_t;\\\n"
        "static inline int NS ## string_start(NS ## builder_t *B)\\\n"
        "{ return flatcc_builder_start_string(B); }\\\n"
        "static inline NS ## string_ref_t NS ## string_end(NS ## builder_t *B)\\\n"
        "{ return flatcc_builder_end_string(B); }\\\n"
        "static inline NS ## ref_t NS ## string_create(NS ## builder_t *B, const char *s, size_t len)\\\n"
        "{ return flatcc_builder_create_string(B, s, len); }\\\n"
        "static inline NS ## ref_t NS ## string_create_str(NS ## builder_t *B, const char *s)\\\n"
        "{ return flatcc_builder_create_string_str(B, s); }\\\n"
        "static inline NS ## ref_t NS ## string_create_strn(NS ## builder_t *B, const char *s, size_t len)\\\n"
        "{ return flatcc_builder_create_string_strn(B, s, len); }\\\n"
        "static inline NS ## string_ref_t NS ## string_clone(NS ## builder_t *B, NS ## string_t string)\\\n"
        "{ __%smemoize(B, string, flatcc_builder_create_string(B, string, NS ## string_len(string))); }\\\n"
        "static inline NS ## string_ref_t NS ## string_slice(NS ## builder_t *B, NS ## string_t string, size_t index, size_t len)\\\n"
        "{ size_t n = NS ## string_len(string); if (index >= n) index = n; n -= index; if (len > n) len = n;\\\n"
        "  return flatcc_builder_create_string(B, string + index, len); }\\\n"
        "__%sbuild_string_ops(NS, NS ## string)\\\n"
        "__%sbuild_offset_vector(NS, NS ## string)\n"
        "\n",
        nsc, nsc, nsc, nsc);
    fprintf(out->fp,
        "#define __%scopy_from_pe(P, P2, N) (*(P) = N ## _read_from_pe(P2), (P))\n"
        "#define __%sfrom_pe(P, N) (*(P) = N ## _read_from_pe(P), (P))\n"
        "#define __%scopy_to_pe(P, P2, N) (N ## _write_to_pe((P), *(P2)), (P))\n"
        "#define __%sto_pe(P, N) (N ## _write_to_pe((P), *(P)), (P))\n",
        nsc, nsc, nsc, nsc);
    fprintf(out->fp,
        "#define __%sdefine_fixed_array_primitives(NS, N, T)\\\n"
        "static inline T *N ## _array_copy(T *p, const T *p2, size_t n)\\\n"
        "{ memcpy(p, p2, n * sizeof(T)); return p; }\\\n"
        "static inline T *N ## _array_copy_from_pe(T *p, const T *p2, size_t n)\\\n"
        "{ size_t i; if (NS ## is_native_pe()) memcpy(p, p2, n * sizeof(T)); else {\\\n"
        "  for (i = 0; i < n; ++i) N ## _copy_from_pe(&p[i], &p2[i]); } return p; }\\\n"
        "static inline T *N ## _array_copy_to_pe(T *p, const T *p2, size_t n)\\\n"
        "{ size_t i; if (NS ## is_native_pe()) memcpy(p, p2, n * sizeof(T)); else {\\\n"
        "  for (i = 0; i < n; ++i) N ## _copy_to_pe(&p[i], &p2[i]); } return p; }\n",
        nsc);
    fprintf(out->fp,
        "#define __%sdefine_scalar_primitives(NS, N, T)\\\n"
        "static inline T *N ## _from_pe(T *p) { return __ ## NS ## from_pe(p, N); }\\\n"
        "static inline T *N ## _to_pe(T *p) { return __ ## NS ## to_pe(p, N); }\\\n"
        "static inline T *N ## _copy(T *p, const T *p2) { *p = *p2; return p; }\\\n"
        "static inline T *N ## _copy_from_pe(T *p, const T *p2)\\\n"
        "{ return __ ## NS ## copy_from_pe(p, p2, N); }\\\n"
        "static inline T *N ## _copy_to_pe(T *p, const T *p2) \\\n"
        "{ return __ ## NS ## copy_to_pe(p, p2, N); }\\\n"
        "static inline T *N ## _assign(T *p, const T v0) { *p = v0; return p; }\\\n"
        "static inline T *N ## _assign_from_pe(T *p, T v0)\\\n"
        "{ *p = N ## _read_from_pe(&v0); return p; }\\\n"
        "static inline T *N ## _assign_to_pe(T *p, T v0)\\\n"
        "{ N ## _write_to_pe(p, v0); return p; }\n"
        "#define __%sbuild_scalar(NS, N, T)\\\n"
        "__ ## NS ## define_scalar_primitives(NS, N, T)\\\n"
        "__ ## NS ## define_fixed_array_primitives(NS, N, T)\\\n"
        "__ ## NS ## build_vector(NS, N, T, sizeof(T), sizeof(T))\n",
        nsc, nsc);

    fprintf(out->fp,
        "/* Depends on generated copy_to/from_pe functions, and the type. */\n"
        "#define __%sdefine_struct_primitives(NS, N)\\\n"
        "static inline N ## _t *N ##_to_pe(N ## _t *p)\\\n"
        "{ if (!NS ## is_native_pe()) { N ## _copy_to_pe(p, p); }; return p; }\\\n"
        "static inline N ## _t *N ##_from_pe(N ## _t *p)\\\n"
        "{ if (!NS ## is_native_pe()) { N ## _copy_from_pe(p, p); }; return p; }\\\n"
        "static inline N ## _t *N ## _clear(N ## _t *p) { return (N ## _t *)memset(p, 0, N ## __size()); }\n"
        "\n"

        /*
         * NOTE: structs can both be inline and independent blocks. They
         * are independent as buffer roots, and also as union members.
         * _clone applied to a struct type name creates a reference to
         * an independent block, but this is ambigous. Structs also
         * support _copy which is the inline equivalent of _clone for
         * inline. There is also the distinction between _clone applied
         * to a field name, clone applied to a type name, and _clone
         * applied to a _vec_push operation. For field names and push
         * operations, _clone is unambigiously inline and similar to
         * _copy. So the ambigiouty is when applying _clone to a type
         * name where _copy and _clone are different. Unions can safely
         * implement clone on structs members via _clone because union
         * members are indendendent blocks whereas push_clone must be
         * implemented with _copy because structs are inline in
         * (non-union) vectors. Structs in union-vectors are independent
         * but these simply the unions clone operation (which is a
         * generated function).
         */
        "/* Depends on generated copy/assign_to/from_pe functions, and the type. */\n"
        "#define __%sbuild_struct(NS, N, S, A, FID, TFID)\\\n"
        "__ ## NS ## define_struct_primitives(NS, N)\\\n"
        "typedef NS ## ref_t N ## _ref_t;\\\n"
        "static inline N ## _t *N ## _start(NS ## builder_t *B)\\\n"
        "{ return (N ## _t *)flatcc_builder_start_struct(B, S, A); }\\\n"
        "static inline N ## _ref_t N ## _end(NS ## builder_t *B)\\\n"
        "{ if (!NS ## is_native_pe()) { N ## _to_pe((N ## _t *)flatcc_builder_struct_edit(B)); }\\\n"
        "  return flatcc_builder_end_struct(B); }\\\n"
        "static inline N ## _ref_t N ## _end_pe(NS ## builder_t *B)\\\n"
        "{ return flatcc_builder_end_struct(B); }\\\n"
        "static inline N ## _ref_t N ## _create(NS ## builder_t *B __ ## N ## _formal_args)\\\n"
        "{ N ## _t *_p = N ## _start(B); if (!_p) return 0; N ##_assign_to_pe(_p __ ## N ## _call_args);\\\n"
        "  return N ## _end_pe(B); }\\\n"
        "static inline N ## _ref_t N ## _clone(NS ## builder_t *B, N ## _struct_t p)\\\n"
        "{ N ## _t *_p; __%smemoize_begin(B, p); _p = N ## _start(B); if (!_p) return 0;\\\n"
        "  N ## _copy(_p, p); __%smemoize_end(B, p, N ##_end_pe(B)); }\\\n"
        "__%sbuild_vector(NS, N, N ## _t, S, A)\\\n"
        "__%sbuild_struct_root(NS, N, A, FID, TFID)\\\n"
        "\n",
        nsc, nsc, nsc, nsc, nsc, nsc);
    fprintf(out->fp,
        "#define __%sstruct_clear_field(p) memset((p), 0, sizeof(*(p)))\n",
        nsc);

    fprintf(out->fp,
        "#define __%sbuild_table(NS, N, K)\\\n"
        "static inline int N ## _start(NS ## builder_t *B)\\\n"
        "{ return flatcc_builder_start_table(B, K); }\\\n"
        "static inline N ## _ref_t N ## _end(NS ## builder_t *B)\\\n"
        "{ FLATCC_ASSERT(flatcc_builder_check_required(B, __ ## N ## _required,\\\n"
        "  sizeof(__ ## N ## _required) / sizeof(__ ## N ## _required[0]) - 1));\\\n"
        "  return flatcc_builder_end_table(B); }\\\n"
        "__%sbuild_offset_vector(NS, N)\n"
        "\n",
        nsc, nsc);

    fprintf(out->fp,
        "#define __%sbuild_table_field(ID, NS, N, TN, TT)\\\n"
        "static inline int N ## _add(NS ## builder_t *B, TN ## _ref_t ref)\\\n"
        "{ TN ## _ref_t *_p; return (ref && (_p = flatcc_builder_table_add_offset(B, ID))) ?\\\n"
        "  ((*_p = ref), 0) : -1; }\\\n"
        "static inline int N ## _start(NS ## builder_t *B)\\\n"
        "{ return TN ## _start(B); }\\\n"
        "static inline int N ## _end(NS ## builder_t *B)\\\n"
        "{ return N ## _add(B, TN ## _end(B)); }\\\n"
        "static inline TN ## _ref_t N ## _create(NS ## builder_t *B __ ## TN ##_formal_args)\\\n"
        "{ return N ## _add(B, TN ## _create(B __ ## TN ## _call_args)); }\\\n"
        "static inline int N ## _clone(NS ## builder_t *B, TN ## _table_t p)\\\n"
        "{ return N ## _add(B, TN ## _clone(B, p)); }\\\n"
        "static inline int N ## _pick(NS ## builder_t *B, TT ## _table_t t)\\\n"
        "{ TN ## _table_t _p = N ## _get(t); return _p ? N ## _clone(B, _p) : 0; }\n"
        "\n",
        nsc);

    fprintf(out->fp,
        "#define __%sbuild_union_field(ID, NS, N, TN, TT)\\\n"
        "static inline int N ## _add(NS ## builder_t *B, TN ## _union_ref_t uref)\\\n"
        "{ NS ## ref_t *_p; TN ## _union_type_t *_pt; if (uref.type == TN ## _NONE) return 0; if (uref.value == 0) return -1;\\\n"
        "  if (!(_pt = (TN ## _union_type_t *)flatcc_builder_table_add(B, ID - 1, sizeof(*_pt), sizeof(*_pt)))) return -1;\\\n"
        "  *_pt = uref.type; if (!(_p = flatcc_builder_table_add_offset(B, ID))) return -1; *_p = uref.value; return 0; }\\\n"
        "static inline int N ## _add_type(NS ## builder_t *B, TN ## _union_type_t type)\\\n"
        "{ TN ## _union_type_t *_pt; if (type == TN ## _NONE) return 0; return (_pt = (TN ## _union_type_t *)flatcc_builder_table_add(B, ID - 1,\\\n"
        "  sizeof(*_pt), sizeof(*_pt))) ? ((*_pt = type), 0) : -1; }\\\n"
        "static inline int N ## _add_value(NS ## builder_t *B, TN ## _union_ref_t uref)\\\n"
        "{ NS ## ref_t *p; if (uref.type == TN ## _NONE) return 0; return (p = flatcc_builder_table_add_offset(B, ID)) ?\\\n"
        "  ((*p = uref.value), 0) : -1; }\\\n"
        "static inline int N ## _clone(NS ## builder_t *B, TN ## _union_t p)\\\n"
        "{ return N ## _add(B, TN ## _clone(B, p)); }\\\n"
        /* `_pick` is not supported on specific union members because the source dictates the type. */
        "static inline int N ## _pick(NS ## builder_t *B, TT ## _table_t t)\\\n"
        "{ TN ## _union_t _p = N ## _union(t); return _p.type ? N ## _clone(B, _p) : 0; }\n"
        "\n",
        nsc);

    fprintf(out->fp,
        "/* M is the union value name and T is its type, i.e. the qualified name. */\n"
        "#define __%sbuild_union_table_value_field(NS, N, NU, M, T)\\\n"
        "static inline int N ## _ ## M ## _add(NS ## builder_t *B, T ## _ref_t ref)\\\n"
        "{ return N ## _add(B, NU ## _as_ ## M (ref)); }\\\n"
        "static inline int N ## _ ## M ## _start(NS ## builder_t *B)\\\n"
        "{ return T ## _start(B); }\\\n"
        "static inline int N ## _ ## M ## _end(NS ## builder_t *B)\\\n"
        "{ T ## _ref_t ref = T ## _end(B);\\\n"
        "  return ref ? N ## _ ## M ## _add(B, ref) : -1; }\\\n"
        "static inline int N ## _ ## M ## _create(NS ## builder_t *B __ ## T ##_formal_args)\\\n"
        "{ T ## _ref_t ref = T ## _create(B __ ## T ## _call_args);\\\n"
        "  return ref ? N ## _add(B, NU ## _as_ ## M(ref)) : -1; }\\\n"
        "static inline int N ## _ ## M ## _clone(NS ## builder_t *B, T ## _table_t t)\\\n"
        "{ T ## _ref_t ref = T ## _clone(B, t);\\\n"
        "  return ref ? N ## _add(B, NU ## _as_ ## M(ref)) : -1; }\n"
        "\n",
        nsc);

    fprintf(out->fp,
        "/* M is the union value name and T is its type, i.e. the qualified name. */\n"
        "#define __%sbuild_union_struct_value_field(NS, N, NU, M, T)\\\n"
        "static inline int N ## _ ## M ## _add(NS ## builder_t *B, T ## _ref_t ref)\\\n"
        "{ return N ## _add(B, NU ## _as_ ## M (ref)); }\\\n"
        "static inline T ## _t *N ## _ ## M ## _start(NS ## builder_t *B)\\\n"
        "{ return T ## _start(B); }\\\n"
        "static inline int N ## _ ## M ## _end(NS ## builder_t *B)\\\n"
        "{ T ## _ref_t ref = T ## _end(B);\\\n"
        "  return ref ? N ## _ ## M ## _add(B, ref) : -1; }\\\n"
        "static inline int N ## _ ## M ## _create(NS ## builder_t *B __ ## T ##_formal_args)\\\n"
        "{ T ## _ref_t ref = T ## _create(B __ ## T ## _call_args);\\\n"
        "  return ref ? N ## _add(B, NU ## _as_ ## M(ref)) : -1; }\\\n"
        "static inline int N ## _ ## M ## _end_pe(NS ## builder_t *B)\\\n"
        "{ T ## _ref_t ref = T ## _end_pe(B);\\\n"
        "  return ref ? N ## _add(B, NU ## _as_ ## M(ref)) : -1; }\\\n"
        "static inline int N ## _ ## M ## _clone(NS ## builder_t *B, T ## _struct_t p)\\\n"
        "{ T ## _ref_t ref = T ## _clone(B, p);\\\n"
        "  return ref ? N ## _add(B, NU ## _as_ ## M(ref)) : -1; }\n",
        nsc);

    fprintf(out->fp,
        "#define __%sbuild_union_string_value_field(NS, N, NU, M)\\\n"
        "static inline int N ## _ ## M ## _add(NS ## builder_t *B, NS ## string_ref_t ref)\\\n"
        "{ return N ## _add(B, NU ## _as_ ## M (ref)); }\\\n"
        "__%sbuild_string_field_ops(NS, N ## _ ## M)\n"
        "\n",
        nsc, nsc);

    fprintf(out->fp,
        "/* NS: common namespace, ID: table field id (not offset), TN: name of type T, TT: name of table type\n"
        " * S: sizeof of scalar type, A: alignment of type T, default value V of type T. */\n"
        "#define __%sbuild_scalar_field(ID, NS, N, TN, T, S, A, V, TT)\\\n"
        "static inline int N ## _add(NS ## builder_t *B, const T v)\\\n"
        "{ T *_p; if (v == V) return 0; if (!(_p = (T *)flatcc_builder_table_add(B, ID, S, A))) return -1;\\\n"
        "  TN ## _assign_to_pe(_p, v); return 0; }\\\n"
        "static inline int N ## _force_add(NS ## builder_t *B, const T v)\\\n"
        "{ T *_p; if (!(_p = (T *)flatcc_builder_table_add(B, ID, S, A))) return -1;\\\n"
        "  TN ## _assign_to_pe(_p, v); return 0; }\\\n"
        "/* Clone does not skip default values and expects pe endian content. */\\\n"
        "static inline int N ## _clone(NS ## builder_t *B, const T *p)\\\n"
        "{ return 0 == flatcc_builder_table_add_copy(B, ID, p, S, A) ? -1 : 0; }\\\n"
        "/* Transferring a missing field is a nop success with 0 as result. */\\\n"
        "static inline int N ## _pick(NS ## builder_t *B, TT ## _table_t t)\\\n"
        "{ const T *_p = N ## _get_ptr(t); return _p ? N ## _clone(B, _p) : 0; }\n"
        "\n",
        nsc);

    fprintf(out->fp,
        "/* NS: common namespace, ID: table field id (not offset), TN: name of type T, TT: name of table type\n"
        " * S: sizeof of scalar type, A: alignment of type T. */\n"
        "#define __%sbuild_scalar_optional_field(ID, NS, N, TN, T, S, A, TT)\\\n"
        "static inline int N ## _add(NS ## builder_t *B, const T v)\\\n"
        "{ T *_p; if (!(_p = (T *)flatcc_builder_table_add(B, ID, S, A))) return -1;\\\n"
        "  TN ## _assign_to_pe(_p, v); return 0; }\\\n"
        "/* Clone does not skip default values and expects pe endian content. */\\\n"
        "static inline int N ## _clone(NS ## builder_t *B, const T *p)\\\n"
        "{ return 0 == flatcc_builder_table_add_copy(B, ID, p, S, A) ? -1 : 0; }\\\n"
        "/* Transferring a missing field is a nop success with 0 as result. */\\\n"
        "static inline int N ## _pick(NS ## builder_t *B, TT ## _table_t t)\\\n"
        "{ const T *_p = N ## _get_ptr(t); return _p ? N ## _clone(B, _p) : 0; }\n"
        "\n",
        nsc);

    fprintf(out->fp,
        "#define __%sbuild_struct_field(ID, NS, N, TN, S, A, TT)\\\n"
        "static inline TN ## _t *N ## _start(NS ## builder_t *B)\\\n"
        "{ return (TN ## _t *)flatcc_builder_table_add(B, ID, S, A); }\\\n"
        "static inline int N ## _end(NS ## builder_t *B)\\\n"
        "{ if (!NS ## is_native_pe()) { TN ## _to_pe((TN ## _t *)flatcc_builder_table_edit(B, S)); } return 0; }\\\n"
        "static inline int N ## _end_pe(NS ## builder_t *B) { return 0; }\\\n"
        "static inline int N ## _create(NS ## builder_t *B __ ## TN ## _formal_args)\\\n"
        "{ TN ## _t *_p = N ## _start(B); if (!_p) return -1; TN ##_assign_to_pe(_p __ ## TN ## _call_args);\\\n"
        "  return 0; }\\\n"
        "static inline int N ## _add(NS ## builder_t *B, const TN ## _t *p)\\\n"
        "{ TN ## _t *_p = N ## _start(B); if (!_p) return -1; TN ##_copy_to_pe(_p, p); return 0; }\\\n"
        "static inline int N ## _clone(NS ## builder_t *B, TN ## _struct_t p)\\\n"
        "{ return 0 == flatcc_builder_table_add_copy(B, ID, p, S, A) ? -1 : 0; }\\\n"
        "static inline int N ## _pick(NS ## builder_t *B, TT ## _table_t t)\\\n"
        "{ TN ## _struct_t _p = N ## _get(t); return _p ? N ## _clone(B, _p) : 0; }\n"
        "\n",
        nsc);

    /* This goes for scalar, struct, and enum vectors. */
    fprintf(out->fp,
        "#define __%sbuild_vector_field(ID, NS, N, TN, T, TT)\\\n"
        "static inline int N ## _add(NS ## builder_t *B, TN ## _vec_ref_t ref)\\\n"
        "{ TN ## _vec_ref_t *_p; return (ref && (_p = flatcc_builder_table_add_offset(B, ID))) ? ((*_p = ref), 0) : -1; }\\\n"
        "static inline int N ## _start(NS ## builder_t *B)\\\n"
        "{ return TN ## _vec_start(B); }\\\n"
        "static inline int N ## _end_pe(NS ## builder_t *B)\\\n"
        "{ return N ## _add(B, TN ## _vec_end_pe(B)); }\\\n"
        "static inline int N ## _end(NS ## builder_t *B)\\\n"
        "{ return N ## _add(B, TN ## _vec_end(B)); }\\\n"
        "static inline int N ## _create_pe(NS ## builder_t *B, const T *data, size_t len)\\\n"
        "{ return N ## _add(B, TN ## _vec_create_pe(B, data, len)); }\\\n"
        "static inline int N ## _create(NS ## builder_t *B, const T *data, size_t len)\\\n"
        "{ return N ## _add(B, TN ## _vec_create(B, data, len)); }\\\n"
        "static inline int N ## _slice(NS ## builder_t *B, TN ## _vec_t vec, size_t index, size_t len)\\\n"
        "{ return N ## _add(B, TN ## _vec_slice(B, vec, index, len)); }\\\n"
        "static inline int N ## _clone(NS ## builder_t *B, TN ## _vec_t vec)\\\n"
        "{ return N ## _add(B, TN ## _vec_clone(B, vec)); }\\\n"
        "static inline int N ## _pick(NS ## builder_t *B, TT ## _table_t t)\\\n"
        "{ TN ## _vec_t _p = N ## _get(t); return _p ? N ## _clone(B, _p) : 0; }\\\n"
        "__%sbuild_vector_ops(NS, N, N, TN, T)\\\n"
        "\n",
        nsc, nsc);

    fprintf(out->fp,
        "#define __%sbuild_offset_vector_field(ID, NS, N, TN, TT)\\\n"
        "static inline int N ## _add(NS ## builder_t *B, TN ## _vec_ref_t ref)\\\n"
        "{ TN ## _vec_ref_t *_p; return (ref && (_p = flatcc_builder_table_add_offset(B, ID))) ? ((*_p = ref), 0) : -1; }\\\n"
        "static inline int N ## _start(NS ## builder_t *B)\\\n"
        "{ return flatcc_builder_start_offset_vector(B); }\\\n"
        "static inline int N ## _end(NS ## builder_t *B)\\\n"
        "{ return N ## _add(B, flatcc_builder_end_offset_vector(B)); }\\\n"
        "static inline int N ## _create(NS ## builder_t *B, const TN ## _ref_t *data, size_t len)\\\n"
        "{ return N ## _add(B, flatcc_builder_create_offset_vector(B, data, len)); }\\\n"
        "__%sbuild_offset_vector_ops(NS, N, N, TN)\\\n"
        "static inline int N ## _clone(NS ## builder_t *B, TN ## _vec_t vec)\\\n"
        "{ return N ## _add(B, TN ## _vec_clone(B, vec)); }\\\n"
        "static inline int N ## _pick(NS ## builder_t *B, TT ## _table_t t)\\\n"
        "{ TN ## _vec_t _p = N ## _get(t); return _p ? N ## _clone(B, _p) : 0; }\n"
        "\n",
        nsc, nsc);

    fprintf(out->fp,
        "/* depends on N ## _add which differs for union member fields and ordinary fields */\\\n"
        "#define __%sbuild_string_field_ops(NS, N)\\\n"
        "static inline int N ## _start(NS ## builder_t *B)\\\n"
        "{ return flatcc_builder_start_string(B); }\\\n"
        "static inline int N ## _end(NS ## builder_t *B)\\\n"
        "{ return N ## _add(B, flatcc_builder_end_string(B)); }\\\n"
        "static inline int N ## _create(NS ## builder_t *B, const char *s, size_t len)\\\n"
        "{ return N ## _add(B, flatcc_builder_create_string(B, s, len)); }\\\n"
        "static inline int N ## _create_str(NS ## builder_t *B, const char *s)\\\n"
        "{ return N ## _add(B, flatcc_builder_create_string_str(B, s)); }\\\n"
        "static inline int N ## _create_strn(NS ## builder_t *B, const char *s, size_t max_len)\\\n"
        "{ return N ## _add(B, flatcc_builder_create_string_strn(B, s, max_len)); }\\\n"
        "static inline int N ## _clone(NS ## builder_t *B, NS ## string_t string)\\\n"
        "{ return N ## _add(B, NS ## string_clone(B, string)); }\\\n"
        "static inline int N ## _slice(NS ## builder_t *B, NS ## string_t string, size_t index, size_t len)\\\n"
        "{ return N ## _add(B, NS ## string_slice(B, string, index, len)); }\\\n"
        "__%sbuild_string_ops(NS, N)\n"
        "\n",
        nsc, nsc);

    fprintf(out->fp,
        "#define __%sbuild_string_field(ID, NS, N, TT)\\\n"
        "static inline int N ## _add(NS ## builder_t *B, NS ## string_ref_t ref)\\\n"
        "{ NS ## string_ref_t *_p; return (ref && (_p = flatcc_builder_table_add_offset(B, ID))) ? ((*_p = ref), 0) : -1; }\\\n"
        "__%sbuild_string_field_ops(NS, N)\\\n"
        "static inline int N ## _pick(NS ## builder_t *B, TT ## _table_t t)\\\n"
        "{ NS ## string_t _p = N ## _get(t); return _p ? N ## _clone(B, _p) : 0; }\n"
        "\n",
        nsc, nsc);

    fprintf(out->fp,
        "#define __%sbuild_table_vector_field(ID, NS, N, TN, TT)\\\n"
        "__%sbuild_offset_vector_field(ID, NS, N, TN, TT)\\\n"
        "__%sbuild_table_vector_ops(NS, N, TN)\n"
        "\n",
        nsc, nsc, nsc);

    fprintf(out->fp,
        "#define __%sbuild_union_vector_field(ID, NS, N, TN, TT)\\\n"
        "static inline int N ## _add(NS ## builder_t *B, TN ## _union_vec_ref_t uvref)\\\n"
        "{ NS ## vec_ref_t *_p; if (!uvref.type || !uvref.value) return uvref.type == uvref.value ? 0 : -1;\\\n"
        "  if (!(_p = flatcc_builder_table_add_offset(B, ID - 1))) return -1; *_p = uvref.type;\\\n"
        "  if (!(_p = flatcc_builder_table_add_offset(B, ID))) return -1; *_p = uvref.value; return 0; }\\\n"
        "static inline int N ## _start(NS ## builder_t *B)\\\n"
        "{ return flatcc_builder_start_union_vector(B); }\\\n"
        "static inline int N ## _end(NS ## builder_t *B)\\\n"
        "{ return N ## _add(B, flatcc_builder_end_union_vector(B)); }\\\n"
        "static inline int N ## _create(NS ## builder_t *B, const TN ## _union_ref_t *data, size_t len)\\\n"
        "{ return N ## _add(B, flatcc_builder_create_union_vector(B, data, len)); }\\\n"
        "__%sbuild_union_vector_ops(NS, N, N, TN)\\\n"
        "static inline int N ## _clone(NS ## builder_t *B, TN ## _union_vec_t vec)\\\n"
        "{ return N ## _add(B, TN ## _vec_clone(B, vec)); }\\\n"
        "static inline int N ## _pick(NS ## builder_t *B, TT ## _table_t t)\\\n"
        "{ TN ## _union_vec_t _p = N ## _union(t); return _p.type ? N ## _clone(B, _p) : 0; }\n"
        "\n",
        nsc, nsc);

    fprintf(out->fp,
        "#define __%sbuild_union_table_vector_value_field(NS, N, NU, M, T)\\\n"
        "static inline int N ## _ ## M ## _push_start(NS ## builder_t *B)\\\n"
        "{ return T ## _start(B); }\\\n"
        "static inline NU ## _union_ref_t *N ## _ ## M ## _push_end(NS ## builder_t *B)\\\n"
        "{ return NU ## _vec_push(B, NU ## _as_ ## M (T ## _end(B))); }\\\n"
        "static inline NU ## _union_ref_t *N ## _ ## M ## _push(NS ## builder_t *B, T ## _ref_t ref)\\\n"
        "{ return NU ## _vec_push(B, NU ## _as_ ## M (ref)); }\\\n"
        "static inline NU ## _union_ref_t *N ## _ ## M ## _push_create(NS ## builder_t *B __ ## T ##_formal_args)\\\n"
        "{ return NU ## _vec_push(B, NU ## _as_ ## M(T ## _create(B __ ## T ## _call_args))); }\\\n"
        "static inline NU ## _union_ref_t *N ## _ ## M ## _push_clone(NS ## builder_t *B, T ## _table_t t)\\\n"
        "{ return NU ## _vec_push(B, NU ## _as_ ## M(T ## _clone(B, t))); }\n"
        "\n",
        nsc);

    fprintf(out->fp,
        "#define __%sbuild_union_struct_vector_value_field(NS, N, NU, M, T)\\\n"
        "static inline T ## _t *N ## _ ## M ## _push_start(NS ## builder_t *B)\\\n"
        "{ return T ## _start(B); }\\\n"
        "static inline NU ## _union_ref_t *N ## _ ## M ## _push_end(NS ## builder_t *B)\\\n"
        "{ return NU ## _vec_push(B, NU ## _as_ ## M (T ## _end(B))); }\\\n"
        "static inline NU ## _union_ref_t *N ## _ ## M ## _push(NS ## builder_t *B, T ## _ref_t ref)\\\n"
        "{ return NU ## _vec_push(B, NU ## _as_ ## M (ref)); }\\\n"
        "static inline NU ## _union_ref_t *N ## _ ## M ## _push_create(NS ## builder_t *B __ ## T ##_formal_args)\\\n"
        "{ return NU ## _vec_push(B, NU ## _as_ ## M(T ## _create(B __ ## T ## _call_args))); }\\\n"
        "static inline NU ## _union_ref_t *N ## _ ## M ## _push_clone(NS ## builder_t *B, T ## _struct_t p)\\\n"
        /* Here we create an independent struct block, so T ## _clone is appropriate as opposed to T ## _copy. */
        "{ return NU ## _vec_push(B, NU ## _as_ ## M(T ## _clone(B, p))); }\n"
        "\n",
        nsc);

    fprintf(out->fp,
        "#define __%sbuild_union_string_vector_value_field(NS, N, NU, M)\\\n"
        "static inline NU ## _union_ref_t *N ## _ ## M ## _push(NS ## builder_t *B, NS ## string_ref_t ref)\\\n"
        "{ return NU ## _vec_push(B, NU ## _as_ ## M (ref)); }\\\n"
        "static inline int N ## _ ## M ## _push_start(NS ## builder_t *B)\\\n"
        "{ return NS ## string_start(B); }\\\n"
        "static inline NU ## _union_ref_t *N ## _ ## M ## _push_end(NS ## builder_t *B)\\\n"
        "{ return NU ## _vec_push(B, NU ## _as_ ## M(NS ## string_end(B))); }\\\n"
        "static inline NU ## _union_ref_t *N ## _ ## M ## _push_create(NS ## builder_t *B, const char *s, size_t len)\\\n"
        "{ return NU ## _vec_push(B, NU ## _as_ ## M(NS ## string_create(B, s, len))); }\\\n"
        "static inline NU ## _union_ref_t *N ## _ ## M ## _push_create_str(NS ## builder_t *B, const char *s)\\\n"
        "{ return NU ## _vec_push(B, NU ## _as_ ## M(NS ## string_create_str(B, s))); }\\\n"
        "static inline NU ## _union_ref_t *N ## _ ## M ## _push_create_strn(NS ## builder_t *B, const char *s, size_t max_len)\\\n"
        "{ return NU ## _vec_push(B, NU ## _as_ ## M(NS ## string_create_strn(B, s, max_len))); }\\\n"
        "static inline NU ## _union_ref_t *N ## _ ## M ## _push_clone(NS ## builder_t *B, NS ## string_t string)\\\n"
        "{ return NU ## _vec_push(B, NU ## _as_ ## M(NS ## string_clone(B, string))); }\\\n"
        "static inline NU ## _union_ref_t *N ## _ ## M ## _push_slice(NS ## builder_t *B, NS ## string_t string, size_t index, size_t len)\\\n"
        "{ return NU ## _vec_push(B, NU ## _as_ ## M(NS ## string_slice(B, string, index, len))); }\n"
        "\n",
        nsc);

     fprintf(out->fp,
        "#define __%sbuild_string_vector_field(ID, NS, N, TT)\\\n"
        "__%sbuild_offset_vector_field(ID, NS, N, NS ## string, TT)\\\n"
        "__%sbuild_string_vector_ops(NS, N)\n"
        "\n",
        nsc, nsc, nsc);

    fprintf(out->fp, "#define __%schar_formal_args , char v0\n", nsc);
    fprintf(out->fp, "#define __%schar_call_args , v0\n", nsc);
    fprintf(out->fp, "#define __%suint8_formal_args , uint8_t v0\n", nsc);
    fprintf(out->fp, "#define __%suint8_call_args , v0\n", nsc);
    fprintf(out->fp, "#define __%sint8_formal_args , int8_t v0\n", nsc);
    fprintf(out->fp, "#define __%sint8_call_args , v0\n", nsc);
    fprintf(out->fp, "#define __%sbool_formal_args , %sbool_t v0\n", nsc, nsc);
    fprintf(out->fp, "#define __%sbool_call_args , v0\n", nsc);
    fprintf(out->fp, "#define __%suint16_formal_args , uint16_t v0\n", nsc);
    fprintf(out->fp, "#define __%suint16_call_args , v0\n", nsc);
    fprintf(out->fp, "#define __%suint32_formal_args , uint32_t v0\n", nsc);
    fprintf(out->fp, "#define __%suint32_call_args , v0\n", nsc);
    fprintf(out->fp, "#define __%suint64_formal_args , uint64_t v0\n", nsc);
    fprintf(out->fp, "#define __%suint64_call_args , v0\n", nsc);
    fprintf(out->fp, "#define __%sint16_formal_args , int16_t v0\n", nsc);
    fprintf(out->fp, "#define __%sint16_call_args , v0\n", nsc);
    fprintf(out->fp, "#define __%sint32_formal_args , int32_t v0\n", nsc);
    fprintf(out->fp, "#define __%sint32_call_args , v0\n", nsc);
    fprintf(out->fp, "#define __%sint64_formal_args , int64_t v0\n", nsc);
    fprintf(out->fp, "#define __%sint64_call_args , v0\n", nsc);
    fprintf(out->fp, "#define __%sfloat_formal_args , float v0\n", nsc);
    fprintf(out->fp, "#define __%sfloat_call_args , v0\n", nsc);
    fprintf(out->fp, "#define __%sdouble_formal_args , double v0\n", nsc);
    fprintf(out->fp, "#define __%sdouble_call_args , v0\n", nsc);
    fprintf(out->fp, "\n");
    fprintf(out->fp, "__%sbuild_scalar(%s, %schar, char)\n", nsc, nsc, nsc);
    fprintf(out->fp, "__%sbuild_scalar(%s, %suint8, uint8_t)\n", nsc, nsc, nsc);
    fprintf(out->fp, "__%sbuild_scalar(%s, %sint8, int8_t)\n", nsc, nsc, nsc);
    fprintf(out->fp, "__%sbuild_scalar(%s, %sbool, %sbool_t)\n", nsc, nsc, nsc, nsc);
    fprintf(out->fp, "__%sbuild_scalar(%s, %suint16, uint16_t)\n", nsc, nsc, nsc);
    fprintf(out->fp, "__%sbuild_scalar(%s, %suint32, uint32_t)\n", nsc, nsc, nsc);
    fprintf(out->fp, "__%sbuild_scalar(%s, %suint64, uint64_t)\n", nsc, nsc, nsc);
    fprintf(out->fp, "__%sbuild_scalar(%s, %sint16, int16_t)\n", nsc, nsc, nsc);
    fprintf(out->fp, "__%sbuild_scalar(%s, %sint32, int32_t)\n", nsc, nsc, nsc);
    fprintf(out->fp, "__%sbuild_scalar(%s, %sint64, int64_t)\n", nsc, nsc, nsc);
    fprintf(out->fp, "__%sbuild_scalar(%s, %sfloat, float)\n", nsc, nsc, nsc);
    fprintf(out->fp, "__%sbuild_scalar(%s, %sdouble, double)\n", nsc, nsc, nsc);
    fprintf(out->fp, "\n");
    fprintf(out->fp, "__%sbuild_string(%s)\n", nsc, nsc);
    fprintf(out->fp, "\n");

    fprintf(out->fp, "__%sbuild_buffer(%s)\n", nsc, nsc);
    gen_epilogue(out);
    fprintf(out->fp, "#endif /* %s_COMMON_BUILDER_H */\n", nscup);
    return 0;
}

static int gen_builder_pretext(fb_output_t *out)
{
    const char *nsc = out->nsc;
    const char *nscup = out->nscup;

    fprintf(out->fp,
        "#ifndef %s_BUILDER_H\n"
        "#define %s_BUILDER_H\n",
        out->S->basenameup, out->S->basenameup);

    fprintf(out->fp, "\n/* " FLATCC_GENERATED_BY " */\n\n");
    fprintf(out->fp, "#ifndef %s_READER_H\n", out->S->basenameup);
    fprintf(out->fp, "#include \"%s_reader.h\"\n", out->S->basename);
    fprintf(out->fp, "#endif\n");
    fprintf(out->fp, "#ifndef %s_COMMON_BUILDER_H\n", nscup);
    fprintf(out->fp, "#include \"%scommon_builder.h\"\n", nsc);
    fprintf(out->fp, "#endif\n");

    fb_gen_c_includes(out, "_builder.h", "_BUILDER_H");

    gen_prologue(out);

    /*
     * Even if defined in the reader header, we must redefine it here
     * because another file might sneak in and update.
     */
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
    return 0;
}

static int get_total_struct_field_count(fb_compound_type_t *ct)
{
    fb_member_t *member;
    fb_symbol_t *sym;
    int count = 0;

    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        if (member->metadata_flags & fb_f_deprecated) {
            continue;
        }
        switch (member->type.type) {
        /* struct arrays count as 1 but struct fields are expanded */
        case vt_compound_type_ref:
            if (member->type.ct->symbol.kind == fb_is_struct) {
                count += get_total_struct_field_count(member->type.ct);
                continue;
            }
            ++count;
            break;
        default:
            ++count;
            break;
        }
    }
    return count;
}

static inline void gen_comma(fb_output_t *out, int index, int count, int is_macro)
{
    char *cont = is_macro ? "\\\n" : "\n";

    if (count == 0) {
        return;
    }
    if (index == 0) {
        if (count > 4) {
            fprintf(out->fp, ",%s  ", cont);
        } else {
            fprintf(out->fp, ", ");
        }
    } else {
        if (index % 4 || count - index <= 2) {
            fprintf(out->fp, ", ");
        } else {
            fprintf(out->fp, ",%s  ", cont);
        }
    }
}

static int gen_builder_struct_args(fb_output_t *out, fb_compound_type_t *ct, int index, int len, int is_macro)
{
    const char *nsc = out->nsc;
    fb_member_t *member;
    fb_symbol_t *sym;
    const char *tname, *tname_ns;
    fb_scoped_name_t snref;

    fb_clear(snref);

    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        if (member->metadata_flags & fb_f_deprecated) {
            continue;
        }
        switch (member->type.type) {
        case vt_fixed_array_compound_type_ref:
            gen_comma(out, index, len, is_macro);
            fb_compound_name(member->type.ct, &snref);
            if (member->type.ct->symbol.kind == fb_is_struct) {
                fprintf(out->fp, "const %s_t v%i[%i]", snref.text, index++, (int)member->type.len);
            } else {
                fprintf(out->fp, "%s_enum_t v%i[%i]", snref.text, index++, (int)member->type.len);
            }
            break;
        case vt_compound_type_ref:
            if (member->type.ct->symbol.kind == fb_is_struct) {
                index = gen_builder_struct_args(out, member->type.ct, index, len, is_macro);
                continue;
            }
            gen_comma(out, index, len, is_macro);
            fb_compound_name(member->type.ct, &snref);
            fprintf(out->fp, "%s_enum_t v%i", snref.text, index++);
            break;
        case vt_fixed_array_type:
            gen_comma(out, index, len, is_macro);
            tname_ns = scalar_type_ns(member->type.st, nsc);
            tname = scalar_type_name(member->type.st);
            fprintf(out->fp, "const %s%s v%i[%i]", tname_ns, tname, index++, (int)member->type.len);
            break;
        case vt_scalar_type:
            gen_comma(out, index, len, is_macro);
            tname_ns = scalar_type_ns(member->type.st, nsc);
            tname = scalar_type_name(member->type.st);
            fprintf(out->fp, "%s%s v%i", tname_ns, tname, index++);
            break;
        default:
            gen_panic(out, "internal error: unexpected struct member type");
            continue;
        }
    }
    return index;
}

static int gen_builder_struct_call_list(fb_output_t *out, fb_compound_type_t *ct, int index, int arg_count, int is_macro)
{
    int i;
    int len = get_total_struct_field_count(ct);

    for (i = 0; i < len; ++i) {
        gen_comma(out, i, arg_count, is_macro);
        fprintf(out->fp, "v%i", index++);
    }
    return index;
}

enum { no_conversion, convert_from_pe, convert_to_pe };

/* Note: returned index is not correct when using from_ptr since it doesn't track arguments, but it shouldn't matter. */
static int gen_builder_struct_field_assign(fb_output_t *out, fb_compound_type_t *ct, int index, int arg_count,
        int conversion, int from_ptr)
{
    const char *nsc = out->nsc;
    fb_member_t *member;
    fb_symbol_t *sym;
    int n, len;
    const char *s;
    int deprecated_index = 0;
    const char *kind, *tprefix;
    fb_scoped_name_t snref;

    fb_clear(snref);
    switch (conversion) {
    case convert_to_pe: kind = "_to_pe"; break;
    case convert_from_pe: kind = "_from_pe"; break;
    default: kind = ""; break;
    }
    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        symbol_name(sym, &n, &s);

        if (index > 0) {
            if (index % 4 == 0) {
                fprintf(out->fp, ";\n  ");
            } else {
                fprintf(out->fp, "; ");
            }
        }
        switch (member->type.type) {
        case vt_fixed_array_compound_type_ref:
            len = (int)member->type.len;
            fb_compound_name(member->type.ct, &snref);
            if (member->metadata_flags & fb_f_deprecated) {
                fprintf(out->fp, "__%sstruct_clear_field(p->__deprecated%i)",
                        nsc, deprecated_index);
                ++deprecated_index;
                ++index;
                continue;
            }
            if (from_ptr) {
                fprintf(out->fp, "%s_array_copy%s(p->%.*s, p2->%.*s, %d)",
                        snref.text, kind, n, s, n, s, len);
            } else {
                fprintf(out->fp, "%s_array_copy%s(p->%.*s, v%i, %d)",
                        snref.text, kind, n, s, index, len);
            }
            ++index;
            continue;
        case vt_compound_type_ref:
            fb_compound_name(member->type.ct, &snref);
            if (member->type.ct->symbol.kind == fb_is_struct) {
                if (member->metadata_flags & fb_f_deprecated) {
                    fprintf(out->fp, "__%sstruct_clear_field(p->__deprecated%i)",
                            nsc, deprecated_index);
                    deprecated_index++;
                    index += get_total_struct_field_count(member->type.ct);
                    continue;
                }
                if (from_ptr) {
                    fprintf(out->fp, "%s_copy%s(&p->%.*s, &p2->%.*s)", snref.text, kind, n, s, n, s);
                    /* `index` does not count children, but it doesn't matter here. */
                    ++index;
                } else {
                    fprintf(out->fp, "%s_assign%s(&p->%.*s", snref.text, kind, n, s);
                    index = gen_builder_struct_call_list(out, member->type.ct, index, arg_count, 0);
                    fprintf(out->fp, ")");
                }
                continue;
            }
            if (member->metadata_flags & fb_f_deprecated) {
                fprintf(out->fp, "__%sstruct_clear_field(p->__deprecated%i)",
                        nsc, deprecated_index);
                ++deprecated_index;
                ++index;
                continue;
            }
            switch (member->size == 1 ? no_conversion : conversion) {
            case convert_from_pe:
                if (from_ptr) {
                    fprintf(out->fp, "%s_copy_from_pe(&p->%.*s, &p2->%.*s)",
                            snref.text, n, s, n, s);
                } else {
                    fprintf(out->fp, "%s_assign_from_pe(&p->%.*s, v%i)",
                            snref.text, n, s, index);
                }
                break;
            case convert_to_pe:
                if (from_ptr) {
                    fprintf(out->fp, "%s_copy_to_pe(&p->%.*s, &p2->%.*s)",
                            snref.text, n, s, n, s);
                } else {
                    fprintf(out->fp, "%s_assign_to_pe(&p->%.*s, v%i)",
                            snref.text, n, s, index);
                }
                break;
            default:
                if (from_ptr) {
                    fprintf(out->fp, "p->%.*s = p2->%.*s", n, s, n, s);
                } else {
                    fprintf(out->fp, "p->%.*s = v%i", n, s, index);
                }
                break;
            }
            ++index;
            continue;
        case vt_fixed_array_type:
            tprefix = scalar_type_prefix(member->type.st);
            len = (int)member->type.len;
            if (member->metadata_flags & fb_f_deprecated) {
                fprintf(out->fp, "__%sstruct_clear_field(p->__deprecated%i)",
                        nsc, deprecated_index);
                ++deprecated_index;
                ++index;
                continue;
            }
            if (from_ptr) {
                fprintf(out->fp, "%s%s_array_copy%s(p->%.*s, p2->%.*s, %d)",
                        nsc, tprefix, kind, n, s, n, s, len);
            } else {
                fprintf(out->fp, "%s%s_array_copy%s(p->%.*s, v%i, %d)",
                        nsc, tprefix, kind, n, s, index, len);
            }
            ++index;
            break;
        case vt_scalar_type:
            tprefix = scalar_type_prefix(member->type.st);
            if (member->metadata_flags & fb_f_deprecated) {
                fprintf(out->fp, "__%sstruct_clear_field(p->__deprecated%i)",
                        nsc, deprecated_index);
                ++deprecated_index;
                ++index;
                continue;
            }
            switch (member->size == 1 ? no_conversion : conversion) {
            case convert_from_pe:
                if (from_ptr) {
                    fprintf(out->fp, "%s%s_copy_from_pe(&p->%.*s, &p2->%.*s)",
                            nsc, tprefix, n, s, n, s);
                } else {
                    fprintf(out->fp, "%s%s_assign_from_pe(&p->%.*s, v%i)",
                            nsc, tprefix, n, s, index);
                }
                break;
            case convert_to_pe:
                if (from_ptr) {
                    fprintf(out->fp, "%s%s_copy_to_pe(&p->%.*s, &p2->%.*s)",
                            nsc, tprefix, n, s, n, s);
                } else {
                    fprintf(out->fp, "%s%s_assign_to_pe(&p->%.*s, v%i)",
                            nsc, tprefix, n, s, index);
                }
                break;
            default:
                if (from_ptr) {
                    fprintf(out->fp, "p->%.*s = p2->%.*s", n, s, n, s);
                } else {
                    fprintf(out->fp, "p->%.*s = v%i", n, s, index);
                }
                break;
            }
            ++index;
            break;
        default:
            gen_panic(out, "internal error: type error");
            continue;
        }
    }
    if (arg_count > 0) {
        fprintf(out->fp, ";\n  ");
    }
    return index;
}

static void gen_builder_struct(fb_output_t *out, fb_compound_type_t *ct)
{
    const char *nsc = out->nsc;
    int arg_count;
    fb_scoped_name_t snt;

    fb_clear(snt);
    assert(ct->symbol.kind == fb_is_struct);

    fb_compound_name(ct, &snt);

    arg_count = get_total_struct_field_count(ct);
    fprintf(out->fp, "#define __%s_formal_args ", snt.text);
    gen_builder_struct_args(out, ct, 0, arg_count, 1);
    fprintf(out->fp, "\n#define __%s_call_args ", snt.text);
    gen_builder_struct_call_list(out, ct, 0, arg_count, 1);
    fprintf(out->fp, "\n");
    fprintf(out->fp,
            "static inline %s_t *%s_assign(%s_t *p",
            snt.text, snt.text, snt.text);
    gen_builder_struct_args(out, ct, 0, arg_count, 0);
    fprintf(out->fp, ")\n{ ");
    gen_builder_struct_field_assign(out, ct, 0, arg_count, no_conversion, 0);
    fprintf(out->fp, "return p; }\n");
    fprintf(out->fp,
            "static inline %s_t *%s_copy(%s_t *p, const %s_t *p2)\n",
            snt.text, snt.text, snt.text, snt.text);
    fprintf(out->fp, "{ ");
    gen_builder_struct_field_assign(out, ct, 0, arg_count, no_conversion, 1);
    fprintf(out->fp, "return p; }\n");
    fprintf(out->fp,
            "static inline %s_t *%s_assign_to_pe(%s_t *p",
            snt.text, snt.text, snt.text);
    gen_builder_struct_args(out, ct, 0, arg_count, 0);
    fprintf(out->fp, ")\n{ ");
    gen_builder_struct_field_assign(out, ct, 0, arg_count, convert_to_pe, 0);
    fprintf(out->fp, "return p; }\n");
    fprintf(out->fp,
            "static inline %s_t *%s_copy_to_pe(%s_t *p, const %s_t *p2)\n",
            snt.text, snt.text, snt.text, snt.text);
    fprintf(out->fp, "{ ");
    gen_builder_struct_field_assign(out, ct, 0, arg_count, convert_to_pe, 1);
    fprintf(out->fp, "return p; }\n");
    fprintf(out->fp,
            "static inline %s_t *%s_assign_from_pe(%s_t *p",
            snt.text, snt.text, snt.text);
    gen_builder_struct_args(out, ct, 0, arg_count, 0);
    fprintf(out->fp, ")\n{ ");
    gen_builder_struct_field_assign(out, ct, 0, arg_count, convert_from_pe, 0);
    fprintf(out->fp, "return p; }\n");
    fprintf(out->fp,
            "static inline %s_t *%s_copy_from_pe(%s_t *p, const %s_t *p2)\n",
            snt.text, snt.text, snt.text, snt.text);
    fprintf(out->fp, "{ ");
    gen_builder_struct_field_assign(out, ct, 0, arg_count, convert_from_pe, 1);
    fprintf(out->fp, "return p; }\n");
    fprintf(out->fp, "__%sbuild_struct(%s, %s, %"PRIu64", %u, %s_file_identifier, %s_type_identifier)\n",
            nsc, nsc, snt.text, (uint64_t)ct->size, ct->align, snt.text, snt.text);

    if (ct->size > 0) {
        fprintf(out->fp, "__%sdefine_fixed_array_primitives(%s, %s, %s_t)\n",
                nsc, nsc, snt.text, snt.text);
    }
}

static int get_create_table_arg_count(fb_compound_type_t *ct)
{
    fb_member_t *member;
    fb_symbol_t *sym;
    int count = 0;

    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        if (member->metadata_flags & fb_f_deprecated) {
            continue;
        }
        ++count;
    }
    return count;
}

static int gen_builder_table_call_list(fb_output_t *out, fb_compound_type_t *ct, int arg_count, int is_macro)
{
    fb_member_t *member;
    fb_symbol_t *sym;
    int index = 0;

    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        if (member->metadata_flags & fb_f_deprecated) {
            continue;
        }
        gen_comma(out, index, arg_count, is_macro);
        fprintf(out->fp, "v%"PRIu64"", (uint64_t)member->id);
        ++index;
    }
    return index;
}


static int gen_required_table_fields(fb_output_t *out, fb_compound_type_t *ct)
{
    const char *nsc = out->nsc;
    fb_member_t *member;
    fb_symbol_t *sym;
    int index;
    int arg_count;
    fb_scoped_name_t snt;

    fb_clear(snt);
    arg_count = get_create_table_arg_count(ct);
    index = 0;
    fb_compound_name(ct, &snt);
    fprintf(out->fp, "static const %svoffset_t __%s_required[] = {", nsc, snt.text);
    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        if (member->metadata_flags & fb_f_deprecated) {
            continue;
        }
        if (member->metadata_flags & fb_f_required) {
            if (index > 0) {
                gen_comma(out, index, arg_count, 0);
            } else {
                fprintf(out->fp, " ");
            }
            fprintf(out->fp, "%u", (unsigned)member->id);
            index++;
        }
    }
    /* Add extra element to avoid null arrays. */
    if (index > 0) {
        fprintf(out->fp, ", 0 };\n");
    } else {
        fprintf(out->fp, " 0 };\n");
    }
    return index;
}

static int gen_builder_table_args(fb_output_t *out, fb_compound_type_t *ct, int arg_count, int is_macro)
{
    const char *nsc = out->nsc;
    fb_symbol_t *sym;
    fb_member_t *member;
    const char *tname, *tname_ns;
    int index;
    fb_scoped_name_t snref;

    fb_clear(snref);
    /* Just to help the comma. */
    index = 0;
    /* We use the id to name arguments so sorted assignment can find the arguments trivially. */
    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        if (member->metadata_flags & fb_f_deprecated) {
            continue;
        }
        gen_comma(out, index++, arg_count, is_macro);
        switch (member->type.type) {
        case vt_compound_type_ref:
            fb_compound_name(member->type.ct, &snref);
            switch (member->type.ct->symbol.kind) {
            case fb_is_struct:
                fprintf(out->fp, "%s_t *v%"PRIu64"", snref.text, (uint64_t)member->id);
                break;
            case fb_is_enum:
                fprintf(out->fp, "%s_enum_t v%"PRIu64"", snref.text, (uint64_t)member->id);
                break;
            case fb_is_table:
                fprintf(out->fp, "%s_ref_t v%"PRIu64"", snref.text, (uint64_t)member->id);
                break;
            case fb_is_union:
                /* Unions jump an index because it is two fields. */
                fprintf(out->fp, "%s_union_ref_t v%"PRIu64"", snref.text, (uint64_t)member->id);
                break;
            default:
                gen_panic(out, "internal error: unexpected table field type");
                continue;
            }
            break;
        case vt_vector_compound_type_ref:
            fb_compound_name(member->type.ct, &snref);
            switch (member->type.ct->symbol.kind) {
            case fb_is_struct:
            case fb_is_enum:
            case fb_is_table:
                fprintf(out->fp, "%s_vec_ref_t v%"PRIu64"", snref.text, (uint64_t)member->id);
                break;
            case fb_is_union:
                fprintf(out->fp, "%s_union_vec_ref_t v%"PRIu64"", snref.text, (uint64_t)member->id);
                break;
            default:
                gen_panic(out, "internal error: unexpected table table type");
                continue;
            }
            break;
        case vt_scalar_type:
            tname_ns = scalar_type_ns(member->type.st, nsc);
            tname = scalar_type_name(member->type.st);
            fprintf(out->fp, "%s%s v%"PRIu64"", tname_ns, tname, (uint64_t)member->id);
            break;
        case vt_vector_type:
            tname = scalar_type_prefix(member->type.st);
            fprintf(out->fp, "%s%s_vec_ref_t v%"PRIu64"", nsc, tname, (uint64_t)member->id);
            break;
        case vt_string_type:
            fprintf(out->fp, "%sstring_ref_t v%"PRIu64"", nsc, (uint64_t)member->id);
            break;
        case vt_vector_string_type:
            fprintf(out->fp, "%sstring_vec_ref_t v%"PRIu64"", nsc, (uint64_t)member->id);
            break;
        default:
            gen_panic(out, "internal error: unexpected table member type");
            continue;
        }
    }
    return index;
}

static int gen_builder_create_table_decl(fb_output_t *out, fb_compound_type_t *ct)
{
    const char *nsc = out->nsc;
    int arg_count;
    fb_scoped_name_t snt;

    fb_clear(snt);
    fb_compound_name(ct, &snt);

    arg_count = get_create_table_arg_count(ct);
    fprintf(out->fp, "#define __%s_formal_args ", snt.text);
    gen_builder_table_args(out, ct, arg_count, 1);
    fprintf(out->fp, "\n#define __%s_call_args ", snt.text);
    gen_builder_table_call_list(out, ct, arg_count, 1);
    fprintf(out->fp, "\n");

    /* `_clone` fw decl must be place before build_table macro and `_create` must be placed after. */
    fprintf(out->fp,
            "static inline %s_ref_t %s_create(%sbuilder_t *B __%s_formal_args);\n",
            snt.text, snt.text, nsc, snt.text);
    return 0;
}

static int gen_builder_create_table(fb_output_t *out, fb_compound_type_t *ct)
{
    const char *nsc = out->nsc;
    fb_member_t *member;
    int n;
    const char *s;
    int patch_union = !(ct->metadata_flags & fb_f_original_order);
    int has_union = 0;
    fb_scoped_name_t snt;

    fb_clear(snt);
    fb_compound_name(ct, &snt);

    fprintf(out->fp,
            "static inline %s_ref_t %s_create(%sbuilder_t *B __%s_formal_args)\n",
            snt.text, snt.text, nsc, snt.text);

    fprintf(out->fp, "{\n    if (%s_start(B)", snt.text);
    for (member = ct->ordered_members; member; member = member->order) {
        if (member->metadata_flags & fb_f_deprecated) {
            continue;
        }
        symbol_name(&member->symbol, &n, &s);
        if (member->type.type == vt_compound_type_ref && member->type.ct->symbol.kind == fb_is_union) {
            has_union = 1;
            if (patch_union) {
                fprintf(out->fp, "\n        || %s_%.*s_add_value(B, v%"PRIu64")", snt.text, n, s, (uint64_t)member->id);
                continue;
            }
        }
        fprintf(out->fp, "\n        || %s_%.*s_add(B, v%"PRIu64")", snt.text, n, s, (uint64_t)member->id);
    }
    if (patch_union && has_union) {
        for (member = ct->ordered_members; member; member = member->order) {
            if (member->metadata_flags & fb_f_deprecated) {
                continue;
            }
            if (member->type.type == vt_compound_type_ref && member->type.ct->symbol.kind == fb_is_union) {
                symbol_name(&member->symbol, &n, &s);
                fprintf(out->fp, "\n        || %s_%.*s_add_type(B, v%"PRIu64".type)", snt.text, n, s, (uint64_t)member->id);
            }
        }
    }
    fprintf(out->fp, ") {\n        return 0;\n    }\n    return %s_end(B);\n}\n\n", snt.text);
    return 0;
}

static int gen_builder_structs(fb_output_t *out)
{
    fb_compound_type_t *ct;

    /* Generate structs in topologically sorted order. */
    for (ct = out->S->ordered_structs; ct; ct = ct->order) {
        gen_builder_struct(out, ct);
        fprintf(out->fp, "\n");
    }
    return 0;
}

static int gen_builder_table(fb_output_t *out, fb_compound_type_t *ct)
{
    const char *nsc = out->nsc;
    fb_scoped_name_t snt;

    fb_clear(snt);
    fb_compound_name(ct, &snt);

    fprintf(out->fp,
            "typedef %sref_t %s_ref_t;\n",
            nsc, snt.text);
    fprintf(out->fp,
            "static %s_ref_t %s_clone(%sbuilder_t *B, %s_table_t t);\n",
            snt.text, snt.text, nsc, snt.text);
    fprintf(out->fp, "__%sbuild_table(%s, %s, %"PRIu64")\n",
            nsc, nsc, snt.text, (uint64_t)ct->count);
    return 0;
}

static int gen_builder_table_prolog(fb_output_t *out, fb_compound_type_t *ct)
{
    const char *nsc = out->nsc;
    fb_scoped_name_t snt;

    fb_clear(snt);
    fb_compound_name(ct, &snt);

    fprintf(out->fp, "__%sbuild_table_prolog(%s, %s, %s_file_identifier, %s_type_identifier)\n",
            nsc, nsc, snt.text, snt.text, snt.text);
    return 0;
}

static int gen_union_fields(fb_output_t *out, const char *st, int n, const char *s,
        fb_compound_type_t *ct, int is_vector)
{
    const char *nsc = out->nsc;
    fb_symbol_t *sym;
    fb_member_t *member;
    const char *su;
    int nu;
    fb_scoped_name_t snref;
    fb_scoped_name_t snu;
    const char *kind = is_vector ? "vector_value" : "value";

    fb_clear(snref);
    fb_clear(snu);
    fb_compound_name(ct, &snref);
    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        symbol_name(sym, &nu, &su);
        switch (member->type.type) {
        case vt_missing:
            break;
        case vt_compound_type_ref:
            fb_compound_name(member->type.ct, &snu);
            switch (member->type.ct->symbol.kind) {
            case fb_is_table:
                fprintf(out->fp,
                        "__%sbuild_union_table_%s_field(%s, %s_%.*s, %s, %.*s, %s)\n",
                        nsc, kind, nsc, st, n, s, snref.text, nu, su, snu.text);
                break;
            case fb_is_struct:
                fprintf(out->fp,
                        "__%sbuild_union_struct_%s_field(%s, %s_%.*s, %s, %.*s, %s)\n",
                        nsc, kind, nsc, st, n, s, snref.text, nu, su, snu.text);
                break;
            default:
                gen_panic(out, "internal error: unexpected union member compound type");
                return -1;
            }
            break;
        case vt_string_type:
            fprintf(out->fp,
                    "__%sbuild_union_string_%s_field(%s, %s_%.*s, %s, %.*s)\n",
                    nsc, kind, nsc, st, n, s, snref.text, nu, su);
            break;
        default:
            gen_panic(out, "internal error: unexpected union member type");
            return -1;
        }
    }
    return 0;
}

static int gen_builder_table_fields(fb_output_t *out, fb_compound_type_t *ct)
{
    const char *nsc = out->nsc;
    fb_member_t *member;
    fb_symbol_t *sym;
    const char *s, *tprefix, *tname, *tname_ns;
    int n;
    int is_optional;
    fb_scoped_name_t snt;
    fb_scoped_name_t snref;
    fb_literal_t literal;

    fb_clear(snt);
    fb_clear(snref);
    fb_compound_name(ct, &snt);

    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        symbol_name(&member->symbol, &n, &s);
        if (member->metadata_flags & fb_f_deprecated) {
            fprintf(out->fp, "/* Skipping build of deprecated field: '%s_%.*s' */\n\n", snt.text, n, s);
            continue;
        }
        is_optional = member->flags & fb_fm_optional;
        switch (member->type.type) {
        case vt_scalar_type:
            tname_ns = scalar_type_ns(member->type.st, nsc);
            tname = scalar_type_name(member->type.st);
            tprefix = scalar_type_prefix(member->type.st);
            if (is_optional) {
                fprintf(out->fp,
                    "__%sbuild_scalar_optional_field(%"PRIu64", %s, %s_%.*s, %s%s, %s%s, %"PRIu64", %u, %s)\n",
                    nsc, (uint64_t)member->id, nsc, snt.text, n, s, nsc, tprefix, tname_ns, tname,
                    (uint64_t)member->size, member->align, snt.text);
            } else {
                print_literal(member->type.st, &member->value, literal);
                fprintf(out->fp,
                    "__%sbuild_scalar_field(%"PRIu64", %s, %s_%.*s, %s%s, %s%s, %"PRIu64", %u, %s, %s)\n",
                    nsc, (uint64_t)member->id, nsc, snt.text, n, s, nsc, tprefix, tname_ns, tname,
                    (uint64_t)member->size, member->align, literal, snt.text);
            }
            break;
        case vt_vector_type:
            tname_ns = scalar_type_ns(member->type.st, nsc);
            tname = scalar_type_name(member->type.st);
            tprefix = scalar_type_prefix(member->type.st);
            fprintf(out->fp,
                "__%sbuild_vector_field(%"PRIu64", %s, %s_%.*s, %s%s, %s%s, %s)\n",
                nsc, (uint64_t)member->id, nsc, snt.text, n, s, nsc, tprefix, tname_ns, tname, snt.text);
            /* [ubyte] vectors can nest buffers. */
            if (member->nest) {
                switch (member->nest->symbol.kind) {
                case fb_is_table:
                    fb_compound_name((fb_compound_type_t *)(&member->nest->symbol), &snref);
                    fprintf(out->fp, "__%sbuild_nested_table_root(%s, %s_%.*s, %s, %s_identifier, %s_type_identifier)\n",
                        nsc, nsc, snt.text, n, s, snref.text, snref.text, snref.text);
                    break;
                case fb_is_struct:
                    fb_compound_name((fb_compound_type_t *)(&member->nest->symbol), &snref);
                    fprintf(out->fp, "__%sbuild_nested_struct_root(%s, %s_%.*s, %s, %u, %s_identifier, %s_type_identifier)\n",
                        nsc, nsc, snt.text, n, s, snref.text,
                        (unsigned)((fb_compound_type_t *)(member->nest))->align, snref.text, snref.text);
                    break;
                default:
                    gen_panic(out, "internal error: unexpected nested type");
                    continue;
                }
            }
            break;
        case vt_string_type:
            fprintf(out->fp,
                "__%sbuild_string_field(%"PRIu64", %s, %s_%.*s, %s)\n",
                nsc, (uint64_t)member->id, nsc, snt.text, n, s, snt.text);
            break;
        case vt_vector_string_type:
            fprintf(out->fp,
                "__%sbuild_string_vector_field(%"PRIu64", %s, %s_%.*s, %s)\n",
                nsc, (uint64_t)member->id, nsc, snt.text, n, s, snt.text);
            break;
        case vt_compound_type_ref:
            fb_compound_name(member->type.ct, &snref);
            switch (member->type.ct->symbol.kind) {
            case fb_is_struct:
                fprintf(out->fp,
                    "__%sbuild_struct_field(%"PRIu64", %s, %s_%.*s, %s, %"PRIu64", %u, %s)\n",
                    nsc, (uint64_t)member->id, nsc, snt.text, n, s, snref.text, (uint64_t)member->size, member->align, snt.text);
                break;
            case fb_is_table:
                fprintf(out->fp,
                    "__%sbuild_table_field(%"PRIu64", %s, %s_%.*s, %s, %s)\n",
                    nsc, (uint64_t)member->id, nsc, snt.text, n, s, snref.text, snt.text);
                break;
            case fb_is_enum:
                if (is_optional) {
                    fprintf(out->fp,
                        "__%sbuild_scalar_optional_field(%"PRIu64", %s, %s_%.*s, %s, %s_enum_t, %"PRIu64", %u, %s)\n",
                        nsc, (uint64_t)member->id, nsc, snt.text, n, s, snref.text, snref.text,
                        (uint64_t)member->size, member->align, snt.text);
                } else {
                    print_literal(member->type.ct->type.st, &member->value, literal);
                    fprintf(out->fp,
                        "__%sbuild_scalar_field(%"PRIu64", %s, %s_%.*s, %s, %s_enum_t, %"PRIu64", %u, %s, %s)\n",
                        nsc, (uint64_t)member->id, nsc, snt.text, n, s, snref.text, snref.text,
                        (uint64_t)member->size, member->align, literal, snt.text);
                }
                break;
            case fb_is_union:
                fprintf(out->fp,
                    "__%sbuild_union_field(%"PRIu64", %s, %s_%.*s, %s, %s)\n",
                    nsc, (uint64_t)member->id, nsc, snt.text, n, s, snref.text, snt.text);
                gen_union_fields(out, snt.text, n, s, member->type.ct, 0);
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
                if (member->type.ct->symbol.flags & fb_indexed) {
                    fprintf(out->fp, "/* vector has keyed elements */\n");
                }
                fprintf(out->fp,
                    "__%sbuild_vector_field(%"PRIu64", %s, %s_%.*s, %s, %s_t, %s)\n",
                    nsc, (uint64_t)member->id, nsc, snt.text, n, s, snref.text, snref.text, snt.text);
                break;
            case fb_is_table:
                if (member->type.ct->symbol.flags & fb_indexed) {
                    fprintf(out->fp, "/* vector has keyed elements */\n");
                }
                fprintf(out->fp,
                    "__%sbuild_table_vector_field(%"PRIu64", %s, %s_%.*s, %s, %s)\n",
                    nsc, (uint64_t)member->id, nsc, snt.text, n, s, snref.text, snt.text);
                break;
            case fb_is_enum:
                fprintf(out->fp,
                    "__%sbuild_vector_field(%"PRIu64", %s, %s_%.*s, %s, %s_enum_t, %s)\n",
                    nsc, (uint64_t)member->id, nsc, snt.text, n, s, snref.text, snref.text, snt.text);
                break;
            case fb_is_union:
                fprintf(out->fp,
                    "__%sbuild_union_vector_field(%"PRIu64", %s, %s_%.*s, %s, %s)\n",
                    nsc, (uint64_t)member->id, nsc, snt.text, n, s, snref.text, snt.text);
                gen_union_fields(out, snt.text, n, s, member->type.ct, 1);
                break;
            default:
                gen_panic(out, "internal error: unexpected vector compound type in table during code generation");
                break;
            }
            break;
        default:
            gen_panic(out, "internal error: unexpected table member type during code generation");
            break;
        }
    }
    fprintf(out->fp, "\n");
    return 0;
}

/*
 * NOTE:
 *
 * Cloning a table might lead to a combinatorial explosion if the source
 * has many shared references in a DAG. In many cases this might not be
 * an issue, but it it is deduplication will be necessary. Deduplication
 * is not specific to cloning but especially relevant here. Because
 * deduplication carries an overhead in runtime and complexity it is not
 * part of the core cloning operation. Cloning of unions and vectors with
 * references have similar concerns.
 *
 * A deduplication operation would internally look like like this:
 *
 *   dedup_clone_table(builder, dedup_map, src_ptr)
 *   {
 *      ref = get_cloned_ref(dedup_map, src_ptr)
 *      if (!ref) {
 *          ref = clone_table(builder, src_ptr);
 *          set_cloned_ref(dedup_map, src_ptr, ref);
 *      }
 *      return ref;
 *   }
 *
 * where dedup_map is a map from a pointer to a builder reference and
 * where the dedup_map is dedicated to a single builder and may cover
 * multiple source buffers as long as they have separate memory
 * locations - otherwise a separate dedup map must be used for each
 * source buffer.
 *
 * Note that the clone operation is not safe without a safe source
 * buffer so clone cannot be used to make a buffer with overlapping data
 * safe (e.g. a string and a table referencing the same memory). Even if
 * the source passes basic verification the result might not. To make
 * clone safe it would be necessariry to remember the type as well, for
 * example by adding a type specifier to the dedup_map.
 *
 * In the following we do not implement deduplication.
 */
static int gen_builder_clone_table(fb_output_t *out, fb_compound_type_t *ct)
{
    const char *nsc = out->nsc;
    fb_member_t *member;
    const char *s;
    int n;
    fb_scoped_name_t snt;
    fb_scoped_name_t snref;

    fb_clear(snt);
    fb_clear(snref);
    fb_compound_name(ct, &snt);

    /*
     * We could optimize this by cloning the entire table memory block
     * and then update update only the references. The builder has
     * direct vtable operations to support this - this would not work
     * properly if there are deprecated fields to be stripped or if the
     * default value has changed - and, more complicated: it is
     * necessary to know what table alignment needs to be which require
     * inspection of all fields, or a worst case assumption. So at least
     * for now, we clone by picking one field at a time.
     */

    fprintf(out->fp,
            "static %s_ref_t %s_clone(%sbuilder_t *B, %s_table_t t)\n",
            snt.text, snt.text, nsc, snt.text);

    fprintf(out->fp,
            "{\n"
            "    __%smemoize_begin(B, t);\n"
            "    if (%s_start(B)", nsc, snt.text);
    for (member = ct->ordered_members; member; member = member->order) {
        if (member->metadata_flags & fb_f_deprecated) {
            continue;
        }
        symbol_name(&member->symbol, &n, &s);
        switch (member->type.type) {
        case vt_scalar_type:
        case vt_vector_type: /* This includes nested buffers - they are just transferred as bytes. */
        case vt_string_type:
        case vt_vector_string_type:
            fprintf(out->fp, "\n        || %s_%.*s_pick(B, t)", snt.text, n, s);
            break;
        case vt_compound_type_ref:
            fb_compound_name(member->type.ct, &snref);
            switch (member->type.ct->symbol.kind) {
            case fb_is_struct:
            case fb_is_table:
            case fb_is_enum:
            case fb_is_union:
                fprintf(out->fp, "\n        || %s_%.*s_pick(B, t)", snt.text, n, s);
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
            case fb_is_table:
            case fb_is_enum:
            case fb_is_union:
                fprintf(out->fp, "\n        || %s_%.*s_pick(B, t)", snt.text, n, s);
                break;
            default:
                gen_panic(out, "internal error: unexpected vector compound type in table during code generation");
                break;
            }
            break;
        default:
            gen_panic(out, "internal error: unexpected table member type during code generation");
            break;
        }
    }
    fprintf(out->fp, ") {\n"
            "        return 0;\n"
            "    }\n"
            "    __%smemoize_end(B, t, %s_end(B));\n}\n", nsc, snt.text);
    return 0;
}

static int gen_builder_enums(fb_output_t *out)
{
    const char *nsc = out->nsc;
    fb_symbol_t *sym;
    int was_here = 0;
    fb_scoped_name_t snt;

    fb_clear(snt);

    for (sym = out->S->symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_enum:
            fb_compound_name((fb_compound_type_t *)sym, &snt);
            fprintf(out->fp,
                "#define __%s_formal_args , %s_enum_t v0\n"
                "#define __%s_call_args , v0\n",
                snt.text, snt.text,
                snt.text);
            fprintf(out->fp, "__%sbuild_scalar(%s, %s, %s_enum_t)\n",
                nsc, nsc, snt.text, snt.text);
            was_here = 1;
            break;
        default:
            continue;
        }
    }
    if (was_here) {
        fprintf(out->fp, "\n");
    }
    return 0;
}

/*
 * Scope resolution is a bit fuzzy in unions -
 *
 * Googles flatc compiler allows dot notation in unions but not enums.
 * C++ generates unqualified enum members (i.e. MyGame.Example.Monster
 * becomes Monster) in the generated enum but still refers to the
 * specific table type in the given namespace. This makes it possible
 * to have name conflicts, and flatc raises these like other enum
 * conficts.
 *
 * We use the same approach and this is why we both look up compound
 * name and symbol name for the same member but the code generator
 * is not concerned with how the scope is parsed or how errors are
 * flagged - it just expects members to be unique.
 */
static int gen_union(fb_output_t *out, fb_compound_type_t *ct)
{
    const char *nsc = out->nsc;
    fb_member_t *member;
    fb_symbol_t *sym;
    const char *s;
    int n;
    fb_scoped_name_t snt;
    fb_scoped_name_t snref;

    fb_clear(snt);
    fb_clear(snref);
    fb_compound_name(ct, &snt);

    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        switch (member->type.type) {
        case vt_compound_type_ref:
            fb_compound_name((fb_compound_type_t *)member->type.ct, &snref);
            symbol_name(sym, &n, &s);
            fprintf(out->fp,
                "static inline %s_union_ref_t %s_as_%.*s(%s_ref_t ref)\n"
                "{ %s_union_ref_t uref; uref.type = %s_%.*s; uref.value = ref; return uref; }\n",
                snt.text, snt.text, n, s, snref.text,
                snt.text, snt.text, n, s);
            break;
        case vt_string_type:
            symbol_name(sym, &n, &s);
            fprintf(out->fp,
                "static inline %s_union_ref_t %s_as_%.*s(%sstring_ref_t ref)\n"
                "{ %s_union_ref_t uref; uref.type = %s_%.*s; uref.value = ref; return uref; }\n",
                snt.text, snt.text, n, s, nsc,
                snt.text, snt.text, n, s);
            break;
        case vt_missing:
            fprintf(out->fp,
                "static inline %s_union_ref_t %s_as_NONE(void)\n"
                "{ %s_union_ref_t uref; uref.type = %s_NONE; uref.value = 0; return uref; }\n",
                snt.text, snt.text, snt.text, snt.text);
            break;
        default:
            gen_panic(out, "internal error: unexpected union value type");
            break;
        }
    }
    fprintf(out->fp,
        "__%sbuild_union_vector(%s, %s)\n\n",
        nsc, nsc, snt.text);
    return 0;
}

static int gen_union_clone(fb_output_t *out, fb_compound_type_t *ct)
{
    const char *nsc = out->nsc;
    fb_member_t *member;
    fb_symbol_t *sym;
    const char *s;
    int n;
    fb_scoped_name_t snt;
    fb_scoped_name_t snref;

    fb_clear(snt);
    fb_clear(snref);
    fb_compound_name(ct, &snt);

    fprintf(out->fp,
            "static %s_union_ref_t %s_clone(%sbuilder_t *B, %s_union_t u)\n{\n    switch (u.type) {\n",
            snt.text, snt.text, nsc, snt.text);

    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        switch (member->type.type) {
        case vt_compound_type_ref:
            fb_compound_name((fb_compound_type_t *)member->type.ct, &snref);
            symbol_name(sym, &n, &s);
            switch (member->type.ct->symbol.kind) {
            case fb_is_table:
                fprintf(out->fp,
                    "    case %u: return %s_as_%.*s(%s_clone(B, (%s_table_t)u.value));\n",
                    (unsigned)member->value.u, snt.text, n, s, snref.text, snref.text);
                break;
            case fb_is_struct:
                fprintf(out->fp,
                    "    case %u: return %s_as_%.*s(%s_clone(B, (%s_struct_t)u.value));\n",
                    (unsigned)member->value.u, snt.text, n, s, snref.text, snref.text);
                break;
            default:
                gen_panic(out, "internal error: unexpected union value type");
                break;
            }
            break;
        case vt_string_type:
            symbol_name(sym, &n, &s);
            fprintf(out->fp,
                "    case %u: return %s_as_%.*s(%sstring_clone(B, u.value));\n",
                (unsigned)member->value.u, snt.text, n, s, nsc);
            break;
        case vt_missing:
            break;
        default:
            gen_panic(out, "internal error: unexpected union value type");
            break;
        }
    }

    /* Unknown unions are dropped. */
    fprintf(out->fp,
            "    default: return %s_as_NONE();\n"
            "    }\n}\n",
            snt.text);
    return 0;
}


static int gen_builder_union_decls(fb_output_t *out)
{
    const char *nsc = out->nsc;
    fb_symbol_t *sym;
    int was_here = 0;
    fb_scoped_name_t snt;

    fb_clear(snt);

    for (sym = out->S->symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_union:
            fb_compound_name((fb_compound_type_t *)sym, &snt);
            fprintf(out->fp,
                "typedef %sunion_ref_t %s_union_ref_t;\n"
                "typedef %sunion_vec_ref_t %s_union_vec_ref_t;\n",
                nsc, snt.text, nsc, snt.text);
            fprintf(out->fp,
                "static %s_union_ref_t %s_clone(%sbuilder_t *B, %s_union_t t);\n",
                snt.text, snt.text, nsc, snt.text);
            was_here = 1;
            break;
        default:
            continue;
        }
    }
    if (was_here) {
        fprintf(out->fp, "\n");
    }
    return 0;
}

static int gen_builder_unions(fb_output_t *out)
{
    fb_symbol_t *sym;

    for (sym = out->S->symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_union:
            gen_union(out, (fb_compound_type_t *)sym);
            gen_union_clone(out, (fb_compound_type_t *)sym);
            fprintf(out->fp, "\n");
            break;
        default:
            continue;
        }
    }
    return 0;
}

static int gen_builder_table_decls(fb_output_t *out)
{
    fb_symbol_t *sym;

    /*
     * Because tables are recursive, we need the type and `start/end/add`
     * operations before the fields. We also need create for push_create
     * but it needs all dependent types, so create is fw declared
     * in a subsequent step. The actual create impl. then follows
     * after the table fields.
     */
    for (sym = out->S->symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_table:
            gen_required_table_fields(out, (fb_compound_type_t *)sym);
            gen_builder_table(out, (fb_compound_type_t *)sym);
            fprintf(out->fp, "\n");
            break;
        default:
            continue;
        }
    }
    for (sym = out->S->symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_table:
            gen_builder_create_table_decl(out, (fb_compound_type_t *)sym);
            gen_builder_table_prolog(out, (fb_compound_type_t *)sym);
            fprintf(out->fp, "\n");
            break;
        default:
            continue;
        }
    }
    return 0;
}

static int gen_builder_tables(fb_output_t *out)
{
    fb_symbol_t *sym;
    for (sym = out->S->symbols; sym; sym = sym->link) {
        switch (sym->kind) {
        case fb_is_table:
            gen_builder_table_fields(out, (fb_compound_type_t *)sym);
            gen_builder_create_table(out, (fb_compound_type_t *)sym);
            gen_builder_clone_table(out, (fb_compound_type_t *)sym);
            fprintf(out->fp, "\n");
            break;
        default:
            continue;
        }
    }
    return 0;
}

static int gen_builder_footer(fb_output_t *out)
{
    gen_epilogue(out);
    fprintf(out->fp,
        "#endif /* %s_BUILDER_H */\n",
        out->S->basenameup);
    return 0;
}

int fb_gen_c_builder(fb_output_t *out)
{
    gen_builder_pretext(out);
    gen_builder_enums(out);
    gen_builder_structs(out);
    gen_builder_union_decls(out);
    gen_builder_table_decls(out);
    gen_builder_unions(out);
    gen_builder_tables(out);
    gen_builder_footer(out);
    return 0;
}
