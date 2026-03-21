#include <stdio.h>
#include <stdlib.h>
#include "flatcc/reflection/reflection_builder.h"
#include "symbols.h"
#include "parser.h"
#include "codegen.h"
#include "fileio.h"
/* Needed to store length prefix. */
#include "catalog.h"

#define BaseType(x) FLATBUFFERS_WRAP_NAMESPACE(reflection_BaseType, x)

static flatbuffers_bool_t is_optional_type(fb_value_t type, int optional, int required)
{
    if (required) return 0;
    if (optional) return 1;
    if (type.type == vt_scalar_type) return 0;
    if (type.type == vt_compound_type_ref && type.ct->symbol.kind == fb_is_enum) return 0;
    return 1;
}

static reflection_Type_ref_t export_type(flatcc_builder_t *B, fb_value_t type)
{
    fb_scalar_type_t st = fb_missing_type;
    int32_t index = -1;
    reflection_BaseType_enum_t base_type = BaseType(None);
    reflection_BaseType_enum_t element = BaseType(None);
    reflection_BaseType_enum_t primitive = BaseType(None);
    uint16_t fixed_length = 0;

    switch (type.type) {
    case vt_scalar_type:
        st = type.st;
        break;
    case vt_vector_type:
        st = type.st;
        base_type = BaseType(Vector);
        break;
    case vt_vector_string_type:
        element = BaseType(String);
        base_type = BaseType(Vector);
        break;
    case vt_vector_compound_type_ref:
        index = (int32_t)type.ct->export_index;
        switch (type.ct->symbol.kind) {
        case fb_is_enum:
            st = type.ct->type.st;
            base_type = BaseType(Vector);
            break;
        case fb_is_struct:
        case fb_is_table:
            base_type = BaseType(Vector);
            element = BaseType(Obj);
            break;
        case fb_is_union:
            base_type = BaseType(Vector);
            element = BaseType(Union);
            break;
        default:
            break;
        }
        break;
    case vt_string_type:
        base_type = BaseType(String);
        break;
    case vt_compound_type_ref:
        index = (int32_t)type.ct->export_index;
        switch (type.ct->symbol.kind) {
        case fb_is_enum:
            st = type.ct->type.st;
            break;
        case fb_is_struct:
        case fb_is_table:
            base_type = BaseType(Obj);
            break;
        case fb_is_union:
            base_type = BaseType(Union);
            break;
        default:
            index = -1;
            break;
        }
        break;
    case vt_fixed_array_type:
        st = type.st;
        base_type = BaseType(Array);
        fixed_length = (uint16_t)type.len;
        break;
    case vt_fixed_array_string_type:
        break;
        element = BaseType(Byte);
        base_type = BaseType(Array);
        fixed_length = (uint16_t)type.len;
        break;
    case vt_fixed_array_compound_type_ref:
        index = (int32_t)type.ct->export_index;
        switch (type.ct->symbol.kind) {
        case fb_is_enum:
            st = type.ct->type.st;
            break;
        case fb_is_struct:
        case fb_is_table:
            element = BaseType(Obj);
            break;
        case fb_is_union:
            element = BaseType(Union);
            break;
        default:
            break;
        }
        base_type = BaseType(Array);
        fixed_length = (uint16_t)type.len;
        break;
    default:
        break;
    }
    /* If st is set, resolve scalar type and set it to base_type or element. */
    switch (st) {
    case fb_missing_type: break;
    case fb_ulong: primitive = BaseType(ULong); break;
    case fb_uint: primitive = BaseType(UInt); break;
    case fb_ushort: primitive = BaseType(UShort); break;
    case fb_ubyte: primitive = BaseType(UByte); break;
    case fb_bool: primitive = BaseType(Bool); break;
    case fb_long: primitive = BaseType(Long); break;
    case fb_int: primitive = BaseType(Int); break;
    case fb_short: primitive = BaseType(Short); break;
    case fb_byte: primitive = BaseType(Byte); break;
    case fb_double: primitive = BaseType(Double); break;
    case fb_float: primitive = BaseType(Float); break;
    /* TODO: Googles flatc tool does not have char arrays so we use Byte as element type */
    case fb_char: primitive = BaseType(Byte); break;
    default: break;
    }

    if (base_type == BaseType(None)) {
        base_type = primitive;
    } else if (base_type == BaseType(Vector) || base_type == BaseType(Array)) {
        if (element == BaseType(None)) {
            element = primitive;
        }
    }
    return reflection_Type_create(B, base_type, element, index, fixed_length);
}

static void export_attributes(flatcc_builder_t *B, fb_metadata_t *m)
{
    for (; m; m = m->link) {
        reflection_KeyValue_vec_push_start(B);
        reflection_KeyValue_key_create_strn(B, m->ident->text, (size_t)m->ident->len);
        if (m->value.type == vt_string) {
            reflection_KeyValue_value_create_strn(B, m->value.s.s, (size_t)m->value.s.len);
        }
        reflection_KeyValue_vec_push_end(B);
    }
}

static void export_fields(flatcc_builder_t *B, fb_compound_type_t *ct)
{
    fb_symbol_t *sym;
    fb_member_t *member;
    flatbuffers_bool_t has_key, deprecated, required, optional, key_processed = 0;
    int64_t default_integer;
    double default_real;

    for (sym = ct->members; sym; sym = sym->link) {
        member = (fb_member_t *)sym;
        /*
         * Unlike `flatc` we allow multiple keys in the parser, but
         * there is no way to tell which key is default in the
         * reflection schema because the fields are sorted, so we only
         * export the default (first) key.
         */
        has_key = !key_processed && (member->metadata_flags & fb_f_key) != 0;
        required = (member->metadata_flags & fb_f_required) != 0;
        default_integer = 0;
        default_real = 0.0;
        deprecated = (member->metadata_flags & fb_f_deprecated) != 0;
        /*
         * Flag is only set when `= null` is used in the schema, but
         * non-scalar types are optional by default and therfore also
         * true in the binary schema.
         */
        optional = is_optional_type(member->type, !!(member->flags & fb_fm_optional), required);

        if ((member->type.type == vt_compound_type_ref || member->type.type == vt_vector_compound_type_ref)
                && member->type.ct->symbol.kind == fb_is_union) {
            reflection_Field_vec_push_start(B);
            reflection_Field_name_start(B);
            reflection_Field_name_append(B, member->symbol.ident->text, (size_t)member->symbol.ident->len);
            reflection_Field_name_append(B, "_type", 5);
            reflection_Field_name_end(B);
            switch(member->type.type) {
            case vt_compound_type_ref:
                reflection_Field_type_create(B, BaseType(UType), BaseType(None), -1, 0);
                break;
            case vt_vector_compound_type_ref:
                reflection_Field_type_create(B, BaseType(Vector), BaseType(UType), -1, 0);
                break;
            }
            reflection_Field_offset_add(B, (uint16_t)(member->id - 1 + 2) * sizeof(flatbuffers_voffset_t));
            reflection_Field_id_add(B, (uint16_t)(member->id - 1));
            reflection_Field_deprecated_add(B, deprecated);
            reflection_Field_vec_push_end(B);
        }
        reflection_Field_vec_push_start(B);
        reflection_Field_name_create(B, member->symbol.ident->text, (size_t)member->symbol.ident->len);
        reflection_Field_type_add(B, export_type(B, member->type));
        switch (ct->symbol.kind) {
        case fb_is_table:
            switch (member->value.type) {
            case vt_uint:
                default_integer = (int64_t)member->value.u;
                break;
            case vt_int:
                default_integer = (int64_t)member->value.i;
                break;
            case vt_bool:
                default_integer = (int64_t)member->value.b;
                break;
            case vt_float:
                default_real = member->value.f;
                break;
            }
            reflection_Field_default_integer_add(B, default_integer);
            reflection_Field_default_real_add(B, default_real);
            reflection_Field_id_add(B, (uint16_t)member->id);
            reflection_Field_offset_add(B, (uint16_t)(member->id + 2) * sizeof(flatbuffers_voffset_t));
            reflection_Field_key_add(B, has_key);
            reflection_Field_required_add(B, required);
            reflection_Field_optional_add(B, optional);
            break;
        case fb_is_struct:
            reflection_Field_offset_add(B, (uint16_t)member->offset);
            break;
        default: break;
        }
        /* Deprecated struct fields not supported by `flatc` but is here as an option. */
        reflection_Field_deprecated_add(B, deprecated);
        if (member->metadata) {
            reflection_Field_attributes_start(B);
            export_attributes(B, member->metadata);
            reflection_Field_attributes_end(B);
        }
        reflection_Field_vec_push_end(B);
        key_processed |= has_key;
    }
}

/* `vec` is filled with references to the constructed objects. */
static void export_objects(flatcc_builder_t *B, object_entry_t *objects, int nobjects,
        reflection_Object_ref_t *object_map)
{
    int i, is_struct;
    fb_compound_type_t *ct;

    for (i = 0; i < nobjects; ++i) {
        ct = objects[i].ct;
        reflection_Object_start(B);
        reflection_Object_name_create_str(B, objects[i].name);
        /*
         * We can post sort-fields because the index is not used, unlike
         * objects and enums.
         */
        reflection_Object_fields_start(B);
        export_fields(B, ct);
        reflection_Object_fields_end(B);
        is_struct = ct->symbol.kind == fb_is_struct;
        if (is_struct) {
            reflection_Object_bytesize_add(B, (int32_t)ct->size);
        }
        reflection_Object_is_struct_add(B, (flatbuffers_bool_t)is_struct);
        reflection_Object_minalign_add(B, ct->align);
        if (ct->metadata) {
            reflection_Object_attributes_start(B);
            export_attributes(B, ct->metadata);
            reflection_Object_attributes_end(B);
        }
        object_map[i] = reflection_Object_end(B);
    }
    reflection_Schema_objects_create(B, object_map, (size_t)nobjects);
}

static void export_enumval(flatcc_builder_t *B, fb_member_t *member, reflection_Object_ref_t *object_map)
{
    int is_union = object_map != 0;

    reflection_EnumVal_vec_push_start(B);
    reflection_EnumVal_name_create(B, member->symbol.ident->text, (size_t)member->symbol.ident->len);
    if (is_union) {
        if (member->type.type == vt_compound_type_ref) {
            /* object is deprecated in favor of union_type to support mixed union types. */
            reflection_EnumVal_object_add(B, object_map[member->type.ct->export_index]);
        }
        reflection_EnumVal_union_type_add(B, export_type(B, member->type));
    }
    reflection_EnumVal_value_add(B, (int64_t)member->value.u);
    reflection_EnumVal_vec_push_end(B);
}

static void export_enums(flatcc_builder_t *B, enum_entry_t *enums, int nenums,
        reflection_Object_ref_t *object_map)
{
    int i, is_union;
    fb_compound_type_t *ct;
    fb_symbol_t *sym;

    reflection_Schema_enums_start(B);
    for (i = 0; i < nenums; ++i) {
        ct = enums[i].ct;
        is_union = ct->symbol.kind == fb_is_union;
        reflection_Enum_vec_push_start(B);
        reflection_Enum_name_create_str(B, enums[i].name);
        reflection_Enum_values_start(B);
        for (sym = ct->members; sym; sym = sym->link) {
            export_enumval(B, (fb_member_t *)sym, is_union ? object_map : 0);
        }
        reflection_Enum_values_end(B);
        reflection_Enum_is_union_add(B, (flatbuffers_bool_t)is_union);
        reflection_Enum_underlying_type_add(B, export_type(B, ct->type));
        if (ct->metadata) {
            reflection_Enum_attributes_start(B);
            export_attributes(B, ct->metadata);
            reflection_Enum_attributes_end(B);
        }
        reflection_Enum_vec_push_end(B);
    }
    reflection_Schema_enums_end(B);
}

static void export_root_type(flatcc_builder_t *B, fb_symbol_t * root_type,
        reflection_Object_ref_t *object_map)
{
    fb_compound_type_t *ct;
    if (root_type) {
        /*
         * We could also store a struct object here, but since the
         * binrary schema says root_table, not root_type as in the text
         * schema, it would be misleading.
         */
        if (root_type->kind == fb_is_table) {
            ct = (fb_compound_type_t *)root_type;
            reflection_Schema_root_table_add(B, object_map[ct->export_index]);
        }
    }
}

static void export_call(flatcc_builder_t *B, fb_member_t *member, reflection_Object_ref_t *object_map)
{
    reflection_RPCCall_vec_push_start(B);
    reflection_RPCCall_name_create(B, member->symbol.ident->text, (size_t)member->symbol.ident->len);
    reflection_RPCCall_request_add(B, object_map[member->req_type.ct->export_index]);
    reflection_RPCCall_response_add(B, object_map[member->type.ct->export_index]);
    if (member->metadata) {
        reflection_RPCCall_attributes_start(B);
        export_attributes(B, member->metadata);
        reflection_RPCCall_attributes_end(B);
    }
    reflection_RPCCall_vec_push_end(B);
}

static void export_services(flatcc_builder_t *B, service_entry_t *services, int nservices,
        reflection_Object_ref_t *object_map)
{
    int i;
    fb_compound_type_t *ct;
    fb_symbol_t *sym;

    reflection_Schema_services_start(B);
    for (i = 0; i < nservices; ++i) {
        ct = services[i].ct;
        reflection_Service_vec_push_start(B);
        reflection_Service_name_create_str(B, services[i].name);
        reflection_Service_calls_start(B);
        for (sym = ct->members; sym; sym = sym->link) {
            export_call(B, (fb_member_t *)sym, object_map);
        }
        reflection_Service_calls_end(B);
        if (ct->metadata) {
            reflection_Service_attributes_start(B);
            export_attributes(B, ct->metadata);
            reflection_Service_attributes_end(B);
        }
        reflection_Service_vec_push_end(B);
    }
    reflection_Schema_services_end(B);
}

static int export_schema(flatcc_builder_t *B, fb_options_t *opts, fb_schema_t *S)
{
    catalog_t catalog;
    reflection_Object_ref_t *object_map = 0;

    if (build_catalog(&catalog, S, opts->bgen_qualify_names, &S->root_schema->scope_index)) {
        return -1;
    }

    if (catalog.nobjects > 0 && !(object_map = malloc((size_t)catalog.nobjects * sizeof(object_map[0])))) {
        clear_catalog(&catalog);
        return -1;
    }

    /* Build the schema. */

    if (opts->bgen_length_prefix) {
        reflection_Schema_start_as_root_with_size(B);
    } else {
        reflection_Schema_start_as_root(B);
    }
    if (S->file_identifier.type == vt_string) {
        reflection_Schema_file_ident_create(B,
                S->file_identifier.s.s, (size_t)S->file_identifier.s.len);
    }
    if (S->file_extension.type == vt_string) {
        reflection_Schema_file_ext_create(B,
                S->file_extension.s.s, (size_t)S->file_extension.s.len);
    }
    export_objects(B, catalog.objects, catalog.nobjects, object_map);
    export_enums(B, catalog.enums, catalog.nenums, object_map);
    export_root_type(B, S->root_type.type, object_map);
    export_services(B, catalog.services, catalog.nservices, object_map);

    reflection_Schema_end_as_root(B);

    /* Clean up support datastructures. */

    clear_catalog(&catalog);
    if (object_map) {
        free(object_map);
    }
    return 0;
}

/*
 * We do not not sort attributes because we would loose ordering
 * information between different attributes, and between same named
 * attributes because the sort is not stable.
 *
 * The C bindings has a scan interface that can find attributes
 * in order of appearance.
 *
 * Field sorting is done on the finished buffer.
 */
static void sort_objects(void *buffer)
{
    size_t i;
    reflection_Schema_table_t schema;
    reflection_Object_vec_t objects;
    reflection_Object_table_t object;
    reflection_Field_vec_t fields;
    reflection_Field_mutable_vec_t mfields;

    schema = reflection_Schema_as_root(buffer);
    objects = reflection_Schema_objects(schema);
    for (i = 0; i < reflection_Object_vec_len(objects); ++i) {
        object = reflection_Object_vec_at(objects, i);
        fields = reflection_Object_fields(object);
        if (fields) {
            mfields = (reflection_Field_mutable_vec_t)fields;
            reflection_Field_vec_sort(mfields);
        }
    }
}

static FILE *open_file(fb_options_t *opts, fb_schema_t *S)
{
    FILE *fp = 0;
    char *path = 0, *ext = 0;
    const char *prefix = opts->outpath ? opts->outpath : "";
    size_t len, prefix_len = strlen(prefix);
    const char *name;

    name = S->basename;
    len = strlen(name);

    ext = fb_create_path_ext(".", flatbuffers_extension);
    /* We generally should not use cgen options here, but in this case it makes sense. */
    if (opts->gen_stdout) {
        return stdout;
    }
    if (opts->gen_outfile) {
        path = fb_create_join_path(prefix, opts->gen_outfile, "", 1);
    } else {
        path = fb_create_join_path_n(prefix, prefix_len, name, len, ext, 1);
    }
    checkmem(path);
    fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "error opening file for writing binary schema: %s\n", path);
    }
    free(path);
    free(ext);
    return fp;
}

static void close_file(FILE *fp)
{
    if (fp && fp != stdout) {
        fclose(fp);
    }
}

/*
 * Normally enums are required to be ascending in the schema and
 * therefore there is no need to sort enums. If not, we export them in
 * the order defined anyway becuase there is no well-defined ordering
 * and blindly sorting the content would just loose more information.
 *
 * In conclusion: find by enum value is only supported when enums are
 * defined in consequtive order.
 *
 * refers to: `opts->ascending_enum`
 *
 * `size` must hold the maximum buffer size.
 * Returns intput buffer if successful and updates size argument.
 */
void *fb_codegen_bfbs_to_buffer(fb_options_t *opts, fb_schema_t *S, void *buffer, size_t *size)
{
    flatcc_builder_t builder, *B;

    B = &builder;
    flatcc_builder_init(B);
    export_schema(B, opts, S);
    if (!flatcc_builder_copy_buffer(B, buffer, *size)) {
        goto done;
    }
    sort_objects(buffer);
done:
    *size = flatcc_builder_get_buffer_size(B);
    flatcc_builder_clear(B);
    return buffer;
}

/*
 * Like to_buffer, but returns allocated buffer.
 * Updates size argument with buffer size if not null.
 * Returned buffer must be deallocatd with `free`.
 * The buffer is malloc aligned which should suffice for reflection buffers.
 */
void *fb_codegen_bfbs_alloc_buffer(fb_options_t *opts, fb_schema_t *S, size_t *size)
{
    flatcc_builder_t builder, *B;
    void *buffer = 0;

    B = &builder;
    flatcc_builder_init(B);
    if (export_schema(B, opts, S)) {
        goto done;
    }
    if (!(buffer = flatcc_builder_finalize_buffer(B, size))) {
        goto done;
    }
    sort_objects(buffer);
done:
    flatcc_builder_clear(B);
    return buffer;
}

int fb_codegen_bfbs_to_file(fb_options_t *opts, fb_schema_t *S)
{
    void *buffer;
    size_t size;
    FILE *fp;
    int ret = -1;

    fp = open_file(opts, S);
    if (!fp) {
        return -1;
    }
    buffer = fb_codegen_bfbs_alloc_buffer(opts, S, &size);
    if (!buffer) {
        fprintf(stderr, "failed to generate binary schema\n");
        goto done;
    }
    if (size != fwrite(buffer, 1, size, fp)) {
        fprintf(stderr, "could not write binary schema to file\n");
        goto done;
    }
    ret = 0;
done:
    if (buffer) {
        free(buffer);
    }
    close_file(fp);
    return ret;
}
