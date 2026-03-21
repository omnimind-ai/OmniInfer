include_directories (
    "${PROJECT_SOURCE_DIR}/external"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/config"
)

set (SOURCES
    # metrohash is not needed unless HT_HASH_FALLBACK is defined.
    # xxhash inline header is now used by default.
    # ${PROJECT_SOURCE_DIR}/external/hash/cmetrohash64.c
    ${PROJECT_SOURCE_DIR}/external/hash/str_set.c
    ${PROJECT_SOURCE_DIR}/external/hash/ptr_set.c
    hash_tables/symbol_table.c
    hash_tables/scope_table.c
    hash_tables/name_table.c
    hash_tables/schema_table.c
    hash_tables/value_set.c
    fileio.c
    parser.c
    semantics.c
    coerce.c
    flatcc.c
    codegen_c.c
    codegen_c_reader.c
    codegen_c_sort.c
    codegen_c_builder.c
    codegen_c_verifier.c
    codegen_c_sorter.c
    codegen_c_json_parser.c
    codegen_c_json_printer.c
    # needed for building binary schema
    ../runtime/builder.c
    ../runtime/emitter.c
    ../runtime/refmap.c
)

if (FLATCC_REFLECTION)
    set (SOURCES ${SOURCES} codegen_schema.c)
endif(FLATCC_REFLECTION)

add_library(flatcc ${SOURCES})

if (FLATCC_INSTALL)
    install(TARGETS flatcc DESTINATION ${lib_dir})
endif()
