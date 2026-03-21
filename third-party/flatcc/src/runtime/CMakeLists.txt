add_library(flatccrt
    builder.c
    emitter.c
    refmap.c
    verifier.c
    json_parser.c
    json_printer.c
)

target_include_directories(flatccrt PUBLIC
    "${PROJECT_SOURCE_DIR}/include")

if (FLATCC_INSTALL)
    install(TARGETS flatccrt DESTINATION ${lib_dir})
endif()
