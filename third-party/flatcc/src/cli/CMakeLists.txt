include_directories (
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/config"
)

add_executable(flatcc_cli
    flatcc_cli.c
)

target_link_libraries(flatcc_cli
    flatcc
)

# Rename because the libflatcc library and the flatcc executable would
# conflict if they had the same target name `flatcc`.
set_target_properties(flatcc_cli PROPERTIES OUTPUT_NAME flatcc)

if (FLATCC_INSTALL)
    install(TARGETS flatcc_cli DESTINATION bin)
endif()
