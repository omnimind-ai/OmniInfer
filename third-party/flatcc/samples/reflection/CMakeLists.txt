include(CTest)

# Pre-generated headers from reflection.fbs are already present in include/flatcc/reflection

set(INC_DIR "${PROJECT_SOURCE_DIR}/include")
set(GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(FBS_DIR "${PROJECT_SOURCE_DIR}/samples/monster")

# Expected binary schema file
set(GEN_SCHEMA "${GEN_DIR}/monster.bfbs")

add_custom_command(
    OUTPUT ${GEN_SCHEMA}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${GEN_DIR}"
    COMMAND flatcc_cli --schema -o "${GEN_DIR}" "${FBS_DIR}/monster.fbs"
    DEPENDS
        flatcc_cli
        "${FBS_DIR}/monster.fbs"
    COMMENT "Generating FlatCC binary schema for monster.fbs"
    VERBATIM
)

add_custom_target(gen_monster_bfbs ALL
    DEPENDS ${GEN_SCHEMA}
)

add_executable(bfbs2json bfbs2json.c)

target_include_directories(bfbs2json PRIVATE "${GEN_DIR}" "${INC_DIR}")
target_link_libraries(bfbs2json flatccrt)
add_dependencies(bfbs2json gen_monster_bfbs)

if (FLATCC_TEST)
    add_test(bfbs2json bfbs2json${CMAKE_EXECUTABLE_SUFFIX} ${GEN_SCHEMA})
endif()
