set(INC_DIR "${PROJECT_SOURCE_DIR}/include")
set(GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(FBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Choose a reliable generated output file
set(GEN_HDR "${GEN_DIR}/monster_reader.h")

add_custom_command(
    OUTPUT ${GEN_HDR}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${GEN_DIR}"
    COMMAND flatcc_cli -a -o "${GEN_DIR}" "${FBS_DIR}/monster.fbs"
    DEPENDS flatcc_cli "${FBS_DIR}/monster.fbs"
    COMMENT "Generating FlatCC output from monster.fbs"
)

add_custom_target(gen_monster_fbs ALL
    DEPENDS ${GEN_HDR}
)

add_executable(monster monster.c)

target_include_directories(monster PRIVATE
    "${GEN_DIR}"
    "${INC_DIR}"
)

target_link_libraries(monster flatccrt)
add_dependencies(monster gen_monster_fbs)

if (FLATCC_TEST)
    add_test(monster monster${CMAKE_EXECUTABLE_SUFFIX})
endif()
