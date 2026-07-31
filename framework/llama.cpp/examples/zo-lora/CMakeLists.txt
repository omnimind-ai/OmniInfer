set(TARGET llama-zo)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/ggml/include
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (GGML_HEXAGON)
    target_compile_definitions(${TARGET} PRIVATE GGML_USE_HEXAGON=1)
endif()
