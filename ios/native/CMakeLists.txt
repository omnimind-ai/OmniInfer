cmake_minimum_required(VERSION 3.22.1)
project("omniinfer-bridge" VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

option(OMNIINFER_BACKEND_LLAMA_CPP "Enable llama.cpp backend" ON)

get_filename_component(OMNIINFER_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

# llama.cpp (already built via add_subdirectory from the parent build script)
if(OMNIINFER_BACKEND_LLAMA_CPP)
    set(LLAMA_CPP_DIR "${OMNIINFER_ROOT}/framework/llama.cpp" CACHE PATH "")
endif()

add_library(omniinfer-bridge STATIC omniinfer_bridge.cpp)

if(OMNIINFER_BACKEND_LLAMA_CPP)
    target_compile_definitions(omniinfer-bridge PRIVATE OMNIINFER_BACKEND_LLAMA_CPP=1)
    target_include_directories(omniinfer-bridge PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LLAMA_CPP_DIR}/include
        ${LLAMA_CPP_DIR}/common
        ${LLAMA_CPP_DIR}/ggml/include
    )
    target_link_libraries(omniinfer-bridge llama common)
endif()

# Install the public header alongside the library.
install(FILES omniinfer_bridge.h DESTINATION include)
