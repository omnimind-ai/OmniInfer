message(STATUS "Using My Hexagon backend")

set(GGML_SOURCES_HTP 
    ggml-htp.cc
    dsprpc_interface.h
    dsprpc_interface.cc
    ggml-htp-impl.h
    htp-cpu-impl.c
    htp-ops.cc
    htp-ops.h
    rpcmem_mapper.cc
    rpcmem_mapper.h
)

ggml_add_backend_library(ggml-htp ${GGML_SOURCES_HTP})

# FIXME(hzx): why NDK build need this to avoid linking errors?
if (CMAKE_SYSTEM_NAME MATCHES "Android")
    target_link_libraries(ggml-htp PRIVATE ggml-cpu)
endif()
