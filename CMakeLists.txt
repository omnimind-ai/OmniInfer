cmake_minimum_required(VERSION 3.14.3)
project(htp_ops C CXX ASM)
# cmake_policy(SET CMP0115 OLD)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

if(HEXAGON_SDK_ROOT)
    include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
else()
    include(${HEXAGON_CMAKE_ROOT}/hexagon_fun.cmake)
endif()
set(ignoreVar ${DEPENDENCIES}${V})

# NOTE(hzx): QAIC generated code are located in ${CMAKE_CURRENT_BINARY_DIR}
set(common_incs
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${CMAKE_CURRENT_BINARY_DIR}
    ${HEXAGON_SDK_ROOT}/incs/
    ${HEXAGON_SDK_ROOT}/incs/stddef/
    ${HEXAGON_SDK_ROOT}/rtos/qurt/
    ${HEXAGON_SDK_ROOT}/utils/examples/
)
include_directories(${common_incs})

if(${OS_TYPE} MATCHES "HLOS")
    # HLOS dynamic binary
    add_library(htp_ops SHARED 
        ${CMAKE_CURRENT_BINARY_DIR}/htp_ops_stub.c
        ${HEXAGON_SDK_ROOT}/utils/examples/dsp_capabilities_utils.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/host/op_export.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/host/session.c
    )

    # Build interface files
    build_idl(include/htp_ops.idl htp_ops)

    # HLOS executable
    add_executable(htp_ops_test
        ${CMAKE_CURRENT_SOURCE_DIR}/src/host/test.c
    )
    
    # Add common HLOS-specific linker options
    set_common_compile_and_link_options(htp_ops_test)

    target_compile_definitions(htp_ops_test PUBLIC VERIFY_PRINT_ERROR)
    if(${CMAKE_SYSTEM_NAME} MATCHES "UbuntuARM")
        target_link_options(htp_ops_test PUBLIC -lpthread -ldl)
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Android")
        target_link_options(htp_ops_test PUBLIC -llog -ldl)
    endif()

    # Find the DSP from provided domain value, default is CDSP
    choose_dsprpc(${DSP_TYPE} dsprpc)
    link_custom_library(htp_ops ${dsprpc})

    add_dependencies(htp_ops_test htp_ops)
    target_link_libraries(htp_ops_test htp_ops)

    # Copy binaries from ship to CMAKE_CURRENT_BINARY_DIR folder
    copy_binaries(htp_ops_test htp_ops)

else()

    # Shared skel library
    add_library(htp_ops_skel SHARED
        ${CMAKE_CURRENT_BINARY_DIR}/htp_ops_skel.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/commu.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/power.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/hmx_mgr.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/worker_pool.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/mmap_mgr.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/op_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/op_tests.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/vtcm_mgr.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/flash_attn_sp_hdim.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/flash_attn.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/mat_mul.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/precompute_table.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/rms_norm.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/mm_benchmark.c
    )
    build_idl(include/htp_ops.idl htp_ops_skel)

    # Enable HMX inline assembly
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mhmx")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mhmx")

    # Link C++ libraries
    target_link_libraries(htp_ops_skel ${HEXAGON_LIB_DIR}/${HEXAGON_ARCH}/G0/pic/libc++abi.so.1)
    target_link_libraries(htp_ops_skel ${HEXAGON_LIB_DIR}/${HEXAGON_ARCH}/G0/pic/libc++.so.1)

    # Copy binaries from ship to CMAKE_CURRENT_BINARY_DIR folder
    copy_binaries(htp_ops_skel)

endif()
