cmake_minimum_required(VERSION 3.22.1)

project("omniinfer-jni" VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

option(OMNIINFER_BACKEND_LLAMA_CPP "Enable llama.cpp backend" ON)
option(OMNIINFER_BACKEND_MNN "Enable MNN backend" OFF)
option(OMNIINFER_BACKEND_EXECUTORCH_QNN "Enable ExecuTorch QNN backend" OFF)

# ABI-specific settings
if(DEFINED ANDROID_ABI)
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_SYSTEM_ARCH "ARM")
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(GGML_SYSTEM_ARCH "x86")
    else()
        message(FATAL_ERROR "Unsupported ABI: ${ANDROID_ABI}")
    endif()
endif()

# Paths relative to OmniInfer repo root
get_filename_component(OMNIINFER_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../../.." ABSOLUTE)

# llama.cpp backend
if(OMNIINFER_BACKEND_LLAMA_CPP)
    set(LLAMA_CPP_DIR "${OMNIINFER_ROOT}/framework/llama.cpp" CACHE PATH "")
    if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "llama.cpp not found at ${LLAMA_CPP_DIR}")
    endif()
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    add_subdirectory(${LLAMA_CPP_DIR} build-llama)
    # Multimodal (vision) support via mtmd
    if(NOT DEFINED LLAMA_INSTALL_VERSION)
        set(LLAMA_INSTALL_VERSION "0.0.0" CACHE STRING "" FORCE)
    endif()
    set(LLAMA_TOOLS_INSTALL OFF CACHE BOOL "" FORCE)
    add_subdirectory(${LLAMA_CPP_DIR}/tools/mtmd build-mtmd)
endif()

# MNN backend
if(OMNIINFER_BACKEND_MNN)
    set(MNN_DIR "${OMNIINFER_ROOT}/framework/mnn" CACHE PATH "")
    if(NOT EXISTS "${MNN_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "MNN not found at ${MNN_DIR}")
    endif()
    set(MNN_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(MNN_BUILD_FOR_ANDROID_COMMAND ON CACHE BOOL "" FORCE)
    set(MNN_BUILD_LLM ON CACHE BOOL "" FORCE)
    set(MNN_BUILD_LLM_OMNI ON CACHE BOOL "" FORCE)
    set(MNN_LOW_MEMORY ON CACHE BOOL "" FORCE)
    set(MNN_CPU_WEIGHT_DEQUANT_GEMM ON CACHE BOOL "" FORCE)
    set(MNN_SUPPORT_TRANSFORMER_FUSE ON CACHE BOOL "" FORCE)
    set(MNN_ARM82 ON CACHE BOOL "" FORCE)
    set(MNN_KLEIDIAI_DEFAULT_ON ON CACHE BOOL "" FORCE)
    set(MNN_BUILD_CONVERTER OFF CACHE BOOL "" FORCE)
    set(MNN_BUILD_TEST OFF CACHE BOOL "" FORCE)
    set(MNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(MNN_BUILD_QUANTOOLS OFF CACHE BOOL "" FORCE)
    set(MNN_BUILD_DEMO OFF CACHE BOOL "" FORCE)
    add_subdirectory(${MNN_DIR} build-mnn)
endif()

# ExecuTorch QNN backend (pre-built libraries + runner source)
if(OMNIINFER_BACKEND_EXECUTORCH_QNN)
    set(ET_PREBUILT_DIR "${OMNIINFER_ROOT}/.local/runtime/android/executorch-qnn-sm8650" CACHE PATH
        "Path to pre-built ExecuTorch QNN libraries")
    if(NOT EXISTS "${ET_PREBUILT_DIR}/lib")
        message(FATAL_ERROR "ExecuTorch pre-built libs not found at ${ET_PREBUILT_DIR}/lib")
    endif()

    # Runner source files (compiled into JNI library)
    set(ET_RUNNER_SRCS
        ${ET_PREBUILT_DIR}/runner_src/runner.cpp
        ${ET_PREBUILT_DIR}/runner_src/decoder_runner.cpp
        ${ET_PREBUILT_DIR}/runner_src/prompt_processor.cpp
        ${ET_PREBUILT_DIR}/runner_src/token_generator.cpp
        ${ET_PREBUILT_DIR}/runner_src/kv_manager.cpp
        ${ET_PREBUILT_DIR}/runner_src/rpc_mem.cpp
        ${ET_PREBUILT_DIR}/runner_src/attention_sink_rope_runner.cpp
        ${ET_PREBUILT_DIR}/runner_src/lhd_token_generator.cpp
    )

    # Import pre-built static libraries
    set(ET_STATIC_LIBS
        executorch executorch_core
        extension_module extension_data_loader extension_tensor
        extension_flat_tensor extension_named_data_map extension_llm_runner
        extension_llm_sampler extension_runner_util extension_evalue_util
        extension_threadpool
        portable_kernels portable_ops_lib kernels_util_all_deps
        quantized_kernels quantized_ops_lib
        full_portable_ops_lib custom_ops
        tokenizers sentencepiece re2 regex_lookahead pcre2-8
        flatccrt cpuinfo pthreadpool etdump bundled_program
        gflags_nothreads
    )
    # Add all abseil libraries
    file(GLOB ET_ABSL_LIBS "${ET_PREBUILT_DIR}/lib/libabsl_*.a")

    foreach(lib_name ${ET_STATIC_LIBS})
        set(lib_path "${ET_PREBUILT_DIR}/lib/lib${lib_name}.a")
        if(EXISTS "${lib_path}")
            add_library(et_${lib_name} STATIC IMPORTED)
            set_target_properties(et_${lib_name} PROPERTIES IMPORTED_LOCATION "${lib_path}")
        endif()
    endforeach()

    # QNN backend shared library (loaded at runtime via dlopen)
    # Not linked at build time — loaded dynamically from device path
endif()

# OmniInfer JNI bridge
add_library(${CMAKE_PROJECT_NAME} SHARED omniinfer_jni.cpp)

if(OMNIINFER_BACKEND_LLAMA_CPP)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE OMNIINFER_BACKEND_LLAMA_CPP=1)
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
            ${LLAMA_CPP_DIR} ${LLAMA_CPP_DIR}/common ${LLAMA_CPP_DIR}/include
            ${LLAMA_CPP_DIR}/ggml/include ${LLAMA_CPP_DIR}/ggml/src
            ${LLAMA_CPP_DIR}/tools/mtmd)
    target_link_libraries(${CMAKE_PROJECT_NAME} llama common mtmd)
endif()

if(OMNIINFER_BACKEND_MNN)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE OMNIINFER_BACKEND_MNN=1)
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
            ${MNN_DIR}/include ${MNN_DIR}/transformers/llm/engine/include ${MNN_DIR}/3rd_party)
    target_link_libraries(${CMAKE_PROJECT_NAME} MNN)
endif()

if(OMNIINFER_BACKEND_EXECUTORCH_QNN)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE OMNIINFER_BACKEND_EXECUTORCH_QNN=1)
    target_sources(${CMAKE_PROJECT_NAME} PRIVATE ${ET_RUNNER_SRCS})
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
            ${ET_PREBUILT_DIR}/include
            ${ET_PREBUILT_DIR}/include/executorch)
    # Link all pre-built static libs that exist
    foreach(lib_name ${ET_STATIC_LIBS})
        if(TARGET et_${lib_name})
            target_link_libraries(${CMAKE_PROJECT_NAME} et_${lib_name})
        endif()
    endforeach()
    # Link abseil libraries
    foreach(absl_lib ${ET_ABSL_LIBS})
        target_link_libraries(${CMAKE_PROJECT_NAME} ${absl_lib})
    endforeach()
    target_link_libraries(${CMAKE_PROJECT_NAME} dl)
endif()

target_link_libraries(${CMAKE_PROJECT_NAME} android log)
