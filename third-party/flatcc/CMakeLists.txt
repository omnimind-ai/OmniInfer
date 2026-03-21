# Debian 12 (Bookworm)
# cmake_minimum_required (VERSION 3.25)

# Appveyor for MSVC 2015
cmake_minimum_required (VERSION 3.16)


# Experimental for generating compile_commands.json so editors with
# clangd language server support can use it. Symlink
# build/Debug/compile_commands.json to project root where it is
# gitignored.
#set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Disable build of tests and samples. Due to custom build step
# dependency on flatcc tool, some custom build configurations may
# experience issues, and this option can then help.
option(FLATCC_TEST "enable tests" ON)

# Only active if FLATCC_TEST is active. Used to ensure that C++ users
# can include generatd C source. Old GCC pre 4.7 won't compile C++ test
# project.
option(FLATCC_CXX_TEST "enable C++ tests" ON)

# Note that linking with flatcc debug libraries may require souce code to also use
# the sanitize flag.
option(FLATCC_DEBUG_CLANG_SANITIZE "enable clang sanitize flag for debug build" ON)

# Conditionally set project languages based on FLATCC_TEST, as C++ is
# only necessary if building the tests.
if (FLATCC_TEST AND FLATCC_CXX_TEST)
    project (FlatCC C CXX)
else()
    project (FlatCC C)
endif()

#
# NOTE: when changing build options, clean the build using on of:
#
#   scripts/cleanall.sh
#   scripts/test.sh
#

# Force use of portable shims such as providing `static_assert`, and
# `stdaligh.h`. Otherwise this option is automatically enabled for some
# known compiler configurations below.
option (FLATCC_PORTABLE
    "include extra headers for compilers that do not support certain C11 features" OFF)

# It is not possible to detect posix_memalign when compiling with
# -std=c11 but aligned_alloc is not always available either.
# This options assumes that posix_memalign is then available.
# Without C11, detection depends on _POSIX_C_SOURCE.
option (FLATCC_GNU_POSIX_MEMALIGN
    "use posix_memalign on gnu systems also when C11 is configured" ON)

# Only build the runtime library - mostly intended in combination with
# FLATCC_INSTALL for cross compiling targets.
option(FLATCC_RTONLY "enable build of runtime library only" OFF)

# Use with or witout FLATCC_RTONLY to enable install targets.
# Libraries are built statically by default, but can CMake's
# cmake -DBUILD_SHARED_LIBS=on can override.
option(FLATCC_INSTALL "enable install targets" OFF)

# Use with debug build with testing enabled only. Enables generation
# of coverage information during build and run. Adds target "coverage"
# which collects data and makes HTML report in build directory
option(FLATCC_COVERAGE "enable coverage" OFF)

# Affects the flatbuffer verify operation. Normally a verify should just
# quickly reject invalid buffers but for troubleshooting, assertions can
# enabled. This requires rebuilding the runtime library and will likely
# break test cases (those that tests that an invalid buffer is invalid).
option (FLATCC_DEBUG_VERIFY
    "assert on verify failure in runtime lib" OFF)

# Print detailed traces of binary buffer contents when calling verify.
option (FLATCC_TRACE_VERIFY
    "assert on verify failure in runtime lib" OFF)

# Some producers allow empty vectors to be misaligned.
# The following setting will cause the verifier to require the index 0
# position to be element aligned even if the vector is empty (otherwise that
# position is only required to be aligned to the preceding size field).
option (FLATCC_ENFORCE_ALIGNED_EMPTY_VECTORS
    "verify includes full alignment check for empty vectors" OFF)

# Reflection is the compilers ability to generate binary schema output
# (.bfbs files). This requires using generated code from
# `reflection.fbs`. During development it may not be possible to
# compile with reflection enabled because it can become impossible to
# fix broken builds. It may also be disabled simple because it isn't
# needed.
option (FLATCC_REFLECTION
    "generation of binary flatbuffer schema files" ON)

# FLATCC_NATIVE_OPTIM and FLATCC_FAST_DOUBLE affects json parsing,
# especially if the content is pretty printed. But it is plenty
# fast without these settings in most cases. Not recommended.
option (FLATCC_NATIVE_OPTIM
    "use machine native optimizations like SSE 4.2" OFF)

# Fast grisu3 string/floating point conversion still depends on strtod
# for about 1-2% of the conversions in order to produce an exact result.
# By allowing a minor difference in the least significant bits, this
# dependeny can be avoided, and speed improved. Some strtod
# implementations call strlen which is really slow on large JSON
# buffers, and catastrophic on buffers that are not zero-terminated -
# regardless of size. Most platforms have a decent strtod these days.
option (FLATCC_FAST_DOUBLE
    "faster but slightly incorrect floating point parser (json)" OFF)

# -Werror is only set for some compiler versions that are believed to
# to not generate any warnings. If the assumption breaks, disable
# this option if the warning is not significant.
option (FLATCC_ALLOW_WERROR "allow -Werror to be configured" ON)

# Experimental setting - sometimes the code branches on a constant
# expression in order to select the best option for a given type size or
# similar. Sometimes compilers don't like that. If this issue surfaces,
# try using this option.
option (FLATCC_IGNORE_CONST_COND "silence const condition warnings" OFF)

if (FLATCC_RTONLY)
    set(FLATCC_TEST off)
endif()

if (FLATCC_TEST)
    enable_testing()
endif()

if (NOT FLATCC_TEST)
    set(FLATCC_COVERAGE off)
endif()

if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
    set(FLATCC_COVERAGE off)
endif()

if (FLATCC_COVERAGE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage -DNDEBUG")
endif()

if (FLATCC_DEBUG_VERIFY)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DFLATCC_DEBUG_VERIFY=1")
endif()

if (FLATCC_TRACE_VERIFY)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DFLATCC_TRACE_VERIFY=1")
endif()

if (FLATCC_ENFORCE_ALIGNED_EMPTY_VECTORS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DFLATCC_ENFORCE_ALIGNED_EMPTY_VECTORS=1")
endif()


if (FLATCC_REFLECTION)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DFLATCC_REFLECTION=1")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DFLATCC_REFLECTION=0")
endif()


if (FLATCC_NATIVE_OPTIM)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -DFLATCC_USE_SSE4_2=1")
endif()

if (FLATCC_FAST_DOUBLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DGRISU3_PARSE_ALLOW_ERROR -DFLATCC_USE_GRISU3=1")
endif()

if (NOT DEFINED FLATCC_INSTALL_LIB)
    set(lib_dir lib)
else()
    set(lib_dir ${FLATCC_INSTALL_LIB})
endif()

# The folder of this directory, as apposed to CMAKE_BINARY_DIR
# which would usually be the build/Release and build/Debug paths
set (dist_dir "${PROJECT_SOURCE_DIR}")
# set (dist_dir "${CMAKE_BINARY_DIR}")

message(STATUS "dist install dir ${dist_dir}")
message(STATUS "lib install dir ${dist_dir}/${lib_dir}")

# Note: for compiling generated C code, warnings of unused functions
# and constants should be turned off - those are plentiful. They are
# silenced for Clang, GCC and MSVC in generated headers.headers.

if (CMAKE_C_COMPILER_ID MATCHES "Clang" AND NOT "${CMAKE_CXX_SIMULATE_ID}" STREQUAL "MSVC")
    # Clang or AppleClang
    message(STATUS "Setting Clang compiler options")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wstrict-prototypes")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wsign-conversion")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wconversion")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -pedantic -Wall -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall -Wextra")
    # Fix broken C++ alignas - either will do
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPORTABLE_PATCH_CPLUSPLUS_STDALIGN")
    if (FLATCC_ALLOW_WERROR)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
    endif()
    if (FLATCC_IGNORE_CONST_COND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-tautological-constant-out-of-range-compare")
    endif()
    if (FLATCC_DEBUG_CLANG_SANITIZE)
        if (CMAKE_BUILD_TYPE MATCHES Debug)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=undefined")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
        endif()
    endif()
    # Suppress warning relaxed in clang-6, see https://reviews.llvm.org/D28148
    if (CMAKE_C_COMPILER_VERSION VERSION_LESS 6)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-missing-field-initializers")
    endif()

    # To get assembly output
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -save-temps")

elseif (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion
                    OUTPUT_VARIABLE GCC_VERSION)
    if (GCC_VERSION VERSION_LESS 4.7)
        message(STATUS "Setting older GNU C compiler options with FLATCC_PORTABLE")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
        # We need stdalign.h
        set(FLATCC_PORTABLE true)
        # Disable C++ test for old compilers known to break due to
        # missing stdalign.h and incomplete stdint.h which is not a
        # priority to fix in portable library for C++ use case.
        # Note: we test the C compiler version not the C++ compiler
        # version, but that is (hopefully) close enough.
        if (FLATCC_CXX_TEST)
            message(STATUS "Disabling C++ tests for GCC pre 4.7")
            set(FLATCC_CXX_TEST false)
        endif()
    else()
        message(STATUS "Setting GNU C compiler options with c11 and Posix")
        if (GCC_VERSION VERSION_LESS 8.0)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -pedantic -Wall -Wextra")
        elseif (NOT (GCC_VERSION VERSION_LESS 8.0))
            # Disable some GCC checks:
            # (warnings exist since 8.0, but are more aggressive in 9.0)
            #
            # -Wstringop-truncation:
            # GCC 9 warns on truncated strncpy into char arrays in FlatBuffer
            # structs, but these are valid as zero-paddded, not zero terminated.
            #
            # -Wno-format-overflow:
            # GCC 9 warns on mistakenly assumed NULL string when
            # printing from a required FlatBuffer string field.
            #
            message(STATUS "Disabling -pedantic for GCC >= 8.0")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -Wall -Wextra")
            message(STATUS "Disabling GNU C compiler warnings: -Wstringop-truncation -Wno-format-overflow")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-stringop-truncation -Wno-format-overflow")
        endif()
        if (NOT (GCC_VERSION VERSION_LESS 11.0))
        # Disable warning on misleading indentation it become more aggressive in 11.0
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-misleading-indentation")
        endif()
        if (FLATCC_GNU_POSIX_MEMALIGN)
            # -std=c11 prevents detection of posix_memalign and aligned_alloc might be missing
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPORTABLE_POSIX_MEMALIGN=1")
        endif()
        if (FLATCC_ALLOW_WERROR)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
        endif()
    endif()
    if (FLATCC_IGNORE_CONST_COND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-type-limits")
    endif()

    # Too aggressive, e.g. main() is not permitted and main with
    # args then yields unused arg warning.
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wstrict-prototypes")

    # In gcc 4.8 it is not possible to suppress this warning using
    # #pragma GCC diagnostic ignored "-Wunused-function"
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-type-limits")

    if (GCC_VERSION VERSION_LESS 4.8)
        # -Wsign-conversion broken for GCC 4.7 conditional operator
    else()
        # Might be disabled if GCC keeps getting more agressive.
        # Incorrectly warns on explicit char to uint32_t casts.
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wsign-conversion")

        # Too aggressive, warns on `x = x + 1;` or `n = -n;`.
        # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wconversion")
    endif()

elseif (CMAKE_C_COMPILER_ID STREQUAL "Intel")
    message(STATUS "Setting Intel C (ICC) compiler options")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -Wall -Wextra")
    # warning #169: expected a declaration
    # Fallthrough attribute does not like semicolon.
    # this is fixed in Intel ICC 2021.5.0, but we only have 2021.4.0 in current CI build.
    # warning #279: controlling expression is constant
    # We have these everywhere in flatcc generated code.
    # warning #188: enumerated type mixed with another type
    # This is not very noisy, but still annoying, e.g. when zeroing an enum.
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -wd169 -wd279 -wd188")
elseif (MSVC) # using STREQUAL here conflicts with string interpretation changes in CMake
    message(STATUS "Setting MSVC C compiler options")
    # -DFLATCC_PORTABLE also required, but set earlier
    # -W3 is the highest warning level that is reasonable.
    # See include/flatcc/portable/pwarnings.h for disabled warnings.
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -W3 -D_CRT_SECURE_NO_WARNINGS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W3 -D_CRT_SECURE_NO_WARNINGS")
    # MSVC 2013 (1800) supports inline variable declations
    # while MSVC 2010 (1600) does not.
    if (MSVC_VERSION STRLESS "1800")
        # Disables monster sample build which uses C99 style variable decls.
        set (FLATCC_NEED_C89_VAR_DECLS true)
    endif()
    set(FLATCC_PORTABLE true)
 elseif (CMAKE_C_COMPILER_ID STREQUAL "XL")
    # IBM's native XLC C compiler in extended C99 mode

    message(STATUS "Setting IBM XL C compiler options")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qlanglvl=extc99")
else()
    # Best effort
    message(STATUS "Best effort settings for compiler: ${CMAKE_C_COMPILER_ID}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
    set(FLATCC_PORTABLE true)
endif()

if (FLATCC_PORTABLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DFLATCC_PORTABLE")
endif()

if (CLANG_VERSION)
    message(STATUS "CLANG_VERSION: ${CLANG_VERSION}")
endif()
if (GCC_VERSION)
    message(STATUS "GCC_VERSION: ${GCC_VERSION}")
endif()
message(STATUS "Configured C_FLAGS: ${CMAKE_C_FLAGS}")

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/${lib_dir})

set(CMAKE_DEBUG_POSTFIX "_d")

if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_EXECUTABLE_SUFFIX "_d${CMAKE_EXECUTABLE_SUFFIX}")
endif()


if (FLATCC_RTONLY)
    # The targets we copy to bin and lib directories, i.e. not tests.
    set(dist_targets
        flatccrt
    )
    add_subdirectory(src/runtime)
else()
    # The targets we copy to bin and lib directories, i.e. not tests.
    set(dist_targets
        flatcc
        flatccrt
        flatcc_cli
    )
    add_subdirectory(src/runtime)
    add_subdirectory(src/compiler)
    add_subdirectory(src/cli)
endif()

# disabled by FLATCC_RTONLY
if (FLATCC_TEST)
    add_subdirectory(test)
    add_subdirectory(samples)
endif()

if (FLATCC_COVERAGE)
    add_custom_target(coverage
        COMMAND lcov --capture --directory src --output-file coverage.info
        COMMAND genhtml coverage.info --output-directory coverage)
endif()

set_target_properties(${dist_targets}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${dist_dir}/${lib_dir}"
    LIBRARY_OUTPUT_DIRECTORY "${dist_dir}/${lib_dir}"
    RUNTIME_OUTPUT_DIRECTORY "${dist_dir}/bin"
)

if (FLATCC_INSTALL)
    install(DIRECTORY include/flatcc DESTINATION include)
endif()

