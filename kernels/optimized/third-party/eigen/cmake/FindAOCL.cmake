
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# FindAOCL.cmake - CMake Module for AMD Optimizing CPU Libraries (AOCL)
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Description:
# ------------
# This CMake module locates and configures AMD Optimizing CPU Libraries (AOCL)
# for high-performance mathematical computing on AMD processors. It searches for
# and sets up the following AOCL components:
#
# 1. AOCL MathLib (libamdlibm): Vector Math Library providing optimized
#    transcendental functions (exp, sin, cos, sqrt, log, etc.) with VRDA
#    (Vector Rapid Double-precision Arithmetic) support for SIMD acceleration
#
# 2. AOCL BLAS (BLIS): Basic Linear Algebra Subprograms optimized for AMD
#    architectures, supporting both single-threaded (libblis) and multithreaded
#    (libblis-mt) execution with OpenMP parallelization
#
# 3. AOCL LAPACK (libflame): Linear Algebra PACKage providing dense matrix
#    factorizations, eigenvalue/eigenvector computations, and linear system
#    solvers optimized for AMD processors
#
# The module automatically detects the appropriate library variants based on
# configuration flags and provides proper linking setup for optimal performance
# on Zen, Zen2, Zen3, Zen4, and Zen5 architectures.
#
# Variables Set:
# --------------
# AOCL_FOUND          - True if AOCL libraries are found
# AOCL_LIBRARIES      - List of AOCL libraries to link against
# AOCL_INCLUDE_DIRS   - Include directories for AOCL headers
# AOCL_BLAS_TYPE      - Type of BLIS library found ("multithreaded" or "single-threaded")
# AOCL_CORE_LIB       - Path to core AOCL math library
# AOCL_BLAS_LIB       - Path to AOCL BLAS library
# AOCL_LAPACK_LIB     - Path to AOCL LAPACK library
#
# Configuration Options:
# ----------------------
# EIGEN_AOCL_BENCH_USE_MT - When ON, searches for multithreaded BLIS first
#                          When OFF, searches for single-threaded BLIS only
#
# # For multithreaded BLIS:
# cmake .. -DEIGEN_AOCL_BENCH_USE_MT=ON
#
# # For single-threaded BLIS:
# cmake .. -DEIGEN_AOCL_BENCH_USE_MT=OFF
#
# Library Search Paths:
# ---------------------
# The module searches for AOCL libraries in the following order:
# 1. ${AOCL_ROOT}/lib (or ${AOCL_ROOT}/lib32 for 32-bit)
# 2. /opt/amd/aocl/lib64 (or /opt/amd/aocl/lib32 for 32-bit)
# 3. ${LIB_INSTALL_DIR}
#
# Expected Library Names:
# -----------------------
# Core MathLib: amdlibm, alm, almfast
# BLAS Single:  blis
# BLAS Multi:   blis-mt
# LAPACK:       flame
#
# Dependencies:
# -------------
# The module automatically links the following system libraries:
# - libm (standard math library)
# - libpthread (POSIX threads)
# - librt (real-time extensions)
#
# Architecture Support:
# ---------------------
# Optimized for AMD Zen family processors (Zen, Zen2, Zen3, Zen4, Zen5)
# with automatic architecture detection and SIMD instruction selection.
#
# Developer:
# ----------
# Name: Sharad Saurabh Bhaskar
# Email: shbhaska@amd.com
#

if(NOT DEFINED AOCL_ROOT)
  if(DEFINED ENV{AOCL_ROOT})
    set(AOCL_ROOT $ENV{AOCL_ROOT})
    if (NOT AOCL_FIND_QUIETLY)
      message(STATUS "AOCL_ROOT set from environment: ${AOCL_ROOT}")
    endif()
  else()
    if (NOT AOCL_FIND_QUIETLY)
      message(WARNING "AOCL_ROOT is not set. AOCL support will be disabled.")
    endif()
    set(AOCL_LIBRARIES "")
  endif()
endif()

if(AOCL_LIBRARIES)
  set(AOCL_FIND_QUIETLY TRUE)
endif()

# Determine default include directories
set(AOCL_INCLUDE_DIRS "")
if(AOCL_ROOT AND EXISTS "${AOCL_ROOT}/include")
  list(APPEND AOCL_INCLUDE_DIRS "${AOCL_ROOT}/include")
endif()
if(EXISTS "/opt/amd/aocl/include")
  list(APPEND AOCL_INCLUDE_DIRS "/opt/amd/aocl/include")
endif()

  if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    # Search for the core AOCL math library.
    find_library(AOCL_CORE_LIB
      NAMES amdlibm alm almfast
      PATHS
        ${AOCL_ROOT}/lib
        /opt/amd/aocl/lib64
        ${LIB_INSTALL_DIR}
    )
    if (NOT AOCL_FIND_QUIETLY)
      if(AOCL_CORE_LIB)
        message(STATUS "Found AOCL core library: ${AOCL_CORE_LIB}")
      else()
        message(WARNING "AOCL core library not found in ${AOCL_ROOT}/lib or default locations.")
      endif()
    endif()

    # Conditional BLIS library search based on MT requirement
    if(EIGEN_AOCL_BENCH_USE_MT)
      # Search for multithreaded BLIS first
      find_library(AOCL_BLAS_LIB
        NAMES blis-mt
        PATHS
          ${AOCL_ROOT}/lib
          /opt/amd/aocl/lib64
          ${LIB_INSTALL_DIR}
      )
      if(AOCL_BLAS_LIB)
        if (NOT AOCL_FIND_QUIETLY)
          message(STATUS "Found AOCL BLAS (MT) library: ${AOCL_BLAS_LIB}")
        endif()
        set(AOCL_BLAS_TYPE "multithreaded")
      else()
        if (NOT AOCL_FIND_QUIETLY)
          message(WARNING "AOCL multithreaded BLAS library not found, falling back to single-threaded.")
        endif()
        find_library(AOCL_BLAS_LIB
          NAMES blis
          PATHS
            ${AOCL_ROOT}/lib
            /opt/amd/aocl/lib64
            ${LIB_INSTALL_DIR}
        )
        set(AOCL_BLAS_TYPE "single-threaded")
      endif()
    else()
      # Search for single-threaded BLIS
      find_library(AOCL_BLAS_LIB
        NAMES blis
        PATHS
          ${AOCL_ROOT}/lib
          /opt/amd/aocl/lib64
          ${LIB_INSTALL_DIR}
      )
      if(AOCL_BLAS_LIB)
        if (NOT AOCL_FIND_QUIETLY)
          message(STATUS "Found AOCL BLAS (ST) library: ${AOCL_BLAS_LIB}")
        endif()
        set(AOCL_BLAS_TYPE "single-threaded")
      else()
        if (NOT AOCL_FIND_QUIETLY)
          message(WARNING "AOCL single-threaded BLAS library not found.")
        endif()
      endif()
    endif()

    # Now search for AOCL LAPACK library.
    find_library(AOCL_LAPACK_LIB
      NAMES flame
      PATHS
        ${AOCL_ROOT}/lib
        /opt/amd/aocl/lib64
        ${LIB_INSTALL_DIR}
    )
    if (NOT AOCL_FIND_QUIETLY)
      if(AOCL_LAPACK_LIB)
        message(STATUS "Found AOCL LAPACK library: ${AOCL_LAPACK_LIB}")
      else()
        message(WARNING "AOCL LAPACK library not found in ${AOCL_ROOT}/lib or default locations.")
      endif()
    endif()

  else()
    # For 32-bit systems, similar search paths.
    find_library(AOCL_CORE_LIB
      NAMES amdlibm alm almfast
      PATHS
        ${AOCL_ROOT}/lib
        /opt/amd/aocl/lib32
        ${LIB_INSTALL_DIR}
    )
    if (NOT AOCL_FIND_QUIETLY)
      if(AOCL_CORE_LIB)
        message(STATUS "Found AOCL core library: ${AOCL_CORE_LIB}")
      else()
        message(WARNING "AOCL core library not found in ${AOCL_ROOT}/lib or default locations.")
      endif()
    endif()

    # Conditional BLIS library search for 32-bit
    if(EIGEN_AOCL_BENCH_USE_MT)
      find_library(AOCL_BLAS_LIB
        NAMES blis-mt
        PATHS
          ${AOCL_ROOT}/lib
          /opt/amd/aocl/lib32
          ${LIB_INSTALL_DIR}
      )
      if(AOCL_BLAS_LIB)
        if (NOT AOCL_FIND_QUIETLY)
          message(STATUS "Found AOCL BLAS (MT) library: ${AOCL_BLAS_LIB}")
        endif()
        set(AOCL_BLAS_TYPE "multithreaded")
      else()
        if (NOT AOCL_FIND_QUIETLY)
          message(WARNING "AOCL multithreaded BLAS library not found, falling back to single-threaded.")
        endif()
        find_library(AOCL_BLAS_LIB
          NAMES blis
          PATHS
            ${AOCL_ROOT}/lib
            /opt/amd/aocl/lib32
            ${LIB_INSTALL_DIR}
        )
        set(AOCL_BLAS_TYPE "single-threaded")
      endif()
    else()
      find_library(AOCL_BLAS_LIB
        NAMES blis
        PATHS
          ${AOCL_ROOT}/lib
          /opt/amd/aocl/lib32
          ${LIB_INSTALL_DIR}
      )
      if(AOCL_BLAS_LIB)
        if (NOT AOCL_FIND_QUIETLY)
          message(STATUS "Found AOCL BLAS (ST) library: ${AOCL_BLAS_LIB}")
        endif()
        set(AOCL_BLAS_TYPE "single-threaded")
      else()
        if (NOT AOCL_FIND_QUIETLY)
          message(WARNING "AOCL single-threaded BLAS library not found.")
        endif()
      endif()
    endif()

    find_library(AOCL_LAPACK_LIB
      NAMES flame
      PATHS
        ${AOCL_ROOT}/lib
        /opt/amd/aocl/lib32
        ${LIB_INSTALL_DIR}
    )
    if (NOT AOCL_FIND_QUIETLY)
      if(AOCL_LAPACK_LIB)
        message(STATUS "Found AOCL LAPACK library: ${AOCL_LAPACK_LIB}")
      else()
        message(WARNING "AOCL LAPACK library not found in ${AOCL_ROOT}/lib or default locations.")
      endif()
    endif()
endif()

# Combine the found libraries into one variable.
if(AOCL_CORE_LIB)
  set(AOCL_LIBRARIES ${AOCL_CORE_LIB})
endif()
if(AOCL_BLAS_LIB)
  list(APPEND AOCL_LIBRARIES ${AOCL_BLAS_LIB})
endif()
if(AOCL_LAPACK_LIB)
  list(APPEND AOCL_LIBRARIES ${AOCL_LAPACK_LIB})
endif()
if(AOCL_LIBRARIES)
  # Link against the standard math and pthread libraries as well as librt
  list(APPEND AOCL_LIBRARIES m pthread rt)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AOCL DEFAULT_MSG AOCL_LIBRARIES AOCL_INCLUDE_DIRS)
mark_as_advanced(AOCL_LIBRARIES AOCL_INCLUDE_DIRS)
