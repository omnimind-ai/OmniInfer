CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

# ---[ Project
PROJECT(FXdiv C CXX)

# ---[ Options.
OPTION(FXDIV_USE_INLINE_ASSEMBLY "Allow use of inline assembly in FXdiv" OFF)
IF("${CMAKE_SOURCE_DIR}" STREQUAL "${PROJECT_SOURCE_DIR}")
  OPTION(FXDIV_BUILD_TESTS "Build FXdiv unit tests" ON)
  OPTION(FXDIV_BUILD_BENCHMARKS "Build FXdiv micro-benchmarks" ON)
ELSE()
  SET(FXDIV_BUILD_TESTS OFF CACHE BOOL "Build FXdiv unit tests")
  SET(FXDIV_BUILD_BENCHMARKS OFF CACHE BOOL "Build FXdiv micro-benchmarks")
ENDIF()

# ---[ CMake options
INCLUDE(GNUInstallDirs)

IF(FXDIV_BUILD_TESTS)
  ENABLE_TESTING()
ENDIF()

# ---[ Download deps
IF(FXDIV_BUILD_TESTS AND NOT DEFINED GOOGLETEST_SOURCE_DIR)
  MESSAGE(STATUS "Downloading Google Test to ${CMAKE_BINARY_DIR}/googletest-source (define GOOGLETEST_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadGoogleTest.cmake "${CMAKE_BINARY_DIR}/googletest-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download")
  SET(GOOGLETEST_SOURCE_DIR "${CMAKE_BINARY_DIR}/googletest-source" CACHE STRING "Google Test source directory")
ENDIF()

IF(FXDIV_BUILD_BENCHMARKS AND NOT DEFINED GOOGLEBENCHMARK_SOURCE_DIR)
  MESSAGE(STATUS "Downloading Google Benchmark to ${CMAKE_BINARY_DIR}/googlebenchmark-source (define GOOGLEBENCHMARK_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadGoogleBenchmark.cmake "${CMAKE_BINARY_DIR}/googlebenchmark-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googlebenchmark-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googlebenchmark-download")
  SET(GOOGLEBENCHMARK_SOURCE_DIR "${CMAKE_BINARY_DIR}/googlebenchmark-source" CACHE STRING "Google Benchmark source directory")
ENDIF()

# ---[ FXdiv library
ADD_LIBRARY(fxdiv INTERFACE)
TARGET_INCLUDE_DIRECTORIES(fxdiv INTERFACE include)
IF(FXDIV_USE_INLINE_ASSEMBLY)
  TARGET_COMPILE_DEFINITIONS(fxdiv INTERFACE FXDIV_USE_INLINE_ASSEMBLY=1)
ELSE()
  TARGET_COMPILE_DEFINITIONS(fxdiv INTERFACE FXDIV_USE_INLINE_ASSEMBLY=0)
ENDIF()

INSTALL(FILES include/fxdiv.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

IF(FXDIV_BUILD_TESTS)
  # ---[ Build google test
  IF(NOT TARGET gtest)
    SET(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    ADD_SUBDIRECTORY(
      "${GOOGLETEST_SOURCE_DIR}"
      "${CMAKE_BINARY_DIR}/googletest")
  ENDIF()

  ADD_EXECUTABLE(multiply-high-test test/multiply-high.cc)
  SET_TARGET_PROPERTIES(multiply-high-test PROPERTIES
    CXX_STANDARD 11
    CXX_EXTENSIONS YES)
  TARGET_LINK_LIBRARIES(multiply-high-test fxdiv gtest gtest_main)
  ADD_TEST(multiply-high multiply-high-test)

  ADD_EXECUTABLE(quotient-test test/quotient.cc)
  SET_TARGET_PROPERTIES(quotient-test PROPERTIES
    CXX_STANDARD 11
    CXX_EXTENSIONS YES)
  TARGET_LINK_LIBRARIES(quotient-test fxdiv gtest gtest_main)
  ADD_TEST(quotient quotient-test)
ENDIF()

IF(FXDIV_BUILD_BENCHMARKS)
  # ---[ Build google benchmark
  IF(NOT TARGET benchmark)
    SET(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    ADD_SUBDIRECTORY(
      "${GOOGLEBENCHMARK_SOURCE_DIR}"
      "${CMAKE_BINARY_DIR}/googlebenchmark")
  ENDIF()

  ADD_EXECUTABLE(init-bench bench/init.cc)
  SET_TARGET_PROPERTIES(init-bench PROPERTIES
    CXX_STANDARD 11
    CXX_EXTENSIONS YES)
  TARGET_LINK_LIBRARIES(init-bench fxdiv benchmark)

  ADD_EXECUTABLE(multiply-bench bench/multiply.cc)
  SET_TARGET_PROPERTIES(multiply-bench PROPERTIES
    CXX_STANDARD 11
    CXX_EXTENSIONS YES)
  TARGET_LINK_LIBRARIES(multiply-bench fxdiv benchmark)

  ADD_EXECUTABLE(divide-bench bench/divide.cc)
  SET_TARGET_PROPERTIES(divide-bench PROPERTIES
    CXX_STANDARD 11
    CXX_EXTENSIONS YES)
  TARGET_LINK_LIBRARIES(divide-bench fxdiv benchmark)

  ADD_EXECUTABLE(quotient-bench bench/quotient.cc)
  SET_TARGET_PROPERTIES(quotient-bench PROPERTIES
    CXX_STANDARD 11
    CXX_EXTENSIONS YES)
  TARGET_LINK_LIBRARIES(quotient-bench fxdiv benchmark)

  ADD_EXECUTABLE(round-down-bench bench/round-down.cc)
  SET_TARGET_PROPERTIES(round-down-bench PROPERTIES
    CXX_STANDARD 11
    CXX_EXTENSIONS YES)
  TARGET_LINK_LIBRARIES(round-down-bench fxdiv benchmark)
ENDIF()
