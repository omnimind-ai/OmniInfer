# FXdiv

[![MIT License](https://img.shields.io/badge/License-MIT%20License-blue.svg)](https://github.com/Maratyszcza/FXdiv/blob/master/LICENSE)
[![Build Status](https://img.shields.io/travis/Maratyszcza/FXdiv.svg)](https://travis-ci.org/Maratyszcza/FXdiv)


Header-only library for division via fixed-point multiplication by inverse

On modern CPUs and GPUs integer division is several times slower than multiplication. FXdiv implements an algorithm to replace an integer division with a multiplication and two shifts. This algorithm improves performance when an application performs repeated divisions by the same divisor.

## Features

- Integer division for `uint32_t`, `uint64_t`, and `size_t`
- Header-only library, no installation or build required
- Compatible with C99, C++, OpenCL, and CUDA
- Uses platform-specific compiler intrinsics for optimal performance
- Covered with unit tests and microbenchmarks

## Example

```c
#include <fxdiv.h>

/* Division of array by a constant: reference implementation */
void divide_array_c(size_t length, uint32_t array[], uint32_t divisor) {
  for (size_t i = 0; i < length; i++) {
    array[i] /= divisor;
  }
}

/* Division of array by a constant: implementation with FXdiv */
void divide_array_fxdiv(size_t length, uint32_t array[], uint32_t divisor) {
  const struct fxdiv_divisor_uint32_t precomputed_divisor =
    fxdiv_init_uint32_t(divisor);
  for (size_t i = 0; i < length; i++) {
    array[i] = fxdiv_quotient_uint32_t(array[i], precomputed_divisor);
  }
}
```

## Status

Currently working features:

| Platform        | uint32_t | uint64_t | size_t   |
| --------------- |:--------:|:--------:|:--------:|
| x86-64 gcc      | Works    | Works    | Works    |
| x86-64 clang    | Works    | Works    | Works    |
| x86-64 MSVC     | Works    | Works    | Works    |
| x86 gcc         | Works    | Works    | Works    |
| x86 clang       | Works    | Works    | Works    |
| x86 MSVC        | Works    | Works    | Works    |
| ARMv7 gcc       | Works    | Works    | Works    |
| ARMv7 clang     | Works    | Works    | Works    |
| ARMv7 MSVC*     | Compiles | Compiles | Compiles |
| ARM64 gcc       | Works    | Works    | Works    |
| ARM64 clang     | Works    | Works    | Works    |
| ARM64 MSVC*     | Compiles | Compiles | Compiles |
| PPC64 gcc       | Works    | Works    | Works    |
| WAsm clang      | Works    | Works    | Works    |
| Asm.js clang    | Works    | Works    | Works    |
| PNaCl clang     | Works    | Works    | Works    |
| CUDA            | Untested | Untested | Untested |
| OpenCL          | Untested | Untested | Untested |

*ARMv7 and ARM64 builds with MSVC are presumed to work, but were only verified to compile successfully

## References

- Granlund, Torbjörn, and Peter L. Montgomery. "Division by invariant integers using multiplication." In ACM SIGPLAN Notices, vol. 29, no. 6, pp. 61-72. ACM, 1994. Available: [gmplib.org/~tege/divcnst-pldi94.pdf](https://gmplib.org/~tege/divcnst-pldi94.pdf)
