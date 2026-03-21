// Benchmarks for matrix logarithm.
// Inverse of matrix exponential, used for Lie group log maps.

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;

#ifndef SCALAR
#define SCALAR double
#endif

typedef SCALAR Scalar;

static void BM_MatrixLog(benchmark::State& state) {
  int n = state.range(0);
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;

  // Generate a matrix close to identity for stable log computation.
  MatrixType A = MatrixType::Identity(n, n) + MatrixType::Random(n, n) / Scalar(n * 2);
  // Ensure A is in the principal branch by computing exp(small matrix).
  A = (MatrixType::Random(n, n) / Scalar(n * 4)).exp();
  MatrixType result(n, n);

  for (auto _ : state) {
    result = A.log();
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
}

template <int N>
static void BM_MatrixLog_Fixed(benchmark::State& state) {
  typedef Matrix<Scalar, N, N> MatrixType;

  MatrixType A = (MatrixType::Random() / Scalar(N * 4)).exp();
  MatrixType result;

  for (auto _ : state) {
    result = A.log();
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_MatrixLog)->Arg(2)->Arg(3)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(64);

BENCHMARK(BM_MatrixLog_Fixed<2>);
BENCHMARK(BM_MatrixLog_Fixed<3>);
BENCHMARK(BM_MatrixLog_Fixed<4>);
