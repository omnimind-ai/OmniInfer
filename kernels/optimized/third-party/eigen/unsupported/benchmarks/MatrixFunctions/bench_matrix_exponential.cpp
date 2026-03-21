// Benchmarks for matrix exponential.
// Critical for Sophus Lie group operations (SLAM, visual odometry).

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;

#ifndef SCALAR
#define SCALAR double
#endif

typedef SCALAR Scalar;

static void BM_MatrixExp(benchmark::State& state) {
  int n = state.range(0);
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;

  // Generate a random matrix with reasonable spectral radius.
  MatrixType A = MatrixType::Random(n, n) / Scalar(n);
  MatrixType result(n, n);

  for (auto _ : state) {
    result = A.exp();
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
}

// Fixed-size specializations for Lie group sizes.
template <int N>
static void BM_MatrixExp_Fixed(benchmark::State& state) {
  typedef Matrix<Scalar, N, N> MatrixType;

  MatrixType A = MatrixType::Random() / Scalar(N);
  MatrixType result;

  for (auto _ : state) {
    result = A.exp();
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
}

// Dynamic sizes: Lie groups (2,3,4) plus larger.
BENCHMARK(BM_MatrixExp)->Arg(2)->Arg(3)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(64)->Arg(128);

// Fixed-size Lie group dimensions.
BENCHMARK(BM_MatrixExp_Fixed<2>);
BENCHMARK(BM_MatrixExp_Fixed<3>);
BENCHMARK(BM_MatrixExp_Fixed<4>);
