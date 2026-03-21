// Benchmarks for matrix power functions: sqrt, pow, cos, sin, cosh, sinh.

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;

typedef double Scalar;
typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

static void BM_MatrixSqrt(benchmark::State& state) {
  int n = state.range(0);
  // SPD matrix has well-defined sqrt.
  Mat tmp = Mat::Random(n, n);
  Mat A = tmp * tmp.transpose() + Mat::Identity(n, n);
  Mat result(n, n);

  for (auto _ : state) {
    result = A.sqrt();
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
}

static void BM_MatrixPow(benchmark::State& state) {
  int n = state.range(0);
  Mat tmp = Mat::Random(n, n);
  Mat A = tmp * tmp.transpose() + Mat::Identity(n, n);
  Mat result(n, n);
  Scalar p = 2.5;

  for (auto _ : state) {
    result = A.pow(p);
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
}

static void BM_MatrixCos(benchmark::State& state) {
  int n = state.range(0);
  Mat A = Mat::Random(n, n) / Scalar(n);
  Mat result(n, n);

  for (auto _ : state) {
    result = A.cos();
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
}

static void BM_MatrixSin(benchmark::State& state) {
  int n = state.range(0);
  Mat A = Mat::Random(n, n) / Scalar(n);
  Mat result(n, n);

  for (auto _ : state) {
    result = A.sin();
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
}

static void BM_MatrixCosh(benchmark::State& state) {
  int n = state.range(0);
  Mat A = Mat::Random(n, n) / Scalar(n);
  Mat result(n, n);

  for (auto _ : state) {
    result = A.cosh();
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
}

static void BM_MatrixSinh(benchmark::State& state) {
  int n = state.range(0);
  Mat A = Mat::Random(n, n) / Scalar(n);
  Mat result(n, n);

  for (auto _ : state) {
    result = A.sinh();
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
}

static void MatPowerSizes(::benchmark::Benchmark* b) {
  for (int n : {4, 8, 16, 32, 64}) {
    b->Arg(n);
  }
}

BENCHMARK(BM_MatrixSqrt)->Apply(MatPowerSizes);
BENCHMARK(BM_MatrixPow)->Apply(MatPowerSizes);
BENCHMARK(BM_MatrixCos)->Apply(MatPowerSizes);
BENCHMARK(BM_MatrixSin)->Apply(MatPowerSizes);
BENCHMARK(BM_MatrixCosh)->Apply(MatPowerSizes);
BENCHMARK(BM_MatrixSinh)->Apply(MatPowerSizes);
