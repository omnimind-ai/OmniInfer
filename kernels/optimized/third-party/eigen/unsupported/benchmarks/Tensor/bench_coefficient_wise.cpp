// Benchmarks for Eigen Tensor coefficient-wise operations.
// Covers activation functions, normalization, and element-wise arithmetic.

#include <benchmark/benchmark.h>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

typedef float Scalar;

// Macro to define a benchmark for a unary tensor operation.
#define BENCH_TENSOR_UNARY(NAME, EXPR)                                        \
  static void BM_##NAME(benchmark::State& state) {                            \
    const int M = state.range(0);                                             \
    const int N = state.range(1);                                             \
    Tensor<Scalar, 2> a(M, N);                                                \
    a.setRandom();                                                            \
    Tensor<Scalar, 2> b(M, N);                                                \
    for (auto _ : state) {                                                    \
      b = EXPR;                                                               \
      benchmark::DoNotOptimize(b.data());                                     \
      benchmark::ClobberMemory();                                             \
    }                                                                         \
    state.SetBytesProcessed(state.iterations() * M * N * sizeof(Scalar) * 2); \
  }

BENCH_TENSOR_UNARY(Exp, a.exp())
BENCH_TENSOR_UNARY(Log, a.abs().log())
BENCH_TENSOR_UNARY(Tanh, a.tanh())
BENCH_TENSOR_UNARY(Sigmoid, a.sigmoid())
BENCH_TENSOR_UNARY(ReLU, a.cwiseMax(Scalar(0)))
BENCH_TENSOR_UNARY(Sqrt, a.abs().sqrt())

// --- Element-wise binary operations ---
static void BM_Add(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);

  Tensor<Scalar, 2> a(M, N);
  Tensor<Scalar, 2> b(M, N);
  Tensor<Scalar, 2> c(M, N);
  a.setRandom();
  b.setRandom();

  for (auto _ : state) {
    c = a + b;
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * M * N * sizeof(Scalar) * 3);
}

static void BM_Mul(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);

  Tensor<Scalar, 2> a(M, N);
  Tensor<Scalar, 2> b(M, N);
  Tensor<Scalar, 2> c(M, N);
  a.setRandom();
  b.setRandom();

  for (auto _ : state) {
    c = a * b;
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * M * N * sizeof(Scalar) * 3);
}

// --- Fused multiply-add ---
static void BM_FMA(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);

  Tensor<Scalar, 2> a(M, N);
  Tensor<Scalar, 2> b(M, N);
  Tensor<Scalar, 2> c(M, N);
  Tensor<Scalar, 2> d(M, N);
  a.setRandom();
  b.setRandom();
  c.setRandom();

  for (auto _ : state) {
    d = a * b + c;
    benchmark::DoNotOptimize(d.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * M * N * sizeof(Scalar) * 4);
}

// --- Rank-4 coefficient-wise (CNN feature maps) ---
static void BM_ReLU_Rank4(benchmark::State& state) {
  const int batch = state.range(0);
  const int C = state.range(1);
  const int H = state.range(2);

  Tensor<Scalar, 4> a(batch, C, H, H);
  Tensor<Scalar, 4> b(batch, C, H, H);
  a.setRandom();

  for (auto _ : state) {
    b = a.cwiseMax(Scalar(0));
    benchmark::DoNotOptimize(b.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * batch * C * H * H * sizeof(Scalar) * 2);
}

static void CwiseSizes(::benchmark::Benchmark* b) {
  for (int size : {256, 1024}) {
    b->Args({size, size});
  }
}

static void Rank4Sizes(::benchmark::Benchmark* b) {
  b->Args({32, 64, 16});
  b->Args({8, 128, 32});
  b->Args({1, 256, 64});
}

BENCHMARK(BM_Exp)->Apply(CwiseSizes);
BENCHMARK(BM_Log)->Apply(CwiseSizes);
BENCHMARK(BM_Tanh)->Apply(CwiseSizes);
BENCHMARK(BM_Sigmoid)->Apply(CwiseSizes);
BENCHMARK(BM_ReLU)->Apply(CwiseSizes);
BENCHMARK(BM_Sqrt)->Apply(CwiseSizes);
BENCHMARK(BM_Add)->Apply(CwiseSizes);
BENCHMARK(BM_Mul)->Apply(CwiseSizes);
BENCHMARK(BM_FMA)->Apply(CwiseSizes);
BENCHMARK(BM_ReLU_Rank4)->Apply(Rank4Sizes);
