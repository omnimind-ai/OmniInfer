// Benchmarks for Eigen Tensor reductions (sum, maximum, mean).
// Tests full and partial reductions, inner vs outer dimension, DefaultDevice and ThreadPoolDevice.

#define EIGEN_USE_THREADS

#include <benchmark/benchmark.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>

using namespace Eigen;

#ifndef SCALAR
#define SCALAR float
#endif

typedef SCALAR Scalar;

// --- Full reduction (rank-2) ---
template <typename ReduceOp>
static void BM_FullReduction(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);

  Tensor<Scalar, 2> A(M, N);
  A.setRandom();

  for (auto _ : state) {
    Tensor<Scalar, 0> result = A.reduce(Eigen::array<int, 2>{0, 1}, ReduceOp());
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * M * N * sizeof(Scalar));
}

// --- Partial reduction along dim 0 (inner dim, ColMajor) ---
static void BM_ReduceInner(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);

  Tensor<Scalar, 2> A(M, N);
  A.setRandom();

  Eigen::array<int, 1> reduce_dims = {0};

  for (auto _ : state) {
    Tensor<Scalar, 1> result = A.sum(reduce_dims);
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * M * N * sizeof(Scalar));
}

// --- Partial reduction along dim 1 (outer dim, ColMajor) ---
static void BM_ReduceOuter(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);

  Tensor<Scalar, 2> A(M, N);
  A.setRandom();

  Eigen::array<int, 1> reduce_dims = {1};

  for (auto _ : state) {
    Tensor<Scalar, 1> result = A.sum(reduce_dims);
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * M * N * sizeof(Scalar));
}

// --- Rank-4 partial reduction (batch x channels x H x W), reduce along spatial dims ---
static void BM_ReduceSpatial(benchmark::State& state) {
  const int batch = state.range(0);
  const int C = state.range(1);
  const int H = state.range(2);

  Tensor<Scalar, 4> A(batch, C, H, H);
  A.setRandom();

  Eigen::array<int, 2> reduce_dims = {2, 3};

  for (auto _ : state) {
    Tensor<Scalar, 2> result = A.sum(reduce_dims);
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * batch * C * H * H * sizeof(Scalar));
}

// --- Full reduction with ThreadPoolDevice ---
static void BM_FullReduction_ThreadPool(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);
  const int threads = state.range(2);

  Tensor<Scalar, 2> A(M, N);
  Tensor<Scalar, 0> result;
  A.setRandom();

  ThreadPool tp(threads);
  ThreadPoolDevice dev(&tp, threads);

  for (auto _ : state) {
    result.device(dev) = A.sum();
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * M * N * sizeof(Scalar));
  state.counters["threads"] = threads;
}

// --- Maximum reduction (rank-2) ---
static void BM_MaxReduction(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);

  Tensor<Scalar, 2> A(M, N);
  A.setRandom();

  for (auto _ : state) {
    Tensor<Scalar, 0> result = A.maximum();
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * M * N * sizeof(Scalar));
}

static void ReductionSizes(::benchmark::Benchmark* b) {
  for (int size : {64, 256, 1024}) {
    b->Args({size, size});
  }
}

static void ThreadPoolReductionSizes(::benchmark::Benchmark* b) {
  for (int size : {256, 1024}) {
    for (int threads : {2, 4, 8}) {
      b->Args({size, size, threads});
    }
  }
}

static void SpatialSizes(::benchmark::Benchmark* b) {
  for (int batch : {1, 8, 32}) {
    for (int c : {64, 128}) {
      for (int h : {16, 32}) {
        b->Args({batch, c, h});
      }
    }
  }
}

BENCHMARK(BM_FullReduction<internal::SumReducer<Scalar>>)->Apply(ReductionSizes)->Name("SumReduction");
BENCHMARK(BM_FullReduction<internal::MaxReducer<Scalar>>)->Apply(ReductionSizes)->Name("MaxReduction_Full");
BENCHMARK(BM_MaxReduction)->Apply(ReductionSizes);
BENCHMARK(BM_ReduceInner)->Apply(ReductionSizes);
BENCHMARK(BM_ReduceOuter)->Apply(ReductionSizes);
BENCHMARK(BM_ReduceSpatial)->Apply(SpatialSizes);
BENCHMARK(BM_FullReduction_ThreadPool)->Apply(ThreadPoolReductionSizes);
