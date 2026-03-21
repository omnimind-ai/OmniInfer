// Benchmarks for Eigen Tensor contraction (generalized GEMM).
// Tests single-threaded (DefaultDevice) and multi-threaded (ThreadPoolDevice) variants.

#define EIGEN_USE_THREADS

#include <benchmark/benchmark.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>

using namespace Eigen;

#ifndef SCALAR
#define SCALAR float
#endif

typedef SCALAR Scalar;

// --- DefaultDevice contraction (rank-2, equivalent to matrix multiply) ---
static void BM_Contraction(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);
  const int K = state.range(2);

  Tensor<Scalar, 2> A(M, K);
  Tensor<Scalar, 2> B(K, N);
  Tensor<Scalar, 2> C(M, N);
  A.setRandom();
  B.setRandom();

  using ContractDims = Tensor<Scalar, 2>::DimensionPair;
  Eigen::array<ContractDims, 1> contract_dims = {ContractDims(1, 0)};

  for (auto _ : state) {
    C = A.contract(B, contract_dims);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] =
      benchmark::Counter(2.0 * M * N * K, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// --- ThreadPoolDevice contraction ---
static void BM_Contraction_ThreadPool(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);
  const int K = state.range(2);
  const int threads = state.range(3);

  Tensor<Scalar, 2> A(M, K);
  Tensor<Scalar, 2> B(K, N);
  Tensor<Scalar, 2> C(M, N);
  A.setRandom();
  B.setRandom();

  ThreadPool tp(threads);
  ThreadPoolDevice dev(&tp, threads);

  using ContractDims = Tensor<Scalar, 2>::DimensionPair;
  Eigen::array<ContractDims, 1> contract_dims = {ContractDims(1, 0)};

  for (auto _ : state) {
    C.device(dev) = A.contract(B, contract_dims);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] =
      benchmark::Counter(2.0 * M * N * K, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
  state.counters["threads"] = threads;
}

// --- Rank-3 batch contraction ---
// Contracts A(batch, M, K) with B(batch, K, N) over batch dim (0<->0)
// and K dim (2<->1), producing C(M, N). This sums over both the batch
// and inner dimensions: C(m, n) = sum_b sum_k A(b, m, k) * B(b, k, n).
static void BM_BatchContraction(benchmark::State& state) {
  const int batch = state.range(0);
  const int M = state.range(1);
  const int N = state.range(2);
  const int K = state.range(3);

  Tensor<Scalar, 3> A(batch, M, K);
  Tensor<Scalar, 3> B(batch, K, N);
  Tensor<Scalar, 2> C(M, N);
  A.setRandom();
  B.setRandom();

  using ContractDims = Tensor<Scalar, 3>::DimensionPair;
  Eigen::array<ContractDims, 2> contract_dims = {ContractDims(0, 0), ContractDims(2, 1)};

  for (auto _ : state) {
    C = A.contract(B, contract_dims);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] = benchmark::Counter(2.0 * batch * M * N * K, benchmark::Counter::kIsIterationInvariantRate,
                                                benchmark::Counter::kIs1000);
}

// --- RowMajor contraction ---
static void BM_Contraction_RowMajor(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);
  const int K = state.range(2);

  Tensor<Scalar, 2, RowMajor> A(M, K);
  Tensor<Scalar, 2, RowMajor> B(K, N);
  Tensor<Scalar, 2, RowMajor> C(M, N);
  A.setRandom();
  B.setRandom();

  using ContractDims = Tensor<Scalar, 2, RowMajor>::DimensionPair;
  Eigen::array<ContractDims, 1> contract_dims = {ContractDims(1, 0)};

  for (auto _ : state) {
    C = A.contract(B, contract_dims);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] =
      benchmark::Counter(2.0 * M * N * K, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

static void ContractionSizes(::benchmark::Benchmark* b) {
  for (int size : {32, 64, 128, 256, 512, 1024}) {
    b->Args({size, size, size});
  }
  // Non-square
  b->Args({256, 256, 1024});
  b->Args({1024, 64, 64});
}

static void ThreadPoolSizes(::benchmark::Benchmark* b) {
  for (int size : {64, 256, 512, 1024}) {
    for (int threads : {2, 4, 8}) {
      b->Args({size, size, size, threads});
    }
  }
}

static void BatchSizes(::benchmark::Benchmark* b) {
  for (int batch : {1, 8, 32}) {
    for (int size : {64, 256}) {
      b->Args({batch, size, size, size});
    }
  }
}

BENCHMARK(BM_Contraction)->Apply(ContractionSizes);
BENCHMARK(BM_Contraction_RowMajor)->Apply(ContractionSizes);
BENCHMARK(BM_Contraction_ThreadPool)->Apply(ThreadPoolSizes);
BENCHMARK(BM_BatchContraction)->Apply(BatchSizes);
