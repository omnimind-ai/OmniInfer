// Benchmarks for Eigen Tensor morphing operations: reshape, slice, chip, pad, stride.

#include <benchmark/benchmark.h>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

typedef float Scalar;

// --- Reshape (zero-cost if no evaluation needed; force eval via assignment) ---
static void BM_Reshape(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);

  Tensor<Scalar, 2> A(M, N);
  A.setRandom();

  Eigen::array<Index, 1> new_shape = {M * N};

  for (auto _ : state) {
    Tensor<Scalar, 1> B = A.reshape(new_shape);
    benchmark::DoNotOptimize(B.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * M * N * sizeof(Scalar));
}

// --- Slice ---
static void BM_Slice(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);

  Tensor<Scalar, 2> A(M, N);
  A.setRandom();

  int sliceM = M / 2;
  int sliceN = N / 2;
  Eigen::array<Index, 2> offsets = {0, 0};
  Eigen::array<Index, 2> extents = {sliceM, sliceN};

  for (auto _ : state) {
    Tensor<Scalar, 2> B = A.slice(offsets, extents);
    benchmark::DoNotOptimize(B.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * sliceM * sliceN * sizeof(Scalar));
}

// --- Chip (extract a sub-tensor along one dimension) ---
static void BM_Chip(benchmark::State& state) {
  const int D0 = state.range(0);
  const int D1 = state.range(1);
  const int D2 = state.range(2);

  Tensor<Scalar, 3> A(D0, D1, D2);
  A.setRandom();

  for (auto _ : state) {
    Tensor<Scalar, 2> B = A.chip(0, 0);
    benchmark::DoNotOptimize(B.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * D1 * D2 * sizeof(Scalar));
}

// --- Pad ---
static void BM_Pad(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);
  const int padSize = state.range(2);

  Tensor<Scalar, 2> A(M, N);
  A.setRandom();

  Eigen::array<std::pair<int, int>, 2> paddings;
  paddings[0] = {padSize, padSize};
  paddings[1] = {padSize, padSize};

  for (auto _ : state) {
    Tensor<Scalar, 2> B = A.pad(paddings);
    benchmark::DoNotOptimize(B.data());
    benchmark::ClobberMemory();
  }
  int outM = M + 2 * padSize;
  int outN = N + 2 * padSize;
  state.SetBytesProcessed(state.iterations() * outM * outN * sizeof(Scalar));
}

// --- Stride ---
static void BM_Stride(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);
  const int stride = state.range(2);

  Tensor<Scalar, 2> A(M, N);
  A.setRandom();

  Eigen::array<Index, 2> strides_arr = {stride, stride};

  for (auto _ : state) {
    Tensor<Scalar, 2> B = A.stride(strides_arr);
    benchmark::DoNotOptimize(B.data());
    benchmark::ClobberMemory();
  }
  int outM = (M + stride - 1) / stride;
  int outN = (N + stride - 1) / stride;
  state.SetBytesProcessed(state.iterations() * outM * outN * sizeof(Scalar));
}

static void MorphSizes(::benchmark::Benchmark* b) {
  for (int size : {256, 1024}) {
    b->Args({size, size});
  }
}

static void ChipSizes(::benchmark::Benchmark* b) {
  b->Args({32, 256, 256});
  b->Args({64, 128, 128});
  b->Args({8, 512, 512});
}

static void PadSizes(::benchmark::Benchmark* b) {
  for (int size : {256, 1024}) {
    for (int pad : {1, 4, 16}) {
      b->Args({size, size, pad});
    }
  }
}

static void StrideSizes(::benchmark::Benchmark* b) {
  for (int size : {256, 1024}) {
    for (int stride : {2, 4}) {
      b->Args({size, size, stride});
    }
  }
}

BENCHMARK(BM_Reshape)->Apply(MorphSizes);
BENCHMARK(BM_Slice)->Apply(MorphSizes);
BENCHMARK(BM_Chip)->Apply(ChipSizes);
BENCHMARK(BM_Pad)->Apply(PadSizes);
BENCHMARK(BM_Stride)->Apply(StrideSizes);
