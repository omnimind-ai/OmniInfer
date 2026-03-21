// Benchmarks for Eigen Tensor shuffling (transpose / permutation).

#include <benchmark/benchmark.h>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

typedef float Scalar;

// --- Rank-2 transpose ---
static void BM_Shuffle2D(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);

  Tensor<Scalar, 2> A(M, N);
  Tensor<Scalar, 2> B(N, M);
  A.setRandom();

  Eigen::array<int, 2> perm = {1, 0};

  for (auto _ : state) {
    B = A.shuffle(perm);
    benchmark::DoNotOptimize(B.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * M * N * sizeof(Scalar) * 2);
}

// --- Identity shuffle (no permutation, measures overhead) ---
static void BM_ShuffleIdentity(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);

  Tensor<Scalar, 2> A(M, N);
  Tensor<Scalar, 2> B(M, N);
  A.setRandom();

  Eigen::array<int, 2> perm = {0, 1};

  for (auto _ : state) {
    B = A.shuffle(perm);
    benchmark::DoNotOptimize(B.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * M * N * sizeof(Scalar) * 2);
}

// --- Rank-3 permutation ---
static void BM_Shuffle3D(benchmark::State& state) {
  const int D0 = state.range(0);
  const int D1 = state.range(1);
  const int D2 = state.range(2);

  Tensor<Scalar, 3> A(D0, D1, D2);
  A.setRandom();

  // Permutation (2, 0, 1)
  Eigen::array<int, 3> perm = {2, 0, 1};

  for (auto _ : state) {
    Tensor<Scalar, 3> B = A.shuffle(perm);
    benchmark::DoNotOptimize(B.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * D0 * D1 * D2 * sizeof(Scalar) * 2);
}

// --- Rank-4 permutation (NCHW -> NHWC layout conversion) ---
static void BM_Shuffle4D_NCHW_to_NHWC(benchmark::State& state) {
  const int N = state.range(0);
  const int C = state.range(1);
  const int H = state.range(2);

  Tensor<Scalar, 4> A(N, C, H, H);
  A.setRandom();

  // NCHW -> NHWC: permute (0, 2, 3, 1)
  Eigen::array<int, 4> perm = {0, 2, 3, 1};

  for (auto _ : state) {
    Tensor<Scalar, 4> B = A.shuffle(perm);
    benchmark::DoNotOptimize(B.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * N * C * H * H * sizeof(Scalar) * 2);
}

static void Shuffle2DSizes(::benchmark::Benchmark* b) {
  for (int size : {256, 1024}) {
    b->Args({size, size});
  }
  b->Args({64, 4096});
  b->Args({4096, 64});
}

static void Shuffle3DSizes(::benchmark::Benchmark* b) {
  b->Args({64, 64, 64});
  b->Args({128, 128, 64});
  b->Args({32, 256, 256});
}

static void Shuffle4DSizes(::benchmark::Benchmark* b) {
  for (int batch : {1, 8}) {
    for (int c : {3, 64}) {
      for (int h : {32, 64}) {
        b->Args({batch, c, h});
      }
    }
  }
}

BENCHMARK(BM_Shuffle2D)->Apply(Shuffle2DSizes);
BENCHMARK(BM_ShuffleIdentity)->Apply(Shuffle2DSizes);
BENCHMARK(BM_Shuffle3D)->Apply(Shuffle3DSizes);
BENCHMARK(BM_Shuffle4D_NCHW_to_NHWC)->Apply(Shuffle4DSizes);
