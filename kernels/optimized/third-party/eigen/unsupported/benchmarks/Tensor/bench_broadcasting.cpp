// Benchmarks for Eigen Tensor broadcasting.
// Tests broadcasting along various dimensions and ranks.

#include <benchmark/benchmark.h>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

typedef float Scalar;

// --- Broadcast row vector {1,N} -> {M,N} ---
static void BM_BroadcastRow(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);

  Tensor<Scalar, 2> row(1, N);
  Tensor<Scalar, 2> result(M, N);
  row.setRandom();

  Eigen::array<int, 2> bcast = {M, 1};

  for (auto _ : state) {
    result = row.broadcast(bcast);
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * M * N * sizeof(Scalar));
}

// --- Broadcast col vector {M,1} -> {M,N} ---
static void BM_BroadcastCol(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);

  Tensor<Scalar, 2> col(M, 1);
  Tensor<Scalar, 2> result(M, N);
  col.setRandom();

  Eigen::array<int, 2> bcast = {1, N};

  for (auto _ : state) {
    result = col.broadcast(bcast);
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * M * N * sizeof(Scalar));
}

// --- Broadcast + element-wise add (bias addition pattern) ---
static void BM_BroadcastAdd(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);

  Tensor<Scalar, 2> mat(M, N);
  Tensor<Scalar, 2> bias(1, N);
  Tensor<Scalar, 2> result(M, N);
  mat.setRandom();
  bias.setRandom();

  Eigen::array<int, 2> bcast = {M, 1};

  for (auto _ : state) {
    result = mat + bias.broadcast(bcast);
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * M * N * sizeof(Scalar) * 2);
}

// --- Rank-4 broadcast (batch x channels x 1 x 1) -> (batch x channels x H x W) ---
static void BM_BroadcastRank4(benchmark::State& state) {
  const int batch = state.range(0);
  const int C = state.range(1);
  const int H = state.range(2);

  Tensor<Scalar, 4> bias(batch, C, 1, 1);
  Tensor<Scalar, 4> result(batch, C, H, H);
  bias.setRandom();

  Eigen::array<int, 4> bcast = {1, 1, H, H};

  for (auto _ : state) {
    result = bias.broadcast(bcast);
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * batch * C * H * H * sizeof(Scalar));
}

static void BroadcastSizes(::benchmark::Benchmark* b) {
  for (int m : {64, 256, 1024}) {
    for (int n : {64, 256, 1024}) {
      b->Args({m, n});
    }
  }
}

static void Rank4Sizes(::benchmark::Benchmark* b) {
  for (int batch : {1, 8}) {
    for (int c : {64, 256}) {
      for (int h : {16, 32}) {
        b->Args({batch, c, h});
      }
    }
  }
}

BENCHMARK(BM_BroadcastRow)->Apply(BroadcastSizes);
BENCHMARK(BM_BroadcastCol)->Apply(BroadcastSizes);
BENCHMARK(BM_BroadcastAdd)->Apply(BroadcastSizes);
BENCHMARK(BM_BroadcastRank4)->Apply(Rank4Sizes);
