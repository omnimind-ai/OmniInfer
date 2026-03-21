// Benchmarks for Eigen Tensor convolution (1D and 2D).

#define EIGEN_USE_THREADS

#include <benchmark/benchmark.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>

using namespace Eigen;

typedef float Scalar;

// --- 1D convolution ---
static void BM_Convolve1D(benchmark::State& state) {
  const int input_size = state.range(0);
  const int kernel_size = state.range(1);

  Tensor<Scalar, 1> input(input_size);
  Tensor<Scalar, 1> kernel(kernel_size);
  input.setRandom();
  kernel.setRandom();

  Eigen::array<int, 1> dims = {0};

  for (auto _ : state) {
    Tensor<Scalar, 1> result = input.convolve(kernel, dims);
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
  double flops = 2.0 * (input_size - kernel_size + 1) * kernel_size;
  state.counters["GFLOPS"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// --- 2D convolution ---
static void BM_Convolve2D(benchmark::State& state) {
  const int H = state.range(0);
  const int W = state.range(1);
  const int kH = state.range(2);
  const int kW = state.range(3);

  Tensor<Scalar, 2> input(H, W);
  Tensor<Scalar, 2> kernel(kH, kW);
  input.setRandom();
  kernel.setRandom();

  Eigen::array<int, 2> dims = {0, 1};

  for (auto _ : state) {
    Tensor<Scalar, 2> result = input.convolve(kernel, dims);
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
  double flops = 2.0 * (H - kH + 1) * (W - kW + 1) * kH * kW;
  state.counters["GFLOPS"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// --- 2D convolution with channels (rank-3: C x H x W, convolve on H,W) ---
static void BM_Convolve2D_Channels(benchmark::State& state) {
  const int C = state.range(0);
  const int H = state.range(1);
  const int kH = state.range(2);

  Tensor<Scalar, 3> input(C, H, H);
  Tensor<Scalar, 2> kernel(kH, kH);
  input.setRandom();
  kernel.setRandom();

  Eigen::array<int, 2> dims = {1, 2};

  for (auto _ : state) {
    Tensor<Scalar, 3> result = input.convolve(kernel, dims);
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
  int outH = H - kH + 1;
  double flops = 2.0 * C * outH * outH * kH * kH;
  state.counters["GFLOPS"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// --- 2D convolution with ThreadPool ---
static void BM_Convolve2D_ThreadPool(benchmark::State& state) {
  const int H = state.range(0);
  const int kH = state.range(1);
  const int threads = state.range(2);

  Tensor<Scalar, 2> input(H, H);
  Tensor<Scalar, 2> kernel(kH, kH);
  Tensor<Scalar, 2> result(H - kH + 1, H - kH + 1);
  input.setRandom();
  kernel.setRandom();

  ThreadPool tp(threads);
  ThreadPoolDevice dev(&tp, threads);

  Eigen::array<int, 2> dims = {0, 1};

  for (auto _ : state) {
    result.device(dev) = input.convolve(kernel, dims);
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
  int outH = H - kH + 1;
  double flops = 2.0 * outH * outH * kH * kH;
  state.counters["GFLOPS"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
  state.counters["threads"] = threads;
}

static void Conv1DSizes(::benchmark::Benchmark* b) {
  for (int input : {128, 512, 2048}) {
    for (int kernel : {3, 5, 11}) {
      b->Args({input, kernel});
    }
  }
}

static void Conv2DSizes(::benchmark::Benchmark* b) {
  for (int hw : {32, 64, 128, 224}) {
    for (int k : {3, 5, 7}) {
      b->Args({hw, hw, k, k});
    }
  }
}

static void Conv2DChannelSizes(::benchmark::Benchmark* b) {
  for (int c : {3, 64, 128}) {
    for (int hw : {16, 32, 56}) {
      for (int k : {3, 5}) {
        b->Args({c, hw, k});
      }
    }
  }
}

static void Conv2DThreadPoolSizes(::benchmark::Benchmark* b) {
  for (int hw : {64, 128, 224}) {
    for (int k : {3, 5}) {
      for (int threads : {2, 4, 8}) {
        b->Args({hw, k, threads});
      }
    }
  }
}

BENCHMARK(BM_Convolve1D)->Apply(Conv1DSizes);
BENCHMARK(BM_Convolve2D)->Apply(Conv2DSizes);
BENCHMARK(BM_Convolve2D_Channels)->Apply(Conv2DChannelSizes);
BENCHMARK(BM_Convolve2D_ThreadPool)->Apply(Conv2DThreadPoolSizes);
