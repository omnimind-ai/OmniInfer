// Benchmarks for Kronecker product (dense and sparse).

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unsupported/Eigen/KroneckerProduct>

using namespace Eigen;

typedef double Scalar;
typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
typedef SparseMatrix<Scalar> SpMat;

// --- Dense Kronecker product ---
static void BM_KroneckerDense(benchmark::State& state) {
  int na = state.range(0);
  int nb = state.range(1);

  Mat A = Mat::Random(na, na);
  Mat B = Mat::Random(nb, nb);

  for (auto _ : state) {
    Mat C = kroneckerProduct(A, B).eval();
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }
  int outSize = na * nb;
  state.counters["output_size"] = outSize;
}

// --- Sparse Kronecker product ---
static void BM_KroneckerSparse(benchmark::State& state) {
  int na = state.range(0);
  int nb = state.range(1);

  // Create sparse identity-like matrices with some fill.
  SpMat A(na, na);
  SpMat B(nb, nb);

  std::vector<Triplet<Scalar>> tripsA, tripsB;
  for (int i = 0; i < na; ++i) {
    tripsA.emplace_back(i, i, 2.0);
    if (i + 1 < na) {
      tripsA.emplace_back(i, i + 1, -1.0);
      tripsA.emplace_back(i + 1, i, -1.0);
    }
  }
  for (int i = 0; i < nb; ++i) {
    tripsB.emplace_back(i, i, 2.0);
    if (i + 1 < nb) {
      tripsB.emplace_back(i, i + 1, -1.0);
      tripsB.emplace_back(i + 1, i, -1.0);
    }
  }
  A.setFromTriplets(tripsA.begin(), tripsA.end());
  B.setFromTriplets(tripsB.begin(), tripsB.end());

  for (auto _ : state) {
    SpMat C = kroneckerProduct(A, B).eval();
    benchmark::DoNotOptimize(C.valuePtr());
    benchmark::ClobberMemory();
  }
  state.counters["output_size"] = na * nb;
}

static void KroneckerSizes(::benchmark::Benchmark* b) {
  for (int na : {4, 8, 16}) {
    for (int nb : {4, 8, 16}) {
      b->Args({na, nb});
    }
  }
}

static void KroneckerSparseSizes(::benchmark::Benchmark* b) {
  for (int na : {16, 32, 64, 128}) {
    for (int nb : {16, 32, 64, 128}) {
      b->Args({na, nb});
    }
  }
}

BENCHMARK(BM_KroneckerDense)->Apply(KroneckerSizes);
BENCHMARK(BM_KroneckerSparse)->Apply(KroneckerSparseSizes);
