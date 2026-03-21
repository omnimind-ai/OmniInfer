// Benchmarks for unsupported iterative solvers: GMRES, MINRES, IDRS, IDRSTABL, BiCGSTABL, DGMRES.

#include <benchmark/benchmark.h>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

using namespace Eigen;

typedef double Scalar;
typedef SparseMatrix<Scalar> SpMat;
typedef Matrix<Scalar, Dynamic, 1> Vec;

// Generate a SPD sparse matrix (Laplacian-like with diagonal dominance).
static SpMat generateSPD(int n, int bandwidth) {
  SpMat A(n, n);
  std::vector<Triplet<Scalar>> trips;
  trips.reserve(n * (2 * bandwidth + 1));
  for (int i = 0; i < n; ++i) {
    Scalar diag = 0;
    for (int j = std::max(0, i - bandwidth); j < std::min(n, i + bandwidth + 1); ++j) {
      if (i != j) {
        Scalar val = -1.0 / (1 + std::abs(i - j));
        trips.emplace_back(i, j, val);
        diag -= val;
      }
    }
    trips.emplace_back(i, i, diag + 1.0);
  }
  A.setFromTriplets(trips.begin(), trips.end());
  return A;
}

// Generate a general (non-symmetric) sparse matrix.
static SpMat generateGeneral(int n, int bandwidth) {
  SpMat A(n, n);
  std::vector<Triplet<Scalar>> trips;
  trips.reserve(n * (2 * bandwidth + 1));
  for (int i = 0; i < n; ++i) {
    Scalar diag = 0;
    for (int j = std::max(0, i - bandwidth); j < std::min(n, i + bandwidth + 1); ++j) {
      if (i != j) {
        Scalar val = -0.5 / (1 + std::abs(i - j));
        if (j > i) val *= 1.5;  // asymmetry
        trips.emplace_back(i, j, val);
        diag += std::abs(val);
      }
    }
    trips.emplace_back(i, i, diag + 1.0);  // diagonal dominance
  }
  A.setFromTriplets(trips.begin(), trips.end());
  return A;
}

// --- GMRES ---
static void BM_GMRES(benchmark::State& state) {
  int n = state.range(0);
  int bw = state.range(1);
  SpMat A = generateGeneral(n, bw);
  Vec b = Vec::Random(n);

  GMRES<SpMat> solver;
  solver.setMaxIterations(1000);
  solver.setTolerance(1e-10);
  solver.compute(A);

  for (auto _ : state) {
    Vec x = solver.solve(b);
    benchmark::DoNotOptimize(x.data());
    benchmark::ClobberMemory();
  }
  state.counters["iterations"] = solver.iterations();
}

// --- DGMRES ---
static void BM_DGMRES(benchmark::State& state) {
  int n = state.range(0);
  int bw = state.range(1);
  SpMat A = generateGeneral(n, bw);
  Vec b = Vec::Random(n);

  DGMRES<SpMat> solver;
  solver.setMaxIterations(1000);
  solver.setTolerance(1e-10);
  solver.compute(A);

  for (auto _ : state) {
    Vec x = solver.solve(b);
    benchmark::DoNotOptimize(x.data());
    benchmark::ClobberMemory();
  }
  state.counters["iterations"] = solver.iterations();
}

// --- MINRES (SPD matrices) ---
static void BM_MINRES(benchmark::State& state) {
  int n = state.range(0);
  int bw = state.range(1);
  SpMat A = generateSPD(n, bw);
  Vec b = Vec::Random(n);

  MINRES<SpMat> solver;
  solver.setMaxIterations(1000);
  solver.setTolerance(1e-10);
  solver.compute(A);

  for (auto _ : state) {
    Vec x = solver.solve(b);
    benchmark::DoNotOptimize(x.data());
    benchmark::ClobberMemory();
  }
  state.counters["iterations"] = solver.iterations();
}

// --- IDRS ---
static void BM_IDRS(benchmark::State& state) {
  int n = state.range(0);
  int bw = state.range(1);
  SpMat A = generateGeneral(n, bw);
  Vec b = Vec::Random(n);

  IDRS<SpMat> solver;
  solver.setMaxIterations(1000);
  solver.setTolerance(1e-10);
  solver.compute(A);

  for (auto _ : state) {
    Vec x = solver.solve(b);
    benchmark::DoNotOptimize(x.data());
    benchmark::ClobberMemory();
  }
  state.counters["iterations"] = solver.iterations();
}

// --- BiCGSTABL ---
static void BM_BiCGSTABL(benchmark::State& state) {
  int n = state.range(0);
  int bw = state.range(1);
  SpMat A = generateGeneral(n, bw);
  Vec b = Vec::Random(n);

  BiCGSTABL<SpMat> solver;
  solver.setMaxIterations(1000);
  solver.setTolerance(1e-10);
  solver.compute(A);

  for (auto _ : state) {
    Vec x = solver.solve(b);
    benchmark::DoNotOptimize(x.data());
    benchmark::ClobberMemory();
  }
  state.counters["iterations"] = solver.iterations();
}

// --- Compare with CG (supported module, SPD only) ---
static void BM_CG_Reference(benchmark::State& state) {
  int n = state.range(0);
  int bw = state.range(1);
  SpMat A = generateSPD(n, bw);
  Vec b = Vec::Random(n);

  ConjugateGradient<SpMat> solver;
  solver.setMaxIterations(1000);
  solver.setTolerance(1e-10);
  solver.compute(A);

  for (auto _ : state) {
    Vec x = solver.solve(b);
    benchmark::DoNotOptimize(x.data());
    benchmark::ClobberMemory();
  }
  state.counters["iterations"] = solver.iterations();
}

// --- Compare with BiCGSTAB (supported module, general) ---
static void BM_BiCGSTAB_Reference(benchmark::State& state) {
  int n = state.range(0);
  int bw = state.range(1);
  SpMat A = generateGeneral(n, bw);
  Vec b = Vec::Random(n);

  BiCGSTAB<SpMat> solver;
  solver.setMaxIterations(1000);
  solver.setTolerance(1e-10);
  solver.compute(A);

  for (auto _ : state) {
    Vec x = solver.solve(b);
    benchmark::DoNotOptimize(x.data());
    benchmark::ClobberMemory();
  }
  state.counters["iterations"] = solver.iterations();
}

static void SolverSizes(::benchmark::Benchmark* b) {
  for (int n : {1000, 10000, 100000}) {
    for (int bw : {5, 20}) {
      b->Args({n, bw});
    }
  }
}

BENCHMARK(BM_GMRES)->Apply(SolverSizes);
BENCHMARK(BM_DGMRES)->Apply(SolverSizes);
BENCHMARK(BM_MINRES)->Apply(SolverSizes);
BENCHMARK(BM_IDRS)->Apply(SolverSizes);
BENCHMARK(BM_BiCGSTABL)->Apply(SolverSizes);
BENCHMARK(BM_CG_Reference)->Apply(SolverSizes);
BENCHMARK(BM_BiCGSTAB_Reference)->Apply(SolverSizes);
