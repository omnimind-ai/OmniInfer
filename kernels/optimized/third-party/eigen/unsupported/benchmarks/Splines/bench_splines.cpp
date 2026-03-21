// Benchmarks for Eigen Spline module.
// Tests fitting, evaluation, and derivative computation.

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <unsupported/Eigen/Splines>

using namespace Eigen;

typedef double Scalar;

// --- Spline fitting (interpolation) ---
template <int Dim, int Degree>
static void BM_SplineFit(benchmark::State& state) {
  const int n = state.range(0);

  typedef Spline<Scalar, Dim> SplineType;
  typedef typename SplineType::PointType PointType;

  // Generate random points.
  Matrix<Scalar, Dim, Dynamic> pts(Dim, n);
  pts.setRandom();

  for (auto _ : state) {
    SplineType spline = SplineFitting<SplineType>::Interpolate(pts, Degree);
    benchmark::DoNotOptimize(spline.knots().data());
    benchmark::ClobberMemory();
  }
}

// --- Spline evaluation ---
template <int Dim, int Degree>
static void BM_SplineEval(benchmark::State& state) {
  const int n = state.range(0);  // number of control points for fitting
  const int neval = 1000;        // number of evaluation points

  typedef Spline<Scalar, Dim> SplineType;

  Matrix<Scalar, Dim, Dynamic> pts(Dim, n);
  pts.setRandom();
  SplineType spline = SplineFitting<SplineType>::Interpolate(pts, Degree);

  // Generate evaluation parameters in [0, 1].
  VectorXd u = VectorXd::LinSpaced(neval, 0, 1);

  for (auto _ : state) {
    for (int i = 0; i < neval; ++i) {
      auto pt = spline(u(i));
      benchmark::DoNotOptimize(pt.data());
    }
    benchmark::ClobberMemory();
  }
  state.counters["Evals/s"] = benchmark::Counter(neval, benchmark::Counter::kIsIterationInvariantRate);
}

// --- Spline derivative evaluation ---
template <int Dim, int Degree>
static void BM_SplineDerivatives(benchmark::State& state) {
  const int n = state.range(0);
  const int neval = 1000;

  typedef Spline<Scalar, Dim> SplineType;

  Matrix<Scalar, Dim, Dynamic> pts(Dim, n);
  pts.setRandom();
  SplineType spline = SplineFitting<SplineType>::Interpolate(pts, Degree);

  VectorXd u = VectorXd::LinSpaced(neval, 0, 1);

  for (auto _ : state) {
    for (int i = 0; i < neval; ++i) {
      auto derivs = spline.derivatives(u(i), 1);
      benchmark::DoNotOptimize(derivs.data());
    }
    benchmark::ClobberMemory();
  }
  state.counters["Evals/s"] = benchmark::Counter(neval, benchmark::Counter::kIsIterationInvariantRate);
}

static void SplineSizes(::benchmark::Benchmark* b) {
  for (int n : {10, 50, 200, 1000}) {
    b->Arg(n);
  }
}

// 2D cubic splines
BENCHMARK(BM_SplineFit<2, 3>)->Apply(SplineSizes)->Name("SplineFit_2D_Cubic");
BENCHMARK(BM_SplineEval<2, 3>)->Apply(SplineSizes)->Name("SplineEval_2D_Cubic");
BENCHMARK(BM_SplineDerivatives<2, 3>)->Apply(SplineSizes)->Name("SplineDerivatives_2D_Cubic");

// 3D cubic splines
BENCHMARK(BM_SplineFit<3, 3>)->Apply(SplineSizes)->Name("SplineFit_3D_Cubic");
BENCHMARK(BM_SplineEval<3, 3>)->Apply(SplineSizes)->Name("SplineEval_3D_Cubic");
BENCHMARK(BM_SplineDerivatives<3, 3>)->Apply(SplineSizes)->Name("SplineDerivatives_3D_Cubic");

// 2D quintic splines
BENCHMARK(BM_SplineFit<2, 5>)->Apply(SplineSizes)->Name("SplineFit_2D_Quintic");
BENCHMARK(BM_SplineEval<2, 5>)->Apply(SplineSizes)->Name("SplineEval_2D_Quintic");
