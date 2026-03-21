// Benchmarks for Eigen AutoDiff module.
// Compares AutoDiff Jacobian computation against NumericalDiff and hand-coded Jacobians.

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>
#include <unsupported/Eigen/NumericalDiff>

using namespace Eigen;

// --- Small functor: Rosenbrock-like (2 inputs -> 2 outputs) ---
struct SmallFunctor {
  typedef Matrix<double, 2, 1> InputType;
  typedef Matrix<double, 2, 1> ValueType;
  typedef Matrix<double, 2, 2> JacobianType;

  enum { InputsAtCompileTime = 2, ValuesAtCompileTime = 2 };

  template <typename T>
  void operator()(const Matrix<T, 2, 1>& x, Matrix<T, 2, 1>* v) const {
    (*v)(0) = T(1) - x(0);
    (*v)(1) = T(10) * (x(1) - x(0) * x(0));
  }
};

// --- Medium functor: chain of operations (6 inputs -> 6 outputs) ---
struct MediumFunctor {
  typedef Matrix<double, 6, 1> InputType;
  typedef Matrix<double, 6, 1> ValueType;
  typedef Matrix<double, 6, 6> JacobianType;

  enum { InputsAtCompileTime = 6, ValuesAtCompileTime = 6 };

  template <typename T>
  void operator()(const Matrix<T, 6, 1>& x, Matrix<T, 6, 1>* v) const {
    (*v)(0) = sin(x(0)) * cos(x(1)) + x(2) * x(2);
    (*v)(1) = exp(x(1) * T(0.1)) + x(3);
    (*v)(2) = x(0) * x(2) - x(4) * x(5);
    (*v)(3) = sqrt(x(3) * x(3) + T(1)) + x(0);
    (*v)(4) = x(4) * x(4) + x(5) * x(5) + x(0) * x(1);
    (*v)(5) = log(x(2) * x(2) + T(1)) + x(3) * x(4);
  }
};

// --- Dynamic-size functor (N inputs -> N outputs) ---
struct DynamicFunctor {
  typedef Matrix<double, Dynamic, 1> InputType;
  typedef Matrix<double, Dynamic, 1> ValueType;
  typedef Matrix<double, Dynamic, Dynamic> JacobianType;

  const int n_;
  DynamicFunctor(int n) : n_(n) {}

  enum { InputsAtCompileTime = Dynamic, ValuesAtCompileTime = Dynamic };

  int inputs() const { return n_; }
  int values() const { return n_; }

  template <typename T>
  void operator()(const Matrix<T, Dynamic, 1>& x, Matrix<T, Dynamic, 1>* v) const {
    v->resize(n_);
    (*v)(0) = T(1) - x(0);
    for (int i = 1; i < n_; ++i) {
      (*v)(i) = T(10) * (x(i) - x(i - 1) * x(i - 1));
    }
  }
};

// Wrapper for NumericalDiff compatibility.
struct SmallFunctorND : SmallFunctor {
  typedef double Scalar;
  int inputs() const { return 2; }
  int values() const { return 2; }
  int operator()(const InputType& x, ValueType& v) const {
    SmallFunctor::operator()(x, &v);
    return 0;
  }
};

struct MediumFunctorND : MediumFunctor {
  typedef double Scalar;
  int inputs() const { return 6; }
  int values() const { return 6; }
  int operator()(const InputType& x, ValueType& v) const {
    MediumFunctor::operator()(x, &v);
    return 0;
  }
};

// --- AutoDiff Jacobian benchmarks ---
template <typename Functor>
static void BM_AutoDiffJacobian(benchmark::State& state, Functor func) {
  AutoDiffJacobian<Functor> adf(func);
  typename Functor::InputType x = Functor::InputType::Random();
  typename Functor::ValueType v;
  typename Functor::JacobianType jac;

  for (auto _ : state) {
    adf(x, &v, &jac);
    benchmark::DoNotOptimize(jac.data());
    benchmark::ClobberMemory();
  }
}

// --- Dynamic AutoDiff Jacobian ---
static void BM_AutoDiffJacobian_Dynamic(benchmark::State& state) {
  int n = state.range(0);
  DynamicFunctor func(n);
  AutoDiffJacobian<DynamicFunctor> adf(func);

  VectorXd x = VectorXd::Random(n);
  VectorXd v(n);
  MatrixXd jac(n, n);

  for (auto _ : state) {
    adf(x, &v, &jac);
    benchmark::DoNotOptimize(jac.data());
    benchmark::ClobberMemory();
  }
}

// --- NumericalDiff benchmarks ---
template <typename Functor>
static void BM_NumericalDiffJacobian(benchmark::State& state, Functor func) {
  NumericalDiff<Functor> ndf(func);
  typename Functor::InputType x = Functor::InputType::Random();
  typename Functor::JacobianType jac;

  for (auto _ : state) {
    ndf.df(x, jac);
    benchmark::DoNotOptimize(jac.data());
    benchmark::ClobberMemory();
  }
}

// --- Hand-coded Jacobian (Rosenbrock) for comparison ---
static void BM_HandCoded_Small(benchmark::State& state) {
  Vector2d x = Vector2d::Random();
  Matrix2d jac;

  for (auto _ : state) {
    jac(0, 0) = -1;
    jac(0, 1) = 0;
    jac(1, 0) = -20 * x(0);
    jac(1, 1) = 10;
    benchmark::DoNotOptimize(jac.data());
    benchmark::ClobberMemory();
  }
}

// --- Scalar AutoDiff evaluation (no Jacobian, just forward pass) ---
static void BM_AutoDiffScalar_Eval(benchmark::State& state) {
  int n = state.range(0);
  using ADScalar = AutoDiffScalar<VectorXd>;
  VectorXd x = VectorXd::Random(n);

  for (auto _ : state) {
    ADScalar sum(0.0, VectorXd::Zero(n));
    for (int i = 0; i < n; ++i) {
      ADScalar xi(x(i), n, i);
      sum += xi * xi + sin(xi);
    }
    benchmark::DoNotOptimize(sum.value());
    benchmark::DoNotOptimize(sum.derivatives().data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK_CAPTURE(BM_AutoDiffJacobian, Small, SmallFunctor());
BENCHMARK_CAPTURE(BM_AutoDiffJacobian, Medium, MediumFunctor());
BENCHMARK(BM_AutoDiffJacobian_Dynamic)->Arg(2)->Arg(6)->Arg(20)->Arg(50)->Arg(100);

BENCHMARK_CAPTURE(BM_NumericalDiffJacobian, Small, SmallFunctorND());
BENCHMARK_CAPTURE(BM_NumericalDiffJacobian, Medium, MediumFunctorND());

BENCHMARK(BM_HandCoded_Small);
BENCHMARK(BM_AutoDiffScalar_Eval)->Arg(2)->Arg(6)->Arg(20)->Arg(50)->Arg(100);
