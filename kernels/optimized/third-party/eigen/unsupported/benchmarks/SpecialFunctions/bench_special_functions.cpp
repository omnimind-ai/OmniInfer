// Benchmarks for special functions beyond what bench_cwise_math.cpp covers.
// Includes Bessel functions, two-argument functions (igamma, betainc),
// and additional functions (lgamma, digamma, zeta, polygamma).

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <unsupported/Eigen/SpecialFunctions>

using namespace Eigen;

// Macro for unary special functions on arrays.
#define BENCH_SPECIAL_UNARY(NAME, EXPR, LO, HI)                                                          \
  template <typename Scalar>                                                                             \
  static void BM_##NAME(benchmark::State& state) {                                                       \
    const Index n = state.range(0);                                                                      \
    using Arr = Array<Scalar, Dynamic, 1>;                                                               \
    Arr a = (Arr::Random(n) + Scalar(1)) * Scalar((double(HI) - double(LO)) / 2.0) + Scalar(LO);         \
    Arr b(n);                                                                                            \
    for (auto _ : state) {                                                                               \
      b = EXPR;                                                                                          \
      benchmark::DoNotOptimize(b.data());                                                                \
    }                                                                                                    \
    state.counters["Elements/s"] = benchmark::Counter(n, benchmark::Counter::kIsIterationInvariantRate); \
    state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar) * 2);                                \
  }

// Macro for binary special functions on arrays.
#define BENCH_SPECIAL_BINARY(NAME, EXPR, LO_A, HI_A, LO_B, HI_B)                                         \
  template <typename Scalar>                                                                             \
  static void BM_##NAME(benchmark::State& state) {                                                       \
    const Index n = state.range(0);                                                                      \
    using Arr = Array<Scalar, Dynamic, 1>;                                                               \
    Arr a = (Arr::Random(n) + Scalar(1)) * Scalar((double(HI_A) - double(LO_A)) / 2.0) + Scalar(LO_A);   \
    Arr b = (Arr::Random(n) + Scalar(1)) * Scalar((double(HI_B) - double(LO_B)) / 2.0) + Scalar(LO_B);   \
    Arr c(n);                                                                                            \
    for (auto _ : state) {                                                                               \
      c = EXPR;                                                                                          \
      benchmark::DoNotOptimize(c.data());                                                                \
    }                                                                                                    \
    state.counters["Elements/s"] = benchmark::Counter(n, benchmark::Counter::kIsIterationInvariantRate); \
    state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar) * 3);                                \
  }

// --- Unary special functions ---
BENCH_SPECIAL_UNARY(Lgamma, Eigen::lgamma(a), 0.1, 20)
BENCH_SPECIAL_UNARY(Digamma, Eigen::digamma(a), 0.1, 20)

// --- Bessel functions (first kind) ---
BENCH_SPECIAL_UNARY(BesselI0, Eigen::bessel_i0(a), 0, 10)
BENCH_SPECIAL_UNARY(BesselI1, Eigen::bessel_i1(a), 0, 10)
BENCH_SPECIAL_UNARY(BesselI0e, Eigen::bessel_i0e(a), 0, 100)
BENCH_SPECIAL_UNARY(BesselI1e, Eigen::bessel_i1e(a), 0, 100)
BENCH_SPECIAL_UNARY(BesselJ0, Eigen::bessel_j0(a), 0, 20)
BENCH_SPECIAL_UNARY(BesselJ1, Eigen::bessel_j1(a), 0, 20)

// --- Bessel functions (second kind) ---
BENCH_SPECIAL_UNARY(BesselY0, Eigen::bessel_y0(a), 0.1, 20)
BENCH_SPECIAL_UNARY(BesselY1, Eigen::bessel_y1(a), 0.1, 20)
BENCH_SPECIAL_UNARY(BesselK0, Eigen::bessel_k0(a), 0.1, 20)
BENCH_SPECIAL_UNARY(BesselK1, Eigen::bessel_k1(a), 0.1, 20)
BENCH_SPECIAL_UNARY(BesselK0e, Eigen::bessel_k0e(a), 0.1, 100)
BENCH_SPECIAL_UNARY(BesselK1e, Eigen::bessel_k1e(a), 0.1, 100)

// --- Two-argument functions ---
BENCH_SPECIAL_BINARY(Igamma, Eigen::igamma(a, b), 0.1, 10, 0.1, 10)
BENCH_SPECIAL_BINARY(Igammac, Eigen::igammac(a, b), 0.1, 10, 0.1, 10)
BENCH_SPECIAL_BINARY(Zeta, Eigen::zeta(a, b), 1.1, 10, 0.1, 10)
BENCH_SPECIAL_BINARY(Polygamma, Eigen::polygamma(a, b), 1, 4, 0.1, 10)

// --- Ternary: betainc ---
template <typename Scalar>
static void BM_Betainc(benchmark::State& state) {
  const Index n = state.range(0);
  using Arr = Array<Scalar, Dynamic, 1>;
  Arr a = (Arr::Random(n) + Scalar(1)) * Scalar(2.5) + Scalar(0.5);  // [0.5, 5.5]
  Arr b = (Arr::Random(n) + Scalar(1)) * Scalar(2.5) + Scalar(0.5);
  Arr x = (Arr::Random(n) + Scalar(1)) * Scalar(0.5);  // [0, 1]
  Arr result(n);
  for (auto _ : state) {
    result = Eigen::betainc(a, b, x);
    benchmark::DoNotOptimize(result.data());
  }
  state.counters["Elements/s"] = benchmark::Counter(n, benchmark::Counter::kIsIterationInvariantRate);
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar) * 4);
}

static void SpecialSizes(::benchmark::Benchmark* b) {
  for (int n : {256, 4096, 65536, 1048576}) b->Arg(n);
}

// --- Register float ---
BENCHMARK(BM_Lgamma<float>)->Apply(SpecialSizes)->Name("Lgamma_float");
BENCHMARK(BM_Digamma<float>)->Apply(SpecialSizes)->Name("Digamma_float");
BENCHMARK(BM_BesselI0<float>)->Apply(SpecialSizes)->Name("BesselI0_float");
BENCHMARK(BM_BesselI1<float>)->Apply(SpecialSizes)->Name("BesselI1_float");
BENCHMARK(BM_BesselI0e<float>)->Apply(SpecialSizes)->Name("BesselI0e_float");
BENCHMARK(BM_BesselI1e<float>)->Apply(SpecialSizes)->Name("BesselI1e_float");
BENCHMARK(BM_BesselJ0<float>)->Apply(SpecialSizes)->Name("BesselJ0_float");
BENCHMARK(BM_BesselJ1<float>)->Apply(SpecialSizes)->Name("BesselJ1_float");
BENCHMARK(BM_BesselY0<float>)->Apply(SpecialSizes)->Name("BesselY0_float");
BENCHMARK(BM_BesselY1<float>)->Apply(SpecialSizes)->Name("BesselY1_float");
BENCHMARK(BM_BesselK0<float>)->Apply(SpecialSizes)->Name("BesselK0_float");
BENCHMARK(BM_BesselK1<float>)->Apply(SpecialSizes)->Name("BesselK1_float");
BENCHMARK(BM_BesselK0e<float>)->Apply(SpecialSizes)->Name("BesselK0e_float");
BENCHMARK(BM_BesselK1e<float>)->Apply(SpecialSizes)->Name("BesselK1e_float");
BENCHMARK(BM_Igamma<float>)->Apply(SpecialSizes)->Name("Igamma_float");
BENCHMARK(BM_Igammac<float>)->Apply(SpecialSizes)->Name("Igammac_float");
BENCHMARK(BM_Betainc<float>)->Apply(SpecialSizes)->Name("Betainc_float");
BENCHMARK(BM_Zeta<float>)->Apply(SpecialSizes)->Name("Zeta_float");
BENCHMARK(BM_Polygamma<float>)->Apply(SpecialSizes)->Name("Polygamma_float");

// --- Register double ---
BENCHMARK(BM_Lgamma<double>)->Apply(SpecialSizes)->Name("Lgamma_double");
BENCHMARK(BM_Digamma<double>)->Apply(SpecialSizes)->Name("Digamma_double");
BENCHMARK(BM_BesselI0<double>)->Apply(SpecialSizes)->Name("BesselI0_double");
BENCHMARK(BM_BesselI1<double>)->Apply(SpecialSizes)->Name("BesselI1_double");
BENCHMARK(BM_BesselJ0<double>)->Apply(SpecialSizes)->Name("BesselJ0_double");
BENCHMARK(BM_BesselJ1<double>)->Apply(SpecialSizes)->Name("BesselJ1_double");
BENCHMARK(BM_BesselY0<double>)->Apply(SpecialSizes)->Name("BesselY0_double");
BENCHMARK(BM_BesselY1<double>)->Apply(SpecialSizes)->Name("BesselY1_double");
BENCHMARK(BM_BesselK0<double>)->Apply(SpecialSizes)->Name("BesselK0_double");
BENCHMARK(BM_BesselK1<double>)->Apply(SpecialSizes)->Name("BesselK1_double");
BENCHMARK(BM_Igamma<double>)->Apply(SpecialSizes)->Name("Igamma_double");
BENCHMARK(BM_Igammac<double>)->Apply(SpecialSizes)->Name("Igammac_double");
BENCHMARK(BM_Betainc<double>)->Apply(SpecialSizes)->Name("Betainc_double");
BENCHMARK(BM_Zeta<double>)->Apply(SpecialSizes)->Name("Zeta_double");
BENCHMARK(BM_Polygamma<double>)->Apply(SpecialSizes)->Name("Polygamma_double");
