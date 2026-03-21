#include <benchmark/benchmark.h>

#include <fxdiv.h>

static void fxdiv_divide_uint32_t(benchmark::State& state) {
	const fxdiv_divisor_uint32_t divisor = fxdiv_init_uint32_t(UINT32_C(65537));
	uint32_t x = 0;
	while (state.KeepRunning()) {
		const fxdiv_result_uint32_t result = fxdiv_divide_uint32_t(x++, divisor);
		benchmark::DoNotOptimize(result);
	}
}
BENCHMARK(fxdiv_divide_uint32_t);

static void fxdiv_divide_uint64_t(benchmark::State& state) {
	const fxdiv_divisor_uint64_t divisor = fxdiv_init_uint64_t(UINT64_C(4294967311));
	uint64_t x = 0;
	while (state.KeepRunning()) {
		const fxdiv_result_uint64_t result = fxdiv_divide_uint64_t(x++, divisor);
		benchmark::DoNotOptimize(result);
	}
}
BENCHMARK(fxdiv_divide_uint64_t);

static void native_divide_uint32_t(benchmark::State& state) {
	uint32_t divisor = UINT32_C(65537);
	benchmark::DoNotOptimize(&divisor);
	uint32_t x = 0;
	while (state.KeepRunning()) {
		const uint32_t quotient = x / divisor;
		const uint32_t remainder = x++ % divisor;
		benchmark::DoNotOptimize(quotient);
		benchmark::DoNotOptimize(remainder);
	}
}
BENCHMARK(native_divide_uint32_t);

static void native_divide_uint64_t(benchmark::State& state) {
	uint64_t divisor = UINT64_C(4294967311);
	benchmark::DoNotOptimize(&divisor);
	uint64_t x = 0;
	while (state.KeepRunning()) {
		const uint64_t quotient = x / divisor;
		const uint64_t remainder = x++ % divisor;
		benchmark::DoNotOptimize(quotient);
		benchmark::DoNotOptimize(remainder);
	}
}
BENCHMARK(native_divide_uint64_t);

BENCHMARK_MAIN();
