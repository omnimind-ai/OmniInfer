#include <benchmark/benchmark.h>

#include <fxdiv.h>

static void fxdiv_round_down_uint32_t(benchmark::State& state) {
	const fxdiv_divisor_uint32_t multiple = fxdiv_init_uint32_t(UINT32_C(65537));
	uint32_t x = 0;
	while (state.KeepRunning()) {
		const uint32_t rounded_x = fxdiv_round_down_uint32_t(x++, multiple);
		benchmark::DoNotOptimize(rounded_x);
	}
}
BENCHMARK(fxdiv_round_down_uint32_t);

static void fxdiv_round_down_uint64_t(benchmark::State& state) {
	const fxdiv_divisor_uint64_t multiple = fxdiv_init_uint64_t(UINT64_C(4294967311));
	uint64_t x = 0;
	while (state.KeepRunning()) {
		const uint64_t rounded_x = fxdiv_round_down_uint64_t(x++, multiple);
		benchmark::DoNotOptimize(rounded_x);
	}
}
BENCHMARK(fxdiv_round_down_uint64_t);

static void native_round_down_uint32_t(benchmark::State& state) {
	uint32_t multiple = UINT32_C(65537);
	benchmark::DoNotOptimize(&multiple);
	uint32_t x = 0;
	while (state.KeepRunning()) {
		const uint32_t rounded_x = x++ / multiple * multiple;
		benchmark::DoNotOptimize(rounded_x);
	}
}
BENCHMARK(native_round_down_uint32_t);

static void native_round_down_uint64_t(benchmark::State& state) {
	uint64_t multiple = UINT64_C(4294967311);
	benchmark::DoNotOptimize(&multiple);
	uint64_t x = 0;
	while (state.KeepRunning()) {
		const uint64_t rounded_x = x++ / multiple * multiple;
		benchmark::DoNotOptimize(rounded_x);
	}
}
BENCHMARK(native_round_down_uint64_t);

BENCHMARK_MAIN();
