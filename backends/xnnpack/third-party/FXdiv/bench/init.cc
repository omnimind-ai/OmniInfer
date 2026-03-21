#include <benchmark/benchmark.h>

#include <fxdiv.h>

static void init_uint32_t(benchmark::State& state) {
	uint32_t d = UINT32_C(0x1971DB6B);
	while (state.KeepRunning()) {
		const fxdiv_divisor_uint32_t divisor = fxdiv_init_uint32_t(d++);
		benchmark::DoNotOptimize(divisor);
	}
}
BENCHMARK(init_uint32_t);

static void init_uint64_t(benchmark::State& state) {
	uint64_t d = UINT64_C(0x425E892B38148FAD);
	while (state.KeepRunning()) {
		const fxdiv_divisor_uint64_t divisor = fxdiv_init_uint64_t(d++);
		benchmark::DoNotOptimize(divisor);
	}
}
BENCHMARK(init_uint64_t);

BENCHMARK_MAIN();
