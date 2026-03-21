#include <benchmark/benchmark.h>

#include <fxdiv.h>

static void fxdiv_mulext_uint32_t(benchmark::State& state) {
	uint32_t c = UINT32_C(0x1971DB6B);
	benchmark::DoNotOptimize(&c);
	uint32_t d = c;
	while (state.KeepRunning()) {
		const uint64_t product = fxdiv_mulext_uint32_t(c, d++);
		benchmark::DoNotOptimize(product);
	}
}
BENCHMARK(fxdiv_mulext_uint32_t);

static void native_mulext_uint32_t(benchmark::State& state) {
	uint32_t c = UINT32_C(0x1971DB6B);
	benchmark::DoNotOptimize(&c);
	uint32_t d = c;
	while (state.KeepRunning()) {
		const uint64_t product = (uint64_t) c * (uint64_t) (d++);
		benchmark::DoNotOptimize(product);
	}
}
BENCHMARK(native_mulext_uint32_t);

static void fxdiv_mulhi_uint32_t(benchmark::State& state) {
	const uint32_t c = UINT32_C(0x1971DB6B);
	uint32_t x = c;
	while (state.KeepRunning()) {
		const uint32_t product = fxdiv_mulhi_uint32_t(c, x++);
		benchmark::DoNotOptimize(product);
	}
}
BENCHMARK(fxdiv_mulhi_uint32_t);

static void fxdiv_mulhi_uint64_t(benchmark::State& state) {
	const uint64_t c = UINT64_C(0x425E892B38148FAD);
	uint64_t x = c;
	while (state.KeepRunning()) {
		const uint64_t product = fxdiv_mulhi_uint64_t(c, x++);
		benchmark::DoNotOptimize(product);
	}
}
BENCHMARK(fxdiv_mulhi_uint64_t);

BENCHMARK_MAIN();
