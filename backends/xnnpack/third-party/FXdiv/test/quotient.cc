#include <gtest/gtest.h>

#include <fxdiv.h>

TEST(uint32_t, cases) {
	EXPECT_EQ(UINT32_C(42) / UINT32_C(7),
		fxdiv_quotient_uint32_t(UINT32_C(42),
			fxdiv_init_uint32_t(UINT32_C(7))));
	EXPECT_EQ(UINT32_C(42) / UINT32_C(6),
		fxdiv_quotient_uint32_t(UINT32_C(42),
			fxdiv_init_uint32_t(UINT32_C(6))));
	EXPECT_EQ(UINT32_C(42) / UINT32_C(5),
		fxdiv_quotient_uint32_t(UINT32_C(42),
			fxdiv_init_uint32_t(UINT32_C(5))));
	EXPECT_EQ(UINT32_C(1) / UINT32_C(1),
		fxdiv_quotient_uint32_t(UINT32_C(1),
			fxdiv_init_uint32_t(UINT32_C(1))));
	EXPECT_EQ(UINT32_MAX / UINT32_C(1),
		fxdiv_quotient_uint32_t(UINT32_MAX,
			fxdiv_init_uint32_t(UINT32_C(1))));
	EXPECT_EQ(UINT32_MAX / UINT32_MAX,
		fxdiv_quotient_uint32_t(UINT32_MAX,
			fxdiv_init_uint32_t(UINT32_MAX)));
	EXPECT_EQ((UINT32_MAX - 1) / UINT32_MAX,
		fxdiv_quotient_uint32_t(UINT32_MAX - 1,
			fxdiv_init_uint32_t(UINT32_MAX)));
	EXPECT_EQ(UINT32_MAX / (UINT32_MAX - 1),
		fxdiv_quotient_uint32_t(UINT32_MAX,
			fxdiv_init_uint32_t(UINT32_MAX - 1)));
	EXPECT_EQ(UINT32_C(0) / UINT32_C(1),
		fxdiv_quotient_uint32_t(UINT32_C(0),
			fxdiv_init_uint32_t(UINT32_C(1))));
}

TEST(uint32_t, divide_by_1) {
	const fxdiv_divisor_uint32_t d = fxdiv_init_uint32_t(UINT32_C(1));
	const uint32_t step = UINT32_C(487);
	for (uint32_t n = 0; n <= UINT32_MAX - step + 1; n += step) {
		EXPECT_EQ(n, fxdiv_quotient_uint32_t(n, d));
	}
}

TEST(uint32_t, divide_zero) {
	const uint32_t step = UINT32_C(491);
	for (uint32_t d = 1; d <= UINT32_MAX - step + 1; d += step) {
		EXPECT_EQ(0,
			fxdiv_quotient_uint32_t(0,
				fxdiv_init_uint32_t(d)));
	}
}

TEST(uint32_t, divide_by_n_minus_1) {
	const uint32_t step = UINT32_C(523);
	for (uint32_t n = 3; n <= UINT32_MAX - step + 1; n += step) {
		EXPECT_EQ(UINT32_C(1),
			fxdiv_quotient_uint32_t(n,
				fxdiv_init_uint32_t(n - 1)));
	}
}

TEST(uint32_t, divide_by_power_of_2) {
	const uint32_t step = UINT32_C(25183);
	for (uint32_t p = 0; p < 32; p += 1) {
		const fxdiv_divisor_uint32_t divisor =
			fxdiv_init_uint32_t(UINT32_C(1) << p);
		for (uint32_t n = 0; n <= UINT32_MAX - step + 1; n += step) {
			EXPECT_EQ(n >> p, fxdiv_quotient_uint32_t(n, divisor));
		}
	}
}

TEST(uint32_t, divide_by_power_of_2_plus_1) {
	const uint32_t step = UINT32_C(25183);
	for (uint32_t p = 0; p < 32; p += 1) {
		const uint32_t d = (UINT32_C(1) << p) + UINT32_C(1);
		const fxdiv_divisor_uint32_t divisor = fxdiv_init_uint32_t(d);
		for (uint32_t n = 0; n <= UINT32_MAX - step + 1; n += step) {
			EXPECT_EQ(n / d, fxdiv_quotient_uint32_t(n, divisor));
		}
	}
}

TEST(uint32_t, divide_by_power_of_2_minus_1) {
	const uint32_t step = UINT64_C(25183);
	for (uint32_t p = 0; p < 32; p += 1) {
		const uint32_t d = (UINT32_C(2) << p) - UINT32_C(1);
		const fxdiv_divisor_uint32_t divisor = fxdiv_init_uint32_t(d);
		for (uint32_t n = 0; n <= UINT32_MAX - step + 1; n += step) {
			EXPECT_EQ(n / d, fxdiv_quotient_uint32_t(n, divisor));
		}
	}
}

TEST(uint32_t, match_native) {
	const uint32_t stepD = UINT32_C(821603);
	const uint32_t stepN = UINT32_C(821641);
	for (uint32_t d = UINT32_MAX; d >= stepD; d -= stepD) {
		const fxdiv_divisor_uint32_t divisor = fxdiv_init_uint32_t(d);
		for (uint32_t n = UINT32_MAX; n >= stepN; n -= stepN) {
			EXPECT_EQ(n / d, fxdiv_quotient_uint32_t(n, divisor));
		}
	}
}

TEST(uint64_t, cases) {
	EXPECT_EQ(UINT64_C(42) / UINT64_C(7),
		fxdiv_quotient_uint64_t(UINT64_C(42),
			fxdiv_init_uint64_t(UINT64_C(7))));
	EXPECT_EQ(UINT64_C(42) / UINT64_C(6),
		fxdiv_quotient_uint64_t(UINT64_C(42),
			fxdiv_init_uint64_t(UINT64_C(6))));
	EXPECT_EQ(UINT64_C(42) / UINT64_C(5),
		fxdiv_quotient_uint64_t(UINT64_C(42),
			fxdiv_init_uint64_t(UINT64_C(5))));
	EXPECT_EQ(UINT64_C(1) / UINT64_C(1),
		fxdiv_quotient_uint64_t(UINT64_C(1),
			fxdiv_init_uint64_t(UINT64_C(1))));
	EXPECT_EQ(UINT64_MAX / UINT64_C(1),
		fxdiv_quotient_uint64_t(UINT64_MAX,
			fxdiv_init_uint64_t(UINT64_C(1))));
	EXPECT_EQ(UINT64_MAX / UINT64_MAX,
		fxdiv_quotient_uint64_t(UINT64_MAX,
			fxdiv_init_uint64_t(UINT64_MAX)));
	EXPECT_EQ((UINT64_MAX - 1) / UINT64_MAX,
		fxdiv_quotient_uint64_t(UINT64_MAX - 1,
			fxdiv_init_uint64_t(UINT64_MAX)));
	EXPECT_EQ(UINT64_MAX / (UINT64_MAX - 1),
		fxdiv_quotient_uint64_t(UINT64_MAX,
			fxdiv_init_uint64_t(UINT64_MAX - 1)));
	EXPECT_EQ(UINT64_C(0) / UINT64_C(1),
		fxdiv_quotient_uint64_t(UINT64_C(0),
			fxdiv_init_uint64_t(UINT64_C(1))));
}

TEST(uint64_t, divide_by_1) {
	const fxdiv_divisor_uint64_t d = fxdiv_init_uint64_t(UINT64_C(1));
	const uint64_t step = UINT64_C(2048116998241);
	for (uint64_t n = 0; n <= UINT64_MAX - step + 1; n += step) {
		EXPECT_EQ(n, fxdiv_quotient_uint64_t(n, d));
	}
}

TEST(uint64_t, divide_zero) {
	const uint64_t step = UINT64_C(8624419641811);
	for (uint64_t d = 1; d <= UINT64_MAX - step + 1; d += step) {
		EXPECT_EQ(0,
			fxdiv_quotient_uint64_t(0,
				fxdiv_init_uint64_t(d)));
	}
}

TEST(uint64_t, divide_by_n_minus_1) {
	const uint64_t step = UINT64_C(8624419641833);
	for (uint64_t n = 3; n <= UINT64_MAX - step + 1; n += step) {
		EXPECT_EQ(UINT64_C(1),
			fxdiv_quotient_uint64_t(n,
				fxdiv_init_uint64_t(n - 1)));
	}
}

TEST(uint64_t, divide_by_power_of_2) {
	const uint64_t step = UINT64_C(93400375993241);
	for (uint32_t p = 0; p < 64; p += 1) {
		const fxdiv_divisor_uint64_t divisor =
			fxdiv_init_uint64_t(UINT64_C(1) << p);
		for (uint64_t n = 0; n <= UINT64_MAX - step + 1; n += step) {
			EXPECT_EQ(n >> p, fxdiv_quotient_uint64_t(n, divisor));
		}
	}
}

TEST(uint64_t, divide_by_power_of_2_plus_1) {
	const uint64_t step = UINT64_C(93400375993241);
	for (uint32_t p = 0; p < 64; p += 1) {
		const uint64_t d = (UINT64_C(1) << p) + UINT64_C(1);
		const fxdiv_divisor_uint64_t divisor = fxdiv_init_uint64_t(d);
		for (uint64_t n = 0; n <= UINT64_MAX - step + 1; n += step) {
			EXPECT_EQ(n / d, fxdiv_quotient_uint64_t(n, divisor));
		}
	}
}

TEST(uint64_t, divide_by_power_of_2_minus_1) {
	const uint64_t step = UINT64_C(93400375993241);
	for (uint32_t p = 0; p < 64; p += 1) {
		const uint64_t d = (UINT64_C(2) << p) - UINT64_C(1);
		const fxdiv_divisor_uint64_t divisor = fxdiv_init_uint64_t(d);
		for (uint64_t n = 0; n <= UINT64_MAX - step + 1; n += step) {
			EXPECT_EQ(n / d, fxdiv_quotient_uint64_t(n, divisor));
		}
	}
}

TEST(uint64_t, match_native) {
	const uint64_t stepD = UINT64_C(7093600525704701);
	const uint64_t stepN = UINT64_C(7093600525704677);
	for (uint64_t d = UINT64_MAX; d >= stepD; d -= stepD) {
		const fxdiv_divisor_uint64_t divisor = fxdiv_init_uint64_t(d);
		for (uint64_t n = UINT64_MAX; n >= stepN; n -= stepN) {
			EXPECT_EQ(n / d, fxdiv_quotient_uint64_t(n, divisor));
		}
	}
}

TEST(size_t, cases) {
	EXPECT_EQ(42 / 7,
		fxdiv_quotient_size_t(42,
			fxdiv_init_size_t(7)));
	EXPECT_EQ(42 / 6,
		fxdiv_quotient_size_t(42,
			fxdiv_init_size_t(6)));
	EXPECT_EQ(42 / 5,
		fxdiv_quotient_size_t(42,
			fxdiv_init_size_t(5)));
	EXPECT_EQ(1 / 1,
		fxdiv_quotient_size_t(1,
			fxdiv_init_size_t(1)));
	EXPECT_EQ(SIZE_MAX / 1,
		fxdiv_quotient_size_t(SIZE_MAX,
			fxdiv_init_size_t(1)));
	EXPECT_EQ(SIZE_MAX / SIZE_MAX,
		fxdiv_quotient_size_t(SIZE_MAX,
			fxdiv_init_size_t(SIZE_MAX)));
	EXPECT_EQ((SIZE_MAX - 1) / SIZE_MAX,
		fxdiv_quotient_size_t(SIZE_MAX - 1,
			fxdiv_init_size_t(SIZE_MAX)));
	EXPECT_EQ(SIZE_MAX / (SIZE_MAX - 1),
		fxdiv_quotient_size_t(SIZE_MAX,
			fxdiv_init_size_t(SIZE_MAX - 1)));
	EXPECT_EQ(0 / 1,
		fxdiv_quotient_size_t(0,
			fxdiv_init_size_t(1)));
}
