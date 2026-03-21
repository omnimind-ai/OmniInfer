#include <gtest/gtest.h>

#include <fxdiv.h>

TEST(uint32_t, cases) {
	EXPECT_EQ(UINT32_C(0),              fxdiv_mulhi_uint32_t(UINT32_C(0),         UINT32_C(0)));
	EXPECT_EQ(UINT32_C(0),              fxdiv_mulhi_uint32_t(UINT32_C(1),         UINT32_C(1)));
	EXPECT_EQ(UINT32_C(0),              fxdiv_mulhi_uint32_t(UINT32_C(1),         UINT32_MAX));
	EXPECT_EQ(UINT32_C(0),              fxdiv_mulhi_uint32_t(UINT32_MAX,          UINT32_C(1)));
	EXPECT_EQ(UINT32_C(1),              fxdiv_mulhi_uint32_t(UINT32_C(65536),     UINT32_C(65536)));
	EXPECT_EQ(UINT32_MAX - UINT32_C(1), fxdiv_mulhi_uint32_t(UINT32_MAX,          UINT32_MAX));
	EXPECT_EQ(UINT32_C(28389652),       fxdiv_mulhi_uint32_t(UINT32_C(123456789), UINT32_C(987654321)));
	EXPECT_EQ(UINT32_C(28389652),       fxdiv_mulhi_uint32_t(UINT32_C(987654321), UINT32_C(123456789)));
}

TEST(uint64_t, cases) {
	EXPECT_EQ(UINT64_C(0),              fxdiv_mulhi_uint64_t(UINT64_C(0),                 UINT64_C(0)));
	EXPECT_EQ(UINT64_C(0),              fxdiv_mulhi_uint64_t(UINT64_C(1),                 UINT64_C(1)));
	EXPECT_EQ(UINT64_C(0),              fxdiv_mulhi_uint64_t(UINT64_C(1),                 UINT64_MAX));
	EXPECT_EQ(UINT64_C(0),              fxdiv_mulhi_uint64_t(UINT64_MAX,                  UINT64_C(1)));
	EXPECT_EQ(UINT64_C(1),              fxdiv_mulhi_uint64_t(UINT64_C(4294967296),        UINT64_C(4294967296)));
	EXPECT_EQ(UINT64_MAX - UINT64_C(1), fxdiv_mulhi_uint64_t(UINT64_MAX,                  UINT64_MAX));
	EXPECT_EQ(UINT64_C(66099812259603), fxdiv_mulhi_uint64_t(UINT64_C(12345678987654321), UINT64_C(98765432123456789)));
	EXPECT_EQ(UINT64_C(66099812259603), fxdiv_mulhi_uint64_t(UINT64_C(98765432123456789), UINT64_C(12345678987654321)));
}

TEST(size_t, cases) {
	EXPECT_EQ(0,            fxdiv_mulhi_size_t(0,        0));
	EXPECT_EQ(0,            fxdiv_mulhi_size_t(1,        1));
	EXPECT_EQ(0,            fxdiv_mulhi_size_t(1,        SIZE_MAX));
	EXPECT_EQ(0,            fxdiv_mulhi_size_t(SIZE_MAX, 1));
	EXPECT_EQ(SIZE_MAX - 1, fxdiv_mulhi_size_t(SIZE_MAX, SIZE_MAX));
}
