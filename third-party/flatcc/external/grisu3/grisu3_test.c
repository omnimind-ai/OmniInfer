#include <inttypes.h>
#include <string.h>
#include <stdio.h>

#include "grisu3_parse.h"
#include "grisu3_print.h"

#define TEST(x, s) do {                                                     \
        if (!(x)) {                                                         \
            fprintf(stderr,                                                 \
                "fail: %s\n"                                                \
                "    input: %s\n"                                           \
                "    expected: %.17g\n"                                     \
                "    got: %.17g\n"                                          \
                "    binary xor: 0x%016"PRId64"\n",                         \
                s, buf, expect, v, (a ^ b));                                \
            return 1;                                                       \
        }                                                                   \
    } while (0)

static int test_parse_double(char *buf)
{
    const char *k, *end;
    double v, expect;
    uint64_t a = 0, b = 0;
    int len = strlen(buf);

    end = buf + len;

    expect = strtod(buf, 0);
    /* Include '\0' in bytes being parsed to make strtod safe. */
    k = grisu3_parse_double(buf, len, &v);

    /* Make sure we parsed and accepted everything. */
    TEST(k == end, "didn't parse to end");

    a = grisu3_cast_uint64_from_double(expect);
    b = grisu3_cast_uint64_from_double(v);

#ifdef GRISU3_PARSE_ALLOW_ERROR
    /*
     * Just where exponent wraps, this assumption will be incorrect.
     * TODO: need next higher double function.
     */
    TEST(a - b <= 1, "binary representation differs by more than lsb");
#else
    /* Binary comparison should match. */
    TEST(expect == v, "double representation differs");
    TEST(a == b, "binary representation differs");
#endif

#if 0
    /* This will print the test data also when correct. */
    TEST(0, "test case passed, just debugging");
#endif

    return 0;
}

/*
 * We currently do not test grisu3_print_double because
 * it is a direct port of dtoa_grisu3 from grisu3.c
 * which presumably has been tested in MathGeoLib.
 *
 * grisu3_parse_double is a new implementation.
 */
int test_suite()
{
    char buf[50];
    int fail = 0;

    fail += test_parse_double("1.23434");
    fail += test_parse_double("1234.34");
    fail += test_parse_double("1234.34e4");
    fail += test_parse_double("1234.34e-4");
    fail += test_parse_double("1.23434E+4");
    fail += test_parse_double("3.2897984798741413E+194");
    fail += test_parse_double("-3.2897984798741413E-194");

    sprintf(buf, "3289798479874141.314124124128497098e109");
    fail += test_parse_double(buf);
    sprintf(buf, "3289798479874141.314124124128497098e209");
    fail += test_parse_double(buf);
    sprintf(buf, "-3289798479874141.314124124128497098e209");
    fail += test_parse_double(buf);
    sprintf(buf, "3289798479874141.314124124128497098e+209");
    fail += test_parse_double(buf);
    sprintf(buf, "-3289798479874141.314124124128497098e-209");
    fail += test_parse_double(buf);

    return fail;
}

void example()
{
    double v;
    const char *buf = "1234.34e-4";
    const char *x, *end;
    char result_buf[50];
    int len;

    fprintf(stderr, "grisu3_parse_double example:\n  parsing '%s' as double\n", buf);
    /* A non-numeric terminator (e.g. '\0') is required to ensure strtod fallback is safe. */
    len = strlen(buf);
    end = buf + len;
    x = grisu3_parse_double(buf, len, &v);
    if (x == 0) {
        fprintf(stderr, "syntax or range error\n");
    } else if (x == buf) {
        fprintf(stderr, "parse double failed\n");
    } else if (x != end) {
        fprintf(stderr, "parse double did not read everything\n");
    } else {
        fprintf(stderr, "got: %.17g\n", v);
    }
    /*
     * TODO: with the current example: the input "0.123434" is printed
     * as "1.23434e-1" which is sub-optimal and different from sprintf.
     *
     * This is not the grisu3 algorithm but a post formatting step
     * in grisu3_print_double (originally dtoa_grisu) and may be a bug
     * in the logic choosing the best print format.
     * sprintf "%.17g" and "%g" both print as "0.123434"
     */
    fprintf(stderr, "grisu3_print_double example:\n  printing %g\n", v);
    grisu3_print_double(v, result_buf);
    fprintf(stderr, "got: %s\n", result_buf);
}

int main()
{
    example();
    fprintf(stderr, "running tests\n");
    if (test_suite()) {
        fprintf(stderr, "GRISU3 PARSE TEST FAILED\n");
        return -1;
    } else {
        fprintf(stderr, "GRISU3 PARSE TEST PASSED\n");
        return 0;
    }
}
