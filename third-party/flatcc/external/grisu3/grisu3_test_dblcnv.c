/*
 * Test cases from Googles Double Conversion Library
 *
 * https://github.com/google/double-conversion/blob/master/test/cctest/test-strtod.cc
 *
 * Added extra tests for grisu parse print roundtrip and negative sign.
 */

#include <string.h>
#include <stdio.h>
#include <math.h>

#include "grisu3_print.h"
#include "grisu3_parse.h"

#define BEGIN_TEST(name) int test_ ## name() {                              \
        int fail = 0; char *id = #name; double v; char *vector;             \
        char buf[1001];

#define END_TEST() return fail; }


void check_double(double x1, double x2, char *id, int line, int *fail)
{
    char tmp[50];
    const char *k;
    int n;
    int failed = 0;
    double v;

    if (x1 != x2) {
        failed = 1;
        fprintf(stderr, "%d: fail (%s): %.17g != %.17g\n",
            line, id, x1, x2);
    } else {
#if 1
        n = grisu3_print_double(x1, tmp);
        if (n >= GRISU3_PRINT_MAX) { /* Leave space for zterm. */
            failed = 1;
            fprintf(stderr, "%d: fail (%s): print length exceeded max: %d, input: %.17g\n",
                    line, id, n, x1);
        } else if ((int)strlen(tmp) != n) {
            failed = 1;
            fprintf(stderr, "%d: fail (%s): print length does not match strlen of output, input: %.17g, got: %s\n",
                    line, id, x1, tmp);
        } else if (!isinf(x1)) {
            /* We do expect print/parse to handle inf. */
            k = grisu3_parse_double(tmp, n, &v);
            if (k == 0 || k == tmp) {
                failed = 1;
                fprintf(stderr, "%d: fail (%s): roundtrip parse failed "
                        "input: %g, printed value %s\n",
                        line, id, x1, tmp);
            } else if (x1 != v) {
                failed = 1;
                fprintf(stderr, "%d: fail (%s): print/parse roundtrip mismatch for "
                        "input: %.17g, got %.17g\n",
                        line, id, x1, v);
            }
        }
#endif
    }
    *fail += failed;
}

#define CHECK_EQ(v1, v2) check_double((v1), (v2), id, __LINE__, &fail)

#define StringToVector(f) f

#define Strtod(f, e) (sprintf(buf, "%se%d", f, e),                          \
        grisu3_parse_double(buf, strlen(buf), &v), v)

#define StrtodChar(f, e) (sprintf(buf, "%se%d", f, e),                      \
        grisu3_parse_double(buf, strlen(buf), &v), v)

#define double_infinity grisu3_double_infinity

BEGIN_TEST(Strtod)
  vector = StringToVector("0");
  CHECK_EQ(0.0, Strtod(vector, 1));
  CHECK_EQ(0.0, Strtod(vector, 2));
  CHECK_EQ(0.0, Strtod(vector, -2));
  CHECK_EQ(0.0, Strtod(vector, -999));
  CHECK_EQ(0.0, Strtod(vector, +999));

  vector = StringToVector("1");
  CHECK_EQ(1.0, Strtod(vector, 0));
  CHECK_EQ(10.0, Strtod(vector, 1));
  CHECK_EQ(100.0, Strtod(vector, 2));
  CHECK_EQ(1e20, Strtod(vector, 20));
  CHECK_EQ(1e22, Strtod(vector, 22));
  CHECK_EQ(1e23, Strtod(vector, 23));

  CHECK_EQ(1e35, Strtod(vector, 35));
  CHECK_EQ(1e36, Strtod(vector, 36));
  CHECK_EQ(1e37, Strtod(vector, 37));
  CHECK_EQ(1e-1, Strtod(vector, -1));
  CHECK_EQ(1e-2, Strtod(vector, -2));
  CHECK_EQ(1e-5, Strtod(vector, -5));
  CHECK_EQ(1e-20, Strtod(vector, -20));
  CHECK_EQ(1e-22, Strtod(vector, -22));
  CHECK_EQ(1e-23, Strtod(vector, -23));
  CHECK_EQ(1e-25, Strtod(vector, -25));
  CHECK_EQ(1e-39, Strtod(vector, -39));

  vector = StringToVector("2");
  CHECK_EQ(2.0, Strtod(vector, 0));
  CHECK_EQ(20.0, Strtod(vector, 1));
  CHECK_EQ(200.0, Strtod(vector, 2));
  CHECK_EQ(2e20, Strtod(vector, 20));
  CHECK_EQ(2e22, Strtod(vector, 22));
  CHECK_EQ(2e23, Strtod(vector, 23));
  CHECK_EQ(2e35, Strtod(vector, 35));
  CHECK_EQ(2e36, Strtod(vector, 36));
  CHECK_EQ(2e37, Strtod(vector, 37));
  CHECK_EQ(2e-1, Strtod(vector, -1));
  CHECK_EQ(2e-2, Strtod(vector, -2));
  CHECK_EQ(2e-5, Strtod(vector, -5));
  CHECK_EQ(2e-20, Strtod(vector, -20));
  CHECK_EQ(2e-22, Strtod(vector, -22));
  CHECK_EQ(2e-23, Strtod(vector, -23));
  CHECK_EQ(2e-25, Strtod(vector, -25));
  CHECK_EQ(2e-39, Strtod(vector, -39));

  vector = StringToVector("9");
  CHECK_EQ(9.0, Strtod(vector, 0));
  CHECK_EQ(90.0, Strtod(vector, 1));
  CHECK_EQ(900.0, Strtod(vector, 2));
  CHECK_EQ(9e20, Strtod(vector, 20));
  CHECK_EQ(9e22, Strtod(vector, 22));
  CHECK_EQ(9e23, Strtod(vector, 23));
  CHECK_EQ(9e35, Strtod(vector, 35));
  CHECK_EQ(9e36, Strtod(vector, 36));
  CHECK_EQ(9e37, Strtod(vector, 37));
  CHECK_EQ(9e-1, Strtod(vector, -1));
  CHECK_EQ(9e-2, Strtod(vector, -2));
  CHECK_EQ(9e-5, Strtod(vector, -5));
  CHECK_EQ(9e-20, Strtod(vector, -20));
  CHECK_EQ(9e-22, Strtod(vector, -22));
  CHECK_EQ(9e-23, Strtod(vector, -23));
  CHECK_EQ(9e-25, Strtod(vector, -25));
  CHECK_EQ(9e-39, Strtod(vector, -39));

  vector = StringToVector("12345");
  CHECK_EQ(12345.0, Strtod(vector, 0));
  CHECK_EQ(123450.0, Strtod(vector, 1));
  CHECK_EQ(1234500.0, Strtod(vector, 2));
  CHECK_EQ(12345e20, Strtod(vector, 20));
  CHECK_EQ(12345e22, Strtod(vector, 22));
  CHECK_EQ(12345e23, Strtod(vector, 23));
  CHECK_EQ(12345e30, Strtod(vector, 30));
  CHECK_EQ(12345e31, Strtod(vector, 31));
  CHECK_EQ(12345e32, Strtod(vector, 32));
  CHECK_EQ(12345e35, Strtod(vector, 35));
  CHECK_EQ(12345e36, Strtod(vector, 36));
  CHECK_EQ(12345e37, Strtod(vector, 37));
  CHECK_EQ(12345e-1, Strtod(vector, -1));
  CHECK_EQ(12345e-2, Strtod(vector, -2));
  CHECK_EQ(12345e-5, Strtod(vector, -5));
  CHECK_EQ(12345e-20, Strtod(vector, -20));
  CHECK_EQ(12345e-22, Strtod(vector, -22));
  CHECK_EQ(12345e-23, Strtod(vector, -23));
  CHECK_EQ(12345e-25, Strtod(vector, -25));
  CHECK_EQ(12345e-39, Strtod(vector, -39));

  vector = StringToVector("12345678901234");
  CHECK_EQ(12345678901234.0, Strtod(vector, 0));
  CHECK_EQ(123456789012340.0, Strtod(vector, 1));
  CHECK_EQ(1234567890123400.0, Strtod(vector, 2));
  CHECK_EQ(12345678901234e20, Strtod(vector, 20));
  CHECK_EQ(12345678901234e22, Strtod(vector, 22));
  CHECK_EQ(12345678901234e23, Strtod(vector, 23));
  CHECK_EQ(12345678901234e30, Strtod(vector, 30));
  CHECK_EQ(12345678901234e31, Strtod(vector, 31));
  CHECK_EQ(12345678901234e32, Strtod(vector, 32));
  CHECK_EQ(12345678901234e35, Strtod(vector, 35));
  CHECK_EQ(12345678901234e36, Strtod(vector, 36));
  CHECK_EQ(12345678901234e37, Strtod(vector, 37));
  CHECK_EQ(12345678901234e-1, Strtod(vector, -1));
  CHECK_EQ(12345678901234e-2, Strtod(vector, -2));
  CHECK_EQ(12345678901234e-5, Strtod(vector, -5));
  CHECK_EQ(12345678901234e-20, Strtod(vector, -20));
  CHECK_EQ(12345678901234e-22, Strtod(vector, -22));
  CHECK_EQ(12345678901234e-23, Strtod(vector, -23));
  CHECK_EQ(12345678901234e-25, Strtod(vector, -25));
  CHECK_EQ(12345678901234e-39, Strtod(vector, -39));

  vector = StringToVector("123456789012345");
  CHECK_EQ(123456789012345.0, Strtod(vector, 0));
  CHECK_EQ(1234567890123450.0, Strtod(vector, 1));
  CHECK_EQ(12345678901234500.0, Strtod(vector, 2));
  CHECK_EQ(123456789012345e20, Strtod(vector, 20));
  CHECK_EQ(123456789012345e22, Strtod(vector, 22));
  CHECK_EQ(123456789012345e23, Strtod(vector, 23));
  CHECK_EQ(123456789012345e35, Strtod(vector, 35));
  CHECK_EQ(123456789012345e36, Strtod(vector, 36));
  CHECK_EQ(123456789012345e37, Strtod(vector, 37));
  CHECK_EQ(123456789012345e39, Strtod(vector, 39));
  CHECK_EQ(123456789012345e-1, Strtod(vector, -1));
  CHECK_EQ(123456789012345e-2, Strtod(vector, -2));
  CHECK_EQ(123456789012345e-5, Strtod(vector, -5));
  CHECK_EQ(123456789012345e-20, Strtod(vector, -20));
  CHECK_EQ(123456789012345e-22, Strtod(vector, -22));
  CHECK_EQ(123456789012345e-23, Strtod(vector, -23));
  CHECK_EQ(123456789012345e-25, Strtod(vector, -25));
  CHECK_EQ(123456789012345e-39, Strtod(vector, -39));
  CHECK_EQ(0.0, StrtodChar("0", 12345));

  CHECK_EQ(0.0, StrtodChar("", 1324));
  CHECK_EQ(0.0, StrtodChar("000000000", 123));
  CHECK_EQ(0.0, StrtodChar("2", -324));
  CHECK_EQ(4e-324, StrtodChar("3", -324));

  // It would be more readable to put non-zero literals on the left side (i.e.
  //   CHECK_EQ(1e-325, StrtodChar("1", -325))), but then Gcc complains that
  // they are truncated to zero.
  CHECK_EQ(0.0, StrtodChar("1", -325));
  CHECK_EQ(0.0, StrtodChar("1", -325));
  CHECK_EQ(0.0, StrtodChar("20000", -328));
  CHECK_EQ(40000e-328, StrtodChar("30000", -328));
  CHECK_EQ(0.0, StrtodChar("10000", -329));
  CHECK_EQ(0.0, StrtodChar("90000", -329));
  CHECK_EQ(0.0, StrtodChar("000000001", -325));
  CHECK_EQ(0.0, StrtodChar("000000001", -325));
  CHECK_EQ(0.0, StrtodChar("0000000020000", -328));
  CHECK_EQ(40000e-328, StrtodChar("00000030000", -328));
  CHECK_EQ(0.0, StrtodChar("0000000010000", -329));
  CHECK_EQ(0.0, StrtodChar("0000000090000", -329));


  // It would be more readable to put the literals (and not double_infinity)
  // on the left side (i.e. CHECK_EQ(1e309, StrtodChar("1", 309))), but then Gcc
  // complains that the floating constant exceeds range of 'double'.

  CHECK_EQ(double_infinity, StrtodChar("1", 309));

  CHECK_EQ(1e308, StrtodChar("1", 308));
  CHECK_EQ(1234e305, StrtodChar("1234", 305));
  CHECK_EQ(1234e304, StrtodChar("1234", 304));

  CHECK_EQ(double_infinity, StrtodChar("18", 307));
  CHECK_EQ(17e307, StrtodChar("17", 307));

  CHECK_EQ(double_infinity, StrtodChar("0000001", 309));

  CHECK_EQ(1e308, StrtodChar("00000001", 308));

  CHECK_EQ(1234e305, StrtodChar("00000001234", 305));
  CHECK_EQ(1234e304, StrtodChar("000000001234", 304));
  CHECK_EQ(double_infinity, StrtodChar("0000000018", 307));
  CHECK_EQ(17e307, StrtodChar("0000000017", 307));
  CHECK_EQ(double_infinity, StrtodChar("1000000", 303));
  CHECK_EQ(1e308, StrtodChar("100000", 303));
  CHECK_EQ(1234e305, StrtodChar("123400000", 300));
  CHECK_EQ(1234e304, StrtodChar("123400000", 299));
  CHECK_EQ(double_infinity, StrtodChar("180000000", 300));
  CHECK_EQ(17e307, StrtodChar("170000000", 300));
  CHECK_EQ(double_infinity, StrtodChar("00000001000000", 303));
  CHECK_EQ(1e308, StrtodChar("000000000000100000", 303));
  CHECK_EQ(1234e305, StrtodChar("00000000123400000", 300));
  CHECK_EQ(1234e304, StrtodChar("0000000123400000", 299));
  CHECK_EQ(double_infinity, StrtodChar("00000000180000000", 300));
  CHECK_EQ(17e307, StrtodChar("00000000170000000", 300));
  CHECK_EQ(1.7976931348623157E+308, StrtodChar("17976931348623157", 292));
  CHECK_EQ(1.7976931348623158E+308, StrtodChar("17976931348623158", 292));
  CHECK_EQ(double_infinity, StrtodChar("17976931348623159", 292));

  // The following number is the result of 89255.0/1e-22. Both floating-point
  // numbers can be accurately represented with doubles. However on Linux,x86
  // the floating-point stack is set to 80bits and the double-rounding
  // introduces an error.
  CHECK_EQ(89255e-22, StrtodChar("89255", -22));

  // Some random values.
  CHECK_EQ(358416272e-33, StrtodChar("358416272", -33));
  CHECK_EQ(104110013277974872254e-225,
           StrtodChar("104110013277974872254", -225));

  CHECK_EQ(123456789e108, StrtodChar("123456789", 108));
  CHECK_EQ(123456789e109, StrtodChar("123456789", 109));
  CHECK_EQ(123456789e110, StrtodChar("123456789", 110));
  CHECK_EQ(123456789e111, StrtodChar("123456789", 111));
  CHECK_EQ(123456789e112, StrtodChar("123456789", 112));
  CHECK_EQ(123456789e113, StrtodChar("123456789", 113));
  CHECK_EQ(123456789e114, StrtodChar("123456789", 114));
  CHECK_EQ(123456789e115, StrtodChar("123456789", 115));

  CHECK_EQ(1234567890123456789012345e108,
           StrtodChar("1234567890123456789012345", 108));
  CHECK_EQ(1234567890123456789012345e109,
           StrtodChar("1234567890123456789012345", 109));
  CHECK_EQ(1234567890123456789012345e110,
           StrtodChar("1234567890123456789012345", 110));
  CHECK_EQ(1234567890123456789012345e111,
           StrtodChar("1234567890123456789012345", 111));
  CHECK_EQ(1234567890123456789012345e112,
           StrtodChar("1234567890123456789012345", 112));
  CHECK_EQ(1234567890123456789012345e113,
           StrtodChar("1234567890123456789012345", 113));
  CHECK_EQ(1234567890123456789012345e114,
           StrtodChar("1234567890123456789012345", 114));
  CHECK_EQ(1234567890123456789012345e115,
           StrtodChar("1234567890123456789012345", 115));
  CHECK_EQ(1234567890123456789052345e108,
           StrtodChar("1234567890123456789052345", 108));
  CHECK_EQ(1234567890123456789052345e109,
           StrtodChar("1234567890123456789052345", 109));
  CHECK_EQ(1234567890123456789052345e110,
           StrtodChar("1234567890123456789052345", 110));
  CHECK_EQ(1234567890123456789052345e111,
           StrtodChar("1234567890123456789052345", 111));
  CHECK_EQ(1234567890123456789052345e112,
           StrtodChar("1234567890123456789052345", 112));
  CHECK_EQ(1234567890123456789052345e113,
           StrtodChar("1234567890123456789052345", 113));
  CHECK_EQ(1234567890123456789052345e114,
           StrtodChar("1234567890123456789052345", 114));
  CHECK_EQ(1234567890123456789052345e115,
           StrtodChar("1234567890123456789052345", 115));
  CHECK_EQ(5.445618932859895e-255,
           StrtodChar("5445618932859895362967233318697132813618813095743952975"
                      "4392982234069699615600475529427176366709107287468930197"
                      "8628345413991790019316974825934906752493984055268219809"
                      "5012176093045431437495773903922425632551857520884625114"
                      "6241265881735209066709685420744388526014389929047617597"
                      "0302268848374508109029268898695825171158085457567481507"
                      "4162979705098246243690189880319928315307816832576838178"
                      "2563074014542859888710209237525873301724479666744537857"
                      "9026553346649664045621387124193095870305991178772256504"
                      "4368663670643970181259143319016472430928902201239474588"
                      "1392338901353291306607057623202353588698746085415097902"
                      "6640064319118728664842287477491068264828851624402189317"
                      "2769161449825765517353755844373640588822904791244190695"
                      "2998382932630754670573838138825217065450843010498555058"
                      "88186560731", -1035));

  // Boundary cases. Boundaries themselves should round to even.
  //
  // 0x1FFFFFFFFFFFF * 2^3 = 72057594037927928
  //                   next: 72057594037927936
  //               boundary: 72057594037927932  should round up.
  CHECK_EQ(72057594037927928.0, StrtodChar("72057594037927928", 0));
  CHECK_EQ(72057594037927936.0, StrtodChar("72057594037927936", 0));
  CHECK_EQ(72057594037927936.0, StrtodChar("72057594037927932", 0));
  CHECK_EQ(72057594037927928.0, StrtodChar("7205759403792793199999", -5));
  CHECK_EQ(72057594037927936.0, StrtodChar("7205759403792793200001", -5));

  // 0x1FFFFFFFFFFFF * 2^10 = 9223372036854774784
  //                    next: 9223372036854775808
  //                boundary: 9223372036854775296 should round up.
  CHECK_EQ(9223372036854774784.0, StrtodChar("9223372036854774784", 0));
  CHECK_EQ(9223372036854775808.0, StrtodChar("9223372036854775808", 0));
  CHECK_EQ(9223372036854775808.0, StrtodChar("9223372036854775296", 0));

  CHECK_EQ(9223372036854774784.0, StrtodChar("922337203685477529599999", -5));
  CHECK_EQ(9223372036854775808.0, StrtodChar("922337203685477529600001", -5));

  // 0x1FFFFFFFFFFFF * 2^50 = 10141204801825834086073718800384
  //                    next: 10141204801825835211973625643008
  //                boundary: 10141204801825834649023672221696 should round up.
  //
  CHECK_EQ(10141204801825834086073718800384.0,
           StrtodChar("10141204801825834086073718800384", 0));
  CHECK_EQ(10141204801825835211973625643008.0,
           StrtodChar("10141204801825835211973625643008", 0));
  CHECK_EQ(10141204801825835211973625643008.0,
           StrtodChar("10141204801825834649023672221696", 0));
  CHECK_EQ(10141204801825834086073718800384.0,
           StrtodChar("1014120480182583464902367222169599999", -5));
  CHECK_EQ(10141204801825835211973625643008.0,
           StrtodChar("1014120480182583464902367222169600001", -5));
  // 0x1FFFFFFFFFFFF * 2^99 = 5708990770823838890407843763683279797179383808
  //                    next: 5708990770823839524233143877797980545530986496
  //                boundary: 5708990770823839207320493820740630171355185152
  // The boundary should round up.
  CHECK_EQ(5708990770823838890407843763683279797179383808.0,
           StrtodChar("5708990770823838890407843763683279797179383808", 0));
  CHECK_EQ(5708990770823839524233143877797980545530986496.0,
           StrtodChar("5708990770823839524233143877797980545530986496", 0));
  CHECK_EQ(5708990770823839524233143877797980545530986496.0,
           StrtodChar("5708990770823839207320493820740630171355185152", 0));
  CHECK_EQ(5708990770823838890407843763683279797179383808.0,
           StrtodChar("5708990770823839207320493820740630171355185151999", -3));
  CHECK_EQ(5708990770823839524233143877797980545530986496.0,
           StrtodChar("5708990770823839207320493820740630171355185152001", -3));

  // The following test-cases got some public attention in early 2011 when they
  // sent Java and PHP into an infinite loop.
  CHECK_EQ(2.225073858507201e-308, StrtodChar("22250738585072011", -324));
  CHECK_EQ(2.22507385850720138309e-308,
           StrtodChar("22250738585072011360574097967091319759348195463516456480"
                      "23426109724822222021076945516529523908135087914149158913"
                      "03962110687008643869459464552765720740782062174337998814"
                      "10632673292535522868813721490129811224514518898490572223"
                      "07285255133155755015914397476397983411801999323962548289"
                      "01710708185069063066665599493827577257201576306269066333"
                      "26475653000092458883164330377797918696120494973903778297"
                      "04905051080609940730262937128958950003583799967207254304"
                      "36028407889577179615094551674824347103070260914462157228"
                      "98802581825451803257070188608721131280795122334262883686"
                      "22321503775666622503982534335974568884423900265498198385"
                      "48794829220689472168983109969836584681402285424333066033"
                      "98508864458040010349339704275671864433837704860378616227"
                      "71738545623065874679014086723327636718751", -1076));
END_TEST()


/* Non-google test */
BEGIN_TEST(grisu3_print_double)
  vector = "13";
  CHECK_EQ(13e-2, Strtod(vector, -2));
  CHECK_EQ(13e-3, Strtod(vector, -3));

  vector = "-13";
  CHECK_EQ(-13e-2, Strtod(vector, -2));
  CHECK_EQ(-13e-3, Strtod(vector, -3));
  vector = "-1";
  CHECK_EQ(-1e-2, Strtod(vector, -2));
  CHECK_EQ(-1e-3, Strtod(vector, -3));

  CHECK_EQ(-1e1, StrtodChar("-1", 1));
  CHECK_EQ(-1e+1, StrtodChar("-1", 1));
  CHECK_EQ(-1e-0, StrtodChar("-1", -0));
  CHECK_EQ(-1e-1, StrtodChar("-1", -1));
  CHECK_EQ(-1e-2, StrtodChar("-1", -2));
  CHECK_EQ(-1e-3, StrtodChar("-1", -3));
  CHECK_EQ(-1e-4, StrtodChar("-1", -4));

  CHECK_EQ(-12e1, StrtodChar("-12", 1));
  CHECK_EQ(-12e+1, StrtodChar("-12", 1));
  CHECK_EQ(-12e-0, StrtodChar("-12", -0));
  CHECK_EQ(-12e-1, StrtodChar("-12", -1));
  CHECK_EQ(-12e-2, StrtodChar("-12", -2));
  CHECK_EQ(-12e-3, StrtodChar("-12", -3));
  CHECK_EQ(-12e-4, StrtodChar("-12", -4));

  CHECK_EQ(-123e1, StrtodChar("-123", 1));
  CHECK_EQ(-123e+1, StrtodChar("-123", 1));
  CHECK_EQ(-123e-0, StrtodChar("-123", -0));
  CHECK_EQ(-123e-1, StrtodChar("-123", -1));
  CHECK_EQ(-123e-2, StrtodChar("-123", -2));
  CHECK_EQ(-123e-3, StrtodChar("-123", -3));
  CHECK_EQ(-123e-4, StrtodChar("-123", -4));

  CHECK_EQ(-1234e1, StrtodChar("-1234", 1));
  CHECK_EQ(-1234e+1, StrtodChar("-1234", 1));
  CHECK_EQ(-1234e-0, StrtodChar("-1234", -0));
  CHECK_EQ(-1234e-1, StrtodChar("-1234", -1));
  CHECK_EQ(-1234e-2, StrtodChar("-1234", -2));
  CHECK_EQ(-1234e-3, StrtodChar("-1234", -3));
  CHECK_EQ(-1234e-4, StrtodChar("-1234", -4));

  CHECK_EQ(-12345e1, StrtodChar("-12345", 1));
  CHECK_EQ(-12345e+1, StrtodChar("-12345", 1));
  CHECK_EQ(-12345e-0, StrtodChar("-12345", -0));
  CHECK_EQ(-12345e-1, StrtodChar("-12345", -1));
  CHECK_EQ(-12345e-2, StrtodChar("-12345", -2));
  CHECK_EQ(-12345e-3, StrtodChar("-12345", -3));
  CHECK_EQ(-12345e-4, StrtodChar("-12345", -4));

  CHECK_EQ(-12345e-5, StrtodChar("-12345", -5));
  CHECK_EQ(-12345e-6, StrtodChar("-12345", -6));
  CHECK_EQ(-12345e-7, StrtodChar("-12345", -7));
  CHECK_EQ(-12345e-8, StrtodChar("-12345", -8));
  CHECK_EQ(-12345e-9, StrtodChar("-12345", -9));
  CHECK_EQ(-12345e-10, StrtodChar("-12345", -10));
END_TEST()

int main()
{
    int fail = 0;

    fail += test_Strtod();
    fail += test_grisu3_print_double();

    if (fail) {
        fprintf(stderr, "FAILURE\n");
        return -1;
    }
    fprintf(stderr, "SUCCESS\n");
    return 0;
}
