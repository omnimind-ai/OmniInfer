// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2007 Julien Pommier
// Copyright (C) 2009-2019 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2018-2025 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_ARCH_GENERIC_PACKET_MATH_TRIG_H
#define EIGEN_ARCH_GENERIC_PACKET_MATH_TRIG_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

//----------------------------------------------------------------------
// Trigonometric Functions
//----------------------------------------------------------------------

// Enum for selecting which function to compute. SinCos is intended to compute
// pairs of Sin and Cos of the even entries in the packet, e.g.
// SinCos([a, *, b, *]) = [sin(a), cos(a), sin(b), cos(b)].
enum class TrigFunction : uint8_t { Sin, Cos, Tan, SinCos };

// The following code is inspired by the following stack-overflow answer:
//   https://stackoverflow.com/questions/30463616/payne-hanek-algorithm-implementation-in-c/30465751#30465751
// It has been largely optimized:
//  - By-pass calls to frexp.
//  - Aligned loads of required 96 bits of 2/pi. This is accomplished by
//    (1) balancing the mantissa and exponent to the required bits of 2/pi are
//    aligned on 8-bits, and (2) replicating the storage of the bits of 2/pi.
//  - Avoid a branch in rounding and extraction of the remaining fractional part.
// Overall, I measured a speed up higher than x2 on x86-64.
inline float trig_reduce_huge(float xf, Eigen::numext::int32_t* quadrant) {
  using Eigen::numext::int32_t;
  using Eigen::numext::int64_t;
  using Eigen::numext::uint32_t;
  using Eigen::numext::uint64_t;

  const double pio2_62 = 3.4061215800865545e-19;     // pi/2 * 2^-62
  const uint64_t zero_dot_five = uint64_t(1) << 61;  // 0.5 in 2.62-bit fixed-point format

  // 192 bits of 2/pi for Payne-Hanek reduction
  // Bits are introduced by packet of 8 to enable aligned reads.
  static const uint32_t two_over_pi[] = {
      0x00000028, 0x000028be, 0x0028be60, 0x28be60db, 0xbe60db93, 0x60db9391, 0xdb939105, 0x9391054a, 0x91054a7f,
      0x054a7f09, 0x4a7f09d5, 0x7f09d5f4, 0x09d5f47d, 0xd5f47d4d, 0xf47d4d37, 0x7d4d3770, 0x4d377036, 0x377036d8,
      0x7036d8a5, 0x36d8a566, 0xd8a5664f, 0xa5664f10, 0x664f10e4, 0x4f10e410, 0x10e41000, 0xe4100000};

  uint32_t xi = numext::bit_cast<uint32_t>(xf);
  // Below, -118 = -126 + 8.
  //   -126 is to get the exponent,
  //   +8 is to enable alignment of 2/pi's bits on 8 bits.
  // This is possible because the fractional part of x as only 24 meaningful bits.
  uint32_t e = (xi >> 23) - 118;
  // Extract the mantissa and shift it to align it wrt the exponent
  xi = ((xi & 0x007fffffu) | 0x00800000u) << (e & 0x7);

  uint32_t i = e >> 3;
  uint32_t twoopi_1 = two_over_pi[i - 1];
  uint32_t twoopi_2 = two_over_pi[i + 3];
  uint32_t twoopi_3 = two_over_pi[i + 7];

  // Compute x * 2/pi in 2.62-bit fixed-point format.
  uint64_t p;
  p = uint64_t(xi) * twoopi_3;
  p = uint64_t(xi) * twoopi_2 + (p >> 32);
  p = (uint64_t(xi * twoopi_1) << 32) + p;

  // Round to nearest: add 0.5 and extract integral part.
  uint64_t q = (p + zero_dot_five) >> 62;
  *quadrant = int(q);
  // Now it remains to compute "r = x - q*pi/2" with high accuracy,
  // since we have p=x/(pi/2) with high accuracy, we can more efficiently compute r as:
  //   r = (p-q)*pi/2,
  // where the product can be be carried out with sufficient accuracy using double precision.
  p -= q << 62;
  return float(double(int64_t(p)) * pio2_62);
}

template <TrigFunction Func, typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
#if EIGEN_COMP_GNUC_STRICT
    __attribute__((optimize("-fno-unsafe-math-optimizations")))
#endif
    Packet
    psincos_float(const Packet& _x) {
  typedef typename unpacket_traits<Packet>::integer_packet PacketI;

  const Packet cst_2oPI = pset1<Packet>(0.636619746685028076171875f);  // 2/PI
  const Packet cst_rounding_magic = pset1<Packet>(12582912);           // 2^23 for rounding
  const PacketI csti_1 = pset1<PacketI>(1);
  const Packet cst_sign_mask = pset1frombits<Packet>(static_cast<Eigen::numext::uint32_t>(0x80000000u));

  Packet x = pabs(_x);

  // Scale x by 2/Pi to find x's octant.
  Packet y = pmul(x, cst_2oPI);

  // Rounding trick to find nearest integer:
  Packet y_round = padd(y, cst_rounding_magic);
  EIGEN_OPTIMIZATION_BARRIER(y_round)
  PacketI y_int = preinterpret<PacketI>(y_round);  // last 23 digits represent integer (if abs(x)<2^24)
  y = psub(y_round, cst_rounding_magic);           // nearest integer to x * (2/pi)

// Subtract y * Pi/2 to reduce x to the interval -Pi/4 <= x <= +Pi/4
// using "Extended precision modular arithmetic"
#if defined(EIGEN_VECTORIZE_FMA)
  // This version requires true FMA for high accuracy.
  // It provides a max error of 1ULP up to (with absolute_error < 5.9605e-08):
  constexpr float huge_th = (Func == TrigFunction::Sin) ? 117435.992f : 71476.0625f;
  x = pmadd(y, pset1<Packet>(-1.57079601287841796875f), x);
  x = pmadd(y, pset1<Packet>(-3.1391647326017846353352069854736328125e-07f), x);
  x = pmadd(y, pset1<Packet>(-5.390302529957764765544681040410068817436695098876953125e-15f), x);
#else
  // Without true FMA, the previous set of coefficients maintain 1ULP accuracy
  // up to x<15.7 (for sin), but accuracy is immediately lost for x>15.7.
  // We thus use one more iteration to maintain 2ULPs up to reasonably large inputs.

  // The following set of coefficients maintain 1ULP up to 9.43 and 14.16 for sin and cos respectively.
  // and 2 ULP up to:
  constexpr float huge_th = (Func == TrigFunction::Sin) ? 25966.f : 18838.f;
  x = pmadd(y, pset1<Packet>(-1.5703125), x);  // = 0xbfc90000
  EIGEN_OPTIMIZATION_BARRIER(x)
  x = pmadd(y, pset1<Packet>(-0.000483989715576171875), x);  // = 0xb9fdc000
  EIGEN_OPTIMIZATION_BARRIER(x)
  x = pmadd(y, pset1<Packet>(1.62865035235881805419921875e-07), x);                      // = 0x342ee000
  x = pmadd(y, pset1<Packet>(5.5644315544167710640977020375430583953857421875e-11), x);  // = 0x2e74b9ee

// For the record, the following set of coefficients maintain 2ULP up
// to a slightly larger range:
// const float huge_th = ComputeSine ? 51981.f : 39086.125f;
// but it slightly fails to maintain 1ULP for two values of sin below pi.
// x = pmadd(y, pset1<Packet>(-3.140625/2.), x);
// x = pmadd(y, pset1<Packet>(-0.00048351287841796875), x);
// x = pmadd(y, pset1<Packet>(-3.13855707645416259765625e-07), x);
// x = pmadd(y, pset1<Packet>(-6.0771006282767103812147979624569416046142578125e-11), x);

// For the record, with only 3 iterations it is possible to maintain
// 1 ULP up to 3PI (maybe more) and 2ULP up to 255.
// The coefficients are: 0xbfc90f80, 0xb7354480, 0x2e74b9ee
#endif

  if (predux_any(pcmp_le(pset1<Packet>(huge_th), pabs(_x)))) {
    const int PacketSize = unpacket_traits<Packet>::size;
    EIGEN_ALIGN_TO_BOUNDARY(sizeof(Packet)) float vals[PacketSize];
    EIGEN_ALIGN_TO_BOUNDARY(sizeof(Packet)) float x_cpy[PacketSize];
    EIGEN_ALIGN_TO_BOUNDARY(sizeof(Packet)) Eigen::numext::int32_t y_int2[PacketSize];
    pstoreu(vals, pabs(_x));
    pstoreu(x_cpy, x);
    pstoreu(y_int2, y_int);
    for (int k = 0; k < PacketSize; ++k) {
      float val = vals[k];
      if (val >= huge_th && (numext::isfinite)(val)) x_cpy[k] = trig_reduce_huge(val, &y_int2[k]);
    }
    x = ploadu<Packet>(x_cpy);
    y_int = ploadu<PacketI>(y_int2);
  }

  // Get the polynomial selection mask from the second bit of y_int
  // We'll calculate both (sin and cos) polynomials and then select from the two.
  Packet poly_mask = preinterpret<Packet>(pcmp_eq(pand(y_int, csti_1), pzero(y_int)));

  Packet x2 = pmul(x, x);

  // Evaluate the cos(x) polynomial. (-Pi/4 <= x <= Pi/4)
  Packet y1 = pset1<Packet>(2.4372266125283204019069671630859375e-05f);
  y1 = pmadd(y1, x2, pset1<Packet>(-0.00138865201734006404876708984375f));
  y1 = pmadd(y1, x2, pset1<Packet>(0.041666619479656219482421875f));
  y1 = pmadd(y1, x2, pset1<Packet>(-0.5f));
  y1 = pmadd(y1, x2, pset1<Packet>(1.f));

  // Evaluate the sin(x) polynomial. (Pi/4 <= x <= Pi/4)
  // octave/matlab code to compute those coefficients:
  //    x = (0:0.0001:pi/4)';
  //    A = [x.^3 x.^5 x.^7];
  //    w = ((1.-(x/(pi/4)).^2).^5)*2000+1;         # weights trading relative accuracy
  //    c = (A'*diag(w)*A)\(A'*diag(w)*(sin(x)-x)); # weighted LS, linear coeff forced to 1
  //    printf('%.64f\n %.64f\n%.64f\n', c(3), c(2), c(1))
  //
  Packet y2 = pset1<Packet>(-0.0001959234114083702898469196984621021329076029360294342041015625f);
  y2 = pmadd(y2, x2, pset1<Packet>(0.0083326873655616851693794799871284340042620897293090820312500000f));
  y2 = pmadd(y2, x2, pset1<Packet>(-0.1666666203982298255503735617821803316473960876464843750000000000f));
  y2 = pmul(y2, x2);
  y2 = pmadd(y2, x, x);

  // Select the correct result from the two polynomials.
  // Compute the sign to apply to the polynomial.
  // sin: sign = second_bit(y_int) xor signbit(_x)
  // cos: sign = second_bit(y_int+1)
  Packet sign_bit = (Func == TrigFunction::Sin) ? pxor(_x, preinterpret<Packet>(plogical_shift_left<30>(y_int)))
                                                : preinterpret<Packet>(plogical_shift_left<30>(padd(y_int, csti_1)));
  sign_bit = pand(sign_bit, cst_sign_mask);  // clear all but left most bit

  if ((Func == TrigFunction::SinCos) || (Func == TrigFunction::Tan)) {
    // TODO(rmlarsen): Add single polynomial for tan(x) instead of paying for sin+cos+div.
    Packet peven = peven_mask(x);
    Packet ysin = pselect(poly_mask, y2, y1);
    Packet ycos = pselect(poly_mask, y1, y2);
    Packet sign_bit_sin = pxor(_x, preinterpret<Packet>(plogical_shift_left<30>(y_int)));
    Packet sign_bit_cos = preinterpret<Packet>(plogical_shift_left<30>(padd(y_int, csti_1)));
    sign_bit_sin = pand(sign_bit_sin, cst_sign_mask);  // clear all but left most bit
    sign_bit_cos = pand(sign_bit_cos, cst_sign_mask);  // clear all but left most bit
    y = (Func == TrigFunction::SinCos) ? pselect(peven, pxor(ysin, sign_bit_sin), pxor(ycos, sign_bit_cos))
                                       : pdiv(pxor(ysin, sign_bit_sin), pxor(ycos, sign_bit_cos));
  } else {
    y = (Func == TrigFunction::Sin) ? pselect(poly_mask, y2, y1) : pselect(poly_mask, y1, y2);
    y = pxor(y, sign_bit);
  }
  return y;
}

template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet psin_float(const Packet& x) {
  return psincos_float<TrigFunction::Sin>(x);
}

template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet pcos_float(const Packet& x) {
  return psincos_float<TrigFunction::Cos>(x);
}

template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet ptan_float(const Packet& x) {
  return psincos_float<TrigFunction::Tan>(x);
}

// Trigonometric argument reduction for double for inputs smaller than 15.
// Reduces trigonometric arguments for double inputs where x < 15. Given an argument x and its corresponding quadrant
// count n, the function computes and returns the reduced argument t such that x = n * pi/2 + t.
template <typename Packet>
Packet trig_reduce_small_double(const Packet& x, const Packet& q) {
  // Pi/2 split into 2 values
  const Packet cst_pio2_a = pset1<Packet>(-1.570796325802803);
  const Packet cst_pio2_b = pset1<Packet>(-9.920935184482005e-10);

  Packet t;
  t = pmadd(cst_pio2_a, q, x);
  t = pmadd(cst_pio2_b, q, t);
  return t;
}

// Trigonometric argument reduction for double for inputs smaller than 1e14.
// Reduces trigonometric arguments for double inputs where x < 1e14. Given an argument x and its corresponding quadrant
// count n, the function computes and returns the reduced argument t such that x = n * pi/2 + t.
template <typename Packet>
Packet trig_reduce_medium_double(const Packet& x, const Packet& q_high, const Packet& q_low) {
  // Pi/2 split into 4 values
  const Packet cst_pio2_a = pset1<Packet>(-1.570796325802803);
  const Packet cst_pio2_b = pset1<Packet>(-9.920935184482005e-10);
  const Packet cst_pio2_c = pset1<Packet>(-6.123234014771656e-17);
  const Packet cst_pio2_d = pset1<Packet>(1.903488962019325e-25);

  Packet t;
  t = pmadd(cst_pio2_a, q_high, x);
  t = pmadd(cst_pio2_a, q_low, t);
  t = pmadd(cst_pio2_b, q_high, t);
  t = pmadd(cst_pio2_b, q_low, t);
  t = pmadd(cst_pio2_c, q_high, t);
  t = pmadd(cst_pio2_c, q_low, t);
  t = pmadd(cst_pio2_d, padd(q_low, q_high), t);
  return t;
}

template <TrigFunction Func, typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
#if EIGEN_COMP_GNUC_STRICT
    __attribute__((optimize("-fno-unsafe-math-optimizations")))
#endif
    Packet
    psincos_double(const Packet& x) {
  typedef typename unpacket_traits<Packet>::integer_packet PacketI;
  typedef typename unpacket_traits<PacketI>::type ScalarI;

  const Packet cst_sign_mask = pset1frombits<Packet>(static_cast<Eigen::numext::uint64_t>(0x8000000000000000u));

  // If the argument is smaller than this value, use a simpler argument reduction
  const double small_th = 15;
  // If the argument is bigger than this value, use the non-vectorized std version
  const double huge_th = 1e14;

  const Packet cst_2oPI = pset1<Packet>(0.63661977236758134307553505349006);  // 2/PI
  // Integer Packet constants
  const PacketI cst_one = pset1<PacketI>(ScalarI(1));
  // Constant for splitting
  const Packet cst_split = pset1<Packet>(1 << 24);

  Packet x_abs = pabs(x);

  // Scale x by 2/Pi
  PacketI q_int;
  Packet s;

  // TODO Implement huge angle argument reduction
  if (EIGEN_PREDICT_FALSE(predux_any(pcmp_le(pset1<Packet>(small_th), x_abs)))) {
    Packet q_high = pmul(pfloor(pmul(x_abs, pdiv(cst_2oPI, cst_split))), cst_split);
    Packet q_low_noround = psub(pmul(x_abs, cst_2oPI), q_high);
    q_int = pcast<Packet, PacketI>(padd(q_low_noround, pset1<Packet>(0.5)));
    Packet q_low = pcast<PacketI, Packet>(q_int);
    s = trig_reduce_medium_double(x_abs, q_high, q_low);
  } else {
    Packet qval_noround = pmul(x_abs, cst_2oPI);
    q_int = pcast<Packet, PacketI>(padd(qval_noround, pset1<Packet>(0.5)));
    Packet q = pcast<PacketI, Packet>(q_int);
    s = trig_reduce_small_double(x_abs, q);
  }

  // All the upcoming approximating polynomials have even exponents
  Packet ss = pmul(s, s);

  // Padé approximant of cos(x)
  // Assuring < 1 ULP error on the interval [-pi/4, pi/4]
  // cos(x) ~= (80737373*x^8 - 13853547000*x^6 + 727718024880*x^4 - 11275015752000*x^2 + 23594700729600)/(147173*x^8 +
  // 39328920*x^6 + 5772800880*x^4 + 522334612800*x^2 + 23594700729600)
  // MATLAB code to compute those coefficients:
  //    syms x;
  //    cosf = @(x) cos(x);
  //    pade_cosf = pade(cosf(x), x, 0, 'Order', 8)
  const Packet cn4 = pset1<Packet>(80737373);
  const Packet cn3 = pset1<Packet>(-13853547000);
  const Packet cn2 = pset1<Packet>(727718024880);
  const Packet cn1 = pset1<Packet>(-11275015752000);
  const Packet cn0 = pset1<Packet>(23594700729600);  // shared with cd0
  const Packet cd3 = pset1<Packet>(147173);
  const Packet cd2 = pset1<Packet>(39328920);
  const Packet cd1 = pset1<Packet>(5772800880);
  const Packet cd0 = pset1<Packet>(522334612800);
  Packet sc1_num = pmadd(ss, cn4, cn3);
  Packet sc2_num = pmadd(sc1_num, ss, cn2);
  Packet sc3_num = pmadd(sc2_num, ss, cn1);
  Packet sc4_num = pmadd(sc3_num, ss, cn0);
  Packet sc1_denum = pmadd(ss, cd3, cd2);
  Packet sc2_denum = pmadd(sc1_denum, ss, cd1);
  Packet sc3_denum = pmadd(sc2_denum, ss, cd0);
  Packet sc4_denum = pmadd(sc3_denum, ss, cn0);
  Packet scos = pdiv(sc4_num, sc4_denum);

  // Padé approximant of sin(x)
  // Assuring < 1 ULP error on the interval [-pi/4, pi/4]
  // sin(x) ~= (x*(4585922449*x^8 - 1066023933480*x^6 + 83284044283440*x^4 - 2303682236856000*x^2 +
  // 15605159573203200))/(45*(1029037*x^8 + 345207016*x^6 + 61570292784*x^4 + 6603948711360*x^2 + 346781323848960))
  // MATLAB code to compute those coefficients:
  //    syms x;
  //    sinf = @(x) sin(x);
  //    pade_sinf = pade(sinf(x), x, 0, 'Order', 8, 'OrderMode', 'relative')
  const Packet sn4 = pset1<Packet>(4585922449);
  const Packet sn3 = pset1<Packet>(-1066023933480);
  const Packet sn2 = pset1<Packet>(83284044283440);
  const Packet sn1 = pset1<Packet>(-2303682236856000);
  const Packet sn0 = pset1<Packet>(15605159573203200);
  const Packet sd3 = pset1<Packet>(1029037);
  const Packet sd2 = pset1<Packet>(345207016);
  const Packet sd1 = pset1<Packet>(61570292784);
  const Packet sd0_inner = pset1<Packet>(6603948711360);
  const Packet sd0 = pset1<Packet>(346781323848960);
  const Packet cst_45 = pset1<Packet>(45);
  Packet ss1_num = pmadd(ss, sn4, sn3);
  Packet ss2_num = pmadd(ss1_num, ss, sn2);
  Packet ss3_num = pmadd(ss2_num, ss, sn1);
  Packet ss4_num = pmadd(ss3_num, ss, sn0);
  Packet ss1_denum = pmadd(ss, sd3, sd2);
  Packet ss2_denum = pmadd(ss1_denum, ss, sd1);
  Packet ss3_denum = pmadd(ss2_denum, ss, sd0_inner);
  Packet ss4_denum = pmadd(ss3_denum, ss, sd0);
  Packet ssin = pdiv(pmul(s, ss4_num), pmul(cst_45, ss4_denum));

  Packet poly_mask = preinterpret<Packet>(pcmp_eq(pand(q_int, cst_one), pzero(q_int)));

  Packet sign_sin = pxor(x, preinterpret<Packet>(plogical_shift_left<62>(q_int)));
  Packet sign_cos = preinterpret<Packet>(plogical_shift_left<62>(padd(q_int, cst_one)));
  Packet sign_bit, sFinalRes;
  if (Func == TrigFunction::Sin) {
    sign_bit = sign_sin;
    sFinalRes = pselect(poly_mask, ssin, scos);
  } else if (Func == TrigFunction::Cos) {
    sign_bit = sign_cos;
    sFinalRes = pselect(poly_mask, scos, ssin);
  } else if (Func == TrigFunction::Tan) {
    // TODO(rmlarsen): Add single polynomial for tan(x) instead of paying for sin+cos+div.
    sign_bit = pxor(sign_sin, sign_cos);
    sFinalRes = pdiv(pselect(poly_mask, ssin, scos), pselect(poly_mask, scos, ssin));
  } else if (Func == TrigFunction::SinCos) {
    Packet peven = peven_mask(x);
    sign_bit = pselect(peven, sign_sin, sign_cos);
    sFinalRes = pselect(pxor(peven, poly_mask), scos, ssin);
  }
  sign_bit = pand(sign_bit, cst_sign_mask);  // clear all but left most bit
  sFinalRes = pxor(sFinalRes, sign_bit);

  // If the inputs values are higher than that a value that the argument reduction can currently address, compute them
  // using the C++ standard library.
  // TODO Remove it when huge angle argument reduction is implemented
  if (EIGEN_PREDICT_FALSE(predux_any(pcmp_le(pset1<Packet>(huge_th), x_abs)))) {
    const int PacketSize = unpacket_traits<Packet>::size;
    EIGEN_ALIGN_TO_BOUNDARY(sizeof(Packet)) double sincos_vals[PacketSize];
    EIGEN_ALIGN_TO_BOUNDARY(sizeof(Packet)) double x_cpy[PacketSize];
    pstoreu(x_cpy, x);
    pstoreu(sincos_vals, sFinalRes);
    for (int k = 0; k < PacketSize; ++k) {
      double val = x_cpy[k];
      if (std::abs(val) > huge_th && (numext::isfinite)(val)) {
        if (Func == TrigFunction::Sin) {
          sincos_vals[k] = std::sin(val);
        } else if (Func == TrigFunction::Cos) {
          sincos_vals[k] = std::cos(val);
        } else if (Func == TrigFunction::Tan) {
          sincos_vals[k] = std::tan(val);
        } else if (Func == TrigFunction::SinCos) {
          sincos_vals[k] = k % 2 == 0 ? std::sin(val) : std::cos(val);
        }
      }
    }
    sFinalRes = ploadu<Packet>(sincos_vals);
  }
  return sFinalRes;
}

template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet psin_double(const Packet& x) {
  return psincos_double<TrigFunction::Sin>(x);
}

template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet pcos_double(const Packet& x) {
  return psincos_double<TrigFunction::Cos>(x);
}

template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet ptan_double(const Packet& x) {
  return psincos_double<TrigFunction::Tan>(x);
}

template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
    std::enable_if_t<std::is_same<typename unpacket_traits<Packet>::type, float>::value, Packet>
    psincos_selector(const Packet& x) {
  return psincos_float<TrigFunction::SinCos, Packet>(x);
}

template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
    std::enable_if_t<std::is_same<typename unpacket_traits<Packet>::type, double>::value, Packet>
    psincos_selector(const Packet& x) {
  return psincos_double<TrigFunction::SinCos, Packet>(x);
}

//----------------------------------------------------------------------
// Inverse Trigonometric Functions
//----------------------------------------------------------------------

// Generic implementation of acos(x).
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet pacos_float(const Packet& x_in) {
  typedef typename unpacket_traits<Packet>::type Scalar;
  static_assert(std::is_same<Scalar, float>::value, "Scalar type must be float");

  const Packet cst_one = pset1<Packet>(Scalar(1));
  const Packet cst_pi = pset1<Packet>(Scalar(EIGEN_PI));
  const Packet p6 = pset1<Packet>(Scalar(2.36423197202384471893310546875e-3));
  const Packet p5 = pset1<Packet>(Scalar(-1.1368644423782825469970703125e-2));
  const Packet p4 = pset1<Packet>(Scalar(2.717843465507030487060546875e-2));
  const Packet p3 = pset1<Packet>(Scalar(-4.8969544470310211181640625e-2));
  const Packet p2 = pset1<Packet>(Scalar(8.8804088532924652099609375e-2));
  const Packet p1 = pset1<Packet>(Scalar(-0.214591205120086669921875));
  const Packet p0 = pset1<Packet>(Scalar(1.57079637050628662109375));

  // For x in [0:1], we approximate acos(x)/sqrt(1-x), which is a smooth
  // function, by a 6'th order polynomial.
  // For x in [-1:0) we use that acos(-x) = pi - acos(x).
  const Packet neg_mask = psignbit(x_in);
  const Packet abs_x = pabs(x_in);

  // Evaluate the polynomial using Horner's rule:
  //   P(x) = p0 + x * (p1 +  x * (p2 + ... (p5 + x * p6)) ... ) .
  // We evaluate even and odd terms independently to increase
  // instruction level parallelism.
  Packet x2 = pmul(x_in, x_in);
  Packet p_even = pmadd(p6, x2, p4);
  Packet p_odd = pmadd(p5, x2, p3);
  p_even = pmadd(p_even, x2, p2);
  p_odd = pmadd(p_odd, x2, p1);
  p_even = pmadd(p_even, x2, p0);
  Packet p = pmadd(p_odd, abs_x, p_even);

  // The polynomial approximates acos(x)/sqrt(1-x), so
  // multiply by sqrt(1-x) to get acos(x).
  // Conveniently returns NaN for arguments outside [-1:1].
  Packet denom = psqrt(psub(cst_one, abs_x));
  Packet result = pmul(denom, p);
  // Undo mapping for negative arguments.
  return pselect(neg_mask, psub(cst_pi, result), result);
}

// Generic implementation of asin(x).
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet pasin_float(const Packet& x_in) {
  typedef typename unpacket_traits<Packet>::type Scalar;
  static_assert(std::is_same<Scalar, float>::value, "Scalar type must be float");

  constexpr float kPiOverTwo = static_cast<float>(EIGEN_PI / 2);

  const Packet cst_half = pset1<Packet>(0.5f);
  const Packet cst_one = pset1<Packet>(1.0f);
  const Packet cst_two = pset1<Packet>(2.0f);
  const Packet cst_pi_over_two = pset1<Packet>(kPiOverTwo);

  const Packet abs_x = pabs(x_in);
  const Packet sign_mask = pandnot(x_in, abs_x);
  const Packet invalid_mask = pcmp_lt(cst_one, abs_x);

  // For arguments |x| > 0.5, we map x back to [0:0.5] using
  // the transformation x_large = sqrt(0.5*(1-x)), and use the
  // identity
  //   asin(x) = pi/2 - 2 * asin( sqrt( 0.5 * (1 - x)))

  const Packet x_large = psqrt(pnmadd(cst_half, abs_x, cst_half));
  const Packet large_mask = pcmp_lt(cst_half, abs_x);
  const Packet x = pselect(large_mask, x_large, abs_x);
  const Packet x2 = pmul(x, x);

  // For |x| < 0.5 approximate asin(x)/x by an 8th order polynomial with
  // even terms only.
  constexpr float alpha[] = {5.08838854730129241943359375e-2f, 3.95139865577220916748046875e-2f,
                             7.550220191478729248046875e-2f, 0.16664917767047882080078125f, 1.00000011920928955078125f};
  Packet p = ppolevl<Packet, 4>::run(x2, alpha);
  p = pmul(p, x);

  const Packet p_large = pnmadd(cst_two, p, cst_pi_over_two);
  p = pselect(large_mask, p_large, p);
  // Flip the sign for negative arguments.
  p = pxor(p, sign_mask);
  // Return NaN for arguments outside [-1:1].
  return por(invalid_mask, p);
}

template <typename Scalar>
struct patan_reduced {
  template <typename Packet>
  static EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet run(const Packet& x);
};

template <>
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet patan_reduced<double>::run(const Packet& x) {
  constexpr double alpha[] = {2.6667153866462208e-05, 3.0917513112462781e-03, 5.2574296781008604e-02,
                              3.0409318473444424e-01, 7.5365702534987022e-01, 8.2704055405494614e-01,
                              3.3004361289279920e-01};

  constexpr double beta[] = {
      2.7311202462436667e-04, 1.0899150928962708e-02, 1.1548932646420353e-01, 4.9716458728465573e-01, 1.0,
      9.3705509168587852e-01, 3.3004361289279920e-01};

  Packet x2 = pmul(x, x);
  Packet p = ppolevl<Packet, 6>::run(x2, alpha);
  Packet q = ppolevl<Packet, 6>::run(x2, beta);
  return pmul(x, pdiv(p, q));
}

// Computes elementwise atan(x) for x in [-1:1] with 2 ulp accuracy.
template <>
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet patan_reduced<float>::run(const Packet& x) {
  constexpr float alpha[] = {1.12026982009410858154296875e-01f, 7.296695709228515625e-01f, 8.109951019287109375e-01f};

  constexpr float beta[] = {1.00917108356952667236328125e-02f, 2.8318560123443603515625e-01f, 1.0f,
                            8.109951019287109375e-01f};

  Packet x2 = pmul(x, x);
  Packet p = ppolevl<Packet, 2>::run(x2, alpha);
  Packet q = ppolevl<Packet, 3>::run(x2, beta);
  return pmul(x, pdiv(p, q));
}

template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet generic_atan(const Packet& x_in) {
  typedef typename unpacket_traits<Packet>::type Scalar;

  constexpr Scalar kPiOverTwo = static_cast<Scalar>(EIGEN_PI / 2);

  const Packet cst_signmask = pset1<Packet>(Scalar(-0.0));
  const Packet cst_one = pset1<Packet>(Scalar(1));
  const Packet cst_pi_over_two = pset1<Packet>(kPiOverTwo);

  //   "Large": For |x| > 1, use atan(1/x) = sign(x)*pi/2 - atan(x).
  //   "Small": For |x| <= 1, approximate atan(x) directly by a polynomial
  //            calculated using Rminimax.

  const Packet abs_x = pabs(x_in);
  const Packet x_signmask = pand(x_in, cst_signmask);
  const Packet large_mask = pcmp_lt(cst_one, abs_x);
  const Packet x = pselect(large_mask, preciprocal(abs_x), abs_x);
  const Packet p = patan_reduced<Scalar>::run(x);
  // Apply transformations according to the range reduction masks.
  Packet result = pselect(large_mask, psub(cst_pi_over_two, p), p);
  // Return correct sign
  return pxor(result, x_signmask);
}

//----------------------------------------------------------------------
// Hyperbolic Functions
//----------------------------------------------------------------------

#ifdef EIGEN_FAST_MATH

/** \internal \returns the hyperbolic tan of \a a (coeff-wise)
    Doesn't do anything fancy, just a 9/8-degree rational interpolant which
    is accurate up to a couple of ulps in the (approximate) range [-8, 8],
    outside of which tanh(x) = +/-1 in single precision. The input is clamped
    to the range [-c, c]. The value c is chosen as the smallest value where
    the approximation evaluates to exactly 1.

    This implementation works on both scalars and packets.
*/
template <typename T>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS T ptanh_float(const T& a_x) {
  // Clamp the inputs to the range [-c, c] and set everything
  // outside that range to 1.0. The value c is chosen as the smallest
  // floating point argument such that the approximation is exactly 1.
  // This saves clamping the value at the end.
#ifdef EIGEN_VECTORIZE_FMA
  const T plus_clamp = pset1<T>(8.01773357391357422f);
  const T minus_clamp = pset1<T>(-8.01773357391357422f);
#else
  const T plus_clamp = pset1<T>(7.90738964080810547f);
  const T minus_clamp = pset1<T>(-7.90738964080810547f);
#endif
  const T x = pmax(pmin(a_x, plus_clamp), minus_clamp);

  // The following rational approximation was generated by rminimax
  // (https://gitlab.inria.fr/sfilip/rminimax) using the following
  // command:
  // $ ratapprox --function="tanh(x)" --dom='[-8.67,8.67]' --num="odd"
  //   --den="even" --type="[9,8]" --numF="[SG]" --denF="[SG]" --log
  //   --output=tanhf.sollya --dispCoeff="dec"

  // The monomial coefficients of the numerator polynomial (odd).
  constexpr float alpha[] = {1.394553628e-8f, 2.102733560e-5f, 3.520756727e-3f, 1.340216100e-1f};

  // The monomial coefficients of the denominator polynomial (even).
  constexpr float beta[] = {8.015776984e-7f, 3.326951409e-4f, 2.597254514e-2f, 4.673548340e-1f, 1.0f};

  // Since the polynomials are odd/even, we need x^2.
  const T x2 = pmul(x, x);
  const T x3 = pmul(x2, x);

  T p = ppolevl<T, 3>::run(x2, alpha);
  T q = ppolevl<T, 4>::run(x2, beta);
  // Take advantage of the fact that the constant term in p is 1 to compute
  // x*(x^2*p + 1) = x^3 * p + x.
  p = pmadd(x3, p, x);

  // Divide the numerator by the denominator.
  return pdiv(p, q);
}

#else

/** \internal \returns the hyperbolic tan of \a a (coeff-wise).
    On the domain [-1.25:1.25] we use an approximation of the form
    tanh(x) ~= x^3 * (P(x) / Q(x)) + x, where P and Q are polynomials in x^2.
    For |x| > 1.25, tanh is implemented as tanh(x) = 1 - (2 / (1 + exp(2*x))).

    This implementation has a maximum error of 1 ULP (measured with AVX2+FMA).

    This implementation works on both scalars and packets.
*/
template <typename T>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS T ptanh_float(const T& x) {
  // The polynomial coefficients were computed using Rminimax:
  // % ./ratapprox --function="tanh(x)-x" --dom='[-1.25,1.25]' --num="[x^3,x^5]" --den="even"
  //     --type="[3,4]" --numF="[SG]" --denF="[SG]" --log --dispCoeff="dec" --output=tanhf.solly
  constexpr float alpha[] = {-1.46725140511989593505859375e-02f, -3.333333432674407958984375e-01f};
  constexpr float beta[] = {1.570280082523822784423828125e-02, 4.4401752948760986328125e-01, 1.0f};
  const T x2 = pmul(x, x);
  const T x3 = pmul(x2, x);
  const T p = ppolevl<T, 1>::run(x2, alpha);
  const T q = ppolevl<T, 2>::run(x2, beta);
  const T small_tanh = pmadd(x3, pdiv(p, q), x);

  const T sign_mask = pset1<T>(-0.0f);
  const T abs_x = pandnot(x, sign_mask);
  constexpr float kSmallThreshold = 1.25f;
  const T large_mask = pcmp_lt(pset1<T>(kSmallThreshold), abs_x);
  // Fast exit if all elements are small.
  if (!predux_any(large_mask)) {
    return small_tanh;
  }

  //  Compute as 1 - (2 / (1 + exp(2*x)))
  const T one = pset1<T>(1.0f);
  const T two = pset1<T>(2.0f);
  const T s = pexp_float<T, true>(pmul(two, abs_x));
  const T abs_tanh = psub(one, pdiv(two, padd(s, one)));

  // Handle infinite inputs and set sign bit.
  constexpr float kHugeThreshold = 16.0f;
  const T huge_mask = pcmp_lt(pset1<T>(kHugeThreshold), abs_x);
  const T x_sign = pand(sign_mask, x);
  const T large_tanh = por(x_sign, pselect(huge_mask, one, abs_tanh));
  return pselect(large_mask, large_tanh, small_tanh);
}

#endif  // EIGEN_FAST_MATH

/** \internal \returns the hyperbolic tan of \a a (coeff-wise)
    This uses a 19/18-degree rational interpolant which
    is accurate up to a couple of ulps in the (approximate) range [-18.7, 18.7],
    outside of which tanh(x) = +/-1 in single precision. The input is clamped
    to the range [-c, c]. The value c is chosen as the smallest value where
    the approximation evaluates to exactly 1.

    This implementation works on both scalars and packets.
*/
template <typename T>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS T ptanh_double(const T& a_x) {
  // Clamp the inputs to the range [-c, c] and set everything
  // outside that range to 1.0. The value c is chosen as the smallest
  // floating point argument such that the approximation is exactly 1.
  // This saves clamping the value at the end.
#ifdef EIGEN_VECTORIZE_FMA
  const T plus_clamp = pset1<T>(17.6610191624600077);
  const T minus_clamp = pset1<T>(-17.6610191624600077);
#else
  const T plus_clamp = pset1<T>(17.714196154005176);
  const T minus_clamp = pset1<T>(-17.714196154005176);
#endif
  const T x = pmax(pmin(a_x, plus_clamp), minus_clamp);
  // The following rational approximation was generated by rminimax
  // (https://gitlab.inria.fr/sfilip/rminimax) using the following
  // command:
  // $ ./ratapprox --function="tanh(x)" --dom='[-18.72,18.72]'
  //   --num="odd" --den="even" --type="[19,18]" --numF="[D]"
  //   --denF="[D]" --log --output=tanh.sollya --dispCoeff="dec"

  // The monomial coefficients of the numerator polynomial (odd).
  constexpr double alpha[] = {2.6158007860482230e-23, 7.6534862268749319e-19, 3.1309488231386680e-15,
                              4.2303918148209176e-12, 2.4618379131293676e-09, 6.8644367682497074e-07,
                              9.3839087674268880e-05, 5.9809711724441161e-03, 1.5184719640284322e-01};

  // The monomial coefficients of the denominator polynomial (even).
  constexpr double beta[] = {6.463747022670968018e-21, 5.782506856739003571e-17,
                             1.293019623712687916e-13, 1.123643448069621992e-10,
                             4.492975677839633985e-08, 8.785185266237658698e-06,
                             8.295161192716231542e-04, 3.437448108450402717e-02,
                             4.851805297361760360e-01, 1.0};

  // Since the polynomials are odd/even, we need x^2.
  const T x2 = pmul(x, x);
  const T x3 = pmul(x2, x);

  // Interleave the evaluation of the numerator polynomial p and
  // denominator polynomial q.
  T p = ppolevl<T, 8>::run(x2, alpha);
  T q = ppolevl<T, 9>::run(x2, beta);
  // Take advantage of the fact that the constant term in p is 1 to compute
  // x*(x^2*p + 1) = x^3 * p + x.
  p = pmadd(x3, p, x);

  // Divide the numerator by the denominator.
  return pdiv(p, q);
}

template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet patanh_float(const Packet& x) {
  typedef typename unpacket_traits<Packet>::type Scalar;
  static_assert(std::is_same<Scalar, float>::value, "Scalar type must be float");

  // For |x| in [0:0.5] we use a polynomial approximation of the form
  // P(x) = x + x^3*(alpha[4] + x^2 * (alpha[3] + x^2 * (... x^2 * alpha[0]) ... )).
  constexpr float alpha[] = {0.1819281280040740966796875f, 8.2311116158962249755859375e-2f,
                             0.14672131836414337158203125f, 0.1997792422771453857421875f, 0.3333373963832855224609375f};
  const Packet x2 = pmul(x, x);
  const Packet x3 = pmul(x, x2);
  Packet p = ppolevl<Packet, 4>::run(x2, alpha);
  p = pmadd(x3, p, x);

  // For |x| in ]0.5:1.0] we use atanh = 0.5*ln((1+x)/(1-x));
  const Packet half = pset1<Packet>(0.5f);
  const Packet one = pset1<Packet>(1.0f);
  Packet r = pdiv(padd(one, x), psub(one, x));
  r = pmul(half, plog(r));

  const Packet x_gt_half = pcmp_le(half, pabs(x));
  const Packet x_eq_one = pcmp_eq(one, pabs(x));
  const Packet x_gt_one = pcmp_lt(one, pabs(x));
  const Packet sign_mask = pset1<Packet>(-0.0f);
  const Packet x_sign = pand(sign_mask, x);
  const Packet inf = pset1<Packet>(std::numeric_limits<float>::infinity());
  return por(x_gt_one, pselect(x_eq_one, por(x_sign, inf), pselect(x_gt_half, r, p)));
}

template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet patanh_double(const Packet& x) {
  typedef typename unpacket_traits<Packet>::type Scalar;
  static_assert(std::is_same<Scalar, double>::value, "Scalar type must be double");
  // For x in [-0.5:0.5] we use a rational approximation of the form
  // R(x) = x + x^3*P(x^2)/Q(x^2), where P is or order 4 and Q is of order 5.
  constexpr double alpha[] = {3.3071338469301391e-03, -4.7129526768798737e-02, 1.8185306179826699e-01,
                              -2.5949536095445679e-01, 1.2306328729812676e-01};

  constexpr double beta[] = {-3.8679974580640881e-03, 7.6391885763341910e-02,  -4.2828141436397615e-01,
                             9.8733495886883648e-01,  -1.0000000000000000e+00, 3.6918986189438030e-01};

  const Packet x2 = pmul(x, x);
  const Packet x3 = pmul(x, x2);
  Packet p = ppolevl<Packet, 4>::run(x2, alpha);
  Packet q = ppolevl<Packet, 5>::run(x2, beta);
  Packet y_small = pmadd(x3, pdiv(p, q), x);

  // For |x| in ]0.5:1.0] we use atanh = 0.5*ln((1+x)/(1-x));
  const Packet half = pset1<Packet>(0.5);
  const Packet one = pset1<Packet>(1.0);
  Packet y_large = pdiv(padd(one, x), psub(one, x));
  y_large = pmul(half, plog(y_large));

  const Packet x_gt_half = pcmp_le(half, pabs(x));
  const Packet x_eq_one = pcmp_eq(one, pabs(x));
  const Packet x_gt_one = pcmp_lt(one, pabs(x));
  const Packet sign_mask = pset1<Packet>(-0.0);
  const Packet x_sign = pand(sign_mask, x);
  const Packet inf = pset1<Packet>(std::numeric_limits<double>::infinity());
  return por(x_gt_one, pselect(x_eq_one, por(x_sign, inf), pselect(x_gt_half, y_large, y_small)));
}

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_ARCH_GENERIC_PACKET_MATH_TRIG_H
