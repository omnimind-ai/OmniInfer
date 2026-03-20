/**=============================================================================
@file
    hvx_internal.h

@brief
    Header file for HVX routines.

Copyright (c) 2020-2021 Qualcomm Technologies Incorporated.
All Rights Reserved. Qualcomm Proprietary and Confidential.
=============================================================================**/

#ifndef _HVX_INTERNAL_H
#define _HVX_INTERNAL_H

#include <stdint.h>
#include <stddef.h>
#include <hexagon_types.h>

#define vmem(A)     *((HVX_Vector *)(A))
#define vmemu(A)    *((HVX_UVector *)(A))

#define HVX_INLINE_ALWAYS inline __attribute__((unused,always_inline))

#ifndef LOG2VLEN
#define LOG2VLEN    7
#endif
#define VLEN        (1<<LOG2VLEN)    // HVX vector - number of int8_t elements
#define VLEN_SHORT  (1<<LOG2VLEN)>>1 // HVX vector - number of int16_t elements
#define VLEN_WORD   (1<<LOG2VLEN)>>2 // HVX vector - number of int32_t elements

#define IEEE_VSF_EXPLEN         (8)
#define IEEE_VSF_EXPBIAS        (127)
#define IEEE_VSF_EXPMASK        (0xFF)
#define IEEE_VSF_MANTLEN        (23)
#define IEEE_VSF_MANTMASK       (0x7FFFFF)
#define IEEE_VSF_MIMPMASK       (0x800000)

#define IEEE_VHF_EXPLEN         (5)
#define IEEE_VHF_EXPBIAS        (15)
#define IEEE_VHF_EXPMASK        (0x1F)
#define IEEE_VHF_MANTLEN        (10)
#define IEEE_VHF_MANTMASK       (0x3FF)
#define IEEE_VHF_MIMPMASK       (0x400)

typedef union
{
    HVX_VectorPair VV;
    struct
    {
        HVX_Vector lo;
        HVX_Vector hi;
    } V;
} HVX_DV;

typedef union {
    uint32_t    uint32_array[32];
    uint32_t    qf32_array[32];
    float       float32_array[32];
    HVX_Vector  vector;
} qhl_hvx_vector_array;

static HVX_INLINE_ALWAYS void l2fetch(const void *p, uint32_t stride,
                                      uint32_t width, uint32_t height,
                                      uint32_t dir)
{
    uint64_t control = HEXAGON_V64_CREATE_H(dir, stride, width, height);
    __asm__ __volatile__ (" l2fetch(%0,%1) " : :"r"(p),"r"(control));
}

// Return whether address is aligned.

static HVX_INLINE_ALWAYS int32_t is_aligned(const void *addr, uint32_t align)
{
    return ((size_t) addr & (align - 1)) == 0;
}

// Return whether 'n' elements from vector are in the one chunk of 'chunk_size'.

static HVX_INLINE_ALWAYS int32_t is_in_one_chunk(void *addr, uint32_t n,
                                                 uint32_t chunk_size)
{
    uint32_t left_off = (size_t) addr & (chunk_size - 1);
    uint32_t right_off = left_off + n;
    return right_off <= chunk_size;
}

// This function stores the first n bytes from vector vin to address 'addr'.
// n must be in range 1..128 and addr may have any alignment. Does one or
// two masked stores.

static HVX_INLINE_ALWAYS void vstu_variable(void *addr, uint32_t n,
                                            HVX_Vector vin)
{
    // Rotate as needed.
    vin = Q6_V_vlalign_VVR(vin, vin, (size_t) addr);

    uint32_t left_off = (size_t) addr & 127;
    uint32_t right_off = left_off + n;

    HVX_VectorPred ql_not = Q6_Q_vsetq_R((size_t) addr);
    HVX_VectorPred qr = Q6_Q_vsetq2_R(right_off);

    if (right_off > 128)
    {
        Q6_vmem_QRIV(qr, (HVX_Vector*) addr + 1, vin);
        // all 1's
        qr = Q6_Q_vcmp_eq_VbVb(vin, vin);
    }

    ql_not = Q6_Q_or_QQn(ql_not, qr);
    Q6_vmem_QnRIV(ql_not, (HVX_Vector*) addr, vin);
}

// This function stores the first n bytes from
// vector pair vin to address 'addr'.

static HVX_INLINE_ALWAYS void vstdu_variable(void *addr, uint32_t n,
                                             HVX_VectorPair vin)
{
    vstu_variable(addr, n > 128 ? 128 : n, Q6_V_lo_W(vin));

    if (n > 128)
    {
        vstu_variable((HVX_Vector*) addr + 1, n - 128, Q6_V_hi_W(vin));
    }
}

// 32x32 fractional multiply - expands to two ops
//  equiv to :
//    p  = (a*b + 0x40000000) >> 31     [with rounding]
//    p  = a*b >> 31                    [without rounding]
// The 'sat' only takes effect when both inputs
// are -0x80000000 and causes the result to saturate to 0x7fffffff

static HVX_INLINE_ALWAYS HVX_Vector Q6_Vw_vmpy_VwVw_s1_rnd_sat(HVX_Vector vu,
                                                               HVX_Vector vv)
{
    return Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(Q6_Vw_vmpye_VwVuh(vu, vv),
                                                  vu, vv);
}

static HVX_INLINE_ALWAYS HVX_Vector Q6_Vw_vmpy_VwVw_s1_sat(HVX_Vector vu,
                                                           HVX_Vector vv)
{
    return Q6_Vw_vmpyoacc_VwVwVh_s1_sat_shift(Q6_Vw_vmpye_VwVuh(vu, vv),
                                              vu, vv);
}

static HVX_INLINE_ALWAYS HVX_VectorPair Q6_W_vmpy_VwVw(HVX_Vector vu,
                                                       HVX_Vector vv)
{
    return Q6_W_vmpyoacc_WVwVh(Q6_W_vmpye_VwVuh(vu, vv), vu, vv);
}

static HVX_INLINE_ALWAYS uint16_t fp16_to_bits(__fp16 *x)
{
    union { __fp16 f; uint16_t i; } fp16 = { .f = *x };
    return fp16.i;
}

//  truncate(x)
//  given a vector of float x,
//  return the vector of integers resulting from dropping all fractional bits
//  no checking performed for overflow - could be extended to return maxint
//
// truncate float to int
static HVX_INLINE_ALWAYS HVX_Vector qhmath_hvx_vw_truncate_vsf(HVX_Vector vin) {
    HVX_Vector mask_mant_v  = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_Vector mask_impl_v  = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_Vector const_zero_v = Q6_V_vzero();

    HVX_VectorPred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, vin);

    HVX_Vector expval_v = vin >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    // negative exp == fractional value
    HVX_VectorPred q_negexp  = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);

    HVX_Vector rshift_v = IEEE_VSF_MANTLEN - expval_v;  // fractional bits - exp shift

    HVX_Vector mant_v = vin & mask_mant_v;                  // obtain mantissa
    HVX_Vector vout = Q6_Vw_vadd_VwVw(mant_v, mask_impl_v); // add implicit 1.0
    vout = Q6_Vw_vasr_VwVw(vout, rshift_v);  // shift to obtain truncated integer
    vout = Q6_V_vmux_QVV(q_negexp, const_zero_v, vout);     // expval<0 -> 0

    HVX_Vector neg_vout = -vout;
    vout = Q6_V_vmux_QVV(q_negative, neg_vout, vout);     // handle negatives
    return(vout);
}

// truncate float to float
static HVX_INLINE_ALWAYS HVX_Vector qhmath_hvx_vsf_truncate_vsf(HVX_Vector vin) {
    HVX_Vector mask_mant_v   = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_Vector const_mnlen_v = Q6_V_vsplat_R(IEEE_VSF_MANTLEN);
    HVX_Vector const_zero_v  = Q6_V_vzero();

    HVX_Vector expval_v = vin >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    HVX_VectorPred q_negexp  = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);
    HVX_VectorPred q_expltmn = Q6_Q_vcmp_gt_VwVw(const_mnlen_v, expval_v);

    mask_mant_v >>= expval_v;
    HVX_Vector not_mask_v = Q6_V_vnot_V(mask_mant_v);        // frac to clear
    HVX_Vector vout_clrfrac = Q6_V_vand_VV(vin, not_mask_v); // clear frac

    HVX_Vector vout = vin;                               // integral no change
    vout = Q6_V_vmux_QVV(q_expltmn, vout_clrfrac, vout); // expval<mant -> clr frac
    vout = Q6_V_vmux_QVV(q_negexp, const_zero_v, vout);  // expval<0  -> 0

    return(vout);
}

// truncate half float to short
static HVX_INLINE_ALWAYS HVX_Vector qhmath_hvx_vh_truncate_vhf(HVX_Vector vin) {
    HVX_Vector const_mnlen_v = Q6_Vh_vsplat_R(IEEE_VHF_MANTLEN);
    HVX_Vector mask_mant_v   = Q6_Vh_vsplat_R(IEEE_VHF_MANTMASK);
    HVX_Vector mask_impl_v   = Q6_Vh_vsplat_R(IEEE_VHF_MIMPMASK);
    HVX_Vector const_emask_v = Q6_Vh_vsplat_R(IEEE_VHF_EXPMASK);
    HVX_Vector const_ebias_v = Q6_Vh_vsplat_R(IEEE_VHF_EXPBIAS);
    HVX_Vector const_zero_v  = Q6_V_vzero();
    HVX_Vector const_one_v   = Q6_Vh_vsplat_R(1);

    HVX_VectorPred q_negative = Q6_Q_vcmp_gt_VhVh(const_zero_v, vin);

    HVX_Vector expval_v = Q6_Vh_vasr_VhVh(vin, const_mnlen_v);
    expval_v = Q6_V_vand_VV(expval_v, const_emask_v);
    expval_v = Q6_Vh_vsub_VhVh(expval_v, const_ebias_v);

    // negative exp == fractional value
    HVX_VectorPred q_negexp = Q6_Q_vcmp_gt_VhVh(const_zero_v, expval_v);

    // fractional bits - exp shift
    HVX_Vector rshift_v = Q6_Vh_vsub_VhVh(const_mnlen_v, expval_v);

    HVX_Vector mant_v = vin & mask_mant_v;                  // obtain mantissa
    HVX_Vector vout = Q6_Vh_vadd_VhVh(mant_v, mask_impl_v); // add implicit 1.0
    vout = Q6_Vh_vasr_VhVh(vout, rshift_v);  // shift to obtain truncated integer
    vout = Q6_V_vmux_QVV(q_negexp, const_zero_v, vout);     // expval<0 -> 0

    // HVX_Vector neg_vout = -vout;
    HVX_Vector not_vout = Q6_V_vnot_V(vout);
    HVX_Vector neg_vout = Q6_Vh_vadd_VhVh(not_vout, const_one_v);
    vout = Q6_V_vmux_QVV(q_negative, neg_vout, vout);     // handle negatives
    return(vout);
}

// truncate half float to half float
static HVX_INLINE_ALWAYS HVX_Vector qhmath_hvx_vhf_truncate_vhf(HVX_Vector vin) {
    HVX_Vector const_mnlen_v = Q6_Vh_vsplat_R(IEEE_VHF_MANTLEN);
    HVX_Vector mask_mant_v   = Q6_Vh_vsplat_R(IEEE_VHF_MANTMASK);
    HVX_Vector const_emask_v = Q6_Vh_vsplat_R(IEEE_VHF_EXPMASK);
    HVX_Vector const_ebias_v = Q6_Vh_vsplat_R(IEEE_VHF_EXPBIAS);
    HVX_Vector const_zero_v  = Q6_V_vzero();

    HVX_Vector expval_v = Q6_Vh_vasr_VhR(vin, IEEE_VHF_MANTLEN);
    expval_v = Q6_V_vand_VV(expval_v, const_emask_v);
    expval_v = Q6_Vh_vsub_VhVh(expval_v, const_ebias_v);

    HVX_VectorPred q_negexp  = Q6_Q_vcmp_gt_VhVh(const_zero_v, expval_v);
    HVX_VectorPred q_expltmn = Q6_Q_vcmp_gt_VhVh(const_mnlen_v, expval_v);

    mask_mant_v = Q6_Vh_vasr_VhVh(mask_mant_v, expval_v);
    HVX_Vector not_mask_v = Q6_V_vnot_V(mask_mant_v);        // frac to clear
    HVX_Vector vout_clrfrac = Q6_V_vand_VV(vin, not_mask_v); // clear frac

    HVX_Vector vout = vin;                               // integral no change
    vout = Q6_V_vmux_QVV(q_expltmn, vout_clrfrac, vout); // expval<mant -> clr frac
    vout = Q6_V_vmux_QVV(q_negexp, const_zero_v, vout);  // expval<0  -> 0

    return(vout);
}

// round float to float
static HVX_INLINE_ALWAYS HVX_Vector qhmath_hvx_vsf_round_vsf(HVX_Vector vin) {
    HVX_Vector mask_mant_v    = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_Vector mask_impl_v    = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_Vector const_zero_v   = Q6_V_vzero();
    HVX_Vector const_one_v    = Q6_V_vsplat_R(1);
    HVX_Vector const_negone_v = Q6_V_vsplat_R(-1);

    HVX_Vector expval_v = vin >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    HVX_VectorPred q_neg1exp = Q6_Q_vcmp_gt_VwVw(const_negone_v, expval_v);

    // add 0.5 (add 1, 1 bit right of decimal)
    HVX_Vector expval1_v = Q6_Vw_vadd_VwVw(expval_v, const_one_v);
    HVX_Vector half_addin_v = mask_impl_v >> expval1_v;
    HVX_Vector vout = Q6_Vw_vadd_VwVw(vin, half_addin_v);

    // reload possibly changed exponent
    expval_v = vout >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    // clear any remaining fraction
    mask_mant_v >>= expval_v;
    HVX_Vector not_mask_v = Q6_V_vnot_V(mask_mant_v);
    vout = Q6_V_vand_VV(vout, not_mask_v);

    vout = Q6_V_vmux_QVV(q_neg1exp, const_zero_v, vout);  // expval<-1 -> 0

    return(vout);
}

// round half float to half float
static HVX_INLINE_ALWAYS HVX_Vector qhmath_hvx_vhf_round_vhf(HVX_Vector vin) {
    HVX_Vector mask_mant_v    = Q6_Vh_vsplat_R(IEEE_VHF_MANTMASK);
    HVX_Vector mask_impl_v    = Q6_Vh_vsplat_R(IEEE_VHF_MIMPMASK);
    HVX_Vector const_emask_v  = Q6_Vh_vsplat_R(IEEE_VHF_EXPMASK);
    HVX_Vector const_ebias_v  = Q6_Vh_vsplat_R(IEEE_VHF_EXPBIAS);
    HVX_Vector const_zero_v   = Q6_V_vzero();
    HVX_Vector const_one_v    = Q6_Vh_vsplat_R(1);
    HVX_Vector const_negone_v = Q6_Vh_vsplat_R(-1);

    HVX_Vector expval_v = Q6_Vh_vasr_VhR(vin, IEEE_VHF_MANTLEN);
    expval_v = Q6_V_vand_VV(expval_v, const_emask_v);
    expval_v = Q6_Vh_vsub_VhVh(expval_v, const_ebias_v);

    HVX_VectorPred q_neg1exp = Q6_Q_vcmp_gt_VhVh(const_negone_v, expval_v);

    // add 0.5 (add 1, 1 bit right of decimal)
    HVX_Vector expval1_v = Q6_Vh_vadd_VhVh(expval_v, const_one_v);
    HVX_Vector half_addin_v = Q6_Vh_vasr_VhVh(mask_impl_v, expval1_v);
    HVX_Vector vout = Q6_Vh_vadd_VhVh(vin, half_addin_v);

    // reload possibly changed exponent
    expval_v = Q6_Vh_vasr_VhR(vout, IEEE_VHF_MANTLEN);
    expval_v = Q6_V_vand_VV(expval_v, const_emask_v);
    expval_v = Q6_Vh_vsub_VhVh(expval_v, const_ebias_v);

    // clear any remaining fraction
    mask_mant_v = Q6_Vh_vasr_VhVh(mask_mant_v, expval_v);
    HVX_Vector not_mask_v = Q6_V_vnot_V(mask_mant_v);
    vout = Q6_V_vand_VV(vout, not_mask_v);

    vout = Q6_V_vmux_QVV(q_neg1exp, const_zero_v, vout);  // expval<-1 -> 0

    return(vout);
}

// qhmath_hvx_vsf_floor_vsf(x)
//  given a vector of float x,
//  return the vector of largest integer valued float <= x
//
static HVX_INLINE_ALWAYS HVX_Vector qhmath_hvx_vsf_floor_vsf(HVX_Vector vin) {
    HVX_Vector mask_mant_v    = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_Vector mask_impl_v    = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_Vector const_mnlen_v  = Q6_V_vsplat_R(IEEE_VSF_MANTLEN);
    HVX_Vector const_zero_v   = Q6_V_vzero();
    HVX_Vector const_negone_v = Q6_V_vsplat_R(0xbf800000);  // -1 IEEE vsf

    // initialization (no changes)
    HVX_VectorPred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, vin);

    HVX_Vector expval_v = vin >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    HVX_VectorPred q_negexp  = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);
    HVX_VectorPred q_expltmn = Q6_Q_vcmp_gt_VwVw(const_mnlen_v, expval_v);
    HVX_VectorPred q_negexp_pos = Q6_Q_vcmp_gtand_QVwVw(q_negexp, vin, const_zero_v);
    HVX_VectorPred q_negexp_neg = Q6_Q_vcmp_gtand_QVwVw(q_negexp, const_zero_v, vin);

    // if expval < 0 (q_negexp)   // <0, floor is 0
    //    if vin > 0
    //       floor = 0
    //    if vin < 0
    //       floor = -1
    // if expval < mant_len (q_expltmn) // >0, but fraction may exist
    //    get sign (q_negative)
    //    mask >> expval          // fraction bits to mask off
    //    vout = ~(mask)          // apply mask to remove fraction
    //    if (qneg) // negative floor is one less (more, sign bit for neg)
    //      vout += ((impl_mask) >> expval)
    //    if (mask && vin)
    //      vout = vin
    // else                       // already an integer
    //    ; // no change

    // compute floor
    mask_mant_v >>= expval_v;
    HVX_Vector neg_addin_v = mask_impl_v >> expval_v;
    HVX_Vector vout_neg_addin = Q6_Vw_vadd_VwVw(vin, neg_addin_v);
    HVX_Vector vout = Q6_V_vmux_QVV(q_negative, vout_neg_addin, vin);

    HVX_Vector mask_chk_v = Q6_V_vand_VV(vin, mask_mant_v);  // chk if bits set
    HVX_VectorPred q_integral = Q6_Q_vcmp_eq_VwVw(const_zero_v, mask_chk_v);

    HVX_Vector not_mask_v = Q6_V_vnot_V(mask_mant_v);  // frac bits to clear
    HVX_Vector vfrfloor_v = Q6_V_vand_VV(vout, not_mask_v); // clear frac bits

    vout = vin;
    vout = Q6_V_vmux_QVV(q_expltmn, vfrfloor_v, vout);    // expval<mant
    vout = Q6_V_vmux_QVV(q_integral, vin, vout);          // integral values
    vout = Q6_V_vmux_QVV(q_negexp_pos, const_zero_v, vout);  // expval<0 x>0 -> 0
    vout = Q6_V_vmux_QVV(q_negexp_neg, const_negone_v, vout);  // expval<0 x<0 -> -1
    return vout;
}

// qhmath_hvx_vhf_floor_vhf(x)
//  given a vector of half float x,
//  return the vector of largest integer valued half float <= x
//
static HVX_INLINE_ALWAYS HVX_Vector qhmath_hvx_vhf_floor_vhf(HVX_Vector vin) {
    HVX_Vector mask_mant_v    = Q6_Vh_vsplat_R(IEEE_VHF_MANTMASK);
    HVX_Vector mask_impl_v    = Q6_Vh_vsplat_R(IEEE_VHF_MIMPMASK);
    HVX_Vector const_mnlen_v  = Q6_Vh_vsplat_R(IEEE_VHF_MANTLEN);
    HVX_Vector const_emask_v  = Q6_Vh_vsplat_R(IEEE_VHF_EXPMASK);
    HVX_Vector const_ebias_v  = Q6_Vh_vsplat_R(IEEE_VHF_EXPBIAS);
    HVX_Vector const_zero_v   = Q6_V_vzero();
    HVX_Vector const_negone_v = Q6_Vh_vsplat_R(0xbc00);  // -1 IEEE vhf

    // initialization (no changes)
    HVX_VectorPred q_negative = Q6_Q_vcmp_gt_VhVh(const_zero_v, vin);

    HVX_Vector expval_v = Q6_Vh_vasr_VhR(vin, IEEE_VHF_MANTLEN);
    expval_v = Q6_V_vand_VV(expval_v, const_emask_v);
    expval_v = Q6_Vh_vsub_VhVh(expval_v, const_ebias_v);

    HVX_VectorPred q_negexp  = Q6_Q_vcmp_gt_VhVh(const_zero_v, expval_v);
    HVX_VectorPred q_expltmn = Q6_Q_vcmp_gt_VhVh(const_mnlen_v, expval_v);
    HVX_VectorPred q_negexp_pos = Q6_Q_vcmp_gtand_QVhVh(q_negexp, vin, const_zero_v);
    HVX_VectorPred q_negexp_neg = Q6_Q_vcmp_gtand_QVhVh(q_negexp, const_zero_v, vin);

    // if expval < 0 (q_negexp)   // <0, floor is 0
    //    if vin > 0
    //       floor = 0
    //    if vin < 0
    //       floor = -1
    // if expval < mant_len (q_expltmn) // >0, but fraction may exist
    //    get sign (q_negative)
    //    mask >> expval          // fraction bits to mask off
    //    vout = ~(mask)          // apply mask to remove fraction
    //    if (qneg) // negative floor is one less (more, sign bit for neg)
    //      vout += ((impl_mask) >> expval)
    //    if (mask && vin)
    //      vout = vin
    // else                       // already an integer
    //    ; // no change

    // compute floor
    mask_mant_v = Q6_Vh_vasr_VhVh(mask_mant_v, expval_v);
    HVX_Vector neg_addin_v = Q6_Vh_vasr_VhVh(mask_impl_v, expval_v);
    HVX_Vector vout_neg_addin = Q6_Vh_vadd_VhVh(vin, neg_addin_v);
    HVX_Vector vout = Q6_V_vmux_QVV(q_negative, vout_neg_addin, vin);

    HVX_Vector mask_chk_v = Q6_V_vand_VV(vin, mask_mant_v);  // chk if bits set
    HVX_VectorPred q_integral = Q6_Q_vcmp_eq_VhVh(const_zero_v, mask_chk_v);

    HVX_Vector not_mask_v = Q6_V_vnot_V(mask_mant_v);  // frac bits to clear
    HVX_Vector vfrfloor_v = Q6_V_vand_VV(vout, not_mask_v); // clear frac bits

    vout = vin;
    vout = Q6_V_vmux_QVV(q_expltmn, vfrfloor_v, vout);    // expval<mant
    vout = Q6_V_vmux_QVV(q_integral, vin, vout);          // integral values
    vout = Q6_V_vmux_QVV(q_negexp_pos, const_zero_v, vout);  // expval<0 x>0 -> 0
    vout = Q6_V_vmux_QVV(q_negexp_neg, const_negone_v, vout);  // expval<0 x<0 -> -1
    return vout;
}

// qhmath_hvx_vsf_ceil_vsf(x)
//  given a vector of float x,
//  return the vector of largest integer valued float <= x
//
static HVX_INLINE_ALWAYS HVX_Vector qhmath_hvx_vsf_ceil_vsf(HVX_Vector vin) {
    HVX_Vector mask_mant_v    = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_Vector mask_impl_v    = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_Vector const_mnlen_v  = Q6_V_vsplat_R(IEEE_VSF_MANTLEN);
    HVX_Vector const_zero_v   = Q6_V_vzero();
    HVX_Vector const_one_v    = Q6_V_vsplat_R(0x3f800000);  // +1 IEEE vsf

    // initialization (no changes)
    HVX_VectorPred q_positive = Q6_Q_vcmp_gt_VwVw(vin, const_zero_v);

    HVX_Vector expval_v = vin >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    HVX_VectorPred q_negexp  = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);
    HVX_VectorPred q_expltmn = Q6_Q_vcmp_gt_VwVw(const_mnlen_v, expval_v);
    HVX_VectorPred q_negexp_pos = Q6_Q_vcmp_gtand_QVwVw(q_negexp, vin, const_zero_v);
    HVX_VectorPred q_negexp_neg = Q6_Q_vcmp_gtand_QVwVw(q_negexp, const_zero_v, vin);

    // if expval < 0 (q_negexp)   // <0, ceil is 0
    //    if vin > 0
    //       ceil = 1
    //    if vin < 0
    //       ceil = 0
    // if expval < mant_len (q_expltmn) // >0, but fraction may exist
    //    get sign (q_negative)
    //    mask >> expval          // fraction bits to mask off
    //    vout = ~(mask)          // apply mask to remove fraction
    //    if (qpos) // positive ceil is one more
    //      vout += ((impl_mask) >> expval)
    //    if (mask && vin)
    //      vout = vin
    // else                       // already an integer
    //    ; // no change

    // compute ceil
    mask_mant_v >>= expval_v;
    HVX_Vector pos_addin_v = mask_impl_v >> expval_v;
    HVX_Vector vout_pos_addin = Q6_Vw_vadd_VwVw(vin, pos_addin_v);
    HVX_Vector vout = Q6_V_vmux_QVV(q_positive, vout_pos_addin, vin);

    HVX_Vector mask_chk_v = Q6_V_vand_VV(vin, mask_mant_v);  // chk if bits set
    HVX_VectorPred q_integral = Q6_Q_vcmp_eq_VwVw(const_zero_v, mask_chk_v);

    HVX_Vector not_mask_v = Q6_V_vnot_V(mask_mant_v);  // frac bits to clear
    HVX_Vector vfrceil_v = Q6_V_vand_VV(vout, not_mask_v); // clear frac bits

    vout = vin;
    vout = Q6_V_vmux_QVV(q_expltmn, vfrceil_v, vout);    // expval<mant
    vout = Q6_V_vmux_QVV(q_integral, vin, vout);          // integral values
    vout = Q6_V_vmux_QVV(q_negexp_pos, const_one_v, vout);  // expval<0 x>0 -> 1
    vout = Q6_V_vmux_QVV(q_negexp_neg, const_zero_v, vout);  // expval<0 x<0 -> 0
    return vout;
}

// qhmath_hvx_vhf_ceil_vhf(x)
//  given a vector of half float x,
//  return the vector of largest integer valued half float <= x
//
static HVX_INLINE_ALWAYS HVX_Vector qhmath_hvx_vhf_ceil_vhf(HVX_Vector vin) {
    HVX_Vector mask_mant_v    = Q6_Vh_vsplat_R(IEEE_VHF_MANTMASK);
    HVX_Vector mask_impl_v    = Q6_Vh_vsplat_R(IEEE_VHF_MIMPMASK);
    HVX_Vector const_mnlen_v  = Q6_Vh_vsplat_R(IEEE_VHF_MANTLEN);
    HVX_Vector const_emask_v  = Q6_Vh_vsplat_R(IEEE_VHF_EXPMASK);
    HVX_Vector const_ebias_v  = Q6_Vh_vsplat_R(IEEE_VHF_EXPBIAS);
    HVX_Vector const_zero_v   = Q6_V_vzero();
    HVX_Vector const_one_v    = Q6_Vh_vsplat_R(0x3c00);  // +1 IEEE vhf

    // initialization (no changes)
    HVX_VectorPred q_positive = Q6_Q_vcmp_gt_VhVh(vin, const_zero_v);

    HVX_Vector expval_v = Q6_Vh_vasr_VhR(vin, IEEE_VHF_MANTLEN);
    expval_v = Q6_V_vand_VV(expval_v, const_emask_v);
    expval_v = Q6_Vh_vsub_VhVh(expval_v, const_ebias_v);

    HVX_VectorPred q_negexp  = Q6_Q_vcmp_gt_VhVh(const_zero_v, expval_v);
    HVX_VectorPred q_expltmn = Q6_Q_vcmp_gt_VhVh(const_mnlen_v, expval_v);
    HVX_VectorPred q_negexp_pos = Q6_Q_vcmp_gtand_QVhVh(q_negexp, vin, const_zero_v);
    HVX_VectorPred q_negexp_neg = Q6_Q_vcmp_gtand_QVhVh(q_negexp, const_zero_v, vin);

    // if expval < 0 (q_negexp)   // <0, ceil is 0
    //    if vin > 0
    //       ceil = 1
    //    if vin < 0
    //       ceil = 0
    // if expval < mant_len (q_expltmn) // >0, but fraction may exist
    //    get sign (q_negative)
    //    mask >> expval          // fraction bits to mask off
    //    vout = ~(mask)          // apply mask to remove fraction
    //    if (qpos) // positive ceil is one more
    //      vout += ((impl_mask) >> expval)
    //    if (mask && vin)
    //      vout = vin
    // else                       // already an integer
    //    ; // no change

    // compute ceil
    mask_mant_v = Q6_Vh_vasr_VhVh(mask_mant_v, expval_v);
    HVX_Vector pos_addin_v = Q6_Vh_vasr_VhVh(mask_impl_v, expval_v);
    HVX_Vector vout_pos_addin = Q6_Vh_vadd_VhVh(vin, pos_addin_v);
    HVX_Vector vout = Q6_V_vmux_QVV(q_positive, vout_pos_addin, vin);

    HVX_Vector mask_chk_v = Q6_V_vand_VV(vin, mask_mant_v);  // chk if bits set
    HVX_VectorPred q_integral = Q6_Q_vcmp_eq_VhVh(const_zero_v, mask_chk_v);

    HVX_Vector not_mask_v = Q6_V_vnot_V(mask_mant_v);  // frac bits to clear
    HVX_Vector vfrceil_v = Q6_V_vand_VV(vout, not_mask_v); // clear frac bits

    vout = vin;
    vout = Q6_V_vmux_QVV(q_expltmn, vfrceil_v, vout);    // expval<mant
    vout = Q6_V_vmux_QVV(q_integral, vin, vout);          // integral values
    vout = Q6_V_vmux_QVV(q_negexp_pos, const_one_v, vout);  // expval<0 x>0 -> 0
    vout = Q6_V_vmux_QVV(q_negexp_neg, const_zero_v, vout);  // expval<0 x<0 -> -1
    return vout;
}

// floor + 0.5
// #define USE_ROUND_IN_FLOOR

/**
 * floor with round single float
 * Vu input sf vector
 * Vd output sf vector, floored
 * return: Vw vector with floored integers
 * Note that this function is depreciated and will be removed in future releases.
 **/
static HVX_INLINE_ALWAYS HVX_Vector Q6_Vw_vfloor_VsfVsf(HVX_Vector Vu, HVX_Vector *Vd)
{
    HVX_Vector round_f;
    HVX_Vector o_i_v;
    HVX_Vector o_f_v;
    HVX_Vector expval_v;
    HVX_Vector mantissa_v;
    HVX_Vector mantissa_shift_v;
    HVX_Vector mask;
    HVX_Vector const_zero_v;
    HVX_Vector const_v;

    const_zero_v = Q6_V_vzero();

#ifdef USE_ROUND_IN_FLOOR
    const_v = Q6_V_vsplat_R(0x3f000000); // 0.5
    round_f = Q6_Vqf32_vadd_VsfVsf(Vu, const_v);
    round_f = Q6_Vsf_equals_Vqf32(round_f);
#else
    round_f = Vu;
#endif

    const_v = Q6_V_vsplat_R(0x7FFFFFFF);
    HVX_VectorPred qpred_negative_vq = Q6_Q_vcmp_gt_VuwVuw(round_f, const_v);

    // const_v = Q6_V_vsplat_R(23);
    expval_v = round_f >> 23;
    expval_v &= 0xFF;
//    HVX_VectorPred qpred_denormalized_vq = Q6_Q_vcmp_eq_VwVw(expval_v, const_zero_v); // XXX: not supported
    expval_v -= 127;

    HVX_VectorPred qpred_negativexp_vq = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);
    // clip negative exponent to zero
    expval_v = Q6_Vw_vmax_VwVw(expval_v, const_zero_v);

    mantissa_shift_v = 23 - expval_v;
    mantissa_shift_v = Q6_Vw_vmax_VwVw(mantissa_shift_v, const_zero_v);

    mantissa_v = round_f;
    mantissa_v &= ((1 << 23) - 1);

    mantissa_v >>= mantissa_shift_v;

    o_i_v = 1 << expval_v;
    o_i_v = Q6_V_vmux_QVV(qpred_negativexp_vq, const_zero_v, o_i_v);

    o_i_v += mantissa_v;

    // fixing sign of integer value
    HVX_Vector negative_i_v = -o_i_v;
    o_i_v = Q6_V_vmux_QVV(qpred_negative_vq, negative_i_v, o_i_v);

    mask = (1 << mantissa_shift_v);
    mask = -mask;
    round_f &= mask;

//    o_f_v = round_f;
    o_f_v = Q6_V_vmux_QVV(qpred_negativexp_vq, const_zero_v, round_f);

    *Vd = o_f_v;
    return o_i_v;
}


/**
 * floor with round single float
 * Vu input hf vector
 * Vd output hf vector, floored
 * return: Vw vector with floored 16bit half
 * Note that this function is depreciated and will be removed in future releases.
 **/
static HVX_INLINE_ALWAYS HVX_Vector Q6_Vh_vfloor_VhfVhf(HVX_Vector Vu, HVX_Vector *Vd)
{
    HVX_Vector round_f;
    HVX_Vector o_i_v;
    HVX_Vector o_f_v;
    HVX_Vector expval_v;
    HVX_Vector mantissa_v;
    HVX_Vector mantissa_shift_v;
    HVX_Vector mask;
    HVX_Vector const_zero_v;
    HVX_Vector const_v;

    const_zero_v = Q6_V_vzero();

#ifdef USE_ROUND_IN_FLOOR
    const_v = Q6_Vh_vsplat_R(0x3800); // 0.5 in __fp16
    round_f = Q6_V_vadd_VhfVhf(Vu, const_v);
    round_f = Q6_Vhf_equals_V(round_f);
#else
    round_f = Vu;
#endif

    const_v = Q6_Vh_vsplat_R(0x7FFF);
    HVX_VectorPred qpred_negative_vq = Q6_Q_vcmp_gt_VuhVuh(round_f, const_v);

    // const_v = Q6_V_vsplat_R(23);
    expval_v = Q6_Vuh_vlsr_VuhR(round_f, 10);
    const_v = Q6_Vh_vsplat_R(0x1F);
    expval_v =  Q6_V_vand_VV(expval_v, const_v);
//    HVX_VectorPred qpred_denormalized_vq = Q6_Q_vcmp_eq_VwVw(expval_v, const_zero_v); // XXX: not supported
    const_v = Q6_Vh_vsplat_R(0x000F); // 15
    expval_v =  Q6_Vh_vsub_VhVh(expval_v, const_v); // exponent - offset 15

    HVX_VectorPred qpred_negativexp_vq = Q6_Q_vcmp_gt_VhVh(const_zero_v, expval_v);
    // clip negative exponent to zero
    expval_v = Q6_Vh_vmax_VhVh(expval_v, const_zero_v);

    const_v = Q6_Vh_vsplat_R(0x000A); // 10
    mantissa_shift_v = Q6_Vh_vsub_VhVh(const_v, expval_v);
    mantissa_shift_v = Q6_Vh_vmax_VhVh(mantissa_shift_v, const_zero_v);

    mantissa_v = round_f;
    const_v = Q6_Vh_vsplat_R(0x03FF); // ((1 << 10) - 1)
    mantissa_v = Q6_V_vand_VV(mantissa_v, const_v);

    mantissa_v =  Q6_Vh_vlsr_VhVh(mantissa_v, mantissa_shift_v);

    const_v = Q6_Vh_vsplat_R(0x0001); // 1
    o_i_v = Q6_Vh_vasl_VhVh(const_v, expval_v);            // 1 << expval_v
    o_i_v = Q6_V_vmux_QVV(qpred_negativexp_vq, const_zero_v, o_i_v);

    o_i_v = Q6_Vh_vadd_VhVh(o_i_v, mantissa_v);    // o_i_v += mantissa_v;

    // fixing sign of integer value
    HVX_Vector negative_i_v = Q6_Vh_vsub_VhVh(const_zero_v, o_i_v); // check this - o_i_v
    o_i_v = Q6_V_vmux_QVV(qpred_negative_vq, negative_i_v, o_i_v);

    mask = Q6_Vh_vasl_VhVh(const_v, mantissa_shift_v);//(1 << mantissa_shift_v);
    mask = Q6_Vh_vsub_VhVh(const_zero_v, mask); // check this - mask
    round_f = Q6_V_vand_VV(round_f, mask);

//    o_f_v = round_f;
    o_f_v = Q6_V_vmux_QVV(qpred_negativexp_vq, const_zero_v, round_f);

    *Vd = o_f_v;
    return o_i_v;
}

static inline HVX_Vector vqf32_from_int(HVX_Vector src)
{
    HVX_Vector const_126 = Q6_V_vsplat_R(0x0000007e);
    HVX_Vector const31 = Q6_V_vsplat_R(31);
    HVX_Vector mant = src;
    HVX_Vector exp = Q6_Vw_vnormamt_Vw(mant);
    mant = Q6_Vw_vasl_VwVw(mant, exp);
    exp = Q6_Vw_vsub_VwVw(const31, exp);
    exp = Q6_Vw_vadd_VwVw(exp, const_126);
    return Q6_V_vor_VV(mant, exp);
}

static inline HVX_Vector vqf16_from_int(HVX_Vector src)
{
    HVX_Vector const_14 = Q6_Vh_vsplat_R(0x000e);
    HVX_Vector const15 = Q6_Vh_vsplat_R(15);
    HVX_Vector mant = src;
    HVX_Vector exp = Q6_Vh_vnormamt_Vh(mant);
    mant = Q6_Vh_vasl_VhVh(mant, exp);
    exp = Q6_Vh_vsub_VhVh(const15, exp);
    exp = Q6_Vh_vadd_VhVh(exp, const_14);
    return Q6_V_vor_VV(mant, exp);
}

#endif /* _HVX_INTERNAL_H */

