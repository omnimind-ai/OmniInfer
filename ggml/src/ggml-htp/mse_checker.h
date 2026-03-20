#pragma once

#include "ggml.h"
#include "ggml-impl.h"
#include "htp_layout.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float * data;      // data pointer, row-major storage: data[i*n + j] represents element (i,j)
    int64_t m;         // number of rows
    int64_t n;         // number of columns
    bool allocated;    // data is allocated by this structure (need to be freed)
} simple_matrix_t;


typedef struct {
    double sum;             // average of the difference sum
    double mse_sum;         // MSE accumulated sum
    double mse_max;         // maximum MSE
    double mse_min;         // minimum MSE
    int count;              // number of calculations
    double max_abs_error;   // maximum absolute error
} mse_stats_t;

// global statistics variables
static mse_stats_t g_mse_stats = {0.0, 0.0, 0.0, 1e30, 0, 0.0};



// create and allocate memory
static inline simple_matrix_t * simple_matrix_create(int64_t m, int64_t n) {
    simple_matrix_t * mat = (simple_matrix_t *)malloc(sizeof(simple_matrix_t));
    GGML_ASSERT(mat);
    
    mat->data = (float *)malloc(sizeof(float) * (size_t)(m * n));
    if (!mat->data) {
        free(mat);
        return NULL;
    }
    
    mat->m = m;
    mat->n = n;
    mat->allocated = true;
    return mat;
}

// free memory
static inline void simple_matrix_free(simple_matrix_t * mat) {
    if (!mat) return;
    if (mat->allocated && mat->data) {
        free(mat->data);
    }
    free(mat);
}



// extract data from GGML tensor to row-major matrix
static inline simple_matrix_t * simple_matrix_from_ggml_tensor(
    const struct ggml_tensor * tensor
) {
    if (tensor->type != GGML_TYPE_F32) {
        fprintf(stderr, "[MSE] Error: Only F32 output tensors supported\n");
        return NULL;
    }
    
    int64_t m = tensor->ne[1];  // number of rows
    int64_t n = tensor->ne[0];  // number of columns
    
    simple_matrix_t * mat = simple_matrix_create(m, n);
    if (!mat) return NULL;
    
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            const float * elem = (const float *)(
                (const char *)tensor->data + 
                i * tensor->nb[1] + 
                j * tensor->nb[0]
            );
            mat->data[i * n + j] = *elem;
        }
    }
    
    return mat;
}



// compute CPU ref for F16 weight × F32 activation
static inline simple_matrix_t * compute_cpu_ref_w16a32(
    const struct ggml_tensor * weight,
    const struct ggml_tensor * act
) {
    const int64_t k = weight->ne[0];
    const int64_t n = weight->ne[1];
    const int64_t m = ggml_nrows(act);
    
    simple_matrix_t * result = simple_matrix_create(m, n);
    GGML_ASSERT(result);
    
    // convert weight to GGML column-major (standard layout)
    ggml_fp16_t * w_ref_f16 = (ggml_fp16_t *)malloc(sizeof(ggml_fp16_t) * (size_t)(k * n));
    GGML_ASSERT(w_ref_f16);
    
    const ggml_fp16_t * w_perm = (const ggml_fp16_t *)weight->data;
    if ((k % 32 == 0) && (n % 32 == 0)) {
        htp_w16_tile32_to_ggml_f16(w_perm, w_ref_f16, (size_t)k, (size_t)n);
    } else {
        // non-aligned case
        for (int ii = 0; ii < k; ++ii) {
            for (int jj = 0; jj < n; ++jj) {
                int idx = htp_w16_tile32_index(ii, jj, (int)k, (int)n);
                w_ref_f16[jj * (int)k + ii] = w_perm[idx];
            }
        }
    }
    
    // compute matrix multiplication: result[m,n] = act[m,k] × weight[k,n]
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int64_t p = 0; p < k; ++p) {
                const float * ap = (const float *)(
                    (const char *)act->data + 
                    i * act->nb[1] + 
                    p * act->nb[0]
                );
                
                ggml_fp16_t a16 = ggml_fp32_to_fp16(*ap);
                ggml_fp16_t w16 = w_ref_f16[j * k + p];
                
                acc += (float)(ggml_fp16_to_fp32(a16) * ggml_fp16_to_fp32(w16));
            }
            result->data[i * n + j] = acc;
        }
    }
    
    free(w_ref_f16);
    return result;
}

// generic CPU ref computation entry
static inline simple_matrix_t * compute_cpu_ref(
    const struct ggml_tensor * weight,
    const struct ggml_tensor * act
) {
    // currently only supports F16 weight × F32 activation
    if (weight->type == GGML_TYPE_F16 && act->type == GGML_TYPE_F32) {
        return compute_cpu_ref_w16a32(weight, act);
    }
    
    fprintf(stderr, "[MSE] Unsupported types: weight=%s, act=%s\n",
            ggml_type_name(weight->type), ggml_type_name(act->type));
    return NULL;
}



// compute MSE and related statistics between two matrices
static inline double compute_mse(
    const simple_matrix_t * mat1,
    const simple_matrix_t * mat2,
    double * out_max_abs_error,
    double * out_sum,
    double * out_max_abs_origin,
    double * out_max_abs_ref
) {
    GGML_ASSERT(mat1);
    GGML_ASSERT(mat2);
    GGML_ASSERT(mat1->m == mat2->m && mat1->n == mat2->n);
    
    double mse = 0.0;
    double max_abs_err = 0.0;
    int64_t total = mat1->m * mat1->n;
    
    for (int64_t i = 0; i < total; ++i) {
        double diff = (double)mat1->data[i] - (double)mat2->data[i];
        double abs_err = fabs(diff);
        
        mse += diff * diff;
        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            (*out_max_abs_origin) = (double)mat1->data[i];
            (*out_max_abs_ref) = (double)mat2->data[i];
        }
        (*out_sum) += abs_err;
    }
    
    mse /= (double)total;
    
    if (out_max_abs_error) {
        *out_max_abs_error = max_abs_err;
    }
    
    return mse;
}



// reset statistics
static inline void mse_stats_reset(void) {
    g_mse_stats.mse_sum = 0.0;
    g_mse_stats.mse_max = 0.0;
    g_mse_stats.mse_min = 1e30;
    g_mse_stats.count = 0;
    g_mse_stats.max_abs_error = 0.0;
}

// update statistics
static inline void mse_stats_update(double mse, double max_abs_error, double sum) {
    g_mse_stats.mse_sum += mse;
    g_mse_stats.sum += sum;
    g_mse_stats.count++;
    
    if (mse > g_mse_stats.mse_max) {
        g_mse_stats.mse_max = mse;
    }
    if (mse < g_mse_stats.mse_min) {
        g_mse_stats.mse_min = mse;
    }
    if (max_abs_error > g_mse_stats.max_abs_error) {
        g_mse_stats.max_abs_error = max_abs_error;
    }
}

// get current average MSE
static inline double mse_stats_get_average(void) {
    if (g_mse_stats.count == 0) return 0.0;
    return g_mse_stats.mse_sum / (double)g_mse_stats.count;
}

// get current average sum
static inline double mse_stats_get_sum_avg(void) {
    if (g_mse_stats.count == 0) return 0.0;
    return g_mse_stats.sum / (double)g_mse_stats.count;
}

// print current statistics
static inline void mse_stats_print(const char * op_name, double current_mse, double current_max_abs_err, double diff_sum, double max_abs_origin, double max_abs_ref) {
    double mse_avg = mse_stats_get_average();
    double sum_avg = mse_stats_get_sum_avg();

    fprintf(stderr, "[MSE] %s: SUM=%.6f (avg=%.6f), MSE=%.6f (avg=%.6f, max=%.6f) | MaxAbsErr=%.6f | MaxAbsOrigin=%.6f MaxAbsRef=%.6f | count=%d\n",
            op_name ? op_name : "unknown",
            diff_sum,
            sum_avg,
            current_mse,
            mse_avg,
            g_mse_stats.mse_max,
            current_max_abs_err,
            max_abs_origin,
            max_abs_ref,
            g_mse_stats.count);
    // SUM: the sum of the difference
    // avg: the average of the difference
    // MSE: the MSE of the calculation
    // avg: the average of the MSE
    // max: the maximum of the MSE
    // MaxAbsErr: the maximum absolute error (absolute value)
    // MaxAbsOrigin: the maximum absolute error (original value)
    // MaxAbsRef: the maximum absolute error (reference value)
    // count: the number of calculations
}

// print final summary (call seems to have bugs)
static inline void mse_stats_print_summary(void) {
    double mse_avg = mse_stats_get_average();
    double sum_avg = mse_stats_get_sum_avg();
    
    fprintf(stderr, "\n");
    fprintf(stderr, "[MSE] ============ Summary ============\n");
    fprintf(stderr, "[MSE] Total operations: %d\n", g_mse_stats.count);
    fprintf(stderr, "[MSE] Average MSE:     %.6f\n", mse_avg);
    fprintf(stderr, "[MSE] Average SUM:     %.6f\n", sum_avg);
    fprintf(stderr, "[MSE] Max MSE:         %.6f\n", g_mse_stats.mse_max);
    fprintf(stderr, "[MSE] Min MSE:         %.6f\n", g_mse_stats.mse_min);
    fprintf(stderr, "[MSE] Max Abs Error:   %.6f\n", g_mse_stats.max_abs_error);
    fprintf(stderr, "[MSE] ===================================\n\n");
}


// external interface
// check MSE for a single MUL_MAT operation
static inline void check_mul_mat_mse(
    const struct ggml_tensor * output_tensor,  // XPU result
    const struct ggml_tensor * weight,
    const struct ggml_tensor * act,
    const char * op_name
) {
    // convert XPU output to simple matrix
    simple_matrix_t * xpu_result = simple_matrix_from_ggml_tensor(output_tensor);
    GGML_ASSERT(xpu_result);
    
    // compute CPU ref
    simple_matrix_t * cpu_ref = compute_cpu_ref(weight, act);
    if(cpu_ref == NULL) {
        fprintf(stderr, "[MSE] Failed to compute CPU ref\n");
        simple_matrix_free(xpu_result);
        return;
    }
    
    fprintf(stderr, "first 2 value: [%f,%f] vs [%f,%f]\n", xpu_result->data[0], xpu_result->data[1], cpu_ref->data[0], cpu_ref->data[1]);

    // compute MSE
    double max_abs_error = 0.0, sum = 0.0, max_abs_origin = 0.0, max_abs_ref = 0.0;
    double mse = compute_mse(xpu_result, cpu_ref, &max_abs_error, &sum, &max_abs_origin, &max_abs_ref);
    
    if (mse >= 0.0) {
        // update statistics
        mse_stats_update(mse, max_abs_error, sum);
        
        // print result
        mse_stats_print(op_name, mse, max_abs_error, sum, max_abs_origin, max_abs_ref);
    } else {
        fprintf(stderr, "[MSE] MSE computation failed\n");
    }
    
    simple_matrix_free(xpu_result);
    simple_matrix_free(cpu_ref);
}

#ifdef __cplusplus
}
#endif

