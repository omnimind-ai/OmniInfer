#pragma once

#include <stdbool.h>

#include "ggml-cpu/ggml-cpu-impl.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

bool htp_ops_support_op(const struct ggml_tensor * dst);
int  htp_ops_compute_op(struct ggml_compute_params * params, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
