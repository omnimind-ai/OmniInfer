#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API bool ggml_backend_is_htp(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_htp_reg(void);

// buffer type check
bool ggml_backend_buft_is_rpcmem(ggml_backend_buffer_type_t buft);

#ifdef __cplusplus
}
#endif
