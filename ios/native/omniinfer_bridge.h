#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int64_t OmniInferHandle;
typedef bool (*OmniInferTokenCallback)(const char *token, void *userdata);
typedef void (*OmniInferMetricsCallback)(const char *metrics, void *userdata);

OmniInferHandle omniinfer_init(const char *config_json);
const char *omniinfer_generate(OmniInferHandle handle,
                               const char *system_prompt,
                               const char *user_prompt,
                               const char *request_json,
                               OmniInferTokenCallback on_token,
                               OmniInferMetricsCallback on_metrics,
                               void *userdata);
void omniinfer_free_string(const char *str);
bool omniinfer_load_history(OmniInferHandle handle,
                            const char **roles,
                            const char **contents,
                            int count);
void omniinfer_set_think_mode(OmniInferHandle handle, bool enabled);
void omniinfer_reset(OmniInferHandle handle);
void omniinfer_cancel(OmniInferHandle handle);
void omniinfer_free(OmniInferHandle handle);
const char *omniinfer_collect_diagnostics_json(OmniInferHandle handle);

#ifdef __cplusplus
}
#endif
