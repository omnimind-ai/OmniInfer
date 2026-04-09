#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace omniinfer {

struct InferenceMetrics {
  int prompt_tokens = 0;
  int generated_tokens = 0;
  int64_t prefill_us = 0;
  int64_t decode_us = 0;
};

class InferenceBackend {
public:
  virtual ~InferenceBackend() = default;

  virtual bool load(const std::string& model_path, const std::string& config_json,
                    const std::string& native_lib_dir, int n_threads, int n_ctx) = 0;

  // on_token returns false to stop generation.
  virtual std::string generate(
      const std::string& system_prompt,
      const std::string& user_prompt,
      bool thinking_enabled,
      std::atomic<bool>& cancelled,
      std::function<bool(const std::string& token)> on_token,
      const std::string& tools_json = "",
      const std::string& tool_choice = "",
      const std::string& messages_json = "") = 0;

  virtual bool load_history(
      const std::vector<std::pair<std::string, std::string>>& messages) = 0;

  virtual void reset() = 0;

  virtual InferenceMetrics get_metrics() = 0;

  virtual const char* name() const = 0;
};

}  // namespace omniinfer
