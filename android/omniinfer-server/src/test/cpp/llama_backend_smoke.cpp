#include "backend_llama_cpp.h"

#include <iostream>
#include <fstream>
#include <iterator>

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) {
    std::cerr << "usage: omniinfer-llama-smoke MODEL.gguf [IMAGE]\n";
    return 2;
  }
  std::vector<std::vector<uint8_t>> images;
  if (argc == 3) {
    std::ifstream file(argv[2], std::ios::binary);
    if (!file) return 2;
    images.emplace_back(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    if (images.front().empty()) return 2;
  }
  const std::string prompt = (images.empty() ? "" : "<image>\n") +
      std::string("What is the capital of France, and what is 12 times 7? Answer briefly.");
  omniinfer::LlamaCppBackend backend;
  if (!backend.load(argv[1], R"({"llama_device":"none"})", "", 8, 2048)) {
    std::cerr << "model load failed\n";
    return 1;
  }
  for (bool thinking : {false, true}) {
    backend.reset();
    std::atomic<bool> cancelled{false};
    std::atomic<bool> graceful_stop{false};
    const auto response = backend.generate(
        "", prompt,
        thinking, cancelled, {}, "", "", "", images, 256, graceful_stop,
        R"({"temperature":0,"seed":42})");
    const auto metrics = backend.get_metrics();
    std::cout << "thinking=" << thinking << " prompt_tokens=" << metrics.prompt_tokens
              << " generated_tokens=" << metrics.generated_tokens << "\n"
              << response << "\n";
    if (response.find("Paris") == std::string::npos || response.find("84") == std::string::npos ||
        metrics.prompt_tokens < 10 || metrics.generated_tokens <= 0 ||
        (!images.empty() && metrics.image_tokens <= 0)) {
      std::cerr << "correctness gate failed\n";
      return 1;
    }
  }
  return 0;
}
