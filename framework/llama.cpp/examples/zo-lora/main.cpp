#include "arg.h"
#include "batch-layout.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"
#include "llama-adapter.h"
#include "llama-model.h"
#include "llama.h"
#include "loss.h"
#include "log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace {

namespace fs = std::filesystem;

constexpr uint64_t DEFAULT_SEED = 1337;

enum class run_mode {
    CPU,
    COOP,
};

enum class lora_exec_mode {
    RUNTIME,
    FUSED_HTP,
};

struct options {
    run_mode mode = run_mode::CPU;
    lora_exec_mode lora_exec = lora_exec_mode::RUNTIME;
    bool lora_exec_set = false;
    bool pipeline = false;
    bool antithetic = false;
    int warmup_steps = 2;
    int steps = 2000;
    int eval_step = 50;
    int batch_size = 4;
    int seq_len = 128;
    int max_train = 1000;
    int max_eval = 1000;
    int rank = 8;
    float alpha = 16.0f;
    float epsilon = 1e-2f;
    float learning_rate = 5e-5f;
    uint64_t seed = DEFAULT_SEED;
    std::string train_path;
    std::string eval_path;
    std::string lora_out;
    std::string hexagon_arch = "auto";
};

struct sst2_sample {
    std::vector<llama_token> tokens;
    std::array<llama_token, 2> class_tokens = { LLAMA_TOKEN_NULL, LLAMA_TOKEN_NULL };
    int label = 0;
};

struct lora_tensor_state {
    std::string name;
    llama_adapter_lora_weight * weight = nullptr;
    int64_t rank = 0;
    int64_t output_dim = 0;
    int64_t padded_output_dim = 0;
    std::vector<float> master;
    std::vector<ggml_fp16_t> plus_f16;
    std::vector<ggml_fp16_t> minus_f16;
    std::vector<ggml_fp16_t> pair_f16;
    std::vector<ggml_fp16_t> f16_io;
};

class backend_host_write_transactions {
public:
    explicit backend_host_write_transactions(const std::vector<lora_tensor_state> & tensors) {
        for (const auto & tensor : tensors) {
            if (tensor.weight == nullptr || tensor.weight->b_pair == nullptr || tensor.weight->b_pair->buffer == nullptr) {
                throw std::runtime_error("missing HTP B pair for " + tensor.name);
            }
            ggml_backend_buffer_t buffer = tensor.weight->b_pair->buffer;
            if (std::find(buffers.begin(), buffers.end(), buffer) == buffers.end()) {
                buffers.push_back(buffer);
            }
        }

        for (ggml_backend_buffer_t buffer : buffers) {
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buffer);
            ggml_backend_dev_t device = ggml_backend_buft_get_device(buft);
            ggml_backend_reg_t reg = device == nullptr ? nullptr : ggml_backend_dev_backend_reg(device);
            auto begin = reg == nullptr ? nullptr : (ggml_backend_buffer_host_write_t)
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_buffer_host_write_begin");
            auto end = reg == nullptr ? nullptr : (ggml_backend_buffer_host_write_t)
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_buffer_host_write_end");
            if (begin == nullptr || end == nullptr) {
                close();
                throw std::runtime_error("HTP backend does not support host-write transactions");
            }
            if (!begin(buffer)) {
                close();
                throw std::runtime_error("failed to begin HTP host-write transaction");
            }
            entries.push_back({ buffer, end });
        }
    }

    ~backend_host_write_transactions() {
        close();
    }

    void finish() {
        bool success = true;
        for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
            success = it->end(it->buffer) && success;
        }
        entries.clear();
        if (!success) {
            throw std::runtime_error("failed to publish HTP host-write transaction");
        }
    }

private:
    struct entry {
        ggml_backend_buffer_t buffer;
        ggml_backend_buffer_host_write_t end;
    };

    void close() noexcept {
        for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
            it->end(it->buffer);
        }
        entries.clear();
    }

    std::vector<ggml_backend_buffer_t> buffers;
    std::vector<entry> entries;
};

struct noise_plan {
    int step = -1;
    std::vector<std::vector<float>> tensors;
};

struct timed_noise_plan {
    noise_plan plan;
    int64_t elapsed_us = 0;
};

struct output_logits {
    std::vector<std::array<float, 2>> plus;
    std::vector<std::array<float, 2>> minus;
    std::vector<int32_t> final_indices;
    int32_t real_tokens = 0;
    int32_t padding_tokens = 0;
    int32_t backend_tokens = 0;
    int64_t batch_us = 0;
    int64_t elapsed_us = 0;
    int64_t logits_us = 0;
};

struct timed_loss {
    float value = 0.0f;
    int64_t elapsed_us = 0;
};

struct loss_snapshot {
    int32_t n_vocab = 0;
    std::vector<float> logits;
    std::vector<int32_t> labels;
};

struct step_timing {
    int64_t sample_us = 0;
    int64_t batch_us = 0;
    int64_t noise_prepare_us = 0;
    int64_t perturb_us = 0;
    int64_t upload_plus_us = 0;
    int64_t upload_minus_us = 0;
    int64_t upload_pair_us = 0;
    int64_t upload_us = 0;
    int64_t decode_plus_us = 0;
    int64_t decode_minus_us = 0;
    int64_t decode_pair_us = 0;
    int64_t decode_us = 0;
    int64_t logits_us = 0;
    int64_t loss_plus_us = 0;
    int64_t loss_minus_us = 0;
    int64_t loss_wait_us = 0;
    int64_t loss_work_us = 0;
    int64_t loss_us = 0;
    int64_t update_us = 0;
    int64_t pipeline_noise_work_us = 0;
    int64_t pipeline_wait_us = 0;
    int64_t step_wall_us = 0;
    int64_t tokens_plus = 0;
    int64_t tokens_minus = 0;
    int64_t tokens_pair = 0;
    int64_t tokens_total = 0;
    int64_t tokens_real = 0;
    int64_t tokens_padding = 0;
    int64_t tokens_backend = 0;
};

struct zo_forward_result {
    float loss_plus = 0.0f;
    float loss_minus = 0.0f;
    step_timing timing;
};

struct save_timing {
    int64_t save_us = 0;
    int64_t fresh_load_us = 0;
    int64_t wall_us = 0;
};

struct eval_result {
    int correct = 0;
    int total = 0;
    float loss = 0.0f;
    int64_t elapsed_us = 0;
    int64_t tokens_real = 0;
    int64_t tokens_padding = 0;
    int64_t tokens_backend = 0;
};

struct serialized_tensor {
    std::string name;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    std::vector<ggml_fp16_t> data;
};

using adapter_ptr = std::unique_ptr<llama_adapter_lora, decltype(&llama_adapter_lora_free)>;

volatile std::sig_atomic_t g_stop_requested = 0;

void signal_handler(int) {
    g_stop_requested = 1;
}

bool ends_with(const std::string & value, const char * suffix) {
    const size_t suffix_len = std::strlen(suffix);
    return value.size() >= suffix_len && value.compare(value.size() - suffix_len, suffix_len, suffix) == 0;
}

bool is_target_name(const std::string & name) {
    static const char * suffixes[] = {
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    };
    for (const char * suffix : suffixes) {
        if (ends_with(name, suffix)) {
            return true;
        }
    }
    return false;
}

std::string trim(std::string value) {
    auto is_space = [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\r' || c == '\n';
    };
    while (!value.empty() && is_space((unsigned char) value.back())) {
        value.pop_back();
    }
    size_t first = 0;
    while (first < value.size() && is_space((unsigned char) value[first])) {
        ++first;
    }
    if (first != 0) {
        value.erase(0, first);
    }
    return value;
}

bool split_arg(const std::string & arg, const char * key, std::string & value) {
    const std::string prefix = std::string(key) + "=";
    if (arg == key) {
        value.clear();
        return true;
    }
    if (arg.compare(0, prefix.size(), prefix) == 0) {
        value = arg.substr(prefix.size());
        return true;
    }
    return false;
}

template <typename T>
T parse_number(const std::string & value, const char * name);

template <>
int parse_number<int>(const std::string & value, const char * name) {
    size_t end = 0;
    int result;
    try {
        result = std::stoi(value, &end);
    } catch (const std::exception &) {
        throw std::runtime_error(std::string("invalid integer for ") + name + ": " + value);
    }
    if (end != value.size()) {
        throw std::runtime_error(std::string("invalid integer for ") + name + ": " + value);
    }
    return result;
}

template <>
float parse_number<float>(const std::string & value, const char * name) {
    size_t end = 0;
    float result;
    try {
        result = std::stof(value, &end);
    } catch (const std::exception &) {
        throw std::runtime_error(std::string("invalid float for ") + name + ": " + value);
    }
    if (end != value.size() || !std::isfinite(result)) {
        throw std::runtime_error(std::string("invalid float for ") + name + ": " + value);
    }
    return result;
}

template <>
uint64_t parse_number<uint64_t>(const std::string & value, const char * name) {
    size_t end = 0;
    uint64_t result;
    try {
        result = std::stoull(value, &end);
    } catch (const std::exception &) {
        throw std::runtime_error(std::string("invalid integer for ") + name + ": " + value);
    }
    if (end != value.size()) {
        throw std::runtime_error(std::string("invalid integer for ") + name + ": " + value);
    }
    return result;
}

bool parse_bool(const std::string & value, const char * name) {
    if (value == "true" || value == "1" || value == "on" || value == "yes") {
        return true;
    }
    if (value == "false" || value == "0" || value == "off" || value == "no") {
        return false;
    }
    throw std::runtime_error(std::string("invalid boolean for ") + name + ": " + value);
}

void print_usage(int, char ** argv) {
    std::fprintf(stderr,
        "\nusage: %s -m MODEL --train-data train.tsv [common llama.cpp options] [options]\n\n"
        "ZO-LoRA options:\n"
        "  --mode cpu|coop                 execution backend (default: cpu)\n"
        "  --lora-exec runtime|fused-htp  LoRA graph path\n"
        "  --pipeline BOOL                 overlap the next NoisePlan with HTP decode\n"
        "  --antithetic BOOL               run paired +/- perturbations in one HTP decode\n"
        "  --warmup-steps N                untimed deterministic warmup steps (default: 2)\n"
        "  --train-data PATH               strict GLUE SST-2 training TSV (required)\n"
        "  --eval-data PATH                validation TSV (required unless evaluation is disabled)\n"
        "  --max-train N                   maximum training rows (default: 1000)\n"
        "  --max-eval N                    maximum validation rows (default: 1000)\n"
        "  --steps N                       ZO steps (default: 2000)\n"
        "  --eval-step N                   validation interval, -1 disables (default: 50)\n"
        "  --batch-size N                  training batch size (default: 4)\n"
        "  --seq-len N                     maximum prompt length (default: 128)\n"
        "  --epsilon F                     perturbation scale (default: 1e-2)\n"
        "  --lr F                          ZO learning rate (default: 5e-5)\n"
        "  --seed N                        deterministic seed (default: 1337)\n"
        "  --rank 8|16|24|32               generated Adapter rank (default: 8)\n"
        "  --alpha F                       generated Adapter alpha (default: 16)\n"
        "  --lora-out PATH                 output Adapter GGUF path\n"
        "  --hexagon-arch auto|v68|v69|v73|v75|v79|v81\n"
        "\nIf --lora is omitted, an F16 ZO-LoRA Adapter is generated from MODEL.\n\n",
        argv[0]);
}

void prescan_mode(int argc, char ** argv, options & opts) {
    for (int i = 1; i < argc; ++i) {
        std::string value;
        if (!split_arg(argv[i], "--mode", value)) {
            continue;
        }
        if (value.empty()) {
            if (++i >= argc) {
                throw std::runtime_error("missing value for --mode");
            }
            value = argv[i];
        }
        if (value == "cpu") {
            opts.mode = run_mode::CPU;
        } else if (value == "coop") {
            opts.mode = run_mode::COOP;
        } else {
            throw std::runtime_error("invalid --mode: " + value);
        }
    }
}

std::vector<char *> parse_custom_args(int argc, char ** argv, options & opts) {
    std::vector<char *> filtered;
    filtered.reserve((size_t) argc);
    filtered.push_back(argv[0]);

    auto set_string = [](std::string & dst, const std::string & value, const char * name) {
        if (value.empty()) {
            throw std::runtime_error(std::string("empty value for ") + name);
        }
        dst = value;
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        std::string value;
        auto take_value = [&](const char * name) {
            if (!value.empty()) {
                return value;
            }
            if (++i >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return std::string(argv[i]);
        };

        if (split_arg(arg, "--mode", value)) {
            value = take_value("--mode");
            opts.mode = value == "cpu" ? run_mode::CPU : value == "coop" ? run_mode::COOP :
                throw std::runtime_error("invalid --mode: " + value);
        } else if (split_arg(arg, "--lora-exec", value)) {
            value = take_value("--lora-exec");
            if (value == "runtime") {
                opts.lora_exec = lora_exec_mode::RUNTIME;
            } else if (value == "fused-htp") {
                opts.lora_exec = lora_exec_mode::FUSED_HTP;
            } else {
                throw std::runtime_error("invalid --lora-exec: " + value);
            }
            opts.lora_exec_set = true;
        } else if (split_arg(arg, "--pipeline", value)) {
            opts.pipeline = parse_bool(take_value("--pipeline"), "--pipeline");
        } else if (split_arg(arg, "--antithetic", value)) {
            opts.antithetic = parse_bool(take_value("--antithetic"), "--antithetic");
        } else if (split_arg(arg, "--warmup-steps", value)) {
            opts.warmup_steps = parse_number<int>(take_value("--warmup-steps"), "--warmup-steps");
        } else if (split_arg(arg, "--train-data", value)) {
            set_string(opts.train_path, take_value("--train-data"), "--train-data");
        } else if (split_arg(arg, "--eval-data", value)) {
            set_string(opts.eval_path, take_value("--eval-data"), "--eval-data");
        } else if (split_arg(arg, "--max-train", value)) {
            opts.max_train = parse_number<int>(take_value("--max-train"), "--max-train");
        } else if (split_arg(arg, "--max-eval", value)) {
            opts.max_eval = parse_number<int>(take_value("--max-eval"), "--max-eval");
        } else if (split_arg(arg, "--steps", value)) {
            opts.steps = parse_number<int>(take_value("--steps"), "--steps");
        } else if (split_arg(arg, "--eval-step", value)) {
            opts.eval_step = parse_number<int>(take_value("--eval-step"), "--eval-step");
        } else if (split_arg(arg, "--batch-size", value)) {
            opts.batch_size = parse_number<int>(take_value("--batch-size"), "--batch-size");
        } else if (split_arg(arg, "--seq-len", value)) {
            opts.seq_len = parse_number<int>(take_value("--seq-len"), "--seq-len");
        } else if (split_arg(arg, "--epsilon", value)) {
            opts.epsilon = parse_number<float>(take_value("--epsilon"), "--epsilon");
        } else if (split_arg(arg, "--lr", value)) {
            opts.learning_rate = parse_number<float>(take_value("--lr"), "--lr");
        } else if (split_arg(arg, "--seed", value)) {
            opts.seed = parse_number<uint64_t>(take_value("--seed"), "--seed");
        } else if (split_arg(arg, "--rank", value)) {
            opts.rank = parse_number<int>(take_value("--rank"), "--rank");
        } else if (split_arg(arg, "--alpha", value)) {
            opts.alpha = parse_number<float>(take_value("--alpha"), "--alpha");
        } else if (split_arg(arg, "--lora-out", value)) {
            set_string(opts.lora_out, take_value("--lora-out"), "--lora-out");
        } else if (split_arg(arg, "--hexagon-arch", value)) {
            set_string(opts.hexagon_arch, take_value("--hexagon-arch"), "--hexagon-arch");
        } else if (opts.mode == run_mode::CPU && (arg == "-dev" || arg == "--device")) {
            if (++i >= argc) {
                throw std::runtime_error("missing value for --device");
            }
        } else if (opts.mode == run_mode::CPU &&
                   (arg.rfind("--device=", 0) == 0 || arg.rfind("-dev=", 0) == 0)) {
            continue;
        } else {
            filtered.push_back(argv[i]);
        }
    }
    return filtered;
}

void validate_options(options & opts) {
    if (!opts.lora_exec_set) {
        opts.lora_exec = opts.mode == run_mode::COOP ? lora_exec_mode::FUSED_HTP : lora_exec_mode::RUNTIME;
    }
    if (opts.steps <= 0 || opts.batch_size <= 0 || opts.max_train <= 0 || opts.max_eval <= 0) {
        throw std::runtime_error("steps, batch-size, max-train and max-eval must be positive");
    }
    if (opts.warmup_steps < 0) {
        throw std::runtime_error("warmup-steps must be non-negative");
    }
    if (opts.eval_step == 0 || opts.eval_step < -1) {
        throw std::runtime_error("eval-step must be positive or -1");
    }
    if (opts.train_path.empty()) {
        throw std::runtime_error("--train-data is required");
    }
    if (opts.eval_step != -1 && opts.eval_path.empty()) {
        throw std::runtime_error("--eval-data is required unless --eval-step is -1");
    }
    if (opts.seq_len < 4) {
        throw std::runtime_error("seq-len must be at least 4");
    }
    if (!(opts.epsilon > 0.0f) || !(opts.learning_rate > 0.0f) || !(opts.alpha > 0.0f)) {
        throw std::runtime_error("epsilon, lr and alpha must be positive");
    }
    if (opts.rank != 8 && opts.rank != 16 && opts.rank != 24 && opts.rank != 32) {
        throw std::runtime_error("rank must be one of 8, 16, 24, 32");
    }
    if (opts.mode == run_mode::CPU && opts.lora_exec != lora_exec_mode::RUNTIME) {
        throw std::runtime_error("CPU mode only supports --lora-exec runtime");
    }
    if (opts.mode == run_mode::COOP && opts.lora_exec != lora_exec_mode::FUSED_HTP) {
        throw std::runtime_error("coop mode only supports --lora-exec fused-htp");
    }
    if ((opts.pipeline || opts.antithetic) && opts.mode != run_mode::COOP) {
        throw std::runtime_error("pipeline and antithetic modes require --mode coop");
    }
    static const char * supported_arches[] = { "auto", "v68", "v69", "v73", "v75", "v79", "v81" };
    bool valid_arch = false;
    for (const char * arch : supported_arches) {
        valid_arch = valid_arch || opts.hexagon_arch == arch;
    }
    if (!valid_arch) {
        throw std::runtime_error("hexagon-arch must be auto, v68, v69, v73, v75, v79 or v81");
    }
}

void set_env(const char * key, const char * value) {
#if defined(_WIN32)
    _putenv_s(key, value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv(key);
    } else {
        setenv(key, value, 1);
    }
#endif
}

void configure_backend_environment(const options & opts) {
    set_env("GGML_HEXAGON_ARCH", opts.hexagon_arch == "auto" ? nullptr : opts.hexagon_arch.c_str());
}

uint64_t mix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

uint64_t stable_hash(const std::string & value) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

class stable_rng {
public:
    explicit stable_rng(uint64_t seed) : state(seed) {}

    uint64_t next_u64() {
        state = mix64(state);
        return state;
    }

    double uniform_open() {
        return ((next_u64() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    }

    float uniform(float low, float high) {
        return low + (high - low) * (float) uniform_open();
    }

    size_t uniform_index(size_t upper) {
        if (upper == 0) {
            throw std::runtime_error("cannot sample from an empty range");
        }
        return (size_t) (next_u64() % upper);
    }

    float normal() {
        if (has_spare) {
            has_spare = false;
            return spare;
        }
        const double radius = std::sqrt(-2.0 * std::log(uniform_open()));
        const double angle = 6.28318530717958647692 * uniform_open();
        spare = (float) (radius * std::sin(angle));
        has_spare = true;
        return (float) (radius * std::cos(angle));
    }

private:
    uint64_t state;
    bool has_spare = false;
    float spare = 0.0f;
};

uint64_t domain_seed(uint64_t seed, uint64_t domain, uint64_t item) {
    return mix64(seed ^ mix64(domain) ^ mix64(item));
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int32_t count = llama_tokenize(vocab, text.data(), (int32_t) text.size(), nullptr, 0, false, false);
    if (count == INT32_MIN) {
        throw std::runtime_error("tokenization overflow");
    }
    if (count < 0) {
        count = -count;
    }
    std::vector<llama_token> result((size_t) count);
    const int32_t actual = llama_tokenize(
        vocab, text.data(), (int32_t) text.size(), result.data(), count, false, false);
    if (actual < 0) {
        throw std::runtime_error("tokenization failed");
    }
    result.resize((size_t) actual);
    return result;
}

sst2_sample make_sample(
        const llama_vocab * vocab,
        const std::string & sentence,
        int label,
        int seq_len,
        size_t line_number) {
    const std::string prompt = sentence + " It was";
    const auto body = tokenize(vocab, sentence);
    const auto raw_prompt = tokenize(vocab, prompt);
    const auto bad = tokenize(vocab, prompt + " terrible");
    const auto good = tokenize(vocab, prompt + " great");

    size_t body_prefix = 0;
    while (body_prefix < body.size() && body_prefix < raw_prompt.size() &&
           body[body_prefix] == raw_prompt[body_prefix]) {
        ++body_prefix;
    }
    const size_t suffix_tokens = raw_prompt.size() - body_prefix;
    if (body_prefix == 0 || suffix_tokens == 0) {
        throw std::runtime_error("SST-2 line " + std::to_string(line_number) +
            ": tokenizer cannot preserve a complete ' It was' suffix and one body token");
    }
    auto check_candidate = [&](const std::vector<llama_token> & candidate, const char * name) {
        if (candidate.size() != raw_prompt.size() + 1 ||
            !std::equal(raw_prompt.begin(), raw_prompt.end(), candidate.begin())) {
            throw std::runtime_error("SST-2 line " + std::to_string(line_number) +
                ": ' " + name + "' must append exactly one token");
        }
    };
    check_candidate(bad, "terrible");
    check_candidate(good, "great");
    if (bad.back() == good.back()) {
        throw std::runtime_error("SST-2 line " + std::to_string(line_number) +
            ": verbalizer tokens must differ");
    }

    const llama_token bos = llama_vocab_bos(vocab);
    if (bos == LLAMA_TOKEN_NULL) {
        throw std::runtime_error("SST-2 construction requires a BOS token");
    }
    const size_t bos_count = 1;
    const size_t minimum = bos_count + suffix_tokens + 1u;
    if ((size_t) seq_len < minimum) {
        throw std::runtime_error("seq-len is too small for BOS, one body token and the SST-2 suffix");
    }

    std::vector<llama_token> kept = raw_prompt;
    const size_t capacity = (size_t) seq_len - bos_count;
    if (kept.size() > capacity) {
        const size_t keep_body = capacity - suffix_tokens;
        kept.erase(kept.begin(), kept.begin() + (std::ptrdiff_t) (body_prefix - keep_body));
    }

    sst2_sample sample;
    sample.tokens.reserve(kept.size() + bos_count);
    sample.tokens.push_back(bos);
    sample.tokens.insert(sample.tokens.end(), kept.begin(), kept.end());
    sample.class_tokens = { bad.back(), good.back() };
    sample.label = label;
    return sample;
}

std::vector<sst2_sample> load_sst2(
        const std::string & path,
        const llama_vocab * vocab,
        int max_samples,
        int seq_len) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open SST-2 TSV: " + path);
    }

    std::vector<sst2_sample> samples;
    samples.reserve((size_t) max_samples);
    std::string line;
    size_t line_number = 0;
    bool first_line = true;
    while ((int) samples.size() < max_samples && std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (first_line && line.size() >= 3 &&
            (unsigned char) line[0] == 0xef && (unsigned char) line[1] == 0xbb && (unsigned char) line[2] == 0xbf) {
            line.erase(0, 3);
        }

        const size_t tab = line.find('\t');
        if (tab == std::string::npos || line.find('\t', tab + 1) != std::string::npos) {
            throw std::runtime_error("SST-2 line " + std::to_string(line_number) + ": expected exactly two TSV columns");
        }
        const std::string sentence = trim(line.substr(0, tab));
        const std::string label_text = trim(line.substr(tab + 1));
        if (first_line && sentence == "sentence" && label_text == "label") {
            first_line = false;
            continue;
        }
        first_line = false;
        if (sentence.empty()) {
            throw std::runtime_error("SST-2 line " + std::to_string(line_number) + ": empty sentence");
        }
        if (label_text != "0" && label_text != "1") {
            throw std::runtime_error("SST-2 line " + std::to_string(line_number) + ": label must be 0 or 1");
        }
        samples.push_back(make_sample(vocab, sentence, label_text[0] - '0', seq_len, line_number));
    }
    if (input.bad()) {
        throw std::runtime_error("failed while reading SST-2 TSV: " + path);
    }
    if (samples.empty()) {
        throw std::runtime_error("SST-2 TSV contains no samples: " + path);
    }
    return samples;
}

std::vector<const sst2_sample *> training_batch(
        const std::vector<sst2_sample> & samples,
        int batch_size,
        int step,
        uint64_t seed) {
    stable_rng rng(domain_seed(seed, 0x44415441ULL, (uint64_t) step));
    std::vector<const sst2_sample *> result;
    result.reserve((size_t) batch_size);
    for (int i = 0; i < batch_size; ++i) {
        result.push_back(&samples[rng.uniform_index(samples.size())]);
    }
    return result;
}

timed_loss measure_snapshot_loss(const loss_snapshot & snapshot) {
    const int64_t started = ggml_time_us();
    if (snapshot.n_vocab <= 0 || snapshot.labels.empty() ||
        snapshot.logits.size() != snapshot.labels.size() * (size_t) snapshot.n_vocab) {
        throw std::runtime_error("invalid full-vocabulary logits snapshot");
    }
    timed_loss result;
    double sum = 0.0;
    for (size_t sample = 0; sample < snapshot.labels.size(); ++sample) {
        const int32_t label = snapshot.labels[sample];
        if (label < 0 || label >= snapshot.n_vocab) {
            throw std::runtime_error("invalid full-vocabulary loss label");
        }
        sum += zo_lora::vocabulary_cross_entropy(
            snapshot.logits.data() + sample * (size_t) snapshot.n_vocab,
            snapshot.n_vocab,
            label);
    }
    result.value = (float) (sum / snapshot.labels.size());
    result.elapsed_us = ggml_time_us() - started;
    return result;
}

llama_token padding_token(const llama_vocab * vocab) {
    llama_token token = llama_vocab_pad(vocab);
    if (token == LLAMA_TOKEN_NULL) {
        token = llama_vocab_eos(vocab);
    }
    if (token == LLAMA_TOKEN_NULL) {
        token = llama_vocab_bos(vocab);
    }
    return token == LLAMA_TOKEN_NULL ? 0 : token;
}

output_logits run_forward(
        llama_context * ctx,
        llama_adapter_lora * adapter,
        llama_batch & batch,
        const std::vector<const sst2_sample *> & samples,
        bool paired,
        int32_t lora_side,
        bool right_padding,
        llama_token pad_token) {
    output_logits result;
    const int64_t batch_started = ggml_time_us();
    common_batch_clear(batch);
    std::vector<size_t> lengths;
    lengths.reserve(samples.size());
    for (const sst2_sample * sample : samples) {
        lengths.push_back(sample->tokens.size());
    }
    const zo_lora::batch_layout layout = zo_lora::make_batch_layout(lengths, paired, right_padding);
    result.final_indices.reserve(layout.sequences.size());
    for (const zo_lora::batch_sequence_layout & sequence : layout.sequences) {
        const auto & tokens = samples[sequence.sample]->tokens;
        int32_t final_index = -1;
        for (size_t position = 0; position < tokens.size(); ++position) {
            final_index = batch.n_tokens;
            common_batch_add(batch, tokens[position], (llama_pos) position,
                { (llama_seq_id) sequence.sequence_id }, position + 1 == tokens.size());
        }
        for (size_t position = tokens.size(); position < sequence.backend_tokens; ++position) {
            common_batch_add(batch, pad_token, (llama_pos) position,
                { (llama_seq_id) sequence.sequence_id }, false);
        }
        result.final_indices.push_back(final_index);
    }
    if (batch.n_tokens <= 0 || batch.n_tokens != layout.backend_tokens ||
        (paired && batch.n_tokens % 2 != 0)) {
        throw std::runtime_error("invalid decode batch shape");
    }
    if ((paired && lora_side != -1) || (!paired && (lora_side < 0 || lora_side > 1))) {
        throw std::runtime_error("invalid decode LoRA side");
    }

    llama_memory_clear(llama_get_memory(ctx), false);
    result.batch_us = ggml_time_us() - batch_started;
    if (adapter->zo_fused_htp) {
        llama_adapter_lora_set_zo_side(adapter, lora_side);
    }
    const int64_t started = ggml_time_us();
    const int32_t status = llama_decode(ctx, batch);
    if (status != 0) {
        throw std::runtime_error("llama_decode failed with status " + std::to_string(status));
    }
    llama_synchronize(ctx);

    result.elapsed_us = ggml_time_us() - started;
    result.real_tokens = (int32_t) layout.real_tokens;
    result.padding_tokens = (int32_t) layout.padding_tokens;
    result.backend_tokens = (int32_t) layout.backend_tokens;
    result.plus.resize(samples.size());
    if (paired) {
        result.minus.resize(samples.size());
    }
    auto copy_side = [&](size_t side, std::vector<std::array<float, 2>> & destination) {
        for (size_t i = 0; i < samples.size(); ++i) {
            const float * row = llama_get_logits_ith(ctx, result.final_indices[side * samples.size() + i]);
            if (row == nullptr) {
                throw std::runtime_error("failed to retrieve a requested logits row");
            }
            destination[i] = {
                row[samples[i]->class_tokens[0]],
                row[samples[i]->class_tokens[1]],
            };
        }
    };
    const int64_t logits_started = ggml_time_us();
    copy_side(0, result.plus);
    if (paired) {
        copy_side(1, result.minus);
    }
    result.logits_us = ggml_time_us() - logits_started;
    return result;
}

int32_t target_token(const sst2_sample & sample, int32_t n_vocab) {
    if (sample.label < 0 || (size_t) sample.label >= sample.class_tokens.size()) {
        throw std::runtime_error("invalid SST-2 label");
    }
    const llama_token token = sample.class_tokens[(size_t) sample.label];
    if (token < 0 || token >= n_vocab) {
        throw std::runtime_error("invalid SST-2 target token");
    }
    return token;
}

timed_loss measure_current_loss(
        llama_context * ctx,
        const output_logits & output,
        size_t side,
        const std::vector<const sst2_sample *> & samples,
        int32_t n_vocab) {
    const int64_t started = ggml_time_us();
    const size_t offset = side * samples.size();
    if (n_vocab <= 0 || samples.empty() || offset + samples.size() > output.final_indices.size()) {
        throw std::runtime_error("invalid current logits for full-vocabulary loss");
    }
    double sum = 0.0;
    for (size_t sample = 0; sample < samples.size(); ++sample) {
        const float * row = llama_get_logits_ith(ctx, output.final_indices[offset + sample]);
        if (row == nullptr) {
            throw std::runtime_error("failed to retrieve full-vocabulary logits row");
        }
        sum += zo_lora::vocabulary_cross_entropy(
            row, n_vocab, target_token(*samples[sample], n_vocab));
    }
    timed_loss result;
    result.value = (float) (sum / samples.size());
    result.elapsed_us = ggml_time_us() - started;
    return result;
}

loss_snapshot copy_current_loss_logits(
        llama_context * ctx,
        const output_logits & output,
        size_t side,
        const std::vector<const sst2_sample *> & samples,
        int32_t n_vocab) {
    const size_t offset = side * samples.size();
    if (n_vocab <= 0 || samples.empty() || offset + samples.size() > output.final_indices.size()) {
        throw std::runtime_error("invalid current logits for full-vocabulary snapshot");
    }
    loss_snapshot snapshot;
    snapshot.n_vocab = n_vocab;
    snapshot.logits.resize(samples.size() * (size_t) n_vocab);
    snapshot.labels.resize(samples.size());
    for (size_t sample = 0; sample < samples.size(); ++sample) {
        const float * row = llama_get_logits_ith(ctx, output.final_indices[offset + sample]);
        if (row == nullptr) {
            throw std::runtime_error("failed to snapshot full-vocabulary logits row");
        }
        std::memcpy(snapshot.logits.data() + sample * (size_t) n_vocab,
            row, (size_t) n_vocab * sizeof(float));
        snapshot.labels[sample] = target_token(*samples[sample], n_vocab);
    }
    return snapshot;
}

eval_result evaluate(
        llama_context * ctx,
        llama_adapter_lora * adapter,
        llama_batch & batch,
        const std::vector<sst2_sample> & dataset,
        int batch_size,
        int32_t n_vocab,
        bool right_padding,
        llama_token pad_token) {
    eval_result result;
    const int64_t started = ggml_time_us();
    double loss_sum = 0.0;
    for (size_t start = 0; start < dataset.size(); start += (size_t) batch_size) {
        if (g_stop_requested) {
            break;
        }
        const size_t end = std::min(dataset.size(), start + (size_t) batch_size);
        std::vector<const sst2_sample *> samples;
        samples.reserve(end - start);
        for (size_t i = start; i < end; ++i) {
            samples.push_back(&dataset[i]);
        }
        const output_logits output = run_forward(ctx, adapter, batch, samples, false, 0, right_padding, pad_token);
        const timed_loss batch_loss = measure_current_loss(ctx, output, 0, samples, n_vocab);
        loss_sum += (double) batch_loss.value * samples.size();
        result.tokens_real += output.real_tokens;
        result.tokens_padding += output.padding_tokens;
        result.tokens_backend += output.backend_tokens;
        for (size_t i = 0; i < samples.size(); ++i) {
            const auto & logits = output.plus[i];
            const int predicted = logits[1] > logits[0] ? 1 : 0;
            result.correct += predicted == samples[i]->label;
            ++result.total;
        }
    }
    result.elapsed_us = ggml_time_us() - started;
    result.loss = result.total > 0 ? (float) (loss_sum / result.total) : 0.0f;
    return result;
}

noise_plan make_noise_plan(const std::vector<lora_tensor_state> & tensors, int step, uint64_t seed) {
    noise_plan plan;
    plan.step = step;
    plan.tensors.resize(tensors.size());
    for (size_t tensor_index = 0; tensor_index < tensors.size(); ++tensor_index) {
        auto & noise = plan.tensors[tensor_index];
        noise.resize(tensors[tensor_index].master.size());
        stable_rng rng(domain_seed(seed, 0x4e4f495345ULL ^ stable_hash(tensors[tensor_index].name), (uint64_t) step));
        for (float & value : noise) {
            value = rng.normal();
        }
    }
    return plan;
}

timed_noise_plan measure_noise_plan(
        const std::vector<lora_tensor_state> & tensors,
        int step,
        uint64_t seed) {
    const int64_t started = ggml_time_us();
    timed_noise_plan result;
    result.plan = make_noise_plan(tensors, step, seed);
    result.elapsed_us = ggml_time_us() - started;
    return result;
}

void upload_f16(ggml_tensor * tensor, const std::vector<float> & source, std::vector<ggml_fp16_t> & io) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F16 || ggml_nelements(tensor) != (int64_t) source.size()) {
        throw std::runtime_error("invalid F16 LoRA tensor upload");
    }
    io.resize(source.size());
    ggml_fp32_to_fp16_row(source.data(), io.data(), (int64_t) source.size());
    ggml_backend_tensor_set(tensor, io.data(), 0, io.size() * sizeof(io[0]));
}

void upload_f16(ggml_tensor * tensor, const std::vector<ggml_fp16_t> & source) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F16 || ggml_nelements(tensor) != (int64_t) source.size()) {
        throw std::runtime_error("invalid encoded F16 LoRA tensor upload");
    }
    ggml_backend_tensor_set(tensor, source.data(), 0, source.size() * sizeof(source[0]));
}

void prepare_perturbations(
        std::vector<lora_tensor_state> & tensors,
        const noise_plan & plan,
        float epsilon,
        bool htp) {
    if (plan.tensors.size() != tensors.size()) {
        throw std::runtime_error("NoisePlan tensor count mismatch");
    }
    for (size_t tensor_index = 0; tensor_index < tensors.size(); ++tensor_index) {
        auto & tensor = tensors[tensor_index];
        const auto & noise = plan.tensors[tensor_index];
        if (noise.size() != tensor.master.size()) {
            throw std::runtime_error("NoisePlan tensor shape mismatch");
        }
        if (htp) {
            const size_t side_stride = (size_t) tensor.padded_output_dim*(size_t) tensor.rank;
            tensor.pair_f16.assign(2*side_stride, ggml_fp32_to_fp16(0.0f));
            tensor.plus_f16.clear();
            tensor.minus_f16.clear();
            for (int64_t output = 0; output < tensor.output_dim; ++output) {
                for (int64_t rank = 0; rank < tensor.rank; ++rank) {
                    const size_t source = (size_t) output*(size_t) tensor.rank + (size_t) rank;
                    const size_t destination = (size_t) rank*(size_t) tensor.padded_output_dim + (size_t) output;
                    tensor.pair_f16[destination] = ggml_fp32_to_fp16(tensor.master[source] + epsilon*noise[source]);
                    tensor.pair_f16[side_stride + destination] = ggml_fp32_to_fp16(tensor.master[source] - epsilon*noise[source]);
                }
            }
        } else {
            tensor.plus_f16.resize(tensor.master.size());
            tensor.minus_f16.resize(tensor.master.size());
            tensor.pair_f16.clear();
            for (size_t index = 0; index < tensor.master.size(); ++index) {
                tensor.plus_f16[index] = ggml_fp32_to_fp16(tensor.master[index] + epsilon*noise[index]);
                tensor.minus_f16[index] = ggml_fp32_to_fp16(tensor.master[index] - epsilon*noise[index]);
            }
        }
    }
}

void upload_serial(std::vector<lora_tensor_state> & tensors, bool plus) {
    for (auto & tensor : tensors) {
        upload_f16(tensor.weight->b, plus ? tensor.plus_f16 : tensor.minus_f16);
    }
}

void upload_pair(std::vector<lora_tensor_state> & tensors) {
    backend_host_write_transactions transactions(tensors);
    for (auto & tensor : tensors) {
        if (tensor.weight->b_pair == nullptr || tensor.weight->b_pair->type != GGML_TYPE_F16 ||
            ggml_nelements(tensor.weight->b_pair) != (int64_t) tensor.pair_f16.size()) {
            throw std::runtime_error("missing HTP B pair for " + tensor.name);
        }
        ggml_backend_tensor_set(tensor.weight->b_pair, tensor.pair_f16.data(), 0,
            tensor.pair_f16.size() * sizeof(tensor.pair_f16[0]));
    }
    transactions.finish();
}

void upload_master(std::vector<lora_tensor_state> & tensors, bool htp) {
    std::unique_ptr<backend_host_write_transactions> transactions;
    if (htp) {
        transactions = std::make_unique<backend_host_write_transactions>(tensors);
    }
    for (auto & tensor : tensors) {
        if (htp) {
            if (tensor.weight->b_pair == nullptr) {
                throw std::runtime_error("missing HTP B pair for " + tensor.name);
            }
            const size_t side_stride = (size_t) tensor.padded_output_dim*(size_t) tensor.rank;
            tensor.pair_f16.assign(2*side_stride, ggml_fp32_to_fp16(0.0f));
            for (int64_t output = 0; output < tensor.output_dim; ++output) {
                for (int64_t rank = 0; rank < tensor.rank; ++rank) {
                    const size_t source = (size_t) output*(size_t) tensor.rank + (size_t) rank;
                    const size_t destination = (size_t) rank*(size_t) tensor.padded_output_dim + (size_t) output;
                    const ggml_fp16_t value = ggml_fp32_to_fp16(tensor.master[source]);
                    tensor.pair_f16[destination] = value;
                    tensor.pair_f16[side_stride + destination] = value;
                }
            }
            ggml_backend_tensor_set(tensor.weight->b_pair, tensor.pair_f16.data(), 0,
                tensor.pair_f16.size() * sizeof(tensor.pair_f16[0]));
        } else {
            upload_f16(tensor.weight->b, tensor.master, tensor.f16_io);
        }
    }
    if (transactions != nullptr) {
        transactions->finish();
    }
}

void apply_update(std::vector<lora_tensor_state> & tensors, const noise_plan & plan, float scale) {
    if (plan.tensors.size() != tensors.size()) {
        throw std::runtime_error("NoisePlan tensor count mismatch during update");
    }
    for (size_t tensor_index = 0; tensor_index < tensors.size(); ++tensor_index) {
        auto & master = tensors[tensor_index].master;
        const auto & noise = plan.tensors[tensor_index];
        for (size_t i = 0; i < master.size(); ++i) {
            master[i] += scale * noise[i];
        }
    }
}

zo_forward_result run_zo_forwards(
        std::vector<lora_tensor_state> & tensors,
        llama_context * ctx,
        llama_adapter_lora * adapter,
        llama_batch & batch,
        const std::vector<const sst2_sample *> & samples,
        bool antithetic,
        bool htp,
        int32_t n_vocab,
        llama_token pad_token) {
    zo_forward_result result;
    if (htp) {
        const int64_t started = ggml_time_us();
        upload_pair(tensors);
        result.timing.upload_pair_us = ggml_time_us() - started;
    }
    if (antithetic) {
        const output_logits output = run_forward(ctx, adapter, batch, samples, true, -1, true, pad_token);
        result.timing.batch_us = output.batch_us;
        result.timing.decode_pair_us = output.elapsed_us;
        result.timing.logits_us = output.logits_us;
        result.timing.tokens_pair = output.backend_tokens;
        result.timing.tokens_real += output.real_tokens;
        result.timing.tokens_padding += output.padding_tokens;
        result.timing.tokens_backend += output.backend_tokens;

        const timed_loss plus_loss = measure_current_loss(ctx, output, 0, samples, n_vocab);
        const timed_loss minus_loss = measure_current_loss(ctx, output, 1, samples, n_vocab);
        result.loss_plus = plus_loss.value;
        result.loss_minus = minus_loss.value;
        result.timing.loss_plus_us = plus_loss.elapsed_us;
        result.timing.loss_minus_us = minus_loss.elapsed_us;
    } else {
        int64_t started = ggml_time_us();
        if (!htp) {
            upload_serial(tensors, true);
            result.timing.upload_plus_us = ggml_time_us() - started;
        }

        output_logits plus_output = run_forward(ctx, adapter, batch, samples, false, 0, htp, pad_token);
        result.timing.batch_us += plus_output.batch_us;
        result.timing.decode_plus_us = plus_output.elapsed_us;
        result.timing.logits_us += plus_output.logits_us;
        result.timing.tokens_plus = plus_output.backend_tokens;
        result.timing.tokens_real += plus_output.real_tokens;
        result.timing.tokens_padding += plus_output.padding_tokens;
        result.timing.tokens_backend += plus_output.backend_tokens;
        started = ggml_time_us();
        loss_snapshot plus_snapshot = copy_current_loss_logits(ctx, plus_output, 0, samples, n_vocab);
        result.timing.logits_us += ggml_time_us() - started;
        auto plus_loss_future = std::async(std::launch::async,
            [snapshot = std::move(plus_snapshot)]() {
                return measure_snapshot_loss(snapshot);
            });

        started = ggml_time_us();
        if (!htp) {
            upload_serial(tensors, false);
            result.timing.upload_minus_us = ggml_time_us() - started;
        }

        const output_logits minus_output = run_forward(ctx, adapter, batch, samples, false, 1, htp, pad_token);
        result.timing.batch_us += minus_output.batch_us;
        result.timing.decode_minus_us = minus_output.elapsed_us;
        result.timing.logits_us += minus_output.logits_us;
        result.timing.tokens_minus = minus_output.backend_tokens;
        result.timing.tokens_real += minus_output.real_tokens;
        result.timing.tokens_padding += minus_output.padding_tokens;
        result.timing.tokens_backend += minus_output.backend_tokens;
        const timed_loss minus_loss = measure_current_loss(ctx, minus_output, 0, samples, n_vocab);

        const int64_t loss_wait_started = ggml_time_us();
        const timed_loss plus_loss = plus_loss_future.get();
        result.timing.loss_wait_us = ggml_time_us() - loss_wait_started;
        result.loss_plus = plus_loss.value;
        result.loss_minus = minus_loss.value;
        result.timing.loss_plus_us = plus_loss.elapsed_us;
        result.timing.loss_minus_us = minus_loss.elapsed_us;
    }

    result.timing.upload_us = result.timing.upload_plus_us +
        result.timing.upload_minus_us + result.timing.upload_pair_us;
    result.timing.decode_us = result.timing.decode_plus_us +
        result.timing.decode_minus_us + result.timing.decode_pair_us;
    result.timing.loss_work_us = result.timing.loss_plus_us + result.timing.loss_minus_us;
    result.timing.loss_us = antithetic ? result.timing.loss_work_us :
        result.timing.loss_minus_us + result.timing.loss_wait_us;
    result.timing.tokens_total = result.timing.tokens_plus +
        result.timing.tokens_minus + result.timing.tokens_pair;
    if (result.timing.tokens_total != result.timing.tokens_backend ||
        result.timing.tokens_real + result.timing.tokens_padding != result.timing.tokens_backend) {
        throw std::runtime_error("inconsistent ZO token accounting");
    }
    return result;
}

void log_step_timing(int step, bool paired, const step_timing & timing) {
    LOG_INF(
        "timing kind=train step=%d path=%s sample_us=%lld batch_us=%lld noise_prepare_us=%lld "
        "perturb_us=%lld upload_us=%lld upload_plus_us=%lld upload_minus_us=%lld upload_pair_us=%lld "
        "decode_us=%lld decode_plus_us=%lld decode_minus_us=%lld decode_pair_us=%lld logits_us=%lld "
        "loss_us=%lld loss_work_us=%lld loss_plus_us=%lld loss_minus_us=%lld loss_wait_us=%lld update_us=%lld "
        "pipeline_noise_work_us=%lld pipeline_wait_us=%lld step_wall_us=%lld "
        "tokens_total=%lld tokens_plus=%lld tokens_minus=%lld tokens_pair=%lld "
        "tokens_real=%lld tokens_padding=%lld tokens_backend=%lld "
        "real_tokens_per_s=%.3f backend_tokens_per_s=%.3f\n",
        step, paired ? "paired" : "serial",
        (long long) timing.sample_us,
        (long long) timing.batch_us,
        (long long) timing.noise_prepare_us,
        (long long) timing.perturb_us,
        (long long) timing.upload_us,
        (long long) timing.upload_plus_us,
        (long long) timing.upload_minus_us,
        (long long) timing.upload_pair_us,
        (long long) timing.decode_us,
        (long long) timing.decode_plus_us,
        (long long) timing.decode_minus_us,
        (long long) timing.decode_pair_us,
        (long long) timing.logits_us,
        (long long) timing.loss_us,
        (long long) timing.loss_work_us,
        (long long) timing.loss_plus_us,
        (long long) timing.loss_minus_us,
        (long long) timing.loss_wait_us,
        (long long) timing.update_us,
        (long long) timing.pipeline_noise_work_us,
        (long long) timing.pipeline_wait_us,
        (long long) timing.step_wall_us,
        (long long) timing.tokens_total,
        (long long) timing.tokens_plus,
        (long long) timing.tokens_minus,
        (long long) timing.tokens_pair,
        (long long) timing.tokens_real,
        (long long) timing.tokens_padding,
        (long long) timing.tokens_backend,
        timing.decode_us == 0 ? 0.0 : timing.tokens_real*1e6/timing.decode_us,
        timing.decode_us == 0 ? 0.0 : timing.tokens_backend*1e6/timing.decode_us);
}

template<typename Getter>
void log_timing_summary(
        const std::vector<step_timing> & timings,
        const char * metric,
        const char * unit,
        Getter getter) {
    if (timings.empty()) {
        return;
    }
    std::vector<int64_t> values;
    values.reserve(timings.size());
    long double sum = 0.0;
    for (const step_timing & timing : timings) {
        const int64_t value = getter(timing);
        values.push_back(value);
        sum += value;
    }
    std::sort(values.begin(), values.end());
    const size_t p50_index = ((values.size() * 50 + 99) / 100) - 1;
    const size_t p95_index = ((values.size() * 95 + 99) / 100) - 1;
    LOG_INF("timing_summary kind=train metric=%s count=%zu avg=%.3f p50=%lld p95=%lld unit=%s\n",
        metric, timings.size(), (double) (sum / timings.size()),
        (long long) values[p50_index], (long long) values[p95_index], unit);
}

void log_training_summary(const std::vector<step_timing> & timings) {
    LOG_INF("timing_summary kind=train count=%zu\n", timings.size());
    if (timings.empty()) {
        return;
    }
    log_timing_summary(timings, "sample", "us", [](const step_timing & value) { return value.sample_us; });
    log_timing_summary(timings, "batch", "us", [](const step_timing & value) { return value.batch_us; });
    log_timing_summary(timings, "noise_prepare", "us", [](const step_timing & value) { return value.noise_prepare_us; });
    log_timing_summary(timings, "perturb", "us", [](const step_timing & value) { return value.perturb_us; });
    log_timing_summary(timings, "upload", "us", [](const step_timing & value) { return value.upload_us; });
    log_timing_summary(timings, "upload_plus", "us", [](const step_timing & value) { return value.upload_plus_us; });
    log_timing_summary(timings, "upload_minus", "us", [](const step_timing & value) { return value.upload_minus_us; });
    log_timing_summary(timings, "upload_pair", "us", [](const step_timing & value) { return value.upload_pair_us; });
    log_timing_summary(timings, "decode", "us", [](const step_timing & value) { return value.decode_us; });
    log_timing_summary(timings, "decode_plus", "us", [](const step_timing & value) { return value.decode_plus_us; });
    log_timing_summary(timings, "decode_minus", "us", [](const step_timing & value) { return value.decode_minus_us; });
    log_timing_summary(timings, "decode_pair", "us", [](const step_timing & value) { return value.decode_pair_us; });
    log_timing_summary(timings, "logits", "us", [](const step_timing & value) { return value.logits_us; });
    log_timing_summary(timings, "loss", "us", [](const step_timing & value) { return value.loss_us; });
    log_timing_summary(timings, "loss_work", "us", [](const step_timing & value) { return value.loss_work_us; });
    log_timing_summary(timings, "loss_wait", "us", [](const step_timing & value) { return value.loss_wait_us; });
    log_timing_summary(timings, "update", "us", [](const step_timing & value) { return value.update_us; });
    log_timing_summary(timings, "pipeline_noise_work", "us", [](const step_timing & value) { return value.pipeline_noise_work_us; });
    log_timing_summary(timings, "pipeline_wait", "us", [](const step_timing & value) { return value.pipeline_wait_us; });
    log_timing_summary(timings, "step_wall", "us", [](const step_timing & value) { return value.step_wall_us; });
    log_timing_summary(timings, "tokens_total", "tokens", [](const step_timing & value) { return value.tokens_total; });
    log_timing_summary(timings, "tokens_real", "tokens", [](const step_timing & value) { return value.tokens_real; });
    log_timing_summary(timings, "tokens_padding", "tokens", [](const step_timing & value) { return value.tokens_padding; });
    log_timing_summary(timings, "tokens_backend", "tokens", [](const step_timing & value) { return value.tokens_backend; });
}

std::vector<lora_tensor_state> collect_adapter_state(llama_adapter_lora * adapter, const llama_model * model, bool htp) {
    if (adapter == nullptr || !std::isfinite(adapter->alpha) || adapter->alpha <= 0.0f) {
        throw std::runtime_error("Adapter alpha must be finite and positive");
    }
    std::vector<std::pair<std::string, llama_adapter_lora_weight *>> ordered;
    ordered.reserve(adapter->ab_map.size());
    for (auto & item : adapter->ab_map) {
        ordered.push_back({ item.first, &item.second });
    }
    std::sort(ordered.begin(), ordered.end(), [](const auto & left, const auto & right) {
        return left.first < right.first;
    });
    if (ordered.empty()) {
        throw std::runtime_error("Adapter has no LoRA target tensors");
    }

    std::vector<lora_tensor_state> result;
    result.reserve(ordered.size());
    int64_t uniform_rank = -1;
    for (const auto & item : ordered) {
        const std::string & name = item.first;
        llama_adapter_lora_weight * weight = item.second;
        const ggml_tensor * base = model->get_tensor(name.c_str());
        if (!is_target_name(name) || base == nullptr || weight->a == nullptr || weight->b == nullptr ||
            base->ne[2] != 1 || base->ne[3] != 1 ||
            weight->a->ne[2] != 1 || weight->a->ne[3] != 1 ||
            weight->b->ne[2] != 1 || weight->b->ne[3] != 1) {
            throw std::runtime_error("invalid ZO-LoRA target: " + name);
        }
        if (weight->a->type != GGML_TYPE_F16 || weight->b->type != GGML_TYPE_F16) {
            throw std::runtime_error("ZO-LoRA requires F16 A/B tensors: " + name);
        }
        const int64_t rank = weight->b->ne[0];
        if ((rank != 8 && rank != 16 && rank != 24 && rank != 32) || weight->a->ne[1] != rank) {
            throw std::runtime_error("invalid ZO-LoRA rank for " + name);
        }
        if (uniform_rank == -1) {
            uniform_rank = rank;
        } else if (uniform_rank != rank) {
            throw std::runtime_error("Adapter ranks are not uniform");
        }
        if (base->ne[0] != weight->a->ne[0] || base->ne[1] != weight->b->ne[1]) {
            throw std::runtime_error("Adapter/base shape mismatch for " + name);
        }
        if (htp) {
            if ((base->type != GGML_TYPE_Q4_0 &&
                 base->type != GGML_TYPE_Q8_0 &&
                 base->type != GGML_TYPE_F16) ||
                weight->b_pair == nullptr || weight->b_pair->type != GGML_TYPE_F16 ||
                weight->b_pair->ne[0] != ((weight->b->ne[1] + 63)/64)*64 ||
                weight->b_pair->ne[1] != rank || weight->b_pair->ne[2] != 2 || weight->b_pair->ne[3] != 1 ||
                !ggml_is_contiguous(weight->b_pair)) {
                throw std::runtime_error("HTP fused LoRA workspace mismatch for " + name);
            }
        }

        lora_tensor_state state;
        state.name = name;
        state.weight = weight;
        state.rank = rank;
        state.output_dim = weight->b->ne[1];
        state.padded_output_dim = ((state.output_dim + 63)/64)*64;
        state.master.resize((size_t) ggml_nelements(weight->b));
        state.f16_io.resize(state.master.size());
        ggml_backend_tensor_get(weight->b, state.f16_io.data(), 0, state.f16_io.size() * sizeof(state.f16_io[0]));
        ggml_fp16_to_fp32_row(state.f16_io.data(), state.master.data(), (int64_t) state.master.size());
        result.push_back(std::move(state));
    }
    return result;
}

std::vector<serialized_tensor> generated_adapter_tensors(const llama_model * model, int rank, uint64_t seed) {
    std::vector<std::pair<std::string, const ggml_tensor *>> targets;
    for (const auto & item : llama_internal_get_tensor_map(model)) {
        if (is_target_name(item.first) && item.second != nullptr && ggml_n_dims(item.second) == 2) {
            targets.push_back({ item.first, item.second });
        }
    }
    std::sort(targets.begin(), targets.end(), [](const auto & left, const auto & right) {
        return left.first < right.first;
    });
    if (targets.empty()) {
        throw std::runtime_error("base model has no standard ZO-LoRA target tensors");
    }

    std::vector<serialized_tensor> tensors;
    tensors.reserve(targets.size() * 2);
    for (const auto & target : targets) {
        const int64_t input_dim = target.second->ne[0];
        const int64_t output_dim = target.second->ne[1];
        if (input_dim <= 0 || output_dim <= 0) {
            throw std::runtime_error("invalid base tensor shape for " + target.first);
        }
        serialized_tensor a;
        a.name = target.first + ".lora_a";
        a.ne0 = input_dim;
        a.ne1 = rank;
        a.data.resize((size_t) input_dim * (size_t) rank);
        const float bound = 1.0f / std::sqrt((float) input_dim);
        stable_rng rng(domain_seed(seed, 0x41494e4954ULL, stable_hash(target.first)));
        for (ggml_fp16_t & value : a.data) {
            value = ggml_fp32_to_fp16(rng.uniform(-bound, bound));
        }
        tensors.push_back(std::move(a));

        serialized_tensor b;
        b.name = target.first + ".lora_b";
        b.ne0 = rank;
        b.ne1 = output_dim;
        b.data.assign((size_t) rank * (size_t) output_dim, ggml_fp32_to_fp16(0.0f));
        tensors.push_back(std::move(b));
    }
    return tensors;
}

std::vector<serialized_tensor> trained_adapter_tensors(
        llama_adapter_lora * adapter,
        const std::vector<lora_tensor_state> & states) {
    std::vector<serialized_tensor> tensors;
    tensors.reserve(states.size() * 2);
    for (const lora_tensor_state & state : states) {
        const llama_adapter_lora_weight & weight = adapter->ab_map.at(state.name);
        serialized_tensor a;
        a.name = state.name + ".lora_a";
        a.ne0 = weight.a->ne[0];
        a.ne1 = weight.a->ne[1];
        a.data.resize((size_t) ggml_nelements(weight.a));
        ggml_backend_tensor_get(weight.a, a.data.data(), 0, a.data.size() * sizeof(a.data[0]));
        tensors.push_back(std::move(a));

        serialized_tensor b;
        b.name = state.name + ".lora_b";
        b.ne0 = state.rank;
        b.ne1 = state.output_dim;
        b.data.resize(state.master.size());
        ggml_fp32_to_fp16_row(state.master.data(), b.data.data(), (int64_t) state.master.size());
        tensors.push_back(std::move(b));
    }
    return tensors;
}

void write_adapter_gguf(
        const fs::path & path,
        const std::string & architecture,
        float alpha,
        const std::vector<serialized_tensor> & tensors) {
    gguf_context_ptr gguf(gguf_init_empty());
    if (!gguf) {
        throw std::runtime_error("failed to allocate GGUF context");
    }
    gguf_set_val_str(gguf.get(), "general.type", "adapter");
    gguf_set_val_str(gguf.get(), "general.architecture", architecture.c_str());
    gguf_set_val_str(gguf.get(), "general.name", "SST-2 ZO-LoRA");
    gguf_set_val_str(gguf.get(), "adapter.type", "lora");
    gguf_set_val_f32(gguf.get(), "adapter.lora.alpha", alpha);

    ggml_init_params params = {
        /*.mem_size   =*/ tensors.size() * ggml_tensor_overhead() + 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr tensor_ctx(ggml_init(params));
    if (!tensor_ctx) {
        throw std::runtime_error("failed to allocate Adapter tensor context");
    }
    for (const serialized_tensor & item : tensors) {
        ggml_tensor * tensor = ggml_new_tensor_2d(tensor_ctx.get(), GGML_TYPE_F16, item.ne0, item.ne1);
        ggml_set_name(tensor, item.name.c_str());
        gguf_add_tensor(gguf.get(), tensor);
        gguf_set_tensor_data(gguf.get(), item.name.c_str(), item.data.data());
    }
    if (!gguf_write_to_file(gguf.get(), path.string().c_str(), false)) {
        throw std::runtime_error("failed to write Adapter GGUF: " + path.string());
    }
}

void sync_file(const fs::path & path) {
#if !defined(_WIN32)
    const int fd = open(path.string().c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error("open for fsync failed: " + path.string() + ": " + std::strerror(errno));
    }
    const int status = fsync(fd);
    const int saved_errno = errno;
    close(fd);
    if (status != 0) {
        throw std::runtime_error("fsync failed: " + path.string() + ": " + std::strerror(saved_errno));
    }
#else
    (void) path;
#endif
}

void sync_directory(const fs::path & path) {
#if !defined(_WIN32)
    const int fd = open(path.string().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error("open directory for fsync failed: " + path.string() + ": " + std::strerror(errno));
    }
    const int status = fsync(fd);
    const int saved_errno = errno;
    close(fd);
    if (status != 0) {
        throw std::runtime_error("directory fsync failed: " + path.string() + ": " + std::strerror(saved_errno));
    }
#else
    (void) path;
#endif
}

fs::path absolute_normal(const fs::path & path) {
    return fs::absolute(path).lexically_normal();
}

fs::path choose_output_path(const options & opts) {
    if (!opts.lora_out.empty()) {
        const fs::path output = absolute_normal(opts.lora_out);
        if (fs::exists(output)) {
            throw std::runtime_error("output path already exists: " + output.string());
        }
        if (!output.parent_path().empty() && !fs::is_directory(output.parent_path())) {
            throw std::runtime_error("output directory does not exist: " + output.parent_path().string());
        }
        return output;
    }

    std::time_t now = std::time(nullptr);
    std::tm local = {};
#if defined(_WIN32)
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char timestamp[32];
    if (std::strftime(timestamp, sizeof(timestamp), "%Y%m%d-%H%M", &local) == 0) {
        throw std::runtime_error("failed to format output timestamp");
    }
    const std::string stem = std::string("sst2-zo-lora-") + timestamp;
    for (int collision = 0; collision < 1000; ++collision) {
        char suffix[16] = {};
        if (collision != 0) {
            std::snprintf(suffix, sizeof(suffix), "-%02d", collision);
        }
        const fs::path candidate = absolute_normal(stem + suffix + ".gguf");
        if (!fs::exists(candidate)) {
            return candidate;
        }
    }
    throw std::runtime_error("could not choose a unique timestamped Adapter path");
}

fs::path temporary_sibling(const fs::path & output, const char * tag) {
    const fs::path directory = output.parent_path();
    const std::string base = "." + output.filename().string() + "." + tag;
#if defined(_WIN32)
    const long process_id = 0;
#else
    const long process_id = (long) getpid();
#endif
    for (int collision = 0; collision < 1000; ++collision) {
        const fs::path candidate = directory /
            (base + "." + std::to_string(process_id) + "." + std::to_string(collision));
        if (!fs::exists(candidate)) {
            return candidate;
        }
    }
    throw std::runtime_error("could not choose a temporary Adapter path");
}

class temporary_file {
public:
    explicit temporary_file(fs::path path) : file_path(std::move(path)) {}
    ~temporary_file() {
        std::error_code error;
        fs::remove(file_path, error);
    }
    const fs::path & path() const { return file_path; }
    void release() { file_path.clear(); }

private:
    fs::path file_path;
};

save_timing save_adapter_atomic(
        const fs::path & output,
        llama_model * model,
        llama_adapter_lora * adapter,
        const std::vector<lora_tensor_state> & states) {
    save_timing timing;
    const int64_t wall_started = ggml_time_us();
    const fs::path partial_path = temporary_sibling(output, "partial");
    temporary_file partial(partial_path);
    int64_t save_started = ggml_time_us();
    write_adapter_gguf(partial_path, model->arch_name(), adapter->alpha, trained_adapter_tensors(adapter, states));
    sync_file(partial_path);
    timing.save_us += ggml_time_us() - save_started;

    const int64_t fresh_load_started = ggml_time_us();
    adapter_ptr verification(llama_adapter_lora_init_zo(
        model, partial_path.string().c_str(), adapter->zo_fused_htp, adapter->zo_paired),
        llama_adapter_lora_free);
    if (!verification) {
        throw std::runtime_error("fresh-load verification failed for saved Adapter");
    }
    if (verification->alpha != adapter->alpha) {
        throw std::runtime_error("fresh-load verification alpha mismatch");
    }
    const auto verified = collect_adapter_state(verification.get(), model, adapter->zo_fused_htp);
    if (verified.size() != states.size()) {
        throw std::runtime_error("fresh-load verification target count mismatch");
    }
    for (size_t tensor_index = 0; tensor_index < states.size(); ++tensor_index) {
        if (verified[tensor_index].name != states[tensor_index].name ||
            verified[tensor_index].master.size() != states[tensor_index].master.size()) {
            throw std::runtime_error("fresh-load verification tensor mismatch");
        }
        const auto & expected_weight = adapter->ab_map.at(states[tensor_index].name);
        const auto & verified_weight = verification->ab_map.at(states[tensor_index].name);
        if (expected_weight.a->type != GGML_TYPE_F16 || verified_weight.a->type != GGML_TYPE_F16 ||
            !ggml_are_same_shape(expected_weight.a, verified_weight.a)) {
            throw std::runtime_error("fresh-load verification A shape mismatch");
        }
        std::vector<ggml_fp16_t> expected_a((size_t) ggml_nelements(expected_weight.a));
        std::vector<ggml_fp16_t> verified_a(expected_a.size());
        ggml_backend_tensor_get(expected_weight.a, expected_a.data(), 0, expected_a.size() * sizeof(expected_a[0]));
        ggml_backend_tensor_get(verified_weight.a, verified_a.data(), 0, verified_a.size() * sizeof(verified_a[0]));
        if (expected_a != verified_a) {
            throw std::runtime_error("fresh-load verification A value mismatch for " + states[tensor_index].name);
        }
        for (size_t value_index = 0; value_index < states[tensor_index].master.size(); ++value_index) {
            const float expected = ggml_fp16_to_fp32(ggml_fp32_to_fp16(states[tensor_index].master[value_index]));
            if (verified[tensor_index].master[value_index] != expected) {
                throw std::runtime_error("fresh-load verification value mismatch for " + states[tensor_index].name);
            }
        }
    }
    verification.reset();
    timing.fresh_load_us = ggml_time_us() - fresh_load_started;

    save_started = ggml_time_us();
#if !defined(_WIN32)
#if defined(SYS_renameat2)
    constexpr unsigned int rename_noreplace = 1;
    if (syscall(SYS_renameat2, AT_FDCWD, partial_path.string().c_str(),
                AT_FDCWD, output.string().c_str(), rename_noreplace) != 0) {
        const int rename_errno = errno;
        if (rename_errno != ENOSYS && rename_errno != EINVAL && rename_errno != EOPNOTSUPP) {
            throw std::runtime_error("atomic rename failed: " + output.string() + ": " + std::strerror(rename_errno));
        }
        if (link(partial_path.string().c_str(), output.string().c_str()) != 0) {
            throw std::runtime_error("atomic publish failed: " + output.string() + ": " + std::strerror(errno));
        }
        if (unlink(partial_path.string().c_str()) != 0) {
            const int unlink_errno = errno;
            unlink(output.string().c_str());
            throw std::runtime_error("temporary Adapter cleanup failed: " + std::string(std::strerror(unlink_errno)));
        }
    }
#else
    if (link(partial_path.string().c_str(), output.string().c_str()) != 0) {
        throw std::runtime_error("atomic publish failed: " + output.string() + ": " + std::strerror(errno));
    }
    if (unlink(partial_path.string().c_str()) != 0) {
        const int unlink_errno = errno;
        unlink(output.string().c_str());
        throw std::runtime_error("temporary Adapter cleanup failed: " + std::string(std::strerror(unlink_errno)));
    }
#endif
#else
    if (fs::exists(output)) {
        throw std::runtime_error("output path appeared during training: " + output.string());
    }
    std::error_code error;
    fs::rename(partial_path, output, error);
    if (error) {
        throw std::runtime_error("atomic rename failed: " + error.message());
    }
#endif
    partial.release();
    sync_directory(output.parent_path());
    timing.save_us += ggml_time_us() - save_started;
    timing.wall_us = ggml_time_us() - wall_started;
    return timing;
}

bool device_name_is_htp(ggml_backend_dev_t device) {
    if (device == nullptr) {
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    if (reg == nullptr) {
        return false;
    }
    const char * name = ggml_backend_reg_name(reg);
    return name != nullptr && std::strcmp(name, "HTP") == 0;
}

void log_configuration(const options & opts, const common_params & params, const std::string & input_lora, const fs::path & output) {
    LOG_INF("\nllama-zo configuration:\n");
    LOG_INF("  mode         = %s\n", opts.mode == run_mode::CPU ? "cpu" : "coop");
    LOG_INF("  lora_exec    = %s\n", opts.lora_exec == lora_exec_mode::RUNTIME ? "runtime" : "fused-htp");
    LOG_INF("  pipeline     = %s\n", opts.pipeline ? "true" : "false");
    LOG_INF("  antithetic   = %s\n", opts.antithetic ? "true" : "false");
    LOG_INF("  warmup_steps = %d\n", opts.warmup_steps);
    LOG_INF("  model        = %s\n", params.model.path.c_str());
    LOG_INF("  lora_input   = %s\n", input_lora.empty() ? "<auto-generate>" : input_lora.c_str());
    LOG_INF("  lora_output  = %s\n", output.string().c_str());
    LOG_INF("  train_data   = %s\n", opts.train_path.c_str());
    LOG_INF("  eval_data    = %s\n", opts.eval_step == -1 ? "<disabled>" : opts.eval_path.c_str());
    LOG_INF("  batch/seq    = %d/%d\n", opts.batch_size, opts.seq_len);
    LOG_INF("  steps/eval   = %d/%d\n", opts.steps, opts.eval_step);
    LOG_INF("  objective    = full-vocabulary cross-entropy\n");
    LOG_INF("  epsilon/lr   = %.8g/%.8g\n", (double) opts.epsilon, (double) opts.learning_rate);
    LOG_INF("  seed         = %llu\n\n", (unsigned long long) opts.seed);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    options opts;
    bool backend_initialized = false;
    std::string final_adapter_path;

    try {
        prescan_mode(argc, argv, opts);
        std::vector<char *> filtered = parse_custom_args(argc, argv, opts);
        validate_options(opts);
        configure_backend_environment(opts);

#if !defined(GGML_USE_HEXAGON)
        if (opts.mode == run_mode::COOP) {
            throw std::runtime_error("coop mode requires a GGML_HEXAGON build");
        }
#endif

        common_init();

        common_params params;
        params.escape = false;
        params.warmup = false;
        if (!common_params_parse((int) filtered.size(), filtered.data(), params, LLAMA_EXAMPLE_COMMON, print_usage)) {
            return 1;
        }
        if (params.sampling.backend_sampling) {
            throw std::runtime_error("full-vocabulary loss does not support --backend-sampling");
        }

        if (params.model.path.empty()) {
            throw std::runtime_error("--model is required");
        }
        if (params.lora_adapters.size() > 1) {
            throw std::runtime_error("llama-zo accepts at most one --lora Adapter");
        }
        std::string input_lora;
        if (!params.lora_adapters.empty()) {
            if (params.lora_adapters.front().scale != 1.0f) {
                throw std::runtime_error("the input Adapter multiplier must be exactly 1");
            }
            input_lora = params.lora_adapters.front().path;
        }
        params.lora_adapters.clear();

        const fs::path output_path = choose_output_path(opts);
        if (!input_lora.empty() && absolute_normal(input_lora) == output_path) {
            throw std::runtime_error("input and output Adapter paths must differ");
        }

        const int64_t sequence_factor = opts.antithetic ? 2 : 1;
        const int64_t max_sequences_64 = sequence_factor * opts.batch_size;
        const int64_t max_tokens_64 = max_sequences_64 * opts.seq_len;
        if (max_sequences_64 > INT32_MAX || max_tokens_64 > INT32_MAX) {
            throw std::runtime_error("batch-size and seq-len exceed llama.cpp batch limits");
        }
        if (opts.mode == run_mode::COOP && max_tokens_64 > 1024) {
            throw std::runtime_error("HTP fused LoRA supports at most 1024 padded tokens per decode");
        }
        const int max_sequences = (int) max_sequences_64;
        const int max_tokens = (int) max_tokens_64;
        params.n_parallel = max_sequences;
        params.n_sequences = max_sequences;
        params.n_ctx = std::max(params.n_ctx, max_tokens);
        params.n_batch = std::max(params.n_batch, max_tokens);
        params.n_ubatch = std::max(params.n_ubatch, max_tokens);
        params.n_outputs_max = std::max(params.n_outputs_max, max_sequences);
        if (opts.mode == run_mode::CPU) {
            params.devices.clear();
            params.n_gpu_layers = 0;
            params.main_gpu = -1;
            params.fit_params = false;
            params.no_op_offload = true;
            params.no_kv_offload = true;
            params.split_mode = LLAMA_SPLIT_MODE_NONE;
        } else {
            if (params.devices.empty() || params.devices.front() == nullptr || !device_name_is_htp(params.devices.front())) {
                throw std::runtime_error("coop mode requires an explicit HTP device, for example --device HTP0");
            }
            for (ggml_backend_dev_t device : params.devices) {
                if (device == nullptr) {
                    break;
                }
                if (!device_name_is_htp(device)) {
                    throw std::runtime_error("coop mode forbids non-HTP devices");
                }
            }
        }

        log_configuration(opts, params, input_lora, output_path);

        const int64_t init_model_started = ggml_time_us();
        llama_backend_init();
        backend_initialized = true;
        llama_numa_init(params.numa);
        auto init = common_init_from_params(params);
        if (!init || init->model() == nullptr || init->context() == nullptr) {
            throw std::runtime_error("failed to initialize model and context");
        }
        const int64_t init_model_us = ggml_time_us() - init_model_started;
        llama_model * model = init->model();
        llama_context * ctx = init->context();

        const int64_t init_adapter_started = ggml_time_us();
        std::unique_ptr<temporary_file> bootstrap_file;
        if (input_lora.empty()) {
            const fs::path bootstrap_path = temporary_sibling(output_path, "bootstrap");
            bootstrap_file.reset(new temporary_file(bootstrap_path));
            const auto generated = generated_adapter_tensors(model, opts.rank, opts.seed);
            write_adapter_gguf(bootstrap_path, model->arch_name(), opts.alpha, generated);
            input_lora = bootstrap_path.string();
            LOG_INF("generated ZO-LoRA bootstrap Adapter with %zu targets\n", generated.size() / 2);
        }

        adapter_ptr adapter(llama_adapter_lora_init_zo(
            model, input_lora.c_str(), opts.mode == run_mode::COOP, opts.antithetic), llama_adapter_lora_free);
        if (!adapter) {
            throw std::runtime_error("failed to load ZO-LoRA Adapter: " + input_lora);
        }
        auto tensor_states = collect_adapter_state(adapter.get(), model, opts.mode == run_mode::COOP);
        if (bootstrap_file) {
            bootstrap_file.reset();
        }

        llama_adapter_lora * adapter_raw = adapter.get();
        float adapter_scale = 1.0f;
        if (llama_set_adapters_lora(ctx, &adapter_raw, 1, &adapter_scale) != 0) {
            throw std::runtime_error("failed to apply ZO-LoRA Adapter");
        }
        const int64_t init_adapter_us = ggml_time_us() - init_adapter_started;

        const int64_t init_data_started = ggml_time_us();
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = vocab == nullptr ? 0 : llama_vocab_n_tokens(vocab);
        if (n_vocab <= 0) {
            throw std::runtime_error("model has no usable vocabulary");
        }
        const auto train_data = load_sst2(opts.train_path, vocab, opts.max_train, opts.seq_len);
        std::vector<sst2_sample> validation_data;
        if (opts.eval_step != -1) {
            validation_data = load_sst2(opts.eval_path, vocab, opts.max_eval, opts.seq_len);
        }
        const int64_t init_data_us = ggml_time_us() - init_data_started;
        LOG_INF("loaded SST-2 rows: train=%zu validation=%zu\n", train_data.size(), validation_data.size());

        const int64_t init_batch_started = ggml_time_us();
        llama_batch batch = llama_batch_init(max_tokens, 0, max_sequences);
        struct batch_guard {
            llama_batch value;
            ~batch_guard() { llama_batch_free(value); }
        } batch_owner { batch };
        const llama_token pad_token = padding_token(vocab);
        const bool htp = opts.mode == run_mode::COOP;
        const int64_t init_batch_us = ggml_time_us() - init_batch_started;

        const int64_t init_upload_started = ggml_time_us();
        upload_master(tensor_states, htp);
        const int64_t init_upload_us = ggml_time_us() - init_upload_started;
        auto run_evaluation = [&](int step) {
            upload_master(tensor_states, htp);
            const eval_result result = evaluate(ctx, adapter.get(), batch_owner.value, validation_data, opts.batch_size,
                n_vocab, htp, pad_token);
            LOG_INF("[eval] step=%d loss=%.6f accuracy=%.4f (%d/%d) time=%.3f s "
                    "tokens_real=%lld tokens_padding=%lld tokens_backend=%lld\n",
                step, (double) result.loss,
                result.total == 0 ? 0.0 : (double) result.correct / result.total,
                result.correct, result.total, result.elapsed_us / 1e6,
                (long long) result.tokens_real,
                (long long) result.tokens_padding,
                (long long) result.tokens_backend);
        };

        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        if (opts.warmup_steps > 0) {
            const uint64_t warmup_data_seed = domain_seed(opts.seed, 0x5741524d44415441ULL, 0);
            const uint64_t warmup_noise_seed = domain_seed(opts.seed, 0x5741524d4e4f4953ULL, 0);
            const auto warmup_samples = training_batch(
                train_data, opts.batch_size, 0, warmup_data_seed);
            for (int warmup = 0; warmup < opts.warmup_steps; ++warmup) {
                if (g_stop_requested) {
                    break;
                }
                const int64_t warmup_started = ggml_time_us();
                timed_noise_plan timed_plan = measure_noise_plan(tensor_states, warmup, warmup_noise_seed);
                const int64_t perturb_started = ggml_time_us();
                prepare_perturbations(tensor_states, timed_plan.plan, opts.epsilon, htp);
                const int64_t perturb_us = ggml_time_us() - perturb_started;
                const zo_forward_result result = run_zo_forwards(
                    tensor_states, ctx, adapter.get(), batch_owner.value, warmup_samples, opts.antithetic, htp,
                    n_vocab, pad_token);
                const int64_t warmup_wall_us = ggml_time_us() - warmup_started;
                LOG_INF(
                    "timing kind=warmup index=%d path=%s noise_prepare_us=%lld perturb_us=%lld "
                    "upload_us=%lld decode_us=%lld loss_us=%lld step_wall_us=%lld "
                    "tokens_real=%lld tokens_padding=%lld tokens_backend=%lld\n",
                    warmup + 1, opts.antithetic ? "paired" : "serial",
                    (long long) timed_plan.elapsed_us,
                    (long long) perturb_us,
                    (long long) result.timing.upload_us,
                    (long long) result.timing.decode_us,
                    (long long) result.timing.loss_us,
                    (long long) warmup_wall_us,
                    (long long) result.timing.tokens_real,
                    (long long) result.timing.tokens_padding,
                    (long long) result.timing.tokens_backend);
            }
            upload_master(tensor_states, htp);
        }

        if (!g_stop_requested && opts.eval_step != -1) {
            run_evaluation(0);
        }

        int completed_steps = 0;
        std::vector<step_timing> training_timings;
        training_timings.reserve((size_t) opts.steps);
        noise_plan current_plan;
        if (opts.pipeline && !g_stop_requested && opts.steps > 0) {
            current_plan = make_noise_plan(tensor_states, 0, opts.seed);
        }

        for (int step = 0; step < opts.steps; ++step) {
            if (g_stop_requested) {
                LOG_INF("stop requested at step boundary %d\n", completed_steps);
                break;
            }
            step_timing timing;
            const int64_t step_started = ggml_time_us();

            int64_t started = ggml_time_us();
            const auto samples = training_batch(train_data, opts.batch_size, step, opts.seed);
            timing.sample_us = ggml_time_us() - started;

            noise_plan plan;
            if (opts.pipeline) {
                plan = std::move(current_plan);
            } else {
                timed_noise_plan timed_plan = measure_noise_plan(tensor_states, step, opts.seed);
                plan = std::move(timed_plan.plan);
                timing.noise_prepare_us = timed_plan.elapsed_us;
            }
            if (plan.step != step) {
                throw std::runtime_error("NoisePlan step mismatch");
            }
            started = ggml_time_us();
            prepare_perturbations(tensor_states, plan, opts.epsilon, htp);
            timing.perturb_us = ggml_time_us() - started;

            std::future<timed_noise_plan> next_plan_future;
            const bool has_next = opts.pipeline && step + 1 < opts.steps;
            if (has_next) {
                next_plan_future = std::async(std::launch::async, [&tensor_states, &opts, step]() {
                    return measure_noise_plan(tensor_states, step + 1, opts.seed);
                });
            }

            zo_forward_result forward = run_zo_forwards(
                tensor_states, ctx, adapter.get(), batch_owner.value, samples, opts.antithetic, htp,
                n_vocab, pad_token);
            forward.timing.sample_us = timing.sample_us;
            forward.timing.noise_prepare_us = timing.noise_prepare_us;
            forward.timing.perturb_us = timing.perturb_us;
            timing = forward.timing;

            const float directional_derivative = (forward.loss_plus - forward.loss_minus) / (2.0f * opts.epsilon);
            started = ggml_time_us();
            apply_update(tensor_states, plan, -opts.learning_rate * directional_derivative);
            timing.update_us = ggml_time_us() - started;
            ++completed_steps;

            if (has_next) {
                const int64_t wait_started = ggml_time_us();
                timed_noise_plan next_plan = next_plan_future.get();
                timing.pipeline_wait_us = ggml_time_us() - wait_started;
                timing.pipeline_noise_work_us = next_plan.elapsed_us;
                current_plan = std::move(next_plan.plan);
            }
            timing.step_wall_us = ggml_time_us() - step_started;
            training_timings.push_back(timing);
            LOG_INF("step %d/%d loss_plus=%.6f loss_minus=%.6f g=%.8e decode=%.3f ms step=%.3f ms\n",
                completed_steps, opts.steps, (double) forward.loss_plus, (double) forward.loss_minus,
                (double) directional_derivative, timing.decode_us / 1000.0, timing.step_wall_us / 1000.0);
            log_step_timing(completed_steps, opts.antithetic, timing);

            if (g_stop_requested) {
                LOG_INF("stop requested after completed step %d\n", completed_steps);
                break;
            }
            if (opts.eval_step != -1 &&
                (completed_steps % opts.eval_step == 0 || completed_steps == opts.steps)) {
                run_evaluation(completed_steps);
            }
        }
        log_training_summary(training_timings);

        const int64_t final_upload_started = ggml_time_us();
        upload_master(tensor_states, htp);
        const int64_t final_upload_us = ggml_time_us() - final_upload_started;
        const save_timing final_save = save_adapter_atomic(output_path, model, adapter.get(), tensor_states);
        final_adapter_path = output_path.string();
        LOG_INF(
            "timing kind=run init_model_us=%lld init_adapter_us=%lld init_data_us=%lld "
            "init_batch_us=%lld init_upload_us=%lld final_upload_us=%lld "
            "save_us=%lld fresh_load_us=%lld save_wall_us=%lld\n",
            (long long) init_model_us,
            (long long) init_adapter_us,
            (long long) init_data_us,
            (long long) init_batch_us,
            (long long) init_upload_us,
            (long long) final_upload_us,
            (long long) final_save.save_us,
            (long long) final_save.fresh_load_us,
            (long long) final_save.wall_us);

        llama_set_adapters_lora(ctx, nullptr, 0, nullptr);
        adapter.reset();
        init.reset();
        llama_backend_free();
        backend_initialized = false;
    } catch (const std::exception & error) {
        LOG_ERR("llama-zo: %s\n", error.what());
        if (backend_initialized) {
            llama_backend_free();
        }
        return 1;
    }

    std::printf("adapter_path=%s\n", final_adapter_path.c_str());
    std::fflush(stdout);
    return 0;
}
