#include "arg.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "console.h"
#include "chat.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "llama-model.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <memory>
#include <vector>
#include <limits.h>
#include <cinttypes>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

static volatile bool g_is_generating = false;
static volatile bool g_is_interrupted = false;

struct mtmd_binary_header {
    char     magic[4];      // Should be "MTMD"
    uint32_t version;       // Version of the file format, e.g., 1
    uint32_t n_tokens;      // Total number of tokens/vectors
    uint32_t n_embd_dims;   // Dimension of each embedding vector (e.g., 4096)
    uint32_t n_pos_dims;    // Dimension of each position_id (1 for normal, 3 for M-RoPE)
    uint32_t embd_type;     // GGML type of embedding data (e.g., GGML_TYPE_F32)
    uint32_t pos_type;      // GGML type of position data (e.g., GGML_TYPE_I32)
    uint32_t reserved[5];   // Reserved for future use, set to 0
};
static_assert(sizeof(mtmd_binary_header) == 48, "mtmd_binary_header size is not 48 bytes");

std::string escape_piece(const std::string & piece) {
    std::string escaped;
    escaped.reserve(piece.size());
    for (const char ch : piece) {
        switch (ch) {
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            case '"':  escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            default:   escaped.push_back(ch); break;
        }
    }
    return escaped;
}

void write_embeddings_plain(
    const std::string & path,
    const std::vector<llama_token> & tokens,
    const std::vector<float> & embeddings,
    const llama_vocab * vocab,
    int32_t n_embd) {
    std::ofstream fout(path, std::ios::out | std::ios::trunc);
    if (!fout) {
        throw std::runtime_error("failed to open output file: " + path);
    }

    fout << "# tokens: " << tokens.size() << "\n";
    fout << "# dims  : " << n_embd << "\n";

    for (size_t i = 0; i < tokens.size(); ++i) {
        const llama_token token = tokens[i];
        std::string piece;
        if (token == -1) {
            piece = "[IMAGE]";
        } else {
            piece = escape_piece(common_token_to_piece(vocab, token, true));
        }

        fout << "tok " << i << " id " << token << " piece \"" << piece << "\"\n";
        const float * row = embeddings.data() + i * n_embd;
        for (int32_t j = 0; j < n_embd; ++j) {
            fout << row[j];
            if (j + 1 < n_embd) {
                fout << ' ';
            }
        }
        fout << "\n";
    }
}

static void write_positions_plain(
    const std::string & path,
    const std::vector<llama_pos> & positions,
    size_t n_tokens,
    int32_t n_pos_dims) {
    std::ofstream fout(path, std::ios::out | std::ios::trunc);
    if (!fout) {
        throw std::runtime_error("failed to open output file: " + path);
    }

    fout << "# tokens: " << n_tokens << "\n";
    fout << "# dims  : " << n_pos_dims << "\n";

    for (size_t i = 0; i < n_tokens; ++i) {
        fout << "tok " << i << " : ";
        for (int32_t d = 0; d < n_pos_dims; ++d) {
            fout << positions[d * n_tokens + i];
            if (d + 1 < n_pos_dims) {
                fout << ' ';
            }
        }
        fout << "\n";
    }
}

static void write_data_binary(
    const std::string & path,
    size_t n_tokens,
    int32_t n_embd_dims,
    int32_t n_pos_dims,
    const std::vector<float> & embeddings,
    const std::vector<llama_pos> & positions) {
    
    std::ofstream fout(path, std::ios::out | std::ios::binary);
    if (!fout) {
        throw std::runtime_error("failed to open binary output file: " + path);
    }

    mtmd_binary_header header;
    strncpy(header.magic, "MTMD", 4);
    header.version     = 1;
    header.n_tokens    = static_cast<uint32_t>(n_tokens);
    header.n_embd_dims = static_cast<uint32_t>(n_embd_dims);
    header.n_pos_dims  = static_cast<uint32_t>(n_pos_dims);
    header.embd_type   = GGML_TYPE_F32;
    header.pos_type    = GGML_TYPE_I32;
    memset(header.reserved, 0, sizeof(header.reserved));

    fout.write(reinterpret_cast<const char*>(&header), sizeof(header));

    fout.write(reinterpret_cast<const char*>(embeddings.data()), embeddings.size() * sizeof(float));

    fout.write(reinterpret_cast<const char*>(positions.data()), positions.size() * sizeof(llama_pos));

    if (!fout) {
        throw std::runtime_error("error writing data to binary file: " + path);
    }

    fout.close();
}

bool get_text_embeddings(
    llama_model* model,
    const std::vector<llama_token>& tokens,
    std::vector<float>& embeddings_out) {

    if (tokens.empty()) {
        return true;
    }

    const int32_t n_embd = llama_model_n_embd(model);
    embeddings_out.resize(tokens.size() * n_embd);

    constexpr size_t ctx_mem_size = 512 * 1024;
    std::vector<uint8_t> ctx_buffer(ctx_mem_size);

    struct ggml_init_params ggml_params {
        /*.mem_size   =*/ ctx_buffer.size(),
        /*.mem_buffer =*/ ctx_buffer.data(),
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx = ggml_init(ggml_params);
    if (!ctx) {
        LOG_ERR("Error: failed to init ggml context for text embeddings\n");
        return false;
    }

    ggml_tensor * t_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens.size());
    ggml_set_name(t_tokens, "tokens");

    ggml_tensor * embedding_source = model->tok_embd;
    if (!embedding_source) {
        LOG_ERR("Error: could not find token_embd.weight tensor\n");
        ggml_free(ctx);
        return false;
    }


    ggml_tensor * t_rows = ggml_get_rows(ctx, embedding_source, t_tokens);
    ggml_set_name(t_rows, "embedding_rows");
    ggml_set_output(t_rows);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, t_rows);

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        LOG_ERR("Error: failed to init ggml CPU backend\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        LOG_ERR("Error: failed to allocate tensors on CPU backend\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(t_tokens, tokens.data(), 0, tokens.size() * sizeof(llama_token));

    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR("Error: ggml graph compute failed with status %d\n", status);
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(t_rows, embeddings_out.data(), 0, embeddings_out.size() * sizeof(float));

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);

    return true;
}


static void show_additional_info(int /*argc*/, char ** argv) {
    LOG(
        "Experimental CLI for multimodal\n\n"
        "Usage: %s [options] -m <model> --mmproj <mmproj> --image <image> --audio <audio> -p <prompt>\n\n"
        "  -m and --mmproj are required\n"
        "  -hf user/repo can replace both -m and --mmproj in most cases\n"
        "  --image, --audio and -p are optional, if NOT provided, the CLI will run in chat mode\n"
        "  to disable using GPU for mmproj model, add --no-mmproj-offload\n",
        argv[0]
    );
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (g_is_generating) {
            g_is_generating = false;
        } else {
            console::cleanup();
            if (g_is_interrupted) {
                _exit(1);
            }
            g_is_interrupted = true;
        }
    }
}
#endif

struct mtmd_cli_context {
    mtmd::context_ptr ctx_vision;
    common_init_result llama_init;

    llama_model       * model;
    llama_context     * lctx;
    const llama_vocab * vocab;
    common_sampler    * smpl;
    llama_batch         batch;
    int                 n_batch;

    mtmd::bitmaps bitmaps;

    // note: we know that gemma3 template is "linear", meaning each turn is completely separated to another
    // so here we don't need to keep track of chat history
    common_chat_templates_ptr tmpls;

    // support for legacy templates (models not having EOT token)
    llama_tokens antiprompt_tokens;

    int n_threads    = 1;
    llama_pos n_past = 0;

    mtmd_cli_context(common_params & params) : llama_init(common_init_from_params(params)) {
        model = llama_init.model.get();
        lctx = llama_init.context.get();
        vocab = llama_model_get_vocab(model);
        smpl = common_sampler_init(model, params.sampling);
        n_threads = params.cpuparams.n_threads;
        batch = llama_batch_init(1, 0, 1); // batch for next token generation
        n_batch = params.n_batch;

        if (!model || !lctx) {
            exit(1);
        }

        if (!llama_model_chat_template(model, nullptr) && params.chat_template.empty()) {
            LOG_ERR("Model does not have chat template.\n");
            LOG_ERR("  For old llava models, you may need to use '--chat-template vicuna'\n");
            LOG_ERR("  For MobileVLM models, use '--chat-template deepseek'\n");
            LOG_ERR("  For Mistral Small 3.1, use '--chat-template mistral-v7'\n");
            exit(1);
        }

        tmpls = common_chat_templates_init(model, params.chat_template);
        LOG_INF("%s: chat template example:\n%s\n", __func__, common_chat_format_example(tmpls.get(), params.use_jinja, params.default_template_kwargs).c_str());

        init_vision_context(params);

        if (params.chat_template == "vicuna") {
            antiprompt_tokens = common_tokenize(lctx, "ASSISTANT:", false, true);
        } else if (params.chat_template == "deepseek") {
            antiprompt_tokens = common_tokenize(lctx, "###", false, true);
        }
    }

    ~mtmd_cli_context() {
        llama_batch_free(batch);
        common_sampler_free(smpl);
    }

    void init_vision_context(common_params & params) {
        const char * clip_path = params.mmproj.path.c_str();
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu = params.mmproj_use_gpu;
        mparams.print_timings = true;
        mparams.n_threads = params.cpuparams.n_threads;
        mparams.verbosity = params.verbosity > 0 ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_INFO;
        ctx_vision.reset(mtmd_init_from_file(clip_path, model, mparams));
        if (!ctx_vision.get()) {
            LOG_ERR("Failed to load vision model from %s\n", clip_path);
            exit(1);
        }
    }

    bool check_antiprompt(const llama_tokens & generated_tokens) {
        if (antiprompt_tokens.empty() || generated_tokens.size() < antiprompt_tokens.size()) {
            return false;
        }
        return std::equal(
            generated_tokens.end() - antiprompt_tokens.size(),
            generated_tokens.end(),
            antiprompt_tokens.begin()
        );
    }

    bool load_media(const std::string & fname) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_vision.get(), fname.c_str()));
        if (!bmp.ptr) {
            return false;
        }
        bitmaps.entries.push_back(std::move(bmp));
        return true;
    }
};

static int generate_response(mtmd_cli_context & ctx, int n_predict) {
    llama_tokens generated_tokens;
    for (int i = 0; i < n_predict; i++) {
        if (i > n_predict || !g_is_generating || g_is_interrupted) {
            LOG("\n");
            break;
        }

        llama_token token_id = common_sampler_sample(ctx.smpl, ctx.lctx, -1);
        generated_tokens.push_back(token_id);
        common_sampler_accept(ctx.smpl, token_id, true);

        if (llama_vocab_is_eog(ctx.vocab, token_id) || ctx.check_antiprompt(generated_tokens)) {
            LOG("\n");
            break;
        }

        LOG("%s", common_token_to_piece(ctx.lctx, token_id).c_str());
        fflush(stdout);

        if (g_is_interrupted) {
            LOG("\n");
            break;
        }

        common_batch_clear(ctx.batch);
        common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);
        if (llama_decode(ctx.lctx, ctx.batch)) {
            LOG_ERR("failed to decode token\n");
            return 1;
        }
    }
    return 0;
}

static int eval_message(mtmd_cli_context & ctx, common_chat_msg & msg, bool add_bos = false) {
    common_chat_templates_inputs tmpl_inputs;
    tmpl_inputs.messages = {msg};
    tmpl_inputs.add_generation_prompt = true;
    tmpl_inputs.use_jinja = false;
    auto formatted_chat = common_chat_templates_apply(ctx.tmpls.get(), tmpl_inputs);
    LOG_DBG("formatted_chat.prompt: %s\n", formatted_chat.prompt.c_str());

    mtmd_input_text text;
    text.text          = formatted_chat.prompt.c_str();
    text.add_special   = add_bos;
    text.parse_special = true;

    if (g_is_interrupted) return 0;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = ctx.bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(ctx.ctx_vision.get(),
                        chunks.ptr.get(),
                        &text,
                        bitmaps_c_ptr.data(),
                        bitmaps_c_ptr.size());
    if (res != 0) {
        LOG_ERR("Unable to tokenize prompt, res = %d\n", res);
        return 1;
    }

    ctx.bitmaps.entries.clear();

    llama_pos new_n_past;
    if (mtmd_helper_eval_chunks(ctx.ctx_vision.get(),
                ctx.lctx, // lctx
                chunks.ptr.get(), // chunks
                ctx.n_past, // n_past
                0, // seq_id
                ctx.n_batch, // n_batch
                true, // logits_last
                &new_n_past)) {
        LOG_ERR("Unable to eval prompt\n");
        return 1;
    }

    ctx.n_past = new_n_past;

    LOG("\n");

    return 0;
}

int main(int argc, char ** argv) {
    ggml_time_init();

    common_params params;
    params.sampling.temp = 0.2; // lower temp by default for better quality

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_additional_info)) {
        return 1;
    }

    common_init();

    if (params.mmproj.path.empty()) {
        show_additional_info(argc, argv);
        LOG_ERR("ERR: Missing --mmproj argument\n");
        return 1;
    }

    mtmd_cli_context ctx(params);
    LOG("%s: loading model: %s\n", __func__, params.model.path.c_str());

    int n_predict = params.n_predict < 0 ? INT_MAX : params.n_predict;

    // Ctrl+C handling
    {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    if (g_is_interrupted) return 130;

    LOG_INF("params.prompt: %s\n", params.prompt.c_str());
    if (params.prompt.find(mtmd_default_marker()) == std::string::npos) {
        for (size_t i = 0; i < params.image.size(); i++) {
            params.prompt += mtmd_default_marker();
        }
    }
    common_chat_msg msg;
    msg.role = "user";
    msg.content = params.prompt;
    for (const auto & image : params.image) {
        if (!ctx.load_media(image)) {
            return 1;
        }
    }

    common_chat_templates_inputs tmpl_inputs;
    tmpl_inputs.messages = {msg};
    tmpl_inputs.add_generation_prompt = true;
    tmpl_inputs.use_jinja = false;
    auto formatted_chat = common_chat_templates_apply(ctx.tmpls.get(), tmpl_inputs);

    mtmd_input_text text;
    text.text          = formatted_chat.prompt.c_str();
    text.add_special   = true;
    text.parse_special = true;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = ctx.bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(ctx.ctx_vision.get(), chunks.ptr.get(), &text, bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
    if (res != 0) {
        LOG_ERR("Unable to tokenize prompt, res = %d\n", res);
        return 1;
    }

    size_t total_tokens = 0;
    size_t total_text_tokens = 0;
    size_t total_image_tokens = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks.ptr.get()); ++i) {
        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
        size_t n_chunk_tokens = mtmd_input_chunk_get_n_tokens(chunk);
        total_tokens += n_chunk_tokens;
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            total_text_tokens += n_chunk_tokens;
        } else {
            total_image_tokens += n_chunk_tokens;
        }
    }

    const int32_t n_embd = llama_model_n_embd(ctx.model);
    const int32_t n_pos_dims = mtmd_decode_use_mrope(ctx.ctx_vision.get()) ? 3 : 1;

    std::vector<llama_token> final_tokens(total_tokens);
    std::vector<float> final_embeddings(total_tokens * n_embd);
    std::vector<llama_pos> final_positions(total_tokens * n_pos_dims, 0);

    llama_pos current_pos = 0;
    size_t token_offset = 0;

    for (size_t i = 0; i < mtmd_input_chunks_size(chunks.ptr.get()); ++i) {
        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
        auto chunk_type = mtmd_input_chunk_get_type(chunk);

        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            LOG_INF("Processing text chunk at pos %lld...\n", (long long)current_pos);
            size_t n_text_tokens;
            const llama_token* text_tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_text_tokens);
            
            std::copy(text_tokens, text_tokens + n_text_tokens, final_tokens.begin() + token_offset);

            std::vector<float> text_embeddings;
            std::vector<llama_token> tokens_vec(text_tokens, text_tokens + n_text_tokens);
            if (!get_text_embeddings(ctx.model, tokens_vec, text_embeddings)) {
                LOG_ERR("Failed to get text embeddings.\n");
                return 1;
            }
            std::copy(text_embeddings.begin(), text_embeddings.end(), final_embeddings.begin() + token_offset * n_embd);

            for (size_t j = 0; j < n_text_tokens; ++j) {
                const llama_pos pos = current_pos + j;
                for (int32_t d = 0; d < n_pos_dims; ++d) {
                    final_positions[d * total_tokens + token_offset + j] = pos;
                }
            }
            
            current_pos += n_text_tokens;

        } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            LOG_INF("Processing image chunk at pos %lld...\n", (long long)current_pos);
            if (mtmd_encode_chunk(ctx.ctx_vision.get(), chunk) != 0) {
                LOG_ERR("Failed to encode image chunk.\n");
                return 1;
            }
            
            std::vector<float>& image_embeddings = mtmd_get_output_embd_vector(ctx.ctx_vision.get());
            std::copy(image_embeddings.begin(), image_embeddings.end(), final_embeddings.begin() + token_offset * n_embd);

            const auto* image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
            const size_t n_img_tokens = mtmd_image_tokens_get_n_tokens(image_tokens);

            std::fill(final_tokens.begin() + token_offset, final_tokens.begin() + token_offset + n_img_tokens, -1);

            if (mtmd_decode_use_mrope(ctx.ctx_vision.get())) {
                const int nx = mtmd_image_tokens_get_nx(image_tokens);
                const int ny = mtmd_image_tokens_get_ny(image_tokens);
                
                for (int y = 0; y < ny; ++y) {
                    for (int x = 0; x < nx; ++x) {
                        size_t token_idx_in_chunk = y * nx + x;
                        size_t global_token_idx = token_offset + token_idx_in_chunk;

                        if (n_pos_dims == 3) {
                            final_positions[total_tokens * 0 + global_token_idx] = current_pos;
                            final_positions[total_tokens * 1 + global_token_idx] = current_pos + y;
                            final_positions[total_tokens * 2 + global_token_idx] = current_pos + x;
                        }
                    }
                }
            } else {
                for (size_t j = 0; j < n_img_tokens; ++j) {
                    const llama_pos pos = current_pos + j;
                    for (int32_t d = 0; d < n_pos_dims; ++d) {
                        final_positions[d * total_tokens + token_offset + j] = pos;
                    }
                }
            }            
            current_pos += mtmd_input_chunk_get_n_pos(chunk);
        }
        token_offset += mtmd_input_chunk_get_n_tokens(chunk);
    }

    const std::string txt_embd_path = "mtmd_embeddings.txt";
    const std::string txt_pos_path = "mtmd_positions.txt";
    const std::string bin_path = "mtmd_data.bin"; // New binary file path

    LOG_INF("Writing combined embeddings to %s\n", txt_embd_path.c_str());
    write_embeddings_plain(txt_embd_path, final_tokens, final_embeddings, ctx.vocab, n_embd);

    LOG_INF("Writing combined position IDs to %s\n", txt_pos_path.c_str());
    write_positions_plain(txt_pos_path, final_positions, total_tokens, n_pos_dims);

    LOG_INF("Writing combined binary data to %s\n", bin_path.c_str());
    try {
        write_data_binary(bin_path, total_tokens, n_embd, n_pos_dims, final_embeddings, final_positions);
    } catch (const std::exception& e) {
        LOG_ERR("Error: %s\n", e.what());
        return 1;
    }


    LOG_INF("\n--- Embedding Extraction Summary ---\n");
    LOG_INF("ViT (Image) Embeddings Shape : [%zu, %d]\n", total_image_tokens, n_embd);
    LOG_INF("Text Embeddings Shape        : [%zu, %d]\n", total_text_tokens, n_embd);
    LOG_INF("Total Combined Embeddings Shape: [%zu, %d]\n", total_tokens, n_embd);
    LOG_INF("Position IDs Shape           : [%zu, %d]\n", total_tokens, n_pos_dims);
    LOG_INF("\nOutput files generated:\n");
    LOG_INF("  - %s (human-readable embeddings)\n", txt_embd_path.c_str());
    LOG_INF("  - %s (human-readable position IDs)\n", txt_pos_path.c_str());
    LOG_INF("  - %s (self-describing binary data)\n", bin_path.c_str());
    LOG_INF("------------------------------------\n");

    if (g_is_interrupted) LOG("\nInterrupted by user\n");
    LOG("\n\n");
    llama_perf_context_print(ctx.lctx);
    return g_is_interrupted ? 130 : 0;
}
