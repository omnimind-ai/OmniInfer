/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/token_generator.h>
#include <numeric>
using executorch::aten::TensorImpl;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;

namespace example {
template <typename T>
TokenGenerator<T>::TokenGenerator(
    tokenizers::Tokenizer* tokenizer,
    DecoderRunner* decoder_runner,
    KVManager<T>* kv_manager,
    const std::string& method_name,
    std::unique_ptr<std::unordered_set<uint64_t>>&& eos_ids,
    Metadata metadata,
    executorch::llm::Stats* stats)
    : tokenizer_(tokenizer),
      decoder_runner_(decoder_runner),
      kv_manager_(kv_manager),
      method_name_(method_name),
      eos_ids_(std::move(eos_ids)),
      stats_(stats),
      metadata_(metadata) {
  k_cache_in_.resize(metadata_.num_layers);
  v_cache_in_.resize(metadata_.num_layers);
  k_cache_out_.resize(metadata_.num_layers);
  v_cache_out_.resize(metadata_.num_layers);

  // Calculate I/O size
  input_toks_.size = metadata_.ar_len * sizeof(int64_t);
  inputs_embeds_.size = metadata_.ar_len * metadata_.hidden_size * sizeof(uint16_t);
  freqs_cos_sin0_.size = metadata_.ar_len * 64 * sizeof(float);
  freqs_cos_sin1_.size = metadata_.ar_len * 64 * sizeof(float);
  attention_mask_.size =
      metadata_.ar_len * metadata_.context_len * sizeof(uint16_t);

  switch (metadata_.cache_mode) {
    case CacheMode::StaticCahce:
      attention_mask_.size =
           metadata_.ar_len * metadata_.context_len * sizeof(uint16_t);
      window_attention_mask_.size = 0;
      break;
    case CacheMode::HybridCache:
      attention_mask_.size =
          metadata_.ar_len * metadata_.context_len * sizeof(uint16_t);
      window_attention_mask_.size =
          metadata_.ar_len * metadata_.context_len * sizeof(uint16_t);
      break;
    default:
      ET_CHECK_MSG(false, "Unsupported llama cache mode");
      break;
  }

  logits_.size = metadata_.ar_len * metadata_.vocab_size * sizeof(uint16_t);
}
template <typename T>
void TokenGenerator<T>::init_io(
    IMemAlloc* buffer_manager,
    Result<MethodMeta> method_meta) {
  size_t idx = 0;
  input_tensors_.reserve(method_meta->num_inputs());
  output_tensors_.reserve(method_meta->num_outputs());
  // [I]: input_tokens
  Result<TensorInfo> input_toks = method_meta->input_tensor_meta(idx++);
  input_toks_.data =
      reinterpret_cast<int64_t*>(buffer_manager->allocate(input_toks_.size));
  input_toks_.tensor = std::make_unique<TensorImpl>(
      input_toks->scalar_type(),
      input_toks->sizes().size(),
      const_cast<TensorImpl::SizesType*>(input_toks->sizes().data()),
      input_toks_.data,
      const_cast<TensorImpl::DimOrderType*>(input_toks->dim_order().data()));
  input_tensors_.emplace_back(input_toks_.tensor.get());
  buffer_manager->add_memory_info(
      input_toks_.data, input_toks_.size, input_toks.get());

  // [I]: attention_mask
  Result<TensorInfo> attention_mask = method_meta->input_tensor_meta(idx++);
  attention_mask_.data = reinterpret_cast<uint16_t*>(
      buffer_manager->allocate(attention_mask_.size));
  attention_mask_.tensor = std::make_unique<TensorImpl>(
      attention_mask->scalar_type(),
      attention_mask->sizes().size(),
      const_cast<TensorImpl::SizesType*>(attention_mask->sizes().data()),
      attention_mask_.data,
      const_cast<TensorImpl::DimOrderType*>(
          attention_mask->dim_order().data()));
  input_tensors_.emplace_back(attention_mask_.tensor.get());
  buffer_manager->add_memory_info(
      attention_mask_.data, attention_mask_.size, attention_mask.get());

  // [I]: inputs_embeds
  Result<TensorInfo> inputs_embeds = method_meta->input_tensor_meta(idx++);
  inputs_embeds_.data = reinterpret_cast<uint16_t*>(
      buffer_manager->allocate(inputs_embeds_.size));
  inputs_embeds_.tensor = std::make_unique<TensorImpl>(
      inputs_embeds->scalar_type(),
      inputs_embeds->sizes().size(),
      const_cast<TensorImpl::SizesType*>(inputs_embeds->sizes().data()),
      inputs_embeds_.data,
      const_cast<TensorImpl::DimOrderType*>(inputs_embeds->dim_order().data()));
  input_tensors_.emplace_back(inputs_embeds_.tensor.get());
  buffer_manager->add_memory_info(
      inputs_embeds_.data, inputs_embeds_.size, inputs_embeds.get());
    
  // [I]: freqs_cos_sin0
    Result<TensorInfo> freqs_cos_sin0 = method_meta->input_tensor_meta(idx++);
    freqs_cos_sin0_.data = reinterpret_cast<float*>(
        buffer_manager->allocate(freqs_cos_sin0_.size));
    freqs_cos_sin0_.tensor = std::make_unique<TensorImpl>(
        freqs_cos_sin0->scalar_type(),
        freqs_cos_sin0->sizes().size(),
        const_cast<TensorImpl::SizesType*>(freqs_cos_sin0->sizes().data()),
        freqs_cos_sin0_.data,
        const_cast<TensorImpl::DimOrderType*>(freqs_cos_sin0->dim_order().data()));
    input_tensors_.emplace_back(freqs_cos_sin0_.tensor.get());
    buffer_manager->add_memory_info(
        freqs_cos_sin0_.data, freqs_cos_sin0_.size, freqs_cos_sin0.get());

  // [I]: freqs_cos_sin1
    Result<TensorInfo> freqs_cos_sin1 = method_meta->input_tensor_meta(idx++);
    freqs_cos_sin1_.data = reinterpret_cast<float*>(
        buffer_manager->allocate(freqs_cos_sin1_.size));
    freqs_cos_sin1_.tensor = std::make_unique<TensorImpl>(
        freqs_cos_sin1->scalar_type(),
        freqs_cos_sin1->sizes().size(),
        const_cast<TensorImpl::SizesType*>(freqs_cos_sin1->sizes().data()),
        freqs_cos_sin1_.data,
        const_cast<TensorImpl::DimOrderType*>(freqs_cos_sin1->dim_order().data()));
    input_tensors_.emplace_back(freqs_cos_sin1_.tensor.get());
    buffer_manager->add_memory_info(
        freqs_cos_sin1_.data, freqs_cos_sin1_.size, freqs_cos_sin1.get());

  // [I]: sliding window attention_mask
  if (metadata_.cache_mode == CacheMode::HybridCache) {
    Result<TensorInfo> window_attention_mask =
        method_meta->input_tensor_meta(idx++);
    window_attention_mask_.data = reinterpret_cast<uint16_t*>(
        buffer_manager->allocate(window_attention_mask_.size));
    window_attention_mask_.tensor = std::make_unique<TensorImpl>(
        window_attention_mask->scalar_type(),
        window_attention_mask->sizes().size(),
        const_cast<TensorImpl::SizesType*>(
            window_attention_mask->sizes().data()),
        window_attention_mask_.data,
        const_cast<TensorImpl::DimOrderType*>(
            window_attention_mask->dim_order().data()));
    input_tensors_.emplace_back(window_attention_mask_.tensor.get());
    buffer_manager->add_memory_info(
        window_attention_mask_.data,
        window_attention_mask_.size,
        window_attention_mask.get());
  }


  // [I] kv_cache
  size_t index = idx; // bypass input_tokens, atten_mask, inputs_embeds, input_pos
  for (int cache_group = 0; cache_group < 2; ++cache_group) {
    std::vector<std::vector<std::unique_ptr<TensorImpl>>>& cache =
        (cache_group == 0 ? k_cache_in_ : v_cache_in_);
    std::vector<std::vector<KVCache<T>>> cache_ptrs = (cache_group == 0)
        ? kv_manager_->get_k_cache_()
        : kv_manager_->get_v_cache_();
    for (int layer = 0; layer < metadata_.num_layers; ++layer) {
      for (int head = 0; head < metadata_.num_heads; ++head, ++index) {
        Result<TensorInfo> kv_cache = method_meta->input_tensor_meta(index);

        T* cache_ptr = cache_ptrs[layer][head].buffer;

        cache[layer].emplace_back(std::make_unique<TensorImpl>(
            kv_cache->scalar_type(),
            kv_cache->sizes().size(),
            const_cast<TensorImpl::SizesType*>(kv_cache->sizes().data()),
            cache_ptr,
            const_cast<TensorImpl::DimOrderType*>(
                kv_cache->dim_order().data())));
        input_tensors_.emplace_back(cache[layer][head].get());
        buffer_manager->add_memory_info(
            cache_ptr, cache[layer][head]->nbytes(), kv_cache.get());
      }
    }
  }

  // [O]: logits
  Result<TensorInfo> logits = method_meta->output_tensor_meta(0);
  logits_.data =
      reinterpret_cast<uint16_t*>(buffer_manager->allocate(logits_.size));
  logits_.tensor = std::make_unique<TensorImpl>(
      logits->scalar_type(),
      logits->sizes().size(),
      const_cast<TensorImpl::SizesType*>(logits->sizes().data()),
      logits_.data,
      const_cast<TensorImpl::DimOrderType*>(logits->dim_order().data()));
  output_tensors_.emplace_back(logits_.tensor.get());
  buffer_manager->add_memory_info(logits_.data, logits_.size, logits.get());

  // [O] kv_cache
  index = 1;
  for (int cache_group = 0; cache_group < 2; ++cache_group) {
    std::vector<std::vector<std::unique_ptr<TensorImpl>>>& cache =
        (cache_group == 0 ? k_cache_out_ : v_cache_out_);
    std::vector<std::vector<KVCache<T>>> cache_ptrs = (cache_group == 0)
        ? kv_manager_->get_k_cache_()
        : kv_manager_->get_v_cache_();
    for (int layer = 0; layer < metadata_.num_layers; ++layer) {
      for (int head = 0; head < metadata_.num_heads; ++head, ++index) {
        Result<TensorInfo> kv_cache = method_meta->output_tensor_meta(index);
        T* cache_ptr = cache_ptrs[layer][head].output_buffer;
        cache[layer].emplace_back(std::make_unique<TensorImpl>(
            kv_cache->scalar_type(),
            kv_cache->sizes().size(),
            const_cast<TensorImpl::SizesType*>(kv_cache->sizes().data()),
            cache_ptr,
            const_cast<TensorImpl::DimOrderType*>(
                kv_cache->dim_order().data())));
        output_tensors_.emplace_back(cache[layer][head].get());
        buffer_manager->add_memory_info(
            cache_ptr, cache[layer][head]->nbytes(), kv_cache.get());
      }
    }
  }
  // Prepare the vector of EValue to run inference
  inputs_.reserve(input_tensors_.size());
  for (auto& input_tensor : input_tensors_) {
    inputs_.emplace_back(std::move(input_tensor));
  }
}

template <typename T>
const std::vector<uint16_t>& TokenGenerator<T>::get_all_logits() {
  return token_all_logits_;
}

// This function only considers the case where token_generator_ar_len equals 1.
template <typename T>
void TokenGenerator<T>::prepare_io(uint64_t cur_token, const std::vector<float>& freqs_cos, const std::vector<float>& freqs_sin, int64_t start_pos) {
  // update input_tok
  *input_toks_.data =
      metadata_.use_int64_token ? cur_token : static_cast<int32_t>(cur_token);
  // update position_ids
//   *input_pos_.data = static_cast<int32_t>(start_pos);
  // update freqs_cos_sin
  std::memcpy(
      freqs_cos_sin0_.data,
      freqs_cos.data() + start_pos * 64,
      64 * sizeof(float));
  std::memcpy(
      freqs_cos_sin1_.data,
      freqs_sin.data() + start_pos * 64,
      64 * sizeof(float));
}

template <typename T>
Result<int64_t> TokenGenerator<T>::generate(
    std::vector<uint64_t> tokens,
    std::vector<float> freqs_cos,
    std::vector<float> freqs_sin,
    int64_t start_pos,
    int32_t seq_len,
    std::function<void(const std::string&)> token_callback,
    bool dump_logits) {
  ET_CHECK_MSG(
      !tokens.empty(), "Token generation loop shouldn't take empty tokens");
  int64_t pos = start_pos; // position in the sequence

  // Token after prefill
  uint64_t cur_token = tokens.back();
  uint64_t prev_token;
  // Rearrange KV cache first
  kv_manager_->rearrange_cache(metadata_.ar_len);
  std::vector<int32_t> attention_map(metadata_.ar_len);
  std::iota(attention_map.begin(), attention_map.end(), -1);

  // Initialize attention mask with current position
  kv_manager_->init_attention_mask(
      attention_mask_.data, attention_map, metadata_.ar_len, pos);
  // Initialize window attention mask with current position
  if (metadata_.cache_mode == CacheMode::HybridCache) {
    kv_manager_->init_attention_mask(
        window_attention_mask_.data,
        attention_map,
        metadata_.ar_len,
        pos,
        metadata_.sliding_window);
  }

  // Initialize the output of the module
  ET_CHECK_MSG(
      decoder_runner_->set_outputs(method_name_, output_tensors_) ==
          executorch::runtime::Error::Ok,
      "Failed to set output tensor for module %s",
      method_name_.c_str());
  // Generate our tokens
  while (pos < seq_len - 1) {
    // Fill in the token and position data

    prepare_io(cur_token, freqs_cos, freqs_sin, pos);
    // Only update data pointer of the cache to the tensor for SHIFT_POINTER
    // mode
    bool updated = kv_manager_->update_cache_tensor(
        k_cache_in_,
        k_cache_out_,
        v_cache_in_,
        v_cache_out_,
        metadata_.ar_len,
        pos);
    // Only update the output of module for SHIFT_POINTER mode
    if (updated) {
      // Update the output of the module
      ET_CHECK_MSG(
          decoder_runner_->set_outputs(method_name_, output_tensors_) ==
              executorch::runtime::Error::Ok,
          "Failed to set output tensor for module %s",
          method_name_.c_str());
    }
    // Run inference
    auto logits_res = decoder_runner_->step(method_name_, inputs_);
    if (dump_logits) {
      token_all_logits_.insert(
          token_all_logits_.end(),
          logits_.data,
          logits_.data + metadata_.ar_len * metadata_.vocab_size);
    }
    ET_CHECK_OK_OR_RETURN_ERROR(logits_res.error());
    executorch::aten::Tensor& logits_tensor = logits_res.get();

    prev_token = cur_token;

    stats_->on_sampling_begin();
    cur_token =
        decoder_runner_->logits_to_token(logits_tensor, metadata_.ar_len);
    stats_->on_sampling_end();

    // Update KV Cache with the output results
    kv_manager_->update_cache(metadata_.ar_len, pos, metadata_.ar_len, {});
    // Update attention mask with current position
    kv_manager_->update_attention_mask(
        attention_mask_.data, metadata_.ar_len, pos, metadata_.ar_len);
    if (metadata_.cache_mode == CacheMode::HybridCache) {
      kv_manager_->update_attention_mask(
          window_attention_mask_.data,
          metadata_.ar_len,
          pos,
          metadata_.ar_len,
          metadata_.sliding_window);
    }
    pos++;

    // print the token as string, decode it with the Tokenizer object
    token_callback(
        ET_UNWRAP_TOKENIZER(tokenizer_->decode(prev_token, cur_token)));

    // data-dependent terminating condition: we have n_eos_ number of EOS
    if (eos_ids_->count(cur_token) > 0) {
      printf("\n");
      ET_LOG(Info, "\nReached to the end of generation");
      break;
    }
  }
  return pos - start_pos;
}
// Explicit instantiations
template class TokenGenerator<uint16_t>;
template class TokenGenerator<uint8_t>;

} // namespace example
