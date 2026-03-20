import argparse
from typing import Dict

import torch
from transformers import Qwen2_5_VLForConditionalGeneration

from torchtune.models.convert_weights import get_mapped_key

# Standard _FROM_META weight mapping of Meta weights to TorchTune + additional bias weight mappings.
_QWEN_2_FROM_META = {
    "tok_embeddings.weight": "embed_tokens.weight",
    "norm.weight": "norm.weight",
    "layers.{}.attention.wk.weight": "layers.{}.self_attn.k_proj.weight",
    "layers.{}.attention.wk.bias": "layers.{}.self_attn.k_proj.bias",
    "layers.{}.attention.wq.weight": "layers.{}.self_attn.q_proj.weight",
    "layers.{}.attention.wq.bias": "layers.{}.self_attn.q_proj.bias",
    "layers.{}.attention.wv.weight": "layers.{}.self_attn.v_proj.weight",
    "layers.{}.attention.wv.bias": "layers.{}.self_attn.v_proj.bias",
    "layers.{}.attention.wo.weight": "layers.{}.self_attn.o_proj.weight",
    "layers.{}.attention_norm.weight": "layers.{}.input_layernorm.weight",
    "layers.{}.ffn_norm.weight": "layers.{}.post_attention_layernorm.weight",
    "layers.{}.feed_forward.w1.weight": "layers.{}.mlp.gate_proj.weight",
    "layers.{}.feed_forward.w2.weight": "layers.{}.mlp.down_proj.weight",
    "layers.{}.feed_forward.w3.weight": "layers.{}.mlp.up_proj.weight",
    "visual.blocks.{}.attn.proj.bias": "visual.blocks.{}.attn.o_proj.bias",
    "visual.blocks.{}.attn.proj.weight": "visual.blocks.{}.attn.o_proj.weight",
    "visual.blocks.{}.attn.qkv.weight": "visual.blocks.{}.attn.qkv_proj.weight",
    "visual.blocks.{}.attn.qkv.bias": "visual.blocks.{}.attn.qkv_proj.bias",
    "visual.blocks.{}.mlp.gate_proj.weight": "visual.blocks.{}.mlp.w1.weight",
    "visual.blocks.{}.mlp.up_proj.weight": "visual.blocks.{}.mlp.w3.weight",
    "visual.blocks.{}.mlp.down_proj.weight": "visual.blocks.{}.mlp.w2.weight",
    "visual.blocks.{}.mlp.gate_proj.bias": "visual.blocks.{}.mlp.w1.bias",
    "visual.blocks.{}.mlp.up_proj.bias": "visual.blocks.{}.mlp.w3.bias",
    "visual.blocks.{}.mlp.down_proj.bias": "visual.blocks.{}.mlp.w2.bias",
    "visual.blocks.{}.norm1.weight": "visual.blocks.{}.norm1.scale",
    "visual.blocks.{}.norm2.weight": "visual.blocks.{}.norm2.scale",
    "visual.merger.ln_q.weight": "visual.merger.ln_q.scale",
    "visual.merger.mlp.{}.weight": "visual.merger.mlp.{}.weight",
    "visual.merger.mlp.{}.bias": "visual.merger.mlp.{}.bias",
    "visual.patch_embed.proj.weight": "visual.patch_embed.proj.weight",
    "visual.blocks.{}.attn.output_proj.bias": "visual.blocks.{}.attn.output_proj.bias"
}


def qwen_2_tune_to_meta(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from torchtune's format to Meta's format. This function
    doesn't handle any sharding or splitting of state dicts. It follows the
    state_dict IN -> state_dict OUT pattern.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in torchtune's format.

    Returns:
        Dict[str, torch.Tensor]: State dict in Meta's format.
    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _QWEN_2_FROM_META.items()}

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, inverted_mapping_dict)
        converted_state_dict[new_key] = value

    # 0.5b and 1.5b models share the same weights for tok_embeddings and output embeddings, see https://github.com/QwenLM/Qwen2.5/issues/733.
    converted_state_dict["output.weight"] = converted_state_dict[
        "tok_embeddings.weight"
    ]

    return converted_state_dict

def convert_weights(input_dir: str, output_file: str) -> None:
    print("Loading model using transformers...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(input_dir, torch_dtype=torch.float32)

    print("Extracting state_dict...")
    state_dict = model.model.language_model.state_dict()

    print("Converting state_dict...")
    converted_state_dict = qwen_2_tune_to_meta(state_dict)

    print("Saving converted state_dict...")
    torch.save(converted_state_dict, output_file)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen2 weights to Meta format."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to directory containing checkpoint files",
    )
    parser.add_argument("output", type=str, help="Path to the output checkpoint")

    args = parser.parse_args()
    convert_weights(args.input_dir, args.output)


if __name__ == "__main__":
    main()
