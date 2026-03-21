# tokenizers
C++ implementations for various tokenizers (sentencepiece, tiktoken etc). Useful for other PyTorch repos such as torchchat, ExecuTorch to build LLM runners using ExecuTorch stack or AOT Inductor stack.

## Installation (from source)
```
git clone git@github.com:meta-pytorch/tokenizers.git
cd ~/tokenizers
git submodule update --init --recursive
pip install -e .
```

## SentencePiece tokenizer
Depend on https://github.com/google/sentencepiece from Google.

## Tiktoken tokenizer
Adapted from https://github.com/sewenew/tokenizer.

## Huggingface tokenizer
Compatible with https://github.com/huggingface/tokenizers/.

## Llama2.c tokenizer
Adapted from https://github.com/karpathy/llama2.c.

## Tekken tokenizer
Mistral's Tekken tokenizer (v7) with full support for special tokens, multilingual text, and instruction-tuned conversations. Provides significant efficiency gains for AI workloads:
- **Special token recognition**: [INST], [/INST], [AVAILABLE_TOOLS], etc. as single tokens
- **Multilingual support**: Complete Unicode handling including emojis and complex scripts
- **Production-ready**: 100% decode accuracy with comprehensive test coverage
- **Python bindings**: Full compatibility with mistral-common ecosystem

## License

tokenizers is released under the [BSD 3 license](LICENSE). (Additional
code in this distribution is covered by the MIT and Apache Open Source
licenses.) However you may have other legal obligations that govern
your use of content, such as the terms of service for third-party
models.
