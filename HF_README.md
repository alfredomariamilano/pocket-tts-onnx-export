---
license: cc-by-4.0
library_name: onnxruntime
base_model: kyutai/pocket-tts
tags:
  - text-to-speech
  - onnx
  - onnxruntime
  - pocket-tts
---

# ottomate/pocket-tts-ONNX

ONNX export artifacts for Pocket TTS.

## Contents

- tokenizer.json
- tokenizer_config.json
- onnx/flow_lm_main.onnx
- onnx/flow_lm_flow.onnx
- onnx/mimi_encoder.onnx
- onnx/mimi_decoder.onnx
- onnx/text_conditioner.onnx
- optional *_int8.onnx quantized variants

## Tokenizer Export

- Exported from SentencePiece `tokenizer.model` using SentencePiece Unigram.
- Special token behavior is explicit (no implicit BOS/EOS append).
- Designed for parity with raw SentencePiece IDs for ONNX text conditioner input.

## Attribution

Derived from Kyutai Pocket TTS:
https://huggingface.co/kyutai/pocket-tts
