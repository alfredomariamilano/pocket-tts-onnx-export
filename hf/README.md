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

- tokenizer.json (for @huggingface/tokenizers compatibility)
- onnx/flow_lm_main.onnx
- onnx/flow_lm_flow.onnx
- onnx/mimi_encoder.onnx
- onnx/mimi_decoder.onnx
- onnx/text_conditioner.onnx
- optional *_int8.onnx quantized variants

## Attribution

This repository is derived from Kyutai's Pocket TTS model:
https://huggingface.co/kyutai/pocket-tts

## License

This model card uses the same license field as the upstream model card:
cc-by-4.0

Also follow upstream usage conditions and prohibited-use policy documented at:
https://huggingface.co/kyutai/pocket-tts
