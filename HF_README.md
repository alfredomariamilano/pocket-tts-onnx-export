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
- embeddings/ (and embeddings_v2/) – voice‑cloning safetensors downloaded from upstream.  Each safetensors file is also converted in-place to a plain `.bin` payload plus a small JSON describing the tensor shape, enabling direct use by the Pocket‑TTS runtime.  These auxiliary files share the same stem (e.g. `foo.bin` and `foo.json`).
- reference_sample.wav – optional audio used as default voice prompt by runtime

## Tokenizer Export

- Exported from SentencePiece `tokenizer.model` using SentencePiece Unigram.
- Special token behavior is explicit (no implicit BOS/EOS append).
- Designed for parity with raw SentencePiece IDs for ONNX text conditioner input.
- Runtime contract for `onnx/text_conditioner.onnx`:
  - input name: `token_ids`
  - input dtype: `tensor(int64)`
  - input shape: `[1, seq_len]`
  - tokenize with `add_special_tokens=false`, then cast/pack to int64 `[1, seq_len]`.

Validation command:

```bash
uv run python scripts/validate_onnx_contracts.py
```

Validation outputs:

- `onnx/validation_report.json`
- `onnx/validation_report.txt`

## Attribution

Derived from Kyutai Pocket TTS:
https://huggingface.co/kyutai/pocket-tts
