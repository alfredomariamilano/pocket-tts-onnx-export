# Migration Note: Tokenizer + ONNX Runtime Contract

This export set enforces a strict runtime contract for Electron/ONNX integration.

## Artifact Contract

- Canonical tokenizer artifacts are:
  - `hf/tokenizer.json`
  - `hf/tokenizer_config.json`
- `text_conditioner.onnx` input is exactly:
  - `token_ids`
  - dtype: `tensor(int64)`
  - shape: `[1, seq_len]` (dynamic `seq_len`)
- Runtime must pass integer token IDs (no float token tensors).
- Downstream ONNX files keep previous model roles and interfaces:
  - `flow_lm_main.onnx`
  - `flow_lm_flow.onnx`
  - `mimi_encoder.onnx`
  - `mimi_decoder.onnx`
  - optional `*_int8.onnx` quantized variants

## Expected Runtime Input Formatting

1. Load `hf/tokenizer.json`.
2. Encode text with `add_special_tokens=false`.
3. Convert IDs to `int64` and reshape to `[1, seq_len]`.
4. Call `text_conditioner.onnx` with `{ token_ids: int64_tensor }`.

## Breaking Changes vs Previous Export Behavior

- Export now fails if `weights/tokenizer.model` is missing (previously tokenizer regeneration could be skipped).
- Export path includes explicit integer-ID checks for tokenizer output.
- `TextConditionerWrapper` enforces `torch.int64` tokens during export/verification.

## Validation

Run:

```bash
uv run python scripts/validate_onnx_contracts.py
```

Outputs:

- `hf/onnx/validation_report.json`
- `hf/onnx/validation_report.txt`
