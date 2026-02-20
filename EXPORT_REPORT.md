# Pocket-TTS Export Report (2026-02-20)

## Exact Commands Run

```powershell
$env:PYTHONIOENCODING='utf-8'; uv run python export.py --quantize 2>&1 | Tee-Object -FilePath hf/onnx/export_run.log
$env:PYTHONIOENCODING='utf-8'; uv run python scripts/tokenizer_parity.py 2>&1 | Tee-Object -FilePath hf/onnx/tokenizer_parity.log
$env:PYTHONIOENCODING='utf-8'; uv run python scripts/validate_onnx_contracts.py 2>&1 | Tee-Object -FilePath hf/onnx/validate_onnx_contracts.log
```

## Final Artifact Filenames

- `hf/tokenizer.json`
- `hf/tokenizer_config.json`
- `hf/onnx/text_conditioner.onnx`
- `hf/onnx/mimi_encoder.onnx`
- `hf/onnx/mimi_decoder.onnx`
- `hf/onnx/flow_lm_main.onnx`
- `hf/onnx/flow_lm_flow.onnx`
- `hf/onnx/text_conditioner_int8.onnx`
- `hf/onnx/mimi_encoder_int8.onnx`
- `hf/onnx/mimi_decoder_int8.onnx`
- `hf/onnx/flow_lm_main_int8.onnx`
- `hf/onnx/flow_lm_flow_int8.onnx`

## ONNX Input Signatures

Authoritative full signatures (all inputs with names/types/shapes) are in:
- `hf/onnx/validation_report.json`

Key contract inputs:

- `text_conditioner.onnx`
  - `token_ids`: `tensor(int64)`, shape `[1, seq_len]`

- `mimi_encoder.onnx`
  - `audio`: `tensor(float)`, shape `[1, 1, audio_len]`

- `mimi_decoder.onnx`
  - `latent`: `tensor(float)`, shape `[1, seq_len, 32]`
  - `state_0 ... state_N`: mixed `tensor(bool)` / `tensor(float)` / `tensor(int64)` (full list in `validation_report.json`)

- `flow_lm_main.onnx`
  - `sequence`: `tensor(float)`, shape `[1, seq_len, 32]`
  - `text_embeddings`: `tensor(float)`, shape `[1, text_len, 1024]`
  - `state_0 ... state_11`: alternating `tensor(float)` cache tensors and `tensor(int64)` step counters

- `flow_lm_flow.onnx`
  - `c`: `tensor(float)`, shape `[batch, 1024]`
  - `s`: `tensor(float)`, shape `[batch, 1]`
  - `t`: `tensor(float)`, shape `[batch, 1]`
  - `x`: `tensor(float)`, shape `[batch, 32]`

## Validation Results

- Tokenizer parity: passed (`hf/onnx/tokenizer_parity.log`)
- Tokenizer IDs are integers and packaged as `int64` `[1, seq_len]`: passed (`hf/onnx/validation_report.txt`)
- `text_conditioner.onnx` strict input contract check (`token_ids`/`tensor(int64)`): passed
- Required ONNX model inference: passed
- Quantized ONNX model inference: passed
- Sample end-to-end chain (`tokenizer -> text_conditioner -> flow_lm_main -> flow_lm_flow -> mimi_decoder`): passed

## Breaking Changes vs Previous Export

- Export now **fails hard** when `weights/tokenizer.model` is missing (no silent tokenizer-skip path).
- Tokenizer export now validates integer token IDs immediately after writing `hf/tokenizer.json`.
- `TextConditionerWrapper` now enforces strict `torch.int64` token IDs during export/verification.
