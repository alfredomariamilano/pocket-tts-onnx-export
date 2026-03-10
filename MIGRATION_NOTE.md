# Migration Note

The exporter now treats `hf/tokenizer.json` and `hf/tokenizer_config.json` as canonical runtime artifacts.

Key runtime contract changes:

- `text_conditioner.onnx` input name is `token_ids`.
- `token_ids` must be `int64` with shape `[1, seq_len]`.
- Runtime tokenization must use `add_special_tokens=False`.
- Export now fails if `weights/tokenizer.model` is missing, instead of silently skipping tokenizer artifact generation.

Validation commands:

```bash
uv run python scripts/tokenizer_parity.py
uv run python scripts/validate_onnx_contracts.py
```