# PocketTTS ONNX Export

This package provides a robust pipeline to export PocketTTS models to ONNX, optimized for streaming inference on diverse runtimes.

## Usage

1. Install dependencies:
   ```bash
   git submodule update --init --recursive
   uv sync
   ```

2. Run export:
   ```bash
   # Exports FP32 models to ./hf/onnx and downloads voice embeddings
   uv run python export.py

   # Exports FP32 + INT8 models
   uv run python export.py --quantize

   # Exports FP32 + q4 weight-only MatMul models
   uv run python export.py --quantize --quantize-precision q4

   # Exports FP32 + INT8 + q4 models
   uv run python export.py --quantize --quantize-precision all

   # Skip downloading voice embeddings and reference sample
   uv run python export.py --skip-embeddings

   # Export, quantize, and validate the runtime contract
   uv run python export.py --quantize --validate

   # Validate tokenizer and ONNX runtime contracts directly
   uv run python scripts/validate_onnx_contracts.py
   ```

## Output Artifacts

The pipeline generates 5 ONNX models plus tokenizer artifacts under `hf/`:

| Artifact | Description | Reasoning |
| :--- | :--- | :--- |
| `hf/onnx/text_conditioner.onnx` | Token IDs to embeddings | Lightweight text processing only. |
| `hf/onnx/mimi_encoder.onnx` | Audio to latents | Used for voice cloning reference encoding. |
| `hf/onnx/flow_lm_main.onnx` | Transformer backbone | Stateful autoregressive step with KV-cache updates. |
| `hf/onnx/flow_lm_flow.onnx` | Flow matching net | Stateless ODE step. |
| `hf/onnx/mimi_decoder.onnx` | Latents to audio | Streaming neural codec decoder. |
| `hf/tokenizer.json` | HF-compatible tokenizer JSON | Compatible with JS runtimes and `@huggingface/tokenizers`. |
| `hf/tokenizer_config.json` | Tokenizer runtime metadata | Defines model family and special token defaults. |
| `hf/embeddings/`, `hf/embeddings_v2/` | Voice-cloning assets | Safetensors are mirrored and converted to `.bin` plus `.json` sidecars. |

## Runtime Contract

- Canonical tokenizer files are `hf/tokenizer.json` and `hf/tokenizer_config.json`.
- `text_conditioner.onnx` input contract:
  - name: `token_ids`
  - dtype: `tensor(int64)`
  - shape: `[1, seq_len]`
- Runtime tokenization must use `add_special_tokens=False` and pass IDs as int64.
- Export fails if `weights/tokenizer.model` is missing so tokenizer artifacts stay canonical.

## Technical Implementation

### 1. Split-Architecture for FlowLM
Unlike the unified PyTorch model, we split the flow transformer into `main` and `flow`.

Why: the flow matching process runs an ODE solver loop multiple times. Exporting the loop as a fixed graph makes it too deterministic and prevents runtime noise injection.

Result: the runtime controls the loop so it can inject temperature noise and vary decode steps without re-exporting.

### 2. Monkeypatching for Export
The original PyTorch code uses stateful modules such as `StreamingMultiheadAttention` that manage buffers internally. ONNX requires explicit state passing from inputs to outputs.

Strategy: monkeypatch the relevant modules during export so caches and counters are exposed as distinct graph inputs and outputs.

### 3. Safe Quantization
The exporter supports two MatMul-only quantization modes for compatibility:

- Dynamic INT8 via ONNX Runtime.
- Q4 weight-only MatMul quantization via `MatMulNBitsQuantizer`.

### 4. Tokenizer Export Parity
Tokenizer export is generated from `weights/tokenizer.model` using SentencePiece Unigram and written to `hf/tokenizer.json`.

Validation command:

```bash
uv run python scripts/tokenizer_parity.py
```

## Credits & License

This project includes code from [PocketTTS](https://github.com/kyutai-labs/pocket-tts) by Kyutai Labs.

- Original code license: MIT
- Modifications: this exporter applies ONNX export wrappers and patches around upstream PocketTTS and splits the architecture for improved runtime control.

Upstream PocketTTS is included as a pinned git submodule under `third_party/pocket-tts`.

