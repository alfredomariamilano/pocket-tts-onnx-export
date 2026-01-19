# PocketTTS ONNX Export

This package provides a robust pipeline to export **PocketTTS** models to ONNX (FP32 and INT8), optimized for streaming inference on diverse runtimes (Web, Edge, CPU).

## Usage

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Export**:
    ```bash
    # Exports FP32 models to ./onnx
    python export.py

    # Exports FP32 AND INT8 models (recommended)
    python export.py --quantize
    ```

## Output Artifacts

The pipeline generates **5 distinct ONNX models** in the `onnx/` directory:

| Model | Description | Reasoning |
| :--- | :--- | :--- |
| **`text_conditioner.onnx`** | Tokenizer → Embeddings | Lightweight text processing only. |
| **`mimi_encoder.onnx`** | Audio → Latents | Used for voice cloning (reference audio encoding). |
| **`flow_lm_main.onnx`** | Transformer Backbone | **Stateful AR**. Updates KV-cache and steps. Returns conditioning vector. |
| **`flow_lm_flow.onnx`** | Flow Matching Net | **Stateless**. Solves the ODE step ($v = f(x, t, c)$). |
| **`mimi_decoder.onnx`** | Latents → Audio | Streaming neural codec decoder. |

## Technical Implementation

### 1. Split-Architecture for FlowLM
Unlike the unified PyTorch model, we split the **Flow Transformer** into two parts (`main` and `flow`):
*   **Why?** The Flow Matching process involves a loop (ODE solver) that runs multiple times (e.g., 10 steps). Exporting this as a fixed graph forces extreme determinism, preventing noise injection.
*   **Result**: By splitting `main` (backbone) and `flow` (solver step), we move the control loop to the **Runtime (Python/JS)**. This allows us to:
    1.  **Inject Temperature**: Add noise at each solver step, which is **critical** for breaking deterministic loops and ensuring the model correctly fires the **EOS (End of Sentence)** token.
    2.  **Adjust Quality**: Change the number of LSD steps dynamically without re-exporting.

### 2. Monkeypatching for Export
The original PyTorch code uses stateful modules (e.g., `StreamingMultiheadAttention`) that manage their own buffers. ONNX requires explicit state passing (Inputs $\rightarrow$ Outputs).
*   **Strategy**: We "monkeypatch" these classes during export to expose their internal caches (KV-cache, counters) as distinct **Input/Output** graph nodes. This makes the models purely functional and side-effect free, which is essential for `onnxruntime`.

### 3. Safe Quantization
We use **Dynamic Quantization** targeting `MatMul` (Matrix Multiplication) operators only.
*   **Why?** This ensures broad compatibility (e.g., older CPUs, WebAssembly) and avoids issues with specific operators (like `ConvInteger`) that can be problematic or slow on certain execution providers. It reduces model size by ~70% with negligible quality loss.

## Credits & License

This project includes code from **[PocketTTS](https://github.com/kyutai-labs/pocket-tts)** by **Kyutai Labs**.

*   **Original Code License**: MIT
*   **Modifications**: This exporter package modifies the original source to support ONNX tracing (via monkeypatching and wrappers) and splits the architecture for improved runtime control.

Code from the `pocket_tts` module is redistributed here under the terms of the MIT License.

