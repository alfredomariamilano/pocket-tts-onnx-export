import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


REQUIRED_MODELS = [
    "text_conditioner.onnx",
    "mimi_encoder.onnx",
    "mimi_decoder.onnx",
    "flow_lm_main.onnx",
    "flow_lm_flow.onnx",
]


ORT_TYPE_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int16)": np.int16,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


def _resolved_shape(shape: list[object]) -> list[int]:
    resolved: list[int] = []
    for dim in shape:
        if isinstance(dim, int) and dim >= 0:
            resolved.append(dim)
        else:
            resolved.append(1)
    return resolved


def _value_info_signature(value_info) -> dict[str, object]:
    return {
        "name": value_info.name,
        "dtype": value_info.type,
        "shape": list(value_info.shape),
    }


def _build_dummy_input(value_info) -> np.ndarray:
    dtype = ORT_TYPE_TO_NP.get(value_info.type)
    if dtype is None:
        raise ValueError(f"Unsupported ONNX input dtype: {value_info.type} ({value_info.name})")

    shape = _resolved_shape(list(value_info.shape))

    if dtype == np.bool_:
        return np.zeros(shape, dtype=dtype)

    if np.issubdtype(dtype, np.integer):
        return np.zeros(shape, dtype=dtype)

    return np.zeros(shape, dtype=dtype)


def _collect_signatures(session: ort.InferenceSession) -> dict[str, list[dict[str, object]]]:
    return {
        "inputs": [_value_info_signature(v) for v in session.get_inputs()],
        "outputs": [_value_info_signature(v) for v in session.get_outputs()],
    }


def _assert_text_conditioner_contract(session: ort.InferenceSession) -> None:
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise AssertionError(f"text_conditioner.onnx must have exactly 1 input, found {len(inputs)}")

    token_ids = inputs[0]
    if token_ids.name != "token_ids":
        raise AssertionError(f"text_conditioner input name mismatch: expected 'token_ids', got '{token_ids.name}'")
    if token_ids.type != "tensor(int64)":
        raise AssertionError(
            f"text_conditioner input dtype mismatch: expected tensor(int64), got {token_ids.type}"
        )

    shape = list(token_ids.shape)
    if len(shape) != 2:
        raise AssertionError(f"text_conditioner token_ids must be rank-2 [1, seq_len], got {shape}")
    if shape[0] != 1:
        raise AssertionError(f"text_conditioner token_ids batch dim must be 1, got {shape[0]}")


def _validate_tokenizer_and_get_tokens(tokenizer_json_path: Path, sample_text: str) -> np.ndarray:
    tokenizer = Tokenizer.from_file(str(tokenizer_json_path))
    encoding = tokenizer.encode(sample_text, add_special_tokens=False)

    if not all(isinstance(token_id, int) for token_id in encoding.ids):
        raise AssertionError("Tokenizer produced non-integer token IDs")

    token_ids = np.asarray(encoding.ids, dtype=np.int64)
    if token_ids.ndim != 1:
        raise AssertionError(f"Expected 1D token list before batching, got ndim={token_ids.ndim}")

    batched = token_ids.reshape(1, -1)
    if batched.size == 0:
        raise AssertionError("Tokenizer produced an empty token sequence for sample text")

    return batched


def _run_text_conditioner(session: ort.InferenceSession, token_tensor: np.ndarray) -> np.ndarray:
    outputs = session.run(None, {"token_ids": token_tensor})
    if len(outputs) != 1:
        raise AssertionError(f"Expected 1 output from text_conditioner.onnx, got {len(outputs)}")
    return outputs[0]


def _run_generic_single_inference(session: ort.InferenceSession) -> None:
    feeds: dict[str, np.ndarray] = {}
    for input_info in session.get_inputs():
        feeds[input_info.name] = _build_dummy_input(input_info)
    session.run(None, feeds)


def _build_feeds_from_metadata(session: ort.InferenceSession, overrides: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {}
    override_map = overrides or {}
    for input_info in session.get_inputs():
        if input_info.name in override_map:
            feeds[input_info.name] = override_map[input_info.name]
        else:
            feeds[input_info.name] = _build_dummy_input(input_info)
    return feeds


def _run_end_to_end_chain(
    sessions: dict[str, ort.InferenceSession],
    token_tensor: np.ndarray,
) -> dict[str, object]:
    conditioner = sessions["text_conditioner.onnx"]
    flow_main = sessions["flow_lm_main.onnx"]
    flow_flow = sessions["flow_lm_flow.onnx"]
    mimi_decoder = sessions["mimi_decoder.onnx"]

    text_embeddings = _run_text_conditioner(conditioner, token_tensor).astype(np.float32)

    flow_main_feeds = _build_feeds_from_metadata(
        flow_main,
        overrides={
            "sequence": np.zeros((1, 1, 32), dtype=np.float32),
            "text_embeddings": text_embeddings,
        },
    )
    flow_main_outputs = flow_main.run(None, flow_main_feeds)

    conditioning = flow_main_outputs[0].astype(np.float32)

    flow_flow_feeds = _build_feeds_from_metadata(
        flow_flow,
        overrides={
            "c": conditioning,
            "s": np.array([[0.0]], dtype=np.float32),
            "t": np.array([[1.0]], dtype=np.float32),
            "x": np.zeros((1, 32), dtype=np.float32),
        },
    )
    latent_step = flow_flow.run(None, flow_flow_feeds)[0].astype(np.float32)

    mimi_decoder_feeds = _build_feeds_from_metadata(
        mimi_decoder,
        overrides={
            "latent": latent_step.reshape(1, 1, 32),
        },
    )
    mimi_decoder_outputs = mimi_decoder.run(None, mimi_decoder_feeds)

    return {
        "token_count": int(token_tensor.shape[1]),
        "text_embeddings_shape": list(text_embeddings.shape),
        "conditioning_shape": list(conditioning.shape),
        "latent_step_shape": list(latent_step.shape),
        "audio_frame_shape": list(mimi_decoder_outputs[0].shape),
    }


def validate(
    tokenizer_json_path: Path,
    onnx_dir: Path,
    report_json_path: Path,
    report_text_path: Path,
    sample_text: str,
) -> dict[str, object]:
    if not tokenizer_json_path.exists():
        raise FileNotFoundError(f"Tokenizer JSON not found: {tokenizer_json_path}")

    if not onnx_dir.exists():
        raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}")

    token_tensor = _validate_tokenizer_and_get_tokens(tokenizer_json_path, sample_text)

    sessions: dict[str, ort.InferenceSession] = {}
    signatures: dict[str, dict[str, list[dict[str, object]]]] = {}
    model_inference: dict[str, str] = {}

    for model_name in REQUIRED_MODELS:
        model_path = onnx_dir / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Required ONNX model not found: {model_path}")

        session = ort.InferenceSession(str(model_path))
        sessions[model_name] = session
        signatures[model_name] = _collect_signatures(session)

    _assert_text_conditioner_contract(sessions["text_conditioner.onnx"])

    text_conditioner_out = _run_text_conditioner(sessions["text_conditioner.onnx"], token_tensor)
    model_inference["text_conditioner.onnx"] = f"ok: output_shape={list(text_conditioner_out.shape)}"

    for model_name, session in sessions.items():
        if model_name == "text_conditioner.onnx":
            continue
        _run_generic_single_inference(session)
        model_inference[model_name] = "ok"

    e2e = _run_end_to_end_chain(sessions, token_tensor)

    quantized_models = sorted(p.name for p in onnx_dir.glob("*_int8.onnx"))
    quantized_signatures: dict[str, dict[str, list[dict[str, object]]]] = {}
    quantized_inference: dict[str, str] = {}
    for model_name in quantized_models:
        session = ort.InferenceSession(str(onnx_dir / model_name))
        quantized_signatures[model_name] = _collect_signatures(session)
        _run_generic_single_inference(session)
        quantized_inference[model_name] = "ok"

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python scripts/validate_onnx_contracts.py",
        "tokenizer": {
            "path": str(tokenizer_json_path),
            "sample_text": sample_text,
            "token_count": int(token_tensor.shape[1]),
            "ids_are_integers": True,
            "token_tensor_dtype": str(token_tensor.dtype),
            "token_tensor_shape": list(token_tensor.shape),
        },
        "contract_checks": {
            "text_conditioner_input_name": "token_ids",
            "text_conditioner_input_dtype": "tensor(int64)",
            "text_conditioner_input_shape": "[1, seq_len]",
        },
        "required_model_signatures": signatures,
        "required_model_inference": model_inference,
        "quantized_model_signatures": quantized_signatures,
        "quantized_model_inference": quantized_inference,
        "end_to_end_sample": e2e,
    }

    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "Pocket-TTS ONNX Contract Validation Report",
        f"Timestamp (UTC): {report['timestamp_utc']}",
        f"Command: {report['command']}",
        "",
        "Tokenizer Checks:",
        f"- Path: {tokenizer_json_path}",
        f"- Sample text: {sample_text}",
        f"- IDs are integers: {report['tokenizer']['ids_are_integers']}",
        f"- Token tensor dtype: {report['tokenizer']['token_tensor_dtype']}",
        f"- Token tensor shape: {report['tokenizer']['token_tensor_shape']}",
        "",
        "Required Model Inference:",
    ]

    for model_name, status in model_inference.items():
        lines.append(f"- {model_name}: {status}")

    lines.extend([
        "",
        "Quantized Model Inference:",
    ])

    if quantized_inference:
        for model_name, status in quantized_inference.items():
            lines.append(f"- {model_name}: {status}")
    else:
        lines.append("- None found")

    lines.extend([
        "",
        "End-to-End Sample:",
        f"- Token count: {e2e['token_count']}",
        f"- Text embeddings shape: {e2e['text_embeddings_shape']}",
        f"- Conditioning shape: {e2e['conditioning_shape']}",
        f"- Latent step shape: {e2e['latent_step_shape']}",
        f"- Audio frame shape: {e2e['audio_frame_shape']}",
        "",
        f"JSON signature report: {report_json_path}",
    ])

    report_text_path.parent.mkdir(parents=True, exist_ok=True)
    report_text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate tokenizer + ONNX artifact contracts for Pocket-TTS Electron runtime compatibility."
    )
    parser.add_argument(
        "--tokenizer-json",
        default="hf/tokenizer.json",
        help="Path to tokenizer.json",
    )
    parser.add_argument(
        "--onnx-dir",
        default="hf/onnx",
        help="Path to ONNX artifact directory",
    )
    parser.add_argument(
        "--report-json",
        default="hf/onnx/validation_report.json",
        help="Path for JSON report output",
    )
    parser.add_argument(
        "--report-text",
        default="hf/onnx/validation_report.txt",
        help="Path for text report output",
    )
    parser.add_argument(
        "--sample-text",
        default="Pocket TTS should accept int64 token ids in ONNX runtime.",
        help="Sample sentence for tokenizer + conditioner validation",
    )
    args = parser.parse_args()

    validate(
        tokenizer_json_path=Path(args.tokenizer_json),
        onnx_dir=Path(args.onnx_dir),
        report_json_path=Path(args.report_json),
        report_text_path=Path(args.report_text),
        sample_text=args.sample_text,
    )

    print("✅ ONNX/tokenizer contract validation passed")
    print(f"- Report JSON: {args.report_json}")
    print(f"- Report text: {args.report_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
