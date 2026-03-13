import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tokenizers import Tokenizer

from pocket_tts.default_parameters import DEFAULT_VARIANT
from pocket_tts.models.tts_model import TTSModel, prepare_text_prompt


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
    return {"name": value_info.name, "dtype": value_info.type, "shape": list(value_info.shape)}


def _build_dummy_input(value_info) -> np.ndarray:
    dtype = ORT_TYPE_TO_NP.get(value_info.type)
    if dtype is None:
        raise ValueError(f"Unsupported ONNX input dtype: {value_info.type} ({value_info.name})")
    shape = _resolved_shape(list(value_info.shape))
    if dtype == np.bool_ or np.issubdtype(dtype, np.integer):
        return np.zeros(shape, dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def _collect_signatures(session: ort.InferenceSession) -> dict[str, list[dict[str, object]]]:
    return {
        "inputs": [_value_info_signature(value_info) for value_info in session.get_inputs()],
        "outputs": [_value_info_signature(value_info) for value_info in session.get_outputs()],
    }


def _assert_text_conditioner_contract(session: ort.InferenceSession) -> None:
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise AssertionError(f"text_conditioner.onnx must have exactly 1 input, found {len(inputs)}")
    token_ids = inputs[0]
    if token_ids.name != "token_ids":
        raise AssertionError(f"text_conditioner input name mismatch: expected 'token_ids', got '{token_ids.name}'")
    if token_ids.type != "tensor(int64)":
        raise AssertionError(f"text_conditioner input dtype mismatch: expected tensor(int64), got {token_ids.type}")
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
    if any(input_info.name == "sequence" for input_info in session.get_inputs()) and any(
        input_info.name == "text_embeddings" for input_info in session.get_inputs()
    ):
        for input_info in session.get_inputs():
            if input_info.name == "sequence":
                feeds[input_info.name] = np.zeros((1, 1, 32), dtype=np.float32)
            elif input_info.name == "text_embeddings":
                feeds[input_info.name] = np.zeros((1, 1, 1024), dtype=np.float32)
            elif input_info.name.startswith("state_"):
                shape = _resolved_shape(list(input_info.shape))
                dtype = ORT_TYPE_TO_NP[input_info.type]
                if len(shape) == 5:
                    cache_len = max(shape[2], 2)
                    feeds[input_info.name] = np.zeros((shape[0], shape[1], cache_len, shape[3], shape[4]), dtype=dtype)
                else:
                    feeds[input_info.name] = np.zeros(shape, dtype=dtype)
            else:
                feeds[input_info.name] = _build_dummy_input(input_info)
    else:
        for input_info in session.get_inputs():
            feeds[input_info.name] = _build_dummy_input(input_info)
    session.run(None, feeds)


def _build_feeds_from_metadata(session: ort.InferenceSession, overrides: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {}
    override_map = overrides or {}
    for input_info in session.get_inputs():
        feeds[input_info.name] = override_map.get(input_info.name, _build_dummy_input(input_info))
    return feeds


def _run_end_to_end_chain(sessions: dict[str, ort.InferenceSession], token_tensor: np.ndarray) -> dict[str, object]:
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
    required_cache_len = int(text_embeddings.shape[1]) + 1
    for input_info in flow_main.get_inputs():
        if not input_info.name.startswith("state_"):
            continue
        shape = _resolved_shape(list(input_info.shape))
        dtype = ORT_TYPE_TO_NP[input_info.type]
        if len(shape) == 5:
            flow_main_feeds[input_info.name] = np.zeros(
                (shape[0], shape[1], max(shape[2], required_cache_len), shape[3], shape[4]),
                dtype=dtype,
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
        overrides={"latent": latent_step.reshape(1, 1, 32)},
    )
    mimi_decoder_outputs = mimi_decoder.run(None, mimi_decoder_feeds)

    return {
        "token_count": int(token_tensor.shape[1]),
        "text_embeddings_shape": list(text_embeddings.shape),
        "conditioning_shape": list(conditioning.shape),
        "latent_step_shape": list(latent_step.shape),
        "audio_frame_shape": list(mimi_decoder_outputs[0].shape),
    }


def _run_flow_lm_prompt_parity_check(
    tokenizer_json_path: Path,
    onnx_dir: Path,
    sample_text: str,
    builtin_voice: str,
) -> dict[str, object]:
    hf_dir = onnx_dir.parent
    voice_path = hf_dir / "embeddings_v3" / f"{builtin_voice}.safetensors"
    if not voice_path.exists():
        voice_path = hf_dir / "embeddings_v2" / f"{builtin_voice}.safetensors"
    if not voice_path.exists():
        raise FileNotFoundError(f"Built-in voice not found in embeddings_v3 or embeddings_v2: {builtin_voice}")

    tokenizer = Tokenizer.from_file(str(tokenizer_json_path))
    prepared_text, _ = prepare_text_prompt(sample_text)
    encoding = tokenizer.encode(prepared_text, add_special_tokens=False)
    token_ids_np = np.asarray([encoding.ids], dtype=np.int64)
    token_ids_torch = torch.tensor(token_ids_np, dtype=torch.int64)

    model = TTSModel.load_model(DEFAULT_VARIANT).cpu().eval()
    voice_state = model.get_state_for_audio_prompt(voice_path)
    current_end = model._flow_lm_current_end(voice_state)
    max_gen_len = model._estimate_max_gen_len(token_ids_torch.shape[1])
    required_len = current_end + token_ids_torch.shape[1] + max_gen_len

    text_session = ort.InferenceSession(str(onnx_dir / "text_conditioner.onnx"))
    flow_main_session = ort.InferenceSession(str(onnx_dir / "flow_lm_main.onnx"))
    text_embeddings = text_session.run(None, {"token_ids": token_ids_np})[0].astype(np.float32)

    onnx_state = copy.deepcopy(voice_state)
    model._expand_kv_cache(onnx_state, required_len)
    feeds = {
        "sequence": np.empty((1, 0, model.flow_lm.ldim), dtype=np.float32),
        "text_embeddings": text_embeddings,
    }
    state_names = [input_info.name for input_info in flow_main_session.get_inputs() if input_info.name.startswith("state_")]
    for index, state_name in enumerate(state_names):
        layer = index // 2
        kind = "cache" if index % 2 == 0 else "offset"
        tensor = onnx_state[f"transformer.layers.{layer}.self_attn"][kind]
        feeds[state_name] = tensor.detach().cpu().numpy()
    onnx_outputs = flow_main_session.run(None, feeds)

    python_state = copy.deepcopy(voice_state)
    model._expand_kv_cache(python_state, required_len)
    with torch.no_grad():
        model._run_flow_lm_and_increment_step(python_state, text_tokens=token_ids_torch)

    layer_metrics: list[dict[str, object]] = []
    max_cache_diff = 0.0
    for layer in range(6):
        python_cache = python_state[f"transformer.layers.{layer}.self_attn"]["cache"].detach().cpu().numpy()
        python_offset = int(python_state[f"transformer.layers.{layer}.self_attn"]["offset"].reshape(-1)[0].item())
        onnx_cache = onnx_outputs[2 + layer * 2]
        onnx_offset = int(np.asarray(onnx_outputs[3 + layer * 2]).reshape(-1)[0])
        used_python = python_cache[:, :, :python_offset]
        used_onnx = onnx_cache[:, :, :python_offset]
        diff = np.abs(np.nan_to_num(used_python, nan=0.0) - np.nan_to_num(used_onnx, nan=0.0))
        cache_max_diff = float(diff.max()) if diff.size else 0.0
        max_cache_diff = max(max_cache_diff, cache_max_diff)
        layer_metrics.append(
            {
                "layer": layer,
                "python_offset": python_offset,
                "onnx_offset": onnx_offset,
                "cache_mae": float(diff.mean()) if diff.size else 0.0,
                "cache_max": cache_max_diff,
            }
        )
        if python_offset != onnx_offset:
            raise AssertionError(
                f"flow_lm_main prompt-state offset mismatch at layer {layer}: python={python_offset} onnx={onnx_offset}"
            )

    if max_cache_diff > 1e-3:
        raise AssertionError(f"flow_lm_main prompt-state cache mismatch exceeds tolerance: max_diff={max_cache_diff}")

    return {
        "builtin_voice": builtin_voice,
        "required_cache_length": required_len,
        "max_cache_diff": max_cache_diff,
        "layers": layer_metrics,
    }


def validate(
    tokenizer_json_path: Path,
    onnx_dir: Path,
    report_json_path: Path,
    report_text_path: Path,
    sample_text: str,
    builtin_voice: str,
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
    flow_lm_prompt_parity = _run_flow_lm_prompt_parity_check(
        tokenizer_json_path=tokenizer_json_path,
        onnx_dir=onnx_dir,
        sample_text=sample_text,
        builtin_voice=builtin_voice,
    )

    quantized_models = sorted({*[path.name for path in onnx_dir.glob("*_int8.onnx")], *[path.name for path in onnx_dir.glob("*_q4.onnx")]})
    quantized_signatures: dict[str, dict[str, list[dict[str, object]]]] = {}
    quantized_inference: dict[str, str] = {}
    for model_name in quantized_models:
        session = ort.InferenceSession(str(onnx_dir / model_name))
        quantized_signatures[model_name] = _collect_signatures(session)
        _run_generic_single_inference(session)
        quantized_inference[model_name] = "ok"

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": "uv run python scripts/validate_onnx_contracts.py",
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
        "flow_lm_prompt_parity": flow_lm_prompt_parity,
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
    lines.extend(["", "Quantized Model Inference:"])
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
        "FlowLM Prompt Parity:",
        f"- Built-in voice: {flow_lm_prompt_parity['builtin_voice']}",
        f"- Required cache length: {flow_lm_prompt_parity['required_cache_length']}",
        f"- Max cache diff: {flow_lm_prompt_parity['max_cache_diff']}",
        "",
        f"JSON signature report: {report_json_path}",
    ])

    report_text_path.parent.mkdir(parents=True, exist_ok=True)
    report_text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate tokenizer and ONNX artifact contracts for Pocket-TTS runtime compatibility.")
    parser.add_argument("--tokenizer-json", default="hf/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--onnx-dir", default="hf/onnx", help="Path to ONNX artifact directory")
    parser.add_argument("--report-json", default="hf/onnx/validation_report.json", help="Path for JSON report output")
    parser.add_argument("--report-text", default="hf/onnx/validation_report.txt", help="Path for text report output")
    parser.add_argument("--sample-text", default="Pocket TTS should accept int64 token ids in ONNX runtime.", help="Sample sentence for tokenizer and conditioner validation")
    parser.add_argument(
        "--builtin-voice",
        default="marius",
        help="Built-in voice from hf/embeddings_v3/ (fallback to embeddings_v2/) used for FlowLM prompt-state parity checks",
    )
    args = parser.parse_args()

    validate(
        tokenizer_json_path=Path(args.tokenizer_json),
        onnx_dir=Path(args.onnx_dir),
        report_json_path=Path(args.report_json),
        report_text_path=Path(args.report_text),
        sample_text=args.sample_text,
        builtin_voice=args.builtin_voice,
    )
    print("ONNX and tokenizer contract validation passed")
    print(f"- Report JSON: {args.report_json}")
    print(f"- Report text: {args.report_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())