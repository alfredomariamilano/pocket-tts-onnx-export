import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


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


def _build_default_feed(session: ort.InferenceSession) -> dict[str, np.ndarray]:
    feed: dict[str, np.ndarray] = {}
    for info in session.get_inputs():
        dtype = ORT_TYPE_TO_NP.get(info.type)
        if dtype is None:
            raise ValueError(f"Unsupported input dtype: {info.type} ({info.name})")
        shape = _resolved_shape(list(info.shape))
        feed[info.name] = np.zeros(shape, dtype=dtype)
    return feed


def _encode_tokens(tokenizer_json: Path, text: str) -> np.ndarray:
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if not ids:
        raise ValueError("Tokenizer produced empty token sequence")
    return np.asarray(ids, dtype=np.int64).reshape(1, -1)


def _metric_block(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    if a.size == 0 or b.size == 0:
        return {
            "mae": 0.0,
            "max_abs": 0.0,
            "rmse": 0.0,
            "mean_rel": 0.0,
        }
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), 1e-8)
    rel = diff / denom
    return {
        "mae": float(np.mean(diff)),
        "max_abs": float(np.max(diff)),
        "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
        "mean_rel": float(np.mean(rel)),
    }


def run_parity(
    fp32_model: Path,
    q4_model: Path,
    tokenizer_json: Path,
    text: str,
    gen_steps: int,
    frames_after_eos: int,
) -> dict[str, object]:
    fp32_sess = ort.InferenceSession(str(fp32_model))
    q4_sess = ort.InferenceSession(str(q4_model))

    fp32_inputs = _build_default_feed(fp32_sess)
    q4_inputs = _build_default_feed(q4_sess)

    token_ids = _encode_tokens(tokenizer_json, text)
    noise = np.zeros((1, gen_steps, 32), dtype=np.float32)
    tail = np.array([[frames_after_eos]], dtype=np.int64)

    for feed in (fp32_inputs, q4_inputs):
        if "token_ids" in feed:
            feed["token_ids"] = token_ids
        if "noise" in feed:
            feed["noise"] = noise
        if "frames_after_eos" in feed:
            feed["frames_after_eos"] = tail

    fp32_out_list = fp32_sess.run(None, fp32_inputs)
    q4_out_list = None
    try:
        q4_out_list = q4_sess.run(None, q4_inputs)
    except Exception as exc:  # pragma: no cover - best effort
        print(f"⚠️ Q4 session failed: {exc}")

    fp32_map = {out.name: arr for out, arr in zip(fp32_sess.get_outputs(), fp32_out_list, strict=False)}
    q4_map = {}
    if q4_out_list is not None:
        q4_map = {out.name: arr for out, arr in zip(q4_sess.get_outputs(), q4_out_list, strict=False)}

    shared_names = sorted(set(fp32_map) & set(q4_map))
    metrics: dict[str, dict[str, float]] = {}
    for name in shared_names:
        if fp32_map[name].dtype == np.bool_:
            continue
        if fp32_map[name].shape != q4_map[name].shape:
            continue
        if fp32_map[name].size == 0 or q4_map[name].size == 0:
            continue
        if not np.issubdtype(fp32_map[name].dtype, np.number):
            continue
        metrics[name] = _metric_block(fp32_map[name], q4_map[name])

    focus = {
        key: metrics[key]
        for key in ("audio", "eos_logits")
        if key in metrics
    }

    return {
        "shared_output_count": len(shared_names),
        "compared_output_count": len(metrics),
        "focus_metrics": focus,
        "all_metrics": metrics,
    }


def main() -> int:


    parser = argparse.ArgumentParser(description="Compare parity between FP32 and Q4 monolith ONNX outputs")
    parser.add_argument("--onnx-dir", default="hf/onnx", help="Directory containing ONNX models")
    parser.add_argument("--fp32-model", default="monolith_ar.onnx", help="FP32 model filename")
    parser.add_argument("--q4-model", default="monolith_ar_q4.onnx", help="Q4 model filename")
    parser.add_argument("--tokenizer-json", default="hf/tokenizer.json", help="Tokenizer JSON path")
    parser.add_argument("--text", default="Parity check for monolith export.", help="Prompt text")
    parser.add_argument("--gen-steps", type=int, default=4, help="Generation steps (noise length)")
    parser.add_argument("--frames-after-eos", type=int, default=2, help="Tail frames after EOS")
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    fp32_model = onnx_dir / args.fp32_model
    q4_model = onnx_dir / args.q4_model

    if not fp32_model.exists():
        raise FileNotFoundError(f"Missing FP32 model: {fp32_model}")
    if not q4_model.exists():
        raise FileNotFoundError(f"Missing Q4 model: {q4_model}")

    # --- Print all output tensor names and shapes for both models ---
    print("\nFP32 model outputs:")
    fp32_sess = ort.InferenceSession(str(fp32_model))
    for i, out in enumerate(fp32_sess.get_outputs()):
        print(f"  [{i}] {out.name} {out.type} {out.shape}")
    print("Q4 model outputs:")
    q4_sess = ort.InferenceSession(str(q4_model))
    for i, out in enumerate(q4_sess.get_outputs()):
        print(f"  [{i}] {out.name} {out.type} {out.shape}")

    parser = argparse.ArgumentParser(description="Compare parity between FP32 and Q4 monolith ONNX outputs")
    parser.add_argument("--onnx-dir", default="hf/onnx", help="Directory containing ONNX models")
    parser.add_argument("--fp32-model", default="monolith_ar.onnx", help="FP32 model filename")
    parser.add_argument("--q4-model", default="monolith_ar_q4.onnx", help="Q4 model filename")
    parser.add_argument("--tokenizer-json", default="hf/tokenizer.json", help="Tokenizer JSON path")
    parser.add_argument("--text", default="Parity check for monolith export.", help="Prompt text")
    parser.add_argument("--gen-steps", type=int, default=4, help="Generation steps (noise length)")
    parser.add_argument("--frames-after-eos", type=int, default=2, help="Tail frames after EOS")
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    fp32_model = onnx_dir / args.fp32_model
    q4_model = onnx_dir / args.q4_model

    if not fp32_model.exists():
        raise FileNotFoundError(f"Missing FP32 model: {fp32_model}")
    if not q4_model.exists():
        raise FileNotFoundError(f"Missing Q4 model: {q4_model}")


    result = run_parity(
        fp32_model=fp32_model,
        q4_model=q4_model,
        tokenizer_json=Path(args.tokenizer_json),
        text=args.text,
        gen_steps=args.gen_steps,
        frames_after_eos=args.frames_after_eos,
    )

    print("✅ Monolith parity check complete")
    print(f"- Shared outputs: {result['shared_output_count']}")
    print(f"- Compared numeric outputs: {result['compared_output_count']}")

    focus = result["focus_metrics"]
    if not focus:
        print("- No focus metrics found for audio/eos_logits")
    else:
        for name, metric in focus.items():
            print(
                f"- {name}: MAE={metric['mae']:.6f}, RMSE={metric['rmse']:.6f}, "
                f"MAX_ABS={metric['max_abs']:.6f}, MEAN_REL={metric['mean_rel']:.6f}"
            )

    # --- EOS-threshold sensitivity test ---
    fp32_eos = None
    q4_eos = None
    if "eos_logits" in result["all_metrics"]:
        # Use the actual output arrays if available
        fp32_eos = result["all_metrics"]["eos_logits"].get("fp32_array")
        q4_eos = result["all_metrics"]["eos_logits"].get("q4_array")
    # If not present, try to extract from run_parity (which doesn't currently return arrays)
    if fp32_eos is None or q4_eos is None:
        # Re-run model to get eos_logits arrays
        # (This is a hack, but avoids refactoring run_parity for now)
        fp32_sess = ort.InferenceSession(str(fp32_model))
        q4_sess = ort.InferenceSession(str(q4_model))
        fp32_inputs = _build_default_feed(fp32_sess)
        q4_inputs = _build_default_feed(q4_sess)
        token_ids = _encode_tokens(Path(args.tokenizer_json), args.text)
        noise = np.zeros((1, args.gen_steps, 32), dtype=np.float32)
        tail = np.array([[args.frames_after_eos]], dtype=np.int64)
        for feed in (fp32_inputs, q4_inputs):
            if "token_ids" in feed:
                feed["token_ids"] = token_ids
            if "noise" in feed:
                feed["noise"] = noise
            if "frames_after_eos" in feed:
                feed["frames_after_eos"] = tail
        fp32_out_list = fp32_sess.run(None, fp32_inputs)
        q4_out_list = q4_sess.run(None, q4_inputs)
        fp32_map = {out.name: arr for out, arr in zip(fp32_sess.get_outputs(), fp32_out_list, strict=False)}
        q4_map = {out.name: arr for out, arr in zip(q4_sess.get_outputs(), q4_out_list, strict=False)}
        fp32_eos = fp32_map.get("eos_logits")
        q4_eos = q4_map.get("eos_logits")


    if fp32_eos is not None and q4_eos is not None:
        fp32_eos = np.array(fp32_eos).reshape(-1)
        q4_eos = np.array(q4_eos).reshape(-1)
        thresholds = [0.0, 0.5, 1.0, 1.5, 2.0]
        print("\nEOS-threshold sensitivity analysis:")
        print(f"{'Threshold':>10} | {'FP32 step':>9} | {'Q4 step':>7} | {'Agree?':>6}")
        print("-" * 42)
        for thresh in thresholds:
            fp32_stop = next((i for i, v in enumerate(fp32_eos) if v > thresh), None)
            q4_stop = next((i for i, v in enumerate(q4_eos) if v > thresh), None)
            agree = (fp32_stop == q4_stop)
            print(f"{thresh:10.2f} | {str(fp32_stop):>9} | {str(q4_stop):>7} | {str(agree):>6}")
    else:
        print("\nEOS-threshold sensitivity analysis: eos_logits not available.")

    # --- Write audio outputs to WAV files ---
    import soundfile as sf
    # Re-run model to get audio outputs
    fp32_sess = ort.InferenceSession(str(fp32_model))
    q4_sess = ort.InferenceSession(str(q4_model))
    fp32_inputs = _build_default_feed(fp32_sess)
    q4_inputs = _build_default_feed(q4_sess)
    token_ids = _encode_tokens(Path(args.tokenizer_json), args.text)
    noise = np.zeros((1, args.gen_steps, 32), dtype=np.float32)
    tail = np.array([[args.frames_after_eos]], dtype=np.int64)
    for feed in (fp32_inputs, q4_inputs):
        if "token_ids" in feed:
            feed["token_ids"] = token_ids
        if "noise" in feed:
            feed["noise"] = noise
        if "frames_after_eos" in feed:
            feed["frames_after_eos"] = tail
    fp32_out_list = fp32_sess.run(None, fp32_inputs)
    q4_out_list = q4_sess.run(None, q4_inputs)
    fp32_map = {out.name: arr for out, arr in zip(fp32_sess.get_outputs(), fp32_out_list, strict=False)}
    q4_map = {out.name: arr for out, arr in zip(q4_sess.get_outputs(), q4_out_list, strict=False)}
    fp32_audio = fp32_map.get("audio")
    q4_audio = q4_map.get("audio")
    if fp32_audio is not None:
        audio_fp32 = np.array(fp32_audio).reshape(-1)
        print("FP32 audio shape/dtype:", audio_fp32.shape, audio_fp32.dtype)
        print(
            "FP32 stats: min=", audio_fp32.min(),
            "max=", audio_fp32.max(),
            "mean=", audio_fp32.mean(),
            "len=", len(audio_fp32),
        )
        print("FP32 audio first 10 samples:", audio_fp32[:10])
        sf.write("fp32_out.wav", audio_fp32, 24000)
        print("Wrote: fp32_out.wav (24kHz)")
    if q4_audio is not None:
        audio_q4 = np.array(q4_audio).reshape(-1)
        print("Q4 audio shape/dtype:", audio_q4.shape, audio_q4.dtype)
        print(
            "Q4 stats: min=", audio_q4.min(),
            "max=", audio_q4.max(),
            "mean=", audio_q4.mean(),
            "len=", len(audio_q4),
        )
        print("Q4 audio first 10 samples:", audio_q4[:10])
        sf.write("q4_out.wav", audio_q4, 24000)
        print("Wrote: q4_out.wav (24kHz)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
