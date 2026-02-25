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


def _dummy_input(info) -> np.ndarray:
    dtype = ORT_TYPE_TO_NP[info.type]
    shape = _resolved_shape(list(info.shape))
    if np.issubdtype(dtype, np.integer) or dtype == np.bool_:
        return np.zeros(shape, dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def _encode_tokens(tokenizer_json: Path, text: str) -> np.ndarray:
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    return np.asarray(ids, dtype=np.int64).reshape(1, -1)


def run_monolith_step(session: ort.InferenceSession, token_ids: np.ndarray, steps: int) -> dict[str, object]:
    inputs = {inp.name: _dummy_input(inp) for inp in session.get_inputs()}
    inputs["token_ids"] = token_ids
    inputs["flow_main/sequence"] = np.full((1, 1, 32), np.nan, dtype=np.float32)
    inputs["flow_flow/s"] = np.array([[0.0]], dtype=np.float32)
    inputs["flow_flow/t"] = np.array([[1.0]], dtype=np.float32)

    total_audio_samples = 0
    eos_values: list[float] = []
    audio_frames: list[np.ndarray] = []

    for step in range(steps):
        # sample fresh noise each step (Gaussian) to drive the model
        inputs["flow_flow/x"] = np.random.randn(1, 32).astype(np.float32)
        outputs = session.run(None, inputs)
        output_map = {out.name: arr for out, arr in zip(session.get_outputs(), outputs, strict=False)}

        audio = output_map["mimi/audio_frame"]
        eos = output_map["flow_main/eos_logit"]
        next_latent = output_map["adapter/latent"]

        total_audio_samples += int(audio.shape[-1])
        audio_frames.append(audio.reshape(-1))
        eos_values.append(float(eos.reshape(-1)[0]))

        for idx in range(12):
            inputs[f"flow_main/state_{idx}"] = output_map[f"flow_main/out_state_{idx}"]
        for idx in range(52):
            inputs[f"mimi/state_{idx}"] = output_map[f"mimi/out_state_{idx}"]

        inputs["flow_main/sequence"] = next_latent.astype(np.float32)

    return {
        "steps": steps,
        "total_audio_samples": total_audio_samples,
        "last_eos_logit": eos_values[-1] if eos_values else None,
        "eos_logits": eos_values,
        "audio_frames": audio_frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Runtime smoke test for monolith ONNX exports")
    parser.add_argument("--onnx-dir", default="hf/onnx", help="ONNX directory")
    parser.add_argument("--tokenizer-json", default="hf/tokenizer.json", help="Tokenizer JSON path")
    parser.add_argument("--text", default="Testing single-session monolith runtime.", help="Prompt text")
    parser.add_argument("--steps", type=int, default=4, help="Number of autoregressive steps")
    parser.add_argument("--output-wav", default="runtime_fp32.wav", help="Filename for written audio")
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    tokenizer_json = Path(args.tokenizer_json)

    # make sure any component exports have the expand-shape fix applied;
    # this mirrors what our pytest harness does so that long-run failures
    # appear consistently.
    from onnx_export.export_utils import fix_expand_shapes
    for fname in ("flow_lm_main.onnx", "flow_lm_flow.onnx", "mimi_decoder.onnx"):
        p = onnx_dir / fname
        if p.exists():
            fix_expand_shapes(model_path=str(p))

    token_ids = _encode_tokens(tokenizer_json, args.text)

    step_path = onnx_dir / "monolith_step.onnx"
    if not step_path.exists():
        raise FileNotFoundError(f"Missing {step_path}")

    step_session = ort.InferenceSession(str(step_path))
    summary = run_monolith_step(step_session, token_ids=token_ids, steps=args.steps)

    print("✅ Monolith-step runtime test passed")
    print(f"- Steps: {summary['steps']}")
    print(f"- Total audio samples: {summary['total_audio_samples']}")
    print(f"- Last EOS logit: {summary['last_eos_logit']}")
    # write concatenated audio to file
    import soundfile as sf
    audio_concat = np.concatenate(summary.get('audio_frames', [])) if summary.get('audio_frames') else np.zeros(0, dtype=np.float32)
    if audio_concat.size > 0:
        outname = args.output_wav
        sf.write(outname, audio_concat, 24000)
        print(f'Wrote {outname}')

    ar_path = onnx_dir / "monolith_ar.onnx"
    if ar_path.exists():
        try:
            ar_session = ort.InferenceSession(str(ar_path))
            ar_inputs = {inp.name: _dummy_input(inp) for inp in ar_session.get_inputs()}
            ar_inputs["token_ids"] = token_ids
            ar_inputs["noise"] = np.zeros((1, args.steps, 32), dtype=np.float32)
            ar_inputs["frames_after_eos"] = np.array([[2]], dtype=np.int64)
            ar_outputs = ar_session.run(None, ar_inputs)
            print("✅ Monolith-AR runtime test passed")
            print(f"- Output count: {len(ar_outputs)}")
        except Exception as exc:
            print(f"⚠️ Monolith-AR runtime test failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
