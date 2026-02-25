import numpy as np
import onnxruntime as ort
from pathlib import Path
from onnx_export.export_utils import fix_expand_shapes


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
    return [dim if isinstance(dim, int) and dim >= 0 else 1 for dim in shape]


def _dummy_input(info) -> np.ndarray:
    dtype = ORT_TYPE_TO_NP[info.type]
    shape = _resolved_shape(list(info.shape))
    if np.issubdtype(dtype, np.integer) or dtype == np.bool_:
        return np.zeros(shape, dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def _encode_tokens(tokenizer_json: Path, text: str) -> np.ndarray:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    return np.asarray(ids, dtype=np.int64).reshape(1, -1)


def run_monolith_step(session: ort.InferenceSession, token_ids: np.ndarray, steps: int) -> dict:
    inputs = {inp.name: _dummy_input(inp) for inp in session.get_inputs()}
    inputs["token_ids"] = token_ids
    inputs["flow_main/sequence"] = np.full((1, 1, 32), np.nan, dtype=np.float32)
    inputs["flow_flow/s"] = np.array([[0.0]], dtype=np.float32)
    inputs["flow_flow/t"] = np.array([[1.0]], dtype=np.float32)

    total_audio_samples = 0
    eos_values = []
    audio_frames = []

    for step in range(steps):
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
    }


def _patch_components(onnx_dir: Path):
    # ensure flow and mimi exports have been patched
    for fname in (
        "flow_lm_main.onnx",
        "flow_lm_flow.onnx",
        "mimi_decoder.onnx",
    ):
        path = onnx_dir / fname
        if path.exists():
            fix_expand_shapes(model_path=str(path))


def test_monolith_step_long_run(tmp_path):
    onnx_dir = Path("hf/onnx")
    _patch_components(onnx_dir)

    step_path = onnx_dir / "monolith_step.onnx"
    assert step_path.exists(), "monolith_step.onnx is missing"

    session = ort.InferenceSession(str(step_path))
    tokens = _encode_tokens(Path("hf/tokenizer.json"), "hello world")
    summary = run_monolith_step(session, token_ids=tokens, steps=100)

    # sanity checks
    assert summary["steps"] == 100
    assert summary["total_audio_samples"] > 0
    assert summary["last_eos_logit"] is not None


def test_ar_export_is_optional(tmp_path):
    # ensure the script still works if the AR model is absent
    onnx_dir = Path("hf/onnx")
    ar_path = onnx_dir / "monolith_ar.onnx"
    if ar_path.exists():
        ar_path.unlink()

    # calling the script should not raise
    import subprocess

    subprocess.check_call([
        "uv", "run", "-m", "scripts.test_monolith_runtime",
        "--steps", "1",
    ], cwd=Path(__file__).parents[1])
