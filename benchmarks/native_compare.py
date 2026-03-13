from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch

from pocket_tts import TTSModel


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "comparison_outputs"
CLONE_PROMPT_PATH = OUTPUT_DIR / "clone_prompt.wav"
SPEAKER_PROJ_BIN_PATH = OUTPUT_DIR / "speaker_proj_weight.bin"
SPEAKER_PROJ_JSON_PATH = OUTPUT_DIR / "speaker_proj_weight.json"
BUILTIN_OUTPUT_PATH = OUTPUT_DIR / "python_builtin.wav"
CLONE_OUTPUT_PATH = OUTPUT_DIR / "python_voice_clone.wav"
REPORT_PATH = OUTPUT_DIR / "python_native_report.json"

# Shared defaults (kept in benchmarks/values.json so Python and Node use identical values)
VALUES_PATH = ROOT / "benchmarks" / "values.json"
try:
    _VALUES = json.loads(VALUES_PATH.read_text())
except Exception:
    _VALUES = {}

DEFAULT_BUILTIN_VOICE = _VALUES.get("defaultBuiltinVoice", "marius")
CLONE_PROMPT_VOICE = _VALUES.get("clonePromptVoice", "alba")
CLONE_PROMPT_TEXT = _VALUES.get(
    "clonePromptText",
    "This short clip becomes the voice cloning reference for the runtime comparison.",
)
DEFAULT_TEXT = _VALUES.get(
    "defaultText",
    "This is a direct quality comparison between the ONNX Node runtime and the native Python Pocket TTS runtime.",
)


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_audio(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    sf.write(path, audio.detach().cpu().numpy(), sample_rate)


def save_float32_tensor(path_stem: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
    path_stem.with_suffix(".bin").write_bytes(array.tobytes())
    path_stem.with_suffix(".json").write_text(
        json.dumps({"shape": list(array.shape), "dtype": "float32"}) + "\n",
        encoding="utf-8",
    )


def timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


def prepare_clone_prompt(model: TTSModel) -> dict[str, float | str]:
    if CLONE_PROMPT_PATH.exists():
        return {"path": str(CLONE_PROMPT_PATH), "created": False}

    voice_state, state_ms = timed_call(model.get_state_for_audio_prompt, CLONE_PROMPT_VOICE)
    audio, generate_ms = timed_call(model.generate_audio, voice_state, CLONE_PROMPT_TEXT)
    save_audio(CLONE_PROMPT_PATH, audio, model.sample_rate)
    return {
        "path": str(CLONE_PROMPT_PATH),
        "created": True,
        "stateMs": state_ms,
        "generateMs": generate_ms,
    }


def export_speaker_projection(model: TTSModel) -> str:
    save_float32_tensor(OUTPUT_DIR / "speaker_proj_weight", model.flow_lm.speaker_proj_weight)
    return str(SPEAKER_PROJ_BIN_PATH)


def run_case(model: TTSModel, voice: str | Path, text: str, output_path: Path) -> dict[str, float | str]:
    voice_state, state_ms = timed_call(model.get_state_for_audio_prompt, voice)
    audio, generate_ms = timed_call(model.generate_audio, voice_state, text)
    save_audio(output_path, audio, model.sample_rate)
    return {
        "voice": str(voice),
        "statePreparationMs": state_ms,
        "generationMs": generate_ms,
        "audioSeconds": audio.shape[-1] / model.sample_rate,
        "outputPath": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run native Pocket TTS comparison for built-in and cloned voices.")
    parser.add_argument("--builtin-voice", default=DEFAULT_BUILTIN_VOICE)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--prepare-clone-prompt-only", action="store_true")
    args = parser.parse_args()

    ensure_output_dir()
    model, load_ms = timed_call(TTSModel.load_model)
    clone_prompt_info = prepare_clone_prompt(model)
    speaker_proj_path = export_speaker_projection(model)

    if args.prepare_clone_prompt_only:
        print(
            json.dumps(
                {
                    "modelLoadMs": load_ms,
                    "clonePrompt": clone_prompt_info,
                    "speakerProjWeight": speaker_proj_path,
                },
                indent=2,
            )
        )
        return 0

    builtin = run_case(model, args.builtin_voice, args.text, BUILTIN_OUTPUT_PATH)
    voice_clone = run_case(model, CLONE_PROMPT_PATH, args.text, CLONE_OUTPUT_PATH)

    report = {
        "modelLoadMs": load_ms,
        "builtinVoice": args.builtin_voice,
        "comparisonText": args.text,
        "clonePrompt": clone_prompt_info,
        "speakerProjWeight": speaker_proj_path,
        "builtin": builtin,
        "voiceClone": voice_clone,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())