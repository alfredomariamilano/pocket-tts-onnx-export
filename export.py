import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

OUTPUT_DIR = Path("hf")
ONNX_DIR = OUTPUT_DIR / "onnx"
WEIGHTS_DIR = Path("weights")
REPO_ID = "kyutai/pocket-tts"
WEIGHTS_FILENAME = "tts_b6369a24.safetensors"
TOKENIZER_MODEL_FILENAME = "tokenizer.model"
TOKENIZER_JSON_FILENAME = "tokenizer.json"
TOKENIZER_CONFIG_FILENAME = "tokenizer_config.json"
HF_README_TEMPLATE = Path("HF_README.md")


@dataclass(frozen=True)
class _StateTensorEntry:
    source_key: str
    name: str
    dtype: str
    shape: list[int]
    offset: int
    nbytes: int


def _load_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip().strip('"').strip("'")

    env_path = Path(".env")
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != "HF_TOKEN":
            continue
        cleaned = value.strip().strip('"').strip("'")
        if cleaned:
            os.environ["HF_TOKEN"] = cleaned
            return cleaned
    return None


def install_check() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("huggingface_hub is missing. Please run: uv sync")
        sys.exit(1)


def _download_hf_file(repo_id: str, filename: str, dest_dir: Path, token: str | None) -> Path | None:
    from huggingface_hub import hf_hub_download

    try:
        cached = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / os.path.basename(filename)
        shutil.copyfile(cached, dest_path)
        return dest_path
    except Exception as exc:
        message = str(exc)
        print(f"Could not fetch {filename}: {message}")
        if "403" in message or "forbidden" in message.lower() or "fine-grained" in message.lower():
            print("Your HF_TOKEN may lack permission or you may need to accept the gated repo.")
        return None


def download_weights() -> str | None:
    print(f"\n--- Downloading weights from {REPO_ID} ---")
    WEIGHTS_DIR.mkdir(exist_ok=True)
    token = _load_hf_token()
    if token:
        print("Using HF_TOKEN from environment or .env")
    else:
        print("HF_TOKEN not found; gated repo downloads may fail.")

    weights_path = _download_hf_file(REPO_ID, WEIGHTS_FILENAME, WEIGHTS_DIR, token)
    if weights_path:
        print(f"Downloaded: {weights_path}")

    tokenizer_path = _download_hf_file(REPO_ID, TOKENIZER_MODEL_FILENAME, WEIGHTS_DIR, token)
    if tokenizer_path:
        print(f"Downloaded: {TOKENIZER_MODEL_FILENAME}")

    return token


def _convert_safetensor_to_raw(src_path: Path) -> bool:
    try:
        import numpy as np
        import safetensors.numpy

        def _dtype_to_contract(dtype: np.dtype) -> str | None:
            if dtype == np.dtype("float32"):
                return "float32"
            if dtype == np.dtype("int64"):
                return "int64"
            if dtype == np.dtype("bool"):
                return "bool"
            return None

        def _to_nested_state(tensors: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
            nested: dict[str, dict[str, np.ndarray]] = {}
            for key, value in tensors.items():
                if "/" not in key:
                    continue
                module_name, tensor_name = key.split("/", 1)
                nested.setdefault(module_name, {})[tensor_name] = value
            return nested

        def _flatten_nested_state(nested: dict[str, dict[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
            flattened: list[tuple[str, np.ndarray]] = []
            for module_name in sorted(nested.keys()):
                module_state = nested[module_name]
                for tensor_name in sorted(module_state.keys()):
                    flattened.append((f"{module_name}/{tensor_name}", module_state[tensor_name]))
            return flattened

        data = safetensors.numpy.load_file(str(src_path))
        raw_path = src_path.with_suffix(".bin")
        meta_path = src_path.with_suffix(".json")

        tensor = data.get("audio_prompt")
        if tensor is not None:
            arr = np.array(tensor, copy=False).astype("float32", copy=False)
            arr.tofile(str(raw_path))
            meta_path.write_text(json.dumps({"shape": list(arr.shape), "dtype": "float32"}) + "\n", encoding="utf-8")
            print(f"Converted {src_path.name} -> {raw_path.name} (v1 shape {arr.shape})")
            return True

        nested_state = _to_nested_state(data)
        if not nested_state:
            print(f"{src_path.name} did not match supported schemas, skipping conversion")
            return False

        flattened = _flatten_nested_state(nested_state)
        if not flattened:
            print(f"{src_path.name} had empty v2 model-state tensors, skipping conversion")
            return False

        compact_seq_len = 1000
        for _, tensor_value in flattened:
            arr = np.array(tensor_value, copy=False)
            if arr.dtype == np.dtype("int64") and arr.size > 0:
                seq_hint = int(arr.reshape(-1)[0])
                if 0 < seq_hint < compact_seq_len:
                    compact_seq_len = seq_hint

        entries: list[_StateTensorEntry] = []
        total_bytes = 0
        parts: list[bytes] = []

        for index, (source_key, tensor_value) in enumerate(flattened):
            arr = np.array(tensor_value, copy=False)

            if source_key.endswith("/current_end") and arr.ndim == 1:
                arr = np.asarray([arr.shape[0]], dtype=np.int64)

            contract_dtype = _dtype_to_contract(arr.dtype)
            if contract_dtype is None:
                print(f"Unsupported dtype for {source_key}: {arr.dtype}, skipping conversion")
                return False

            if contract_dtype == "float32" and arr.ndim >= 3 and arr.shape[2] == 1000 and 0 < compact_seq_len < arr.shape[2]:
                arr = arr[:, :, :compact_seq_len, ...]

            contiguous = np.ascontiguousarray(arr)
            blob = contiguous.tobytes(order="C")
            nbytes = len(blob)

            entries.append(
                _StateTensorEntry(
                    source_key=source_key,
                    name=f"state_{index}",
                    dtype=contract_dtype,
                    shape=[int(dim) for dim in contiguous.shape],
                    offset=total_bytes,
                    nbytes=nbytes,
                )
            )
            parts.append(blob)
            total_bytes += nbytes

        with open(raw_path, "wb") as raw_file:
            for part in parts:
                raw_file.write(part)

        meta = {
            "format": "pocket_tts_flow_state_v2",
            "tensor_count": len(entries),
            "total_bytes": total_bytes,
            "tensors": [
                {
                    "name": entry.name,
                    "source_key": entry.source_key,
                    "dtype": entry.dtype,
                    "shape": entry.shape,
                    "offset": entry.offset,
                    "nbytes": entry.nbytes,
                }
                for entry in entries
            ],
        }
        meta_path.write_text(json.dumps(meta) + "\n", encoding="utf-8")
        print(f"Converted {src_path.name} -> {raw_path.name} (v2 tensors: {len(entries)}, bytes: {total_bytes})")
        return True
    except Exception as exc:
        print(f"Failed to convert {src_path.name} to raw: {exc}")
        return False


def download_voice_embeddings(repo_id: str, output_dir: Path, token: str | None) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token) if token else HfApi()
    prefixes = ["embeddings/", "embeddings_v2/", "embeddings_v3/"]

    try:
        all_files = api.list_repo_files(repo_id=repo_id)
        print(f"repository {repo_id} contains {len(all_files)} files")
    except Exception as exc:
        print(f"Failed to list files for {repo_id}: {exc}")
        return

    for prefix in prefixes:
        matched = [path for path in all_files if path.startswith(prefix)]
        if not matched:
            print(f"No files found under '{prefix}'")
            continue

        target_dir = output_dir / prefix
        skipped: list[str] = []
        for repo_path in matched:
            dest = _download_hf_file(repo_id, repo_path, target_dir, token)
            if dest:
                print(f"Downloaded embedding: {repo_path} -> {dest}")
                if dest.suffix == ".safetensors" and not _convert_safetensor_to_raw(dest):
                    skipped.append(dest.name)
        if skipped:
            print("The following safetensors failed conversion:")
            for name in skipped:
                print(f"  - {name}")


def export_tokenizer_json() -> None:
    tokenizer_model_path = WEIGHTS_DIR / TOKENIZER_MODEL_FILENAME
    tokenizer_json_path = OUTPUT_DIR / TOKENIZER_JSON_FILENAME
    tokenizer_config_path = OUTPUT_DIR / TOKENIZER_CONFIG_FILENAME
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not tokenizer_model_path.exists():
        raise FileNotFoundError(
            f"Required tokenizer source file is missing: {tokenizer_model_path}. Cannot regenerate canonical tokenizer artifacts."
        )

    import sentencepiece as spm
    import sentencepiece.sentencepiece_model_pb2 as sp_pb2
    from tokenizers import SentencePieceUnigramTokenizer, Tokenizer

    sys.modules.setdefault("sentencepiece_model_pb2", sp_pb2)
    tokenizer = SentencePieceUnigramTokenizer.from_spm(str(tokenizer_model_path))
    tokenizer.normalizer = None
    tokenizer.save(str(tokenizer_json_path))
    # ensure compact JSON output for the tokenizer definition
    try:
        data = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))
        # no trailing newline; JSON parsers don't require it
        tokenizer_json_path.write_text(
            json.dumps(data, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    except Exception:  # keep original if parsing fails
        pass

    print(f"Exported tokenizer JSON: {tokenizer_json_path}")

    hf_tokenizer = Tokenizer.from_file(str(tokenizer_json_path))
    encoded = hf_tokenizer.encode("Pocket TTS ONNX int64 tokenizer validation.", add_special_tokens=False)
    if not all(isinstance(token_id, int) for token_id in encoded.ids):
        raise TypeError("Exported tokenizer produced non-integer token IDs.")
    print("Tokenizer ID type validation passed")

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_model_path))

    def _safe_piece(piece_id: int, fallback: str | None) -> str | None:
        if piece_id < 0:
            return fallback
        return sp.id_to_piece(piece_id)

    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 4096,
        "padding_side": "right",
        "truncation_side": "right",
        "model_input_names": ["input_ids", "attention_mask"],
        "unk_token": _safe_piece(sp.unk_id(), "<unk>"),
        "pad_token": _safe_piece(sp.pad_id(), "<pad>"),
        "eos_token": _safe_piece(sp.eos_id(), "</s>"),
        "bos_token": _safe_piece(sp.bos_id(), None),
        "add_bos_token": False,
        "add_eos_token": False,
        "legacy": False,
    }
    tokenizer_config_path.write_text(json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported tokenizer config: {tokenizer_config_path}")


def export_hf_readme(template_path: Path = HF_README_TEMPLATE) -> None:
    if not template_path.exists():
        print(f"README template not found: {template_path}. Skipping model card export.")
        return

    readme_path = OUTPUT_DIR / "README.md"

    OUTPUT_DIR.mkdir(exist_ok=True)
    shutil.copyfile(template_path, readme_path)
    print(f"Exported model card: {readme_path}")


def download_reference_sample(repo_id: str, output_dir: Path, token: str | None) -> None:
    dest = _download_hf_file(repo_id, "reference_sample.wav", output_dir, token)
    if dest:
        print(f"Downloaded reference sample: {dest}")
    else:
        print("reference_sample.wav not available")


def clone_audio_prompts_to_embeddings(output_dir: Path) -> None:
    """Clone local audio prompt files into HF embeddings_v3 as safetensors.

    This allows local voice prompts under ./audio_prompts to be treated like the
    built-in voice cloning assets in `hf/embeddings_v3/`.
    """

    prompt_dir = Path("audio_prompts")
    if not prompt_dir.exists() or not prompt_dir.is_dir():
        return

    audio_files = sorted(
        [
            p
            for p in prompt_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".safetensors")
        ]
    )
    if not audio_files:
        return

    try:
        from pocket_tts import TTSModel, export_model_state
    except Exception as exc:
        print(f"Skipping audio prompt cloning: failed to import pocket_tts ({exc})")
        return

    try:
        model = TTSModel.load_model()
    except Exception as exc:
        print(f"Skipping audio prompt cloning: failed to load pocket-tts model ({exc})")
        return

    dest_dir = output_dir / "embeddings_v3"
    dest_dir.mkdir(parents=True, exist_ok=True)

    for audio_file in audio_files:
        dest_safetensors = dest_dir / f"{audio_file.stem}.safetensors"
        try:
            if audio_file.suffix.lower() == ".safetensors":
                shutil.copyfile(audio_file, dest_safetensors)
                print(f"Copied existing safetensors prompt: {audio_file.name}")
            else:
                voice_state = model.get_state_for_audio_prompt(str(audio_file))
                export_model_state(voice_state, str(dest_safetensors))
                print(f"Exported audio prompt state: {dest_safetensors}")
            _convert_safetensor_to_raw(dest_safetensors)
        except Exception as exc:
            print(f"Failed to clone audio prompt {audio_file.name}: {exc}")


def run_export_scripts(skip_embeddings: bool = False) -> None:
    print("\n--- Running export scripts ---")
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    if not skip_embeddings:
        token = _load_hf_token()
        download_voice_embeddings(REPO_ID, OUTPUT_DIR, token)
        download_reference_sample(REPO_ID, OUTPUT_DIR, token)
    else:
        print("Skipping voice embeddings and reference sample download")

    clone_audio_prompts_to_embeddings(OUTPUT_DIR)

    weights_path = str(WEIGHTS_DIR / WEIGHTS_FILENAME)
    output_dir_str = str(ONNX_DIR)

    cmd1 = [sys.executable, "-m", "scripts.export_mimi_and_conditioner", "--output_dir", output_dir_str, "--weights_path", weights_path]
    subprocess.run(cmd1, check=True)
    print("Mimi and conditioner export succeeded")

    cmd2 = [sys.executable, "-m", "scripts.export_flow_lm", "--output_dir", output_dir_str, "--weights_path", weights_path]
    subprocess.run(cmd2, check=True)
    print("FlowLM export succeeded")


def run_quantization(precision: str = "int8", q4_block_size: int = 128) -> None:
    print("\n--- Running quantization ---")
    if not any(ONNX_DIR.glob("*.onnx")):
        print("No models found in output directory to quantize.")
        return

    cmd = [
        sys.executable,
        "-m",
        "scripts.quantize",
        "--input_dir",
        str(ONNX_DIR),
        "--output_dir",
        str(ONNX_DIR),
        "--precision",
        precision,
        "--q4-block-size",
        str(q4_block_size),
    ]
    subprocess.run(cmd, check=True)
    print(f"Quantization succeeded in: {ONNX_DIR.absolute()}")


def run_full_validation() -> None:
    print("\n--- Running full contract validation ---")
    subprocess.run([sys.executable, "-m", "scripts.validate_onnx_contracts"], check=True)
    print("Full contract validation succeeded")


def print_summary() -> None:
    print(f"\nAll done. Artifacts are in: {OUTPUT_DIR.absolute()}")
    if not OUTPUT_DIR.exists():
        print("Output directory missing.")
        return

    output_files: list[Path] = list(ONNX_DIR.glob("*.onnx"))
    for filename in (TOKENIZER_JSON_FILENAME, TOKENIZER_CONFIG_FILENAME):
        path = OUTPUT_DIR / filename
        if path.exists():
            output_files.append(path)
    for subdir_name in ("embeddings", "embeddings_v2", "embeddings_v3"):
        subdir = OUTPUT_DIR / subdir_name
        if subdir.exists():
            output_files.extend(sorted(path for path in subdir.iterdir() if path.is_file()))
    output_files.extend(sorted(OUTPUT_DIR.glob("*.wav")))

    for path in sorted(output_files):
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f" - {path.relative_to(OUTPUT_DIR)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified export script for PocketTTS")
    parser.add_argument("--quantize", action="store_true", help="Run quantization after export")
    parser.add_argument(
        "--quantize-precision",
        choices=["int8", "q4", "all"],
        default="int8",
        help="Quantization precision profile when --quantize is enabled",
    )
    parser.add_argument("--q4-block-size", type=int, default=128, help="Block size for q4 weight-only quantization")
    parser.add_argument("--validate", action="store_true", help="Run tokenizer and ONNX contract validation")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip downloading voice embeddings and reference sample")
    args = parser.parse_args()

    install_check()
    download_weights()
    export_hf_readme()
    export_tokenizer_json()
    run_export_scripts(skip_embeddings=args.skip_embeddings)

    if args.quantize:
        run_quantization(precision=args.quantize_precision, q4_block_size=args.q4_block_size)
    if args.validate:
        run_full_validation()

    print_summary()
