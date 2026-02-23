import os
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Constants
OUTPUT_DIR = Path("hf")
ONNX_DIR = OUTPUT_DIR / "onnx"
WEIGHTS_DIR = Path("weights")
SCRIPTS_DIR = Path("scripts")
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

def install_check():
    try:
        import huggingface_hub
        print("✅ huggingface_hub is installed.")
    except ImportError:
        print("❌ huggingface_hub is missing. Please run: uv sync")
        sys.exit(1)

def _download_hf_file(repo_id: str, filename: str, dest_dir: Path, token: str | None) -> Path | None:
    """Download *filename* from a HF repo into *dest_dir* without polluting it.

    Uses the global HuggingFace cache (default) and copies the downloaded asset
    into the provided destination folder.  Returns the final path or ``None`` on
    failure (warning printed).
    """
    from huggingface_hub import hf_hub_download

    try:
        cached = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / os.path.basename(filename)
        shutil.copyfile(cached, dest_path)
        return dest_path
    except Exception as e:
        # hf_hub_download errors can be noisy.  We always print a one-line
        # summary including the original message; this makes debugging easier
        # while avoiding Groceries™-length traces in the main export log.
        msg = str(e)
        print(f"⚠️ Could not fetch {filename}: {msg}")
        if '403' in msg or 'forbidden' in msg.lower() or 'fine-grained' in msg.lower():
            print("    → your HF_TOKEN may lack permission or you must accept the gated repo")
        return None


def download_weights():
    print(f"\n--- Downloading Weights from {REPO_ID} ---")
    WEIGHTS_DIR.mkdir(exist_ok=True)
    token = _load_hf_token()
    if token:
        print("✅ Using HF_TOKEN from environment/.env")
    else:
        print("⚠️ HF_TOKEN not found; gated repo downloads may fail.")

    # download the main weights file
    path = _download_hf_file(REPO_ID, WEIGHTS_FILENAME, WEIGHTS_DIR, token)
    if path:
        print(f"✅ Downloaded: {path}")

    # Attempt tokeniser file as well
    tokpath = _download_hf_file(REPO_ID, TOKENIZER_MODEL_FILENAME, WEIGHTS_DIR, token)
    if tokpath:
        print(f"✅ Downloaded: {TOKENIZER_MODEL_FILENAME}")
    
    return token


def _convert_safetensor_to_raw(src_path: Path) -> bool:
    """Convert a voice safetensors file into runtime-friendly raw payloads.

    Supported schemas:
    - v1: single ``audio_prompt`` tensor -> ``.bin`` + ``{"shape", "dtype"}``
    - v2: flattened model-state tensors (``module/key`` names) -> ``.bin`` +
      manifest with deterministic ``state_{i}`` entries for ONNX state seeding.
    """
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

        def _flatten_nested_state(
            nested: dict[str, dict[str, np.ndarray]],
        ) -> list[tuple[str, np.ndarray]]:
            flattened: list[tuple[str, np.ndarray]] = []
            for module_name in sorted(nested.keys()):
                module_state = nested[module_name]
                for tensor_name in sorted(module_state.keys()):
                    flattened.append((f"{module_name}/{tensor_name}", module_state[tensor_name]))
            return flattened

        data = safetensors.numpy.load_file(str(src_path))
        raw_path = src_path.with_suffix(".bin")
        meta_path = src_path.with_suffix(".json")

        # v1: single embedding tensor.
        tensor = data.get("audio_prompt")
        if tensor is not None:
            arr = np.array(tensor, copy=False).astype("float32", copy=False)
            arr.tofile(str(raw_path))

            meta = {"shape": list(arr.shape), "dtype": "float32"}
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)

            print(f"✅ Converted {src_path.name} -> {raw_path.name} (v1 shape {arr.shape})")
            return True

        # v2: flattened model-state tensors serialized as module/key.
        nested_state = _to_nested_state(data)
        if not nested_state:
            print(
                f"⚠️ {src_path.name} did not match supported schemas (v1 audio_prompt or v2 model-state keys), skipping conversion"
            )
            return False

        flattened = _flatten_nested_state(nested_state)
        if not flattened:
            print(f"⚠️ {src_path.name} had empty v2 model-state tensors, skipping conversion")
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
                print(
                    f"⚠️ {src_path.name} has unsupported dtype for {source_key}: {arr.dtype}, skipping conversion"
                )
                return False

            if (
                contract_dtype == "float32"
                and arr.ndim >= 3
                and arr.shape[2] == 1000
                and 0 < compact_seq_len < arr.shape[2]
            ):
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

        v2_meta = {
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

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(v2_meta, f)

        print(
            f"✅ Converted {src_path.name} -> {raw_path.name} (v2 state tensors: {len(entries)}, bytes: {total_bytes})"
        )
        return True
    except Exception as e:
        print(f"⚠️ Failed to convert {src_path.name} to raw: {e}")
        return False


def download_voice_embeddings(repo_id: str, output_dir: Path, token: str | None):
    """Download all files under the `embeddings` and `embeddings_v2`
    prefixes from the given HF repo into subdirectories of *output_dir*.

    Files are downloaded from HF using the shared cache to avoid duplication.
    After each successful download we attempt to convert any ``.safetensors``
    asset into a plain ``.bin`` file understood by the Pocket‑TTS runtime.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token) if token else HfApi()
    prefixes = ["embeddings/", "embeddings_v2/"]

    try:
        # list_repo_files returns paths relative to repo root
        all_files = api.list_repo_files(repo_id=repo_id)
        print(f"ℹ️ repository {repo_id} contains {len(all_files)} files")
    except Exception as e:
        print(f"⚠️ Failed to list files for {repo_id}: {e}")
        return

    for prefix in prefixes:
        matched = [p for p in all_files if p.startswith(prefix)]
        if not matched:
            print(f"⚠️ No files found under '{prefix}'")
            continue

        target_dir = output_dir / prefix
        skipped = []  # filenames that could not be converted
        for path in matched:
            dest = _download_hf_file(repo_id, path, target_dir, token)
            if dest:
                print(f"✅ Downloaded embedding: {path} -> {dest}")
                # if we got a safetensors file, convert it so runtime can use it
                if dest.suffix == ".safetensors":
                    ok = _convert_safetensor_to_raw(dest)
                    if not ok:
                        skipped.append(dest.name)
            # helper already printed any failure message
        if skipped:
            print(
                "⚠️ The following safetensors did not match supported schemas or failed to convert:",
            )
            for name in skipped:
                print(f"    - {name}")


def export_tokenizer_json():

    tokenizer_model_path = WEIGHTS_DIR / TOKENIZER_MODEL_FILENAME
    tokenizer_json_path = OUTPUT_DIR / TOKENIZER_JSON_FILENAME
    tokenizer_config_path = OUTPUT_DIR / TOKENIZER_CONFIG_FILENAME
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not tokenizer_model_path.exists():
        raise FileNotFoundError(
            f"Required tokenizer source file is missing: {tokenizer_model_path}. "
            "Cannot regenerate canonical tokenizer artifacts."
        )

    try:
        import sys

        import sentencepiece as spm
        import sentencepiece.sentencepiece_model_pb2 as sp_pb2
        from tokenizers import SentencePieceUnigramTokenizer

        sys.modules.setdefault("sentencepiece_model_pb2", sp_pb2)

        tokenizer = SentencePieceUnigramTokenizer.from_spm(str(tokenizer_model_path))
        tokenizer.normalizer = None
        tokenizer.save(str(tokenizer_json_path))
        print(f"✅ Exported tokenizer JSON: {tokenizer_json_path}")

        from tokenizers import Tokenizer

        hf_tokenizer = Tokenizer.from_file(str(tokenizer_json_path))
        sample_text = "Pocket TTS ONNX int64 tokenizer validation."
        encoded = hf_tokenizer.encode(sample_text, add_special_tokens=False)
        if not all(isinstance(token_id, int) for token_id in encoded.ids):
            raise TypeError(
                "Exported tokenizer produced non-integer token IDs. "
                "This violates ONNX token_ids int64 compatibility."
            )
        print("✅ Tokenizer ID type validation passed (all integer IDs)")

        sp = spm.SentencePieceProcessor(model_file=str(tokenizer_model_path))

        def _safe_piece(piece_id: int, fallback: str | None) -> str | None:
            if piece_id < 0:
                return fallback
            return sp.id_to_piece(piece_id)

        unk_token = _safe_piece(sp.unk_id(), "<unk>")
        pad_token = _safe_piece(sp.pad_id(), "<pad>")
        eos_token = _safe_piece(sp.eos_id(), "</s>")
        bos_token = _safe_piece(sp.bos_id(), None)

        tokenizer_config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": 4096,
            "padding_side": "right",
            "truncation_side": "right",
            "model_input_names": ["input_ids", "attention_mask"],
            "unk_token": unk_token,
            "pad_token": pad_token,
            "eos_token": eos_token,
            "bos_token": bos_token,
            "add_bos_token": False,
            "add_eos_token": False,
            "legacy": False,
        }
        tokenizer_config_path.write_text(
            json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"✅ Exported tokenizer config: {tokenizer_config_path}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to export tokenizer artifacts ({TOKENIZER_JSON_FILENAME}, "
            f"{TOKENIZER_CONFIG_FILENAME}): {e}"
        ) from e


def export_hf_readme(template_path: Path = HF_README_TEMPLATE):
    if not template_path.exists():
        print(f"⚠️ README template not found: {template_path}. Skipping model card export.")
        return

    readme_path = OUTPUT_DIR / "README.md"
    OUTPUT_DIR.mkdir(exist_ok=True)
    shutil.copyfile(template_path, readme_path)
    print(f"✅ Exported model card: {readme_path}")

def download_reference_sample(repo_id: str, output_dir: Path, token: str | None):
    """Attempt to fetch a `reference_sample.wav` from the HF repo.

    This file is useful as a fallback voice prompt on the runtime side
    and is expected by downstream JS/TS helpers.  Failure to obtain it
    simply results in a warning.
    """
    print("🔍 checking for reference_sample.wav in repo")
    dest = _download_hf_file(repo_id, "reference_sample.wav", output_dir, token)
    if dest:
        print(f"✅ Downloaded reference sample: {dest}")
    else:
        print("⚠️ reference_sample.wav not available")


def run_export_scripts(skip_embeddings: bool = False):
    print(f"\n--- Running Export Scripts ---")
    
    # Ensure output directory exists
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    
    # download voice embeddings and optional reference sample
    if not skip_embeddings:
        token = _load_hf_token()
        download_voice_embeddings(REPO_ID, OUTPUT_DIR, token)
        download_reference_sample(REPO_ID, OUTPUT_DIR, token)
    else:
        print("⚠️ Skipping voice embedding download as requested.")
        print("⚠️ Skipping reference sample download as requested.")
    
    # Common arguments
    weights_path = str(WEIGHTS_DIR / WEIGHTS_FILENAME)
    output_dir_str = str(ONNX_DIR)
    
    # 1. Export Mimi & Conditioner
    print("\n[1/2] Exporting Mimi & Text Conditioner...")
    cmd1 = [
        sys.executable,
        "-m",
        "scripts.export_mimi_and_conditioner",
        "--output_dir", output_dir_str,
        "--weights_path", weights_path
    ]
    try:
        subprocess.run(cmd1, check=True)
        print("✅ Mimi/Conditioner Export Success")
    except subprocess.CalledProcessError:
        print("❌ Mimi/Conditioner Export Failed")
        sys.exit(1)

    # 2. Export FlowLM
    print("\n[2/2] Exporting FlowLM (Split Models)...")
    cmd2 = [
        sys.executable,
        "-m",
        "scripts.export_flow_lm",
        "--output_dir", output_dir_str,
        "--weights_path", weights_path
    ]
    try:
        subprocess.run(cmd2, check=True)
        print("✅ FlowLM Export Success")
    except subprocess.CalledProcessError:
        print("❌ FlowLM Export Failed")
        sys.exit(1)

def run_quantization(precision: str = "int8", q4_block_size: int = 128):
    print(f"\n--- [Optional] Running Quantization ---")
    
    # Check if input directory has models
    if not any(ONNX_DIR.glob("*.onnx")):
        print("⚠️ No models found in output directory to quantize.")
        return

    # Quantization Output Directory (Same as input for PocketTTS compatibility)
    QUANT_DIR = ONNX_DIR
    
    cmd = [
        sys.executable,
        "-m",
        "scripts.quantize",
        "--input_dir", str(ONNX_DIR),
        "--output_dir", str(QUANT_DIR),
        "--precision", precision,
        "--q4-block-size", str(q4_block_size),
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Quantization Success! INT8 models in: {QUANT_DIR.absolute()}")
    except subprocess.CalledProcessError:
        print("❌ Quantization Failed")


def run_full_validation():
    print("\n--- Running Full Contract Validation ---")
    cmd = [
        sys.executable,
        "-m",
        "scripts.validate_onnx_contracts",
    ]
    try:
        subprocess.run(cmd, check=True)
        print("✅ Full contract validation success")
    except subprocess.CalledProcessError:
        print("❌ Full contract validation failed")
        sys.exit(1)

def print_summary():
    print(f"\n✅ All Done! Artifacts are in: {OUTPUT_DIR.absolute()}")
    print("Files:")
    if OUTPUT_DIR.exists():
        output_files = list(ONNX_DIR.glob("*.onnx"))
        tokenizer_json = OUTPUT_DIR / TOKENIZER_JSON_FILENAME
        tokenizer_config = OUTPUT_DIR / TOKENIZER_CONFIG_FILENAME
        if tokenizer_json.exists():
            output_files.append(tokenizer_json)
        if tokenizer_config.exists():
            output_files.append(tokenizer_config)
        # include embeddings
        for sub in ("embeddings", "embeddings_v2"):
            subdir = OUTPUT_DIR / sub
            if subdir.exists():
                for f in sorted(subdir.iterdir()):
                    if f.is_file():
                        output_files.append(f)
        # include reference sample if present
        for wav in OUTPUT_DIR.glob("*.wav"):
            output_files.append(wav)
        for f in sorted(output_files):
            s = f.stat().st_size / (1024*1024)
            print(f" - {f.name:<30} ({s:.1f} MB)")
    else:
        print("⚠️ Output directory missing.")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Export Script for PocketTTS")
    parser.add_argument("--quantize", action="store_true", help="Run INT8 quantization after export")
    parser.add_argument(
        "--quantize-precision",
        choices=["int8", "q4", "all"],
        default="int8",
        help="Quantization precision profile when --quantize is enabled",
    )
    parser.add_argument(
        "--q4-block-size",
        type=int,
        default=128,
        help="Block size to use for q4 weight-only quantization",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run full tokenizer/ONNX contract validation after export",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Do not download voice‑embedding safetensors from upstream HF repo",
    )
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
