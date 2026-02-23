import os
import json
import shutil
import subprocess
import sys
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
    """Convert a `.safetensors` file containing an ``audio_prompt`` tensor
    into a raw float32 binary blob and accompanying metadata.

    Returns ``True`` if the conversion succeeded, ``False`` if the file lacked
    the expected key or if an error occurred.  The caller may use the return
    value to accumulate a warning summary instead of printing repeated lines.
    """
    try:
        # delay heavy imports until needed
        import numpy as np
        import safetensors.torch

        data = safetensors.torch.load_file(str(src_path))
        tensor = data.get("audio_prompt")
        if tensor is None:
            print(f"⚠️ {src_path.name} missing 'audio_prompt' tensor, skipping conversion")
            return False

        arr = tensor.cpu().numpy() if hasattr(tensor, "cpu") else np.array(tensor)
        arr = arr.astype("float32")

        raw_path = src_path.with_suffix(".bin")
        arr.tofile(str(raw_path))

        meta = {"shape": list(arr.shape), "dtype": "float32"}
        meta_path = src_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        print(f"✅ Converted {src_path.name} -> {raw_path.name} (shape {arr.shape})")
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
                "⚠️ The following safetensors lacked an audio_prompt or failed to convert:",
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

def run_quantization():
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
        "--output_dir", str(QUANT_DIR)
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
        run_quantization()

    if args.validate:
        run_full_validation()
        
    print_summary()
