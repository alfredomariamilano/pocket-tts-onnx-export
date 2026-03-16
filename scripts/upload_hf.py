import argparse
import os
import subprocess
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


DEFAULT_REPO_ID = "ottomate/pocket-tts-ONNX"
DEFAULT_FOLDER = Path("hf")


def load_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip().strip('"').strip("'")

    env_path = Path(".env")
    if env_path.exists():
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

    raise RuntimeError("HF_TOKEN not found in environment or .env")


def build_readme(repo_id: str) -> str:
    return f"""---
license: cc-by-4.0
library_name: onnxruntime
base_model: kyutai/pocket-tts
tags:
  - text-to-speech
  - onnx
  - onnxruntime
  - pocket-tts
---

# {repo_id}

ONNX export artifacts for Pocket TTS.

## Contents

- tokenizer.json
- tokenizer_config.json
- onnx/flow_lm_main.onnx
- onnx/flow_lm_flow.onnx
- onnx/mimi_encoder.onnx
- onnx/mimi_decoder.onnx
- onnx/text_conditioner.onnx
- optional *_int8.onnx quantized variants
- optional *_q4.onnx quantized variants
- embeddings/, embeddings_v2/, and embeddings_v3/ voice-cloning assets
- reference_sample.wav
"""


def _read_local_hf_readme() -> str | None:
    template_path = Path("HF_README.md")
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return None


def ensure_readme(repo_id: str, readme_path: Path) -> None:
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    content = _read_local_hf_readme() or build_readme(repo_id)
    readme_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload local hf artifacts to Hugging Face Hub.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Target HF repo id (owner/name)")
    parser.add_argument("--folder", default=str(DEFAULT_FOLDER), help="Local folder to upload")
    parser.add_argument("--private", action="store_true", help="Create repo as private")
    parser.add_argument("--commit-message", default=None, help="Commit message for upload")
    parser.add_argument("--skip-readme", action="store_true", help="Do not rewrite hf/README.md before uploading")
    args = parser.parse_args()

    folder_path = Path(args.folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    commit_message = args.commit_message
    if not commit_message:
        try:
            commit_message = subprocess.check_output(["git", "log", "-1", "--pretty=%B"], encoding="utf-8").strip()
        except Exception:
            commit_message = "Upload hf export artifacts"

    token = load_hf_token()
    api = HfApi(token=token)

    if not args.skip_readme:
        ensure_readme(args.repo_id, folder_path / "README.md")

    try:
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    except HfHubHTTPError as exc:
        if "403" not in str(exc):
            raise

    def should_include(path: Path) -> bool:
        rel = path.relative_to(folder_path)
        if "onnx_quant" in rel.parts:
            return False
        if rel.match("onnx/validation_report.*"):
            return False
        if rel.parts and rel.parts[0] in ("embeddings", "embeddings_v2", "embeddings_v3") and rel.suffix == ".safetensors":
            return False
        return True

    files_to_upload = [path for path in folder_path.rglob("*") if path.is_file() and should_include(path)]

    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(folder_path),
        path_in_repo=".",
        commit_message=commit_message,
        allow_patterns=[str(path.relative_to(folder_path)) for path in files_to_upload],
        delete_patterns=["*"],
    )
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()