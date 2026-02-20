import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


DEFAULT_REPO_ID = "ottomate/pocket-tts-ONNX"
DEFAULT_FOLDER = Path("hf")
DEFAULT_README = DEFAULT_FOLDER / "README.md"


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

- tokenizer.json (for @huggingface/tokenizers compatibility)
- onnx/flow_lm_main.onnx
- onnx/flow_lm_flow.onnx
- onnx/mimi_encoder.onnx
- onnx/mimi_decoder.onnx
- onnx/text_conditioner.onnx
- optional *_int8.onnx quantized variants

## Attribution

This repository is derived from Kyutai's Pocket TTS model:
https://huggingface.co/kyutai/pocket-tts

## License

This model card uses the same license field as the upstream model card:
cc-by-4.0

Also follow upstream usage conditions and prohibited-use policy documented at:
https://huggingface.co/kyutai/pocket-tts
"""


def ensure_readme(repo_id: str, readme_path: Path):
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(build_readme(repo_id), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Upload local hf artifacts to Hugging Face Hub.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Target HF repo id (owner/name)")
    parser.add_argument("--folder", default=str(DEFAULT_FOLDER), help="Local folder to upload")
    parser.add_argument("--private", action="store_true", help="Create repo as private")
    parser.add_argument(
        "--commit-message",
        default="Upload hf export artifacts",
        help="Commit message for upload",
    )
    parser.add_argument(
        "--skip-readme",
        action="store_true",
        help="Do not (re)write hf/README.md before uploading",
    )
    args = parser.parse_args()

    folder_path = Path(args.folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    token = load_hf_token()
    api = HfApi(token=token)

    if not args.skip_readme:
        ensure_readme(args.repo_id, folder_path / "README.md")
        print(f"Wrote {folder_path / 'README.md'}")

    try:
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    except HfHubHTTPError as e:
        if "403" in str(e):
            print(
                "Could not create repo in this namespace (403). "
                "Will try uploading in case the repo already exists and token has write access."
            )
        else:
            raise

    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(folder_path),
        path_in_repo=".",
        commit_message=args.commit_message,
    )
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
