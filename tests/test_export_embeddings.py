import sys
from pathlib import Path

import huggingface_hub
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import export as exporter


EMBEDDING_DIM = 1024


class DummyApi:
    def __init__(self, files):
        self._files = files

    def list_repo_files(self, repo_id):
        return list(self._files)


@pytest.mark.parametrize("prefix", ["embeddings/", "embeddings_v2/"])
def test_download_voice_embeddings(tmp_path, monkeypatch, capsys, prefix):
    repo_id = "kyutai/pocket-tts"
    fake_files = [f"{prefix}foo.safetensors", "other/file.txt"]

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: DummyApi(fake_files))

    def fake_download(*args, **kwargs):
        filename = kwargs.get("filename") or (args[1] if len(args) > 1 else None)
        cache_path = tmp_path / "cache" / filename
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.suffix == ".safetensors":
            import numpy as np
            import safetensors.numpy

            arr = np.zeros((1, 2, EMBEDDING_DIM), dtype=np.float32)
            safetensors.numpy.save_file({"audio_prompt": arr}, str(cache_path))
        else:
            cache_path.write_text("")
        return str(cache_path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    out_dir = tmp_path / "hf"
    exporter.download_voice_embeddings(repo_id, out_dir, token=None)

    listing_output = capsys.readouterr().out
    assert f"repository {repo_id} contains" in listing_output

    stem = out_dir / prefix / "foo"
    assert stem.with_suffix(".safetensors").exists()
    assert stem.with_suffix(".bin").exists()
    assert stem.with_suffix(".json").exists()