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

    def list_repo_files(self, repo_id, revision=None):
        return list(self._files)


@pytest.mark.parametrize("language", ["english", "italian"])
def test_download_voice_embeddings_per_language(tmp_path, monkeypatch, capsys, language):
    fake_files = [
        f"languages/{language}/embeddings/foo.safetensors",
        f"languages/{language}/embeddings/bar.safetensors",
        "languages/french/embeddings/other.safetensors",
        "tokenizer.model",
    ]

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
    exporter.download_voice_embeddings(language, out_dir, token=None)

    listing_output = capsys.readouterr().out
    assert f"repository {exporter.VOICE_EMBEDDINGS_REPO} contains" in listing_output

    stem = out_dir / language / "embeddings_v3" / "foo"
    assert stem.with_suffix(".safetensors").exists()
    assert stem.with_suffix(".bin").exists()
    assert stem.with_suffix(".json").exists()
    assert not (out_dir / "embeddings_v3" / "foo.safetensors").exists()
    assert not (out_dir / language / "embeddings_v3" / "other.safetensors").exists()

def test_download_voice_embeddings_uses_pinned_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda token=None: DummyApi(["languages/english/embeddings/foo.safetensors"]),
    )

    revisions: list[str | None] = []

    def fake_download(*args, **kwargs):
        revisions.append(kwargs.get("revision"))
        filename = kwargs["filename"]
        cache_path = tmp_path / "cache" / filename
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np
        import safetensors.numpy

        arr = np.zeros((1, 2, EMBEDDING_DIM), dtype=np.float32)
        safetensors.numpy.save_file({"audio_prompt": arr}, str(cache_path))
        return str(cache_path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    exporter.download_voice_embeddings("english", tmp_path / "hf", token=None)

    assert revisions == [exporter.VOICE_EMBEDDINGS_REVISION]


def test_download_voice_embeddings_missing_language(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: DummyApi(["tokenizer.model"]))

    exporter.download_voice_embeddings("klingon", tmp_path / "hf", token=None)

    assert "No voice embeddings found for language 'klingon'" in capsys.readouterr().out


def test_clone_audio_prompts_to_embeddings(tmp_path, monkeypatch):
    # Create a dummy audio prompt file
    audio_dir = tmp_path / "audio_prompts"
    audio_dir.mkdir(parents=True)
    prompt_path = audio_dir / "announcer.wav"
    prompt_path.write_bytes(b"RIFF\x00\x00\x00\x00")

    # Prepare temp output destination
    out_dir = tmp_path / "hf"

    # Insert a dummy pocket_tts module to avoid loading the real model
    import types

    dummy_module = types.ModuleType("pocket_tts")

    class DummyModel:
        def get_state_for_audio_prompt(self, _):
            import numpy as np

            return {"transformer.layers.0.self_attn/cache": np.zeros((1,), dtype=np.float32)}

    class DummyTTSModel:
        @staticmethod
        def load_model(*args, **kwargs):
            return DummyModel()

    def dummy_export_model_state(state, dest):
        import safetensors.numpy

        safetensors.numpy.save_file(state, dest)

    dummy_module.TTSModel = DummyTTSModel
    dummy_module.export_model_state = dummy_export_model_state

    monkeypatch.setitem(sys.modules, "pocket_tts", dummy_module)

    # Run cloning for a non-default language
    exporter.clone_audio_prompts_to_embeddings(out_dir, language="italian")

    # Expect safetensors + raw artifacts in the language-scoped embeddings_v3
    safetensors_path = out_dir / "italian" / "embeddings_v3" / "announcer.safetensors"
    assert safetensors_path.exists()
    assert safetensors_path.with_suffix(".bin").exists()
    assert safetensors_path.with_suffix(".json").exists()

    # Legacy root-level clone (English model) is kept for backward compatibility
    legacy_path = out_dir / "embeddings_v3" / "announcer.safetensors"
    assert legacy_path.exists()
