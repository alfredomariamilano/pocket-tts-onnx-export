import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.validate_onnx_contracts as validate


class _FakeValueInfo:
    def __init__(self, name: str, shape: list[object]):
        self.name = name
        self.shape = shape


class _FakeSession:
    def __init__(self, inputs: list[_FakeValueInfo]):
        self._inputs = inputs

    def get_inputs(self) -> list[_FakeValueInfo]:
        return self._inputs


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"dummy")
    return path


def test_find_builtin_voice_nested_language_layout(tmp_path):
    onnx_dir = tmp_path / "hf" / "english" / "onnx"
    tokenizer_json = tmp_path / "hf" / "tokenizer.json"
    voice = _touch(tmp_path / "hf" / "embeddings_v3" / "marius.safetensors")

    assert validate._find_builtin_voice(onnx_dir, tokenizer_json, "marius") == voice


def test_find_builtin_voice_flat_layout(tmp_path):
    onnx_dir = tmp_path / "hf" / "onnx"
    tokenizer_json = tmp_path / "hf" / "tokenizer.json"
    voice = _touch(tmp_path / "hf" / "embeddings_v2" / "marius.safetensors")

    assert validate._find_builtin_voice(onnx_dir, tokenizer_json, "marius") == voice


def test_find_builtin_voice_prefers_embeddings_v3(tmp_path):
    onnx_dir = tmp_path / "hf" / "onnx"
    tokenizer_json = tmp_path / "hf" / "tokenizer.json"
    voice_v3 = _touch(tmp_path / "hf" / "embeddings_v3" / "marius.safetensors")
    _touch(tmp_path / "hf" / "embeddings_v2" / "marius.safetensors")

    assert validate._find_builtin_voice(onnx_dir, tokenizer_json, "marius") == voice_v3


def test_find_builtin_voice_missing_raises(tmp_path):
    onnx_dir = tmp_path / "hf" / "english" / "onnx"
    tokenizer_json = tmp_path / "hf" / "tokenizer.json"

    with pytest.raises(FileNotFoundError):
        validate._find_builtin_voice(onnx_dir, tokenizer_json, "marius")


def test_bundle_language_nested_layout():
    assert validate._bundle_language(Path("hf/italian/onnx")) == "italian"
    assert validate._bundle_language(Path("hf/english_2026-04/onnx")) == "english_2026-04"


def test_bundle_language_flat_layout_falls_back_to_default():
    assert validate._bundle_language(Path("hf/onnx")) == validate.DEFAULT_LANGUAGE
    assert validate._bundle_language(Path("hf")) == validate.DEFAULT_LANGUAGE


def test_build_state_feeds_pads_cache_to_contract_length():
    session = _FakeSession(
        [
            _FakeValueInfo("sequence", [1, "seq_len", 32]),
            _FakeValueInfo("state_0", [2, 1, 1000, 16, 64]),
            _FakeValueInfo("state_1", [1]),
        ]
    )
    state = {
        "transformer.layers.0.self_attn": {
            "cache": torch.zeros(2, 1, 295, 16, 64),
            "offset": torch.zeros(1, dtype=torch.long),
        }
    }

    feeds = validate._build_state_feeds(state, ["state_0", "state_1"], session)

    assert feeds["state_0"].shape == (2, 1, 1000, 16, 64)
    assert np.isnan(feeds["state_0"][:, :, 295:]).all()
    assert feeds["state_0"][:, :, :295].shape == (2, 1, 295, 16, 64)
    assert feeds["state_1"].shape == (1,)


def test_build_state_feeds_keeps_cache_at_contract_length():
    session = _FakeSession([_FakeValueInfo("state_0", [2, 1, 1000, 16, 64])])
    state = {
        "transformer.layers.0.self_attn": {
            "cache": torch.zeros(2, 1, 1000, 16, 64),
            "offset": torch.zeros(1, dtype=torch.long),
        }
    }

    feeds = validate._build_state_feeds(state, ["state_0"], session)

    assert feeds["state_0"].shape == (2, 1, 1000, 16, 64)
