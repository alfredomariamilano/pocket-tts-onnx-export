import os
import io
import sys
import json
import pytest
import shutil
from pathlib import Path

# make sure project root is on python path so we can import export.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import export as exporter
import huggingface_hub

# dimensions used by the embedded tensors (must match runtime constant)
EMBEDDING_DIM = 1024

class DummyApi:
    def __init__(self, files):
        self._files = files
    def list_repo_files(self, repo_id):
        return list(self._files)

# (no global fake download; tests provide their own implementations)


@ pytest.mark.parametrize("prefix", ["embeddings/", "embeddings_v2/"])
def test_download_voice_embeddings(tmp_path, monkeypatch, capsys, prefix):
    # setup fake repo listing with one safetensors file and one unrelated
    repo_id = "kyutai/pocket-tts"
    fake_files = [f"{prefix}foo.safetensors", "other/file.txt"]

    # patch HfApi to return our fake listing
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: DummyApi(fake_files))

    # fake_download will create a *real* safetensors file when asked
    def fake_download_inner(*args, **kwargs):
        filename = kwargs.get("filename") or (args[1] if len(args) > 1 else None)
        cache_path = tmp_path / "cache" / filename
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.suffix == ".safetensors":
            # create a tiny tensor with the expected key
            import numpy as np, safetensors.numpy
            arr = np.zeros((1, 2, EMBEDDING_DIM), dtype=np.float32)
            safetensors.numpy.save_file({"audio_prompt": arr}, str(cache_path))
        else:
            cache_path.write_text("")
        return str(cache_path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download_inner)

    out_dir = tmp_path / "hf"
    exporter.download_voice_embeddings(repo_id, out_dir, token=None)

    # debug listing should be printed
    listing_output = capsys.readouterr().out
    assert f"repository {repo_id} contains" in listing_output

    # assert only the file under prefix was written and converted
    safepath = out_dir / prefix / "foo.safetensors"
    assert safepath.exists(), "expected embedding file to be downloaded"
    binpath = out_dir / prefix / "foo.bin"
    assert binpath.exists(), "expected raw .bin conversion file"

    # metadata beside the converted file
    meta = json.loads((out_dir / prefix / "foo.json").read_text())
    assert meta["shape"] == [1, 2, EMBEDDING_DIM]
    assert meta["dtype"] == "float32"

    # unrelated file should not be downloaded
    assert not (out_dir / "other" / "file.txt").exists()
    # ensure no .cache directory was created under output
    assert not any(str(p).endswith('.cache') for p in out_dir.rglob('*')), "no .cache should be under hf"


def test_download_reference_sample(tmp_path, monkeypatch):
    repo_id = "foo/bar"
    called = []

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    def fake_download_inner(*args, **kwargs):
        filename = kwargs.get("filename") or (args[1] if len(args) > 1 else None)
        called.append(filename)
        target = cache_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("")
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download_inner)

    exporter.download_reference_sample(repo_id, tmp_path / "hf", token=None)
    assert "reference_sample.wav" in called
    assert (tmp_path / "hf" / "reference_sample.wav").exists()


def test_download_hf_file_missing(tmp_path, capsys, monkeypatch):
    # simulate hf_hub_download raising a not-found error
    def always_fail(*args, **kwargs):
        raise Exception("An error happened while trying to locate the file on the Hub")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", always_fail)
    result = exporter._download_hf_file("foo/bar", "doesnotexist.safetensors", tmp_path / "hf", token=None)
    assert result is None
    out = capsys.readouterr().out
    assert "Could not fetch doesnotexist.safetensors" in out
    # message should include the original error but not raise an exception


def test_download_hf_file_permission(tmp_path, capsys, monkeypatch):
    # simulate hf_hub_download raising a 403 error
    def forbidden(*args, **kwargs):
        raise Exception("403 Forbidden: you are not allowed")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)
    res = exporter._download_hf_file("foo/bar", "somefile.bin", tmp_path / "hf", token=None)
    assert res is None
    out = capsys.readouterr().out
    assert "Could not fetch somefile.bin" in out
    assert "HF_TOKEN" in out


def test_convert_safetensor_missing_key(tmp_path, capsys):
    # create a safetensors file that matches neither supported schema
    import numpy as np, safetensors.numpy

    arr = np.zeros((1, 3, EMBEDDING_DIM), dtype=np.float32)
    path = tmp_path / "bad.safetensors"
    safetensors.numpy.save_file({"not_audio": arr}, str(path))

    # run conversion helper directly; should return False
    result = exporter._convert_safetensor_to_raw(path)
    assert result is False

    out = capsys.readouterr().out
    assert "did not match supported schemas" in out

    # no artefacts should have been created
    assert not path.with_suffix('.bin').exists()
    assert not path.with_suffix('.json').exists()


def test_download_voice_embeddings_summary(tmp_path, monkeypatch, capsys):
    # simulate two safetensors, one good one unsupported
    repo_id = "kyutai/pocket-tts"
    fake_files = ["embeddings/foo.safetensors", "embeddings/bar.safetensors"]
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: DummyApi(fake_files))

    def fake_download_inner(*args, **kwargs):
        filename = kwargs.get("filename") or (args[1] if len(args) > 1 else None)
        cache_path = tmp_path / "cache" / filename
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.name == "foo.safetensors":
            # valid tensor
            import numpy as np, safetensors.numpy
            arr = np.zeros((1, 2, EMBEDDING_DIM), dtype=np.float32)
            safetensors.numpy.save_file({"audio_prompt": arr}, str(cache_path))
        else:
            # unsupported schema (neither v1 nor v2)
            import numpy as np, safetensors.numpy
            arr = np.zeros((1, 2, EMBEDDING_DIM), dtype=np.float32)
            safetensors.numpy.save_file({"oops": arr}, str(cache_path))
        return str(cache_path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download_inner)

    out_dir = tmp_path / "hf"
    exporter.download_voice_embeddings(repo_id, out_dir, token=None)
    output = capsys.readouterr().out
    assert "Converted foo.safetensors" in output
    assert "did not match supported schemas" in output and "bar.safetensors" in output


def test_convert_safetensor_v2_model_state(tmp_path):
    import numpy as np, safetensors.numpy

    path = tmp_path / "voice_v2.safetensors"
    # Deliberately unsorted keys; converter should produce deterministic state_ order.
    safetensors.numpy.save_file(
        {
            "flow_lm/z_cache": np.zeros((1, 4), dtype=np.float32),
            "flow_lm/a_steps": np.array([7], dtype=np.int64),
            "decoder/enabled": np.array([True], dtype=np.bool_),
        },
        str(path),
    )

    ok = exporter._convert_safetensor_to_raw(path)
    assert ok is True

    bin_path = path.with_suffix(".bin")
    meta_path = path.with_suffix(".json")
    assert bin_path.exists()
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text())
    assert meta["format"] == "pocket_tts_flow_state_v2"
    assert meta["tensor_count"] == 3
    assert len(meta["tensors"]) == 3

    tensor_names = [entry["name"] for entry in meta["tensors"]]
    assert tensor_names == ["state_0", "state_1", "state_2"]

    source_keys = [entry["source_key"] for entry in meta["tensors"]]
    # sorted by module, then tensor key
    assert source_keys == [
        "decoder/enabled",
        "flow_lm/a_steps",
        "flow_lm/z_cache",
    ]

    dtypes = [entry["dtype"] for entry in meta["tensors"]]
    assert dtypes == ["bool", "int64", "float32"]

    total = 0
    for entry in meta["tensors"]:
        assert entry["offset"] == total
        total += entry["nbytes"]
    assert meta["total_bytes"] == total
    assert bin_path.stat().st_size == total


def test_convert_safetensor_v2_current_end_to_int64_scalar(tmp_path):
    import numpy as np, safetensors.numpy

    path = tmp_path / "voice_current_end.safetensors"
    safetensors.numpy.save_file(
        {
            "transformer.layers.0.self_attn/cache": np.zeros((2, 1, 10, 16, 64), dtype=np.float32),
            "transformer.layers.0.self_attn/current_end": np.zeros((10,), dtype=np.float32),
        },
        str(path),
    )

    ok = exporter._convert_safetensor_to_raw(path)
    assert ok is True

    meta = json.loads(path.with_suffix(".json").read_text())
    entries = {entry["source_key"]: entry for entry in meta["tensors"]}
    current_end = entries["transformer.layers.0.self_attn/current_end"]

    assert current_end["dtype"] == "int64"
    assert current_end["shape"] == [1]


def test_run_export_scripts_skip(tmp_path, capsys, monkeypatch):
    # monkeypatch exporter's internal functions to avoid heavy work
    monkeypatch.setattr(exporter, "_load_hf_token", lambda: None)
    monkeypatch.setattr(exporter, "download_voice_embeddings", lambda *args, **kwargs: None)
    monkeypatch.setattr(exporter, "ONNX_DIR", tmp_path / "hf" / "onnx")
    monkeypatch.setattr(exporter, "WEIGHTS_DIR", tmp_path / "weights")

    # intercept subprocess.run to pretend success and capture calls
    class DummyResult:
        def __init__(self):
            self.returncode = 0
    def fake_run(cmd, check=False, **kwargs):
        return DummyResult()
    monkeypatch.setattr(exporter.subprocess, "run", fake_run)

    # ensure dirs exist so export scripts continue
    (tmp_path / "hf" / "onnx").mkdir(parents=True)
    (tmp_path / "weights").mkdir(parents=True)

    exporter.run_export_scripts(skip_embeddings=True)
    captured = capsys.readouterr().out
    assert "Skipping voice embedding download" in captured


def test_print_summary_includes_embeddings(tmp_path, capsys, monkeypatch):
    # create fake output dir with some onnx and embeddings
    fake_hf = tmp_path / "hf"
    fake_hf.mkdir()
    fake_onnx = fake_hf / "onnx"
    fake_onnx.mkdir()
    (fake_onnx / "foo.onnx").write_text("")
    emb = fake_hf / "embeddings"
    emb.mkdir()
    (emb / "voice.safetensors").write_text("")
    # simulate converted files too
    (emb / "voice.bin").write_text("")
    (emb / "voice.json").write_text("{}")
    emb2 = fake_hf / "embeddings_v2"
    emb2.mkdir()
    (emb2 / "other.safetensors").write_text("")
    (emb2 / "other.bin").write_text("")
    (emb2 / "other.json").write_text("{}")
    # add reference sample wav
    (fake_hf / "reference_sample.wav").write_text("")

    # patch constants
    monkeypatch.setattr(exporter, "OUTPUT_DIR", fake_hf)
    monkeypatch.setattr(exporter, "ONNX_DIR", fake_onnx)
    # call summary
    exporter.print_summary()
    captured = capsys.readouterr().out
    assert "foo.onnx" in captured
    assert "voice.safetensors" in captured
    assert "voice.bin" in captured
    assert "voice.json" in captured
    assert "other.safetensors" in captured
    assert "other.bin" in captured
    assert "other.json" in captured
    assert "reference_sample.wav" in captured
