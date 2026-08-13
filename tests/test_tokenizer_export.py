import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import export as exporter


def test_export_language_tokenizer_jsons(tmp_path, monkeypatch):
    hf_dir = tmp_path / "hf"
    bundle_dir = hf_dir / "english"
    (bundle_dir / "onnx").mkdir(parents=True)
    (bundle_dir / "onnx" / "tokenizer.model").write_bytes(b"spm-model")
    (hf_dir / "embeddings_v3").mkdir()
    (hf_dir / "embeddings_v3" / "x.safetensors").write_bytes(b"x")
    (hf_dir / "tokenizer.json").write_text("{}")

    calls: list[tuple] = []

    def fake_export_tokenizer_json(tokenizer_model_path: Path, output_dir: Path) -> None:
        calls.append((tokenizer_model_path, output_dir))

    monkeypatch.setattr(exporter, "export_tokenizer_json", fake_export_tokenizer_json)
    monkeypatch.setattr(exporter, "OUTPUT_DIR", hf_dir)

    exporter.export_language_tokenizer_jsons()

    assert calls == [(bundle_dir / "onnx" / "tokenizer.model", bundle_dir)]


def test_export_language_tokenizer_jsons_skips_bundles_without_model(tmp_path, monkeypatch):
    hf_dir = tmp_path / "hf"
    (hf_dir / "french" / "onnx").mkdir(parents=True)

    calls: list[tuple] = []

    def fake_export_tokenizer_json(tokenizer_model_path: Path, output_dir: Path) -> None:
        calls.append((tokenizer_model_path, output_dir))

    monkeypatch.setattr(exporter, "export_tokenizer_json", fake_export_tokenizer_json)
    monkeypatch.setattr(exporter, "OUTPUT_DIR", hf_dir)

    exporter.export_language_tokenizer_jsons()

    assert calls == []


def test_run_full_validation_uses_language_tokenizer(tmp_path, monkeypatch):
    onnx_dir = tmp_path / "hf" / "italian" / "onnx"
    onnx_dir.mkdir(parents=True)
    (onnx_dir.parent / "tokenizer.json").write_text("{}")

    captured: list[list[str]] = []

    def fake_run(cmd, check=True):
        captured.append(cmd)

    monkeypatch.setattr(exporter.subprocess, "run", fake_run)

    exporter.run_full_validation(onnx_dir)

    assert "scripts.validate_onnx_contracts" in captured[0]
    assert "--tokenizer-json" in captured[0]
    tokenizer_arg = captured[0][captured[0].index("--tokenizer-json") + 1]
    assert tokenizer_arg == str(onnx_dir.parent / "tokenizer.json")


def test_run_full_validation_falls_back_to_default_tokenizer(tmp_path, monkeypatch):
    onnx_dir = tmp_path / "hf" / "italian" / "onnx"
    onnx_dir.mkdir(parents=True)

    captured: list[list[str]] = []

    def fake_run(cmd, check=True):
        captured.append(cmd)

    monkeypatch.setattr(exporter.subprocess, "run", fake_run)

    exporter.run_full_validation(onnx_dir)

    assert "--tokenizer-json" not in captured[0]


def test_export_tokenizer_json_generates_artifacts(tmp_path):
    import sentencepiece as spm

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("pocket tts onnx export tokenizer validation test\n", encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(tmp_path / "sp"),
        vocab_size=24,
        model_type="unigram",
    )

    out_dir = tmp_path / "hf"
    exporter.export_tokenizer_json(tmp_path / "sp.model", out_dir)

    assert (out_dir / "tokenizer.json").exists()
    assert (out_dir / "tokenizer_config.json").exists()
