import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.quantize as quantize


def test_resolve_precisions_variants():
    assert quantize._resolve_precisions("int8") == ["int8"]
    assert quantize._resolve_precisions("q4") == ["q4"]
    assert quantize._resolve_precisions("all") == ["int8", "q4"]


def test_output_filename_suffixes():
    assert quantize._output_filename("flow_lm_main", "int8") == "flow_lm_main_int8.onnx"
    assert quantize._output_filename("flow_lm_main", "q4") == "flow_lm_main_q4.onnx"


def test_main_dispatches_all_precisions(monkeypatch, tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir(parents=True)
    (input_dir / "flow_lm_main.onnx").write_bytes(b"dummy")

    monkeypatch.setattr(quantize, "MODELS_TO_QUANTIZE", ["flow_lm_main"])

    calls: list[tuple] = []

    def fake_int8(input_path: Path, output_path: Path):
        calls.append(("int8", input_path.name, output_path.name))

    def fake_q4(input_path: Path, output_path: Path, block_size: int = 128):
        calls.append(("q4", input_path.name, output_path.name, block_size))

    monkeypatch.setattr(quantize, "quantize_file_int8", fake_int8)
    monkeypatch.setattr(quantize, "quantize_file_q4_weight_only", fake_q4)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quantize.py",
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--precision",
            "all",
            "--q4-block-size",
            "64",
        ],
    )

    quantize.main()

    assert calls == [
        ("int8", "flow_lm_main.onnx", "flow_lm_main_int8.onnx"),
        ("q4", "flow_lm_main.onnx", "flow_lm_main_q4.onnx", 64),
    ]


def test_main_default_precision_int8_only(monkeypatch, tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir(parents=True)
    (input_dir / "flow_lm_flow.onnx").write_bytes(b"dummy")

    monkeypatch.setattr(quantize, "MODELS_TO_QUANTIZE", ["flow_lm_flow"])

    calls: list[tuple] = []

    def fake_int8(input_path: Path, output_path: Path):
        calls.append(("int8", input_path.name, output_path.name))

    monkeypatch.setattr(quantize, "quantize_file_int8", fake_int8)
    monkeypatch.setattr(
        quantize,
        "quantize_file_q4_weight_only",
        lambda *args, **kwargs: calls.append(("q4",)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quantize.py",
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
        ],
    )

    quantize.main()

    assert calls == [("int8", "flow_lm_flow.onnx", "flow_lm_flow_int8.onnx")]