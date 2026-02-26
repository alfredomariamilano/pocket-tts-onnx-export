import sys
from pathlib import Path

import scripts.quantize as quantize


def test_resolve_precisions_variants():
    assert quantize._resolve_precisions("int8") == ["int8"]
    assert quantize._resolve_precisions("q4") == ["q4"]
    assert quantize._resolve_precisions("all") == ["int8", "q4"]


def test_output_filename_suffixes():
    assert quantize._output_filename("flow_lm_main", "int8") == "flow_lm_main_int8.onnx"
    assert quantize._output_filename("flow_lm_main", "q4") == "flow_lm_main_q4.onnx"
    # fused generator name
    assert quantize._output_filename("flow_lm_full", "int8") == "flow_lm_full_int8.onnx"
    assert quantize._output_filename("flow_lm_full", "q4") == "flow_lm_full_q4.onnx"


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


def test_default_op_types_include_gemm():
    # ensure the default op_types tuple covers both MatMul and Gemm
    assert "MatMul" in quantize.quantize_file_int8.__defaults__[0]
    assert "Gemm" in quantize.quantize_file_int8.__defaults__[0]


def test_strip_if_no_reduction(tmp_path):
    """helper should delete quant outputs that are not smaller"""
    inp = tmp_path / "a.onnx"
    out = tmp_path / "b.onnx"
    inp.write_bytes(b"orig")
    # create a file artificially larger
    out.write_bytes(b"bigger than original")

    quantize._maybe_strip_if_no_reduction(inp, out)
    assert not out.exists()


def test_strip_if_reduction_kept(tmp_path):
    """quant output that is smaller should be preserved"""
    inp = tmp_path / "a.onnx"
    out = tmp_path / "b.onnx"
    inp.write_bytes(b"orig")
    out.write_bytes(b"ok")

    quantize._maybe_strip_if_no_reduction(inp, out)
    assert out.exists()
