import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.quantize as quantize


def test_models_to_quantize_has_core_five():
    assert set(quantize.MODELS_TO_QUANTIZE) == {
        "flow_lm_main",
        "flow_lm_flow",
        "mimi_decoder",
        "mimi_encoder",
        "text_conditioner",
    }


def test_quantize_file_skips_missing_input(tmp_path, monkeypatch):
    def fake_quantize_dynamic(*args, **kwargs):
        raise AssertionError("quantize_dynamic must not be called for missing input")

    monkeypatch.setattr(quantize, "quantize_dynamic", fake_quantize_dynamic)

    quantize.quantize_file(tmp_path / "missing.onnx", tmp_path / "out.onnx")


def test_quantize_file_writes_int8_output(tmp_path, monkeypatch):
    input_path = tmp_path / "model.onnx"
    input_path.write_bytes(b"dummy")

    stub_model = object()
    monkeypatch.setattr(quantize.onnx, "load", lambda path: stub_model)
    monkeypatch.setattr(quantize.onnx.shape_inference, "infer_shapes", lambda model: model)
    monkeypatch.setattr(quantize.onnx, "save", lambda model, path: None)

    calls: list[tuple] = []

    def fake_quantize_dynamic(model_input, model_output, **kwargs):
        calls.append((model_input, model_output, kwargs))

    monkeypatch.setattr(quantize, "quantize_dynamic", fake_quantize_dynamic)

    quantize.quantize_file(input_path, tmp_path / "model_int8.onnx")

    assert len(calls) == 1
    _, model_output, kwargs = calls[0]
    assert Path(model_output).name == "model_int8.onnx"
    assert kwargs["weight_type"] == quantize.QuantType.QInt8


def test_main_quantizes_each_model_to_int8(monkeypatch, tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir(parents=True)
    for name in quantize.MODELS_TO_QUANTIZE:
        (input_dir / f"{name}.onnx").write_bytes(b"dummy")

    calls: list[tuple] = []

    def fake_quantize_file(input_path: Path, output_path: Path, op_types=("MatMul",)):
        calls.append((input_path.name, output_path.name, op_types))

    monkeypatch.setattr(quantize, "quantize_file", fake_quantize_file)
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

    assert calls == [
        (f"{name}.onnx", f"{name}_int8.onnx", ["MatMul"]) for name in quantize.MODELS_TO_QUANTIZE
    ]


def test_main_skips_missing_models(monkeypatch, tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir(parents=True)
    (input_dir / "flow_lm_main.onnx").write_bytes(b"dummy")

    calls: list[tuple] = []

    def fake_quantize_file(input_path: Path, output_path: Path, op_types=("MatMul",)):
        if input_path.exists():
            calls.append(output_path.name)

    monkeypatch.setattr(quantize, "quantize_file", fake_quantize_file)
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

    assert calls == ["flow_lm_main_int8.onnx"]
