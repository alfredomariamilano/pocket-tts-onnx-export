from onnx_export.external_data import _normalize_optimize_stages


def test_normalize_optimize_stages_defaults_to_ort_from_legacy_flag():
    assert _normalize_optimize_stages(optimize_stages=None, ort_optimize=True) == ["ort"]
    assert _normalize_optimize_stages(optimize_stages=None, ort_optimize=False) == []


def test_normalize_optimize_stages_parses_csv_in_order():
    assert _normalize_optimize_stages(optimize_stages="onnxsim,ort", ort_optimize=False) == ["onnxsim", "ort"]


def test_normalize_optimize_stages_rejects_unknown_stage():
    try:
        _normalize_optimize_stages(optimize_stages="foo", ort_optimize=True)
    except ValueError as exc:
        assert "Unsupported optimization stage" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown optimization stage")
