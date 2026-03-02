from pathlib import Path

import onnx


def sidecar_path_for_model(model_path: Path, suffix: str = ".onnx_data") -> Path:
    cleaned_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return model_path.with_name(f"{model_path.stem}{cleaned_suffix}")


def _optimize_with_ort(model_path: Path) -> Path | None:
    optimized_path = model_path.with_name(f"{model_path.stem}.ort_optimized.onnx")
    try:
        import onnxruntime as ort

        if optimized_path.exists():
            optimized_path.unlink()

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.optimized_model_filepath = str(optimized_path)

        ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        if optimized_path.exists():
            return optimized_path
    except Exception as e:
        print(f"⚠️ ORT optimization failed for {model_path}: {e}")
        if optimized_path.exists():
            optimized_path.unlink()
    return None


def rewrite_model_external_data(
    model_path: str | Path,
    *,
    use_external_data: bool = True,
    suffix: str = ".onnx_data",
    size_threshold: int = 1024,
    ort_optimize: bool = True,
) -> Path | None:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    sidecar_path = sidecar_path_for_model(path, suffix=suffix)
    optimized_path: Path | None = None
    load_path = path
    try:
        if ort_optimize:
            optimized_path = _optimize_with_ort(path)
            if optimized_path is not None:
                load_path = optimized_path

        model = onnx.load(str(load_path), load_external_data=True)

        if use_external_data:
            onnx.save_model(
                model,
                str(path),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=sidecar_path.name,
                size_threshold=size_threshold,
                convert_attribute=False,
            )
            return sidecar_path

        onnx.save_model(model, str(path), save_as_external_data=False)
        if sidecar_path.exists():
            sidecar_path.unlink()
        return None
    finally:
        if optimized_path is not None and optimized_path.exists():
            try:
                optimized_path.unlink()
            except Exception as e:
                print(f"⚠️ Failed to clean up temporary optimized model {optimized_path}: {e}")