from pathlib import Path

import onnx


SUPPORTED_OPTIMIZE_STAGES = {"onnxsim", "ort"}


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
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
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


def _optimize_with_onnxsim(model_path: Path) -> Path | None:
    optimized_path = model_path.with_name(f"{model_path.stem}.onnxsim.onnx")
    try:
        from onnxsim import simplify

        if optimized_path.exists():
            optimized_path.unlink()

        model = onnx.load(str(model_path), load_external_data=True)
        simplified, check = simplify(model)
        if not check:
            print(f"⚠️ onnxsim reported verification failure for {model_path}")
            return None

        onnx.save_model(simplified, str(optimized_path), save_as_external_data=False)
        return optimized_path
    except Exception as e:
        print(f"⚠️ onnxsim optimization failed for {model_path}: {e}")
        if optimized_path.exists():
            optimized_path.unlink()
    return None


def _normalize_optimize_stages(*, optimize_stages: str | None, ort_optimize: bool) -> list[str]:
    if optimize_stages is None:
        return ["ort"] if ort_optimize else []

    stages = [stage.strip().lower() for stage in optimize_stages.split(",") if stage.strip()]
    for stage in stages:
        if stage not in SUPPORTED_OPTIMIZE_STAGES:
            raise ValueError(
                f"Unsupported optimization stage '{stage}'. Supported stages: {sorted(SUPPORTED_OPTIMIZE_STAGES)}"
            )
    return stages


def rewrite_model_external_data(
    model_path: str | Path,
    *,
    use_external_data: bool = True,
    suffix: str = ".onnx_data",
    size_threshold: int = 1024,
    ort_optimize: bool = True,
    optimize_stages: str | None = None,
) -> Path | None:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    sidecar_path = sidecar_path_for_model(path, suffix=suffix)
    temporary_paths: list[Path] = []
    load_path = path
    try:
        stages = _normalize_optimize_stages(optimize_stages=optimize_stages, ort_optimize=ort_optimize)
        for stage in stages:
            optimized_path: Path | None = None
            if stage == "onnxsim":
                optimized_path = _optimize_with_onnxsim(load_path)
            elif stage == "ort":
                optimized_path = _optimize_with_ort(load_path)

            if optimized_path is not None:
                temporary_paths.append(optimized_path)
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
        for temporary_path in temporary_paths:
            if temporary_path.exists():
                try:
                    temporary_path.unlink()
                except Exception as e:
                    print(f"⚠️ Failed to clean up temporary optimized model {temporary_path}: {e}")