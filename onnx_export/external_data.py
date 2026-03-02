from pathlib import Path

import onnx


def sidecar_path_for_model(model_path: Path, suffix: str = ".onnx_data") -> Path:
    cleaned_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return model_path.with_name(f"{model_path.stem}{cleaned_suffix}")


def rewrite_model_external_data(
    model_path: str | Path,
    *,
    use_external_data: bool = True,
    suffix: str = ".onnx_data",
    size_threshold: int = 1024,
) -> Path | None:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    sidecar_path = sidecar_path_for_model(path, suffix=suffix)
    model = onnx.load(str(path), load_external_data=True)

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