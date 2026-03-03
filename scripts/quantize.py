
import argparse
from pathlib import Path
from typing import Iterable

import onnx
from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from onnx_export.external_data import rewrite_model_external_data

MODELS_TO_QUANTIZE = [
    "flow_lm_main",
    "flow_lm_flow",
    "mimi_decoder",
    "mimi_encoder",
    "text_conditioner",
]

SUPPORTED_PRECISIONS = ("int8", "q4")


def _parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_model_node_filters(entries: Iterable[str] | None) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    if not entries:
        return mapping

    for entry in entries:
        model_name, sep, nodes_raw = entry.partition(":")
        key = model_name.strip()
        if not sep or not key:
            raise ValueError(
                f"Invalid model node filter '{entry}'. Expected format: model_name:node_a,node_b"
            )
        mapping[key] = _parse_csv_list(nodes_raw)
    return mapping


def _model_quant_filters(
    model_name: str,
    include_default: list[str],
    exclude_default: list[str],
    include_overrides: dict[str, list[str]],
    exclude_overrides: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    include = include_overrides.get(model_name, include_default)
    exclude = exclude_overrides.get(model_name, exclude_default)
    return include, exclude


def _resolve_precisions(selection: str) -> list[str]:
    if selection == "all":
        return list(SUPPORTED_PRECISIONS)
    if selection in SUPPORTED_PRECISIONS:
        return [selection]
    raise ValueError(f"Unsupported precision '{selection}'. Expected one of: int8, q4, all")


def _output_filename(model_name: str, precision: str) -> str:
    return f"{model_name}_{precision}.onnx"


def _print_reduction(input_path: Path, output_path: Path) -> None:
    size_orig = input_path.stat().st_size / (1024 * 1024)
    size_quant = output_path.stat().st_size / (1024 * 1024)
    reduction = (size_orig - size_quant) / size_orig * 100
    print(f"  ✅ Complete: {size_orig:.1f}MB -> {size_quant:.1f}MB ({reduction:.1f}% reduction)")


def quantize_file_int8(
    input_path: Path,
    output_path: Path,
    op_types: tuple[str, ...] = ("MatMul",),
    nodes_to_quantize: list[str] | None = None,
    nodes_to_exclude: list[str] | None = None,
) -> None:
    if not input_path.exists():
        print(f"⚠️ Skipping {input_path.name} (not found)")
        return

    print(f"Quantizing {input_path.name} -> {output_path.name} (int8 dynamic)...")
    temp_path = output_path.with_suffix(".temp.onnx")

    try:
        print("  Running shape inference...")
        model = onnx.load(str(input_path))
        model = onnx.shape_inference.infer_shapes(model)
        onnx.save(model, str(temp_path))

        quantize_dynamic(
            model_input=str(temp_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=op_types,
            nodes_to_quantize=nodes_to_quantize,
            nodes_to_exclude=nodes_to_exclude,
            extra_options={"ForceQuantizeNoType": True, "DefaultTensorType": 1},
        )

        _print_reduction(input_path, output_path)
    except Exception as e:
        print(f"  ❌ INT8 quantization failed for {input_path.name}: {e}")
        if output_path.exists():
            output_path.unlink()
    finally:
        if temp_path.exists():
            temp_path.unlink()


def quantize_file_q4_weight_only(
    input_path: Path,
    output_path: Path,
    block_size: int = 128,
    op_types: tuple[str, ...] = ("MatMul",),
    nodes_to_quantize: list[str] | None = None,
    nodes_to_exclude: list[str] | None = None,
) -> None:
    if not input_path.exists():
        print(f"⚠️ Skipping {input_path.name} (not found)")
        return

    print(f"Quantizing {input_path.name} -> {output_path.name} (q4 weight-only)...")
    temp_path = output_path.with_suffix(".temp.onnx")

    try:
        print("  Running shape inference...")
        model = onnx.load(str(input_path))
        model = onnx.shape_inference.infer_shapes(model)
        onnx.save(model, str(temp_path))

        kwargs = {
            "model": str(temp_path),
            "bits": 4,
            "block_size": block_size,
            "quant_format": QuantFormat.QOperator,
            "op_types_to_quantize": op_types,
        }

        if nodes_to_quantize:
            kwargs["nodes_to_quantize"] = nodes_to_quantize
        if nodes_to_exclude:
            kwargs["nodes_to_exclude"] = nodes_to_exclude

        try:
            quantizer = MatMulNBitsQuantizer(**kwargs)
        except TypeError:
            kwargs.pop("nodes_to_quantize", None)
            kwargs.pop("nodes_to_exclude", None)
            print("  ⚠️ MatMulNBitsQuantizer node include/exclude filters unsupported in this ORT build; using op-type-only q4 quantization.")
            quantizer = MatMulNBitsQuantizer(**kwargs)
        quantizer.process()
        quantizer.model.save_model_to_file(str(output_path), use_external_data_format=False)

        _print_reduction(input_path, output_path)
    except Exception as e:
        print(f"  ❌ Q4 quantization failed for {input_path.name}: {e}")
        if output_path.exists():
            output_path.unlink()
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize PocketTTS ONNX models (int8 dynamic and/or q4 weight-only)."
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        default="onnx",
        help="Input directory containing FP32 ONNX models",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="onnx_int8",
        help="Output directory for quantized ONNX models",
    )
    parser.add_argument(
        "--precision",
        choices=["int8", "q4", "all"],
        default="int8",
        help="Quantization precision to emit",
    )
    parser.add_argument(
        "--q4-block-size",
        type=int,
        default=128,
        help="Block size for q4 weight-only quantization",
    )
    parser.add_argument(
        "--nodes-include",
        type=str,
        default="",
        help="Comma-separated node names to quantize (global default for all models)",
    )
    parser.add_argument(
        "--nodes-exclude",
        type=str,
        default="",
        help="Comma-separated node names to keep in FP precision (global default for all models)",
    )
    parser.add_argument(
        "--model-nodes-include",
        action="append",
        default=[],
        help="Per-model include override in format model_name:node_a,node_b (repeatable)",
    )
    parser.add_argument(
        "--model-nodes-exclude",
        action="append",
        default=[],
        help="Per-model exclude override in format model_name:node_x,node_y (repeatable)",
    )
    parser.add_argument(
        "--external-data",
        dest="external_data",
        action="store_true",
        help="Save tensor weights to external data sidecar files",
    )
    parser.add_argument(
        "--no-external-data",
        dest="external_data",
        action="store_false",
        help="Keep tensor weights embedded in the .onnx files",
    )
    parser.add_argument(
        "--external-data-suffix",
        type=str,
        default=".onnx_data",
        help="Suffix for per-model external tensor data files",
    )
    parser.add_argument(
        "--ort-optimize",
        dest="ort_optimize",
        action="store_true",
        help="Run ONNX Runtime graph optimization before final save",
    )
    parser.add_argument(
        "--no-ort-optimize",
        dest="ort_optimize",
        action="store_false",
        help="Disable ONNX Runtime graph optimization before final save",
    )
    parser.add_argument(
        "--optimize-stages",
        type=str,
        default=None,
        help="Comma-separated optimization stages before save (supported: onnxsim,ort)",
    )
    parser.set_defaults(external_data=True, ort_optimize=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    precisions = _resolve_precisions(args.precision)
    include_default = _parse_csv_list(args.nodes_include)
    exclude_default = _parse_csv_list(args.nodes_exclude)
    include_overrides = _parse_model_node_filters(args.model_nodes_include)
    exclude_overrides = _parse_model_node_filters(args.model_nodes_exclude)

    print(f"Starting Quantization: {input_dir} -> {output_dir}")
    print(f"Requested precisions: {', '.join(precisions)}")
    print("Using MatMul-only quantization for broad CPU compatibility.")

    for model_name in MODELS_TO_QUANTIZE:
        in_file = input_dir / f"{model_name}.onnx"
        model_include, model_exclude = _model_quant_filters(
            model_name=model_name,
            include_default=include_default,
            exclude_default=exclude_default,
            include_overrides=include_overrides,
            exclude_overrides=exclude_overrides,
        )

        if model_include:
            print(f"- {model_name}: include node filters = {model_include}")
        if model_exclude:
            print(f"- {model_name}: exclude node filters = {model_exclude}")

        for precision in precisions:
            out_file = output_dir / _output_filename(model_name, precision)
            if precision == "int8":
                quantize_file_int8(
                    in_file,
                    out_file,
                    nodes_to_quantize=model_include or None,
                    nodes_to_exclude=model_exclude or None,
                )
            elif precision == "q4":
                quantize_file_q4_weight_only(
                    in_file,
                    out_file,
                    block_size=args.q4_block_size,
                    nodes_to_quantize=model_include or None,
                    nodes_to_exclude=model_exclude or None,
                )

            if out_file.exists():
                sidecar = rewrite_model_external_data(
                    out_file,
                    use_external_data=args.external_data,
                    suffix=args.external_data_suffix,
                    ort_optimize=args.ort_optimize,
                    optimize_stages=args.optimize_stages,
                )
                if sidecar is not None:
                    print(f"  ↳ external tensor data: {sidecar.name}")

    print("\nQuantization routine finished.")

if __name__ == "__main__":
    main()
