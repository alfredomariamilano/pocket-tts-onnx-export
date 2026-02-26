
import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

# order is irrelevant; included for completeness
# NOTE: "flow_lm_full" is a large fused graph containing Python-style
# loops/Loop ops.  The ONNXRuntime MatMulNBits quantizer currently has
# trouble recognising constant weights inside such subgraphs, so quantizing
# it may not shrink the model and in fact can make the file larger.  We
# detect that later and drop any unhelpful artifacts, so consumers should
# still fall back to the float32 version when necessary.
MODELS_TO_QUANTIZE = [
    "flow_lm_main",
    "flow_lm_flow",
    "flow_lm_full",  # fused AR + flow generator (may not quantize)
    "mimi_decoder",
    "mimi_encoder",
    "text_conditioner",
]

SUPPORTED_PRECISIONS = ("int8", "q4")


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


def _maybe_strip_if_no_reduction(input_path: Path, output_path: Path) -> None:
    """Remove an output file if quantization did not shrink it.

    A few of our models (notably the fused *flow_lm_full* graph) contain
    MatMul operations whose weights are not treated as constants by the
    quantizer.  In those cases the quantized version can actually be larger
    than the float32 original due to metadata/scale tensors being written back
    into the file.  We detect that situation and delete the quantized model so
    consumers fall back to the FP32 variant.
    """
    try:
        orig = input_path.stat().st_size
        new = output_path.stat().st_size
        if new >= orig:
            reduction = (orig - new) / orig * 100
            print(
                f"  ⚠️ Quantized file is not smaller ({orig/1e6:.1f}MB -> {new/1e6:.1f}MB, {reduction:.1f}%).");
            # TODO: keep file
            # print(f"     removing {output_path.name} and keeping float32 model")
            # output_path.unlink()
    except Exception:
        # permission errors or missing file - ignore
        pass


def quantize_file_int8(input_path: Path, output_path: Path, op_types: tuple[str, ...] = ("MatMul", "Gemm")) -> None:
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
            extra_options={"ForceQuantizeNoType": True, "DefaultTensorType": 1},
        )

        _print_reduction(input_path, output_path)
        _maybe_strip_if_no_reduction(input_path, output_path)
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

        quantizer = MatMulNBitsQuantizer(
            model=str(temp_path),
            bits=4,
            block_size=block_size,
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=op_types,
        )
        quantizer.process()
        quantizer.model.save_model_to_file(str(output_path), use_external_data_format=False)

        _print_reduction(input_path, output_path)
        _maybe_strip_if_no_reduction(input_path, output_path)
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
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    precisions = _resolve_precisions(args.precision)

    print(f"Starting Quantization: {input_dir} -> {output_dir}")
    print(f"Requested precisions: {', '.join(precisions)}")
    print("Using MatMul/Gemm quantization for broad CPU compatibility.")

    for model_name in MODELS_TO_QUANTIZE:
        in_file = input_dir / f"{model_name}.onnx"
        for precision in precisions:
            out_file = output_dir / _output_filename(model_name, precision)
            if precision == "int8":
                quantize_file_int8(in_file, out_file)
            elif precision == "q4":
                quantize_file_q4_weight_only(in_file, out_file, block_size=args.q4_block_size)

    print("\nQuantization routine finished.")

if __name__ == "__main__":
    main()
