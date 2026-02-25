import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, checker, helper
from onnx.compose import add_prefix, merge_models
from onnx import version_converter


REQUIRED_MODELS = (
    "text_conditioner.onnx",
    "flow_lm_main.onnx",
    "flow_lm_flow.onnx",
    "mimi_decoder.onnx",
)


def _create_flow_to_latent_adapter(opset_version: int, ir_version: int) -> onnx.ModelProto:
    input_vi = helper.make_tensor_value_info("flow_dir", TensorProto.FLOAT, ["batch", 32])
    output_vi = helper.make_tensor_value_info("latent", TensorProto.FLOAT, ["batch", 1, 32])

    unsqueeze = helper.make_node(
        "Unsqueeze",
        inputs=["flow_dir", "axes"],
        outputs=["latent"],
        name="unsqueeze_latent_step",
    )

    axes_initializer = helper.make_tensor(
        name="axes",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[1],
    )

    graph = helper.make_graph(
        nodes=[unsqueeze],
        name="flow_to_latent_adapter",
        inputs=[input_vi],
        outputs=[output_vi],
        initializer=[axes_initializer],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset_version)],
        producer_name="pocket-tts-onnx-export",
    )
    model.ir_version = ir_version
    return model


def _load_required_models(onnx_dir: Path) -> dict[str, onnx.ModelProto]:
    loaded: dict[str, onnx.ModelProto] = {}
    for filename in REQUIRED_MODELS:
        model_path = onnx_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Required ONNX model not found: {model_path}")
        loaded[filename] = onnx.load(str(model_path))
    return loaded


def _normalize_versions(models: dict[str, onnx.ModelProto]) -> tuple[dict[str, onnx.ModelProto], int, int]:
    target_ir = max(model.ir_version for model in models.values())
    target_opset = max(model.opset_import[0].version for model in models.values())

    normalized: dict[str, onnx.ModelProto] = {}
    for name, model in models.items():
        current_opset = model.opset_import[0].version
        normalized_model = model
        if current_opset != target_opset:
            normalized_model = version_converter.convert_version(normalized_model, target_opset)
        normalized_model.ir_version = target_ir
        normalized[name] = normalized_model

    return normalized, target_ir, target_opset


def export_monolith_step(onnx_dir: Path, output_name: str) -> Path:
    models = _load_required_models(onnx_dir)
    models, target_ir, target_opset = _normalize_versions(models)

    text_conditioner = models["text_conditioner.onnx"]
    flow_main = add_prefix(models["flow_lm_main.onnx"], prefix="flow_main/")
    flow_flow = add_prefix(models["flow_lm_flow.onnx"], prefix="flow_flow/")
    mimi_decoder = add_prefix(models["mimi_decoder.onnx"], prefix="mimi/")

    adapter = add_prefix(
        _create_flow_to_latent_adapter(
            opset_version=target_opset,
            ir_version=target_ir,
        ),
        prefix="adapter/",
    )

    merged = merge_models(
        text_conditioner,
        flow_main,
        io_map=[("embeddings", "flow_main/text_embeddings")],
    )
    merged = merge_models(
        merged,
        flow_flow,
        io_map=[("flow_main/conditioning", "flow_flow/c")],
    )
    merged = merge_models(
        merged,
        adapter,
        io_map=[("flow_flow/flow_dir", "adapter/flow_dir")],
    )
    merged = merge_models(
        merged,
        mimi_decoder,
        io_map=[("adapter/latent", "mimi/latent")],
    )

    existing_output_names = {value_info.name for value_info in merged.graph.output}
    if "adapter/latent" not in existing_output_names:
        latent_output = helper.make_tensor_value_info(
            "adapter/latent",
            TensorProto.FLOAT,
            ["batch", 1, 32],
        )
        merged.graph.output.extend([latent_output])

    checker.check_model(merged)


    output_path = onnx_dir / output_name
    onnx.save(merged, str(output_path))
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an experimental single-file ONNX model for one Pocket-TTS generation step."
    )
    parser.add_argument(
        "--onnx-dir",
        default="hf/onnx",
        help="Directory containing exported split ONNX models",
    )
    parser.add_argument(
        "--output-name",
        default="monolith_step.onnx",
        help="Filename for the merged monolith-step model",
    )
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    out = export_monolith_step(onnx_dir=onnx_dir, output_name=args.output_name)
    print(f"✅ Exported monolith step model: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
