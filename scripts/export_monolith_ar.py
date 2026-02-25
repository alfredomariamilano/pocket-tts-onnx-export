import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, checker, helper


def _vi(name: str, data_type: int, shape: list[int | str]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, data_type, shape)


def _copy_value_info(value_info: onnx.ValueInfoProto, new_name: str | None = None) -> onnx.ValueInfoProto:
    dims: list[int | str] = []
    tensor_type = value_info.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("unk")
    return helper.make_tensor_value_info(new_name or value_info.name, tensor_type.elem_type, dims)


def _find_outputs(step_model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    outputs = {o.name: o for o in step_model.graph.output}

    latent_name = "adapter/latent"
    eos_name = "flow_main/eos_logit"
    audio_name = "mimi/audio_frame"

    required = [latent_name, eos_name, audio_name]
    for name in required:
        if name not in outputs:
            raise ValueError(
                f"monolith_step.onnx missing required output '{name}'. Re-export monolith_step first."
            )

    return outputs


def build_monolith_ar(step_model_path: Path, output_path: Path) -> None:
    step_model = onnx.load(str(step_model_path))
    step_graph = step_model.graph

    step_inputs = {i.name: i for i in step_graph.input}
    step_outputs = _find_outputs(step_model)

    flow_state_input_names = [name for name in step_inputs if name.startswith("flow_main/state_")]
    flow_state_input_names.sort(key=lambda n: int(n.split("_")[-1]))

    mimi_state_input_names = [name for name in step_inputs if name.startswith("mimi/state_")]
    mimi_state_input_names.sort(key=lambda n: int(n.split("_")[-1]))

    flow_state_output_names = [name.replace("state_", "out_state_") for name in flow_state_input_names]
    mimi_state_output_names = [name.replace("state_", "out_state_") for name in mimi_state_input_names]

    if "token_ids" not in step_inputs:
        raise ValueError("monolith_step.onnx input 'token_ids' is required")
    if "flow_main/sequence" not in step_inputs:
        raise ValueError("monolith_step.onnx input 'flow_main/sequence' is required")

    graph_inputs: list[onnx.ValueInfoProto] = [
        _copy_value_info(step_inputs["token_ids"]),
        _vi("noise", TensorProto.FLOAT, [1, "gen_len", 32]),
        _vi("frames_after_eos", TensorProto.INT64, [1, 1]),
    ]
    for name in flow_state_input_names:
        graph_inputs.append(_copy_value_info(step_inputs[name]))
    for name in mimi_state_input_names:
        graph_inputs.append(_copy_value_info(step_inputs[name]))

    graph_outputs: list[onnx.ValueInfoProto] = [
        _vi("audio", TensorProto.FLOAT, ["gen_len", 1, 1, 1920]),
        _vi("eos_logits", TensorProto.FLOAT, ["gen_len", 1, 1]),
    ]
    for name in flow_state_output_names:
        graph_outputs.append(_copy_value_info(step_outputs[name], new_name=f"final/{name}"))
    for name in mimi_state_output_names:
        graph_outputs.append(_copy_value_info(step_outputs[name], new_name=f"final/{name}"))

    trip_count_node = helper.make_node("Gather", ["noise_shape", "axis1"], ["trip_count_vec"], axis=0)
    trip_count_scalar = helper.make_node("Squeeze", ["trip_count_vec", "axes0"], ["trip_count"])

    noise_shape = helper.make_node("Shape", ["noise"], ["noise_shape"])

    init_sequence = helper.make_node("Constant", [], ["init_sequence"], value=helper.make_tensor("init_sequence_t", TensorProto.FLOAT, [1, 1, 32], [float("nan")] * 32))
    init_cond = helper.make_node("Constant", [], ["init_cond"], value=helper.make_tensor("init_cond_t", TensorProto.BOOL, [], [True]))
    init_eos_seen = helper.make_node("Constant", [], ["init_eos_seen"], value=helper.make_tensor("init_eos_seen_t", TensorProto.BOOL, [1, 1], [False]))
    init_frames_after = helper.make_node("Constant", [], ["init_frames_after"], value=helper.make_tensor("init_frames_after_t", TensorProto.INT64, [1, 1], [0]))

    loop_body_inputs: list[onnx.ValueInfoProto] = [
        _vi("iter", TensorProto.INT64, []),
        _vi("cond_in", TensorProto.BOOL, []),
        _copy_value_info(step_inputs["token_ids"], "token_ids_in"),
        _vi("sequence_in", TensorProto.FLOAT, [1, 1, 32]),
        _vi("frames_after_eos_in", TensorProto.INT64, [1, 1]),
        _vi("noise_in", TensorProto.FLOAT, [1, "gen_len", 32]),
        _vi("eos_seen_in", TensorProto.BOOL, [1, 1]),
        _vi("frames_after_in", TensorProto.INT64, [1, 1]),
    ]
    for name in flow_state_input_names:
        loop_body_inputs.append(_copy_value_info(step_inputs[name], f"carry/{name}"))
    for name in mimi_state_input_names:
        loop_body_inputs.append(_copy_value_info(step_inputs[name], f"carry/{name}"))

    loop_body_nodes: list[onnx.NodeProto] = []

    gather_noise = helper.make_node("Gather", ["noise_in", "iter"], ["noise_step"], axis=1)
    const_s = helper.make_node("Constant", [], ["const_s"], value=helper.make_tensor("const_s_t", TensorProto.FLOAT, [1, 1], [0.0]))
    const_t = helper.make_node("Constant", [], ["const_t"], value=helper.make_tensor("const_t_t", TensorProto.FLOAT, [1, 1], [1.0]))

    loop_body_nodes.extend([gather_noise, const_s, const_t])

    rename_input_map = {
        "token_ids": "token_ids_in",
        "flow_main/sequence": "sequence_in",
        "flow_flow/s": "const_s",
        "flow_flow/t": "const_t",
        "flow_flow/x": "noise_step",
    }
    for name in flow_state_input_names:
        rename_input_map[name] = f"carry/{name}"
    for name in mimi_state_input_names:
        rename_input_map[name] = f"carry/{name}"

    prefix = "step/"
    initializer_names = {init.name for init in step_graph.initializer}
    for node in step_graph.node:
        new_inputs = []
        for inp in node.input:
            if inp == "":
                new_inputs.append("")
            elif inp in rename_input_map:
                new_inputs.append(rename_input_map[inp])
            elif inp in initializer_names:
                new_inputs.append(inp)
            else:
                new_inputs.append(f"{prefix}{inp}")
        new_outputs = [f"{prefix}{out}" for out in node.output]
        new_name = f"{prefix}{node.name}" if node.name else ""
        new_node = helper.make_node(node.op_type, new_inputs, new_outputs, name=new_name)
        for attr in node.attribute:
            new_node.attribute.extend([attr])
        loop_body_nodes.append(new_node)

    eos_threshold = helper.make_node(
        "Constant",
        [],
        ["eos_threshold"],
        value=helper.make_tensor("eos_threshold_t", TensorProto.FLOAT, [1, 1], [-4.0]),
    )
    one_i64 = helper.make_node(
        "Constant",
        [],
        ["one_i64"],
        value=helper.make_tensor("one_i64_t", TensorProto.INT64, [1, 1], [1]),
    )
    axes01 = helper.make_node(
        "Constant",
        [],
        ["axes01"],
        value=helper.make_tensor("axes01_t", TensorProto.INT64, [2], [0, 1]),
    )

    eos_bool = helper.make_node("Greater", [f"{prefix}flow_main/eos_logit", "eos_threshold"], ["eos_bool"])
    eos_seen_out = helper.make_node("Or", ["eos_seen_in", "eos_bool"], ["eos_seen_out"])
    frames_after_inc = helper.make_node("Add", ["frames_after_in", "one_i64"], ["frames_after_inc"])
    frames_after_out = helper.make_node("Where", ["eos_seen_out", "frames_after_inc", "frames_after_in"], ["frames_after_out"])

    not_eos_seen = helper.make_node("Not", ["eos_seen_out"], ["not_eos_seen"])
    tail_less = helper.make_node("Less", ["frames_after_out", "frames_after_eos_in"], ["tail_less"])
    continue_tensor = helper.make_node("Or", ["not_eos_seen", "tail_less"], ["continue_tensor"])
    cond_out = helper.make_node("Squeeze", ["continue_tensor", "axes01"], ["cond_out"])

    active_not_eos = helper.make_node("Not", ["eos_seen_in"], ["active_not_eos"])
    active_tail_less = helper.make_node("Less", ["frames_after_in", "frames_after_eos_in"], ["active_tail_less"])
    active = helper.make_node("Or", ["active_not_eos", "active_tail_less"], ["active"])
    active_f = helper.make_node("Cast", ["active"], ["active_f"], to=TensorProto.FLOAT)
    masked_audio = helper.make_node("Mul", [f"{prefix}mimi/audio_frame", "active_f"], ["audio_scan"])

    loop_body_nodes.extend(
        [
            eos_threshold,
            one_i64,
            axes01,
            eos_bool,
            eos_seen_out,
            frames_after_inc,
            frames_after_out,
            not_eos_seen,
            tail_less,
            continue_tensor,
            cond_out,
            active_not_eos,
            active_tail_less,
            active,
            active_f,
            masked_audio,
        ]
    )

    loop_body_outputs: list[onnx.ValueInfoProto] = [
        _vi("cond_out", TensorProto.BOOL, []),
        _copy_value_info(step_inputs["token_ids"], "token_ids_out"),
        _vi("sequence_out", TensorProto.FLOAT, [1, 1, 32]),
        _vi("frames_after_eos_out", TensorProto.INT64, [1, 1]),
        _vi("noise_out", TensorProto.FLOAT, [1, "gen_len", 32]),
        _vi("eos_seen_out", TensorProto.BOOL, [1, 1]),
        _vi("frames_after_out", TensorProto.INT64, [1, 1]),
    ]
    for name in flow_state_output_names:
        loop_body_outputs.append(_copy_value_info(step_outputs[name], f"out/{name}"))
    for name in mimi_state_output_names:
        loop_body_outputs.append(_copy_value_info(step_outputs[name], f"out/{name}"))
    loop_body_outputs.extend(
        [
            _copy_value_info(step_outputs["mimi/audio_frame"], "audio_scan"),
            _copy_value_info(step_outputs["flow_main/eos_logit"], "eos_scan"),
        ]
    )

    passthrough_nodes = [
        helper.make_node("Identity", ["token_ids_in"], ["token_ids_out"]),
        helper.make_node("Identity", [f"{prefix}adapter/latent"], ["sequence_out"]),
        helper.make_node("Identity", ["frames_after_eos_in"], ["frames_after_eos_out"]),
        helper.make_node("Identity", ["noise_in"], ["noise_out"]),
        helper.make_node("Identity", [f"{prefix}flow_main/eos_logit"], ["eos_scan"]),
    ]
    loop_body_nodes.extend(passthrough_nodes)

    for name in flow_state_output_names:
        loop_body_nodes.append(helper.make_node("Identity", [f"{prefix}{name}"], [f"out/{name}"]))
    for name in mimi_state_output_names:
        loop_body_nodes.append(helper.make_node("Identity", [f"{prefix}{name}"], [f"out/{name}"]))

    body_graph = helper.make_graph(
        loop_body_nodes,
        "monolith_ar_body",
        loop_body_inputs,
        loop_body_outputs,
        initializer=list(step_graph.initializer),
        value_info=list(step_graph.value_info),
    )

    loop_inputs = [
        "trip_count",
        "init_cond",
        "token_ids",
        "init_sequence",
        "frames_after_eos",
        "noise",
        "init_eos_seen",
        "init_frames_after",
        *flow_state_input_names,
        *mimi_state_input_names,
    ]

    loop_outputs = [
        "_token_ids_final",
        "_sequence_final",
        "_frames_after_eos_final",
        "_noise_final",
        "_eos_seen_final",
        "_frames_after_final",
        *[f"final/{name}" for name in flow_state_output_names],
        *[f"final/{name}" for name in mimi_state_output_names],
        "audio",
        "eos_logits",
    ]

    loop_node = helper.make_node("Loop", loop_inputs, loop_outputs, body=body_graph)

    axes0_init = helper.make_tensor("axes0", TensorProto.INT64, [1], [0])
    axis1_init = helper.make_tensor("axis1", TensorProto.INT64, [1], [1])

    graph = helper.make_graph(
        [
            noise_shape,
            trip_count_node,
            trip_count_scalar,
            init_sequence,
            init_cond,
            init_eos_seen,
            init_frames_after,
            loop_node,
        ],
        "monolith_ar",
        graph_inputs,
        graph_outputs,
        initializer=[axes0_init, axis1_init],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", max(18, step_model.opset_import[0].version))],
        producer_name="pocket-tts-onnx-export",
    )
    model.ir_version = max(step_model.ir_version, model.ir_version)

    checker.check_model(model)
    onnx.save(model, str(output_path))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an experimental autoregressive monolith ONNX model using Loop over monolith_step"
    )
    parser.add_argument("--onnx-dir", default="hf/onnx", help="Directory containing monolith_step.onnx")
    parser.add_argument("--step-name", default="monolith_step.onnx", help="Monolith-step ONNX filename")
    parser.add_argument("--output-name", default="monolith_ar.onnx", help="Output autoregressive monolith filename")
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    step_path = onnx_dir / args.step_name
    out_path = onnx_dir / args.output_name

    if not step_path.exists():
        raise FileNotFoundError(f"Required step model not found: {step_path}")

    build_monolith_ar(step_model_path=step_path, output_path=out_path)
    print(f"✅ Exported autoregressive monolith model: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
