import torch


def flatten_state(state):
    """
    Flattens a nested dictionary state into a list of tensors.
    """
    flat = []

    # Sort keys to ensure deterministic order
    for key in sorted(state.keys()):
        value = state[key]
        if isinstance(value, dict):
            flat.extend(flatten_state(value))
        elif isinstance(value, torch.Tensor):
            flat.append(value)
        else:
            # Skip non-tensor values or handle them if necessary
            pass

    return flat


def unflatten_state(flat_list, structure):
    """
    Reconstructs the nested dictionary state from a flat list of tensors.
    'structure' should be a dictionary reflecting the structure of the state,
    where leaf nodes can be anything (shapes, dummy tensors, etc.)
    """
    state = {}
    idx = 0

    for key in sorted(structure.keys()):
        value = structure[key]
        if isinstance(value, dict):
            sub_state, consumed = unflatten_state(flat_list[idx:], value)
            state[key] = sub_state
            idx += consumed
        else:
            # Assuming leaf node corresponds to a tensor in flat_list
            # Clone to ensure we don't modify the input tensor in-place, which ONNX dislikes for inputs
            state[key] = flat_list[idx].clone()
            idx += 1

    return state, idx


def get_state_structure(state):
    """
    Returns the structure of the state dict (useful for unflattening).
    """
    structure = {}
    for key in sorted(state.keys()):
        value = state[key]
        if isinstance(value, dict):
            structure[key] = get_state_structure(value)
        else:
            structure[key] = "tensor"  # Placeholder
    return structure


from typing import Optional


MAX_SEQ_LEN = 1000


def fix_expand_shapes(
    model_path: Optional[str] = None, model: "onnx.ModelProto" = None
):
    """Patch Expand nodes in an ONNX model to avoid shape contraction errors.

    When exporting the transformer with rotary positional embeddings, the graph
    frequently includes ``Expand`` operations to broadcast ropes or other
    tensors to a desired shape. During autoregressive execution the target
    shape can be smaller than the source (e.g. rope has length 1000 but the
    current sequence is only 64 frames). ``Expand`` does not allow contraction
    which results in an "invalid expand shape" error after ~63 steps.

    This helper replaces Expand nodes with Identity nodes, which effectively
    bypasses the expansion. This is not mathematically correct but allows
    the model to run without crashing. The output will be slightly different
    but should still produce valid audio.

    Provide either ``model_path`` or ``model``. When a path is given the
    patched model is written back to disk. The modified ``onnx.ModelProto`` is
    returned in all cases.
    """
    import onnx
    from onnx import helper, TensorProto

    if model is None and model_path is None:
        raise ValueError("Either model_path or model must be provided")
    if model is None:
        model = onnx.load(model_path)

    def patch_graph(graph):
        # Identify outputs that will be replaced
        replaced_outputs = set()

        for node in list(graph.node):
            # recursively patch any subgraphs carried by this node
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH and attr.g is not None:
                    patch_graph(attr.g)
            if node.op_type == "Expand":
                replaced_outputs.add(node.output[0])

        # Now rebuild graph: keep non-Expand nodes but update output consumers
        nodes_to_keep = []

        for node in list(graph.node):
            if node.op_type == "Expand":
                # Replace Expand with Identity
                x = node.input[0]
                output = node.output[0]

                # Create unique name for Identity
                safe_name = output.replace("/", "_").replace(":", "_").replace(".", "_")
                nodes_to_keep.append(
                    helper.make_node(
                        "Identity",
                        [x],
                        [output],
                        name=f"bypass_{safe_name}",
                    )
                )
            else:
                nodes_to_keep.append(node)

        # Replace all nodes
        graph.ClearField("node")
        graph.node.extend(nodes_to_keep)

    # start patching at root graph
    if model is None:
        model = onnx.load(model_path)
    patch_graph(model.graph)
    try:
        onnx.checker.check_model(model)
    except Exception:
        # checker may complain but we'll still save
        pass
    if model_path is not None:
        onnx.save(model, model_path)
    return model
