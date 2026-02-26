import pytest

# some of these tests touch PyTorch/ONNX export; skip entire file if torch
# isn't available in the current environment.
torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")

from scripts.export_flow_lm import FlowLMFullWrapper


def make_dummy_flow():
    """Simple stand-in for FlowLMModel that satisfies the attributes
    referenced by the export wrapper.  All computations are no-ops so the
    exported graph is tiny and fast to trace.
    """
    class DummyBackbone(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
        def forward(self, input_, text_embeddings, seq, model_state=None):
            # return zeros of appropriate shape
            B, T, _ = input_.shape
            return torch.zeros((B, T, self.dim), dtype=input_.dtype, device=input_.device)

    class DummyFlowModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ldim = 32
            self.dim = 1024
            self.bos_emb = torch.zeros(self.ldim)
            self.input_linear = torch.nn.Identity()
            self.backbone = DummyBackbone(self.dim)
            self.out_eos = torch.nn.Linear(self.dim, 1)
            torch.nn.init.zeros_(self.out_eos.weight)
            torch.nn.init.zeros_(self.out_eos.bias)
            self.flow_net = torch.nn.Identity()

    return DummyFlowModel()


def test_export_flow_lm_full_contains_loop(tmp_path):
    """Exporting the fused generator should produce an ONNX graph with a
    Loop node driven by the noise length.  A dynamic axis is also added so
    the model can accept variable-sized sequences at runtime.
    """
    flow = make_dummy_flow()
    wrapper = FlowLMFullWrapper(flow, {})

    # script the module to help the exporter handle loops properly; the
    # production script does this wrapped in a try/except so we mimic it here.
    try:
        wrapper = torch.jit.script(wrapper)
    except Exception:
        pass

    dummy_text = torch.randn(1, 1, flow.dim)
    dummy_state = []
    dummy_max = torch.tensor([1], dtype=torch.int64)
    dummy_lsd = torch.tensor([1], dtype=torch.int64)
    dummy_temp = torch.tensor([0.], dtype=torch.float32)
    dummy_after = torch.tensor([1], dtype=torch.int64)
    dummy_st_s = torch.tensor([0.0], dtype=torch.float32)
    dummy_st_t = torch.tensor([1.0], dtype=torch.float32)

    # choose a noise length greater than 1 so we can detect dynamic looping
    dummy_noise = torch.randn(1, 5, flow.ldim)

    out_path = tmp_path / "flow_lm_full_test.onnx"
    torch.onnx.export(
        wrapper,
        (
            dummy_text,
            dummy_state,
            dummy_max,
            dummy_lsd,
            dummy_temp,
            dummy_after,
            dummy_st_s,
            dummy_st_t,
            dummy_noise,
        ),
        str(out_path),
        input_names=(
            ["text_embeddings"]
            + [f"state_{i}" for i in range(len(dummy_state))]
            + [
                "max_frames",
                "lsd_steps",
                "temperature",
                "frames_after_eos",
                "st_s",
                "st_t",
                "noise",
            ]
        ),
        output_names=(["latents", "eos_logits"] + [f"out_state_{i}" for i in range(len(dummy_state))]),
        dynamic_axes={
            "text_embeddings": {1: "text_len"},
            "noise": {1: "max_frames"},
            "st_s": {0: "lsd_steps"},
            "st_t": {0: "lsd_steps"},
        },
        opset_version=17,
        dynamo=False,
    )

    model = onnx.load(str(out_path))
    assert any(node.op_type == "Loop" for node in model.graph.node), "exported graph must contain a Loop op"
