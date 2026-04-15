import argparse
import sys
import beartype.claw
import beartype
import torch
import torch.nn as nn
import os

beartype.claw.beartype_this_package = lambda *args, **kwargs: None
beartype.beartype = lambda *args, **kwargs: (lambda func: func) if not args else args[0]


# Monkeypatch trunc_normal_ to be ONNX-friendly
def patched_trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    with torch.no_grad():
        if std == 0:
            return tensor.fill_(mean).clamp_(a, b)
        return tensor.normal_(mean, std).clamp_(a, b)


torch.nn.init.trunc_normal_ = patched_trunc_normal_

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.default_parameters import DEFAULT_VARIANT
from pocket_tts.modules.stateful_module import init_states
import pocket_tts.modules.transformer as transformer_module
from onnx_export.export_utils import get_state_structure, flatten_state

# ==============================================================================
# 1. MONKEYPATCHES
# ==============================================================================


def patched_complete_kv(
    cache: torch.Tensor, offset: torch.Tensor, k: torch.Tensor, v: torch.Tensor
):
    current_offset = offset.view(-1)[0]
    new_cache = cache.clone()
    new_cache[0, :, current_offset : current_offset + k.shape[1]] = k
    new_cache[1, :, current_offset : current_offset + v.shape[1]] = v
    valid = new_cache[:, :, : current_offset + k.shape[1]]
    return new_cache, valid[0], valid[1]


def patched_append_and_get(self, k: torch.Tensor, v: torch.Tensor, state: dict | None):
    if state is None:
        k_attn = k.permute(0, 2, 1, 3)
        v_attn = v.permute(0, 2, 1, 3)
        pos_k = torch.arange(k_attn.shape[2], device=k_attn.device, dtype=torch.long)
        pos_k = pos_k.view(1, -1).expand(k_attn.shape[0], -1)
        offset = torch.zeros(k_attn.shape[0], device=k_attn.device, dtype=torch.long)
        return k_attn, v_attn, pos_k, offset
    new_cache, cache_k, cache_v = patched_complete_kv(
        state["cache"], state["offset"], k, v
    )
    state["cache"] = new_cache
    k_attn = cache_k.permute(0, 2, 1, 3)
    v_attn = cache_v.permute(0, 2, 1, 3)
    pos_k = torch.arange(k_attn.shape[2], device=k_attn.device, dtype=torch.long)
    pos_k = pos_k.view(1, -1).expand(k_attn.shape[0], -1)
    return k_attn, v_attn, pos_k, state["offset"]


transformer_module.complete_kv = patched_complete_kv
transformer_module._LinearKVCacheBackend.append_and_get = patched_append_and_get

# ==============================================================================
# 2. WRAPPERS
# ==============================================================================


class FlowLMMainWrapper(nn.Module):
    """
    Unified Backbone Model for both Conditioning and AR steps.
    Inputs:
      - sequence: (B, T, 32)
      - text_embeddings: (B, Text, 1024)
      - state_*: KVCache states
    Outputs:
      - conditioning: (B, 1024) - used for Flow step
      - eos_logit: (B, 1) - used for EOS detection
      - out_state_*: Updated states
    """

    def __init__(self, flow_lm, state_structure, eos_threshold=-4.0):
        super().__init__()
        self.flow_lm = flow_lm
        self.state_structure = state_structure
        self.eos_threshold = eos_threshold

    def forward(self, sequence, text_embeddings, state_flat):
        idx = 0

        def unflatten_recursive(struct):
            nonlocal idx
            s = {}
            for k, v in sorted(struct.items()):
                if isinstance(v, dict):
                    e = unflatten_recursive(v)
                else:
                    e = state_flat[idx]
                    idx += 1
                s[k] = e
            return s

        model_state = unflatten_recursive(self.state_structure)

        # Handle BOS replacement (NaN -> bos_emb)
        sequence = torch.where(torch.isnan(sequence), self.flow_lm.bos_emb, sequence)

        input_ = self.flow_lm.input_linear(sequence)

        # Backbone Forward Pass
        # Returns (B, T_new, 1024)
        transformer_out = self.flow_lm.backbone(
            input_, text_embeddings, sequence, model_state=model_state
        )

        # Extract Conditioning
        batch_size = transformer_out.shape[0]
        dim = transformer_out.shape[2]
        dummy_out = torch.zeros(
            (batch_size, 1, dim),
            device=transformer_out.device,
            dtype=transformer_out.dtype,
        )

        # We use [(transformer_out + dummy)]
        augmented_out = torch.cat([transformer_out, dummy_out], dim=1)
        c = augmented_out[:, 0]

        # Extract EOS logit
        eos_logit = self.flow_lm.out_eos(c)

        # State Increment
        seq_len = sequence.shape[1]
        text_len = text_embeddings.shape[1]
        increment = seq_len + text_len

        def recurse_increment(s):
            if "step" in s:
                s["step"] = s["step"] + increment
            if "offset" in s:
                s["offset"] = s["offset"] + increment
            for k, v in s.items():
                if isinstance(v, dict):
                    recurse_increment(v)

        recurse_increment(model_state)

        from onnx_export.export_utils import flatten_state as fs

        out_state = fs(model_state)

        return c, eos_logit, *out_state


class FlowNetWrapper(nn.Module):
    """
    Stateless wrapper for FlowNet.
    Inputs: c (conditioning), s (timestep), t (timestep), x (latent)
    Output: flow_dir
    """

    def __init__(self, flow_lm):
        super().__init__()
        self.flow_net = flow_lm.flow_net

    def forward(self, c, s, t, x):
        return self.flow_net(c, s, t, x)


# ==============================================================================
# 3. EXPORT SCRIPT
# ==============================================================================


def main():
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description="Export FlowLM models to ONNX.")
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="onnx_models",
        help="Directory for output ONNX files",
    )
    parser.add_argument(
        "--weights_path",
        "-w",
        type=str,
        default="weights/tts_b6369a24.safetensors",
        help="Path to weights file used to load FlowLM",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...")
    tts = TTSModel.load_model(DEFAULT_VARIANT).cpu().eval()

    # Reload weights if available to match production
    if os.path.exists(args.weights_path):
        import safetensors.torch

        print(f"Reloading weights from {args.weights_path}...")
        state_dict = safetensors.torch.load_file(args.weights_path)
        # Load only common keys or strict if possible; strict might fail if keys missing in state dict
        # Assuming safe load for now or rely on load_model matching the weights
        try:
            tts.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Failed to reload specified weights: {e}")

    # Init patched state
    STATIC_SEQ_LEN = 1000
    state = init_states(tts.flow_lm, batch_size=1, sequence_length=STATIC_SEQ_LEN)
    structure = get_state_structure(state)
    flat_state = flatten_state(state)

    state_input_names = [f"state_{i}" for i in range(len(flat_state))]
    state_output_names = [f"out_state_{i}" for i in range(len(flat_state))]

    # -------------------------------------------------------------
    # 1. Main Flow Model (Backbone)
    print("\nExporting FlowLM Main Model (Backbone)...")
    main_wrapper = FlowLMMainWrapper(tts.flow_lm, structure)

    # Needs to handle dynamic axes for both seq and text
    dummy_seq = torch.randn(1, 1, tts.flow_lm.ldim)
    dummy_text = torch.randn(1, 1, tts.flow_lm.dim)
    main_args = (dummy_seq, dummy_text, flat_state)

    main_out_path = os.path.join(args.output_dir, "flow_lm_main.onnx")
    main_dynamic_axes = {
        "sequence": {1: "seq_len"},
        "text_embeddings": {1: "text_len"},
    }
    for index, tensor in enumerate(flat_state):
        if tensor.ndim == 5:
            main_dynamic_axes[f"state_{index}"] = {2: f"cache_len_{index}"}
            main_dynamic_axes[f"out_state_{index}"] = {2: f"cache_len_out_{index}"}

    torch.onnx.export(
        main_wrapper,
        main_args,
        main_out_path,
        input_names=["sequence", "text_embeddings"] + state_input_names,
        output_names=["conditioning", "eos_logit"] + state_output_names,
        dynamic_axes=main_dynamic_axes,
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported {main_out_path}")

    # 2. Flow Net Model
    print("\nExporting Flow Net Model...")
    flow_wrapper = FlowNetWrapper(tts.flow_lm)
    # Inputs: c(B, 1024), s(B,1), t(B,1), x(B, 32)
    dummy_c = torch.randn(1, 1024)
    dummy_s = torch.tensor([[0.0]])
    dummy_t = torch.tensor([[1.0]])
    dummy_x = torch.randn(1, 32)

    flow_args = (dummy_c, dummy_s, dummy_t, dummy_x)

    flow_out_path = os.path.join(args.output_dir, "flow_lm_flow.onnx")
    torch.onnx.export(
        flow_wrapper,
        flow_args,
        flow_out_path,
        input_names=["c", "s", "t", "x"],
        output_names=["flow_dir"],
        dynamic_axes={
            "c": {0: "batch"},
            "s": {0: "batch"},
            "t": {0: "batch"},
            "x": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported {flow_out_path}")
    print("\nDone! 2-Model split optimization complete.")


if __name__ == "__main__":
    main()
