import argparse
import sys
import beartype.claw
import torch
import torch.nn as nn
import os
import onnx

beartype.claw.beartype_this_package = lambda *args, **kwargs: None

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.default_parameters import DEFAULT_VARIANT
from pocket_tts.modules.stateful_module import init_states
import pocket_tts.modules.transformer as transformer_module
from onnx_export.export_utils import get_state_structure, flatten_state

# ==============================================================================
# 1. MONKEYPATCHES
# ==============================================================================

def patched_complete_kv(cache: torch.Tensor, offset: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    if offset.numel() > 1 and not torch.all(offset == offset.view(-1)[0]):
        raise ValueError("Linear cache offset must be identical across batch.")
    offset_value = offset.view(-1)[0]

    cache[0, :, offset_value : offset_value + k.shape[1]] = k
    cache[1, :, offset_value : offset_value + v.shape[1]] = v
    valid = cache[:, :, : offset_value + k.shape[1]]
    return valid[0], valid[1]


transformer_module.complete_kv = patched_complete_kv

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
            for k, v in  sorted(struct.items()):
                if isinstance(v, dict): e = unflatten_recursive(v)
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
        transformer_out = self.flow_lm.backbone(input_, text_embeddings, sequence, model_state=model_state)
        
        # Extract Conditioning
        batch_size = transformer_out.shape[0]
        dim = transformer_out.shape[2]
        dummy_out = torch.zeros((batch_size, 1, dim), device=transformer_out.device, dtype=transformer_out.dtype)
        
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
            if "step" in s: s["step"] = s["step"] + increment
            if "offset" in s: s["offset"] = s["offset"] + increment
            for k, v in s.items(): 
                if isinstance(v, dict): recurse_increment(v)
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


class FlowLMFullWrapper(nn.Module):
    """
    Fused version of generateLatents.  Runs autoregressive backbone +
    LSD-integrated flow in a single forward call.  Outputs full latent
    sequence and eos logits for each step, along with final flattened state.

    Inputs:
      - text_embeddings: (B, Text, 1024)
      - state_*: flattened state inputs as before
      - max_frames: [1] int64
      - lsd_steps: [1] int64
      - temperature: [1] float32
      - frames_after_eos: [1] int64  (not currently used for early stopping)
      - st_s: (lsd_steps,) float32
      - st_t: (lsd_steps,) float32
      - noise: (B, max_frames, 32) float32 - per-frame gaussian noise

    Outputs:
      - latents: (B, max_frames, 32)
      - eos_logits: (B, max_frames, 1)
      - out_state_*: final flattened state
    """
    def __init__(self, flow_lm, state_structure, eos_threshold=-4.0):
        super().__init__()
        self.flow_lm = flow_lm
        self.state_structure = state_structure
        self.eos_threshold = eos_threshold

    def forward(
        self,
        text_embeddings,
        state_flat,
        max_frames,
        lsd_steps,
        temperature,
        frames_after_eos,
        st_s,
        st_t,
        noise,
    ):
        # rebuild state dict
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

        # the conditioning pass was previously used to prime the backbone with
        # a BOS token, but it had the side‑effect of materialising large weight
        # tensors as computed constants in the graph which prevented
        # quantization.  we can initialize the autoregressive sequence directly
        # and skip this extra call.
        seq = torch.full((text_embeddings.shape[0], 1, self.flow_lm.ldim), float('nan'))
        seq = torch.where(torch.isnan(seq), self.flow_lm.bos_emb, seq)

        # pre‑allocate outputs
        B = text_embeddings.shape[0]
        latents_accum = []
        eos_accum = []

        dt = 1.0 / torch.clamp(lsd_steps, min=1)

        # iterative generation loop – iterate over the dynamic length of the
        # provided noise tensor rather than converting a scalar to a Python int.
        # This allows ONNX export to emit a Loop operator and support arbitrary
        # utterance lengths instead of unrolling a single step.
        for step in range(noise.shape[1]):
            # autoregressive step
            seq = torch.where(torch.isnan(seq), self.flow_lm.bos_emb, seq)
            ar_in = self.flow_lm.input_linear(seq)
            ar_out = self.flow_lm.backbone(ar_in, text_embeddings, seq, model_state=model_state)
            c = ar_out[:, 0]
            eos_logit = self.flow_lm.out_eos(c)
            eos_accum.append(eos_logit)

            # LSD integration
            x = noise[:, step, :]
            for j in range(lsd_steps.item()):
                flow_dir = self.flow_lm.flow_net(
                    c,
                    st_s[j].view(1, 1),
                    st_t[j].view(1, 1),
                    x,
                )
                # Euler update
                x = x + flow_dir * dt
            latents_accum.append(x)

            # update state counters
            seq_len = seq.shape[1]
            text_len = text_embeddings.shape[1]
            incr = seq_len + text_len
            def recurse_inc(s):
                if 'step' in s: s['step'] = s['step'] + incr
                if 'offset' in s: s['offset'] = s['offset'] + incr
                for k, v in s.items():
                    if isinstance(v, dict): recurse_inc(v)
            recurse_inc(model_state)

            seq = x.unsqueeze(1)
        # stack outputs
        latents_out = torch.stack(latents_accum, dim=1)
        eos_out = torch.stack(eos_accum, dim=1)
        from onnx_export.export_utils import flatten_state as fs
        out_state = fs(model_state)
        return latents_out, eos_out, *out_state


# ==============================================================================
# 3. EXPORT SCRIPT
# ==============================================================================

def main():
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description="Export FlowLM models to ONNX.")
    parser.add_argument("--output_dir", "-o", type=str, default="onnx_models", help="Directory for output ONNX files")
    parser.add_argument("--weights_path", "-w", type=str, default="weights/tts_b6369a24.safetensors", help="Path to weights file used to load FlowLM")
    parser.add_argument(
        "--pytorch-quantize",
        action="store_true",
        help="Apply PyTorch dynamic quantization to the model before export (int8 weights)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...")
    tts = TTSModel.load_model(DEFAULT_VARIANT).cpu().eval()
    if args.pytorch_quantize:
        print("Applying PyTorch dynamic quantization to FlowLM...")
        # quantize all Linear layers in place; this produces qint8 weights
        tts = torch.quantization.quantize_dynamic(tts, {torch.nn.Linear}, dtype=torch.qint8)
    
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
    torch.onnx.export(
        main_wrapper, main_args, main_out_path,
        input_names=["sequence", "text_embeddings"] + state_input_names,
        output_names=["conditioning", "eos_logit"] + state_output_names,
        dynamic_axes={"sequence": {1: "seq_len"}, "text_embeddings": {1: "text_len"}},
        opset_version=14, dynamo=False
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
        flow_wrapper, flow_args, flow_out_path,
        input_names=["c", "s", "t", "x"],
        output_names=["flow_dir"],
        dynamic_axes={"c": {0: "batch"}, "s": {0: "batch"}, "t": {0: "batch"}, "x": {0: "batch"}},
        opset_version=14, dynamo=False
    )
    print(f"Exported {flow_out_path}")

    # -------------------------------------------------------------
    # 3. Fused Generator Model (AR + Flow loops)
    print("\nExporting Fused AR+Flow Generator Model...")
    full_wrapper = FlowLMFullWrapper(tts.flow_lm, structure)

    # scripting the wrapper helps the exporter preserve loops that depend on
    # tensor shapes.  the unscripted version was unrolling the loop at export
    # time using the dummy noise shape, leading to a model that could only
    # ever generate a single frame.
    try:
        full_wrapper = torch.jit.script(full_wrapper)
    except Exception:
        # scripting might fail in some environments; fall back gracefully
        pass

    # dummy placeholders
    dummy_text = torch.randn(1, 1, tts.flow_lm.dim)
    dummy_state = flat_state
    dummy_max = torch.tensor([1], dtype=torch.int64)
    dummy_lsd = torch.tensor([1], dtype=torch.int64)
    dummy_temp = torch.tensor([0.25], dtype=torch.float32)
    dummy_after = torch.tensor([1], dtype=torch.int64)
    dummy_st_s = torch.tensor([0.0], dtype=torch.float32)
    dummy_st_t = torch.tensor([1.0], dtype=torch.float32)
    # use a slightly longer dummy noise so that an unrolled graph still has a
    # reasonable number of steps if scripting fails.
    dummy_noise = torch.randn(1, 8, tts.flow_lm.ldim)

    full_args = (
        dummy_text,
        dummy_state,
        dummy_max,
        dummy_lsd,
        dummy_temp,
        dummy_after,
        dummy_st_s,
        dummy_st_t,
        dummy_noise,
    )

    full_out_path = os.path.join(args.output_dir, "flow_lm_full.onnx")
    torch.onnx.export(
        full_wrapper,
        full_args,
        full_out_path,
        input_names=(
            ["text_embeddings"]
            + state_input_names
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
        output_names=(
            ["latents", "eos_logits"] + state_output_names
        ),
        dynamic_axes={
            "text_embeddings": {1: "text_len"},
            "noise": {1: "max_frames"},
            "st_s": {0: "lsd_steps"},
            "st_t": {0: "lsd_steps"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported {full_out_path}")

    print("\nDone! 3-Model export complete.")

if __name__ == "__main__":
    main()
