from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tokenizers import Tokenizer

from onnx_export.export_utils import flatten_state
from pocket_tts.default_parameters import DEFAULT_VARIANT
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import increment_steps, init_states


MIMI_STATIC_SEQ_LEN = 2048


def _load_flow_state_inputs(session: ort.InferenceSession, model_state: dict[str, dict[str, torch.Tensor]]) -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {}
    state_names = [input_info.name for input_info in session.get_inputs() if input_info.name.startswith("state_")]
    for index, state_name in enumerate(state_names):
        layer = index // 2
        kind = "cache" if index % 2 == 0 else "offset"
        tensor = model_state[f"transformer.layers.{layer}.self_attn"][kind]
        feeds[state_name] = tensor.detach().cpu().numpy()
    return feeds


def _load_mimi_state_inputs(session: ort.InferenceSession, mimi_state: dict) -> dict[str, np.ndarray]:
    flat_state = flatten_state(mimi_state)
    state_names = [input_info.name for input_info in session.get_inputs() if input_info.name.startswith("state_")]
    return {
        state_name: flat_state[index].detach().cpu().numpy()
        for index, state_name in enumerate(state_names)
    }


def _audio_frame_native(model: TTSModel, latent: torch.Tensor, mimi_state: dict) -> torch.Tensor:
    mimi_decoding_input = latent * model.flow_lm.emb_std + model.flow_lm.emb_mean
    transposed = mimi_decoding_input.transpose(-1, -2)
    quantized = model.mimi.quantizer(transposed)
    audio_frame = model.mimi.decode_from_latent(quantized, mimi_state)
    increment_steps(model.mimi, mimi_state, increment=16)
    return audio_frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic native-vs-ONNX PocketTTS step comparison")
    parser.add_argument("--text", default="This is a direct quality comparison between the ONNX Node runtime and the native Python Pocket TTS runtime.")
    parser.add_argument("--voice", default="marius")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--onnx-dir", default="hf/onnx")
    parser.add_argument("--tokenizer", default="hf/tokenizer.json")
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    model = TTSModel.load_model(DEFAULT_VARIANT, temp=0.0, lsd_decode_steps=1).cpu().eval()

    builtin_voice = Path("hf/embeddings_v3") / f"{args.voice}.safetensors"
    if not builtin_voice.exists():
        builtin_voice = Path("hf/embeddings_v2") / f"{args.voice}.safetensors"
    native_flow_state = model.get_state_for_audio_prompt(builtin_voice)
    native_flow_state = copy.deepcopy(native_flow_state)
    onnx_flow_state = copy.deepcopy(native_flow_state)

    prepared = model.flow_lm.conditioner.prepare(args.text)
    token_ids_torch = prepared.tokens
    token_ids_np = token_ids_torch.detach().cpu().numpy().astype(np.int64)
    max_gen_len = model._estimate_max_gen_len(token_ids_torch.shape[1])
    current_end = model._flow_lm_current_end(native_flow_state)
    required_len = current_end + token_ids_torch.shape[1] + max_gen_len
    model._expand_kv_cache(native_flow_state, required_len)
    model._expand_kv_cache(onnx_flow_state, required_len)

    text_session = ort.InferenceSession(str(onnx_dir / "text_conditioner.onnx"))
    flow_main_session = ort.InferenceSession(str(onnx_dir / "flow_lm_main.onnx"))
    flow_net_session = ort.InferenceSession(str(onnx_dir / "flow_lm_flow.onnx"))
    mimi_session = ort.InferenceSession(str(onnx_dir / "mimi_decoder.onnx"))

    onnx_text_embeddings = text_session.run(None, {"token_ids": token_ids_np})[0].astype(np.float32)
    with torch.no_grad():
        native_text_embeddings = model.flow_lm.conditioner(prepared).detach().cpu().numpy().astype(np.float32)

    prompt_feeds = {
        "sequence": np.empty((1, 0, model.flow_lm.ldim), dtype=np.float32),
        "text_embeddings": onnx_text_embeddings,
        **_load_flow_state_inputs(flow_main_session, onnx_flow_state),
    }
    onnx_prompt_outputs = flow_main_session.run(None, prompt_feeds)
    with torch.no_grad():
        model._run_flow_lm_and_increment_step(native_flow_state, text_tokens=token_ids_torch)

    state_names = [input_info.name for input_info in flow_main_session.get_inputs() if input_info.name.startswith("state_")]
    for index, state_name in enumerate(state_names):
        layer = index // 2
        kind = "cache" if index % 2 == 0 else "offset"
        tensor = torch.from_numpy(onnx_prompt_outputs[2 + index])
        onnx_flow_state[f"transformer.layers.{layer}.self_attn"][kind] = tensor

    native_mimi_state = init_states(model.mimi, batch_size=1, sequence_length=MIMI_STATIC_SEQ_LEN)
    onnx_mimi_state = copy.deepcopy(native_mimi_state)

    metrics: dict[str, object] = {
        "text_embeddings_max_diff": float(np.max(np.abs(native_text_embeddings - onnx_text_embeddings))),
        "steps": [],
        "first_step_mimi_state": [],
    }

    previous_native = torch.full((1, 1, model.flow_lm.ldim), float("nan"), dtype=model.flow_lm.dtype)
    previous_onnx = np.full((1, 1, model.flow_lm.ldim), np.nan, dtype=np.float32)

    for generation_step in range(args.steps):
        with torch.no_grad():
            native_latent, native_eos = model._run_flow_lm_and_increment_step(
                native_flow_state,
                backbone_input_latents=previous_native,
            )

        main_outputs = flow_main_session.run(
            None,
            {
                "sequence": previous_onnx,
                "text_embeddings": np.empty((1, 0, model.flow_lm.dim), dtype=np.float32),
                **_load_flow_state_inputs(flow_main_session, onnx_flow_state),
            },
        )
        conditioning = main_outputs[0].astype(np.float32)
        eos_logit = float(np.asarray(main_outputs[1]).reshape(-1)[0])
        for index, state_name in enumerate(state_names):
            layer = index // 2
            kind = "cache" if index % 2 == 0 else "offset"
            onnx_flow_state[f"transformer.layers.{layer}.self_attn"][kind] = torch.from_numpy(main_outputs[2 + index])

        flow_dir = flow_net_session.run(
            None,
            {
                "c": conditioning,
                "s": np.asarray([[0.0]], dtype=np.float32),
                "t": np.asarray([[1.0]], dtype=np.float32),
                "x": np.zeros((1, model.flow_lm.ldim), dtype=np.float32),
            },
        )[0].astype(np.float32)
        onnx_latent = flow_dir[:, None, :]

        native_audio = _audio_frame_native(model, native_latent, native_mimi_state)
        onnx_audio_outputs = mimi_session.run(
            None,
            {
                "latent": onnx_latent.astype(np.float32),
                **_load_mimi_state_inputs(mimi_session, onnx_mimi_state),
            },
        )
        onnx_audio = onnx_audio_outputs[0].astype(np.float32)
        mimi_state_names = [input_info.name for input_info in mimi_session.get_inputs() if input_info.name.startswith("state_")]
        flat_mimi_state = flatten_state(onnx_mimi_state)
        for index, state_name in enumerate(mimi_state_names):
            flat_mimi_state[index].copy_(torch.from_numpy(onnx_audio_outputs[1 + index]))

        native_latent_np = native_latent.detach().cpu().numpy().astype(np.float32)
        native_audio_np = native_audio.detach().cpu().numpy().astype(np.float32)
        if generation_step == 0:
            native_flat_mimi_state = flatten_state(native_mimi_state)
            mimi_state_names = [input_info.name for input_info in mimi_session.get_inputs() if input_info.name.startswith("state_")]
            for index, state_name in enumerate(mimi_state_names):
                onnx_state_tensor = np.asarray(onnx_audio_outputs[1 + index])
                native_state_tensor = native_flat_mimi_state[index].detach().cpu().numpy()
                if native_state_tensor.dtype == np.bool_ or onnx_state_tensor.dtype == np.bool_:
                    diff = np.logical_xor(native_state_tensor.astype(bool), onnx_state_tensor.astype(bool)).astype(np.float32)
                else:
                    diff = np.abs(np.nan_to_num(native_state_tensor, nan=0.0) - np.nan_to_num(onnx_state_tensor, nan=0.0))
                metrics["first_step_mimi_state"].append(
                    {
                        "state": state_name,
                        "onnx_nan": int(np.isnan(onnx_state_tensor).sum()),
                        "native_nan": int(np.isnan(native_state_tensor).sum()),
                        "max_diff": float(diff.max()) if diff.size else 0.0,
                        "mae": float(diff.mean()) if diff.size else 0.0,
                    }
                )
        metrics["steps"].append(
            {
                "step": generation_step,
                "native_eos": bool(native_eos.item()),
                "onnx_eos": bool(eos_logit > -4.0),
                "latent_max_diff": float(np.max(np.abs(native_latent_np - onnx_latent))),
                "latent_mae": float(np.mean(np.abs(native_latent_np - onnx_latent))),
                "native_audio_nan": int(np.isnan(native_audio_np).sum()),
                "onnx_audio_nan": int(np.isnan(onnx_audio).sum()),
                "audio_max_diff": float(np.max(np.abs(native_audio_np - onnx_audio))),
                "audio_mae": float(np.mean(np.abs(native_audio_np - onnx_audio))),
            }
        )

        previous_native = native_latent
        previous_onnx = onnx_latent

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())