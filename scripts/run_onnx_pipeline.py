import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# we reuse helpers from the test and export modules
from onnx_export.export_utils import flatten_state
from pocket_tts import TTSModel
import soundfile as sf


def _encode_tokens(tokenizer_json: Path, text: str) -> np.ndarray:
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    return np.asarray(ids, dtype=np.int64).reshape(1, -1)


def _resolved_shape(shape: list[object]) -> list[int]:
    # match helper from other scripts
    resolved: list[int] = []
    for dim in shape:
        if isinstance(dim, int) and dim >= 0:
            resolved.append(dim)
        else:
            resolved.append(1)
    return resolved


def _build_dummy_input(info) -> np.ndarray:
    # copied from validate script
    type_map = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int16)": np.int16,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    dtype = type_map[info.type]
    shape = _resolved_shape(list(info.shape))
    if np.issubdtype(dtype, np.integer) or dtype == np.bool_:
        return np.zeros(shape, dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full text->audio generation using the split ONNX exports."
    )
    parser.add_argument(
        "--onnx-dir",
        default="hf/onnx",
        help="directory containing exported onnx models",
    )
    parser.add_argument(
        "--tokenizer-json", default="hf/tokenizer.json", help="tokenizer json file"
    )
    parser.add_argument(
        "--text", default="Hello world, this is a test.", help="prompt text to speak"
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="optional audio prompt for voice conditioning (path or HF voice id)",
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="maximum autoregressive steps"
    )
    parser.add_argument(
        "--eos-threshold",
        type=float,
        default=-4.0,
        help="stop generation when eos logit exceeds this value",
    )
    parser.add_argument(
        "--no-eos",
        action="store_true",
        help="ignore EOS logits and run full step count",
    )
    parser.add_argument(
        "--use-onnx-decoder",
        action="store_true",
        help="use onnx mimi_decoder model instead of PyTorch for audio decoding. "
        "Note: ONNX decoder has a known issue where audio gets quieter over time.",
    )
    parser.add_argument(
        "--output-wav", default="onnx_full.wav", help="output wav filename"
    )

    args = parser.parse_args()
    if args.use_onnx_decoder:
        print(
            "WARNING: ONNX decoder has a known issue where audio gets progressively quieter."
        )
        print("         For production use, prefer PyTorch decoder (default).")
    print(f"decoder mode: {'ONNX' if args.use_onnx_decoder else 'PyTorch'}")
    onnx_dir = Path(args.onnx_dir)
    tokenizer_json = Path(args.tokenizer_json)

    if not tokenizer_json.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_json}")
    if not onnx_dir.exists():
        raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}")

    # tokenize input text
    token_ids = _encode_tokens(tokenizer_json, args.text)

    # load pyro model once; we need it for decoding
    tts = TTSModel.load_model()

    # prepare flow state: if a voice is given we condition, otherwise start from clean NaN caches
    flow_state_feeds: dict[str, np.ndarray] = {}
    if args.voice:
        voice_state = tts.get_state_for_audio_prompt(args.voice)
        # the returned object is already the flow_lm state dict
        # make sure caches are large enough for onnx export (1000)
        tts._expand_kv_cache(voice_state, sequence_length=1000)
        flow_list = flatten_state(voice_state)
        for idx, tensor in enumerate(flow_list):
            flow_state_feeds[f"state_{idx}"] = tensor.cpu().numpy()
    else:
        # start from default model state (NaNs) so generation behaves correctly
        from pocket_tts.modules.stateful_module import init_states

        init_state = init_states(tts.flow_lm, batch_size=1, sequence_length=1000)
        for idx, tensor in enumerate(flatten_state(init_state)):
            flow_state_feeds[f"state_{idx}"] = tensor.cpu().numpy()

    # load sessions
    conditioner = ort.InferenceSession(str(onnx_dir / "text_conditioner.onnx"))
    flow_main = ort.InferenceSession(str(onnx_dir / "flow_lm_main.onnx"))
    flow_flow = ort.InferenceSession(str(onnx_dir / "flow_lm_flow.onnx"))

    # prepare optional ONNX Mimi decoder
    onnx_decoder = None
    mimi_state_feeds: dict[str, np.ndarray] = {}
    if args.use_onnx_decoder:
        onnx_decoder = ort.InferenceSession(str(onnx_dir / "mimi_decoder.onnx"))
        # initialize Mimi state with the same sequence length used during export.
        # the export script used a STATIC_SEQ_LEN of 1000 tokens, so we reuse that
        # regardless of the actual number of steps requested.  This avoids
        # mismatches between the ONNX model's fixed input shape and our feeds.
        from pocket_tts.modules.stateful_module import init_states

        mimi_steps_per_latent = int(tts.mimi.encoder_frame_rate / tts.mimi.frame_rate)
        # use export static length
        mimi_sequence_length = 1000
        # if we want to be fancy we could inspect onnx_decoder inputs for state_* shape
        mimi_state = init_states(
            tts.mimi, batch_size=1, sequence_length=mimi_sequence_length
        )
        for idx, tensor in enumerate(flatten_state(mimi_state)):
            mimi_state_feeds[f"state_{idx}"] = tensor.cpu().numpy()

    # run text conditioning
    text_embeddings = conditioner.run(None, {"token_ids": token_ids})[0].astype(
        np.float32
    )

    # helper to populate missing flow_main inputs with zeros
    def build_flow_main_feeds(
        overrides: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        feeds: dict[str, np.ndarray] = {}
        ov = overrides or {}
        for inp in flow_main.get_inputs():
            if inp.name in ov:
                feeds[inp.name] = ov[inp.name]
            elif inp.name == "sequence":
                # placeholder; caller should override
                feeds[inp.name] = np.full((1, 1, 32), np.nan, dtype=np.float32)
            elif inp.name == "text_embeddings":
                # caller should override
                feeds[inp.name] = np.zeros(
                    (1, 0, text_embeddings.shape[-1]), dtype=np.float32
                )
            else:
                feeds[inp.name] = _build_dummy_input(inp)
        # merge any provided flow_state
        for k, v in flow_state_feeds.items():
            feeds[k] = v
        return feeds

    # run prompt step (sequence blank, text_embeddings full)
    prompt_feeds = build_flow_main_feeds(
        overrides={
            "sequence": np.full((1, 1, 32), np.nan, dtype=np.float32),
            "text_embeddings": text_embeddings,
        }
    )
    prompt_outs = flow_main.run(None, prompt_feeds)
    conditioning = prompt_outs[0].astype(np.float32)
    eos_logit = prompt_outs[1].astype(np.float32)
    # update flow_state_feeds with out_state_*
    for out_info, value in zip(flow_main.get_outputs()[2:], prompt_outs[2:]):
        # mapping out_state_i -> state_i
        name = out_info.name.replace("out_", "")
        flow_state_feeds[name] = value

    # empty text embeddings for subsequent steps
    empty_text_embeddings = np.zeros(
        (1, 0, text_embeddings.shape[-1]), dtype=np.float32
    )

    # prepare PyTorch Mimi state for decoding
    from pocket_tts.modules.stateful_module import init_states, increment_steps

    mimi_steps_per_latent = int(tts.mimi.encoder_frame_rate / tts.mimi.frame_rate)
    mimi_sequence_length = args.steps * mimi_steps_per_latent
    mimi_state = init_states(
        tts.mimi, batch_size=1, sequence_length=mimi_sequence_length
    )

    audio_frames: list[np.ndarray] = []
    current_latent = np.full((1, 1, 32), np.nan, dtype=np.float32)
    for step in range(args.steps):
        # run flow_main step with last latent
        feeds = build_flow_main_feeds(
            overrides={
                "sequence": current_latent,
                "text_embeddings": empty_text_embeddings,
            }
        )
        outs = flow_main.run(None, feeds)
        conditioning = outs[0].astype(np.float32)
        eos_logit = outs[1].astype(np.float32)
        # update state
        for out_info, val in zip(flow_main.get_outputs()[2:], outs[2:]):
            flow_state_feeds[out_info.name.replace("out_", "")] = val

        # sample latent via flow_flow
        noise = np.random.randn(1, 32).astype(np.float32)
        flow_dir = flow_flow.run(
            None,
            {
                "c": conditioning,
                "s": np.array([[0.0]], dtype=np.float32),
                "t": np.array([[1.0]], dtype=np.float32),
                "x": noise,
            },
        )[0]
        # adapter: unsqueeze axis=1
        next_latent = flow_dir[:, None, :]

        # decode with Mimi (ONNX or PyTorch)
        if args.use_onnx_decoder and onnx_decoder is not None:
            # prepare feeds for ONNX decoder
            onnx_feeds: dict[str, np.ndarray] = {"latent": next_latent}
            onnx_feeds.update(mimi_state_feeds)
            onnx_outs = onnx_decoder.run(None, onnx_feeds)
            audio_frame = onnx_outs[0]
            # update state feeds
            for idx, val in enumerate(onnx_outs[1:]):
                mimi_state_feeds[f"state_{idx}"] = val
            # ensure 1‑D float array like PyTorch path
            audio_frame = np.asarray(audio_frame).reshape(-1)
        else:
            # convert latent to torch tensor so we can use model parameters
            import torch

            latent_t = torch.from_numpy(next_latent)
            mimi_input = latent_t * tts.flow_lm.emb_std + tts.flow_lm.emb_mean
            # swap last two dimensions (B,1,32) -> (B,32,1)
            transposed = mimi_input.transpose(-1, -2)
            quantized = tts.mimi.quantizer(transposed)
            audio_frame = tts.mimi.decode_from_latent(quantized, mimi_state)
            increment_steps(tts.mimi, mimi_state, increment=mimi_steps_per_latent)
            # audio_frame shape [B,1,1,1920]
            audio_frame = audio_frame.reshape(-1).cpu().detach().numpy()

        audio_frames.append(audio_frame)

        current_latent = next_latent

        # break on EOS unless user disabled it
        if not args.no_eos:
            if eos_logit.size and float(eos_logit.reshape(-1)[0]) > args.eos_threshold:
                break

    print(
        f"generated {len(audio_frames)} frames, total samples={sum(f.size for f in audio_frames)}"
    )

    if audio_frames:
        audio_concat = np.concatenate(audio_frames)
        sf.write(args.output_wav, audio_concat, 24000)
        print(f"wrote {args.output_wav}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
