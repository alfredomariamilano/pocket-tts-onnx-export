import os
import sys
from pathlib import Path

# make sure project root is on python path so we can import scripts modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import onnxruntime as ort

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.default_parameters import DEFAULT_VARIANT
from pocket_tts.modules.stateful_module import init_states
from onnx_export.export_utils import flatten_state, get_state_structure
from onnx_export.wrappers import MimiWrapper

# reuse export helper only when needed (importing triggers patches)
# the module will be imported dynamically inside the test


def test_mimi_decoder_parity(tmp_path):
    """Verify that the ONNX mimi_decoder produces identical output to PyTorch.

    This test exports a fresh ONNX model into a temporary directory, then runs a
    single random latent through both the PyTorch wrapper and the ONNX runtime,
    comparing audio frames and all internal state tensors.
    """

    # load reference PyTorch model
    tts = TTSModel.load_model(DEFAULT_VARIANT)
    tts.eval()

    # build a small initial state
    # use a reasonably large sequence_length to avoid cache sizing issues
    mimi_state = init_states(tts.mimi, batch_size=1, sequence_length=1000)
    flat_state = flatten_state(mimi_state)
    state_structure = get_state_structure(mimi_state)

    # locate or generate ONNX decoder. If hf/onnx already contains a model,
    # just use it; otherwise attempt an export but patch rope_offset to avoid
    # beartype errors during tracing.
    mimi_path = Path("hf/onnx/mimi_decoder.onnx")
    if not mimi_path.exists():
        # import export helper now (triggers several monkeypatches)
        from scripts import export_mimi_and_conditioner

        # remember original functions so we can undo patch later
        import pocket_tts.modules.transformer as transformer_module
        orig_complete_kv = transformer_module.complete_kv
        # patch rope_offset to accept tensors (remove beartype wrapper)
        if hasattr(transformer_module._LinearKVCacheBackend.rope_offset, "__wrapped__"):
            transformer_module._LinearKVCacheBackend.rope_offset = (
                transformer_module._LinearKVCacheBackend.rope_offset.__wrapped__
            )

        flow_path, mimi_path, _ = export_mimi_and_conditioner.export_models(output_dir=str(tmp_path))
        assert mimi_path is not None and os.path.exists(mimi_path)

        # restore transformer behaviour
        transformer_module.complete_kv = orig_complete_kv

    # now we have mimi_path pointing to a usable onnx file

    # create the PyTorch wrapper and run one step
    wrapper = MimiWrapper(tts.mimi, state_structure, emb_std=tts.flow_lm.emb_std, emb_mean=tts.flow_lm.emb_mean)

    latent = torch.randn(1, 1, tts.flow_lm.ldim)
    with torch.no_grad():
        pt_out = wrapper(latent, flat_state)

    pt_audio = pt_out[0].numpy()
    pt_states = [s.numpy() for s in pt_out[1:]]

    sess = ort.InferenceSession(mimi_path)

    # prepare ONNX inputs
    onnx_inputs = {"latent": latent.numpy()}
    for idx, tensor in enumerate(flat_state):
        onnx_inputs[f"state_{idx}"] = tensor.numpy()

    onnx_outs = sess.run(None, onnx_inputs)
    onnx_audio = onnx_outs[0]
    onnx_states = onnx_outs[1:]

    # compare outputs
    np.testing.assert_allclose(pt_audio, onnx_audio, rtol=1e-4, atol=1e-4)
    assert len(pt_states) == len(onnx_states)
    for pt_s, onnx_s in zip(pt_states, onnx_states):
        np.testing.assert_allclose(pt_s, onnx_s, rtol=1e-4, atol=1e-4)

    # if we reach here, the decoder is numerically equivalent for a single step
