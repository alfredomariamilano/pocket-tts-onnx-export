import numpy as np
import onnxruntime as ort
from pocket_tts import TTSModel

# helper from run_onnx_pipeline

def _resolved_shape(shape: list[object]) -> list[int]:
    resolved: list[int] = []
    for dim in shape:
        if isinstance(dim, int) and dim >= 0:
            resolved.append(dim)
        else:
            resolved.append(1)
    return resolved

def _build_dummy_input(info) -> np.ndarray:
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

# generate conditioning using ONNX text_conditioner + flow_lm_main
ort_sess_cond = ort.InferenceSession('hf/onnx/text_conditioner.onnx')
ort_sess_main = ort.InferenceSession('hf/onnx/flow_lm_main.onnx')

# pick real text tokens for comparison
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file('hf/tokenizer.json')
text = "Hello world this is a test"
ids = tokenizer.encode(text, add_special_tokens=False).ids
token_ids = np.asarray(ids, dtype=np.int64).reshape(1, -1)

# also prepare PyTorch model for comparison
model = TTSModel.load_model()
import torch

# compute PyTorch text embeddings
pt_token_tensor = torch.from_numpy(token_ids)
pt_text_embeddings = model.flow_lm.conditioner(pt_token_tensor).detach().numpy()
print('pt_text_embeddings stats', pt_text_embeddings.min(), pt_text_embeddings.max(), pt_text_embeddings.mean(), pt_text_embeddings.std())

# build dummy state dictionary for onnx
state_dict = {}
for inp in ort_sess_main.get_inputs():
    if inp.name.startswith('state_'):
        state_dict[inp.name] = _build_dummy_input(inp)

def run_main(conditioning):
    feeds = {}
    for inp in ort_sess_main.get_inputs():
        if inp.name == 'sequence':
            feeds['sequence'] = np.full((1,1,32), np.nan, dtype=np.float32)
        elif inp.name == 'text_embeddings':
            feeds['text_embeddings'] = conditioning
        elif inp.name.startswith('state_'):
            feeds[inp.name] = state_dict[inp.name]
        else:
            feeds[inp.name] = np.zeros(tuple(_resolved_shape(inp.shape)), dtype=np.float32)
    outs = ort_sess_main.run(None, feeds)
    return outs[0].astype(np.float32), outs[1].astype(np.float32)

conditioning, eos = run_main(ort_sess_cond.run(None, {'token_ids': token_ids})[0].astype(np.float32))
print('onnx conditioning stats', conditioning.min(),conditioning.max(),conditioning.mean(),conditioning.std(), 'eos', eos)
print('pt conditioning stats', pt_text_embeddings.min(), pt_text_embeddings.max(), pt_text_embeddings.mean(), pt_text_embeddings.std())

# flow module comparison
noise = np.random.randn(1,32).astype(np.float32)
# onnx flow
ort_sess_flow = ort.InferenceSession('hf/onnx/flow_lm_flow.onnx')
flow_out = ort_sess_flow.run(None, {'c': conditioning, 's': np.array([[0.0]],dtype=np.float32), 't': np.array([[1.0]],dtype=np.float32), 'x': noise})[0]
print('onnx flow stats', flow_out.min(),flow_out.max(),flow_out.mean(),flow_out.std())

# pytorch flow
model = TTSModel.load_model()
# ensure same device cpu
torch_flow = model.flow_lm.flow_net
import torch
c_t = torch.from_numpy(conditioning)
s_t = torch.from_numpy(np.array([[0.0]],dtype=np.float32))
t_t = torch.from_numpy(np.array([[1.0]],dtype=np.float32))
x_t = torch.from_numpy(noise)
pt_out = torch_flow(c_t, s_t, t_t, x_t).detach().numpy()
print('pt flow stats', pt_out.min(),pt_out.max(),pt_out.mean(),pt_out.std())
