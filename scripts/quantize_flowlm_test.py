import torch
from pocket_tts.model import TTSModel

print('loading model')
tts = TTSModel.load_model().cpu().eval()
print('quantizing flow_lm only')
tts.flow_lm = torch.quantization.quantize_dynamic(tts.flow_lm, {torch.nn.Linear}, dtype=torch.qint8)
print('done quantizing, attempting init_states')
from pocket_tts.modules.stateful_module import init_states
state = init_states(tts.flow_lm, batch_size=1, sequence_length=100)
print('init_states successful, state keys', state.keys())
