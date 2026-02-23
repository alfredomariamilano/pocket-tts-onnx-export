import safetensors.torch, torch
from huggingface_hub import hf_hub_download

repo = 'kyutai/pocket-tts'
path = hf_hub_download(repo_id=repo, filename='embeddings/alba.safetensors')
print('downloaded to', path)
loaded = safetensors.torch.load_file(path)
print('keys', loaded.keys())
tens = loaded['audio_prompt']
print('shape', tens.shape, 'dtype', tens.dtype)
