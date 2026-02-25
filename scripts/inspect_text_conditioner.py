from tokenizers import Tokenizer
import numpy as np
import onnxruntime as ort

# load tokenizer
tokenizer = Tokenizer.from_file('hf/tokenizer.json')
text = "Hello world this is a test"
ids = tokenizer.encode(text, add_special_tokens=False).ids
token_ids = np.asarray(ids, dtype=np.int64).reshape(1, -1)

sess = ort.InferenceSession('hf/onnx/text_conditioner.onnx')
out = sess.run(None, {'token_ids': token_ids})[0]
print('text_embeddings shape', out.shape)
print('stats', out.min(), out.max(), out.mean(), out.std())
print('some values', out.flatten()[:10])
