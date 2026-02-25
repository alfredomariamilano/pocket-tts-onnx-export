import numpy as np
import soundfile as sf

d1, sr1 = sf.read('onnx_full.wav')
d2, sr2 = sf.read('reference.wav')
print('sr', sr1, sr2)
# make lengths equal
minlen = min(len(d1), len(d2))
d1 = d1[:minlen]
d2 = d2[:minlen]
print('shape', d1.shape)
print('MAE', np.mean(np.abs(d1-d2)))
print('RMSE', np.sqrt(np.mean((d1-d2)**2)))
print('d1 stats', d1.min(), d1.max(), d1.mean(), d1.std())
print('d2 stats', d2.min(), d2.max(), d2.mean(), d2.std())
