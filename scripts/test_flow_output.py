import onnxruntime as ort
import numpy as np
sess=ort.InferenceSession('hf/onnx/flow_lm_flow.onnx')
c=np.zeros((1,1024),dtype=np.float32)
s=np.array([[0.0]],dtype=np.float32)
t=np.array([[1.0]],dtype=np.float32)
x=np.random.randn(1,32).astype(np.float32)
out=sess.run(None,{'c':c,'s':s,'t':t,'x':x})
print('flow out shape', out[0].shape, 'stats', out[0].min(),out[0].max(),out[0].mean(),out[0].std())
