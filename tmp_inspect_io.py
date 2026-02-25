import onnxruntime as ort
s=ort.InferenceSession('hf/onnx/monolith_step.onnx')
print('INPUTS')
for i,x in enumerate(s.get_inputs()):
    print(i,x.name,x.type,x.shape)
print('OUTPUTS')
for i,x in enumerate(s.get_outputs()):
    print(i,x.name,x.type,x.shape)
