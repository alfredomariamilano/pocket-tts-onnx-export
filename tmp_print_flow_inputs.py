import onnxruntime as ort
import sys

path = sys.argv[1]
sess = ort.InferenceSession(path)
print('Inputs:')
for inp in sess.get_inputs():
    print(inp.name, inp.type, inp.shape)
print('Outputs:')
for out in sess.get_outputs():
    print(out.name, out.type, out.shape)
