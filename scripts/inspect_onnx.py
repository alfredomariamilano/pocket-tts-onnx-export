import sys
import onnxruntime as ort

if len(sys.argv) != 2:
    print("usage: python inspect_onnx.py <model.onnx>")
    sys.exit(1)
path = sys.argv[1]
sess = ort.InferenceSession(path)
print("Model", path)
print("Inputs:")
for inp in sess.get_inputs():
    print(inp.name, inp.type, inp.shape)
print("Outputs:")
for out in sess.get_outputs():
    print(out.name, out.type, out.shape)
