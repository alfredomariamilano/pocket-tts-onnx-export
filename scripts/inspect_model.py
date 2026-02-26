import onnx
import sys

path = sys.argv[1]
print('loading', path)
model = onnx.load(path)
print('nodes:', len(model.graph.node))
print('initializers:', len(model.graph.initializer))
print('first 20 ops:')
for n in model.graph.node[:20]:
    print(' ', n.op_type, n.name)

# show any quantization-related op types
ops = sorted(set(n.op_type for n in model.graph.node))
print('\nquant-related ops:')
for o in ops:
    if any(k in o.lower() for k in ('mat', 'quant', 'dequant', 'qlinear')):
        print(' ', o)
print('\nfirst few graph inputs:')
for inp in model.graph.input[:20]:
    print(' ', inp.name)

# analyze some MatMul nodes to see which input is constant
init_names = {init.name for init in model.graph.initializer}
# report biggest initializers which represent the weight tensors we expect to
# quantize; if these are unhandled ops we'll know which ones to focus on.
def _tensor_size(init):
    prod = 1
    for d in init.dims:
        prod *= d
    return prod
print('\nlargest initializers:')
for init in sorted(model.graph.initializer, key=_tensor_size, reverse=True)[:10]:
    print('  ', init.name, 'shape', list(init.dims), 'dtype', onnx.TensorProto.DataType.Name(init.data_type))

# count how many Gemm vs MatMul nodes
matmuls = sum(1 for n in model.graph.node if n.op_type == 'MatMul')
gemms = sum(1 for n in model.graph.node if n.op_type == 'Gemm')
print(f"\nnode counts: MatMul={matmuls}, Gemm={gemms}")
print('\nSample MatMul nodes with input const status:')
count = 0
for n in model.graph.node:
    if n.op_type == 'MatMul' and count < 15:
        print(' -', n.name, 'inputs', list(n.input))
        for inp in n.input:
            print('    ', inp, 'initial? ', inp in init_names)
        count += 1
# show a Gemm node example
gemm_shown = False
for n in model.graph.node:
    if n.op_type == 'Gemm':
        print('\nexample Gemm', n.name, 'inputs', list(n.input))
        for inp in n.input:
            print('    ', inp, 'initial? ', inp in init_names)
        break
