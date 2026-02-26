import onnx

m = onnx.load('hf/onnx/flow_lm_full.onnx')
used = set(i for n in m.graph.node for i in n.input)
print('total initializers', len(m.graph.initializer))
for init in m.graph.initializer:
    size = 1
    for d in init.dims:
        size *= d
    users = [n.name for n in m.graph.node if init.name in n.input]
    if users:
        print('init', init.name, 'shape', init.dims, 'elements', size, 'used by', users[:3])
    else:
        print('init', init.name, 'shape', init.dims, 'elements', size, 'unused')
