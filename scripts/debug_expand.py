import onnx

model = onnx.load('hf/onnx/monolith_step.onnx')
for node in model.graph.node:
    if node.op_type == 'Expand' and 'self_attn' in node.name:
        print('Found Expand node', node.name)
        print(' Inputs:', node.input)
        for attr in node.attribute:
            print(' attr', attr.name, onnx.helper.printable_attribute_value(attr))
        break
else:
    print('no matching Expand node')
