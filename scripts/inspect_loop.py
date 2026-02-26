import onnx
import sys

path = sys.argv[1]
model = onnx.load(path)
ops = set(n.op_type for n in model.graph.node)
print('op types present:', sorted(ops))

for node in model.graph.node:
    if node.op_type in ('Loop','Scan','If','While'):
        print('Found control flow node', node.op_type, node.name)
        for attr in node.attribute:
            if attr.name == 'body':
                body = attr.g
                print(' body inputs', [inp.name for inp in body.input])
                print(' body initializers', [init.name for init in body.initializer])
                print(' body nodes count', len(body.node))

print('Done')
