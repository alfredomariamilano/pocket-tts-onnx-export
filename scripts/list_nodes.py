import onnx, sys

if len(sys.argv) < 3:
    print("Usage: python list_nodes.py <model.onnx> <substring> [--unused-init]")
    sys.exit(1)

path = sys.argv[1]
substr = sys.argv[2]
show_unused = '--unused-init' in sys.argv
model = onnx.load(path)
if show_unused:
    used = set(i for n in model.graph.node for i in n.input)
    print('unused initializers:')
    for init in model.graph.initializer:
        if init.name not in used:
            print(' ', init.name, init.dims)
else:
    for n in model.graph.node:
        if substr in n.name or any(substr in o for o in n.output) or any(substr in i for i in n.input):
            print(f"{n.op_type} {n.name} inputs={list(n.input)} outputs={list(n.output)}")
