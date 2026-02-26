from pathlib import Path
import scripts.quantize as q

inp = Path('hf/onnx/flow_lm_flow.onnx')
out = Path('hf/onnx_temp/flow_lm_flow_q4.onnx')
print('sizes before', inp.stat().st_size)
q.quantize_file_q4_weight_only(inp, out)
print('exists', out.exists(), 'size', out.stat().st_size if out.exists() else 'NA')
