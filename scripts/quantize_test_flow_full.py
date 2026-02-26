from pathlib import Path
from scripts.quantize import quantize_file_int8, quantize_file_q4_weight_only

input_path = Path('hf/onnx/flow_lm_full.onnx')
int8_path = Path('hf/onnx_temp/flow_lm_full_int8.onnx')
q4_path = Path('hf/onnx_temp/flow_lm_full_q4.onnx')

print('--- INT8 quantization ---')
quantize_file_int8(input_path, int8_path)
print('--- Q4 quantization ---')
quantize_file_q4_weight_only(input_path, q4_path)
print('done')
