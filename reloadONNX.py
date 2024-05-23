'''重新加载onnx文件，让其能显示tensor形状'''
import onnx
from onnx import shape_inference
model = "onnx/2.onnx"
onnx.save(onnx.shape_inference.infer_shapes(onnx.load(model)),model)