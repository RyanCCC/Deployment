from openvino.runtime import Core
from openvino.inference_engine import IECore
from openvino.offline_transformations import serialize
ie = Core()
onnx_model_path = r"./yolov5s.onnx"
model_onnx = ie.read_model(model=onnx_model_path)
compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")
serialize(model=model_onnx, model_path="exported_onnx_model.xml", weights_path="exported_onnx_model.bin")