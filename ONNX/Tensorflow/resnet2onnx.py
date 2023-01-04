import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import onnxruntime as rt
import tf2onnx

# 加载图像
img_path = './images/test.jpg'

img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 加载模型
model = ResNet50(weights='imagenet')

preds = model.predict(x)
print('Keras Predicted:', decode_predictions(preds, top=3)[0])
# model.save(os.path.join("./models", model.name))

# Convert to ONNX using the Python Api
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = model.name + ".onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]

# RUN the ONNX model
providers = ['CPUExecutionProvider']
m = rt.InferenceSession(output_path, providers=providers)
onnx_pred = m.run(output_names, {"input": x})

print('ONNX Predicted:', decode_predictions(onnx_pred[0], top=3)[0])


'''
Convert ONNX using the command line
!python -m tf2onnx.convert --opset 13 \
    --saved-model {os.path.join("/models", model.name)} \
    --output  {os.path.join("/models", model.name + ".onnx")}
'''

'''
可参考：https://github.com/onnx/tensorflow-onnx/tree/main/tutorials
'''