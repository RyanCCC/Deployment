import tf2onnx
import onnxruntime as rt
import tensorflow as tf


yolox = tf.keras.models.load_model('./models/yolox_model')

# 加载图像

print('load model')
