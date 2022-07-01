from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np

model_xml = '../YOLOV5/yolov5s.xml'
model_bin = '../YOLOV5/yolov5s.bin'
image = './images/giraffe.jpg'
ie = IECore()
net = IENetwork(model=model_xml, weights=model_bin)
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
batch,channel,height,width  = net.inputs[input_blob].shape
image = cv2.imread(image)
image = cv2.resize(image, (width, height))
image = image.transpose((2, 0, 1))
exec_net = ie.load_network(network=net, device_name='CPU')
res = exec_net.infer(inputs={input_blob: image})


print('finish')