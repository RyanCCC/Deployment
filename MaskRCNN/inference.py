from PIL import Image
import tensorflow as tf
import numpy as np
from utils.anchors import get_anchors
from utils.utils import mold_inputs,unmold_detections
from utils import visualize
from utils.config import Config
import os
from glob import glob
import json
import base64
import tf2onnx
from tqdm import tqdm
import onnxruntime as ort

img_pattern  = './samples/*.jpg'
model_path = './models/building_model'
class_path = './data/building.names'

class intEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.intc):
            return float(obj)
    
        return json.JSONEncoder.default(self, obj)

def get_class(classes_path):
        classes_path = os.path.expanduser(classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        class_names.insert(0,"BG")
        return class_names

def img2base64(image_path):
    with open(image_path, 'rb') as f:
        image = f.read()
    image_base64 = str(base64.b64encode(image), encoding='utf-8')
    return image_base64 

class_names = get_class(class_path)
def get_config():
    class InferenceConfig(Config):
        NUM_CLASSES = len(class_names)
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.8
        NAME = "Customer"
        RPN_ANCHOR_SCALES =  (16, 32, 64, 128, 256)
        IMAGE_MIN_DIM = 512
        IMAGE_MAX_DIM = 512
        IMAGE_SHAPE =  [512, 512 ,3]

    config = InferenceConfig()
    return config

InferenceConfig = get_config()
model = tf.keras.models.load_model(model_path)
model.summary()

# 将模型导出为onnx
output_path = os.path.join('./models/mask_rcnn_13.onnx')
# model_proto, _ = tf2onnx.convert.from_keras(model, opset=13, output_path=output_path)
# output_names = [n.name for n in model_proto.graph.output]
# print(output_names)

# load onnx model 
outputs_names = ['mrcnn_detection', 'mrcnn_class', 'mrcnn_bbox', 'mrcnn_mask', 'ROI', 'rpn_class', 'rpn_bbox']
onnx_model = ort.InferenceSession(output_path)



for image_name in tqdm(glob(img_pattern)):
    image = Image.open(image_name).convert('RGB')
    imageWidth, imageHeight = image.size
    image = [np.array(image)]
    molded_images, image_metas, windows = mold_inputs(InferenceConfig,image)
    image_shape = molded_images[0].shape
    anchors = get_anchors(InferenceConfig,image_shape)
    anchors = np.broadcast_to(anchors, (1,) + anchors.shape)
    
    detections, _, _, mrcnn_mask, _, _, _ =onnx_model.run(outputs_names, {"input_image":molded_images.astype(np.float32), "input_image_meta":image_metas.astype(np.float32), "input_anchors":anchors.astype(np.float32)})
    
    detections, _, _, mrcnn_mask, _, _, _ =model.predict([molded_images, image_metas, anchors], verbose=0)
    final_rois, final_class_ids, final_scores, final_masks =unmold_detections(detections[0], mrcnn_mask[0],image[0].shape, molded_images[0].shape,windows[0])

    r = {
        "rois": final_rois,
        "class_ids": final_class_ids,
        "scores": final_scores,
        "masks": final_masks,
    }


    drawed_image, shapes = visualize.display_instances(image[0], r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'])

    drawed_image.show()