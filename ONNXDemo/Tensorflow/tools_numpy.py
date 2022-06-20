from PIL import Image
import numpy as np
# import tensorflow as tf
from nms import non_max_suppression

def sigmoid(x):  
    return np.exp(-np.logaddexp(0, -x))

def yolo_correct_boxes(box_xy, box_wh, image_shape, input_shape, letterbox_image=True):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape, np.float64)
    image_shape = np.array(image_shape, np.float64)

    if letterbox_image:
        new_shape = np.round(image_shape * np.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=1)
    return boxes

def resize_image(image, input_shape = [640,640], letterbox_image=True):
    size = input_shape
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

def preprocess_input(image):
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image

def prediction(model, image_data):
    out_boxes, out_scores, out_classes = model([image_data], training=False)
    return out_boxes, out_scores, out_classes

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def DecodeBox(outputs,image_shape, input_shape, class_names,confidence=0.7, max_boxes=100):
    num_classes = len(class_names)
    outputs = outputs[:-1]
    batch_size = np.shape(outputs[0])[0]
    grids = []
    strides = []
    hw = [np.shape(x)[1:3] for x in outputs]
    outputs = np.concatenate([np.reshape(x, [batch_size, -1, 5 + num_classes]) for x in outputs], axis = 1)
    for i in range(len(hw)):
        grid_x, grid_y  = np.meshgrid(np.arange(hw[i][1]), np.arange(hw[i][0]))
        grid = np.reshape(np.stack((grid_x, grid_y), 2), (1, -1, 2))
        shape  = np.shape(grid)[:2]
        grids.append(np.array(grid, np.float32))
        strides.append(np.ones((shape[0], shape[1], 1)) * input_shape[0] / np.array(hw[i][0], np.float32))
    grids = np.concatenate(grids, axis=1)
    strides = np.concatenate(strides, axis=1)
    box_xy = (outputs[..., :2] + grids) * strides / np.array(input_shape[::-1], np.float32)
    box_wh = np.exp(outputs[..., 2:4]) * strides / np.array(input_shape[::-1], np.float32)
    box_confidence  = sigmoid(outputs[..., 4:5])
    box_class_probs = sigmoid(outputs[..., 5: ])
    boxes = yolo_correct_boxes(box_xy, box_wh, image_shape, input_shape)
    box_scores  = box_confidence * box_class_probs

    mask = box_scores >= confidence
    max_boxes_tensor = max_boxes
    boxes_out   = np.empty(shape=[0, 4])
    scores_out  = np.array([])
    classes_out = np.array([], dtype=np.int32)
    for c in range(num_classes):
        class_boxes =np.array(boxes[mask[..., c]])
        class_box_scores = np.array(box_scores[..., c][mask[..., c]])
        '''
        TODO: IMPLEMENT NON_MAX_SUPPRESSION BY NUMPY
        '''
        # nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=0.5)
        # nms_index = nms_index.numpy
        nms_index = non_max_suppression(class_boxes, class_box_scores, threshold=0.5)

        if len(class_boxes) == 0:
            continue
        class_boxes = class_boxes[nms_index]
        class_box_scores = class_box_scores[nms_index]
        classes = np.ones_like(class_box_scores, 'int32') * c

        boxes_out = np.append(boxes_out, class_boxes, axis=0)
        scores_out = np.append(scores_out, class_box_scores)
        classes_out = np.append(classes_out, classes)


    return boxes_out, scores_out, classes_out