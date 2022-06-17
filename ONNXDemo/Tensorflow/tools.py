from PIL import Image
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf

def yolo_correct_boxes(box_xy, box_wh, image_shape, input_shape, letterbox_image=True):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    if letterbox_image:
        new_shape = K.round(image_shape * K.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = K.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]])
    boxes *= K.concatenate([image_shape, image_shape])
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

def DecodeBox(outputs,image_shape, input_shape, class_names,confidence=0.7, max_boxes=100, ):
    num_classes = len(class_names)
    outputs = outputs[:-1]
    batch_size = K.shape(outputs[0])[0]
    grids = []
    strides = []
    hw = [K.shape(x)[1:3] for x in outputs]
    outputs = tf.concat([tf.reshape(x, [batch_size, -1, 5 + num_classes]) for x in outputs], axis = 1)
    for i in range(len(hw)):
        grid_x, grid_y  = tf.meshgrid(tf.range(hw[i][1]), tf.range(hw[i][0]))
        grid = tf.reshape(tf.stack((grid_x, grid_y), 2), (1, -1, 2))
        shape  = tf.shape(grid)[:2]
        grids.append(tf.cast(grid, K.dtype(outputs)))
        strides.append(tf.ones((shape[0], shape[1], 1)) * input_shape[0] / tf.cast(hw[i][0], K.dtype(outputs)))
    grids = tf.concat(grids, axis=1)
    strides = tf.concat(strides, axis=1)
    box_xy = (outputs[..., :2] + grids) * strides / K.cast(input_shape[::-1], K.dtype(outputs))
    box_wh = tf.exp(outputs[..., 2:4]) * strides / K.cast(input_shape[::-1], K.dtype(outputs))
    box_confidence  = K.sigmoid(outputs[..., 4:5])
    box_class_probs = K.sigmoid(outputs[..., 5: ])
    boxes = yolo_correct_boxes(box_xy, box_wh, image_shape, input_shape)
    box_scores  = box_confidence * box_class_probs

    mask = box_scores >= confidence
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_out   = []
    scores_out  = []
    classes_out = []
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[..., c])
        class_box_scores = tf.boolean_mask(box_scores[..., c], mask[..., c])
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=0.5)

        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c

        boxes_out.append(class_boxes)
        scores_out.append(class_box_scores)
        classes_out.append(classes)
    boxes_out = K.concatenate(boxes_out, axis=0)
    scores_out = K.concatenate(scores_out, axis=0)
    classes_out = K.concatenate(classes_out, axis=0)

    return boxes_out, scores_out, classes_out