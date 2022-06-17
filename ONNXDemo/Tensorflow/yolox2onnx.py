import tf2onnx
import onnxruntime as rt
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tools import * 
import colorsys


image = './images/test.jpg'
input_shape = [640,640]

yolox = tf.keras.models.load_model('./models/yolox_model')

# 加载图像
image = Image.open(image)
image = cvtColor(image)
image_data = resize_image(image)
image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
outputs = prediction(yolox, image_data) 

# Decode outputs
classes_path='./village.names'
class_names = get_classes(classes_path)
font = ImageFont.truetype(font='./simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
thickness = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))
out_boxes, out_scores, out_classes = DecodeBox(outputs,input_image_shape, input_shape, class_names)

# 在图像上画图
num_classes = len(class_names)
hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
for i, c in list(enumerate(out_classes)):
    predicted_class = class_names[int(c)]
    box = out_boxes[i]
    score = out_scores[i]
    top, left, bottom, right = box

    top = max(0, np.floor(top).astype('int32'))
    left = max(0, np.floor(left).astype('int32'))
    bottom = min(image.size[1], np.floor(bottom).astype('int32'))
    right = min(image.size[0], np.floor(right).astype('int32'))
    label = '{} {:.2f}'.format(predicted_class, score)
    draw = ImageDraw.Draw(image)
    label_size = draw.textsize(label, font)
    label = label.encode('utf-8')
    print(label, top, left, bottom, right)
            
    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])

    for i in range(thickness):
        draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
    draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
    del draw
image.show()
print('finish')