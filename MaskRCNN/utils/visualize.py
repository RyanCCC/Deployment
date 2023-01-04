import os
import sys
import random
import itertools
import colorsys
import numpy as np

from skimage.measure import find_contours
from PIL import Image
import cv2
ROOT_DIR = os.path.abspath("../")

sys.path.append(ROOT_DIR)

def random_colors(N, bright=True):
    """
    生成随机颜色
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """
    打上mask图标
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), 
                      show_mask=True, show_bbox=False,
                      colors=None, captions=None):
    # instance的数量
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    colors = colors or random_colors(N)

    # 当masked_image为原图时是在原图上绘制
    # 如果不想在原图上绘制，可以把masked_image设置成等大小的全0矩阵
    masked_image = np.array(image,np.uint8)
    shapes = []
    for i in range(N):
        color = colors[i]

        # 该部分用于显示bbox
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            cv2.rectangle(masked_image, (x1, y1), (x2, y2), (color[0] * 255,color[1] * 255,color[2] * 255), 2)

        # 该部分用于显示文字与置信度
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(masked_image, caption, (x1, y1 + 8), font, 1, (255, 255, 255), 2)

        # 该部分用于显示语义分割part
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # 画出语义分割的范围
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask

        # 画出语义分割边界框
        '''
        参数说明:
        mode: cv2.RETE_EXTERNAL  仅检索极端的外部轮廓，为所有轮廓设置了层次hierarchy[i][2]= hierarchy[i][3]=-1
              cv2.RETR_LIST 在不建立任何层次关系的情况下检索所有轮廓
              cv2.RETR_CCOMP 检索所有轮廓并将其组织为两级层次结构。在顶层，组件具有外部边界；在第二层，有孔的边界。如果所连接零部件的孔内还有其他轮廓，则该轮廓仍将放置在顶层。
              cv2.RETR_TREE 检索所有轮廓，并重建嵌套轮廓的完整层次
              cv2.RETE_FLOODFILL 输入图像也可以是32位的整形图像 CV——32SC1
        method: cv2.CHAIN_APPROX_NONE 存储所有的轮廓点，任何一个包含一两个点的子序列（不改变顺序索引的连续的）相邻
                cv2.CHAIN_APPROX_SIMPLE  压缩水平，垂直和对角线段，仅保留其端点。 例如，一个直立的矩形轮廓编码有4个点
                cv2.CHAIN_APPROX_TC89_L1 和 cv2.CHAIN_APPROX_TC89_KCOS 近似算法
        '''
        contours, h= cv2.findContours(padded_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.polylines(masked_image, contours, 1, (color[0] * 255,color[1] * 255,color[2] * 255), 2)

        # 根据轮廓线画出近似轮廓
        epsilon = 0.01*cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True)
        cv2.polylines(masked_image, [approx], 1, (color[0] * 255,color[1] * 255,color[2] * 255), 2)
        shapes.append((label, [approx]))
            

    img = Image.fromarray(np.uint8(masked_image))
    return img, shapes