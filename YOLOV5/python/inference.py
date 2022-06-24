'''
使用openvino对视频进行异步推理
'''
from charset_normalizer import from_path
import cv2
from cv2 import rotate
import numpy as np
import yaml
from openvino.inference_engine import IECore
from openvino.runtime import Core  # the version of openvino >= 2022.1
import random
from PIL import Image
import skvideo.io

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), scaleup=False, stride=32):
    """
    将图片缩放调整到指定大小,1920x1080的图片最终会缩放到640x384的大小，和YOLOv4的letterbox不一样
    Resize and pad image while meeting stride-multiple constraints
    :param img: 原图 hwc
    :param new_shape: 缩放后的最长边大小
    :param color: pad的颜色
    :param auto: True：进行矩形填充  False：直接进行resize
    :param scale_up: True：仍进行上采样 False：不进行上采样
    :return: img: letterbox后的图片 HWC
             ratio: wh ratios
             (dw, dh): w和h的pad
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # 只进行下采样 因为上采样会让图片模糊
    # (for better test mAP) scale_up = False 对于大于new_shape（r<1）的原图进行缩放,小于new_shape（r>1）的不变
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)
    ratio = r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # 这里的取余操作可以保证padding后的图片是32的整数倍(416x416)，如果是(512x512)可以保证是64的整数倍
    # dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    # 在较小边的两侧进行pad, 而不是在一侧pad
    # divide padding into 2 sides
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img,ratio,(dw,dh)

def iou(b1,b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:,0], b2[:,1], b2[:,2], b2[:,3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)

    area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / np.maximum((area_b1+area_b2-inter_area),1e-6)
    return iou

#非极大值抑制函数
def non_max_suppression(boxes, conf_thres=0.5, nms_thres=0.4, ratio=1, pad=(20,20)):
    # 取出batch_size
    bs = np.shape(boxes)[0]
    # xywh___ to____ xyxy
    shape_boxes = np.zeros_like(boxes[:,:,:4])
    shape_boxes[:, :, 0] = boxes[:, :, 0] - boxes[:, :, 2] / 2
    shape_boxes[:, :, 1] = boxes[:, :, 1] - boxes[:, :, 3] / 2
    shape_boxes[:, :, 2] = boxes[:, :, 0] + boxes[:, :, 2] / 2
    shape_boxes[:, :, 3] = boxes[:, :, 1] + boxes[:, :, 3] / 2
    boxes[:, :, :4] = shape_boxes
    boxes[:, :, 5:] *= boxes[:, :, 4:5]

    # output存放每一张图片的预测结果，推理阶段一般是一张图片
    output = []
    for i in range(bs):
        predictions = boxes[i]  # 预测位置xyxy  shape==(12700,85)
        score = np.max(predictions[:, 5:], axis=-1)
        # score = predictions[:,4]  # 存在物体置信度,shape==12700
        mask = score > conf_thres  # 物体置信度阈值mask==[False,False,True......],shape==12700,True将会被保留，False列将会被删除
        detections = predictions[mask]  # 第一次筛选  shape==(115,85)
        class_conf = np.expand_dims(np.max(detections[:,5:],axis=-1),axis=-1)  # 获取每个预测框预测的类别置信度
        class_pred = np.expand_dims(np.argmax(detections[:,5:],axis=-1),axis=-1)  # 获取每个预测框的类别下标
        # 结果堆叠，(num_boxes,位置信息4+包含物体概率1+类别置信度1+类别序号1)
        detections = np.concatenate([detections[:,:4],class_conf,class_pred],axis=-1)  # shape=(numbox,7)

        unique_class = np.unique(detections[:,-1])  # 取出包含的所有类别
        if len(unique_class)==0:
            continue
        best_box = []
        for c in unique_class:
            # 取出类别为c的预测结果
            cls_mask = detections[:,-1] == c
            detection = detections[cls_mask] # shape=(82,7)

            # 包含物体类别概率从高至低排列
            scores = detection[:,4]
            arg_sort = np.argsort(scores)[::-1]  # 返回的是索引
            detection = detection[arg_sort]

            while len(detection) != 0:
                best_box.append(detection[0])
                if len(detection) == 1:
                    break
                # 计算当前置信度最大的框和其它预测框的iou
                ious = iou(best_box[-1],detection[1:])
                detection = detection[1:][ious < nms_thres]  # 小于nms_thres将被保留，每一轮至少减少一个
        output.append(best_box)

    boxes_loc = []
    conf_loc = []
    class_loc = []
    if len(output):
        for i in range(len(output)):
            pred = output[i]
            for i, det in enumerate(pred):
                if len(det):
                    # 将框坐标调整回原始图像中
                    det[0] = (det[0] - pad[0]) / ratio
                    det[2] = (det[2] - pad[0]) / ratio
                    det[1] = (det[1] - pad[1]) / ratio
                    det[3] = (det[3] - pad[1]) / ratio
                    boxes_loc.append([det[0],det[1],det[2],det[3]])
                    conf_loc.append(det[4])
                    class_loc.append(det[5])
    return boxes_loc,conf_loc,class_loc

def plot_one_box(img,boxes,conf,clas_id,line_thickness=3,names=None):
    # 画位置框
    # tl = 框的线宽  要么等于line_thickness要么根据原图im长宽信息自适应生成一个
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(boxes[0]), int(boxes[1])), (int(boxes[2]),int(boxes[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    # 画类别信息框
    label = f'{names[int(clas_id)]} {conf:.2f}'
    tf = max(tl - 1, 1)  # label字体的线宽 font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

if __name__ == '__main__':
    # # 样例使用的COCO 80类别数据集
    names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone','microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush']
    conf_thres = 0.5
    nms_thres = 0.4

    #frame = cv2.imread(img_path)
    model_xml=r"../yolov5s.xml"
    model_bin=r"../yolov5s.bin"
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    #双请求，自动分配GPU及CPU资源：
    exec_net = ie.load_network(network=net, num_requests=2, device_name="HETERO:GPU,CPU")
    input_layer = next(iter(net.input_info))

    is_async_mode = True
    video_path = '../video/test.MOV'
    metadata = skvideo.io.ffprobe(video_path)
    rotate_angle = 0
    try:
        d = metadata['video'].get('tag')[0]
        if d.setdefault('@key') == 'rotate':
            rotate_angle = int(d.setdefault('@value'))
    except:
        pass
    cap = cv2.VideoCapture(video_path)
    #fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
    #摄像头一次读入的帧数：
    number_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #如果该值无效，默认其帧数为1，实际是指无法获取；如能获取，则该值为实际输入帧数
    number_input_frames = 1 if number_input_frames != -1 and number_input_frames < 0 else number_input_frames

    wait_key_code = 1
    #异步模式下定义了当前帧及下一帧的帧ID：
    cur_request_id = 0
    next_request_id = 1

    if number_input_frames != 1:
        is_async_mode = True
        #给出异步模式的初始第一帧：
        ret, frame = cap.read()
    else:
        is_async_mode = False
        wait_key_code = 0

    # 推理摄像头
    
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        # 异步模式是指启动第一张图片的AI推理计算后，无需等待AI推理计算结束，直接采集第二张图片，并完成图像预处理，然后再检查第一张图片推理计算是否完毕，若完毕，处理输出结果，
        # 这样的好处是，并行执行了第一张图片的AI推理计算任务和第二张图片的图像采集和预处理任务，减少了等待时间，提高了硬件的利用率，也提高了吞吐量。
        if is_async_mode:
            #异步模式，捕获帧提供给下一个请求：
            ret, next_frame = cap.read()
        else:
            #常规同步模式，捕获帧提供给当前请求
            ret, frame = cap.read()
        if not ret:
            break
        # 是否需要旋转frame
        frame = Image.fromarray(frame)
        frame = np.array(frame.rotate(-rotate_angle))
        # frame.show()


        if is_async_mode:
            request_id = next_request_id
            img, ratio, (dw,dh) = letterbox(frame)
        else:
            request_id = cur_request_id
            img, ratio, (dw,dh) = letterbox(frame)

        # np.ascontiguousarray()将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
        blob = cv2.dnn.blobFromImage(np.ascontiguousarray(img), 1/255.0, (img.shape[0], img.shape[1]), swapRB=True, crop=False)
        infer_request_handle=exec_net.start_async(request_id=request_id,inputs={input_layer: blob})
        #0代表推理结果已出现：
        if infer_request_handle.wait(-1) == 0:
            #这里直接使用新版yolo5的3合1输出：
            res = infer_request_handle.output_blobs["output"]
            outs = res.buffer
            boxes_loc,conf_loc,class_loc = non_max_suppression(outs, conf_thres=conf_thres, nms_thres=nms_thres,
                                                               ratio=ratio, pad=(dw,dh))

            # 可视化
            for i in range(len(boxes_loc)):
                boxes = boxes_loc[i]
                conf = conf_loc[i]
                clas_id = class_loc[i]
                plot_one_box(frame, boxes, conf, clas_id, line_thickness=3, names=names)

            cv2.imshow("result", frame)
            key=cv2.waitKey(1)

            if is_async_mode:
                #异步模式下，下一帧及当前帧请求号互换：
                cur_request_id, next_request_id = next_request_id, cur_request_id
                frame = next_frame
            # ESC按键按下时退出程序：
            if key == 27:
               break
    #while循环结束时释放资源：
    cv2.destroyAllWindows()