## YOLOV5部署

### 模型转换

在[YOLOV5](https://github.com/ultralytics/yolov5)上下载代码。模型转换的过程是将yolov5的pt文件先转换成ONNX模型，再转换成Openvino模型。

1. ONNX转换

下载`yolov5s.pt`，执行工程上面的`export.py`文件：
```
python export.py --weights ./yolov5s.pt --device '0' --batch-size 1 --imgsz (640, 640) --iou-thres 0.6 --conf-thres 0.65 --include ('onnx')
```
执行成功后，在目录下会生成`yolov5s.onnx`文件。

2. Openvino转换

执行当前路径下的`onnx2openvino.py`文件。生成`*.bin`和`*.xml`文件。

### Openvino Python

请看：inference文件


### Openvino C++