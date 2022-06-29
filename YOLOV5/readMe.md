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

请看：`inference.py`文件

### Openvino C++



### 注意

1. 关于TensorRT加速

   C++的加速效果更好，但是涉及到要会用C++把自己的工作写一遍才行，现在开源出来包括网上很多方法都是用C++来加速的，这个方法更加灵活，适应性更强（多输入多输出之类），而且最终只会生成一个engine，看起来很简洁。
   这部分如果有C++基础的真的强烈推荐，可以参考如下工程：
   yolov5：https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5
   yoloX：https://github.com/Megvii-BaseDetection/YOLOX

   不会C++可以考虑用Torchtrt来加速，上述就提供了一个加速yolov5的例子，当然需要把一些加速不了的东西拿出来。此外，这个方法遇到多输入多输出也可以用一些折中的方法进行加速，比如把可以加速的部分划分成多个模型加速，加速不了的用原来的方法计算，这样依然可以获得很高的速度收益。
   这类方法简单有效，适合不精通C++但需要加速的人群，可以参考如下工程：
   yoloX：https://github.com/Megvii-BaseDetection/YOLOX
   Ocean：https://github.com/researchmm/TracKit/blob/master/lib/tutorial/Ocean/ocean.md

### 参考

1. [C++ yolov5 TensorRT](https://blog.csdn.net/qq_34919792/article/details/120650792)
2. [C++ yolov5 openvino](https://blog.csdn.net/qq_40700822/article/details/115709175)