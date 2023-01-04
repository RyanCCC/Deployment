#include<ctime>
#include "yoloxmodel.h"
#include "yolox_dnn.h"
using namespace cv::dnn;
using namespace cv;

/*
* ONNX安装地址：https://onnxruntime.ai/
* 安装可参考：https://blog.csdn.net/qq_19865329/article/details/115945454
* 安装命令：Install-Package Microsoft.ML.OnnxRuntime -Version 1.12.0
* 可参考：https://blog.csdn.net/qq_34124780/article/details/121079317
*/


/*
* 模型下载：
* yolox_pytorch1.7_onnxv6_1_3_640_640.onnx：https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime
*/

/*
* 环境加载步骤：
* 1. 初始化环境、会话等
* 2. 会话中加载模型，得到模型的输入和输出节点
* 3. 调用API得到模型的返回值
*/
using namespace std;


int main()
{
    //使用DNN操作模型
    std::string model = "./models/yolox_pytorch1.7_onnxv6_1_3_640_640.onnx";
    YOLO yolox_dnn;
    Net net;
    bool flag = yolox_dnn.initModel(net, model, false);
    if (flag) {
        cout << "Init model successful";
    }
    else {
        cout << "Init model fail";
    }
    //推理
    float* pred_data = NULL;
    
    Mat img = imread("./test.jpg");
    yolox_dnn.Detect(img, net, pred_data);
    
    
    



    //记录程序运行时间
    auto start_time = clock();
    const wchar_t* model_path = L"./models/model_yolox_13.onnx";
    yoloxmodelinference yolox(model_path);
    printf("Using Onnxruntime C++ API\n");
    //以图像进行推理
    cv::Mat image = cv::imread("./test.jpg");
    cv::resize(image, image, { 640, 640 }, 0.0, 0.0, cv::INTER_CUBIC);
    //cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    auto result = yolox.predict(image);
    size_t input_tensor_size = 3 * 640 * 640;
    std::vector<float> input_tensor_values(input_tensor_size);
    //初始化一个数据（演示用）
    for (unsigned int i = 0; i < input_tensor_size; i++)
    {
        input_tensor_values[i] = (float)i / (input_tensor_size + 1);
    }
    float* results = nullptr;
    try
    {
        results = yolox.predict_test(input_tensor_values);
    }
    catch (Ort::Exception& e)
    {
        delete results;
        printf("%s\n", e.what());
    }
    auto end_time = std::clock();
    printf("Proceed exits after %.2f seconds", static_cast<float>(end_time - start_time) / 1000);
    printf("Done!\n");
    return 0;

}



/*原始没有封装代码：
* //初始化环境，每个进程一个环境，环境保留了线程池和其他状态信息
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolox");
    //初始化session选项
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible opitmizations
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    //加载模型
    const wchar_t* model_path = L"./models/model_yolox_13.onnx";
    printf("Using Onnxruntime C++ API\n");
    Ort::Session session(env, model_path, session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    //输出模型输入节点的数量
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<const char*> output_node_names(num_output_nodes);
    std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                           // Otherwise need vector<vector<>>

    printf("Number of inputs = %zu\n", num_input_nodes);
    //迭代所有的输入节点
    for (int i = 0; i < num_input_nodes; i++) {
        //输出输入节点的名称
        char* input_name = session.GetInputName(i, allocator);
        printf("Input %d : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // 输出输入节点的类型
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        input_node_dims = tensor_info.GetShape();
        //输入节点的打印维度
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        //打印各个维度的大小
        for (int j = 0; j < input_node_dims.size(); j++)
            printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
        //batch_size=1
        input_node_dims[0] = 1;
    }
    //打印输出节点信息，方法类似
    for (int i = 0; i < num_output_nodes; i++)
    {
        char* output_name = session.GetOutputName(i, allocator);
        printf("Output: %d name=%s\n", i, output_name);
        output_node_names[i] = output_name;
        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Output %d : type=%d\n", i, type);
        auto output_node_dims = tensor_info.GetShape();
        printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
        for (int j = 0; j < input_node_dims.size(); j++)
            printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
    }

    //自定义测试用的Tensor对象
    size_t input_tensor_size = 3 * 640 * 640;
    std::vector<float> input_tensor_values(input_tensor_size);
    for (unsigned int i = 0; i < input_tensor_size; i++)
        input_tensor_values[i] = (float)i / (input_tensor_size + 1);

    //为输入对象创建一个Tensor对象
    try {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);

        //推理
        auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), num_output_nodes);
        float* floatarr = output_tensors.front().GetTensorMutableData<float>();
        cout << "Number of outputs = " << output_tensors.size() << endl;
    }
    catch (Ort::Exception& e)
    {
        cout << e.what() << endl;
    }
    auto end_time = clock();
    cout << "Proceed exit after " << static_cast<float>(end_time - start_time) / CLOCKS_PER_SEC << " seconds";
    cout << "Done!" << endl;
    return 0;
*/
