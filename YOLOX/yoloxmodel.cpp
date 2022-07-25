#include "yoloxmodel.h"



yoloxmodelinference::yoloxmodelinference(const wchar_t* onnx_model_path):session(nullptr), env(nullptr) {
    //初始化环境，每个进程一个环境,环境保留了线程池和其他状态信息
    this->env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "yolox");
    //初始化Session选项
    Ort::SessionOptions session_options;
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // 创建Session并把模型加载到内存中
    this->session = Ort::Session(env, onnx_model_path, session_options);
    //输入输出节点数量和名称
    this->num_input_nodes = session.GetInputCount();
    this->num_output_nodes = session.GetOutputCount();
    for (int i = 0; i < this->num_input_nodes; i++)
    {
        auto input_node_name = session.GetInputName(i, allocator);
        this->input_node_names.push_back(input_node_name);
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        this->input_node_dims = tensor_info.GetShape();
    }
    for (int i = 0; i < this->num_output_nodes; i++)
    {
        auto output_node_name = session.GetOutputName(i, allocator);
        this->output_node_names.push_back(output_node_name);
    }
}

float* yoloxmodelinference::predict_test(std::vector<float> input_tensor_values, int batch_size)
{
    this->input_node_dims[0] = batch_size;
    auto input_tensor_size = input_tensor_values.size();
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), this->num_output_nodes);
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    return floatarr;
}

cv::Mat yoloxmodelinference::predict(cv::Mat& input_tensor, int batch_size, int index)
{
    //图像预处理
    int input_tensor_size = input_tensor.cols * input_tensor.rows * 3;
    std::size_t counter = 0;
    std::vector<float> input_data(input_tensor_size);
    std::vector<float> output_data;
    try {
        for (unsigned k = 0; k < 3; k++) {
            for (unsigned i = 0; i < input_tensor.rows; i++)
            {
                for (unsigned j = 0; j < input_tensor.cols; j++)
                {
                    input_data[counter++] = static_cast<float>(input_tensor.at<cv::Vec3b>(i, j)[k]) / 255.0;
                }
            }
        }
    }
    catch (cv::Exception& e)
    {
        printf(e.what());
    }
    try
    {

        output_data = this->predict(input_data);
    }
    catch (Ort::Exception& e)
    {
        throw e;
    }
    cv::Mat output_tensor(output_data);
    output_tensor = output_tensor.reshape(1, { 320,320 }) * 255.0;
    std::cout << output_tensor.rows << " " << output_tensor.cols  << std::endl;
    return output_tensor;
}

std::vector<float> yoloxmodelinference::predict(std::vector<float>& input_tensor_values, int batch_size, int index)
{
    this->input_node_dims[0] = batch_size;
    this->output_node_dims[0] = batch_size;
    float* floatarr = nullptr;
    try
    {
        std::vector<const char*>output_node_names;
        if (index != -1)
        {
            output_node_names = { this->output_node_names[index] };
        }
        else
        {
            output_node_names = this->output_node_names;
        }
        this->input_node_dims[0] = batch_size;
        auto input_tensor_size = input_tensor_values.size();
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
        auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
        assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
        floatarr = output_tensors.front().GetTensorMutableData<float>();
    }
    catch (Ort::Exception& e)
    {
        throw e;
    }
    int64_t output_tensor_size = 1;
    for (auto& it : this->output_node_dims)
    {
        output_tensor_size *= it;
    }
    std::vector<float>results(output_tensor_size);
    for (unsigned i = 0; i < output_tensor_size; i++)
    {
        results[i] = floatarr[i];
    }
    return results;
  
}
