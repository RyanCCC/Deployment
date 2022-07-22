#include "yoloxmodel.h"

yoloxmodelinference::yoloxmodelinference(const wchar_t* onnx_model_path):session(nullptr), env(nullptr) {
    //��ʼ��������ÿ������һ������,�����������̳߳غ�����״̬��Ϣ
    this->env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "yolox");
    //��ʼ��Sessionѡ��
    Ort::SessionOptions session_options;
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // ����Session����ģ�ͼ��ص��ڴ���
    this->session = Ort::Session(env, onnx_model_path, session_options);
    //��������ڵ�����������
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

float* yoloxmodelinference::predict(std::vector<float> input_tensor_values, int batch_size)
{
    this->input_node_dims[0] = batch_size;
    auto input_tensor_size = input_tensor_values.size();
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), this->num_output_nodes);
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    return floatarr;
}
