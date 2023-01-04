#pragma once
#ifndef yoloxmodel
#define yoloxmodel

#include <iostream>
#include <vector>
#include <assert.h>
#include<onnxruntime_cxx_api.h>
#include<ctime>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

/*
* onnxruntime进行推理
*/

class yoloxmodelinference {
public:
	yoloxmodelinference(const wchar_t* onnx_model_path);
	float* predict_test(std::vector<float>input_tensor_values, int batch_size = 1);
	cv::Mat predict(cv::Mat& input_tensor, int batch_size = 1, int index = 0);
	std::vector<float> predict(std::vector<float>& input_data, int batch_size = 1, int index = 0);
private:
	Ort::Env env;
	Ort::Session session;
	Ort::AllocatorWithDefaultOptions allocator;
	std::vector<const char*>input_node_names;
	std::vector<const char*>output_node_names;
	std::vector<int64_t> input_node_dims;
	std::vector<int64_t> output_node_dims;
	std::size_t num_output_nodes;
	std::size_t num_input_nodes;
	const int netWidth = 640;
	const int netHeight = 640;
	const int strideSize = 3;//stride size

	float boxThreshold = 0.25;
};


#endif // !yoloxmodel
