#pragma once
#ifndef yoloxmodel
#define yoloxmodel

#include <iostream>
#include <vector>
#include <assert.h>
#include<onnxruntime_cxx_api.h>
#include<ctime>


class yoloxmodelinference {
public:
	yoloxmodelinference(const wchar_t* onnx_model_path);
	float* predict(std::vector<float>input_tensor_values, int batch_size = 1);
private:
	Ort::Env env;
	Ort::Session session;
	Ort::AllocatorWithDefaultOptions allocator;
	std::vector<const char*>input_node_names;
	std::vector<const char*>output_node_names;
	std::vector<int64_t> input_node_dims;
	std::size_t num_output_nodes;
	std::size_t num_input_nodes;

};


#endif // !yoloxmodel
