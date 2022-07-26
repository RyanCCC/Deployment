#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>

struct Output {
	//类别
	int id;
	//置信度
	float confidence;
	//矩形框
	cv::Rect box;
};

class YOLO {
public:
	YOLO() {

	}
	~YOLO(){}
	bool initModel(cv::dnn::Net& net, std::string& netPath, bool isCuda);
	std::vector<Output>& Detect(cv::Mat& image, cv::dnn::Net& net);

private:
	//网络输入的shape
	const int netWidth = 640;   //ONNX图片输入宽度
	const int netHeight = 640;  //ONNX图片输入高度
	const int strideSize = 3;   //stride size
	float boxThreshold = 0.25;
	float classThreshold = 0.25;
	float nmsThreshold = 0.45;
	float nmsScoreThreshold = boxThreshold * classThreshold;

};