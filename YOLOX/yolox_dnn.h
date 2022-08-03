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

struct GridAndStride
{
	int grid0;
	int grid1;
	int stride;
};

class YOLO {
public:
	YOLO() {

	}
	~YOLO(){}
	bool initModel(cv::dnn::Net& net, std::string& netPath, bool isCuda);
	void Detect(cv::Mat& image, cv::dnn::Net& net, float* pdata);
	//float* blobFromImage(cv::Mat& img);
	void decodeBox(float* prob, std::vector<Output>& objects, float scale, const int img_w, const int img_h);
	void generate_grids_and_stride(std::vector<int>& strides, std::vector<GridAndStride>& grid_strides);
	void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, float* feat_blob, float prob_threshold, std::vector<Output>& objects);

private:
	//网络输入的shape
	const int netWidth = 640;   //ONNX图片输入宽度
	const int netHeight = 640;  //ONNX图片输入高度
	float boxThreshold = 0.25;
	float classThreshold = 0.25;
	float nmsThreshold = 0.45;
	float nmsScoreThreshold = boxThreshold * classThreshold;
	int NUM_CLASSES = 80;

};