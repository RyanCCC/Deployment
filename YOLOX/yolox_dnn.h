#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>

struct Output {
	//���
	int id;
	//���Ŷ�
	float confidence;
	//���ο�
	cv::Rect box;
};

class YOLO {
public:
	YOLO() {

	}
	~YOLO(){}
	bool initModel(cv::dnn::Net& net, std::string& netPath, bool isCuda);
	//float* blobFromImage(cv::Mat& img);
	std::vector<Output>& Detect(cv::Mat& image, cv::dnn::Net& net);
	void decodeBox();

private:
	//���������shape
	const int netWidth = 640;   //ONNXͼƬ������
	const int netHeight = 640;  //ONNXͼƬ����߶�
	const int strideSize = 3;   //stride size
	float boxThreshold = 0.25;
	float classThreshold = 0.25;
	float nmsThreshold = 0.45;
	float nmsScoreThreshold = boxThreshold * classThreshold;

};