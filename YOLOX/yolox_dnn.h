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
	//���������shape
	const int netWidth = 640;   //ONNXͼƬ������
	const int netHeight = 640;  //ONNXͼƬ����߶�
	float boxThreshold = 0.25;
	float classThreshold = 0.25;
	float nmsThreshold = 0.45;
	float nmsScoreThreshold = boxThreshold * classThreshold;
	int NUM_CLASSES = 80;

};