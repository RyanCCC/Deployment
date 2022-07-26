#include "yolox_dnn.h"

bool YOLO::initModel(cv::dnn::Net& net, std::string& netPath, bool isCuda)
{
	try {
		net = cv::dnn::readNet(netPath);
	}
	catch (const std::exception& e) {
		std::cout << e.what() << std::endl;
		return false;
	}
	//cuda
	//if (isCuda) {
	//	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	//	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
	//}
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	return true;
}
