#include "yolox_dnn.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;

bool YOLO::initModel(Net& net, string& netPath, bool isCuda)
{
	try {
		net = readNet(netPath);
	}
	catch (const exception& e) {
		cout << e.what() << std::endl;
		return false;
	}
	//cuda
	//if (isCuda) {
	//	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	//	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
	//}
	net.setPreferableBackend(DNN_BACKEND_DEFAULT);
	net.setPreferableTarget(DNN_TARGET_CPU);
	return true;
}

std::vector<Output>& YOLO::Detect(Mat& image, Net& net)
{
	Mat blob;
	vector<Output> output;
	int col = image.cols;
	int row = image.rows;
	int maxLen = MAX(col, row);
	Mat netInput = image.clone();
	//resize image
	if (maxLen > 1.2 * col || maxLen > 1.2 * row) {
		Mat resizeImg = Mat::zeros(maxLen, maxLen, CV_8UC3);
		image.copyTo(resizeImg(Rect(0, 0, col, row)));
		netInput = resizeImg;
	}
	vector<Ptr<Layer>> layer;
	vector<String>layer_names;
	layer_names = net.getLayerNames();
	//cv::Scalar(104, 117, 123)¡¢cv::Scalar(114, 114,114)
	blobFromImage(netInput, blob, 1/255.0, Size(netWidth, netHeight), Scalar(0, 0, 0), true, false);
	net.setInput(blob);
	vector<cv::Mat> netOutputImg;
	net.forward(netOutputImg, net.getUnconnectedOutLayersNames());
	cout << "finish" << endl;
	return output;




}
