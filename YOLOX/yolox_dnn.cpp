#include "yolox_dnn.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;


/*
* https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/TensorRT/cpp/yolox.cpp
*/

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

void YOLO::Detect(Mat& image, Net& net, float* pdata)
{
	Mat blob;
	vector<Output> yolox_result;
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
	pdata = (float*)netOutputImg[0].data;
	decodeBox(pdata, yolox_result, 1, 1, 1);
	cout << "finish" << endl;
}

void YOLO::generate_grids_and_stride(std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
	for (auto stride : strides)
	{
		int num_grid_y = netHeight / stride;
		int num_grid_x = netWidth / stride;
		for (int g1 = 0; g1 < num_grid_y; g1++)
		{
			for (int g0 = 0; g0 < num_grid_x; g0++)
			{
				grid_strides.push_back({ g0, g1, stride });
			}
		}
	}
}

void YOLO::generate_yolox_proposals(std::vector<GridAndStride> grid_strides, float* feat_blob, float prob_threshold, std::vector<Output>& objects)
{

	const int num_anchors = grid_strides.size();

	for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
	{
		const int grid0 = grid_strides[anchor_idx].grid0;
		const int grid1 = grid_strides[anchor_idx].grid1;
		const int stride = grid_strides[anchor_idx].stride;

		const int basic_pos = anchor_idx * (NUM_CLASSES + 5);

		// yolox/models/yolo_head.py decode logic
		float x_center = (feat_blob[basic_pos + 0] + grid0) * stride;
		float y_center = (feat_blob[basic_pos + 1] + grid1) * stride;
		float w = exp(feat_blob[basic_pos + 2]) * stride;
		float h = exp(feat_blob[basic_pos + 3]) * stride;
		float x0 = x_center - w * 0.5f;
		float y0 = y_center - h * 0.5f;

		float box_objectness = feat_blob[basic_pos + 4];
		for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++)
		{
			float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
			float box_prob = box_objectness * box_cls_score;
			if (box_prob > prob_threshold)
			{
				Output obj;
				obj.box.x = x0;
				obj.box.y = y0;
				obj.box.width = w;
				obj.box.height = h;
				obj.id = class_idx;
				obj.confidence = box_prob;

				objects.push_back(obj);
			}

		} // class loop

	} // point anchor loop
}

static inline float intersection_area(const Output& a, const Output& b)
{
	cv::Rect_<float> inter = a.box & b.box;
	return inter.area();
}

static void nms_sorted_bboxes(const std::vector<Output>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
	picked.clear();

	const int n = faceobjects.size();

	std::vector<float> areas(n);
	for (int i = 0; i < n; i++)
	{
		areas[i] = faceobjects[i].box.area();
	}

	for (int i = 0; i < n; i++)
	{
		const Output& a = faceobjects[i];

		int keep = 1;
		for (int j = 0; j < (int)picked.size(); j++)
		{
			const Output& b = faceobjects[picked[j]];

			// intersection over union
			float inter_area = intersection_area(a, b);
			float union_area = areas[i] + areas[picked[j]] - inter_area;
			// float IoU = inter_area / union_area
			if (inter_area / union_area > nms_threshold)
				keep = 0;
		}

		if (keep)
			picked.push_back(i);
	}
}

void YOLO::decodeBox(float* prob, std::vector<Output>& objects, float scale, const int img_w, const int img_h)
{
	std::vector<Output> proposals;
	std::vector<int> strides = { 8, 16, 32 };
	std::vector<GridAndStride> grid_strides;
	generate_grids_and_stride(strides, grid_strides);
	generate_yolox_proposals(grid_strides, prob, nmsScoreThreshold, proposals);
	std::cout << "num of boxes before nms: " << proposals.size() << std::endl;
	//¾­¹ýNMSËã·¨
	std::vector<int> picked;
	nms_sorted_bboxes(proposals, picked, nmsThreshold);
	int count = picked.size();

	std::cout << "num of boxes: " << count << std::endl;

	objects.resize(count);
	for (int i = 0; i < count; i++)
	{
		objects[i] = proposals[picked[i]];

		// adjust offset to original unpadded
		float x0 = (objects[i].box.x) / scale;
		float y0 = (objects[i].box.y) / scale;
		float x1 = (objects[i].box.x + objects[i].box.width) / scale;
		float y1 = (objects[i].box.y + objects[i].box.height) / scale;

		// clip
		x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
		y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
		x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
		y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

		objects[i].box.x = x0;
		objects[i].box.y = y0;
		objects[i].box.width = x1 - x0;
		objects[i].box.height = y1 - y0;
	}
}



