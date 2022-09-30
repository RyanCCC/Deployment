#include<fstream>
#include<sstream>
#include<iostream>
#include<opencv2/dnn.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

struct Net_config
{
	float confThreshold;
	float nmsThreshold;
	string modelpath;
};

class YOLOV7
{
public:
	YOLOV7(Net_config);
	void detect(Mat& frame);
private:
	int inpWidth;
	int inpHeight;
	vector<string> class_names;
	int num_class;
	float confThreshold;
	float nmsThreshold;
	Net net;
	void drawPred(float conf, int left, int top, int right, int bottom, Mat& Frame, int classid);
};

//初始化网络
YOLOV7::YOLOV7(Net_config config) {
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;

	this->net = readNet(config.modelpath);
	ifstream ifs("coco.names");
	string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	this->num_class = class_names.size();

	size_t pos = config.modelpath.find("_");
	int len = config.modelpath.length() - 6 - pos;
	string hxw = config.modelpath.substr(pos + 1, len);
	pos = hxw.find("x");
	string h = hxw.substr(0, pos);
	len = hxw.length() - pos;
	string w = hxw.substr(pos + 1, len);
	this->inpHeight = stoi(h);
	this->inpWidth = stoi(w);
}

//Draw the predicted bounding box
void YOLOV7::drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid) {
	
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 2);

	//label
	string label = format("%.2f", conf);
	label = this->class_names[classid] + ":" + label;

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
}

void YOLOV7::detect(Mat& frame) {
	Mat blob = blobFromImage(frame, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

	int num_proposal = outs[0].size[0];
	int nout = outs[0].size[1];
	if (outs[0].dims > 2) {
		num_proposal = outs[0].size[1];
		nout = outs[0].size[2];
		outs[0] = outs[0].reshape(0, num_proposal);
	}
	//generate proposal
	vector<float> confidences;
	vector<Rect> boxes;
	vector<int> classIds;

	float ratioh = (float)frame.rows / this->inpHeight, ratiow = (float)frame.cols / this->inpWidth;
	int n = 0, row_ind = 0;
	float* pdata = (float*)outs[0].data;
	//筛选出高于阈值的box
	for (n = 0; n < num_proposal; n++) {
		float box_score = pdata[4];
		if (box_score > this->confThreshold) {
			Mat scores = outs[0].row(row_ind).colRange(5, nout);
			Point classIdPoint;
			double max_class_socre;
			minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
			max_class_socre *= box_score;
			if (max_class_socre > this->confThreshold)
			{
				const int class_idx = classIdPoint.x;
				float cx = pdata[0] * ratiow;  ///cx
				float cy = pdata[1] * ratioh;   ///cy
				float w = pdata[2] * ratiow;   ///w
				float h = pdata[3] * ratioh;  ///h

				int left = int(cx - 0.5 * w);
				int top = int(cy - 0.5 * h);

				confidences.push_back((float)max_class_socre);
				boxes.push_back(Rect(left, top, (int)(w), (int)(h)));
				classIds.push_back(class_idx);
			}
		}
		row_ind++;
		pdata += nout;
	}

	//NMS
	vector<int> indices;
	dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		this->drawPred(confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame, classIds[idx]);
	}
}

int main()
{
	Net_config YOLOV7_nets = { 0.3, 0.5, "../../models/yolov7_480x640.onnx" };
	YOLOV7 net(YOLOV7_nets);
	string imgpath = "./samples/dog.jpg";
	Mat srcimg = imread(imgpath);
	net.detect(srcimg);

	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}


