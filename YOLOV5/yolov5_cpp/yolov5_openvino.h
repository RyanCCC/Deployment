#pragma once
#ifndef  YOLOV5VINO_H
#define YOLOV5VINO_H

#include <fstream>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#define NOT_NCS2

using namespace cv;
using namespace dnn;
using namespace std;
using namespace InferenceEngine;

class YOLOV5
{
public:
    typedef struct {
        float prob;
        std::string name;
        cv::Rect rect;
    } Object;
    YOLOV5();
    ~YOLOV5();
    //��ʼ��
    bool init(string xml_path, double cof_threshold, double nms_area_threshold);
    //�ͷ���Դ
    bool uninit();
    //����ͼ���ȡ���
    bool process_frame(Mat& inframe, vector<Object>& detected_objects);

private:
    double sigmoid(double x);
    vector<int> get_anchors(int net_grid);
    bool parse_yolov5(const Blob::Ptr& blob, int net_grid, float cof_threshold,
        vector<Rect>& o_rect, vector<float>& o_rect_cof);
    Rect detet2origin(const Rect& dete_rect, float rate_to, int top, int left);
    //�洢��ʼ����õĿ�ִ������
    ExecutableNetwork _network;
    OutputsDataMap _outputinfo;
    string _input_name;
    //������
    string _xml_path;                             //OpenVINOģ��xml�ļ�·��
    double _cof_threshold;                //���Ŷ���ֵ,���㷽���ǿ����Ŷȳ�����Ʒ�������Ŷ�
    double _nms_area_threshold;  //nms��С�ص������ֵ
};




#endif // ! YOLOV5VINO_H