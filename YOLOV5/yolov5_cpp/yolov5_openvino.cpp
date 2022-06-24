#include "yolov5_openvino.h"


/**
reference:
    https://blog.csdn.net/hh1357102/article/details/124956796
    https://github.com/fb029ed/yolov5_cpp_openvino
**/


YOLOV5::YOLOV5(){}
YOLOV5::~YOLOV5(){}

bool YOLOV5::parse_yolov5(const Blob::Ptr& blob, int net_grid, float cof_threshold, vector<Rect>& o_rect, vector<float>& o_rect_cof) {
    vector<int> anchors = get_anchors(net_grid);
    LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
    const float* output_blob = blobMapped.as<float*>();
    //int item_size = 6;
    int item_size = 85;
    size_t anchor_n = 3;
    for (int n = 0; n < anchor_n; ++n)
        for (int i = 0; i < net_grid; ++i)
            for (int j = 0; j < net_grid; ++j)
            {
                double box_prob = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 4];
                box_prob = sigmoid(box_prob);
                if (box_prob < cof_threshold)
                    continue;

                double x = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 0];
                double y = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 1];
                double w = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 2];
                double h = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 3];

                double max_prob = 0;
                int idx = 0;
                for (int t = 5; t < 85; ++t) {
                    double tp = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + t];
                    tp = sigmoid(tp);
                    if (tp > max_prob) {
                        max_prob = tp;
                        idx = t;
                    }
                }
                float cof = box_prob * max_prob;
                if (cof < cof_threshold)
                    continue;

                x = (sigmoid(x) * 2 - 0.5 + j) * 640.0f / net_grid;
                y = (sigmoid(y) * 2 - 0.5 + i) * 640.0f / net_grid;
                w = pow(sigmoid(w) * 2, 2) * anchors[n * 2];
                h = pow(sigmoid(h) * 2, 2) * anchors[n * 2 + 1];

                double r_x = x - w / 2;
                double r_y = y - h / 2;
                Rect rect = Rect(round(r_x), round(r_y), round(w), round(h));
                o_rect.push_back(rect);
                o_rect_cof.push_back(cof);
            }
    if (o_rect.size() == 0) return false;
    else return true;
}

bool YOLOV5::init(string xml_path, double cof_threshold, double nms_area_threshold) {
    _xml_path = xml_path;
    _cof_threshold = cof_threshold;
    _nms_area_threshold = nms_area_threshold;
    Core ie;
    auto cnnNetwork = ie.ReadNetwork(_xml_path);
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    InputInfo::Ptr& input = inputInfo.begin()->second;
    _input_name = inputInfo.begin()->first;
    input->setPrecision(Precision::FP32);
    input->getInputData()->setLayout(Layout::NCHW);
    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    SizeVector& inSizeVector = inputShapes.begin()->second;
    cnnNetwork.reshape(inputShapes);
    _outputinfo = OutputsDataMap(cnnNetwork.getOutputsInfo());
    for (auto& output : _outputinfo) {
        output.second->setPrecision(Precision::FP32);
    }
    //_network =  ie.LoadNetwork(cnnNetwork, "GPU");
    _network = ie.LoadNetwork(cnnNetwork, "CPU");
    return true;
}

bool YOLOV5::uninit() {
    return true;
}

bool YOLOV5::process_frame(Mat& inframe, vector<Object>& detected_objects) {
    if (inframe.empty()) {
        cout << "processing..." << endl;
        return false;
    }
    resize(inframe, inframe, Size(640, 640));
    cvtColor(inframe, inframe, COLOR_BGR2RGB);
    size_t img_size = 640 * 640;
    InferRequest::Ptr infer_request = _network.CreateInferRequestPtr();
    Blob::Ptr frameBlob = infer_request->GetBlob(_input_name);
    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(frameBlob)->wmap();
    float* blob_data = blobMapped.as<float*>();
    //nchw
    for (size_t row = 0; row < 640; row++) {
        for (size_t col = 0; col < 640; col++) {
            for (size_t ch = 0; ch < 3; ch++) {
                blob_data[img_size * ch + row * 640 + col] = float(inframe.at<Vec3b>(row, col)[ch]) / 255.0f;
            }
        }
    }
    infer_request->Infer();
    vector<Rect> origin_rect;
    vector<float> origin_rect_cof;
    int s[3] = { 80,40,20 };
    int i = 0;
    for (auto& output : _outputinfo) {
        auto output_name = output.first;
        Blob::Ptr blob = infer_request->GetBlob(output_name);
        parse_yolov5(blob, s[i], _cof_threshold, origin_rect, origin_rect_cof);
        ++i;
    }
    vector<int> final_id;
    dnn::NMSBoxes(origin_rect, origin_rect_cof, _cof_threshold, _nms_area_threshold, final_id);
    for (int i = 0; i < final_id.size(); ++i) {
        Rect resize_rect = origin_rect[final_id[i]];
        detected_objects.push_back(Object{
            origin_rect_cof[final_id[i]],
            "",resize_rect
            });
    }
    return true;
}

double YOLOV5::sigmoid(double x) {
    return (1 / (1 + exp(-x)));
}

vector<int> YOLOV5::get_anchors(int net_grid) {
    vector<int> anchors(6);
    int a80[6] = { 10,13, 16,30, 33,23 };
    int a40[6] = { 30,61, 62,45, 59,119 };
    int a20[6] = { 116,90, 156,198, 373,326 };
    if (net_grid == 80) {
        anchors.insert(anchors.begin(), a80, a80 + 6);
    }
    else if (net_grid == 40) {
        anchors.insert(anchors.begin(), a40, a40 + 6);
    }
    else if (net_grid == 20) {
        anchors.insert(anchors.begin(), a20, a20 + 6);
    }
    return anchors;
}