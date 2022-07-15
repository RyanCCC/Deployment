#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

struct YOLOX {
    int input_width;
    int input_height;
} yolox;


Mat resizeKeepAspectRatio(const Mat& img, int target_width = 500) {
    int width = img.cols;
    int height = img.rows;
    Mat square = cv::Mat::zeros(target_width, target_width, img.type());
    int max_dim = (width >= height) ? width : height;
    float scale = ((float)target_width) / max_dim;
    cv::Rect roi;
    if (width >= height)
    {
        roi.width = target_width;
        roi.x = 0;
        roi.height = height * scale;
        roi.y = (target_width - roi.height) / 2;
    }
    else
    {
        roi.y = 0;
        roi.height = target_width;
        roi.width = width * scale;
        roi.x = (target_width - roi.width) / 2;
    }

    //imshow("square", square(roi));

    cv::resize(img, square(roi), roi.size());

    return square;
}
Mat resizeImg(const Mat& img, int target_width = 500, int target_height = 500) {
    int width = img.cols;
    int height = img.rows;
    Mat square = cv::Mat::zeros(target_width, target_height, img.type());
    cv::Rect roi;
    roi.width = target_width;
    roi.x = 0;
    roi.height = target_height;
    roi.y = (target_width - roi.height) / 2;

    cv::resize(img, square, square.size());

    return square;
}

//YOLOV5的预处理
static inline Mat preprocess_img(Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.cols;
        x = 0;
        y = (input_h - h) / 2;
    }
    else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    Mat re(h, w, CV_8UC3);
    resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

int main()
{
    //YOLOX 预处理
    Mat src = imread("../src/20210803173137.jpg");
    yolox.input_height = 640;
    yolox.input_width = 640;
    int ori_width = src.rows;
    int ori_height = src.cols;
    int ori_depth = src.channels();
    //1. resize
    auto keep_resize_img = resizeKeepAspectRatio(src);
    auto resize_img = resizeImg(src);
    Mat process_img = preprocess_img(src, yolox.input_width, yolox.input_height);
    
    //2. expand dimension
    int size_1[4] = { 1, ori_width, ori_height, ori_depth };
    Mat output_(4, size_1, src.type(), src.data);

    cout << "output_: " << output_.size << endl;


    imshow("Test", src);
    imshow("keep_resize", keep_resize_img);
    imshow("resize", resize_img);
    imshow("preprocess", process_img);
    waitKey(0);
    destroyAllWindows();
    return 0;
}
