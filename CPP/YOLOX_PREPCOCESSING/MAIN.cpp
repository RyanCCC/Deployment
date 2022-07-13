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

    imshow("square", square(roi));

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
    
    //2. expand dimension


    imshow("Test", src);
    imshow("keep_resize", keep_resize_img);
    imshow("resize", resize_img);
    waitKey(0);
    destroyAllWindows();
    return 0;
}
