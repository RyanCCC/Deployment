#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

struct YOLOX {
    int input_width;
    int input_height;
} yolox;


int main()
{
    //YOLOX 预处理
    Mat src = imread("../src/20210803173137.jpg");
    yolox.input_height = 640;
    yolox.input_width = 640;
    //1. resize
    
    //2. expand dimension


    imshow("Test", src);
    waitKey(0);
    destroyAllWindows();
    return 0;
}
