/*
实现NMS算法
*/
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

typedef struct Bbox{
    int x1;
    int y1;
    int x2;
    int y2;
    float score;
}Bbox;


bool sort_score(Bbox box1, Bbox box2){
    return (box1.score > box2.score);
}

//计算iou
float cal_iou(Bbox box1, Bbox box2) {
    auto max_x = max(box1.x1, box2.x1);
    auto max_y = max(box1.y1, box2.y1);

    auto min_x = min(box1.x2, box2.x2);
    auto min_y = min(box1.y2, box2.y2);


    //没有重叠的情况
    if (min_x <= max_x || min_y <= max_y) {
        return 0;
    }
    //计算重叠面积
    float over_area = (min_x - max_x) * (min_y - max_y);
    float area_a = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    float area_b = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    float iou = over_area / (area_a + area_b - over_area);
    return iou;
}

/*
计算NMS:
1. 获取当前类别下所有box的信息
2. 根据confident从大到小进行排序
3. 计算最大confidence对应的bbox与剩下的bbox的IOU，移除大于IOU阈值的bbox
4. 对剩下的bbox重复进行2、3步，直到不能再移除bbox为止
*/

vector<Bbox> nms(vector<Bbox>& vec_boxs, float threshold) {
    vector<Bbox>  res;
    while (vec_boxs.size() > 0)
    {
        sort(vec_boxs.begin(), vec_boxs.end(), sort_score);
        res.push_back(vec_boxs[0]);
        for (int i = 0; i < vec_boxs.size() - 1; i++)
        {
            float iou_value = cal_iou(vec_boxs[0], vec_boxs[i + 1]);
            if (iou_value > threshold)
            {
                vec_boxs.erase(vec_boxs.begin()+i + 1);
            }
        }
        vec_boxs.erase(vec_boxs.begin());  // res 已经保存，所以可以将最大的删除了

    }
    return res;
}

int main() {
    cout << "init two bounding box: bounding_box1 and bounding_box2." << endl;
    Bbox box1 = { 10, 10, 30, 30, 0.9 };
    Bbox box2 = { 13, 13, 34, 34, 0.6 };
    vector<Bbox> bboxs = { box1, box2 };
    vector<Bbox> result = nms(bboxs, 0.5);
    cout << "finish. The number of boxes is: " << result.size() << endl;
    for (int i = 0; i < result.size(); i++) {
        Bbox result_box = result[i];
        auto box_x1 = result_box.x1;
        auto box_y1 = result_box.y1;
        auto box_x2 = result_box.x2;
        auto box_y2 = result_box.y2;
        auto confidence = result_box.score;
        cout << "box (x1, y1): " << box_x1 << "," << box_y1 
            <<" (x1, y2):"<< box_x2<<", " << box_y2 <<" confidence: "<< confidence << endl;
    }
    return 0;
}
