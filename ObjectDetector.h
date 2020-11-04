//
// Created by Sveta Morkva on 10/3/20.
//

#ifndef KPI_LAB2_OBJECTDETECTOR_H
#define KPI_LAB2_OBJECTDETECTOR_H

#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>

class ObjectDetector {
public:
    ObjectDetector(bool brisk);

    //SVM
    void train(std::vector<cv::Mat> images, std::vector<int> labels, cv::Ptr<cv::ml::SVM> &model);
    //DTree
    void train(std::vector<cv::Mat> images, std::vector<int> labels, cv::Ptr<cv::ml::DTrees> &model);
    void predict(std::vector<cv::Mat> images, std::vector<int> labels, cv::ml::StatModel *model);
    void predictVideo(const std::string &filename, cv::VideoCapture video, cv::ml::StatModel *model);


private:
    cv::Mat getDescriptors(std::vector<cv::Mat> images);

    int num_clusters = 8;
    int num_descr = 30;
    bool mBriskDescr = true;
    cv::Rect curImgBoundRect;
};


#endif //KPI_LAB2_OBJECTDETECTOR_H
