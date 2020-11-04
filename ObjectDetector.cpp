//
// Created by Sveta Morkva on 10/3/20.
//

#include "ObjectDetector.h"
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <boost/filesystem.hpp>
#include "opencv2/viz.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cvstd.hpp>

using namespace cv;
using namespace cv::ml;

namespace fs = boost::filesystem;

ObjectDetector::ObjectDetector(bool brisk) : mBriskDescr(brisk) {}

void ObjectDetector::train(std::vector<Mat> images, std::vector<int> labels, Ptr<ml::SVM> &model) {
    Mat samples = getDescriptors(images);
    std::string filename = "../trained_model/trained_";

//    if(fs::exists(filename)){
//        std::cout << "Found trained model: " << filename << ", loading it." << std::endl;
//        svm = SVM::load(filename);
//        return;
//    }
    model = SVM::create();
    model->setType(ml::SVM::C_SVC);
    model->setKernel(ml::SVM::LINEAR);
    model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    model->train(samples, ROW_SAMPLE, labels);
    std::cout << "Trained "<< (mBriskDescr ? "brisk" : "surf") << " SVM model." <<  std::endl;
    filename += mBriskDescr ? "svm_brisk.xml" : "svm_surf.xml";
    model->save(filename);
}

void ObjectDetector::train(std::vector<cv::Mat> images, std::vector<int> labels, cv::Ptr<cv::ml::DTrees> &model) {
    Mat samples = getDescriptors(images);
    std::string filename = "../trained_model/trained_";

//    if(fs::exists(filename)){
//        std::cout << "Found trained model: " << filename << ", loading it." << std::endl;
//        svm = SVM::load(filename);
//        return;
//    }
    model = DTrees::create();
    model->setCVFolds(1);
    model->setMaxDepth(15);
    model->train(samples, ROW_SAMPLE, labels);
    std::cout << "Trained "<< (mBriskDescr ? "brisk" : "surf") << " DTREES model." <<  std::endl;
    filename += mBriskDescr ? "dtrees_brisk.xml" : "drees_surf.xml";
    model->save(filename);
}

void ObjectDetector::predict(std::vector<Mat> images, std::vector<int> labels, cv::ml::StatModel *model) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    Mat samples = getDescriptors(images);

    Mat predict_labels;
    model->predict(samples, predict_labels);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    printf("Time for detection and prediction: %f\n", time.count() / 1000000.0);
    int truePositivesS = 0, falsePositivesS = 0, truePositivesM = 0, falsePositivesM = 0,
            trueNegatives = 0, falseNegatives = 0, allS = 0, allM = 0;
    for (int i = 0; i < predict_labels.rows; i++) {
        if (labels[i] == 1) {
            predict_labels.at<float>(i, 0) == labels[i] ? truePositivesS++ : falsePositivesS++;
            allS++;
        } else if (labels[i] == 2) {
            predict_labels.at<float>(i, 0) == labels[i] ? truePositivesM++ : falsePositivesM++;
            allM++;
        } else {
            predict_labels.at<float>(i, 0) == labels[i] ? trueNegatives++ : falseNegatives++;
        }
    }
    printf("False positive sneaker: %d\nFalse positive money: %d\nFalse negative: %d\n", falsePositivesS, falsePositivesM,
           falseNegatives);
    double precisonS = truePositivesS * 1.0/(truePositivesS+falsePositivesS);
    double precisonM = truePositivesM * 1.0/(truePositivesM+falsePositivesM);
    double recallS = truePositivesS * 1.0/allS;
    double recallM = truePositivesM * 1.0/allM;
    double f1S = 2*precisonS*recallS/(precisonS+recallS);
    double f1M = 2*precisonM*recallM/(precisonM+recallM);
    printf("Precision sneaker: %f\nRecall sneaker: %f\nF1-score sneaker: %f\n", precisonS, recallS, f1S);
    printf("Precision money: %f\nRecall money: %f\nF1-score money: %f\n", precisonM, recallM, f1M);
}


cv::Mat ObjectDetector::getDescriptors(std::vector<Mat> images) {

    Ptr<DescriptorExtractor> extractor;
    if (mBriskDescr) {
        extractor = BRISK::create();
    } else {
        extractor = xfeatures2d::SURF::create();
    }

    Mat outSamples = Mat::zeros(images.size(), extractor->descriptorSize() * num_descr, CV_32FC1);
    for (int i = 0; i < images.size(); i++) {
        std::vector<KeyPoint> keypoints;
        cv::Mat descriptors;
        std::vector<int> clustNames;
        std::map<int, std::vector<Point2f>> clusters;
        auto img = images[i];
        extractor->detect(img, keypoints);
        extractor->compute(img, keypoints, descriptors);
        if (keypoints.size() < num_clusters) {
            continue;
        }
        cv::Mat points(keypoints.size(), descriptors.cols + 2, CV_32FC1);
        for (int z = 0; z < keypoints.size(); z++) {
            auto p = keypoints[z].pt;
            for (int j = 0; j < descriptors.cols; j++) {
                points.at<float>(z, j) = mBriskDescr ? descriptors.at<uchar>(z, j) : descriptors.at<float>(z, j);
            }
            points.at<float>(z, descriptors.cols) = p.x;
            points.at<float>(z, descriptors.cols + 1) = p.y;
        }
        kmeans(points, num_clusters, clustNames,
               TermCriteria(TermCriteria::MAX_ITER, 10, 1.0), 3,
               KMEANS_PP_CENTERS);
        Mat img_keypoints;
        for (int j = 0; j < points.rows; j++) {
            Point2f p = {points.at<float>(j, descriptors.cols),
                         points.at<float>(j, descriptors.cols + 1)};
            clusters[clustNames[j]].push_back(p);
        }

        int biggestClusterNum = 0;
        int biggestClusterName;
        for (auto cl: clusters) {
            if (biggestClusterNum < cl.second.size()*0.95) {
                biggestClusterName = cl.first;
                biggestClusterNum = cl.second.size();
            } else {
                auto r1 = boundingRect(cl.second);
                auto r2 = boundingRect(clusters[biggestClusterName]);
                Point center_of_rect1 = (r1.br() + r1.tl())*0.5;
                Point center_of_rect2 = (r2.br() + r2.tl())*0.5;
                Point center_of_img = {img.cols/2, img.rows/2};
                double dist1 = cv::norm(center_of_img-center_of_rect1);
                double dist2 = cv::norm(center_of_img-center_of_rect2);
                if (dist1 < dist2) {
                    biggestClusterName = cl.first;
                    biggestClusterNum = cl.second.size();
                }
            }

        }
        int num = 0;
        for (int z = 0; z < descriptors.rows; z++) {
            if (clustNames[z] == biggestClusterName) {
                for (int j = 0; j < descriptors.cols; j++) {
                    outSamples.at<float>(i, num*descriptors.cols+j) = points.at<float>(z, j);
                }
                if (++num == num_descr) {
                    break;
                }
            }
        }
        curImgBoundRect = boundingRect(clusters[biggestClusterName]);
//        rectangle(img, r, Scalar(255, 0, 0));
//        imshow("SURF Keypoints", img);
//        waitKey();
    }
    return outSamples;
}

void ObjectDetector::predictVideo(const std::string &filename, cv::VideoCapture video, cv::ml::StatModel *model) {
    if (!video.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return;
    }
    std::cout << "Processing video: " << filename << std::endl;
    std::vector<cv::Mat> resultVideo;
    while (true) {
        Mat frame;
        video >> frame;
        if (frame.empty()) {
            break;
        }

        cvtColor(frame, frame, COLOR_BGR2GRAY);
        auto img = getDescriptors({frame});
        float label = model->predict(img);

        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);

        Scalar color;
        std::string overlay;
        if (label == 0) {
            color = cv::viz::Color::red();
            overlay = "None objects were found.";
        }else if (label == 1) {
            color = cv::viz::Color::green();
            overlay = "Sneaker object was found.";
        }else {
            color = cv::viz::Color::green();
            overlay = "Money object was found.";
        }
        if (label != 0) {
            rectangle(frame, curImgBoundRect, cv::viz::Color::green(), 2);
        }

        auto point = Point(frame.cols / 2, frame.rows - frame.rows /4);
        cv::putText(frame, overlay, point, FONT_HERSHEY_PLAIN, 1.0, cv::viz::Color::white(), 2.0);
        resultVideo.push_back(frame);
    }

    VideoWriter writer = cv::VideoWriter();
    auto size = cv::Size(static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH)),
                         static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT)));

    //avi format
    if (!writer.open(filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, size)) {
        std::cout << "Error: cant open file to write" << std::endl;
        return;
    }

    for (const auto &frame: resultVideo) {
        writer.write(frame);
    }
}

