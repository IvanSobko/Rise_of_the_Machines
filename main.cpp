#include "ObjectDetector.h"

#include <iostream>
#include <map>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

std::vector<std::string> get_file_list(const std::string& path)
{
    std::vector<std::string> m_file_list;
    if (fs::is_directory(path)) {
        fs::path apk_path(path);
        fs::directory_iterator end;

        for (fs::directory_iterator i(apk_path); i != end; ++i) {
            const fs::path cp = (*i);
            if (cp.extension() == ".jpg" || cp.extension() == ".jpeg") {
                m_file_list.push_back(cp.string());
            }
        }
    }
    return m_file_list;
}

void preprocessData() {
    std::ofstream fileLabels;
    fileLabels.open("../metrics/train_data.csv");
    fileLabels << "Image_name, Correct label\n";
    fs::create_directory("../images/train");
    fs::create_directory("../images/test");
    const auto &briskData = get_file_list("../images/brisk_dataset");
    const auto &noneData = get_file_list("../images/none_dataset");
    const auto &surfData = get_file_list("../images/surf_dataset");
    int num = 0;
    for (size_t i = 0; i < briskData.size() * 0.8; i++) {
        auto newfilename = "../images/train/" + std::to_string(num++) + ".jpg";
        fs::copy_file(briskData[i], newfilename, fs::copy_option::overwrite_if_exists);
        fileLabels << newfilename << "," << 1 << "\n";
    }
    for (size_t i = 0; i < surfData.size() * 0.8; i++) {
        auto newfilename = "../images/train/" + std::to_string(num++) + ".jpg";
        fs::copy_file(surfData[i], newfilename, fs::copy_option::overwrite_if_exists);
        fileLabels << newfilename << "," << 2 << "\n";
    }
    for (size_t i = 0; i < noneData.size() * 0.8; i++) {
        auto newfilename = "../images/train/" + std::to_string(num++) + ".jpg";
        fs::copy_file(noneData[i], newfilename, fs::copy_option::overwrite_if_exists);
        fileLabels << newfilename << "," << 0 << "\n";
    }
    num = 0;
    for (size_t i = briskData.size() * 0.8 - 1; i < briskData.size(); i++) {
        auto newfilename = "../images/test/" + std::to_string(num++) + ".jpg";
        fs::copy_file(briskData[i], newfilename, fs::copy_option::overwrite_if_exists);
        fileLabels << newfilename << "," << 1 << "\n";
    }
    for (size_t i = surfData.size() * 0.8 - 1; i < surfData.size() * 0.8; i++) {
        auto newfilename = "../images/test/" + std::to_string(num++) + ".jpg";
        fs::copy_file(surfData[i], newfilename, fs::copy_option::overwrite_if_exists);
        fileLabels << newfilename << "," << 2 << "\n";
    }
    for (size_t i = noneData.size() * 0.8 - 1; i < noneData.size(); i++) {
        auto newfilename = "../images/test/" + std::to_string(num++) + ".jpg";
        fs::copy_file(noneData[i], newfilename, fs::copy_option::overwrite_if_exists);
        fileLabels << newfilename << "," << 0 << "\n";
    }
    fileLabels.close();
}

std::map<std::string, int> read()
{
    std::ifstream fin;
    std::string line;
    std::map<std::string, int> data;
    // Open an existing file
    fin.open("../metrics/train_data.csv");
    std::getline(fin, line, '\n');
    while(std::getline(fin, line, '\n')) {
        auto it = line.find(",");
        auto filename = line.substr(0, it);
        auto type = std::stoi(line.substr(it + 1));

        data[filename] = type;
    }
    fin.close();
    return data;
}

void trainClassificator() {
    std::cout << "Getting train and test files..." << std::endl;
    const auto &trainFileNames = get_file_list("../images/train");
    const auto &testFileNames = get_file_list("../images/test");
    ObjectDetector detectorBrisk(true), detectorSurf(false);
    std::vector<cv::Mat> trainImage;
    std::vector<cv::Mat> testImage;
    std::vector<int> labelsTrain, labelsTest;
    auto data = read();
    for (int i = 0; i < trainFileNames.size(); i++) {
        auto train = cv::imread(trainFileNames[i], cv::IMREAD_GRAYSCALE);

        if (train.empty()) {
            std::cerr << "Warning: Could not train image!" << std::endl;
            continue;
        }
        labelsTrain.push_back(data[trainFileNames[i]]);
        trainImage.push_back(train);
    }

    cv::Ptr<cv::ml::SVM> SVMbrisk, SVMsurf;
    cv::Ptr<cv::ml::DTrees> TREEBrisk, TREESurf;
    detectorBrisk.train(trainImage, labelsTrain, SVMbrisk);
    detectorSurf.train(trainImage, labelsTrain, SVMsurf);
    detectorBrisk.train(trainImage, labelsTrain, TREEBrisk);
    detectorSurf.train(trainImage, labelsTrain, TREESurf);


    for (int i = 0; i < testFileNames.size(); i++) {
        auto test = cv::imread(testFileNames[i], cv::IMREAD_GRAYSCALE);

        if (test.empty()) {
            std::cerr << "Warning: Could not train image!" << std::endl;
            continue;
        }
        labelsTest.push_back(data[testFileNames[i]]);
        testImage.push_back(test);
    }
    std::cout << "\nBrisk and SVM metrics:" << std::endl;
    detectorBrisk.predict(testImage, labelsTest, SVMbrisk);
    std::cout << "\nSURF and SVM metrics:" << std::endl;
    detectorSurf.predict(testImage, labelsTest, SVMsurf);
    std::cout << "\nBrisk and DTREE metrics:" << std::endl;
    detectorBrisk.predict(testImage, labelsTest, TREEBrisk);
    std::cout << "\nSURF and DTREE metrics:" << std::endl;
    detectorSurf.predict(testImage, labelsTest, TREESurf);

    detectorBrisk.predictVideo("../Video/result_money_svm_brisk.avi", cv::VideoCapture("../Video/money.mp4"), SVMbrisk);
    detectorSurf.predictVideo("../Video/result_money_svm_surf.avi", cv::VideoCapture("../Video/money.mp4"), SVMsurf);

    detectorBrisk.predictVideo("../Video/result_money_tree_brisk.avi", cv::VideoCapture("../Video/money.mp4"), TREEBrisk);
    detectorSurf.predictVideo("../Video/result_money_tree_surf.avi", cv::VideoCapture("../Video/money.mp4"), TREESurf);

    detectorBrisk.predictVideo("../Video/result_sneaker_svm_brisk.avi", cv::VideoCapture("../Video/Sneaker.mp4"), SVMbrisk);
    detectorSurf.predictVideo("../Video/result_sneaker_svm_surf.avi", cv::VideoCapture("../Video/Sneaker.mp4"), SVMsurf);

    detectorBrisk.predictVideo("../Video/result_sneaker_tree_brisk.avi", cv::VideoCapture("../Video/Sneaker.mp4"), TREEBrisk);
    detectorSurf.predictVideo("../Video/result_sneaker_tree_surf.avi", cv::VideoCapture("../Video/Sneaker.mp4"), TREESurf);
}

int main() {
    preprocessData();
    trainClassificator();
    return 0;
}
