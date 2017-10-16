//
// Created by uelordi on 10/10/17.
//

#include "DeepFaceDetection.h"
#include "logger.h"
#define LOG_TAG "DeepFaceDetection"
bool DeepFaceDetection::init(std::string deepNetFile, std::string weightsFile) {

    dlib::deserialize(deepNetFile) >> m_detModel;
    m_imageChannels.resize(3);
    m_clahe = cv::createCLAHE();
    m_clahe->setClipLimit(4);
    return false;
}
int DeepFaceDetection::process(const cv::Mat &  in, std::vector<cv::Rect> *faces) {

    cv::Mat processImg;
    applyCLAHE(in, &processImg);
    cv::resize(processImg, processImg, cv::Size(160,120));
    dlib::cv_image<dlib::rgb_pixel> dlib_image(in);
    dlib::assign_image(m_res, dlib_image);
    LOGV("process");
    auto dets = m_detModel(m_res);
    LOGV("process1");
    unsigned long nDets = dets.size();
    if (nDets == 0) {
        LOGD("No detections found");
        return 0;
    }
//    landmarks->resize(nDets);
    faces->resize(nDets);
    int i = 0;
    cv::Point2f p;
    for (auto&& d : dets) {
        int w = abs(d.rect.right() - d.rect.left());
        int h = abs(d.rect.bottom() - d.rect.top());
        (*faces)[i] = cv::Rect(d.rect.left(), d.rect.top(), w, h);
    }
}

void DeepFaceDetection::drawFaces(cv::Mat *inOut, std::vector<cv::Rect> faces) {
    for(int i = 0; i< faces.size(); i++) {
        cv::rectangle(*inOut, faces[i].tl(), faces[i].br(), cv::Scalar(0, 255, 0), 1, 8, 0);
    }
}
// TODO put the apply CLAHE function in the sources
void DeepFaceDetection::applyCLAHE(const cv::Mat &src, cv::Mat *dst) {
    cv::Mat clahe_img;
    cv::Mat clahe_img2;
    cvtColor(src, clahe_img, CV_BGR2Lab);

    // Extract the L channel
    cv::split(clahe_img, m_imageChannels);

    // Apply the CLAHE algorithm to the L channel
    m_clahe->apply(m_imageChannels[0], clahe_img2);

    // Merge the the color planes back into an Lab image
    clahe_img2.copyTo(m_imageChannels[0]);
    merge(m_imageChannels, clahe_img);

    // Convert back to RGB
    cvtColor(clahe_img, *dst, CV_Lab2BGR);
}

