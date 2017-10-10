//
// Created by uelordi on 10/10/17.
//

#include "DeepFaceDetection.h"

bool DeepFaceDetection::init(std::string deepNetFile, std::string weightsFile) {

    dlib::deserialize(deepNetFile) >> m_detModel;

    return false;
}

void DeepFaceDetection::process(const cv::Mat & in) {
    dlib::cv_image<dlib::rgb_pixel> dlib_image(in);
    dlib::assign_image(m_res, in);
// TODO correct the neural network compilation problem
    auto dets = m_detModel(m_res);
//-------------------------

}

