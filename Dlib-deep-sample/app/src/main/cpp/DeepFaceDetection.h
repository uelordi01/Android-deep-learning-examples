//
// Created by uelordi on 10/10/17.
//

#ifndef DLIB_DEEP_SAMPLE_DEEPFACEDETECTION_H
#define DLIB_DEEP_SAMPLE_DEEPFACEDETECTION_H

#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

class DeepFaceDetection {
public:
    bool init(std::string deepNetFile, std::string weightsFile);
    int process(const cv::Mat & in, std::vector<cv::Rect> *faces);
    void drawFaces(cv::Mat *inOut, std::vector<cv::Rect> faces);
private:
    // TODO put the apply CLAHE function in the sources
    void applyCLAHE(const cv::Mat &src, cv::Mat *dst);
    // Detection network structure
    template <long num_filters, typename SUBNET> using con5d =
    dlib::con<num_filters, 5, 5, 2, 2, SUBNET>;
    template <long num_filters, typename SUBNET> using con5 =
    dlib::con<num_filters, 5, 5, 1, 1, SUBNET>;
    template <typename SUBNET> using downsampler =
    dlib::relu<dlib::affine<con5d<32,
            dlib::relu<dlib::affine<con5d<32,
                    dlib::relu<dlib::affine<con5d<16, SUBNET>>>>>>>>>;
    template <typename SUBNET> using rcon5 =
    dlib::relu<dlib::affine<con5<45, SUBNET>>>;
    using det_net =
    dlib::loss_mmod<
            dlib::con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<
                                     downsampler<
                                     dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>>;

    det_net m_detModel;
    dlib::matrix<dlib::rgb_pixel> m_res;
    cv::Ptr<cv::CLAHE> m_clahe;
    std::vector<cv::Mat> m_imageChannels;
    cv::Mat clahe_img1;
    cv::Mat clahe_img2;


};


#endif //DLIB_DEEP_SAMPLE_DEEPFACEDETECTION_H
