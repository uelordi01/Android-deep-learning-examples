
#include <string>
#include "native-lib.h"
//#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include "fps.h"
//OpenCV includes:
Fps *fpsCounter;
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
JNIEXPORT void JNICALL
Java_org_uelordi_deepsamples_dlib_JniManager_init(JNIEnv *env, jclass type) {

    // TODO
}
JNIEXPORT void JNICALL
Java_org_uelordi_deepsamples_dlib_JniManager_process(JNIEnv *env, jclass type, jlong colorImage,
                                                       jlong greyImage) {

    // TODO
    cv::Mat  &inMat =*(cv::Mat *) colorImage;
    float fps = fpsCounter->checkFps();
    std::stringstream ss;
    ss.precision(4);
    ss << "FPS "<< fps;
    cv::putText(inMat,ss.str().c_str(), cv::Point(15,15),  CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0));
}

JNIEXPORT void JNICALL
Java_org_uelordi_deepsamples_dlib_JniManager_start(JNIEnv *env, jclass type) {

    // TODO
    fpsCounter = new Fps();
    fpsCounter->start();

}

JNIEXPORT void JNICALL
Java_org_uelordi_deepsamples_dlib_JniManager_stop(JNIEnv *env, jclass type) {

    // TODO

}
