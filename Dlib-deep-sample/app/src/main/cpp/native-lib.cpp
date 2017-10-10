
#include <string>
#include "native-lib.h"
#include "opencv2/opencv.hpp"
#include "fps.h"
#include "DeepFaceDetection.h"
//OpenCV includes:
Fps *fpsCounter;
DeepFaceDetection *m_net;
JNIEXPORT void JNICALL
Java_org_uelordi_deepsamples_dlib_JniManager_init(JNIEnv *env, jclass type,
                                                  jstring neuralNet_,
                                                  jstring weights_filename_) {
    const char *netFile = env->GetStringUTFChars(neuralNet_, 0);
    const char *weightsFile = env->GetStringUTFChars(weights_filename_, 0);

        m_net = new DeepFaceDetection();
        m_net->init(netFile, weightsFile);
}
JNIEXPORT void JNICALL
Java_org_uelordi_deepsamples_dlib_JniManager_process(JNIEnv *env, jclass type, jlong colorImage,
                                                       jlong greyImage) {


    cv::Mat  &colorMat =*(cv::Mat *) colorImage;
    cv::Mat  &greyMat =*(cv::Mat *) greyImage;

    /*
     * make Processing
     */

    /*
     * check the fps:
     */
    float fps = fpsCounter->checkFps();
    cv::putText(colorMat,
                fpsCounter->getFpsText().c_str(),
                cv::Point(15,15),
                CV_FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(0,255,0));
    /*
     * finish chek fps:
     */
}

JNIEXPORT void JNICALL
Java_org_uelordi_deepsamples_dlib_JniManager_start(JNIEnv *env, jclass type) {

    // TODO define the starting variables to process
    fpsCounter = new Fps();
    fpsCounter->start();
}

JNIEXPORT void JNICALL
Java_org_uelordi_deepsamples_dlib_JniManager_stop(JNIEnv *env, jclass type) {

    // TODO delete every object when you finished
    delete(m_net);
}