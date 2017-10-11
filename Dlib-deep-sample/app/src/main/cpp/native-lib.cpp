
#include <string>
#include "native-lib.h"
#include "opencv2/opencv.hpp"
#include "fps.h"
#include "DeepFaceDetection.h"
#include <mutex>
#include <condition_variable>
//OpenCV includes:
Fps *fpsCounter;
DeepFaceDetection *m_net;
bool mIsTheThreadStart;
cv::Mat shared_camera_image;
std::thread * workerThread;
std::mutex mCameraMutex;
std::unique_lock<std::mutex> lock(mCameraMutex);
std::condition_variable mCameraCondition;
void process ();
JNIEXPORT void JNICALL
Java_org_uelordi_deepsamples_dlib_JniManager_init(JNIEnv *env, jclass type,
                                                  jstring neuralNet_,
                                                  jstring weights_filename_) {
    const char *netFile = env->GetStringUTFChars(neuralNet_, 0);
    const char *weightsFile = env->GetStringUTFChars(weights_filename_, 0);
        mIsTheThreadStart = false;
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
//    mCameraMutex.lock();
        colorMat.copyTo(shared_camera_image);
        process();
//    mCameraMutex.unlock();
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
    mIsTheThreadStart = true;
//    workerThread = new std::thread(process);
}

JNIEXPORT void JNICALL
Java_org_uelordi_deepsamples_dlib_JniManager_stop(JNIEnv *env, jclass type) {

    // TODO delete every object when you finished
    workerThread->join();
    delete(m_net);
}
void process(void) {
    cv::Mat processingImage;
    while(mIsTheThreadStart) {
//        mCameraMutex.
//        mCameraCondition.wait(mCameraMutex);
        cv::cvtColor(shared_camera_image, processingImage, CV_BGRA2RGB);
        std::vector<cv::Rect> faces;
        m_net->process(processingImage, &faces);
        m_net->drawFaces(&shared_camera_image, faces);
    }
}