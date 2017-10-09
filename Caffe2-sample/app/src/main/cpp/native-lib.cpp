#include "native-lib.h"
#include "fps.h"


#define CAFFE2_USE_LITE_PROTO 1
#include <caffe2/core/predictor.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/timer.h>
#include "caffe2/core/init.h"


#include "opencv2/core.hpp"
#define IMG_H 227
#define IMG_W 227
#define IMG_C 3
#define MAX_DATA_SIZE IMG_H * IMG_W * IMG_C
#define alog(...) __android_log_print(ANDROID_LOG_ERROR, "F8DEMO", __VA_ARGS__);
#define PROTOBUF_USE_DLLS 1
//#define min(a,b) ((a) > (b)) ? (b) : (a)
//#define max(a,b) ((a) > (b)) ? (a) : (b)

//

#include "classes.h"

static caffe2::NetDef _initNet, _predictNet;
static caffe2::Predictor *_predictor;
static char raw_data[MAX_DATA_SIZE];
static float input_data[MAX_DATA_SIZE];
static caffe2::Workspace ws;
using namespace cv;
//OpenCV includes:
Fps *fpsCounter;
JNIEXPORT void JNICALL
Java_org_uelordi_deepsamples_caffe2_JniManager_init(JNIEnv *env, jclass type,   jobject assetManager) {

    // TODO
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    alog("Attempting to load protobuf netdefs...");
    loadToNetDef(mgr, &_initNet,   "squeeze_init_net.pb");
    loadToNetDef(mgr, &_predictNet,"squeeze_predict_net.pb");
    alog("done.");
    alog("Instantiating predictor...");
    _predictor = new caffe2::Predictor(_initNet, _predictNet);
    alog("done.")
}
JNIEXPORT void JNICALL
Java_org_uelordi_deepsamples_caffe2_JniManager_process(JNIEnv *env, jclass type, jlong colorImage,
                                                       jlong greyImage) {

//    // TODO
    cv::Mat  &inMat =*(cv::Mat *) colorImage;
//    int h = inMat.rows;
//    int w = inMat.cols;
//    auto h_offset = max(0, (h - IMG_H) / 2);
//    auto w_offset = max(0, (w - IMG_W) / 2);
//
//    auto iter_h = IMG_H;
//    auto iter_w = IMG_W;
//    if (h < IMG_H) {
//        iter_h = h;
//    }
//    if (w < IMG_W) {
//        iter_w = w;
//    }
    caffe2::TensorCPU input;
//    if (infer_HWC) {
//        input.Resize(std::vector<int>({IMG_H, IMG_W, IMG_C}));
//    } else {
//        input.Resize(std::vector<int>({1, IMG_C, IMG_H, IMG_W}));
//    }
//    memcpy(input.mutable_data<float>(), input_data, IMG_H * IMG_W * IMG_C * sizeof(float));
//    caffe2::Predictor::TensorVector input_vec{&input};
//    caffe2::Predictor::TensorVector output_vec;
//    caffe2::Timer t;
//    t.Start();
//    _predictor->run(input_vec, &output_vec);
//    float fps = 1000/t.MilliSeconds();
//    total_fps += fps;
//    avg_fps = total_fps / iters_fps;
//    total_fps -= avg_fps;
////    float fps = fpsCounter->checkFps();
//    std::stringstream ss;
//    ss.precision(4);
//    ss << "FPS "<< fps;
//    cv::putText(inMat,ss.str().c_str(), cv::Point(15,15),  CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0));
}

JNIEXPORT void JNICALL
Java_org_uelordi_deepsamples_caffe2_JniManager_start(JNIEnv *env, jclass type) {

    // TODO
    fpsCounter = new Fps();
    fpsCounter->start();

}

JNIEXPORT void JNICALL
Java_org_uelordi_deepsamples_caffe2_JniManager_stop(JNIEnv *env, jclass type) {

    // TODO

}

void loadToNetDef(AAssetManager* mgr, caffe2::NetDef* net, const char *filename) {
    AAsset* asset = AAssetManager_open(mgr, filename, AASSET_MODE_BUFFER);
    assert(asset != nullptr);
    const void *data = AAsset_getBuffer(asset);
    assert(data != nullptr);
    off_t len = AAsset_getLength(asset);
    assert(len != 0);
    if (!net->ParseFromArray(data, len)) {
        alog("Couldn't parse net from data.\n");
    }
    AAsset_close(asset);
}
