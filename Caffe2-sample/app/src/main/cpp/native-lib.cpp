#include "native-lib.h"
#include "fps.h"
//#include <dlib/dnn.h>
//#include <dlib/gui_widgets.h>
//#include <dlib/clustering.h>
//#include <dlib/string.h>
//#include <dlib/image_io.h>
//#include <dlib/image_processing/frontal_face_detector.h>
//#define CAFFE2_USE_LITE_PROTO 1
#include <caffe2/core/predictor.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/timer.h>
#include "caffe2/core/init.h"


#include "opencv2/imgproc.hpp"


#define IMG_H 227
#define IMG_W 227
#define IMG_C 3
#define MAX_DATA_SIZE IMG_H * IMG_W * IMG_C
#define alog(...) __android_log_print(ANDROID_LOG_ERROR, "F8DEMO", __VA_ARGS__);
#define PROTOBUF_USE_DLLS 1

bool infer_HWC = false;
#define min(a,b) ((a) > (b)) ? (b) : (a)
#define max(a,b) ((a) > (b)) ? (a) : (b)

//

#include "classes.h"

static caffe2::NetDef _initNet, _predictNet;
static caffe2::Predictor *_predictor;
static char raw_data[MAX_DATA_SIZE];
static float  input_data [MAX_DATA_SIZE];
static caffe2::Workspace ws;
using namespace cv;
//OpenCV includes:
Fps *fpsCounter;
float avg_fps = 0.0;
float total_fps = 0.0;
int iters_fps = 10;
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
    alog("processing");
    int h = inMat.rows;
    int w = inMat.cols;
    cv::Mat process_image;
    auto h_offset = max(0, (h - IMG_H) / 2);
    auto w_offset = max(0, (w - IMG_W) / 2);
    cv::resize(inMat, process_image, cv::Size(IMG_H, IMG_W));
    cv::cvtColor(process_image, process_image, CV_BGRA2BGR);
//    input_data = convertToFloatImage(process_image);
//
    auto iter_h = IMG_H;
    auto iter_w = IMG_W;
    if (h < IMG_H) {
        iter_h = h;
    }
    if (w < IMG_W) {
        iter_w = w;
    }
    caffe2::TensorCPU input;
    if (infer_HWC) {
        input.Resize(std::vector<int>({IMG_H, IMG_W, IMG_C}));
    } else {
        input.Resize(std::vector<int>({1, IMG_C, IMG_H, IMG_W}));
    }

//    memcpy(input.mutable_data<float>(), process_image.data, IMG_H * IMG_W * IMG_C * sizeof(float));
//    caffe2::Predictor::TensorVector input_vec{&input};
//    caffe2::Predictor::TensorVector output_vec;
//    caffe2::Timer t;
//    t.Start();
//    _predictor->run(input_vec, &output_vec);
//    float fps = 1000/t.MilliSeconds();
//
//    constexpr int k = 5;
//    float max[k] = {0};
//    int max_index[k] = {0};
//    int capacity = output_vec.capacity();
//    // Find the top-k results manually.
//    if ( capacity > 0) {
//        for (auto output : output_vec) {
//            int output_size =  output->size();
//            for (auto i = 0; i < output_size; ++i) {
//                for (auto j = 0; j < k; ++j) {
//                    if (output->template data<float>()[i] > max[j]) {
//                        for (auto _j = k - 1; _j > j; --_j) {
//                            max[_j - 1] = max[_j];
//                            max_index[_j - 1] = max_index[_j];
//                        }
//                        max[j] = output->template data<float>()[i];
//                        max_index[j] = i;
//                        goto skip;
//                    }
//                }
//                skip:;
//            }
//        }
//    }
//
////    total_fps += fps;
////    avg_fps = total_fps / iters_fps;
////    total_fps -= avg_fps;
//////    float fps = fpsCounter->checkFps();
//    std::stringstream ss;
//    ss.precision(4);
//    ss << "FPS "<< fps;
//    cv::putText(inMat,ss.str().c_str(), cv::Point(15,15),  CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0));
//    int ystartingPoint = 30;
//
//    for (auto j = 0; j < k; ++j) {
//        std::stringstream ss1;
//        ss1 << j << ": " << imagenet_classes[max_index[j]] << " - " << max[j] * 100 << "%\n";
//        cv::putText(inMat, ss1.str(), cv::Point(15, ystartingPoint), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,0));
//    }
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

//float * convertToFloatImage(const Mat & img) {
//    static float data[MAX_DATA_SIZE];
//    cv::Mat res;
//    img.convertTo(res, CV_32FC3);
//    for(int i = 0; i< img.rows; i++) {
//        for(int j=0; j<img.cols; j++) {
//            Vec3b intensity = res.at<Vec3b>(j, i);
//            float blue = static_cast<float>(intensity.val[0]/255);
//            float green = static_cast<float>(intensity.val[1]/255);
//            float red = static_cast<float>(intensity.val[2]/255);
//            alog("red %4.2d green %4.2d blue %4.2d", red, green, blue);
//        }
//    }
//}

JNIEXPORT jstring JNICALL
Java_org_uelordi_deepsamples_caffe2_JniManager_processYUVFrame(JNIEnv *env, jobject instance,
                                                               jint h, jint w, jbyteArray Y_,
                                                               jbyteArray U_, jbyteArray V_,
                                                               jint rowStride, jint pixelStride,
                                                               jboolean r_hwc) {


    // TODO
    if (!_predictor) {
        return env->NewStringUTF("Loading...");
    }
    jsize Y_len = env->GetArrayLength(Y_);
    jbyte * Y_data = env->GetByteArrayElements(Y_, 0);
    assert(Y_len <= MAX_DATA_SIZE);
    jsize U_len = env->GetArrayLength(U_);
    jbyte * U_data = env->GetByteArrayElements(U_, 0);
    assert(U_len <= MAX_DATA_SIZE);
    jsize V_len = env->GetArrayLength(V_);
    jbyte * V_data = env->GetByteArrayElements(V_, 0);
    assert(V_len <= MAX_DATA_SIZE);

#define min(a,b) ((a) > (b)) ? (b) : (a)
#define max(a,b) ((a) > (b)) ? (a) : (b)

    auto h_offset = max(0, (h - IMG_H) / 2);
    auto w_offset = max(0, (w - IMG_W) / 2);

    auto iter_h = IMG_H;
    auto iter_w = IMG_W;
    if (h < IMG_H) {
        iter_h = h;
    }
    if (w < IMG_W) {
        iter_w = w;
    }

    for (auto i = 0; i < iter_h; ++i) {
        jbyte* Y_row = &Y_data[(h_offset + i) * w];
        jbyte* U_row = &U_data[(h_offset + i) / 4 * rowStride];
        jbyte* V_row = &V_data[(h_offset + i) / 4 * rowStride];
        for (auto j = 0; j < iter_w; ++j) {
            // Tested on Pixel and S7.
            char y = Y_row[w_offset + j];
            char u = U_row[pixelStride * ((w_offset+j)/pixelStride)];
            char v = V_row[pixelStride * ((w_offset+j)/pixelStride)];

            float b_mean = 104.00698793f;
            float g_mean = 116.66876762f;
            float r_mean = 122.67891434f;

            auto b_i = 0 * IMG_H * IMG_W + j * IMG_W + i;
            auto g_i = 1 * IMG_H * IMG_W + j * IMG_W + i;
            auto r_i = 2 * IMG_H * IMG_W + j * IMG_W + i;

            if (infer_HWC) {
                b_i = (j * IMG_W + i) * IMG_C;
                g_i = (j * IMG_W + i) * IMG_C + 1;
                r_i = (j * IMG_W + i) * IMG_C + 2;
            }
/*
  R = Y + 1.402 (V-128)
  G = Y - 0.34414 (U-128) - 0.71414 (V-128)
  B = Y + 1.772 (U-V)
 */
            input_data[r_i] = -r_mean + (float) ((float) min(255., max(0., (float) (y + 1.402 * (v - 128)))));
            input_data[g_i] = -g_mean + (float) ((float) min(255., max(0., (float) (y - 0.34414 * (u - 128) - 0.71414 * (v - 128)))));
            input_data[b_i] = -b_mean + (float) ((float) min(255., max(0., (float) (y + 1.772 * (u - v)))));

        }
    }

    caffe2::TensorCPU input;
    if (infer_HWC) {
        input.Resize(std::vector<int>({IMG_H, IMG_W, IMG_C}));
    } else {
        input.Resize(std::vector<int>({1, IMG_C, IMG_H, IMG_W}));
    }
    memcpy(input.mutable_data<float>(), input_data, IMG_H * IMG_W * IMG_C * sizeof(float));
    caffe2::Predictor::TensorVector input_vec{&input};
    caffe2::Predictor::TensorVector output_vec;
    caffe2::Timer t;
    t.Start();
    _predictor->run(input_vec, &output_vec);
    float fps = 1000/t.MilliSeconds();
    total_fps += fps;
    avg_fps = total_fps / iters_fps;
    total_fps -= avg_fps;

    constexpr int k = 5;
    float max[k] = {0};
    int max_index[k] = {0};
    // Find the top-k results manually.
    if (output_vec.capacity() > 0) {
        for (auto output : output_vec) {
            for (auto i = 0; i < output->size(); ++i) {
                for (auto j = 0; j < k; ++j) {
                    if (output->template data<float>()[i] > max[j]) {
                        for (auto _j = k - 1; _j > j; --_j) {
                            max[_j - 1] = max[_j];
                            max_index[_j - 1] = max_index[_j];
                        }
                        max[j] = output->template data<float>()[i];
                        max_index[j] = i;
                        goto skip;
                    }
                }
                skip:;
            }
        }
    }
    std::ostringstream stringStream;
    stringStream << avg_fps << " FPS\n";

    for (auto j = 0; j < k; ++j) {
        stringStream << j << ": " << imagenet_classes[max_index[j]] << " - " << max[j] * 100 << "%\n";
    }
    alog("%s", stringStream.str().c_str());
    return env->NewStringUTF(stringStream.str().c_str());
}