//
// Created by uelordi on 6/10/17.
//

#ifndef CAPTUREONLY_NATIVE_LIB_CPP_H
#define CAPTUREONLY_NATIVE_LIB_CPP_H
#include <jni.h>
#include <build_android/caffe2/proto/caffe2.pb.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include "opencv2/core.hpp"

void loadToNetDef(AAssetManager* mgr, caffe2::NetDef* net, const char *filename);
#ifdef __cplusplus
extern "C" {
#endif
JNIEXPORT void JNICALL
Java_org_uelordi_deepsamples_caffe2_JniManager_init(JNIEnv *env, jclass type,  jobject assetManager);

JNIEXPORT void JNICALL
Java_org_uelordi_deepsamples_caffe2_JniManager_process(JNIEnv *env, jclass type, jlong colorImage,
                                                       jlong greyImage);
JNIEXPORT void JNICALL
Java_org_uelordi_deepsamples_caffe2_JniManager_start(JNIEnv *env, jclass type);

JNIEXPORT void JNICALL
Java_org_uelordi_deepsamples_caffe2_JniManager_stop(JNIEnv *env, jclass type);

JNIEXPORT jstring JNICALL
Java_org_uelordi_deepsamples_caffe2_JniManager_processYUVFrame(JNIEnv *env, jobject instance,
                                                               jint h, jint w, jbyteArray Y_,
                                                               jbyteArray U_, jbyteArray V_,
                                                               jint rowStride, jint pixelStride,
                                                               jboolean r_hwc);

#ifdef __cplusplus
}
#endif
#endif //CAPTUREONLY_NATIVE_LIB_CPP_H
//float * convertToFloatImage(const cv::Mat & img);