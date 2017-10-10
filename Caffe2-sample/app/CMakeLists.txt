cmake_minimum_required(VERSION 3.4.1)
SET(OpenCV_FOUND true )
FIND_PACKAGE(OpenCV REQUIRED)
SET(PROTOBUF_PROTOC_EXECUTABLE ${caffe2_DIR}/build_host_protoc/bin/protoc)
ADD_SUBDIRECTORY(${dlib_DIR} ${CMAKE_CURRENT_BINARY_DIR}/dlib)

add_library( # Sets the name of the library.
             native-lib
             SHARED
             src/main/cpp/native-lib.h
             src/main/cpp/native-lib.cpp
             src/main/cpp/fps.h)


find_library( # Sets the name of the path variable.
              log-lib
              log )

find_library(
        android-lib
        android
        )

include(AndroidNdkModules)
android_ndk_import_module_cpufeatures()
INCLUDE_DIRECTORIES(${caffe2_DIR}
                    ${caffe2_DIR}/build_android
                    ${caffe2_DIR}/third_party
                    ${caffe2_DIR}/third_party/eigen
                    src/main
                    ${OpenCV_INCLUDE_DIRS}
                    )
add_library(
    caffe2
    STATIC
    IMPORTED
    )
set_target_properties(
    caffe2
    PROPERTIES IMPORTED_LOCATION
    ${caffe2_DIR}/build_android/caffe2/libCaffe2_CPU.a
    )
add_library(
    thread_pool
    STATIC
    IMPORTED
    )
set_target_properties(
    thread_pool
    PROPERTIES IMPORTED_LOCATION
    ${caffe2_DIR}/build_android/libCAFFE2_PTHREADPOOL.a
    )
add_library(
    glog
    SHARED
    IMPORTED
    )
set_target_properties(
    glog
    PROPERTIES IMPORTED_LOCATION
     ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libglog.so
    )

add_library(
    protobuf
    SHARED
    IMPORTED
    )
set_target_properties(
    protobuf
    PROPERTIES IMPORTED_LOCATION
    ${caffe2_DIR}/build_android/third_party/protobuf/cmake/libprotobuf.a
    )

add_library (
    NNPACK
    STATIC
    IMPORTED
    )
set_target_properties (
    NNPACK
    PROPERTIES IMPORTED_LOCATION
    ${caffe2_DIR}/build_android/libCAFFE2_NNPACK.a
    )

include_directories( src/main/cpp )

find_library(
     log-lib
     log
     )

target_link_libraries(
                       native-lib
                       -Wl,--whole-archive
                       caffe2
                       -Wl,--no-whole-archive
                       NNPACK
                       thread_pool
                       dlib
                       glog
                       protobuf
                       cpufeatures
                       ${log-lib}
                       ${android-lib})