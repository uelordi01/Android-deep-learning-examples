cmake_minimum_required(VERSION 3.4.1)
SET(OpenCV_FOUND true )
FIND_PACKAGE(OpenCV REQUIRED)
# ADD_SUBDIRECTORY(${OpenCV_DIR} ${CMAKE_CURRENT_BINARY_DIR}/opencv)
ADD_SUBDIRECTORY(${dlib_DIR} ${CMAKE_CURRENT_BINARY_DIR}/dlib)
SET(PROJECT_SRCS
         src/main/cpp/native-lib.h
         src/main/cpp/native-lib.cpp
         src/main/cpp/fps.h)
add_library( # Sets the name of the library.
             native-lib
             SHARED
             ${PROJECT_SRCS})

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )


target_link_libraries( # Specifies the target library.
                   native-lib
                   ${OpenCV_LIBS}
                   dlib
                   ${log-lib} )