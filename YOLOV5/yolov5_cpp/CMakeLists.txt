cmake_minimum_required(VERSION 3.5)
project(yolov5_cpp)
set(CMAKE_CXX_STANDARD 11)
include_directories(
    #OpenVINO推理引擎的头文件
    C:/Program Files (x86)/IntelSWTools/openvino_2020.4.287/deployment_tools/inference_engine/include
    C:/Program Files (x86)/IntelSWTools/openvino_2020.4.287/deployment_tools/ngraph/include
)

#查找必要的依赖包
set(OpenCV_DIR "C:/opencv/build/x64/vc15/lib")
find_package(OpenCV REQUIRED)
set(InferenceEngine_DIR "C:/Program Files (x86)/IntelSWTools/openvino_2020.4.287/deployment_tools/inference_engine/share")
find_package(InferenceEngine)
set(ngraph_DIR "C:/Program Files (x86)/IntelSWTools/openvino_2020.4.287/deployment_tools/ngraph/cmake")
find_package(ngraph REQUIRED)
set(ngraph_LIBRARIES "C:/Program Files (x86)/IntelSWTools/openvino_2020.4.287/deployment_tools/ngraph/lib")
set(ngraph_INCLUDE_DIRS "C:/Program Files (x86)/IntelSWTools/openvino_2020.4.287/deployment_tools/ngraph/include")


# 编译detector.so
add_library (detector SHARED ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_include_directories(detector
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PUBLIC ${OpenCV_INCLUDE_DIR}
        PUBLIC ${InferenceEngine_INCLUDE_DIRS}
#        PUBLIC ${ngraph_INCLUDE_DIRS}
)
target_link_libraries(detector
        ${OpenCV_LIBS}
        ${InferenceEngine_LIBRARIES}
        ${ngraph_LIBRARIES}
)

add_executable(detect_test main.cpp)
target_link_libraries(detect_test
        ${OpenCV_LIBS}
        detector
)