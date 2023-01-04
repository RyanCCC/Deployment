cmake_minimum_required(VERSION 2.8)
project(YOLOX_PREPROCESSING)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

# 添加opencv
set(OpenCV_DIR 'your opencv path')
find_package(OpenCV REQUIRED)

#打印opencv相关信息
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# 添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(main main.cpp)

# Link your application with OpenCV libraries
target_link_libraries(main PRIVATE ${OpenCV_LIBS})
