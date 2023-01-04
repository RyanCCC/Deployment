#1. 指定：CMAKE版本
cmake_minimum_required(VERSION 3.0)

project(DLLDemo)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-m64")

#声明引入子项目（目录）
ADD_SUBDIRECTORY(lib)

# 生成测试dll的可执行文件
 add_executable(main main.cpp)
