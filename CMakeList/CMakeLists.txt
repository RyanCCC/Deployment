#1. 指定：CMAKE版本
cmake_minimum_required(VERSION 2.8)

#2. project name：指定项目名称
project(CMakeList)

#3. 头文件目录
include_directories(include)

#4. 源文件目录，将其赋值给DIR_SRCS
aux_source_directory(src DIR_SRCS)

#5. 设置库文件名称hello
add_library(hello ${DIR_SRCS})

#6. 生成可执行文件名为main
add_executable(main main.cpp)

#7. 将库文件链接到可执行文件
target_link_libraries(main hello)