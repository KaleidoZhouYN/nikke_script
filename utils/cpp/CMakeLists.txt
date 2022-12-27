set( CMAKE_CXX_COMPILER "C:/MinGW/bin/g++.exe")
set( CMAKE_C_COMPILER "C:/MinGW/bin/gcc.exe")
cmake_minimum_required(VERSION 3.25)
project(nikke)

include_directories("C:/Users/zhouyinan/Downloads/opencv/build/include")
add_subdirectory(src)

add_executable(test_screen test_screen.cpp)
target_link_libraries(test_screen win)