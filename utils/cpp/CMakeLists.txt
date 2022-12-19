project(NIKKE_SCRIPT)
cmake_minimum_required(VERSION 3.25)
include_directories("C:\Users\zhouyinan\Downloads\opencv\build\include")
add_subdirectory(src)

add_executable(test_screen test_screen.cpp)
target_link_libraries(test_screen win)