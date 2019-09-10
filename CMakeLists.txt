# Boilerplate
cmake_minimum_required(VERSION 3.7)
project(HandTrack)

# Extra settings
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

add_compile_options(-Wall -Wextra -Wpedantic)

# Find all .h .cpp files
file(GLOB_RECURSE SOURCES
    "headers/*.h"
    "main.cpp"
    "sources/*.cpp"
)
add_executable(${PROJECT_NAME} ${SOURCES})


# Finding libs and dependencies
include(FindPkgConfig)

find_package(OpenCV REQUIRED)
target_link_libraries(
    ${PROJECT_NAME} 
    ${OpenCV_LIBS}
)
