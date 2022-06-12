cmake_minimum_required(VERSION 3.1.0)

project(GetRScamera)

set(CMAKE_BUILD_TYPE "Release")

# set( CMAKE_CXX_COMPILER "g++" )
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 ${SSE_FLAGS} -g -O3 -march=native")

# # Enable C++14
# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

FIND_PACKAGE(realsense2 REQUIRED)
FIND_PACKAGE(OpenCV 3 REQUIRED)
find_package(Sophus REQUIRED)

include_directories( 
    ${realsense2_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS} 
    ${Sophus_INCLUDE_DIRS}
    "/usr/include/eigen3/"
    )

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/lib)

add_library(algorithm SHARED
        DirectPoseEstimation.cpp
        JacobianAccumulator.cpp
        )

target_link_libraries(algorithm 
    # ${realsense2_LIBRARY}
    ${OpenCV_LIBS} 
    )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

add_executable(${PROJECT_NAME} 
    GetRScamera.cpp
    )

target_link_libraries( ${PROJECT_NAME} 
    ${realsense2_LIBRARY}
    ${OpenCV_LIBS} 
    algorithm
    )


