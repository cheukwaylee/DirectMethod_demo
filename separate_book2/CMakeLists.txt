cmake_minimum_required(VERSION 2.8)
project(ch8)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 ${SSE_FLAGS} -g -O3 -march=native")

find_package(OpenCV 3 REQUIRED)
find_package(Sophus REQUIRED)
# find_package(Pangolin REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        # ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3/"
        # ${Pangolin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/lib)

add_library(algorithm SHARED
        DirectPoseEstimation.cpp
        JacobianAccumulator.cpp
        )

target_link_libraries(algorithm 
    ${OpenCV_LIBS} 
    # ${Pangolin_INCLUDE_DIRS} # ?
    )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

# add_executable(optical_flow optical_flow.cpp)
# target_link_libraries(optical_flow ${OpenCV_LIBS})

add_executable(direct_method 
        direct_method.cpp
        )

target_link_libraries(direct_method 
        ${OpenCV_LIBS} 
        # ${Pangolin_LIBRARIES}
        algorithm
        )
