cmake_minimum_required(VERSION 3.1.0)

project(main_using_slambook1)

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++14 -O3" )

# 添加cmake模块路径
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

FIND_PACKAGE(realsense2 REQUIRED)
FIND_PACKAGE(OpenCV 3 REQUIRED)
# find_package(Sophus REQUIRED)
find_package(G2O)
# find_package(Pangolin REQUIRED) # ?

include_directories( 
    ${realsense2_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS} 
    # ${Sophus_INCLUDE_DIRS}
    "/usr/include/eigen3/"
    ${G2O_INCLUDE_DIRS} 
    # ${Pangolin_INCLUDE_DIRS} # ?
    )

set( G2O_LIBS 
    g2o_core g2o_types_sba g2o_solver_csparse g2o_stuff g2o_csparse_extension 
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/lib)

add_library(algorithm SHARED
        poseEstimation.cpp
        # visualization/Viewer.cc
        )

target_link_libraries(algorithm 
    ${OpenCV_LIBS} 
    ${G2O_LIBS}
    # ${Pangolin_INCLUDE_DIRS} # ?
    )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

add_executable(${PROJECT_NAME} 
    main.cpp
    )

target_link_libraries( ${PROJECT_NAME} 
    ${realsense2_LIBRARY}
    ${OpenCV_LIBS}
    algorithm
    )


