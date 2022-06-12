#ifndef DIRECTPOSEESTIMATION_H
#define DIRECTPOSEESTIMATION_H

#pragma once

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <mutex>

#include "JacobianAccumulator.h"

// class DirectPoseEstimation
// {
// public:
//     DirectPoseEstimation();
//     ~DirectPoseEstimation();

// private:
// };

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const std::vector<double> depth_ref,
    Sophus::SE3d &T21);

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const std::vector<double> depth_ref,
    Sophus::SE3d &T21);

#endif