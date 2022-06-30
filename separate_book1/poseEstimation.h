#ifndef POSEESTIMATION_H
#define POSEESTIMATION_H

#pragma once

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
#include <ctime>
#include <climits>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace std;
using namespace g2o;

// 一次测量的值，包括一个世界坐标系下三维点与一个灰度值
struct Measurement
{
    Measurement(Eigen::Vector3d p, float g) : pos_world(p), grayscale(g) {}
    Eigen::Vector3d pos_world;
    float grayscale;
};

inline Eigen::Vector3d project2Dto3D(
    int x, int y, int d,
    float fx, float fy, float cx, float cy, float scale)
{
    float zz = float(d) / scale;
    float xx = zz * (x - cx) / fx;
    float yy = zz * (y - cy) / fy;
    return Eigen::Vector3d(xx, yy, zz);
}

inline Eigen::Vector2d project3Dto2D(
    float x, float y, float z,
    float fx, float fy, float cx, float cy)
{
    float u = fx * x / z + cx;
    float v = fy * y / z + cy;
    return Eigen::Vector2d(u, v);
}

// 直接法估计位姿
// 输入：测量值（空间点的灰度），新的灰度图，相机内参；
// 输出：相机位姿
// 返回：true为成功，false失败
bool poseEstimationDirect(
    const vector<Measurement> &measurements,
    cv::Mat *gray,
    Eigen::Matrix3f &intrinsics,
    Eigen::Isometry3d &Tcw);

// project a 3d point into an image plane, the error is photometric error
// an unary edge with one vertex SE3Expmap (the pose of camera)
// 边edge   误差项
// 顶点vertex 待优化的变量 pose
class EdgeSE3ProjectDirect : public BaseUnaryEdge<1, double, VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // 构造函数
    EdgeSE3ProjectDirect() {}

    EdgeSE3ProjectDirect(
        Eigen::Vector3d point,
        float fx, float fy, float cx, float cy,
        cv::Mat *image)
        : x_world_(point), fx_(fx), fy_(fy), cx_(cx), cy_(cy), image_(image)
    {
    }

    virtual void computeError();
    virtual void linearizeOplus();

    // dummy read and write functions because we don't care...
    virtual bool read(std::istream &in) {}
    virtual bool write(std::ostream &out) const {}

protected:
    // Bilinear interpolation
    // get a gray scale value from reference image
    inline float getPixelValue(float x, float y)
    {
        uchar *data = &image_->data[int(y) * image_->step + int(x)];
        float xx = x - floor(x);
        float yy = y - floor(y);
        return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[image_->step] +
            xx * yy * data[image_->step + 1]);
    }

public:
    Eigen::Vector3d x_world_;                 // 3D point in world frame
    float cx_ = 0, cy_ = 0, fx_ = 0, fy_ = 0; // Camera intrinsics
    cv::Mat *image_ = nullptr;                // reference image
};

#endif