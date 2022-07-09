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

#define IR_WHITE_DOTS_THRESHOLD 170

enum getPixelValueDirection
{
    normal,
    u_plus,
    u_minus,
    v_plus,
    v_minus
};

// 一次测量的值，包括一个世界坐标系下三维点与一个灰度值
struct Measurement
{
    Measurement(Eigen::Vector3d p, float g) : pos_world(p), grayscale(g) {}
    Eigen::Vector3d pos_world;
    float grayscale;
};

struct RelativePoseResult
{
    RelativePoseResult(Eigen::Isometry3d Tcw, double tframe)
        : Tcw_(Tcw), tframe_(tframe) {}

    Eigen::Isometry3d Tcw_;
    double tframe_;
};

//从关联文件中提取这些需要加载的图像的路径和时间戳
void LoadImages(const string &strAssociationFilename,
                vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD,
                vector<double> &vTimestamps,
                vector<int> &vHeightLimit,
                vector<int> &vWidthLimit);

// inline void pickMeasurement(
//     const cv::Mat &last_left,
//     const cv::Mat &last_depth,
//     cv::RNG &rng,
//     std::vector<Measurement> &measurements,
//     bool usingFAST = false,
//     int nPoints = 2000, int boarder = 20)
// {
//     measurements.clear();

//     if (usingFAST)
//     {
//         std::vector<cv::KeyPoint> keypoints;
//         cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
//         detector->detect(last_left, keypoints);
//         for (auto kp : keypoints)
//         {
//             // 去掉邻近边缘处的点
//             if (kp.pt.x < boarder || kp.pt.y < boarder ||
//                 (kp.pt.x + boarder) > last_left.cols ||
//                 (kp.pt.y + boarder) > last_left.rows)
//                 continue;
//             ushort d = last_depth.ptr<ushort>(cvRound(kp.pt.y))[cvRound(kp.pt.x)];
//             if (d == 0)
//                 continue;
//             d /= 5000;
//             Eigen::Vector3d p3d = project2Dto3D(kp.pt.x, kp.pt.y, d, fx, fy, cx, cy, 1.0f);
//             float grayscale = float(last_left.ptr<uchar>(cvRound(kp.pt.y))[cvRound(kp.pt.x)]);
//             measurements.push_back(Measurement(p3d, grayscale));
//         }
//     }
//     else // using random
//     {
//         for (int i = 0; i < nPoints; i++)
//         {
//             int x = rng.uniform(boarder, last_left.cols - boarder); // don't pick pixels close to boarder
//             int y = rng.uniform(boarder, last_left.rows - boarder); // don't pick pixels close to boarder
//             // int disparity = disparity_img.at<uchar>(y, x);
//             // double depth = fx * baseline / disparity; // you know this is disparity to depth
//             ushort d = last_depth.ptr<ushort>(y)[x];
//             if (d == 0)
//                 continue;
//             d /= 5000;
//             Eigen::Vector3d p3d = project2Dto3D(x, y, d, fx, fy, cx, cy, 1.0f);
//             float grayscale = float(last_left.ptr<uchar>(y)[x]);
//             measurements.push_back(Measurement(p3d, grayscale));
//         }
//     }
// }

inline Eigen::Vector3d project2Dto3D(
    int x, int y, float d,
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
        : x_world_(point),
          fx_(fx), fy_(fy), cx_(cx), cy_(cy),
          image_(image)
    {
        // image_height_ = image_->cols;
        // image_width_ = image_->rows;
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
        // boundary check
        if (x < 0)
            x = 0;
        if (y < 0)
            y = 0;
        if (x >= image_->cols)
            x = image_->cols - 1;
        if (y >= image_->rows)
            y = image_->rows - 1;

        uchar *data = &image_->data[int(y) * image_->step + int(x)];
        float xx = x - floor(x);
        float yy = y - floor(y);
        return float(
            (1 - xx) * (1 - yy) * data[0] +      // z00 lower-left
            xx * (1 - yy) * data[1] +            // z10 lower-right
            (1 - xx) * yy * data[image_->step] + // z01 upper-left
            xx * yy * data[image_->step + 1]     // z11 upper-right
        );
    }

    inline float getPixelValue_opt(float x, float y,
                                   getPixelValueDirection opt)
    {
        // boundary check
        if (x < 0)
            x = 0;
        if (y < 0)
            y = 0;
        if (x >= image_->cols)
            x = image_->cols - 1;
        if (y >= image_->rows)
            y = image_->rows - 1;

        uchar *data = &image_->data[int(y) * image_->step + int(x)];

        bool isWhiteDot = (data[0] > IR_WHITE_DOTS_THRESHOLD) ||
                          (data[1] > IR_WHITE_DOTS_THRESHOLD) ||
                          (data[image_->step] > IR_WHITE_DOTS_THRESHOLD) ||
                          (data[image_->step + 1] > IR_WHITE_DOTS_THRESHOLD);
        while (isWhiteDot)
        {
            if (opt == u_plus)
            {
                if (x >= image_->cols - 1)
                    break;
                x++;
            }
            else if (opt == u_minus)
            {
                if (x < 1)
                    break;
                x--;
            }
            else if (opt == v_plus)
            {
                if (y >= image_->rows - 1)
                    break;
                y++;
            }
            else if (opt == v_minus)
            {
                if (y < 1)
                    break;
                y--;
            }
            data = &image_->data[int(y) * image_->step + int(x)];
            isWhiteDot = (data[0] > IR_WHITE_DOTS_THRESHOLD) ||
                         (data[1] > IR_WHITE_DOTS_THRESHOLD) ||
                         (data[image_->step] > IR_WHITE_DOTS_THRESHOLD) ||
                         (data[image_->step + 1] > IR_WHITE_DOTS_THRESHOLD);
        }

        float xx = x - floor(x);
        float yy = y - floor(y);
        return float(
            (1 - xx) * (1 - yy) * data[0] +      // z00 lower-left
            xx * (1 - yy) * data[1] +            // z10 lower-right
            (1 - xx) * yy * data[image_->step] + // z01 upper-left
            xx * yy * data[image_->step + 1]     // z11 upper-right
        );
    }

public:
    Eigen::Vector3d x_world_;                 // 3D point in world frame
    float cx_ = 0, cy_ = 0, fx_ = 0, fy_ = 0; // Camera intrinsics
    cv::Mat *image_ = nullptr;                // reference image

    // private:
    //     static int image_height_;
    //     static int image_width_;
};

#endif