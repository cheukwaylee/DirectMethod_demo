#include <librealsense2/rs.hpp>

// #include <opencv2/opencv.hpp>
// #include <sophus/se3.hpp>
#include <boost/format.hpp>
// // #include <pangolin/pangolin.h>
// #include <mutex>

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

#include "poseEstimation.h"
#include "Converter.h"

using namespace std;
using namespace g2o;

#define width 640
#define height 480
#define fps 30

#ifndef _CAMERA_PARAMETER_
#define _CAMERA_PARAMETER_
// Camera intrinsics
const double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline
double baseline = 0.573;
#endif // !_CAMERA_PARAMETER_

// paths
// string left_file = "/home/cw/thesis_dataset/d435i_py/dark_notIR/000012.png";
// string depth_file = "/home/cw/thesis_dataset/d435i_py/dark_notIR/depth_000012.png";
// boost::format fmt_others("/home/cw/thesis_dataset/d435i_py/dark_notIR/%06d.png"); // other files
//从命令行输入参数中得到关联文件的路径
string strAssociationFilename = "../fr1_xyz.txt"; // path_to_association // TODO
const string traj_filename = "../res.txt";        // output

int main(int argc, char **argv)
{
    //按顺序存放需要读取的彩色图像、深度图像的路径，以及对应的时间戳的变量
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    //从关联文件中加载这些信息
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    //彩色图像和深度图像数据的一致性检查
    int nImages = vstrImageFilenamesRGB.size();
    if (vstrImageFilenamesRGB.empty())
    {
        cerr << endl
             << "No images found in provided path." << endl;
        return 1;
    }
    else if (vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size())
    {
        cerr << endl
             << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    vector<float> vTimesTrack; // 保存每一帧的处理时间
    vTimesTrack.resize(nImages);

    // DirectMethod-related algorithm variable
    cv::Mat last_depth;
    cv::Mat last_left;
    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 2000;
    int boarder = 20;

    std::vector<Measurement> measurements;
    Eigen::Matrix3f K;
    K << fx, 0.f, cx, 0.f, fy, cy, 0.f, 0.f, 1.0f;

    std::vector<RelativePoseResult> vRelativePoseResult; // res
    // cv::Mat last_left = cv::imread(left_file, 0);
    // cv::Mat last_depth = cv::imread(depth_file, 0);
    // measurements.clear();

    for (int ni = 0; ni < nImages; ni++) // ni 当前正在处理第ni张图
    {
        // TODO 读入新图 放在 pic_left pic_depth pic_left_timestamp
        cv::Mat pic_left = cv::imread(string(argv[3]) + "/" + vstrImageFilenamesRGB[ni], CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat pic_depth = cv::imread(string(argv[3]) + "/" + vstrImageFilenamesD[ni], CV_LOAD_IMAGE_UNCHANGED);
        double pic_left_timestamp = vTimestamps[ni];

        if (last_depth.empty() || last_left.empty())
        {
        }
        else
        {
            pickMeasurement(
                last_left, last_depth,
                rng,
                measurements, // output
                false, nPoints, boarder);

            chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
            Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity(); // res
            poseEstimationDirect(measurements, &pic_left, K, Tcw);
            chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
            chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
            cout << "direct method costs time: " << time_used.count() * 1000 << " ms" << endl;
            cout << "Tcw=" << endl
                 << Tcw.matrix() << endl;
            vRelativePoseResult.push_back(RelativePoseResult(Tcw, pic_left_timestamp));

            // plot the feature points
            cv::Mat img_show(height * 2, width, CV_8UC1);
            last_left.copyTo(img_show(cv::Rect(0, 0, width, height)));
            pic_left.copyTo(img_show(cv::Rect(0, height, width, height)));
            cvtColor(img_show, img_show, cv::COLOR_GRAY2RGB);

            cv::Mat img_res;
            cvtColor(pic_left, img_res, cv::COLOR_GRAY2RGB);
            for (Measurement m : measurements)
            {
                if (rand() > RAND_MAX / 5)
                    continue;
                Eigen::Vector3d p = m.pos_world;
                Eigen::Vector2d pixel_prev = project3Dto2D(p(0, 0), p(1, 0), p(2, 0), fx, fy, cx, cy);
                Eigen::Vector3d p2 = Tcw * m.pos_world;
                Eigen::Vector2d pixel_now = project3Dto2D(p2(0, 0), p2(1, 0), p2(2, 0), fx, fy, cx, cy);
                if (pixel_now(0, 0) < 0 ||
                    pixel_now(0, 0) >= width ||
                    pixel_now(1, 0) < 0 ||
                    pixel_now(1, 0) >= height)
                    continue;
                float b = 255 * float(rand()) / RAND_MAX;
                float g = 255 * float(rand()) / RAND_MAX;
                float r = 255 * float(rand()) / RAND_MAX;
                cv::circle(img_show, cv::Point2d(pixel_prev(0, 0), pixel_prev(1, 0)), 8, cv::Scalar(b, g, r), 2);
                cv::circle(img_show, cv::Point2d(pixel_now(0, 0), pixel_now(1, 0) + height), 8, cv::Scalar(b, g, r), 2);
                cv::line(img_show, cv::Point2d(pixel_prev(0, 0), pixel_prev(1, 0)), cv::Point2d(pixel_now(0, 0), pixel_now(1, 0) + height), cv::Scalar(b, g, r), 1);

                cv::circle(img_res, cv::Point2f(pixel_now[0], pixel_now[1]), 2, cv::Scalar(0, 250, 0), 2);
                cv::line(img_res, cv::Point2f(pixel_prev[0], pixel_prev[1]), cv::Point2f(pixel_now[0], pixel_now[1]),
                         cv::Scalar(0, 250, 0));
            }
            cv::imshow("matching", img_show);
            cv::imshow("tracking", img_res);
            cv::waitKey(1);
        }

        last_left = pic_left.clone();   // CV_8UC1
        last_depth = pic_depth.clone(); // CV_16U

        // Display in a GUI
        // cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
        // cv::imshow("Display Image", color);
        // cv::imshow("Display depth", pic_depth * 15);
        // cv::imshow("Display pic_left", pic_left);
        // cv::imshow("Display pic_right", pic_right);
        // cv::waitKey(1);
    }

    ofstream f;
    f.open(traj_filename.c_str());
    //这个可以理解为，在输出浮点数的时候使用0.3141592654这样的方式而不是使用科学计数法
    f << fixed; /// Generate floating-point output in fixed-point notation.
    for (RelativePoseResult res : vRelativePoseResult)
    {
        cv::Mat Tcw = Converter::toCvMat(res.Tcw_); // Eigen::Isometry3d to cv::Mat
        double timestamp = res.tframe_;

        //然后分解出旋转矩阵
        cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
        //以及平移向量
        cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);
        //用四元数表示旋转
        vector<float> q = Converter::toQuaternion(Rwc);
        //然后按照给定的格式输出到文件中
        f << setprecision(6) << timestamp << " " << setprecision(9) << twc.at<float>(0)
          << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0]
          << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    //操作完毕，关闭文件并且输出调试信息
    f.close();
    std::cout << std::endl
              << "trajectory saved!" << std::endl;
    return 0;
}
