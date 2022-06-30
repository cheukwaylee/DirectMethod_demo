#include <librealsense2/rs.hpp>

// #include <opencv2/opencv.hpp>
// #include <sophus/se3.hpp>
// // #include <boost/format.hpp>
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

using namespace std;
using namespace g2o;

#define width 640
#define height 480
#define fps 30

#ifndef _CAMERA_PARAMETER_
#define _CAMERA_PARAMETER_
// Camera intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline
double baseline = 0.573;
#endif // !_CAMERA_PARAMETER_

int main(int argc, char **argv)
try
{
    // judge whether devices is exist or not
    rs2::context ctx;
    auto list = ctx.query_devices(); // Get a snapshot of currently connected devices
    if (list.size() == 0)
        throw std::runtime_error("No device detected. Is it plugged in?");
    rs2::device dev = list.front();

    rs2::frameset frames;
    // Contruct a pipeline which abstracts the device
    rs2::pipeline pipe; //创建一个通信管道//https://baike.so.com/doc/1559953-1649001.html pipeline的解释
                        // Create a configuration for configuring the pipeline with a non default profile
    rs2::config cfg;    //创建一个以非默认配置的配置用来配置管道
    // Add desired streams to configuration
    //  cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps); //向配置添加所需的流
    cfg.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);
    cfg.enable_stream(RS2_STREAM_INFRARED, 1, width, height, RS2_FORMAT_Y8, fps);
    // cfg.enable_stream(RS2_STREAM_INFRARED, 2, width, height, RS2_FORMAT_Y8, fps);

    // start stream
    rs2::pipeline_profile selection = pipe.start(cfg); //指示管道使用所请求的配置启动流

    // set emitter power
    // rs2::device selected_device = selection.get_device();
    auto &selected_device = dev;
    auto depth_sensor = selected_device.first<rs2::depth_sensor>();
    if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED))
    {
        depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1.f); // Enable emitter
        depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0.f); // Disable emitter
    }
    if (depth_sensor.supports(RS2_OPTION_LASER_POWER))
    {
        // Query min and max values:
        auto range = depth_sensor.get_option_range(RS2_OPTION_LASER_POWER);
        depth_sensor.set_option(RS2_OPTION_LASER_POWER, range.max); // Set max power
        depth_sensor.set_option(RS2_OPTION_LASER_POWER, 0.f);       // Disable laser
    }

    // DirectMethod-related algorithm variable
    cv::Mat last_depth;
    cv::Mat last_left;

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 3500;
    int boarder = 20;

    std::vector<Measurement> measurements;
    Eigen::Matrix3f K;
    K << fx, 0.f, cx, 0.f, fy, cy, 0.f, 0.f, 1.0f;
    // Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity(); // res
    // std::vector<Eigen::Vector2d> pixel_history;

    std::vector<cv::KeyPoint> keypoints;
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();

    while (true)
    {
        frames = pipe.wait_for_frames(); //等待所有配置的流生成框架

        // // Align to depth
        // rs2::align align_to_depth(RS2_STREAM_DEPTH);
        // frames = align_to_depth.process(frames);

        // Get each frame
        // rs2::frame color_frame = frames.get_color_frame();
        rs2::frame depth_frame = frames.get_depth_frame();
        rs2::video_frame ir_frame_left = frames.get_infrared_frame(1);
        // rs2::video_frame ir_frame_right = frames.get_infrared_frame(2);

        // Creating OpenCV Matrix from a color image
        // cv::Mat color(cv::Size(width, height), CV_8UC3, (void *)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat pic_depth(cv::Size(width, height), CV_16U, (void *)depth_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat pic_left(cv::Size(width, height), CV_8UC1, (void *)ir_frame_left.get_data());
        // cv::Mat pic_right(cv::Size(width, height), CV_8UC1, (void *)ir_frame_right.get_data());

        if (last_depth.empty() || last_left.empty())
        {
        }
        else
        {
            measurements.clear();

            bool usingFAST = false;
            if (usingFAST)
            {
                keypoints.clear();
                detector->detect(last_left, keypoints);
                for (auto kp : keypoints)
                {
                    // 去掉邻近边缘处的点
                    if (kp.pt.x < boarder || kp.pt.y < boarder ||
                        (kp.pt.x + boarder) > last_left.cols || (kp.pt.y + boarder) > last_left.rows)
                        continue;
                    ushort d = last_depth.ptr<ushort>(cvRound(kp.pt.y))[cvRound(kp.pt.x)];
                    if (d == 0)
                        continue;
                    Eigen::Vector3d p3d = project2Dto3D(kp.pt.x, kp.pt.y, d, fx, fy, cx, cy, 1.0f);
                    float grayscale = float(last_left.ptr<uchar>(cvRound(kp.pt.y))[cvRound(kp.pt.x)]);
                    measurements.push_back(Measurement(p3d, grayscale));
                }
            }
            else
            {
                for (int i = 0; i < nPoints; i++)
                {
                    int x = rng.uniform(boarder, last_left.cols - boarder); // don't pick pixels close to boarder
                    int y = rng.uniform(boarder, last_left.rows - boarder); // don't pick pixels close to boarder
                    // int disparity = disparity_img.at<uchar>(y, x);
                    // double depth = fx * baseline / disparity; // you know this is disparity to depth
                    ushort d = last_depth.ptr<ushort>(y)[x];
                    if (d == 0)
                        continue;
                    Eigen::Vector3d p3d = project2Dto3D(x, y, d, fx, fy, cx, cy, 1.0f);
                    float grayscale = float(last_left.ptr<uchar>(y)[x]);
                    measurements.push_back(Measurement(p3d, grayscale));
                }
            }

            chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
            Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity(); // res
            poseEstimationDirect(measurements, &pic_left, K, Tcw);
            chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
            chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
            cout << "direct method costs time: " << time_used.count() << " seconds." << endl;
            cout << "Tcw=" << endl
                 << Tcw.matrix() << endl;

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
        cv::imshow("Display depth", pic_depth * 15);
        // cv::imshow("Display pic_left", pic_left);
        // cv::imshow("Display pic_right", pic_right);
        cv::waitKey(1);
    }
    return 0;
}

// error
catch (const rs2::error &e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception &e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
