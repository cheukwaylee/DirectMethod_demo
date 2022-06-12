#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
// #include <boost/format.hpp>
// #include <pangolin/pangolin.h>
// #include <mutex>

#include "DirectPoseEstimation.h"

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
    // cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
    // cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);

    // get depth scale
    // float depth_scale = get_depth_scale(profile.get_device());

    // start stream
    pipe.start(cfg); //指示管道使用所请求的配置启动流

    // DirectMethod-related algorithm variable
    cv::Mat last_depth;
    cv::Mat last_left;
    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 2000;
    int boarder = 20;
    Sophus::SE3d T_cur_ref; // res

    VecVector2d pixels_ref;
    std::vector<double> depth_ref;

    while (true)
    {
        frames = pipe.wait_for_frames(); //等待所有配置的流生成框架

        // // Align to depth
        // rs2::align align_to_depth(RS2_STREAM_DEPTH);
        // frames = align_to_depth.process(frames);

        // // Get imu data
        // if (rs2::motion_frame accel_frame = frames.first_or_default(RS2_STREAM_ACCEL))
        // {
        //     rs2_vector accel_sample = accel_frame.get_motion_data();
        //     std::cout << "Accel:" << accel_sample.x << ", " << accel_sample.y << ", " << accel_sample.z << std::endl;
        // }
        // if (rs2::motion_frame gyro_frame = frames.first_or_default(RS2_STREAM_GYRO))
        // {
        //     rs2_vector gyro_sample = gyro_frame.get_motion_data();
        //     std::cout << "Gyro:" << gyro_sample.x << ", " << gyro_sample.y << ", " << gyro_sample.z << std::endl;
        // }

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
            pixels_ref.clear();
            depth_ref.clear();

            // generate pixels in ref and load depth data
            for (int i = 0; i < nPoints; i++)
            {
                int x = rng.uniform(boarder, last_left.cols - boarder); // don't pick pixels close to boarder
                int y = rng.uniform(boarder, last_left.rows - boarder); // don't pick pixels close to boarder
                // int disparity = disparity_img.at<uchar>(y, x);
                // double depth = fx * baseline / disparity; // you know this is disparity to depth
                double depth = last_depth.at<ushort>(y, x);
                depth_ref.push_back(depth);
                pixels_ref.push_back(Eigen::Vector2d(x, y));
            }

            // try single layer by uncomment this line
            // DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
            DirectPoseEstimationSingleLayer(last_left, pic_left, pixels_ref, depth_ref, T_cur_ref);
        }

        last_depth = pic_depth.clone(); // CV_16U
        last_left = pic_left.clone();   // CV_8UC1

        // Display in a GUI
        // cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
        // cv::imshow("Display Image", color);
        cv::imshow("Display depth", pic_depth * 15);
        cv::imshow("Display pic_left", pic_left);
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
