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

// // visualize trajectory
// #include <thread>
// #include "visualizeTraj.h"

using namespace std;
using namespace g2o;

// #define width 640
// #define height 480
#define width 620
#define height 460
// Camera intrinsics // TODO d435i parameter
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;

int main(int argc, char **argv)
{
    // if (argc != 2)
    // {
    //     cerr << "input images path plz" << endl;
    //     return -1;
    // }
    // const string path_to_rgb = "/home/cw/thesis_dataset/d435i_py/20220705_test/20220705_231522";
    // const string path_to_rgb = "/home/cw/thesis_dataset/d435i_py/20220705_test/20220706_224022";
    const string path_to_rgb = argv[1];
    const string path_to_depth = path_to_rgb;
    const string strAssociationFilename = path_to_rgb + "/associated.txt"; // path_to_association
    const string traj_filename = "../res.txt";                             // output

    //按顺序存放需要读取的彩色图像、深度图像的路径，以及对应的时间戳的变量
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    vector<int> vHeightLimit;
    vector<int> vWidthLimit;

    //从关联文件中加载这些信息
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps,
               vHeightLimit, vWidthLimit);

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

    // // visualize trajectory
    // visualizeTraj::step();
    // std::thread visualizeTraj_loop;
    // visualizeTraj_loop = std::thread(visualizeTraj::run);

    // DirectMethod-related algorithm variable
    cv::Mat last_depth;
    cv::Mat last_left;
    int last_left_heightLimit, last_left_widthLimit;

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 3000;
    int boarder = 10;

    std::vector<Measurement> measurements;
    std::vector<Measurement> first_frame_measurements;
    bool first_frame_flag = true;
    Eigen::Matrix3f K;
    K << fx, 0.f, cx, 0.f, fy, cy, 0.f, 0.f, 1.0f;

    std::vector<RelativePoseResult> vRelativePoseResult;            // res
    Eigen::Isometry3d T_curr_world = Eigen::Isometry3d::Identity(); // current wrt world

    for (int ni = 0; ni < nImages; ni++) // ni 当前正在处理第ni张图
    {
        // TODO 读入新图 放在 pic_left pic_depth pic_left_timestamp
        cv::Mat pic_left = cv::imread(path_to_rgb + "/" + vstrImageFilenamesRGB[ni], CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat pic_depth = cv::imread(path_to_depth + "/" + vstrImageFilenamesD[ni], CV_LOAD_IMAGE_UNCHANGED);
        double pic_left_timestamp = vTimestamps[ni];
        int pic_left_heightLimit = vHeightLimit[ni];
        int pic_left_widthLimit = vWidthLimit[ni];
        // cout << "pic_left_heightLimit, pic_left_widthLimit "
        //      << pic_left_heightLimit << " , " << pic_left_widthLimit << endl;

        if (pic_left.channels() == 3)
        {
            cvtColor(pic_left, pic_left, cv::COLOR_RGB2GRAY);
        }

        /* Tetris
        // cv::Mat pic_left_post = cv::Mat::zeros(height - 0, width - 0, CV_8UC1);
        // cv::Mat pic_depth_post = cv::Mat::zeros(height - 0, width - 0, CV_16U);
        // int cols = pic_left.cols, rows = pic_left.rows;
        // for (int y = boarder, y_post = 0; y < rows - boarder; ++y)
        // {
        //     bool isLineVaild = false;

        //     u_char *pic_left_post_ptr = pic_left_post.ptr<uchar>(y_post);
        //     ushort *pic_depth_post_ptr = pic_depth_post.ptr<ushort>(y_post);

        //     for (int x = boarder, x_post = 0; x < cols - boarder; ++x)
        //     {
        //         u_char grayscale = last_left.ptr<uchar>(y)[x];
        //         ushort d = last_depth.ptr<ushort>(y)[x];
        //         if (grayscale < 180)
        //         {
        //             // cout << float(grayscale) << endl;
        //             // cout << d << endl;

        //             pic_left_post_ptr[x_post] = 155;
        //             pic_depth_post_ptr[x_post] = 155;
        //             ++x_post;
        //             isLineVaild = true;
        //         }
        //     }
        //     if (isLineVaild)
        //         ++y_post;
        // }
        // cv::imshow("pic_left_post", pic_left_post);
        // cv::imshow("pic_depth_post", pic_depth_post);
        // cv::waitKey();
        */

        if (last_depth.empty() || last_left.empty())
        {
        }
        else
        {
            // pickMeasurement(
            //     last_left, last_depth,
            //     rng,
            //     measurements, // output
            //     false, nPoints, boarder);
            measurements.clear();

            const bool usingFAST = false;
            if (usingFAST)
            {
                /* usingFAST
                // std::vector<cv::KeyPoint> keypoints;
                // cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
                // detector->detect(last_left, keypoints);
                // for (auto kp : keypoints)
                // {
                //     // 去掉邻近边缘处的点
                //     if (kp.pt.x < boarder || kp.pt.y < boarder ||
                //         (kp.pt.x + boarder) > last_left.cols ||
                //         (kp.pt.y + boarder) > last_left.rows)
                //         continue;
                //     float d = last_depth.ptr<ushort>(cvRound(kp.pt.y))[cvRound(kp.pt.x)];
                //     if (d == 0)
                //         continue;
                //     d /= 5000;
                //     Eigen::Vector3d p3d = project2Dto3D(kp.pt.x, kp.pt.y, d, fx, fy, cx, cy, 1.0f);
                //     float grayscale = float(last_left.ptr<uchar>(cvRound(kp.pt.y))[cvRound(kp.pt.x)]);
                //     measurements.push_back(Measurement(p3d, grayscale));
                // }
                */
            }
            else // using random
            {
                // 1-version
                for (int i = 0; i < nPoints; i++)
                {
                    int x = rng.uniform(boarder, last_left.cols - boarder); // don't pick pixels close to boarder
                    int y = rng.uniform(boarder, last_left.rows - boarder); // don't pick pixels close to boarder
                    // int disparity = disparity_img.at<uchar>(y, x);
                    // double depth = fx * baseline / disparity; // you know this is disparity to depth
                    float d = float(last_depth.ptr<ushort>(y)[x]);
                    float grayscale = float(last_left.ptr<uchar>(y)[x]);
                    if (grayscale > IR_WHITE_DOTS_THRESHOLD || d == 0)
                        continue;
                    // d /= 5000; // TODO d435i png sclar and depth?
                    Eigen::Vector3d p3d = project2Dto3D(x, y, d, fx, fy, cx, cy, 1.0f);
                    measurements.push_back(Measurement(p3d, grayscale));
                }

                // // 2-version
                // for (int i = 0; i < nPoints; i++)
                // {
                //     if (last_left_heightLimit > height || last_left_heightLimit < 0 ||
                //         last_left_widthLimit > width || last_left_widthLimit < 0)
                //     {
                //         last_left_heightLimit = height;
                //         last_left_widthLimit = width;
                //     }
                //     // cout << "last_left_heightLimit, last_left_widthLimit "
                //     //      << last_left_heightLimit << " , " << last_left_widthLimit << endl;

                //     int x = rng.uniform(0, last_left_widthLimit);  // don't pick pixels close to boarder
                //     int y = rng.uniform(0, last_left_heightLimit); // don't pick pixels close to boarder
                //     // int disparity = disparity_img.at<uchar>(y, x);
                //     // double depth = fx * baseline / disparity; // you know this is disparity to depth
                //     float d = float(last_depth.ptr<ushort>(y)[x]);
                //     float grayscale = float(last_left.ptr<uchar>(y)[x]);
                //     if (d == 0)
                //         continue;
                //     // d /= 5000; // TODO d435i png sclar and depth?
                //     Eigen::Vector3d p3d = project2Dto3D(x, y, d, fx, fy, cx, cy, 1.0f);
                //     measurements.push_back(Measurement(p3d, grayscale));
                // }
            }

            if (first_frame_flag & !measurements.empty())
            {
                first_frame_measurements = measurements;
                first_frame_flag = false;
            }

            chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
            Eigen::Isometry3d T_curr_last = Eigen::Isometry3d::Identity(); // res: current wrt last
            // T_curr_last(0, 3) = std::stod(argv[2]);
            // T_curr_last(1, 3) = std::stod(argv[3]);
            // T_curr_last(2, 3) = std::stod(argv[4]);
            poseEstimationDirect(measurements, &pic_left, K, T_curr_last);
            chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
            chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);

            T_curr_world = T_curr_last * T_curr_world; // curr wrt last * last wrt world
            cout << "direct method cost time: " << time_used.count() * 1000 << " ms" << endl;
            cout.flags(ios::fixed);
            cout.precision(6);
            cout << "T_curr_world=  @" << pic_left_timestamp << endl
                 << T_curr_world.matrix() << endl
                 << endl;
            // cout << "T_curr_last=" << endl
            //      << T_curr_last.matrix() << endl
            //      << endl;
            vRelativePoseResult.push_back(RelativePoseResult(T_curr_world, pic_left_timestamp));

            // plot the feature points

            // cv::Mat img_show(height * 2, width, CV_8UC1);
            // last_left.copyTo(img_show(cv::Rect(0, 0, width, height)));
            // pic_left.copyTo(img_show(cv::Rect(0, height, width, height)));
            // cvtColor(img_show, img_show, cv::COLOR_GRAY2RGB);

            cv::Mat img_adjTracking;
            cvtColor(pic_left, img_adjTracking, cv::COLOR_GRAY2RGB);

            cv::Mat img_firstTracking;
            cvtColor(pic_left, img_firstTracking, cv::COLOR_GRAY2RGB);

            // cout << "size of first_frame_measurements " << first_frame_measurements.size() << endl;
            // First_Tracking
            for (Measurement m : first_frame_measurements)
            {
                Eigen::Vector3d p = m.pos_world;
                Eigen::Vector2d pixel_first = project3Dto2D(p(0, 0), p(1, 0), p(2, 0), fx, fy, cx, cy);
                Eigen::Vector3d p2 = T_curr_world * m.pos_world;
                Eigen::Vector2d pixel_now = project3Dto2D(p2(0, 0), p2(1, 0), p2(2, 0), fx, fy, cx, cy);

                cv::circle(img_firstTracking,
                           cv::Point2f(pixel_first[0], pixel_first[1]), 1,
                           cv::Scalar(0, 250, 0), 1);
                cv::line(img_firstTracking,
                         cv::Point2f(pixel_first[0], pixel_first[1]),
                         cv::Point2f(pixel_now[0], pixel_now[1]),
                         cv::Scalar(0, 250, 0));
            }

            // Adj_Tracking
            for (Measurement m : measurements)
            {
                // if (rand() > RAND_MAX / 5)
                //     continue;
                Eigen::Vector3d p = m.pos_world;
                Eigen::Vector2d pixel_prev = project3Dto2D(p(0, 0), p(1, 0), p(2, 0), fx, fy, cx, cy);
                Eigen::Vector3d p2 = T_curr_last * m.pos_world;
                Eigen::Vector2d pixel_now = project3Dto2D(p2(0, 0), p2(1, 0), p2(2, 0), fx, fy, cx, cy);
                if (pixel_now(0, 0) < 0 ||
                    pixel_now(0, 0) >= width ||
                    pixel_now(1, 0) < 0 ||
                    pixel_now(1, 0) >= height)
                    continue;

                // float b = 255 * float(rand()) / RAND_MAX;
                // float g = 255 * float(rand()) / RAND_MAX;
                // float r = 255 * float(rand()) / RAND_MAX;
                // cv::circle(img_show, cv::Point2d(pixel_prev(0, 0), pixel_prev(1, 0)), 8, cv::Scalar(b, g, r), 2);
                // cv::circle(img_show, cv::Point2d(pixel_now(0, 0), pixel_now(1, 0) + height), 8, cv::Scalar(b, g, r), 2);
                // cv::line(img_show, cv::Point2d(pixel_prev(0, 0), pixel_prev(1, 0)), cv::Point2d(pixel_now(0, 0), pixel_now(1, 0) + height), cv::Scalar(b, g, r), 1);

                cv::circle(img_adjTracking,
                           cv::Point2f(pixel_now[0], pixel_now[1]), 1,
                           cv::Scalar(0, 250, 0), 1);
                cv::line(img_adjTracking,
                         cv::Point2f(pixel_prev[0], pixel_prev[1]),
                         cv::Point2f(pixel_now[0], pixel_now[1]),
                         cv::Scalar(0, 250, 0));
            }
            // cv::imshow("matching", img_show);
            cv::imshow("Adj_Tracking", img_adjTracking);
            cv::imshow("First_Tracking", img_firstTracking);
            cv::waitKey(120);
        }

        last_left = pic_left.clone();   // CV_8UC1
        last_depth = pic_depth.clone(); // CV_16U

        last_left_heightLimit = pic_left_heightLimit;
        last_left_widthLimit = pic_left_widthLimit;

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
        cv::Mat Tcw = ORB_SLAM2::Converter::toCvMat(res.Tcw_.matrix()); // Eigen::Isometry3d to cv::Mat
        double timestamp = res.tframe_;

        //然后分解出旋转矩阵
        cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
        //以及平移向量
        cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);
        //用四元数表示旋转
        vector<float> q = ORB_SLAM2::Converter::toQuaternion(Rwc);
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
