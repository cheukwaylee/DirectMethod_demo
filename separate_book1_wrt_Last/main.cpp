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

// #define TUM1_fr1_xyz "../res_TUM1_fr1_xyz.txt"
// #define ETH3d_plant_dark "../res_ETH3d_plant_dark.txt"
#define ETH3d_einstein_global_light_changes_1 "../res_ETH3d_einstein_global_light_changes_1.txt"
// #define ETH3d_einstein_global_light_changes_2 "../res_ETH3d_einstein_global_light_changes_2.txt"
// #define ETH3d_einstein_global_light_changes_3 "../res_ETH3d_einstein_global_light_changes_3.txt"

#ifdef ETH3d_einstein_global_light_changes_1
#define ETH3d_parameter
string path_to_rgb = "/home/cw/thesis_dataset/eth-3d/einstein_global_light_changes_1_mono/einstein_global_light_changes_1";
string path_to_depth = "/home/cw/thesis_dataset/eth-3d/einstein_global_light_changes_1_rgbd/einstein_global_light_changes_1";
string strAssociationFilename = path_to_depth + "/associated_low_bright.txt"; // path_to_association
const string traj_filename = ETH3d_einstein_global_light_changes_1;           // output
#endif                                                                        // ETH3d_einstein_global_light_changes_1

#ifdef ETH3d_einstein_global_light_changes_2
#define ETH3d_parameter
string path_to_rgb = "/home/cw/thesis_dataset/eth-3d/einstein_global_light_changes_2_mono/einstein_global_light_changes_2";
string path_to_depth = "/home/cw/thesis_dataset/eth-3d/einstein_global_light_changes_2_rgbd/einstein_global_light_changes_2";
string strAssociationFilename = path_to_depth + "/associated.txt";  // path_to_association
const string traj_filename = ETH3d_einstein_global_light_changes_2; // output
#endif                                                              // ETH3d_einstein_global_light_changes_2

#ifdef ETH3d_einstein_global_light_changes_3
#define ETH3d_parameter
string path_to_rgb = "/home/cw/thesis_dataset/eth-3d/einstein_global_light_changes_3_mono/einstein_global_light_changes_3";
string path_to_depth = "/home/cw/thesis_dataset/eth-3d/einstein_global_light_changes_3_rgbd/einstein_global_light_changes_3";
string strAssociationFilename = path_to_depth + "/associated.txt";  // path_to_association
const string traj_filename = ETH3d_einstein_global_light_changes_3; // output
#endif                                                              // ETH3d_einstein_global_light_changes_3

#ifdef TUM1_fr1_xyz
#define width 640 // TUM1
#define height 480
// Camera intrinsics
const double fx = 517.306408, fy = 516.469215,
             cx = 318.643040, cy = 255.313989;
// double baseline = 0.573; // baseline
string strAssociationFilename = "../fr1_xyz.txt"; // path_to_association
string path_to_rgb = "/home/cw/thesis_dataset/data_tum_rgbd/rgbd_dataset_freiburg1_xyz";
string path_to_depth = "/home/cw/thesis_dataset/data_tum_rgbd/rgbd_dataset_freiburg1_xyz";
const string traj_filename = TUM1_fr1_xyz; // output
#endif                                     // TUM1_fr1_xyz

#ifdef ETH3d_plant_dark
#define ETH3d_parameter
string strAssociationFilename = "/home/cw/thesis_dataset/eth-3d/plant_dark_rgbd/plant_dark/associated.txt"; // path_to_association
string path_to_rgb = "/home/cw/thesis_dataset/eth-3d/plant_dark_mono/plant_dark";
string path_to_depth = "/home/cw/thesis_dataset/eth-3d/plant_dark_rgbd/plant_dark";
const string traj_filename = ETH3d_plant_dark; // output
#endif                                         // ETH3d_plant_dark

#ifdef ETH3d_parameter
#define width 739 // ETH-3d
#define height 458
// Camera intrinsics
const double fx = 726.28741455078, fy = 726.28741455078,
             cx = 354.6496887207, cy = 186.46566772461;
#endif // ETH3d_parameter

// #define fps 30
// paths
// string left_file = "/home/cw/thesis_dataset/d435i_py/dark_notIR/000012.png";
// string depth_file = "/home/cw/thesis_dataset/d435i_py/dark_notIR/depth_000012.png";
// boost::format fmt_others("/home/cw/thesis_dataset/d435i_py/dark_notIR/%06d.png"); // other files
//??????????????????????????????????????????????????????

int main(int argc, char **argv)
{
    //??????????????????????????????????????????????????????????????????????????????????????????????????????
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    //????????????????????????????????????
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    //???????????????????????????????????????????????????
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

    vector<float> vTimesTrack; // ??????????????????????????????
    vTimesTrack.resize(nImages);

    // DirectMethod-related algorithm variable
    cv::Mat last_depth;
    cv::Mat last_left;
    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 3200;
    int boarder = 15;

    std::vector<Measurement> measurements;
    Eigen::Matrix3f K;
    K << fx, 0.f, cx, 0.f, fy, cy, 0.f, 0.f, 1.0f;

    std::vector<RelativePoseResult> vRelativePoseResult; // res
    // cv::Mat last_left = cv::imread(left_file, 0);
    // cv::Mat last_depth = cv::imread(depth_file, 0);
    // measurements.clear();
    Eigen::Isometry3d T_curr_world = Eigen::Isometry3d::Identity(); // current wrt world

    for (int ni = 0; ni < nImages; ni++) // ni ?????????????????????ni??????
    {
        // TODO ???????????? ?????? pic_left pic_depth pic_left_timestamp
        cv::Mat pic_left = cv::imread(path_to_rgb + "/" + vstrImageFilenamesRGB[ni], CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat pic_depth = cv::imread(path_to_depth + "/" + vstrImageFilenamesD[ni], CV_LOAD_IMAGE_UNCHANGED);
        double pic_left_timestamp = vTimestamps[ni];

        cvtColor(pic_left, pic_left, cv::COLOR_RGB2GRAY);

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

            bool usingFAST = false;
            if (usingFAST)
            {
                std::vector<cv::KeyPoint> keypoints;
                cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
                detector->detect(last_left, keypoints);
                for (auto kp : keypoints)
                {
                    // ???????????????????????????
                    if (kp.pt.x < boarder || kp.pt.y < boarder ||
                        (kp.pt.x + boarder) > last_left.cols ||
                        (kp.pt.y + boarder) > last_left.rows)
                        continue;
                    float d = last_depth.ptr<ushort>(cvRound(kp.pt.y))[cvRound(kp.pt.x)];
                    if (d == 0)
                        continue;
                    d /= 5000;
                    Eigen::Vector3d p3d = project2Dto3D(kp.pt.x, kp.pt.y, d, fx, fy, cx, cy, 1.0f);
                    float grayscale = float(last_left.ptr<uchar>(cvRound(kp.pt.y))[cvRound(kp.pt.x)]);
                    measurements.push_back(Measurement(p3d, grayscale));
                }
            }
            else // using random
            {
                for (int i = 0; i < nPoints; i++)
                {
                    int x = rng.uniform(boarder, last_left.cols - boarder); // don't pick pixels close to boarder
                    int y = rng.uniform(boarder, last_left.rows - boarder); // don't pick pixels close to boarder
                    // int disparity = disparity_img.at<uchar>(y, x);
                    // double depth = fx * baseline / disparity; // you know this is disparity to depth
                    float d = last_depth.ptr<ushort>(y)[x];
                    if (d == 0)
                        continue;
                    d /= 5000;
                    Eigen::Vector3d p3d = project2Dto3D(x, y, d, fx, fy, cx, cy, 1.0f);
                    float grayscale = float(last_left.ptr<uchar>(y)[x]);
                    measurements.push_back(Measurement(p3d, grayscale));
                }
            }

            chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
            Eigen::Isometry3d T_curr_last = Eigen::Isometry3d::Identity(); // res: current wrt last
            poseEstimationDirect(measurements, &pic_left, K, T_curr_last);
            chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
            chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);

            T_curr_world = T_curr_last * T_curr_world; // curr wrt last * last wrt world
            cout << "direct method costs time: " << time_used.count() * 1000 << " ms" << endl;
            cout << "T_curr_world=" << endl
                 << T_curr_world.matrix() << endl;
            // cout << "T_curr_last=" << endl
            //      << T_curr_last.matrix() << endl;
            vRelativePoseResult.push_back(RelativePoseResult(T_curr_world, pic_left_timestamp));

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
                Eigen::Vector3d p2 = T_curr_last * m.pos_world;
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
    //?????????????????????????????????????????????????????????0.3141592654?????????????????????????????????????????????
    f << fixed; /// Generate floating-point output in fixed-point notation.
    for (RelativePoseResult res : vRelativePoseResult)
    {
        cv::Mat Tcw = ORB_SLAM2::Converter::toCvMat(res.Tcw_.matrix()); // Eigen::Isometry3d to cv::Mat
        double timestamp = res.tframe_;

        //???????????????????????????
        cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
        //??????????????????
        cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);
        // const float random_range = 0.0;
        // twc.at<float>(0) = rng.uniform(-random_range, random_range);
        // twc.at<float>(1) = rng.uniform(-random_range, random_range);
        // twc.at<float>(2) = rng.uniform(-random_range, random_range);
        //????????????????????????
        vector<float> q = ORB_SLAM2::Converter::toQuaternion(Rwc);
        //?????????????????????????????????????????????
        f << setprecision(6) << timestamp << " " << setprecision(9) << twc.at<float>(0)
          << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0]
          << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    //???????????????????????????????????????????????????
    f.close();
    std::cout << std::endl
              << "trajectory saved!" << std::endl;
    return 0;
}
