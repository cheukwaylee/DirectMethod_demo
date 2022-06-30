#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>
// #include <pangolin/pangolin.h>
#include <mutex>

#include "DirectPoseEstimation.h"

using namespace std;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
    VecVector2d;

// Camera intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline
double baseline = 0.573;

// // paths
// string left_file = "../left.png";
// string disparity_file = "../disparity.png";
// boost::format fmt_others("../%06d.png"); // other files
string left_file = "/home/cw/thesis_dataset/d435i_py/dark_notIR/000012.png";
string disparity_file = "/home/cw/thesis_dataset/d435i_py/dark_notIR/depth_000012.png";
boost::format fmt_others("/home/cw/thesis_dataset/d435i_py/dark_notIR/%06d.png"); // other files
#define CURRENT_IMG_BEGIN 13
#define CURRENT_IMG_END 290

// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

int main(int argc, char **argv)
{

    cv::Mat left_img = cv::imread(left_file, 0);
    cv::Mat disparity_img = cv::imread(disparity_file, 0);

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 2000;
    int boarder = 20;
    VecVector2d pixels_ref;
    vector<double> depth_ref;

    // generate pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++)
    {
        int x = rng.uniform(boarder, left_img.cols - boarder); // don't pick pixels close to boarder
        int y = rng.uniform(boarder, left_img.rows - boarder); // don't pick pixels close to boarder
        // int disparity = disparity_img.at<uchar>(y, x);
        // double depth = fx * baseline / disparity; // you know this is disparity to depth
        double depth = disparity_img.at<uchar>(y, x);
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }

    // estimates 01~05.png's pose using this information
    Sophus::SE3d T_cur_ref;

    for (int i = CURRENT_IMG_BEGIN; i < CURRENT_IMG_END; i++) // 1-6
    {                                                         // 1~10
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);
        // try single layer by uncomment this line
        // DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
        DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
    }
    return 0;
}
