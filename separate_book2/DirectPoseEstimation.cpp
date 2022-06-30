#include "DirectPoseEstimation.h"

// DirectPoseEstimation::DirectPoseEstimation()
// {

// }

// DirectPoseEstimation::~DirectPoseEstimation()
// {

// }

void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const std::vector<double> depth_ref,
    Sophus::SE3d &T21)
{
    const int iterations = 30
    ; // 10
    double cost = 0, lastCost = 0;
    auto t1 = std::chrono::steady_clock::now();
    JacobianAccumulator jaco_accu(img1, img2, px_ref, depth_ref, T21);

    for (int iter = 0; iter < iterations; iter++)
    {
        jaco_accu.reset();
        // cv::parallel_for_(cv::Range(0, px_ref.size()),
        //                   std::bind(&JacobianAccumulator::accumulate_jacobian,
        //                             &jaco_accu,
        //                             std::placeholders::_1));
        jaco_accu.accumulate_jacobian();

        Matrix6d H = jaco_accu.hessian();
        Vector6d b = jaco_accu.bias();

        // solve update and put it into estimation
        Vector6d update = H.ldlt().solve(b);

        T21 = Sophus::SE3d::exp(update) * T21;
        cost = jaco_accu.cost_func();

        if (std::isnan(update[0]))
        {
            // sometimes occurred when we have a black or white patch and H is irreversible
            std::cout << "update is nan" << std::endl;
            break;
        }
        if (iter > 0 && cost > lastCost)
        {
            // std::cout << "cost increased: " << cost << ", " << lastCost << std::endl;
            break;
        }
        if (update.norm() < 1e-3)
        {
            // converge
            break;
        }

        lastCost = cost;
        // std::cout << "iteration: " << iter << ", cost: " << cost << std::endl;
    }

    std::cout << "T21 = \n"
              << T21.matrix() << std::endl;
    auto t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1); // unit: second
    std::cout << "single layer time: " << time_used.count() * 1000 << " ms" << std::endl;

    // plot the projected pixels here
    cv::Mat img2_show;
    cv::cvtColor(img2, img2_show, CV_GRAY2BGR);
    VecVector2d projection = jaco_accu.projected_points();
    for (size_t i = 0; i < px_ref.size(); ++i)
    {
        auto p_ref = px_ref[i];
        auto p_cur = projection[i];
        if (p_cur[0] > 0 && p_cur[1] > 0)
        {
            cv::circle(img2_show,
                       cv::Point2f(p_cur[0], p_cur[1]), 2,
                       cv::Scalar(0, 250, 0), 2);
            cv::line(img2_show,
                     cv::Point2f(p_ref[0], p_ref[1]),
                     cv::Point2f(p_cur[0], p_cur[1]),
                     cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("current", img2_show);
    cv::waitKey();
}

void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const std::vector<double> depth_ref,
    Sophus::SE3d &T21)
{

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    std::vector<cv::Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++)
    {
        if (i == 0)
        {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        }
        else
        {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    double fxG = fx, fyG = fy, cxG = cx, cyG = cy; // backup the old values
    for (int level = pyramids - 1; level >= 0; level--)
    {
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level
        for (auto &px : px_ref)
        {
            px_ref_pyr.push_back(scales[level] * px);
        }

        // scale fx, fy, cx, cy in different pyramid levels
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
    }
}
