#include "JacobianAccumulator.h"
// #include <sophus/types.hpp>

void JacobianAccumulator::accumulate_jacobian(const cv::Range &range)
{
    // parameters
    const int half_patch_size = 1;

    int cnt_good = 0;
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;

    for (size_t i = range.start; i < range.end; i++)
    {
        // compute the projection in the second image
        // 参考帧的pixel + 对应深度 --> 参考帧的空间三维点
        Eigen::Vector3d point_ref =
            depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - cx) / fx,
                                           (px_ref[i][1] - cy) / fy,
                                           1);

        // 参考帧恢复的空间三维点 经过前后帧变换 --> 当前帧的空间三维点
        Eigen::Vector3d point_cur = T21 * point_ref;
        if (point_cur[2] < 0) // depth invalid
            continue;

        // 当前帧的空间三维点 --> 当前帧的pixel
        float u = fx * point_cur[0] / point_cur[2] + cx;
        float v = fy * point_cur[1] / point_cur[2] + cy;
        if (u < half_patch_size || u > img2.cols - half_patch_size ||
            v < half_patch_size || v > img2.rows - half_patch_size)
            continue;

        // 参考帧的pixel 经过T21之后 重投影到当前帧的pixel
        projection[i] = Eigen::Vector2d(u, v);

        double X = point_cur[0], Y = point_cur[1], Z = point_cur[2],
               Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;
        cnt_good++;

        // and compute error and jacobian in a PATCH
        for (int x = -half_patch_size; x <= half_patch_size; x++)
            for (int y = -half_patch_size; y <= half_patch_size; y++)
            {
                // error = reference frame's（锚点灰度） - current(to be optimized)
                double error = GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) -
                               GetPixelValue(img2, u + x, v + y);

                Matrix26d J_pixel_xi;
                Eigen::Vector2d J_img_pixel;

                J_pixel_xi(0, 0) = fx * Z_inv; // same as the slambook2
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fx * X * Z2_inv;
                J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fx * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fy * Z_inv;
                J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fy * X * Z_inv;

                J_img_pixel = Eigen::Vector2d(
                    0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
                    0.5 * (GetPixelValue(img2, u + x, v + 1 + y) - GetPixelValue(img2, u + x, v - 1 + y)));

                // total jacobian
                Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

                hessian += J * J.transpose();
                bias += -error * J;
                cost_tmp += error * error;
            }
    }

    if (cnt_good)
    {
        std::unique_lock<std::mutex> lck(hessian_mutex);
        // hessian_mutex.lock();

        std::cout << "paraellel enable debuging" << std::endl;
        // std::cout << hessian << std::endl;
        // std::cout << bias << std::endl;
        // std::cout << "cnt_good " << cnt_good << std::endl;
        // std::cout << cost_tmp / cnt_good << std::endl;

        //! bug
        // set hessian, bias and cost
        H += hessian;
        b += bias;
        cost += cost_tmp / double(cnt_good);

        // hessian_mutex.unlock();
    }
}

void JacobianAccumulator::accumulate_jacobian()
{
    // parameters
    const int half_patch_size = 1;

    int cnt_good = 0;
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;

    for (size_t i = 0; i < px_ref.size(); i++)
    {
        // std::cout << "xuna" << i << std::endl;
        // compute the projection in the second image
        Eigen::Vector3d point_ref =
            depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - cx) / fx, (px_ref[i][1] - cy) / fy, 1);
        Eigen::Vector3d point_cur = T21 * point_ref;
        if (point_cur[2] < 0) // depth invalid
            continue;

        float u = fx * point_cur[0] / point_cur[2] + cx;
        float v = fy * point_cur[1] / point_cur[2] + cy;
        if (u < half_patch_size || u > img2.cols - half_patch_size ||
            v < half_patch_size || v > img2.rows - half_patch_size)
            continue;

        projection[i] = Eigen::Vector2d(u, v);
        double X = point_cur[0], Y = point_cur[1], Z = point_cur[2],
               Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;
        cnt_good++;

        // and compute error and jacobian
        for (int x = -half_patch_size; x <= half_patch_size; x++)
            for (int y = -half_patch_size; y <= half_patch_size; y++)
            {

                double error = GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) -
                               GetPixelValue(img2, u + x, v + y);
                Matrix26d J_pixel_xi;
                Eigen::Vector2d J_img_pixel;

                J_pixel_xi(0, 0) = fx * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fx * X * Z2_inv;
                J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fx * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fy * Z_inv;
                J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fy * X * Z_inv;

                J_img_pixel = Eigen::Vector2d(
                    0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
                    0.5 * (GetPixelValue(img2, u + x, v + 1 + y) - GetPixelValue(img2, u + x, v - 1 + y)));

                // total jacobian
                Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

                hessian += J * J.transpose();
                bias += -error * J;
                cost_tmp += error * error;
            }
    }

    if (cnt_good) // cnt_good
    {
        std::unique_lock<std::mutex> lck(hessian_mutex);
        // hessian_mutex.lock();

        std::cout
            << "paraellel disable debuging" << std::endl;
        // std::cout << &hessian << std::endl;
        // std::cout << bias << std::endl;
        // std::cout << "cnt_good " << cnt_good << std::endl;
        // std::cout << cost_tmp / cnt_good << std::endl;

        // ! bug
        // set hessian, bias and cost
        H += hessian;
        b += bias;
        cost += cost_tmp / cnt_good;

        // hessian_mutex.unlock();
    }
    return;
}
