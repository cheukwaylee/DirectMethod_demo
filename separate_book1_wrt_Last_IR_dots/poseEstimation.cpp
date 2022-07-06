#include "poseEstimation.h"

// virtual void EdgeSE3ProjectDirect::linearizeOplus()
// {
// }

// plus in manifold
void EdgeSE3ProjectDirect::linearizeOplus()
{
    if (level() == 1)
    {
        _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
        return;
    }

    VertexSE3Expmap *vtx = static_cast<VertexSE3Expmap *>(_vertices[0]);
    Eigen::Vector3d xyz_trans = vtx->estimate().map(x_world_); // q in book

    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double invz = 1.0 / xyz_trans[2];
    double invz_2 = invz * invz;

    // 3d to 2d
    float u = x * fx_ * invz + cx_;
    float v = y * fy_ * invz + cy_;

    // jacobian from se3 to u,v
    // NOTE that in g2o the Lie algebra is (\omega, \epsilon),
    // where \omega is so(3) and \epsilon the translation
    Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;

    jacobian_uv_ksai(0, 0) = -x * y * invz_2 * fx_;
    jacobian_uv_ksai(0, 1) = (1 + (x * x * invz_2)) * fx_;
    jacobian_uv_ksai(0, 2) = -y * invz * fx_;
    jacobian_uv_ksai(0, 3) = invz * fx_;
    jacobian_uv_ksai(0, 4) = 0;
    jacobian_uv_ksai(0, 5) = -x * invz_2 * fx_;

    jacobian_uv_ksai(1, 0) = -(1 + y * y * invz_2) * fy_;
    jacobian_uv_ksai(1, 1) = x * y * invz_2 * fy_;
    jacobian_uv_ksai(1, 2) = x * invz * fy_;
    jacobian_uv_ksai(1, 3) = 0;
    jacobian_uv_ksai(1, 4) = invz * fy_;
    jacobian_uv_ksai(1, 5) = -y * invz_2 * fy_;

    Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;

    jacobian_pixel_uv(0, 0) = (getPixelValue(u + 1, v) - getPixelValue(u - 1, v)) / 2;
    jacobian_pixel_uv(0, 1) = (getPixelValue(u, v + 1) - getPixelValue(u, v - 1)) / 2;

    _jacobianOplusXi = jacobian_pixel_uv * jacobian_uv_ksai;
}

void EdgeSE3ProjectDirect::computeError()
{
    const VertexSE3Expmap *v = static_cast<const VertexSE3Expmap *>(_vertices[0]);
    // 3d to 2d
    // u = fx * x / z + cx;
    // v = fy * y / z + cy;
    Eigen::Vector3d x_local = v->estimate().map(x_world_);
    float x = x_local[0] * fx_ / x_local[2] + cx_;
    float y = x_local[1] * fy_ / x_local[2] + cy_;

    // check x,y is in the image
    if (x - 4 < 0 || (x + 4) > image_->cols ||
        (y - 4) < 0 || (y + 4) > image_->rows)
    {
        _error(0, 0) = 0.0;
        this->setLevel(1); // 不优化这一项？
    }
    else
    {
        _error(0, 0) = getPixelValue(x, y) - _measurement; // current frame - reference frame's grayscale value as Measurement
    }
}

// 直接法估计位姿
// 输入：测量值（空间点的灰度）， 新的灰度图， 相机内参；
// 输出：相机位姿
// 返回：true为成功，false失败
bool poseEstimationDirect(
    const vector<Measurement> &measurements,
    cv::Mat *gray,
    Eigen::Matrix3f &K,
    Eigen::Isometry3d &Tcw)
{
    // 初始化g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>> DirectBlock; // 求解的向量是6＊1的

    DirectBlock::LinearSolverType *linearSolver = new g2o::LinearSolverDense<DirectBlock::PoseMatrixType>();

    // debug
    // DirectBlock *solver_ptr = new DirectBlock(linearSolver);
    DirectBlock *solver_ptr = new DirectBlock(std::unique_ptr<DirectBlock::LinearSolverType>(linearSolver));

    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr ); // G-N
    // g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(std::unique_ptr<DirectBlock>(solver_ptr)); // G-N
    // g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr); // L-M
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<DirectBlock>(solver_ptr)); // L-M

    /* debug tips
    //g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);  // line 356
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> (linearSolver));
    //g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr); // line 357
    g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<DirectBlock> (solver_ptr));
    */

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap();            // Vertex: optimization variable
    pose->setEstimate(g2o::SE3Quat(Tcw.rotation(), Tcw.translation())); // optimization variable initial guess
    pose->setId(0);
    optimizer.addVertex(pose);

    // 添加边
    // 边：误差项，每一个被采样的像素点都构成一个误差
    int id = 1;
    for (Measurement m : measurements)
    {
        EdgeSE3ProjectDirect *edge = new EdgeSE3ProjectDirect(
            m.pos_world,
            K(0, 0), K(1, 1), K(0, 2), K(1, 2),
            gray); // Edge: error
        edge->setVertex(0, pose);
        edge->setMeasurement(m.grayscale); // reference frame's grayscale value as Measurement
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        edge->setId(id++);
        optimizer.addEdge(edge);
    }
    cout << "edges in graph: " << optimizer.edges().size() << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30);
    Tcw = pose->estimate();
    return true;
}

//从关联文件中提取这些需要加载的图像的路径和时间戳
void LoadImages(const string &strAssociationFilename,
                vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD,
                vector<double> &vTimestamps)
{
    //输入文件流
    ifstream fAssociation;
    //打开关联文件
    fAssociation.open(strAssociationFilename.c_str());

    //一直读取,知道文件结束
    while (!fAssociation.eof())
    {
        string s; // 每一行
        //读取一行的内容到字符串s中
        getline(fAssociation, s);

        //如果不是空行就可以分析数据了
        if (!s.empty())
        {
            //字符串流
            stringstream ss;
            //字符串格式:  时间戳 rgb图像路径 时间戳 图像路径
            ss << s; // 整行读入 然后按空格间断输出

            double t;
            string sRGB, sD;

            ss >> t;
            vTimestamps.push_back(t);

            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);

            // ! bug? 左右目的时间戳可能不一致是否需要考虑单独处理
            // 没有写入任何一个向量 相当于第二个时间戳被抛弃了
            ss >> t;

            ss >> sD;
            vstrImageFilenamesD.push_back(sD);
        }
    }
}