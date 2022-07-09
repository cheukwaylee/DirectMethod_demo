#include "visualizeTraj.h"

// visualizeTraj::visualizeTraj()
// {
// }

// visualizeTraj::~visualizeTraj()
// {
// }

void visualizeTraj::step()
{
    // create a window and bind its context to the main thread
    //创建一个窗口，绑定属性：窗口名Trajectory Viewer，长1024，宽768
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // // unset the current context from the main thread
    // pangolin::GetBoundWindow()->RemoveCurrent();
}

void visualizeTraj::run()
{
    // fetch the context and bind it to this thread
    pangolin::BindToContext(window_name);

    // we manually need to restore the properties of the context
    glEnable(GL_DEPTH_TEST);

    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
        pangolin::ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin::AxisY));

    // Create Interactive View in window
    pangolin::Handler3D handler(s_cam);
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                                .SetHandler(&handler);

    while (!pangolin::ShouldQuit())
    {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        // Render OpenGL Cube
        pangolin::glDrawColouredCube();

        // Swap frames and Process Events
        pangolin::FinishFrame();
    }

    // unset the current context from the main thread
    pangolin::GetBoundWindow()->RemoveCurrent();
}