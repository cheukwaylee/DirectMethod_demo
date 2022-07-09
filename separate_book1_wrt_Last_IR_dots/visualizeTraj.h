#ifndef VISUALIZETRAJ_H
#define VISUALIZETRAJ_H

#pragma once

// visualize trajectory
// https://github.com/stevenlovegrove/Pangolin/blob/master/examples/HelloPangolinThreads/main.cpp
#include <pangolin/pangolin.h>
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/gl/gl.h>
#include <pangolin/display/display_internal.h>

class visualizeTraj
{
public:
    // visualizeTraj();
    // ~visualizeTraj();
    static void step();
    static void run();

private:
};

#endif