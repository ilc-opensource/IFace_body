/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * */

#pragma once

#include <opencv2/core/core.hpp>

#if WIN32

#if defined(IHumanOriEstimation_EXPORTS)
#define HOEAPI __declspec(dllexport)
#else
#define HOEAPI __declspec(dllimport)
#endif

#else

#define HOEAPI

#endif


enum HumanView
{
    FRONTAL_VIEW = 0,
    LEFT_VIEW = 1,
    BACK_VIEW = 2,
    RIGHT_VIEW = 3,
    UNKNOW_VIEW = 4,
};

struct HumanOriEstimationValue
{
    HumanView humanOriValue;
    bool angleFlag;
    float pan;
    float tilt;
};

HOEAPI
bool humanOriEstimationInit(const char *headPoseConfigFile, const char *modelDir, const char *hboData, const char *leafNodeFile, bool speed = true);

HOEAPI
void estimateHumanOri(cv::Mat colorFrame, cv::Mat depthFrame, cv::Rect HeadShoulder,
                      HumanOriEstimationValue &humanOriEstimation);

HOEAPI
void humanOriEstimationRelease();
