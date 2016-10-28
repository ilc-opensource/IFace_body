/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * Author: Hua Tang, Fei Duan, Haibing Ren, Ziang Li
 * */

#ifndef TRACKING_H
#define TRACKING_H

#include <opencv/cv.h>
#include "color_GMM.h"
#include "RSWrapper.h"

namespace robot
{
int fastTracking(
    CColorGMM & gmm,
    cv::Rect & faceBbox,
    int dumbbellLowerbound,
    cv::Rect & initBbox,
#ifdef LIVE_VIDEO
    RSWrapper * d4p,
#else
    int idxFrameBeg,
    int idxFrameEnd,
    const std::string & dbDir,
#endif
    double distCurr,
    int period);
}

#endif