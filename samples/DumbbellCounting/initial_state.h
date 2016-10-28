/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * Author: Hua Tang, Fei Duan, Haibing Ren, Ziang Li
 * */

#ifndef INITIAL_STATE_H
#define INITIAL_STATE_H

#include "IHumanDetector_RGBD.h"
#include "RSWrapper.h"
#include "util.h"

void readImgFile(const char *sBaseName, int nFrameNo, cv::Mat &colorImage, cv::Mat &depthImage);

bool getInitialState(
    robot::CColorGMM & gmm,
    cv::Rect & faceBbox,
    int & dumbbellLowerbound,
    cv::Rect & initBbox,
#ifdef LIVE_VIDEO
    RSWrapper * d4p,
#else
    int idxFrameBeg,
    int idxFrameEnd,
    const std::string & dbDir,
#endif
    double & distCurr);

#endif