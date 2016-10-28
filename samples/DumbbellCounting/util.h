/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * Author: Hua Tang, Fei Duan, Haibing Ren, Ziang Li
 * */

#ifndef UTIL_H
#define UTIL_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "bwlabel.h"
#include "color_GMM.h"

// Calculate distance of face
ushort calcDepthValue(const ushort *ptrDep, const cv::Size &imgSize, int x, int y);

// Detect dumbbell
void locateDumbbell(robot::CColorGMM &gmm, cv::Mat &frame, cv::Rect &bbox, int minSize);

#endif