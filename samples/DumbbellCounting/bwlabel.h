/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * Author: Hua Tang, Fei Duan, Haibing Ren, Ziang Li
 * */

#ifndef BWLABEL_H
#define BWLABEL_H

#include <opencv/cv.h>
#include <vector>

namespace robot
{
struct conn_comp_t
{
    int label;
    std::vector<cv::Point> coords;
    cv::Rect bbox;
};

int bwlabel(IplImage * img, int * labels, int n = 8);
int bwlabel(const cv::Mat & img, cv::Mat & lblMap, int n = 8);
int bwlabel(const cv::Mat & img, cv::Mat & lblMap,
            std::vector<conn_comp_t> & cc, int n = 8);

int findMaxCc(const std::vector<conn_comp_t> & cc);
bool findMaxCc(const cv::Mat & img, conn_comp_t & ccMax);
}

#endif