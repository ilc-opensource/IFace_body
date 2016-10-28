/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * */

#pragma once

#include <opencv2/core/core.hpp>

#if WIN32

#if defined(IHumanDetector_RGBD_EXPORTS)
#define HDRGBDAPI __declspec(dllexport)
#else
#define HDRGBDAPI __declspec(dllimport)
#endif

#else

#define HDRGBDAPI

#endif

typedef struct HdRect
{
    int	x;
    int	y;
    int	width;
    int	height;
    float view;
    int neighbors;
    double confidence;
    ushort depth;

    HdRect() :
        x(0), y(0), width(0), height(0), view(0), neighbors(0), confidence(0)
    {};
    HdRect(const int& _x, const int& _y, const int& _width, const int& _height,
        float _view = 0.0f, int _neighbors = 0, double _confidence = 0.0) :
        x(_x), y(_y), width(_width), height(_height),
        view(_view), neighbors(_neighbors), confidence(_confidence)
    {};

    HdRect(const HdRect& init) :
        x(init.x), y(init.y), width(init.width), height(init.height),
        view(init.view), neighbors(init.neighbors), confidence(init.confidence)
    {};

    HdRect operator=(const HdRect& _leftOpt)
    {
        x = _leftOpt.x;
        y = _leftOpt.y;
        width = _leftOpt.width;
        height = _leftOpt.height;
        view = _leftOpt.view;
        neighbors = _leftOpt.neighbors;
        confidence = _leftOpt.confidence;
        return *this;
    };
} HdRect;


HDRGBDAPI void initRGBDHumanDetector(double winScaleRatio = 1.4f);

HDRGBDAPI int detectHumanInRGBDImage(cv::Mat rgbImage, cv::Mat depthImage, HdRect*& human);

HDRGBDAPI void humanDetectionRelease();

HDRGBDAPI void setHumanROI(cv::Rect humanROI);

HDRGBDAPI void clearHumanROI();

HDRGBDAPI void setHumanSizeRange(int nMinSize, int nMaxSize);

HDRGBDAPI void clearHumanSizeRange();
