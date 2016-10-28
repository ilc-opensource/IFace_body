/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * */

#pragma once

#include <opencv/cv.h>

#ifdef __cplusplus
extern "C" {
#endif

#if WIN32

#if defined(IFace3DPoseEstimation_EXPORTS)
#define FPEAPI __declspec(dllexport)
#else
#define FPEAPI __declspec(dllimport)
#endif

#else

#define FPEAPI

#endif


FPEAPI
bool iFacePoseEstimation_WithLandmark(unsigned short *lpDepthMap, int nWidth, int nHeight, CvPoint2D32f *Landmark, int nLandmarkNum, double *dPan_Angle, double *dTilt_Angle);

#ifdef __cplusplus
};
#endif
