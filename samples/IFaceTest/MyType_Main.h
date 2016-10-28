/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * */

#ifndef  MY_TYPE_MAIN_H
#define MY_TYPE_MAIN_H

#define FACE_ID_THRESHOLD 0.1
#define MAX_FACE_ID 20
#define FACE_TEMPLATE_MAX_NUM  MAX_FACE_ID
#define MAX_FACE_NUMBER   1600


#define size_smallface  64
#define size_bigface    128

#define ATTRIBUTE_FEATURE_DIM 512

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <opencv2/opencv.hpp> 
#include <opencv/cxcore.h>

using namespace std;

typedef struct Face_Attribute
{
	CvRect FaceRegion;
	int FaceView;

	int FaceID;
	float Prob_FaceID;

	int	   Blink;
	float  Prob_Blink;
	int	   Age;
	float  Prob_Age;
	int	   Smile;
	float  Prob_Smile;
	int	   Gender;
	float  Prob_Gender;	
	double Attribute_Feature[ATTRIBUTE_FEATURE_DIM];
} Face_Attribute;

#endif