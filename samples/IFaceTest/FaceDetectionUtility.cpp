/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * */

#include "opencv/cv.h"
#include "opencv/highgui.h"	
#include "IFaceWrapper.hpp"

#include "MyType.h"
#include "FaceRecognitionUtility.h"

#if defined WIN32 || defined _WIN32 || defined WIN64 || defined _WIN64
#include <windows.h>
#elif defined(__linux__)
#include <time.h>
#endif

IplImage* imgRotation90n(IplImage* srcImage, int angle);

IFaceFaceDetector detector;

long long getTickCounting()
{
#if defined WIN32 || defined WIN64 || defined _WIN64
	LARGE_INTEGER counter;
	QueryPerformanceCounter(&counter);
	return (long long)counter.QuadPart;
#elif(defined __linux__)
	struct timespec tp;
	clock_gettime(CLOCK_MONOTONIC, &tp);
	return (long long)tp.tv_sec * 1000000000 + tp.tv_nsec;
#else
	struct timeval tv;
	struct timezone tz;
	gettimeofday(&tv, &tz);
	return (long long)tv.tv_sec * 1000000 + tv.tv_usec;
#endif
}

double getTickFrequency1()
{
#if defined WIN32 || defined WIN64 || defined _WIN64
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	return (double)freq.QuadPart;
#elif(defined __linux__) || (defined __APPLE__)
	return 1e9;
#else
	return 1e6;
#endif
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Detection Utility
///  written by Ren Haibing 
///   the face model is in the default directory
/// Acknowledge		: modified from Yimin's program
///
/// Function name	: InitFaceDetector
/// Description	    : init face detector 
///
/// Argument		:	
///
/// Return type		: 
///
/// Create Time		: 2014-10-28  14:20
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void InitFaceDetector(const char* modelFile)
{
	tagDetectConfig configParam;
	EnumViewAngle  viewAngle = (EnumViewAngle)VIEW_ANGLE_FRONTAL;
	detector.init(viewAngle, FEA_HAAR, 2, modelFile);//(EnumFeaType)trackerType);
//	detector.config( configParam );
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Detection Utility
///  written by Ren Haibing 
///   the face model is in the default directory
/// Acknowledge		: modified from Yimin's program
///
/// Function name	: FaceDetectionApplication
/// Description	    : face detection and rotate the image with face detection result(in the scenio of no exif file)
///
/// Argument		:	color_image -- source color image
/// Argument		:	gray_image -- source gray image
/// Argument		:	rects -- detected face region
/// Argument		:	MAX_face_numBER -- maximal face number
/// Argument		:	imgExif -- image exif information
///
/// Return type		:  int -- detected face number
///
/// Create Time		: 2014-10-28  16:20
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
#define LARGE_IMAGE_SIZE  20000
#define STANDARD_IMAGE_WIDTH_LONG  2000
#define STANDARD_IMAGE_WIDTH_SMALL  1200
int FaceDetectionApplication(IplImage* color_image, CvRectItem* rects, int MAX_face_numBER, bool bRotateTry)
{
	// 3.1 face detection
	int face_num;
	IplImage *detectImage = NULL;

	if ((color_image->width > LARGE_IMAGE_SIZE) || (color_image->height > LARGE_IMAGE_SIZE))
	{
		int nNewWidth = (color_image->width > color_image->height) ? STANDARD_IMAGE_WIDTH_LONG : STANDARD_IMAGE_WIDTH_SMALL;		
		double dScale = nNewWidth * 1.0 / color_image->width;		
		int nNewHeight = int(color_image->height * dScale);

		detectImage = cvCreateImage(cvSize(nNewWidth, nNewHeight), IPL_DEPTH_8U, color_image->nChannels);
		cvResize(color_image, detectImage);

		detector.setFaceDetectionSizeRange(detectImage);
		//detector.setFaceDetectionROI(detectImage, 0.8);

		face_num = detector.detect(detectImage, rects, 0);

		detector.clearFaceDetectionRange();
		//detector.clearFaceDetectionROI();

		for (int i = 0; i<face_num; i++)
		{
			rects[i].rc.x = int(rects[i].rc.x  / dScale);
			rects[i].rc.y = int(rects[i].rc.y  / dScale);
			rects[i].rc.width = int(rects[i].rc.width  / dScale);
			rects[i].rc.height = int(rects[i].rc.height  / dScale);
		}
	}
	else
	{
		detectImage = cvCloneImage(color_image);
		//detector.setFaceDetectionSizeRange(Detect_Image);
		//detector.setFaceDetectionROI(Detect_Image, 0.8);
		//t1 = getTickCounting();
		
		face_num = detector.detect(detectImage, rects, 0);   //for imagelist input
		
		//t2 = getTickCounting();
		//duration = ((double)(t2 - t1)) / (getTickFrequency1() * 1e-3);
		//printf("detect takes %f ms\n", duration);
		//detector.clearFaceDetectionRange();
		//detector.clearFaceDetectionROI();
	}

	cvReleaseImage(&detectImage);

	return face_num;
}
