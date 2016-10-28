/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * */

#ifndef  FACE_DETECTION_UTILITY_H
#define FACE_DETECTION_UTILITY_H

//#include "MyType.h"
//#include "FaceRecognitionUtility.h"

// Detect frontaql faces
void InitFaceDetector(const char* modelFile);
int FaceDetectionApplication(IplImage* color_image, CvRectItem* rects, int MAX_face_numBER, bool bRotateTry);
long long getTickCounting();
double getTickFrequency1();
#endif