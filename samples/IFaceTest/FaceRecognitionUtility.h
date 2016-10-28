/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * */

#ifndef  FACE_RECOGNITION_UTILITY_H
#define FACE_RECOGNITION_UTILITY_H

#include "opencv/cv.h"
#include "opencv/highgui.h"	
#include "basetypes.hpp"

void InitFaceRecognition(const char *facemodel, const char *blinkmodel, const char *smilemod, const char *gendermod, const char *agemod,  const char *opencv_input_xmlEyeLeftCorner, const char *opencv_input_xmlMthLeftCorner, const char *opencv_input_xmlNose, const char *str_faceModelXML);
void FaceRecognition_Release();
const char *GetRecognizedFaceName(int nFaceSetID);
void FaceRecognitionApplication(IplImage* gray_image, int face_num,CvRectItem* rects);

void GetFrontalFace(cv::Rect *FrontalFaceRegoin, int nFace_Num);

#endif