/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * */

#ifndef  FACE_REGISTRATION_UTILITY_H
#define FACE_REGISTRATION_UTILITY_H

void FaceRegistration_Init(const char *str_facesetxml, const char * xmlEyeLeftCorner, const char * xmlMthLeftCorner, const char *sFaceRecognizer_modelpath);
void FaceRegistration_AddUser(const char *str_facesetxml, char *sUserName, IplImage *m_FaceTemplate[FACE_TEMPLATE_MAX_NUM], int nTotalFaceNum);
void FaceRegistration_Release();

int FaceRegistration_DetectFace(char *sImageFilename, char *sDesFaceName, IplImage *m_FaceTemplate[FACE_TEMPLATE_MAX_NUM], int *nTotalNum);

#endif