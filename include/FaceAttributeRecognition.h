/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * */

#pragma once

#include "IFaceWrapper.hpp"

#if WIN32

#if defined(ISceneClassification_EXPORTS)
#define FARAPI __declspec(dllexport)
#else
#define FARAPI __declspec(dllimport)
#endif

#else

#define FARAPI

#endif

class FARAPI CFaceAttributeRecognition
{
public:
    // land mark detection
    IFaceLandmarkDetector *m_LandmarkDetector;
    CvPoint2D32f m_Landmark6[6+1];
    bool LandmarkDetection_WithHeadShoulder(cv::Mat greyImage, CvRect HeadShoulder);


    IFaceAlignFace *m_CutFace;

    // attribute recognition
    float  m_probBlink, m_probSmile, m_probGender, m_probAge[4], m_probFaceID;
    int    m_bBlink, m_bSmile, m_bGender;  // +1, -1, otherwise 0: no process
    int    m_nAgeID, m_nFaceSetID;
    char   m_sFacename[128];

    IFaceFaceDetector   *m_FaceDetector;
    IFaceBlinkDetector  *m_blinkDetector;
    IFaceSmileDetector  *m_smileDetector;
    IFaceGenderDetector *m_genderDetector;
    IFaceAgeDetector    *m_ageDetector;
    IFaceFaceRecognizer *m_FaceRecognizer;

    void AttributeRecognition();
    int FaceRecognition();
public:
    CFaceAttributeRecognition(void);
    CFaceAttributeRecognition(const char *opencv_config);
    ~CFaceAttributeRecognition(void);
};

