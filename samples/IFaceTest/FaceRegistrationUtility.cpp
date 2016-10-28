/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * */

#include "opencv/cv.h"
#include "opencv/highgui.h"	
#include "IFaceWrapper.hpp"
#include "MyType.h"

IFaceFaceAnalyzer *m_faceAnalyzer = NULL;
IplImage* imgRotation90n(IplImage* srcImage, int angle);
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Registration Utility
///  written by Ren Haibing 
///   the face model is in the default directory
/// Acknowledge		:
///
/// Function name	: FaceRegistration_Init
/// Description	    : init IflibFaceAnalyzer for face registration  
///
/// Argument		:	
///
/// Return type		: 
///
/// Create Time		: 2014-11-18  10:20
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void FaceRegistration_Init(const char *str_facesetxml, const char * xmlEyeLeftCorner, const char * xmlMthLeftCorner, const char *sFaceRecognizer_modelpath)
{
	if(m_faceAnalyzer == NULL)
	{
		EnumTrackerType traType   = TRA_HAAR; //TRA_PF;
		EnumViewAngle   viewAngle = VIEW_ANGLE_FRONTAL; //VIEW_ANGLE_HALF_MULTI; //

		//int  sampleRate = 1;
		int recognizerType = RECOGNIZER_CAS_GLOH;  //RECOGNIZER_BOOST_GB240
		bool bEnableAutoCluster =  false;//false;//true;
		bool bEnableShapeRegressor =  true;//false;//true;

		m_faceAnalyzer = new IFaceFaceAnalyzer(
            viewAngle, traType, 0,
			str_facesetxml, recognizerType, bEnableAutoCluster, bEnableShapeRegressor, xmlEyeLeftCorner, xmlMthLeftCorner, sFaceRecognizer_modelpath);
	}

}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Registration Utility
///  written by Ren Haibing 
///  
/// Acknowledge		:
///
/// Function name	: FaceRegistration_DetectFace
/// Description	    : Given a image file name, detect the face region and insert it to template array  
///
/// Argument		:	sImageFilename -- input image file name
/// Argument		:	sDesFaceName -- save the detected face region
/// Argument		:	m_FaceTemplate -- face template array
/// Argument		:	nTotalFaceNum -- valid face template size
///
/// Return type		: 
///
/// Create Time		: 2014-11-18  10:41
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
#define LARGE_IMAGE_SIZE  2000
#define STANDARD_IMAGE_WIDTH_LONG  2000
#define STANDARD_IMAGE_WIDTH_SMALL  1200

int FaceRegistration_DetectFace(char *sImageFilename, char *sDesFaceName, IplImage *m_FaceTemplate[FACE_TEMPLATE_MAX_NUM], int *nTotalFaceNum)
{
	int nFaceNum = 0;
	IplImage *colorImage = cvLoadImage(sImageFilename);
			
	if (colorImage == NULL)
	{
		return 0;
	}

	if ((colorImage->width > LARGE_IMAGE_SIZE) || (colorImage->height > LARGE_IMAGE_SIZE))
	{
		int nNewWidth = (colorImage->width > colorImage->height) ? STANDARD_IMAGE_WIDTH_LONG : STANDARD_IMAGE_WIDTH_SMALL;
		double dScale = nNewWidth * 1.0 / colorImage->width;
		int nNewHeight = int(colorImage->height * dScale);

		IplImage *detectImage = cvCreateImage(cvSize(nNewWidth, nNewHeight), IPL_DEPTH_8U, colorImage->nChannels);
		cvResize(colorImage, detectImage);

		cvReleaseImage(&colorImage);
		colorImage = detectImage;
	}

	char sFilename[1024];
	sprintf(sFilename, "%s_%d.jpg",sDesFaceName, *nTotalFaceNum);

	bool bGetGoodFace = m_faceAnalyzer->faceDetection(colorImage, 80, sFilename);
	
	if(bGetGoodFace)
	{		
		IplImage *lpFace = m_faceAnalyzer->getBigCutFace();
					
		m_FaceTemplate[*nTotalFaceNum] = cvCreateImage(cvSize(lpFace->width, lpFace->height), IPL_DEPTH_8U,1);
		cvCopy(lpFace, m_FaceTemplate[*nTotalFaceNum]);	

		if (*nTotalFaceNum < FACE_TEMPLATE_MAX_NUM - 1)
		{
			++(*nTotalFaceNum);
			nFaceNum = 1;
		}

	}
	
	cvReleaseImage(&colorImage);

	return nFaceNum;
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Registration Utility
///  written by Ren Haibing 
///  
/// Acknowledge		:
///
/// Function name	: FaceRegistration_AddUser
/// Description	    : register the user with the template array  
///
/// Argument		:	sUserName -- user name
/// Argument		:	m_FaceTemplate -- face template array
/// Argument		:	nTotalFaceNum -- valid face template size
///
/// Return type		: 
///
/// Create Time		: 2014-11-18  10:45
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void FaceRegistration_AddUser(const char *str_facesetxml, char *sUserName, IplImage *m_FaceTemplate[FACE_TEMPLATE_MAX_NUM], int nTotalFaceNum)
{
	int nFaceSetIdx = -1;
	//int nFaceSetID = -1;
	nFaceSetIdx = m_faceAnalyzer->insertEmptyFaceSet(sUserName);
				
	for(int i = 0; i < nTotalFaceNum; ++i)
	{
		m_faceAnalyzer->tryInsertFace(m_FaceTemplate[i], nFaceSetIdx,  true);
		//nFaceSetID = m_faceAnalyzer->getFaceSetID(nFaceSetIdx);
	}

	m_faceAnalyzer->saveFaceModelXML(str_facesetxml);	
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Registration Utility
///  written by Ren Haibing 
///  
/// Acknowledge		:
///
/// Function name	: FaceRegistration_Release
/// Description	    : release the memory for face registration 
///
/// Argument		:
///
/// Return type		: 
///
/// Create Time		: 2014-11-18  10:50
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void FaceRegistration_Release()
{
	if(m_faceAnalyzer)
	{
		delete m_faceAnalyzer;
		m_faceAnalyzer = NULL;
	}
}