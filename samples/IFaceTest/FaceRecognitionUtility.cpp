/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * */

#include "opencv/cv.h"
#include "opencv/highgui.h"	
#include "IFaceWrapper.hpp"

#include "MyType.h"
#include "FaceDetectionUtility.h"
#include "FaceRecognitionUtility.h"
#include "PhotoIndexingUtility.h"

IFaceFaceRecognizer *pfaceRecognizer = NULL;
IFaceBlinkDetector*  pblinkDetector = NULL;
IFaceSmileDetector*  psmileDetector = NULL;
IFaceGenderDetector* pgenderDetector = NULL;
IFaceAgeDetector*    pageDetector = NULL;
IFaceLandmarkDetector* plandmarkDetector = NULL;
IFaceAlignFace cutFace(size_smallface, size_bigface);

int Face_Valid_Flag[MAX_FACE_NUMBER];	
int nFaceSetSize;
Face_Attribute FaceRecognitionResult[MAX_FACE_NUMBER];
Face_Attribute ProfileFaceRecognitionResult[MAX_FACE_NUMBER];
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Recognition Utility
///  written by Ren Haibing 
///   the face model is in the default directory
/// Acknowledge		: modified from Yimin's program
///
/// Function name	: InitFaceRecognition
/// Description	    : init face recognizer 
///
/// Argument		:	
///
/// Return type		: 
///
/// Create Time		: 2014-11-2  14:20
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void InitFaceRecognition(const char *facemodel, const char *blinkmodel, const char *smilemod, const char *gendermod, const char *agemod, const char *opencv_input_xmlEyeLeftCorner, const char *opencv_input_xmlMthLeftCorner, const char *opencv_input_xmlNose,const char *str_faceModelXML)
{
	pfaceRecognizer = new IFaceFaceRecognizer(size_bigface, 2, facemodel);
	pfaceRecognizer->loadFaceModelXML(str_faceModelXML);
	nFaceSetSize = pfaceRecognizer->getFaceSetSize();

	pblinkDetector = new IFaceBlinkDetector(size_smallface, blinkmodel);

	psmileDetector = new IFaceSmileDetector(size_smallface, smilemod);

	pgenderDetector = new IFaceGenderDetector(size_smallface, gendermod);

	pageDetector = new IFaceAgeDetector(size_bigface, agemod);

	plandmarkDetector = new IFaceLandmarkDetector(LDM_6PT, opencv_input_xmlEyeLeftCorner, opencv_input_xmlMthLeftCorner, opencv_input_xmlNose);
}

void FaceRecognition_Release()
{
	if (pfaceRecognizer)
	{
		delete pfaceRecognizer;
		pfaceRecognizer = NULL;
	}		

	if (pblinkDetector)
	{
		delete pblinkDetector;
		pblinkDetector = NULL;
	}		

	if (psmileDetector)
	{
		delete psmileDetector;
		psmileDetector = NULL;
	}		

	if (pgenderDetector)
	{
		delete pgenderDetector;
		pgenderDetector = NULL;
	}		

	if (pageDetector)
	{
		delete pageDetector;
		pageDetector = NULL;
	}		

	if (plandmarkDetector)
	{
		delete plandmarkDetector;
		plandmarkDetector = NULL;
	}		
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Recognition Utility
///  written by Ren Haibing 
/// Acknowledge		: modified from Yimin's program
///
/// Function name	: GetRecognizedFaceName
/// Description	    : Get the face name with the recognized face ID 
///
/// Argument		:	nFaceSetID -- face ID in the face model DB
///
/// Return type		: 
///
/// Create Time		: 2014-11-5  10:50
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
const char *GetRecognizedFaceName(int nFaceSetID)
{
	return pfaceRecognizer->getFaceName(nFaceSetID);
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Recognition Utility
///  written by Ren Haibing 
/// Acknowledge		: modified from Yimin's program
///
/// Function name	: FaceRecognitionApplication
/// Description	    : face identification and face attribute recognition 
///
/// Argument		:	color_image -- source image
/// Argument		:	nFace_Num -- detected face number
/// Argument		:	rects -- detected face region
///
/// Return type		: 
///
/// Create Time		: 2014-11-5  10:56
///
///
/// Side Effect		: int Face_Valid_Flag[MAX_face_numBER] -- face valid flag array
///                   Face_Attribute FaceRecognitionResult[MAX_face_numBER] -- final recogntion result 
///////////////////////////////////////////////////////////////////////////////////////////////
void FaceRecognitionApplication(IplImage* color_image, int nFace_Num, CvRectItem* rects)
{
	
	bool   DoBlink = true, DoSmile = true, DoGender = true, DoAge = true;
	float  smile_threshold, blink_threshold, gender_threshold; 
	int    bBlink = 0, bSmile = 0, bGender = 0;  //+1, -1, otherwise 0: no process 
	int    nAgeID = 0;
	float  probBlink = 0, probSmile = 0, probGender = 0, probAge[4];

	// config landmark detector ------------------------------------


	bool  bLandmark = false;
	CvPoint2D32f   landmark6[6+1]; // consider both 6-pt and 7-pt

	float probFaceID;
	int nFaceSetID;

	// blink/smile/gender/age/face recognize section
	for( int i=0; i< nFace_Num; i++ )
	{
		Face_Valid_Flag[i] = 0;
		bSmile = bBlink = bGender = -1;
		probSmile = 0, probBlink = 0, probGender = 0;		

		// get face rect and id from face tracker
		CvRect rect = rects[i].rc;
		//int    face_trackid = rects[i].fid;
		//float  like = rects[i].prob;
		int    angle= rects[i].angle;
				
		FaceRecognitionResult[i].FaceRegion = rect;
		FaceRecognitionResult[i].FaceView = 0;//frontal view

		// filter out outer faces
		if (rect.x + rect.width  > color_image->width || rect.x < 0) continue;
		if (rect.y + rect.height > color_image->height || rect.y < 0) continue;
		if (rect.width<color_image->width * 0.03) continue;
				
		// Landmark detection -----------------------------------------------------
		bLandmark = plandmarkDetector->detect(color_image, &rect, landmark6, NULL, angle); //for imagelist input
		if(bLandmark == false) continue;
		cutFace.init(color_image, rect, landmark6);

		Face_Valid_Flag[i] = 1;   

		// detect blink----------------------------------------------
		bBlink = 0;	
		probBlink = 0;
		if (DoBlink)
		{
			blink_threshold = pblinkDetector->getDefThreshold();//0.5;
			pblinkDetector->predict(&cutFace, &probBlink);
					
			if(probBlink > blink_threshold )
				bBlink = 0;//1; //eye close
			else 
				bBlink = 1;//0; //eye open
			FaceRecognitionResult[i].Blink = bBlink;
			FaceRecognitionResult[i].Prob_Blink = probBlink;
		}

		// detect smile -----------------------------------------------------------
		bSmile    = 0;	
		probSmile = 0;
		if (DoSmile)
		{	
			smile_threshold = psmileDetector->getDefThreshold(); //0.42;
			psmileDetector->predict(&cutFace, &probSmile);

			if(probSmile > smile_threshold)
				bSmile = 1;  //smile
			else 
				bSmile = 0; //not smile
			FaceRecognitionResult[i].Smile = bSmile;
			FaceRecognitionResult[i].Prob_Smile = probSmile;
		}
			
		//detect gender --------------------------------------------------------
		bGender    = 0;	
		probGender = 0;
		if(DoGender)
		{
			gender_threshold = pgenderDetector->getDefThreshold(); // 0.42;

			//cvSaveImage("c:/temp/gender.jpg", cutFace.getBigCutFace());
			pgenderDetector->predict(&cutFace, &probGender);


			if(probGender > gender_threshold)
				bGender =  1; //female
			else
				bGender =  0; //male
			FaceRecognitionResult[i].Gender = bGender;
			FaceRecognitionResult[i].Prob_Gender = probGender;
		}

		// estmage age -------------------------------------------------------------
		if(DoAge)
		{
			//nAgeID = 0:"Baby", 1:"Kid", 2:"Adult", 3:"Senior"
			nAgeID = pageDetector->predict(&cutFace, probAge);
			FaceRecognitionResult[i].Age = nAgeID;
			FaceRecognitionResult[i].Prob_Age = probAge[nAgeID];
		}

		//Face Recognition ---------------------------------------------------------
		if(bLandmark) // aligned face is needed
		{
			nFaceSetID = pfaceRecognizer->predict(&cutFace, &probFaceID);
			FaceRecognitionResult[i].FaceID = nFaceSetID;
			FaceRecognitionResult[i].Prob_FaceID = probFaceID;
		}

	}//for( int i=0; i< nFace_Num; i++ )
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Recognition Utility
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: GetFrontalFace
/// Description	    : get face region from  face attribute recognition resutl 
///
/// Argument		:	FrontalFaceRegoin -- face region
/// Argument		:	nFace_Num -- face number
///
/// Return type		:  
///
/// Create Time		: 2014-12-29  10:31
///
///
/// Side Effect		: 
///                   
///////////////////////////////////////////////////////////////////////////////////////////////
void GetFrontalFace(cv::Rect *FrontalFaceRegoin, int nFace_Num)
{
	for(int i=0;i<nFace_Num;i++)
	{
		FrontalFaceRegoin[i].x = FaceRecognitionResult[i].FaceRegion.x;
		FrontalFaceRegoin[i].y = FaceRecognitionResult[i].FaceRegion.y;
		FrontalFaceRegoin[i].width = FaceRecognitionResult[i].FaceRegion.width;
		FrontalFaceRegoin[i].height = FaceRecognitionResult[i].FaceRegion.height;
	}
}