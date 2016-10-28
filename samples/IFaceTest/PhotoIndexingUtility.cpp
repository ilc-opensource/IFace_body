/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * */

#include "opencv/cv.h"
#include "opencv/highgui.h"	
#include "IFaceWrapper.hpp"
#include "opencv2/core.hpp"

#include "MyType_Main.h"
#include "FaceDetectionUtility.h"
#include "FaceRecognitionUtility.h"
#include "FaceRegistrationUtility.h"

#include <time.h>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

extern string thumbnail;
extern string faceTestDir;
extern string faceModelXML;

#define ITER_NUM 1

extern Face_Attribute FaceRecognitionResult[MAX_FACE_NUMBER];
extern int Face_Valid_Flag[MAX_FACE_NUMBER];

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
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
bool PhotoIndexing_Init(string& configure_file)
{
	if (configure_file.empty())
	{
		fprintf(stderr, "please confirm your configure file\n");
		return false;
	}
	 
	cv::FileStorage fs;
	string opencv_input_facemod;
	string opencv_input_blinkmod;
	string opencv_input_smilemod;
	string opencv_input_gendermod;
	string opencv_input_agemod;
	string opencv_input_xmlEyeLeftCorner;
	string opencv_input_xmlMthLeftCorner;
	string opencv_input_xmlNose;
	string opencv_input_newSkinLookupTable;

	fs.open(configure_file.c_str(), cv::FileStorage::READ);

	if (!fs.isOpened())
	{
		fprintf(stderr, "invalid configure file: %s\n", configure_file.c_str());
		return false;
	}

	fs["thumbnail"] >> thumbnail;
	fs["faceTestDir"] >> faceTestDir;
	fs["faceModelXML"] >> faceModelXML;
	fs["opencv_input_facemod"] >> opencv_input_facemod;
	fs["opencv_input_blinkmod"] >> opencv_input_blinkmod;
	fs["opencv_input_smilemod"] >> opencv_input_smilemod;
	fs["opencv_input_gendermod"] >> opencv_input_gendermod;
	fs["opencv_input_agemod"] >> opencv_input_agemod;
	fs["opencv_input_xmlEyeLeftCorner"] >> opencv_input_xmlEyeLeftCorner;
	fs["opencv_input_xmlMthLeftCorner"] >> opencv_input_xmlMthLeftCorner;
	fs["opencv_input_xmlNose"] >> opencv_input_xmlNose;
	fs["opencv_input_newSkinLookupTable"] >> opencv_input_newSkinLookupTable;

	fs.release();
	
	const char *str_faceModelXML = faceModelXML.c_str();
	
	fstream ifs;
	ifs.open(str_faceModelXML, fstream::in);
	if (!ifs.is_open())
	{
		fprintf(stderr, "can not open faceModelXML file: %s\n", str_faceModelXML);
		return false;
	}

	cout << "Begin Photo Indexing Initialization " << endl;
	// Face Recognition -------------------------------------------------------
	InitFaceRecognition(opencv_input_facemod.c_str(), opencv_input_blinkmod.c_str(), opencv_input_smilemod.c_str(), opencv_input_gendermod.c_str(), opencv_input_agemod.c_str(), opencv_input_xmlEyeLeftCorner.c_str(), opencv_input_xmlMthLeftCorner.c_str(), opencv_input_xmlNose.c_str(), str_faceModelXML);

	// Face Detector ---------------------------------------------
	InitFaceDetector(opencv_input_newSkinLookupTable.c_str());

	//scene classification --------------------------------------------------------
	//initializeScene("./scene_model");

	//Registration
	FaceRegistration_Init(str_faceModelXML, opencv_input_xmlEyeLeftCorner.c_str(), opencv_input_xmlMthLeftCorner.c_str(), opencv_input_facemod.c_str());

	cout << "Finish photo indexing initialization  " << endl << endl;

	return true;
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///  written by Ren Haibing 
///   
/// Acknowledge		:
///
/// Function name	: PhotoIndexing_Release
/// Description	    : release memory for photo indexing 
///
/// Argument		:	
///
/// Return type		: 
///
/// Create Time		: 2014-11-18  10:32
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void PhotoIndexing_Release()
{
	//destroyScene();
	FaceRecognition_Release();
	FaceRegistration_Release();
//	ReleaseProfileFaceDetector();
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///  written by Zhang, Yimin
///   
/// Acknowledge		:
///
/// Function name	: imgRotation90n
/// Description	    : rotate the image, given the rotation angle 
///
/// Argument		:	srcImage -- input image
/// Argument		:	angle -- rotation angle(1: 90,  2: 180 degree 3: 270 )
///
/// Return type		:  IplImage -- rotated image
///
/// Create Time		: 2014-11-18  10:32
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
//rotate (counter clockwise) an image by angle (1: 90,  2: 180 degree 3: 270 )
IplImage* imgRotation90n(IplImage* srcImage, int angle)
{
	assert(angle==1 || angle ==2 || angle ==3); 
	
	IplImage* dstImage = NULL;

	if (srcImage == NULL)
		return NULL;

	//set the center of rotation   
    CvPoint2D32f center;     
    center.x=float (srcImage->width/2);   
    center.y=float (srcImage->height/2);  

    //set the rotation matrix  
    float m[6];               
    CvMat M = cvMat( 2, 3, CV_32F, m );   
    cv2DRotationMatrix( center, angle*90,1, &M);   

	if (angle==2)
	{
		dstImage = cvCreateImage (cvSize(srcImage->width,srcImage->height), srcImage->depth,srcImage->nChannels);
	    //rotate the image   
        cvWarpAffine(srcImage,dstImage, &M,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0) ); 
	}
	else
	{
		int maxHW = max(srcImage->width,srcImage->height); //the max of Height and Width

	    // Adjust rotation center to dst's center,
	    m[2] += (maxHW - srcImage->width) / 2;
	    m[5] += (maxHW - srcImage->height) / 2;
		dstImage = cvCreateImage (cvSize(srcImage->height,srcImage->width), srcImage->depth,srcImage->nChannels);
		IplImage* tmpImage = cvCreateImage (cvSize(maxHW, maxHW), srcImage->depth,srcImage->nChannels);
	    //rotate the image   
        cvWarpAffine(srcImage,tmpImage, &M,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0) ); 

		if ( (srcImage->width) >= (srcImage->height))
		  cvSetImageROI(tmpImage,cvRect((maxHW-srcImage->height)/2,0, srcImage->height,srcImage->width));
		else
		  cvSetImageROI(tmpImage,cvRect(0,(maxHW-srcImage->width)/2,srcImage->height,srcImage->width));
 
		cvCopy(tmpImage, dstImage);  //just copy the ROI area
		cvReleaseImage(&tmpImage);
	}
	return dstImage;
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///  written by Zhang, Yimin; modified by Ren, Haibing
///   
/// Acknowledge		:
///
/// Function name	: PhotoIndexing_ImageEvaluation
/// Description	    : given image name, generate the label information for photo indexing 
///
/// Argument		:	sFilename -- input image file name
/// Argument		:	*scene_label -- scene classification result
///
/// Return type		:  int -- detected face number
///
/// Create Time		: 2014-11-18  13:10
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
#define LARGE_IMAGE_SIZE  2048
#define STANDARD_IMAGE_WIDTH_LONG  LARGE_IMAGE_SIZE
#define STANDARD_IMAGE_WIDTH_SMALL  1200

int PhotoIndexing_ImageEvaluation(char* sFilename,int *scene_label)
{
	IplImage* color_image = cvLoadImage(sFilename);
	if(color_image == NULL) return -1;

	CvRectItem rects[MAX_FACE_NUMBER];

//	TickMeter	time_Cascades, photo_time_Cascades;
	
	int face_num;
	IplImage *Detect_Image;
	int nNewWidth, nNewHeight;
	double dScale;

	if ((color_image->width>LARGE_IMAGE_SIZE) || (color_image->height>LARGE_IMAGE_SIZE))
	{
		if (color_image->width>color_image->height)
			nNewWidth = STANDARD_IMAGE_WIDTH_LONG;
		else nNewWidth = STANDARD_IMAGE_WIDTH_SMALL;

		dScale = nNewWidth * 1.0 / color_image->width;

		nNewHeight = int(color_image->height * dScale);
		Detect_Image = cvCreateImage(cvSize(nNewWidth, nNewHeight), IPL_DEPTH_8U, color_image->nChannels);
		cvResize(color_image, Detect_Image);

		face_num = FaceDetectionApplication(color_image, rects, MAX_FACE_NUMBER, false);

		for (int i = 0; i<face_num; i++)
		{
			rects[i].rc.x = int(rects[i].rc.x / dScale);
			rects[i].rc.y = int(rects[i].rc.y / dScale);
			rects[i].rc.width = int(rects[i].rc.width / dScale);
			rects[i].rc.height = int(rects[i].rc.height / dScale);
		}
	}
	else
	{

		face_num = FaceDetectionApplication(color_image, rects, MAX_FACE_NUMBER, false);

	}
	// face detection
	//time_Cascades.reset(); time_Cascades.start();
	///long long t1 = getTickCounting();
	
	
	cout << " Face Number: = " << face_num << endl;
	//time_Cascades.stop();
	///long long t2 = getTickCounting();
	///double duration = ((double)(t2 - t1)) / (getTickFrequency1() * 1e-3);
	///printf("face detection takes %f ms\n", duration);
	///cout << "Face number: " << face_num<<endl;
	//cout<<"Face detection time: "<<time_Cascades.getTimeSec()<<endl;

	//3.3 face attribute recognition and identification
	//time_Cascades.reset(); time_Cascades.start();
	///t1 = getTickCounting();
	FaceRecognitionApplication(color_image, face_num, rects);
	///t2 = getTickCounting();
	///duration = ((double)(t2 - t1)) / (getTickFrequency1() * 1e-3);
	///printf("face recognition takes %f ms\n", duration);
	///cout << "Face number: " << face_num << endl;
	//time_Cascades.stop();
	//cout<<"Face recognition time: "<<time_Cascades.getTimeSec()<<endl;


	//3.2 scene classification
	/*time_Cascades.reset(); time_Cascades.start();
	*scene_label = Scene_Classification(color_image, rects, Face_Valid_Flag, face_num);
	time_Cascades.stop();
	cout<<"Scene classification time: "<<time_Cascades.getTimeSec()<<endl;*/

	cvReleaseImage(&color_image);
	return face_num;
}


int PhotoIndex_ImageEvaluation(IplImage* color_image)
{
	if (color_image == NULL) return -1;

	CvRectItem rects[MAX_FACE_NUMBER];

	//	TickMeter	time_Cascades, photo_time_Cascades;

	int face_num;
	IplImage *Detect_Image;
	int nNewWidth, nNewHeight;
	double dScale;

	if ((color_image->width>LARGE_IMAGE_SIZE) || (color_image->height>LARGE_IMAGE_SIZE))
	{
		
		if (color_image->width>color_image->height)
			nNewWidth = STANDARD_IMAGE_WIDTH_LONG;
		else nNewWidth = STANDARD_IMAGE_WIDTH_SMALL;

		dScale = nNewWidth * 1.0 / color_image->width;

		nNewHeight = int(color_image->height * dScale);
		Detect_Image = cvCreateImage(cvSize(nNewWidth, nNewHeight), IPL_DEPTH_8U, color_image->nChannels);
		cvResize(color_image, Detect_Image);

		face_num = FaceDetectionApplication(color_image, rects, MAX_FACE_NUMBER, false);

		for (int i = 0; i<face_num; i++)
		{
			rects[i].rc.x = int(rects[i].rc.x / dScale);
			rects[i].rc.y = int(rects[i].rc.y / dScale);
			rects[i].rc.width = int(rects[i].rc.width / dScale);
			rects[i].rc.height = int(rects[i].rc.height / dScale);
		}
	}
	else
	{

		face_num = FaceDetectionApplication(color_image, rects, MAX_FACE_NUMBER, false);

	}
	// face detection
	//time_Cascades.reset(); time_Cascades.start();
	///long long t1 = getTickCounting();


	cout << " Face Number: = " << face_num << endl;
	//time_Cascades.stop();
	///long long t2 = getTickCounting();
	///double duration = ((double)(t2 - t1)) / (getTickFrequency1() * 1e-3);
	///printf("face detection takes %f ms\n", duration);
	///cout << "Face number: " << face_num<<endl;
	//cout<<"Face detection time: "<<time_Cascades.getTimeSec()<<endl;

	//3.3 face attribute recognition and identification
	//time_Cascades.reset(); time_Cascades.start();
	///t1 = getTickCounting();
	FaceRecognitionApplication(color_image, face_num, rects);
	///t2 = getTickCounting();
	///duration = ((double)(t2 - t1)) / (getTickFrequency1() * 1e-3);
	///printf("face recognition takes %f ms\n", duration);
	///cout << "Face number: " << face_num << endl;
	//time_Cascades.stop();
	//cout<<"Face recognition time: "<<time_Cascades.getTimeSec()<<endl;


	//3.2 scene classification
	/*time_Cascades.reset(); time_Cascades.start();
	*scene_label = Scene_Classification(color_image, rects, Face_Valid_Flag, face_num);
	time_Cascades.stop();
	cout<<"Scene classification time: "<<time_Cascades.getTimeSec()<<endl;*/

	//cvReleaseImage(&color_image);
	return face_num;
}