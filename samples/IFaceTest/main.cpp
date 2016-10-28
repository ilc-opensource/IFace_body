/**
*** Copyright (C) 1985-2011 Intel Corporation.  All rights reserved.
***
*** The information and source code contained herein is the exclusive
*** property of Intel Corporation and may not be disclosed, examined
*** or reproduced in whole or in part without explicit written authorization
*** from the company.
***
*** Embedded Application Lab, Intel Labs China.
**/
#include "RSWrapper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <time.h>

#ifdef WIN32
#include <direct.h>
#define do_mkdir(_X_) _mkdir(_X_)
#define do_rmdir(_X_) _rmdir(_X_)
#else
#include <sys/stat.h>
#include <sys/types.h>
#define do_mkdir(_X_) mkdir(_X_, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)
#define do_rmdir(_X_) rmdir(_X_)
#endif

#include "cxoptions.hpp"

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "basetypes.hpp"
#include "IFaceWrapper.hpp"

#define _MY_DEBUG


#include <opencv2/opencv.hpp>

using namespace std;

//#include <afx.h>
#include "MyType_Main.h"
#include "FaceDetectionUtility.h"
#include "FaceRecognitionUtility.h"
#include "FaceRegistrationUtility.h"
#include "PhotoIndexingUtility.h"

extern Face_Attribute FaceRecognitionResult[MAX_FACE_NUMBER];
extern Face_Attribute ProfileFaceRecognitionResult[MAX_FACE_NUMBER];
extern int Face_Valid_Flag[MAX_FACE_NUMBER];
extern int nFaceSetSize;

string thumbnail;
string faceTestDir;
string faceModelXML;

void usage()
{
	cout << "Usage:" << endl;
	cout << "  -c <configure_file> -r <image_list.txt> <username> " << endl;
	cout << "  -c <configure_file> -i <image_list.txt>" << endl;
	cout << "  -c <configure_file> -i <image_list.txt> -o <output_directory>" << endl;

}

void createDirectoryRecursively(const char *pathname)
{
	char *str = (char*)malloc(sizeof(pathname) + 1);
	strcpy(str, pathname);

	char *ptr = str;
	while (1) {
		ptr = strchr(ptr, '/');
		if (ptr == nullptr)
			ptr = strchr(str, '\\');

		if (ptr == nullptr) {
			do_mkdir(str);
			break;
		}

		if (*(ptr - 1) == '.') {
			++ptr;
			continue;
		}

		char chr = *ptr;
		*ptr = '\0';

		do_mkdir(str);

		*ptr = chr;
		++ptr;
	}
}

int evaluateImgList(string &imgList, string &destFolder)
{
	int len = 0;

	if (imgList.empty()) {
		cout << "Usage: IFaceTest.exe -c <configure_file> -i <image_list.txt> -o <output_directory>" << endl;
		cout << "You must specify a image list" << endl << endl;
		return -1;
	}

	FILE *fpImgList = fopen(imgList.c_str(), "rt");

	if (fpImgList == nullptr) {
		cout << imgList << " doesn't exist" << endl;
		return -1;
	}

	cout << "Image List File: " << imgList << endl;

	if (destFolder.empty()) {
		cout << "You didn't specify directory containing recognition results, results will be saved in where original Images located" << endl << endl;
	}
	else {
		createDirectoryRecursively(destFolder.c_str());
		cout << "You've specified directory containing recognition results: " << destFolder << endl << endl;
	}

	//read image list
	char sPath[1024];
	int imgNo = 0;

#ifdef _MY_DEBUG
	float probFaceID,probAgeID,probGender;
	int nFaceSetID, nAgeID, nGender;

	int color_image_show_width;
	int color_image_show_height;
	double dcolor_image_show_scale;
	IplImage *color_image = NULL, *color_image_show = NULL;
#endif
	
	while(fgets(sPath, 1024, fpImgList))
	{
		int scene_label = 0;

		cout << "---------------------------------------------------------" << endl;

		len = strlen(sPath);
		while (len && isspace(sPath[len - 1])) {
			sPath[len - 1] = '\0';
			len = strlen(sPath);
		}

		if (strlen(sPath) == 0)   continue;

		cout << sPath << endl;

		int face_num = PhotoIndexing_ImageEvaluation(sPath, &scene_label);

#ifdef _MY_DEBUG 
		if (face_num >= 0) {
			color_image = cvLoadImage(sPath);

			color_image_show_width = 1024;
			dcolor_image_show_scale = color_image_show_width * 1.0 / color_image->width;
			color_image_show_height = int(color_image->height * dcolor_image_show_scale);
			color_image_show = cvCreateImage(cvSize(color_image_show_width, color_image_show_height), IPL_DEPTH_8U, color_image->nChannels);
			cvResize(color_image, color_image_show);

			cvReleaseImage(&color_image);

			CvFont *pFont = new CvFont;
			cvInitFont(pFont, CV_FONT_HERSHEY_PLAIN, 4, 4, 2, 8);
			CvFont *pFont2 = new CvFont;
			cvInitFont(pFont2, CV_FONT_HERSHEY_PLAIN, 2, 2, 2, 4);
			for (int i = 0; i < face_num; ++i) 
			{
				CvRect rect = FaceRecognitionResult[i].FaceRegion;
				CvRect Rect_show;
				Rect_show.x = rect.x * dcolor_image_show_scale;
				Rect_show.y = rect.y * dcolor_image_show_scale;
				Rect_show.width = rect.width * dcolor_image_show_scale;
				Rect_show.height = rect.height * dcolor_image_show_scale;

				if (Face_Valid_Flag[i] > 0) {
					probFaceID = FaceRecognitionResult[i].Prob_FaceID;
					nFaceSetID = FaceRecognitionResult[i].FaceID;
					nAgeID = FaceRecognitionResult[i].Age;
					probAgeID = FaceRecognitionResult[i].Prob_Age;
					nGender = FaceRecognitionResult[i].Gender;
					probGender = FaceRecognitionResult[i].Prob_Gender;
	
					if (probFaceID<0.10)  nFaceSetID = -1;
					if (probAgeID < 0.10) nAgeID = -1;
					if (probGender < 0.10) nGender = -1;

					const char *name = GetRecognizedFaceName(nFaceSetID);
					char sFaceName[256];
					strcpy(sFaceName, name);
					iFaceDrawFaceRect(color_image_show, Rect_show), CV_RGB(0, 255, 0);

					char sAge[256];
					switch (nAgeID)
					{//nAgeID = 0:"Baby", 1:"Kid", 2:"Adult", 3:"Senior"
						case 0:
							strcpy(sAge, "Baby");
							break;
						case 1:
							strcpy(sAge, "Kid");
							break;
						case 2:
							strcpy(sAge, "Adult");
							break;
						case 3:
							strcpy(sAge, "Senior");
							break;
						default:
							strcpy(sAge, "Unknown");
							break;
					}

					char sGender[256];
					switch (nGender)
					{//	nGender =  0:"male", 1:"female"
						case 0:
							strcpy(sGender, "Male");
							break;
						case 1:
							strcpy(sGender, "Female");
							break;
						default:
							strcpy(sGender, "Unknown");
							break;
					}
					char text[256];
					memset(text, 0, 256);
					sprintf(text, "%s(%s,%s)", sFaceName, sAge,sGender);
					cvPutText(color_image_show, text, cvPoint(Rect_show.x, (Rect_show.y - 10)), pFont, CV_RGB(0, 255, 0));


					char text1[256];
					memset(text1, 0, 256);

					sprintf(text1, "%s(%f)", sFaceName, probFaceID);
					cvPutText(color_image_show, text1, cvPoint(0,30), pFont2, CV_RGB(0, 255, 0));


					char text2[256];
					memset(text2, 0, 256);

					sprintf(text2, "%s(%f)", sAge, probAgeID);
					cvPutText(color_image_show, text2, cvPoint(0,60), pFont2, CV_RGB(0, 255, 0));

					char text3[256];
					memset(text3, 0, 256);

					sprintf(text3, "%s(%f)", sGender, probGender);
					cvPutText(color_image_show, text3, cvPoint(0,90), pFont2, CV_RGB(0, 255, 0));

				}
				else {
					//draw face detection rectangle
					iFaceDrawFaceRect(color_image_show, Rect_show, CV_RGB(0, 0, 255));
				}

			}

			delete pFont;

			//output face detection and recongition image
			char sImgPath1[1024];
			char tmpStr1[1024];
			strcpy(tmpStr1, sPath);

			if (destFolder.empty()) {
				char* firstdot1 = strrchr(tmpStr1, '.');
				*firstdot1 = '\0';
				strcpy(sImgPath1, tmpStr1);
				sprintf(sImgPath1, "%sFDFR10.jpg", tmpStr1);
			}
			else {
				char* lpPureName = strrchr(tmpStr1, '/');
				if (lpPureName == NULL)
					lpPureName = strrchr(tmpStr1, '\\');
				if (lpPureName == NULL)
					lpPureName = tmpStr1;
				else lpPureName++;
				char* firstdot1 = strrchr(lpPureName, '.');
				*firstdot1 = '\0';
				sprintf(sImgPath1, "%s/%sFDFR20_11.jpg", destFolder.c_str(), lpPureName);
			}

			cvSaveImage(sImgPath1, color_image_show);
			cvReleaseImage(&color_image_show);
		}		
#endif 	

		printf(" %d\r", ++imgNo);		
	}

	fclose(fpImgList);
	return 0;
}

int FaceRegistration(string sImgList, string sUserName)
{
	int len = 0;

	FILE *fpImgList = fopen(sImgList.c_str(), "rt");
	if (fpImgList == nullptr) {
		cout << "Usage: " << "IFaceText -c <configure_file> -r <ImgList.txt> <username> " << endl;
		cout << "Please Confirm Your Image List File: " << sImgList << endl << endl;
		return -1;
	}

	if (sUserName.empty()) {
		cout << "Usage: " << "IFaceText -c <configure_file> -r <ImgList.txt> <username> " << endl;
		cout << "NOTE: username is needed when registeration." << endl << endl;
		return -1;
	}

	cout << "Image List File: " << sImgList << endl;
	cout << "Username for Registeration: " << sUserName << endl << endl;

	//read image list
	char sPath[1024];

	IplImage *m_FaceTemplate[FACE_TEMPLATE_MAX_NUM];

	for (int i = 0; i < FACE_TEMPLATE_MAX_NUM; i++) {
		m_FaceTemplate[i] = NULL;
	}		

	int nTotalFaceNum = 0;

	cout << "Begin Face Detection" << endl;

	while(fgets(sPath, 1024, fpImgList)) {
		len = strlen(sPath);
		while (len && isspace(sPath[len - 1])) {
			sPath[len - 1] = '\0';
			len = strlen(sPath);
		}

		if (strlen(sPath) == 0)   continue;
		
		int nFaceNum = FaceRegistration_DetectFace(sPath, (char*)thumbnail.c_str(), m_FaceTemplate, &nTotalFaceNum);

		cout << sPath << "  :  " << nFaceNum << endl;
	}

	cout << "Finish Face Detection" << endl << endl;

	cout << "Begin Face Registeration" << endl;

	FaceRegistration_AddUser(faceModelXML.c_str(), (char*)sUserName.c_str(), m_FaceTemplate, nTotalFaceNum);

	cout << "Finish Face Registeration" << endl;

	for(int i = 0; i < FACE_TEMPLATE_MAX_NUM; ++i)
	{
		if (m_FaceTemplate[i] != NULL)
		{
			cvReleaseImage(&(m_FaceTemplate[i]));
		}
	}

	fclose(fpImgList);

	return 0;
}

int main(int argc, char* argv[])
{
	string imgList;
	string username;
	string output;
	string configure_xml;
	int run_mode;
	
	int index = 1;
	char* optarg = NULL;
	
	while (index < argc) {
		optarg = argv[index];

		if (!strcmp(optarg, "-c")) {
			if (++index == argc) {
				usage();
				return -1;
			}

			configure_xml = argv[index];
			if (configure_xml[0] == '-') {
				usage();
				return -1;
			}
		}
		else if (!strcmp(optarg, "-r")) {
			if (++index == argc) {
				usage();
				return -1;
			}

			imgList = argv[index];
			if (imgList[0] == '-') {
				usage();
				return -1;
			}
			run_mode = 0;
		}
		else if (!strcmp(optarg, "-i")) {
			if (++index == argc) {
				usage();
				return -1;
			}

			imgList = argv[index];
			if (imgList[0] == '-') {
				usage();
				return -1;
			}
			run_mode = 1;
		}
		else if (!strcmp(optarg, "-o") && (++index < argc)) {
			char* temp = argv[index];
			if (temp[0] != '-') {
				output = temp;
			}
			else {
				--index;
			}
		}
		else {
			username = optarg;
		}

		++index;
	}

	if (run_mode == -1 || ((run_mode == 0) && (username.empty()))) {
		usage();
		return -1;
	}
	
	if (!PhotoIndexing_Init(configure_xml))
	{
		usage();
		return -1;
	}

	if (run_mode)
	{
		long long t1 = getTickCounting();
		evaluateImgList(imgList,output);
		long long t2 = getTickCounting();
		double duration = ((double)(t2 - t1)) / (getTickFrequency1() * 1e-3);
		printf("face detection takes %f ms\n", duration);
	}
	else
	{
		FaceRegistration(imgList, username);
	}

	PhotoIndexing_Release();

	return 0;
}