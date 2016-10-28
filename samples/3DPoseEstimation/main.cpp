/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * */

#include <iostream>
#include <string>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "IHumanDetector_RGBD.h"
#include "IHumanOriEstimation.h"
#include "FaceAttributeRecognition.h"
#include "IFace3DPoseEstimation.h"
#include "RSWrapper.h"

#ifdef WIN32
#include <io.h>
#define do_access(_PATH_, _MODE_) _access(_PATH_, _MODE_)
#else
#include <unistd.h>
#define do_access(_PATH_, _MODE_) access(_PATH_, _MODE_)
#endif

using namespace std;

#define LIVE_VIDEO

bool readImgFile(const char *video_dir, int nFrameNo, cv::Mat &colorImage, cv::Mat &depthImage)
{
    char filename[200];

    memset(filename, 0, 200);
    sprintf(filename, "%s/frame_%05d_RGBImg.jpg", video_dir, nFrameNo);

    colorImage = cv::imread(filename);

    if (!colorImage.data) {
        return false;
    }

    memset(filename, 0, 200);
    sprintf(filename, "%s/frame_%05d_DepthDataConvertedSmooth.dat", video_dir, nFrameNo);

    depthImage = cv::Mat(cv::Size(colorImage.cols, colorImage.rows), CV_16U, 1);

    FILE *fDepthFile = fopen(filename, "rb");

    fread(depthImage.data, sizeof(ushort), depthImage.cols * depthImage.rows, fDepthFile);

    fclose(fDepthFile);

    return true;
}

int main(int argc, char* argv[])
{
    string dbDir;

#ifdef LIVE_VIDEO
    if (argc != 2) {
        fprintf(stderr, "(LIVE_VIDEO MODE) Usage: \n");
        fprintf(stderr, "\t3DPosEstimation.exe <opecv_config_yml>\n");
        return -1;
    }
	cout << "3DPosEstimation in LIVE_VIDEO mode" << endl; 
#else
    if (argc < 5) {
        fprintf(stderr, "(IMGSET MODE) Usage: \n");
        fprintf(stderr, "\t3DPosEstimation.exe <opecv_config_yml> <video_dir> start_index end_index <save_dir>\n");
        return -1;
    }

    char *video_dir = NULL, *save_dir = NULL;
    int len = 0, len2 = 0;
    int start = 0, end = 0;

    video_dir = argv[2];
    start = atoi(argv[3]);
    end = atoi(argv[4]);

    if (argc >= 6) {
        save_dir = argv[5];
        len2 = strlen(save_dir);
    }

    len = strlen(video_dir);

    if (len == 0 || do_access(video_dir, 0) == -1) {
        fprintf(stderr, "Invalid video_dir for 3DPosEstimation: %s\n", video_dir);
        return -1;
    }

    if (len2 == 0 || do_access(save_dir, 0) == -1) {
        fprintf(stderr, "No save result, because save_dir is invalid: %s\n", save_dir);
        len2 = 0;
        save_dir = NULL;
    }

    if (!(0 <= start && start <= end)) {
        fprintf(stderr, "Invalid start_index end_index: %d %d\n", start, end);
        return -1;
    }

    if (video_dir[len - 1] == '/') {
        video_dir[len - 1] = '\0';
    }

    if (len2 && save_dir[len2 - 1] == '/') {
        save_dir[len2 - 1] = '\0';
    }

    cout << "3DPosEstimation in IMAGE_SET mode" << endl;
#endif
	//read configure files
	string configure_file = argv[1];
	if (configure_file.empty())
	{
		fprintf(stderr, "please confirm your configure file\n");
		return false;
	}
	cv::FileStorage fs;
	fs.open(configure_file.c_str(), cv::FileStorage::READ);	
	string opencv_input_newSkinLookupTable;
	string opencv_input_xmlEyeLeftCorner;
	string opencv_input_xmlMthLeftCorner;
	string opencv_input_xmlNose;
	fs["opencv_input_newSkinLookupTable"] >> opencv_input_newSkinLookupTable;
	fs["opencv_input_xmlEyeLeftCorner"] >> opencv_input_xmlEyeLeftCorner;
	fs["opencv_input_xmlMthLeftCorner"] >> opencv_input_xmlMthLeftCorner;
	fs["opencv_input_xmlNose"] >> opencv_input_xmlNose;
	fs.release();
	
	//Init
	IFaceFaceDetector detector;
	EnumViewAngle  viewAngle = (EnumViewAngle)VIEW_ANGLE_FRONTAL;
	detector.init(viewAngle, FEA_HAAR, 2, opencv_input_newSkinLookupTable.c_str());

	IFaceLandmarkDetector* 	plandmarkDetector = new IFaceLandmarkDetector(LDM_6PT, opencv_input_xmlEyeLeftCorner.c_str(), opencv_input_xmlMthLeftCorner.c_str(), opencv_input_xmlNose.c_str());

    cvNamedWindow("3DPoseEstimationResult");
    cv::Mat colorImage, depthImage;

#ifdef LIVE_VIDEO
    int idxImageRes = 1, idxFrameRate = 30;
    RSWrapper depthCam(idxImageRes, idxImageRes, idxFrameRate, idxFrameRate);
    if (!depthCam.init()) {
        std::cerr << "Init. RealSense Failure!" << std::endl;
        return -1;
    }

    uint32_t counts = 0;

    while (true) {
        //Get RGB-D Images
        bool ret = depthCam.capture(0, colorImage, depthImage);
        if (!ret) {
                std::cerr << "Get realsense camera data failure!" << std::endl;
                break;
        }

        std::cout << std::endl << "Get realsense camera data " << ++counts << std::endl;
#else
		

    for (int nFrameNo = start; nFrameNo <= end; ++nFrameNo) {
        bool succ = readImgFile(video_dir, nFrameNo, colorImage, depthImage);

        if (succ == false) continue;
#endif
		IplImage *origin_Image = new IplImage(colorImage);
		CvRectItem rects[1024];
		detector.setFaceDetectionSizeRange(50, 200);
		int nNum = detector.detect(origin_Image, rects, 0);
		detector.clearFaceDetectionRange();
		
		cv::Mat dstImage = colorImage.clone();
        if (nNum) {
            cv::Mat gray_image;
            cvtColor(colorImage, gray_image, CV_BGR2GRAY);

			for (int i = 0; i < nNum; ++i){
				CvRect rect = rects[i].rc;
				bool  bLandmark = false;
				CvPoint2D32f   landmark6[6 + 1]; // consider both 6-pt and 7-pt
				bLandmark = plandmarkDetector->detect(origin_Image, &rect, landmark6, NULL, rects[i].angle); //for imagelist input
				if (bLandmark) {
					double dPan_Angle, dTile_Angle;
					bool bSuccess = iFacePoseEstimation_WithLandmark((unsigned short*)depthImage.data, colorImage.cols, colorImage.rows, landmark6, 6, &dPan_Angle, &dTile_Angle);
					if (bSuccess)
					{
						cout << "Pan degree:" << dPan_Angle << endl;
						cout << "Tilt degree:" << dTile_Angle << endl;
						
						int nLength = rect.width;
						int nLeftLength = abs(dPan_Angle)*1.0 / 45 * nLength;
						if (dPan_Angle > 0)
							cv::rectangle(dstImage,
							cvPoint(rect.x + rect.width*0.5 - nLeftLength, rect.y - 30),
							cvPoint(rect.x + rect.width*0.5, rect.y - 30 + 5),
							cvScalar(0, 0, 255, 0), 3);
						else
							cv::rectangle(dstImage,
							cvPoint(rect.x + rect.width*0.5 - nLeftLength, rect.y - 30),
							cvPoint(rect.x + rect.width*0.5, rect.y - 30 + 5),
							cvScalar(255, 0, 0, 0), 3);

						int nUpLength = abs(dTile_Angle)*1.0 / 45 * nLength;
						
						if (dTile_Angle > 0)
							cv::rectangle(dstImage,
							cvPoint(rect.x + rect.width + 20, rect.y + rect.height*0.5),
							cvPoint(rect.x + rect.width + 20, rect.y + rect.height*0.5 + nUpLength),
							cvScalar(0, 0, 255, 0), 3);
						else
							cv::rectangle(dstImage,
							cvPoint(rect.x + rect.width + 20, rect.y + rect.height*0.5),
							cvPoint(rect.x + rect.width + 20, rect.y + rect.height*0.5 + nUpLength),
							cvScalar(255, 0, 0, 0), 3);

						cv::rectangle(dstImage, rect, CV_RGB(0, 255, 0), 2);
						//cv::putText(dstImage, "Pan", cv::Point(rect.x, rect.y), CV_FONT_NORMAL, 1, cv::Scalar(0, 255, 0), 2);
						//cv::putText(dstImage, "Tilt", cv::Point(rect.x + rect.width, rect.y), CV_FONT_NORMAL, 1, cv::Scalar(0, 255, 0), 2);			
					}
					
				}
				
			}

#ifndef LIVE_VIDEO
            if (save_dir != NULL) {
                char savePath[256];
                memset(savePath, 0, 256);
                sprintf(savePath, "%s/frame_%05d_3DPosEsti.jpg", save_dir, nFrameNo);
                fprintf(stderr, "Saved Image: %s\n", savePath);
                imwrite(savePath, dstImage);
            }
#endif
            imshow("3DPoseEstimationResult", dstImage);

            dstImage.release();
            gray_image.release();
        }
		else
			imshow("3DPoseEstimationResult", dstImage);
		delete origin_Image;
#ifndef LIVE_VIDEO
        colorImage.release();
        depthImage.release();
#endif

        if (cvWaitKey(1) == 27)
            break;
    }

    return 0;
}
