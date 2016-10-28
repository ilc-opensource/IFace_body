/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * */

#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "RSWrapper.h"
#include "IHumanDetector_RGBD.h"
#include "IHumanOriEstimation.h"

using namespace std;

bool InitModel(string configure_file)
{
	if (configure_file.empty())
	{
		fprintf(stderr, "please confirm your configure file\n");
		return false;
	}

	// Initialize Human Detector
	initRGBDHumanDetector(1.3);

	//initialize Human Ori Estimation
	//read configure_file
	cv::FileStorage fs;
	string headPosConfigFile;
	string hboData;
	string modelDir;
	string leafNodeFile;

	fs.open(configure_file, cv::FileStorage::READ);
	if (!fs.isOpened()) {
		fprintf(stderr, "invalid config file\n");
		fs.release();
		return false;
	}

	fs["headPosConfigFile"] >> headPosConfigFile;
	fs["hboData"] >> hboData;
	fs["modelDir"] >> modelDir;
	fs["leafNodeFile"] >> leafNodeFile;

	fs.release();

	humanOriEstimationInit(headPosConfigFile.c_str(), modelDir.c_str(), hboData.c_str(), leafNodeFile.c_str());

	return true;
}
void ReleaseModel()
{
	// Cleanup
	humanDetectionRelease();
	humanOriEstimationRelease();

}

void HuamnOriEstimate(cv::Mat color, cv::Mat depth, cv::Rect rect, string &szPose)
{
	HumanOriEstimationValue hpe_ret;
	estimateHumanOri(color, depth, rect, hpe_ret);
	switch (hpe_ret.humanOriValue) {
	case FRONTAL_VIEW:
		szPose = "Frontal"; break;
	case LEFT_VIEW:
		szPose = "Left"; break;
	case BACK_VIEW:
		szPose = "Back"; break;
	case RIGHT_VIEW:
		szPose = "Right"; break;
	default:
		szPose = "Unknow"; break;
	}
	return;
}
int main(int argc, char *argv[])
{
	if (argc != 2) {
		fprintf(stderr, "Usage: HumanDetection.exe <config_file>\n");
		return -1;
	}

	//Initialize model
	InitModel(argv[1]);
	
	// Initialize camera
	int idxImageRes = 1, idxFrameRate = 60;
	RSWrapper depthCam(idxImageRes, idxImageRes, idxFrameRate, idxFrameRate);
	if (depthCam.init() < 1) {
		cerr << "Init. RealSense Failure!" << endl;
		return -1;
	}

	char szText[100];
	while (true) {
		//Get RGB-D Images
		cv::Mat color, depth;

		bool ret = depthCam.capture(0, color, depth);
		if (!ret) {
			std::cerr << "Get realsense camera data failure!" << std::endl;
			break;
		}

		cv::namedWindow("HD", cv::WINDOW_AUTOSIZE);
		cv::Mat colorImg_HD = color.clone();
	
		// Head detection
		HdRect *heads;
		double dHDTime = double(cv::getTickCount());
		int nNum = detectHumanInRGBDImage(color, depth, heads);
		dHDTime = double((cv::getTickCount() - dHDTime)*1000.0 / cv::getTickFrequency());
		
		vector<HdRect> vecHeads(nNum);
		copy(heads, heads + nNum, vecHeads.begin());

		for (auto it = vecHeads.begin(); it != vecHeads.end(); ++it)
			rectangle(colorImg_HD, cv::Rect(it->x, it->y, it->width, it->height), cv::Scalar(0, 0, 255), 2);
		sprintf(szText, "HD: %.2fms", dHDTime);
		putText(colorImg_HD, szText, cv::Point(0, 40), 0, 1.0, cv::Scalar(0, 255, 0),2);
		
		// Human orientation estimation
		string szPose;
		for (auto it = vecHeads.begin(); it != vecHeads.end(); ++it) {
			double dHPETime = double(cv::getTickCount());
			HuamnOriEstimate(color, depth, cv::Rect(it->x, it->y, it->width, it->height), szPose);
			dHPETime = double((cv::getTickCount() - dHPETime)*1000.0 / cv::getTickFrequency());
	
			sprintf(szText, "%s %.2fms", szPose.c_str(), dHPETime);
			putText(colorImg_HD, szText, cv::Point(it->x, it->y), 0, 1.0, cv::Scalar(0, 255, 0),3);
		}
		
		//Show images
		imshow("HD", colorImg_HD);

		if (cv::waitKey(1) == 27)
			break;
	}

	depthCam.release();
	ReleaseModel();
	cv::destroyAllWindows();
	return 0;
}
