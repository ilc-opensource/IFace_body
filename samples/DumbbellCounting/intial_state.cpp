/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * Author: Hua Tang, Fei Duan, Haibing Ren, Ziang Li
 * */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "initial_state.h"
#include "log.h"

using namespace std;

void readImgFile(const char *sBaseName, int nFrameNo, cv::Mat &colorImage, cv::Mat &depthImage)
{
    char filename[200];

    memset(filename, 0, 200);
    sprintf(filename, "%s%05d_RGBImg.jpg", sBaseName, nFrameNo);

	colorImage = cv::imread(filename);

    memset(filename, 0, 200);
    sprintf(filename, "%s%05d_DepthDataConvertedSmooth.dat", sBaseName, nFrameNo);

	depthImage = cv::Mat(cv::Size(colorImage.cols, colorImage.rows), CV_16U, 1);

    FILE *fDepthFile = fopen(filename, "rb");

    int ret = fread(depthImage.data, sizeof(ushort), depthImage.cols * depthImage.rows, fDepthFile);
	if (ret <= 0) {
		fprintf(stderr, "fread error\n");
	}

    fclose(fDepthFile);
}

bool getInitialState(
    robot::CColorGMM & gmm,
    cv::Rect & faceBbox,
    int & dumbbellLowerbound,
    cv::Rect & initBbox,
#ifdef LIVE_VIDEO
    RSWrapper * d4p,
#else
    int idxFrameBeg,
    int idxFrameEnd,
    const std::string & dbDir,
#endif
    double & distCurr)
{
    cv::Rect bboxLeft, bboxRight;

    // Initialize for human detection
    int faceNum;
    initRGBDHumanDetector();

    // Initialize
    bool hasFace, hasLeftDum, hasRightDum;

#ifdef LIVE_VIDEO
    while (true) {
        cv::Mat frame, depth;

        bool ret = d4p->capture(0, frame, depth);
        if (!ret) {
            std::cerr << "Get realsense camera data failure!" << std::endl;
            break;
        }

        ushort * depthData = (ushort*)depth.data;
#else
    for (int i = idxFrameBeg; i < idxFrameEnd; i++) {
        cv::Mat frame, depth;
        readImgFile(dbDir.c_str(), i, frame, depth);

        ushort *depthData = (ushort*)depth.data;
#endif

        // Initialize
        hasFace = false;
        hasLeftDum = false;
        hasRightDum = false;
        int center = 0, upperBound = 0, leftBound = 0, rightBound = 640, lowerBound = 480;
        HdRect *faces = NULL;
        // Detect face for initial state
        if (!hasFace) {
            faceNum = detectHumanInRGBDImage(frame, depth, faces);
            if (faceNum > 0) {
                int idxMin = -1;
                int minBias = 1000;
                for (int i = 0; i < faceNum; ++i) {
                    int bias = faces[i].x + faces[i].width / 2 - frame.cols / 2;
                    bias = bias > 0 ? bias : abs(bias);
                    if (minBias > bias) {
                        minBias = bias;
                        idxMin = i;
                    }
                }

                HdRect &curFace = faces[idxMin];
                faceBbox = cv::Rect(curFace.x, curFace.y, curFace.width, curFace.height);
                initBbox = faceBbox;

                cv::rectangle(frame, initBbox, cv::Scalar(0, 255, 0), 3);
                if (initBbox.area() < MIN_FACE || initBbox.area() > MAX_FACE) {
                    cv::imshow("DUMBBELL", frame);
					cv::waitKey(1);
                    continue;
                }

                // to get position of center point of faces
                center = initBbox.x + initBbox.width / 2;
                upperBound = initBbox.y;
                leftBound = center - initBbox.width / 2 < 0 ? 0 : center - initBbox.width / 2;
                rightBound = center + initBbox.width / 2 > 640 ? 640 : center + initBbox.width / 2;

                cv::rectangle(frame, faceBbox, cv::Scalar(0, 255, 0), 3);

                int cx = static_cast<int>(faceBbox.x + faceBbox.width / 2);
                int cy = static_cast<int>(faceBbox.y + faceBbox.height / 2);
                distCurr = calcDepthValue(depthData, frame.size(), cx, cy);
                FILE_LOG(logDEBUG) << "Face distance is " << distCurr << " mm";
                hasFace = true;
            }
        }

        //Detect dumbbell for left img and right img
		cv::Mat leftImg = frame(cv::Range(upperBound, lowerBound), cv::Range(leftBound, center));
        if (leftImg.empty()) {
            FILE_LOG(logDEBUG) << "LEFT EMPTY";
            cv::imshow("DUMBBELL", frame);
			cv::waitKey(1);
            continue;
        }
        locateDumbbell(gmm, leftImg, bboxLeft, 3);

		cv::Mat rightImg = frame(cv::Range(upperBound, lowerBound), cv::Range(center, rightBound));
        if (rightImg.empty()) {
            FILE_LOG(logDEBUG) << "RIGHT EMPTY";
            cv::imshow("DUMBBELL", frame);
			cv::waitKey(1);
            continue;
        }
        locateDumbbell(gmm, rightImg, bboxRight, 3);

        if ((std::max<int>(bboxLeft.width, bboxLeft.height) > MIN_DUMBBELL) && (std::max<int>(bboxLeft.width, bboxLeft.height) < MAX_DUMBBELL)) {
            cv::rectangle(frame, cv::Rect(bboxLeft.x + leftBound, bboxLeft.y + upperBound, bboxLeft.width, bboxLeft.height), cv::Scalar(0, 0, 255), 3);
            hasLeftDum = true;
        }

        if ((std::max<int>(bboxRight.width, bboxRight.height) > MIN_DUMBBELL) && (std::max<int>(bboxRight.width, bboxRight.height) < MAX_DUMBBELL)) {
            cv::rectangle(frame, cv::Rect(bboxRight.x + center, bboxRight.y + upperBound, bboxRight.width, bboxRight.height), cv::Scalar(0, 0, 255), 3);
            hasRightDum = true;
        }

        cv::imshow("DUMBBELL", frame);
		cv::waitKey(1);

        dumbbellLowerbound = min(bboxLeft.y + bboxLeft.height / 2, bboxRight.y + bboxRight.height / 2) + upperBound;
        faceBbox.br();
        int dumbbellUpperbound = faceBbox.y + faceBbox.height;

        //Get key to proceed next step
        int ch = cvWaitKey(50);
        if ((ch == 't' || ch == 'T') && hasFace && hasLeftDum && hasRightDum && (dumbbellUpperbound < 240) && ((dumbbellLowerbound - dumbbellUpperbound) > faceBbox.height)) {
            FILE_LOG(logINFO) << "stopping capture...";
            return true;
        }
        if (ch == 'q' || ch == 'Q')
            return false;
    }


    return true;
}