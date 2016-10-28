/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * Author: Hua Tang, Fei Duan, Haibing Ren, Ziang Li
 * */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video.hpp>

#include "tracking.h"
#include "initial_state.h"
#include "util.h"
#include "log.h"

#ifdef USE_OMP
#include <omp.h>
#endif

using namespace std;

namespace robot
{

bool withinFace(int winX, int winY, cv::Rect face)
{
    if (winX > face.x && winX < (face.x + face.width) && winY > face.y && winY < (face.y + face.height)) {
        return true;
    } else {
        return false;
    }
}

bool validRect(int winX, int winY, int width, int height)
{
    if ((width > 4.0) && (height > 4.0) && (winX > 0) && ((winX + width) < 640) && (winY > 0) && ((winY + height) < 480)) {
        return true;
    } else {
        return false;
    }
}

int fastTracking(
    CColorGMM &gmm,
    cv::Rect &faceBbox,
    int dumbbellLowerbound,
    cv::Rect &initBbox,
#ifdef LIVE_VIDEO
    RSWrapper *d4p,
#else
    int idxFrameBeg,
    int idxFrameEnd,
    const std::string &dbDir,
#endif
    double distCurr,
    int period)
{
    int bellLeft = 0, bellRight = 0;
    bool canIncrLeft = true, canIncrRight = true;
    cv::Mat color, depth;
    cv::TermCriteria termCrit(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1);

    ushort *ptrDep = nullptr;
    cv::Mat frame, frameOri;
    cv::Mat deciMap, mask;
    cv::Rect winTrackLeft = initBbox, winTrackRight = initBbox;
    cv::RotatedRect boxTrackLeft, boxTrackRight;
    cv::Rect bboxLeft, bboxRight;

    // boundary settings
    int center = initBbox.x + initBbox.width / 2;
    int upperBound = initBbox.y;
    int leftBound = center - initBbox.width * 2 < 0 ? 0 : center - initBbox.width * 2;      // check whehter within img
    int rightBound = center + initBbox.width * 2 > 640 ? 640 : center + initBbox.width * 2; // check whehter within img
    int lowerBound = 480;                                                                   // check whether within img

    FILE_LOG(logINFO) << "Boundary---------------------------------";
    FILE_LOG(logINFO) << "upper: " << upperBound << "      =========          lower: " << dumbbellLowerbound;
    FILE_LOG(logINFO) << "left:  " << leftBound << "      =========          right: " << rightBound;
	double t0 = (double)cv::getTickCount();

#ifdef LIVE_VIDEO
    while (true) {
        cout << "start tracking..." << endl;
        bool ret = d4p->capture(0, color, depth);
        if (!ret) {
            std::cerr << "Get realsense camera data failure!" << std::endl;
            break;
        }
        cout << "hello" << endl;
#else
    for (int i = idxFrameBeg; i < idxFrameEnd * 50; ++i) {
        readImgFile(dbDir.c_str(), i, color, depth);

        i = i % 700;
        if (i == 0)
        {
            i++;
        }
#endif
        // Set ending timing
		double t = (cv::getTickCount() - t0) / cv::getTickFrequency();
        if (t > period) {
            cout << "ZXTEST end" << endl;
			cv::Mat result(300, 1000, CV_8UC3, cv::Scalar(255, 255, 255));
            char num[1024];
            sprintf(num, "Total number: %d", bellLeft + bellRight);
            cv::putText(result, num, cv::Point(100, 180),
                        CV_FONT_VECTOR0 /*CV_FONT_HERSHEY_PLAIN*/, 3, cv::Scalar(0, 0, 0), 2);
            cv::imshow("result", result);
			cv::waitKey(0); //gai 0719
            break;
        }
        FILE_LOG(logDEBUG) << "Total Time: " << t;

        ptrDep = (ushort*)depth.data;
        if (!ptrDep)
            continue;
        frame = color;
        frameOri = frame.clone();

#ifdef USE_OMP
#pragma omp parallel for
#endif
        // Denoising by depth image
        for (int i = 0; i < frame.rows; i++) {
            uchar * ptr = frame.ptr<uchar>(i);
            for (int j = 0; j < frame.cols; j++) {
                if (abs(ptrDep[i * frame.cols+ j] - distCurr) > 800) {
                    ptr[3 * j] = ptr[3 * j + 1] = ptr[3 * j + 2] = 255;
                }
            }
        }
        cv::imshow("depth", frame);
        cv::line(frameOri, cv::Point(center, 0), cv::Point(center, 479), cv::Scalar(255, 255, 0), 1);
        cv::line(frameOri, cv::Point(0, dumbbellLowerbound), cv::Point(639, dumbbellLowerbound), cv::Scalar(0, 255, 255), 1);
        cv::line(frameOri, cv::Point(0, faceBbox.y + faceBbox.height), cv::Point(639, faceBbox.y + faceBbox.height), cv::Scalar(255, 0, 255), 1);

        char time[10];
        std::sprintf(time, "%0.0f", t);
        cv::putText(frameOri, time, cv::Point(550, 460), CV_FONT_NORMAL /*CV_FONT_HERSHEY_PLAIN*/, 2, cv::Scalar(0, 0, 0));

        // Preprocessing
		cv::GaussianBlur(frame, frame, cv::Size(5, 5), 2, 2);
        gmm.RGBToColorMap(frame, deciMap);
		GaussianBlur(deciMap, deciMap, cv::Size(3, 3), 0, 0);
        cv::threshold(deciMap, deciMap, 90, 255, CV_THRESH_BINARY);
		cv::Mat ker = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::erode(deciMap, deciMap, ker);
        cv::dilate(deciMap, deciMap, ker);

        // Left processing and Right processing
		cv::Mat leftImg = frame(cv::Range(upperBound, lowerBound), cv::Range(leftBound, center));
		cv::Mat deciMapLeft = deciMap(cv::Range(upperBound, lowerBound), cv::Range(leftBound, center));

		cv::Mat rightImg = frame(cv::Range(upperBound, lowerBound), cv::Range(center, rightBound));
		cv::Mat deciMapRight = deciMap(cv::Range(upperBound, lowerBound), cv::Range(center, rightBound));

        // For left dumbbell
        if (winTrackLeft.area() > 3 && winTrackLeft.area() < 10000 && !withinFace(winTrackLeft.x + leftBound, winTrackLeft.y + upperBound, faceBbox)) {
            boxTrackLeft = cv::CamShift(deciMapLeft, winTrackLeft, termCrit);
        } else {
            locateDumbbell(gmm, leftImg, bboxLeft, 3);
            if (std::min<int>(bboxLeft.width, bboxLeft.height) > MIN_DUMBBELL && (std::max<int>(bboxLeft.width, bboxLeft.height) < MAX_DUMBBELL)
                && validRect(bboxLeft.x, bboxLeft.y, bboxLeft.width, bboxLeft.height)) {
                boxTrackLeft = cv::RotatedRect(cv::Point(bboxLeft.x + bboxLeft.width / 2, bboxLeft.y + bboxLeft.height / 2), bboxLeft.size(), 0);
                winTrackLeft = cv::Rect(bboxLeft.x, bboxLeft.y, bboxLeft.width, bboxLeft.height);
                FILE_LOG(logDEBUG) << winTrackLeft.area();
            } else {
                cv::putText(frameOri, "Object Lost! Busy Searching...", cv::Point(10, 30), CV_FONT_VECTOR0 /*CV_FONT_HERSHEY_PLAIN*/, 1, cv::Scalar(0, 0, 255));
                FILE_LOG(logDEBUG) << "Show Image.";
                cv::imshow("DUMBBELL", frameOri);
                cvWaitKey(1);
                continue;
            }
        }

        // For right dumbbell
        if (winTrackRight.area() > 3 && winTrackRight.area() < 10000 && !withinFace(winTrackRight.x + center, winTrackRight.y + upperBound, faceBbox)) {
            boxTrackRight = cv::CamShift(deciMapRight, winTrackRight, termCrit);
        } else {
            locateDumbbell(gmm, rightImg, bboxRight, 3);
            if (std::min<int>(bboxRight.width, bboxRight.height) > MIN_DUMBBELL && (std::max<int>(bboxRight.width, bboxRight.height) < MAX_DUMBBELL)
                && validRect(bboxRight.x, bboxRight.y, bboxRight.width, bboxRight.height)) {
                boxTrackRight = cv::RotatedRect(cv::Point(bboxRight.x + bboxRight.width / 2, bboxRight.y + bboxRight.height / 2), bboxRight.size(), 0);
                winTrackRight = cv::Rect(bboxRight.x, bboxRight.y, bboxRight.width, bboxRight.height);
                FILE_LOG(logDEBUG) << winTrackLeft.area();
            } else {
                cv::putText(frameOri, "Object Lost! Busy Searching...", cv::Point(10, 30), CV_FONT_VECTOR0 /*CV_FONT_HERSHEY_PLAIN*/, 1, cv::Scalar(0, 0, 255));
                FILE_LOG(logDEBUG) << "Show Image.";
                cv::imshow("DUMBBELL", frameOri);
                cvWaitKey(1);
                continue;
            }
        }

        boxTrackLeft = cv::RotatedRect(cv::Point((int)boxTrackLeft.center.x + leftBound, (int)boxTrackLeft.center.y + upperBound), cv::Size2f(boxTrackLeft.size.height, boxTrackLeft.size.width), 0);
        boxTrackRight = cv::RotatedRect(cv::Point((int)boxTrackRight.center.x + center, (int)boxTrackRight.center.y + upperBound), cv::Size2f(boxTrackRight.size.height, boxTrackRight.size.width), 0);
        // left
        if (boxTrackLeft.boundingRect().area() > 0 && validRect((int)boxTrackLeft.center.x, (int)boxTrackLeft.center.y, (int)boxTrackLeft.size.width, (int)boxTrackLeft.size.height)) {
            cv::ellipse(frameOri, boxTrackLeft, cv::Scalar(0, 0, 255), 3, CV_AA);
            cv::line(frameOri, cv::Point(std::max<int>(0, (int)boxTrackLeft.center.x - 4), (int)boxTrackLeft.center.y),
                cv::Point(std::min<int>(frame.cols - 1, (int)boxTrackLeft.center.x + 4), (int)boxTrackLeft.center.y),
                     cv::Scalar(0, 255, 0), 2);
            cv::line(frameOri, cv::Point((int)boxTrackLeft.center.x, std::max<int>((int)boxTrackLeft.center.y - 4, 0)),
                cv::Point((int)boxTrackLeft.center.x, std::min<int>((int)boxTrackLeft.center.y + 4, frame.rows - 1)),
                     cv::Scalar(0, 255, 0), 2);
            if (abs(boxTrackLeft.center.y - boxTrackLeft.size.height / 2 - faceBbox.y) < faceBbox.height && canIncrLeft) {
                ++bellLeft;
                canIncrLeft = false;
                cv::putText(frameOri, "VERY GOOD!", cv::Point(200, 80),
                            CV_FONT_NORMAL /*CV_FONT_HERSHEY_PLAIN*/, 2, cv::Scalar(255, 0, 255));
            } else if (abs(boxTrackLeft.center.y + boxTrackLeft.size.height / 2) > dumbbellLowerbound) {
                canIncrLeft = true;
            } else {
                cv::putText(frameOri, "FIGHTING", cv::Point(100, 80),
                            CV_FONT_NORMAL /*CV_FONT_HERSHEY_PLAIN*/, 2, cv::Scalar(255, 0, 255));
            }
        }
        // right
        if (boxTrackRight.boundingRect().area() > 0 && validRect((int)boxTrackRight.center.x, (int)boxTrackRight.center.y, (int)boxTrackRight.size.width, (int)boxTrackRight.size.height)) {
            cv::ellipse(frameOri, boxTrackRight, cv::Scalar(0, 0, 255), 3, CV_AA);
            cv::line(frameOri, cv::Point(std::max<int>(0, (int)boxTrackRight.center.x - 4), (int)boxTrackRight.center.y),
                     cv::Point(std::min<int>(frame.cols - 1, (int)boxTrackRight.center.x + 4), (int)boxTrackRight.center.y),
                     cv::Scalar(0, 255, 0), 2);
            cv::line(frameOri, cv::Point((int)boxTrackRight.center.x, std::max<int>((int)boxTrackRight.center.y - 4, 0)),
                cv::Point((int)boxTrackRight.center.x, std::min<int>((int)boxTrackRight.center.y + 4, frame.rows - 1)),
                     cv::Scalar(0, 255, 0), 2);
            if (abs(boxTrackRight.center.y - boxTrackRight.size.height / 2 - faceBbox.y) < faceBbox.height && canIncrRight) {
                ++bellRight;
                canIncrRight = false;
                cv::putText(frameOri, "VERY GOOD!", cv::Point(200, 80),
                            CV_FONT_NORMAL /*CV_FONT_HERSHEY_PLAIN*/, 2, cv::Scalar(255, 0, 255));
            } else if (abs(boxTrackRight.center.y + boxTrackRight.size.height / 2) > dumbbellLowerbound) {
                canIncrRight = true;
            } else {
                cv::putText(frameOri, "FIGHTING", cv::Point(100, 80),
                            CV_FONT_NORMAL /*CV_FONT_HERSHEY_PLAIN*/, 2, cv::Scalar(255, 0, 255));
            }
        }

        cv::putText(frameOri, std::to_string(bellLeft), cv::Point(10, 80),
                    CV_FONT_NORMAL /*CV_FONT_HERSHEY_PLAIN*/, 2, cv::Scalar(255, 255, 0));
        cv::putText(frameOri, std::to_string(bellRight), cv::Point(450, 80),
                    CV_FONT_NORMAL /*CV_FONT_HERSHEY_PLAIN*/, 2, cv::Scalar(255, 255, 0));

        FILE_LOG(logDEBUG) << "Show Image.";
        cv::imshow("DUMBBELL", frameOri);

        int ch = cv::waitKey(1);
        if (ch == 27) {
            break;
        }
    }

    return bellLeft + bellRight;
}
}
