/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * Author: Hua Tang, Fei Duan, Haibing Ren, Ziang Li
 * */

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "tracking.h"
#include "initial_state.h"
#include "log.h"

using namespace std;


int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: DumbbellCounting.exe <config_file> \n");
        return -1;
    }

	cv::FileStorage fs;
    string modelFile;
    int period = 0;

	fs.open(argv[1], cv::FileStorage::READ);

    if (!fs.isOpened()) {
        fprintf(stderr, "%s,%d: invalid configure file: %s\n", __FILE__, __LINE__, argv[1]);
        return false;
    }

    fs["modelFile"] >> modelFile;
    fs["period"] >> period;

    cout << "You load color model: " << modelFile << endl;
    cout << "You set up time for dumbbell counting is " << period << endl;

#ifndef LIVE_VIDEO
    string dbDir;
    fs["dbDir"] >> dbDir;
    dbDir = dbDir + "/frame_";
#endif

    fs.release();

    robot::CColorGMM gmm;
    cv::Rect faceBbox, initBbox;
    double distCurr;
    int dumbbellLowerbound = 480;

    // Load dumbbell
#ifdef TEST_COLOR_MODEL
    FILE_LOG(logDEBUG) << "Load the trained color model.";
    gmm.loadLookupTable(modelFile);
    cout << "Dmbbell Color Model Loaded" << endl;
#else
    FILE_LOG(logDEBUG) << "Train a color model by using exemplar image.";

    string trainFile{ "koutu.png" };
    cv::Mat imgTr = cv::imread(trainFile);
    CV_Assert(!imgTr.empty());

    gmm.trainLookupTable(imgTr);
    gmm.saveLookupTable(modelFile);

    cv::Mat decimap;
    gmm.RGBToColorMap(imgTr, decimap);

    cv::imshow("DecisionMap", decimap);
    cv::waitKey(0);

    return 0;
#endif

#ifdef LIVE_VIDEO
    RSWrapper d4p(1, 1, 60, 60);
    if (d4p.init() < 1) {
        cerr << "Init RealSense failed!" << endl;
        return EXIT_FAILURE;
    }
    cout << "RealSense Camera initialized" << endl;
#else
    cout << "This is offline version" << endl;
    int idxFrameBeg = 120, idxFrameEnd = 180;
#endif

    // 'c' to continue, 'q' to exit
    int ch;
    while (1) {
        // Get initial state for tracking
#ifdef LIVE_VIDEO
        if (getInitialState(gmm, faceBbox, dumbbellLowerbound, initBbox, &d4p, distCurr))
#else
        if (getInitialState(gmm, faceBbox, dumbbellLowerbound, initBbox, idxFrameBeg, idxFrameEnd, dbDir, distCurr))
#endif
        {
            int sum_dumbbell = 0;
            // Dumbbell tracking and counting
            cout << "Start tracking....." << endl;
#ifdef LIVE_VIDEO
            sum_dumbbell = robot::fastTracking(gmm, faceBbox, dumbbellLowerbound, initBbox, &d4p, distCurr, period);
#else
            sum_dumbbell = robot::fastTracking(gmm, faceBbox, dumbbellLowerbound, initBbox, idxFrameBeg, idxFrameEnd, dbDir, distCurr, period);
#endif

            // 'c' to continue, 'q' to leave
            while (ch = cvWaitKey(50000)) {
                if (ch == 'c' || ch == 'C') {
                    cvDestroyAllWindows();
                    break;
                }
                if (ch == 'q' || ch == 'Q') break;
            }
            if (ch == 'c' || ch == 'C') continue;
            if (ch == 'q' || ch == 'Q') break;
        } else {
            break;
        }
    }

    cout << "Exit" << endl;

    return 0;
}