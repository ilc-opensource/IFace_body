
/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * Author: Hua Tang, Fei Duan, Haibing Ren, Ziang Li
 * */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "color_GMM.h"
#include "log.h"

using namespace std;

#define EM_REFINE

namespace robot
{
CColorGMM::CColorGMM(void)
{
    clusterNum = 8;
    lookupTableInterval = 4;
    lookupTalbeChannelBin = 256 / lookupTableInterval;
    lookupTableData = (uchar *)malloc(lookupTalbeChannelBin * lookupTalbeChannelBin * lookupTalbeChannelBin);
}

CColorGMM::~CColorGMM(void)
{
    if (lookupTableData != nullptr) {
        free(lookupTableData);
    }
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of CColorGMM
///  written by Ren Haibing
/// Acknowledge		:
///
/// Function name	: TrainLookupTalbe
/// Description	    : train lookup table
///
/// Argument		:
///
/// Return type		:
///
/// Create Time		: 2015-4-27  14:20
///
///
/// Side Effect		:
///
///////////////////////////////////////////////////////////////////////////////////////////////
int CColorGMM::trainLookupTable(const cv::Mat &lpTrainingImage)
{
    cv::GaussianBlur(lpTrainingImage, lpTrainingImage, cvSize(5, 5), 5);

    cv::Mat lpImageHSV;
    cv::cvtColor(lpTrainingImage, lpImageHSV, CV_BGR2HSV);

    int sampleNum = 0;
    int imageWidth = lpTrainingImage.cols;
    int imageHeight = lpTrainingImage.rows;
    uchar *lpImageBuffer;
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (int i = 0; i < imageHeight; ++i) {
        for (int j = 0; j < imageWidth; ++j) {
            lpImageBuffer = lpImageHSV.ptr<uchar>(i, j);
            if ((lpImageBuffer[0] > 0) || (lpImageBuffer[1] > 0) || (lpImageBuffer[2] > 0)) {
                sampleNum++;
            }
        }
    }

    cv::Mat TrainingSamples(sampleNum, 3, CV_8UC1);
    int nSampleNo = 0;
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (int i = 0; i < imageHeight; ++i) {
        for (int j = 0; j < imageWidth; ++j) {
            lpImageBuffer = lpImageHSV.ptr<uchar>(i, j);
            if ((lpImageBuffer[0] > 0) || (lpImageBuffer[1] > 0) || (lpImageBuffer[2] > 0)) {
                TrainingSamples.ptr<uchar>(nSampleNo, 0)[0] = lpImageBuffer[0];
                TrainingSamples.ptr<uchar>(nSampleNo, 1)[0] = lpImageBuffer[1];
                TrainingSamples.ptr<uchar>(nSampleNo, 2)[0] = lpImageBuffer[2];
                nSampleNo++;
            }
        }
    }
	/*
    emParams.means = NULL;
    emParams.probs = NULL;
    emParams.covs = NULL;
    emParams.weights = NULL;
    emParams.nclusters = clusterNum;
    emParams.start_step = CvEM::START_AUTO_STEP;
    emParams.cov_mat_type = CvEM::COV_MAT_SPHERICAL;
    emParams.term_crit.max_iter = 300;
    emParams.term_crit.epsilon = 0.1;*/

    // cvReshape
    FILE_LOG(logINFO) << "Begin to estimate the GMM parameter";
    cv::Mat labels;

#ifdef EM_REFINE
	/*
    CvEM Em1;
    Em1.train(TrainingSamples, cv::Mat(), emParams, &labels);

    emParams.cov_mat_type = CvEM::COV_MAT_DIAGONAL;
    emParams.start_step = CvEM::START_E_STEP;
    emParams.means = Em1.get_means();
    emParams.covs = Em1.get_covs();
    emParams.weights = Em1.get_weights();*/
#endif

    //em.train(TrainingSamples, cv::Mat(), emParams, &labels);

    FILE_LOG(logINFO) << "Begin to calculate the lookup table";
    int halfInteval = (int)(0.5 * lookupTableInterval);
    cv::Mat lookupTableRGBImage(lookupTalbeChannelBin * lookupTalbeChannelBin,
                                lookupTalbeChannelBin, CV_8UC3);
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (int i = 0; i < lookupTalbeChannelBin; ++i) {
        FILE_LOG(logDEBUG) << "Lookup table, step 1 :" << i << "/" << lookupTalbeChannelBin;
        for (int j = 0; j < lookupTalbeChannelBin; ++j) {
            for (int k = 0; k < lookupTalbeChannelBin; k++) {
                lookupTableRGBImage.ptr<uchar>(i * lookupTalbeChannelBin + j, k)[0] =
                    i * lookupTableInterval + halfInteval;
                lookupTableRGBImage.ptr<uchar>(i * lookupTalbeChannelBin + j, k)[1] =
                    j * lookupTableInterval + halfInteval;
                lookupTableRGBImage.ptr<uchar>(i * lookupTalbeChannelBin + j, k)[2] =
                    k * lookupTableInterval + halfInteval;
            }
        }
    }
    cv::Mat lookupTableHSVImage;
    cvtColor(lookupTableRGBImage, lookupTableHSVImage, CV_BGR2HSV);

    double dResult;
    cv::Mat samples(1, 3, CV_8UC1);

#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (int i = 0; i < lookupTalbeChannelBin; ++i) {
        FILE_LOG(logDEBUG) << "Lookup table, step 2 :" << i << "/" << lookupTalbeChannelBin;
        for (int j = 0; j < lookupTalbeChannelBin; ++j) {
            for (int k = 0; k < lookupTalbeChannelBin; ++k) {
                samples.at<uchar>(0, 0) =
                    lookupTableHSVImage.ptr<uchar>(i * lookupTalbeChannelBin + j, k)[0];
                samples.at<uchar>(0, 1) =
                    lookupTableHSVImage.ptr<uchar>(i * lookupTalbeChannelBin + j, k)[1];
                samples.at<uchar>(0, 2) =
                    lookupTableHSVImage.ptr<uchar>(i * lookupTalbeChannelBin + j, k)[2];
                //dResult = em.calcLikelihood(samples);
				dResult = em->predict2(samples,cv::noArray())[0];

                int offset = 30;
                int temp = int(dResult + offset + 3);
                temp = temp < 0 ? 0 : temp;
                temp = temp > offset ? offset : temp;

                temp = temp * 255 / offset;
                temp = temp > 255 ? 255 : temp;

                lookupTableData[(i * lookupTalbeChannelBin + j) * lookupTalbeChannelBin + k] = (uchar)(temp);
            }
        }
    }

    FILE_LOG(logINFO) << "Begin to test training sample";
    // Mat Prob;
    cv::Mat result(lpTrainingImage.rows, lpTrainingImage.cols, CV_8UC1);
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (int i = 0; i < imageHeight; ++i) {
        for (int j = 0; j < imageWidth; ++j) {
            samples.ptr<uchar>(0, 0)[0] = lpImageHSV.ptr<uchar>(i, j)[0];
            samples.ptr<uchar>(0, 1)[0] = lpImageHSV.ptr<uchar>(i, j)[1];
            samples.ptr<uchar>(0, 2)[0] = lpImageHSV.ptr<uchar>(i, j)[2];

            //double test = em.calcLikelihood(samples);
			double test = em->predict2(samples, cv::noArray())[0];
            int offset = 30;
            int temp = (int)test + offset + 3;
            temp = temp < 0 ? 0 : temp;
            temp = temp > offset ? offset : temp;

            temp = temp * 255 / offset;
            temp = temp > 255 ? 255 : temp;

            result.at<uchar>(i, j) = (uchar)(temp);
        }
    }
    imshow("Training sample result image", result);
    // create a lookup table
    FILE_LOG(logINFO) << "Finish registration";

    return 0;
}

int CColorGMM::trainLUT(const cv::Mat & inSamples)
{/*
    emParams.means = NULL;
    emParams.probs = NULL;
    emParams.covs = NULL;
    emParams.weights = NULL;
    emParams.nclusters = clusterNum;
    emParams.start_step = CvEM::START_AUTO_STEP;
    emParams.cov_mat_type = CvEM::COV_MAT_SPHERICAL;
    emParams.term_crit.max_iter = 300;
    emParams.term_crit.epsilon = 0.1;
	*/
    FILE_LOG(logINFO) << "Begin to estimate the GMM parameter";

    cv::Mat labels;
    //em.train(inSamples, cv::Mat(), emParams, &labels);
	em->trainEM(inSamples, cv::noArray(), labels, cv::noArray());

    FILE_LOG(logINFO) << "Begin to calculate the lookup table";
    int halfInteval = (int)(0.5 * lookupTableInterval);
    cv::Mat lookupTableRGBImage(lookupTalbeChannelBin * lookupTalbeChannelBin,
                                lookupTalbeChannelBin, CV_8UC3);
    for (int i = 0; i < lookupTalbeChannelBin; ++i) {
        FILE_LOG(logDEBUG) << "Lookup table, step 1 :" << i << "/" << lookupTalbeChannelBin;
        for (int j = 0; j < lookupTalbeChannelBin; ++j) {
            for (int k = 0; k < lookupTalbeChannelBin; k++) {
                lookupTableRGBImage.ptr<uchar>(i * lookupTalbeChannelBin + j, k)[0] =
                    i * lookupTableInterval + halfInteval;
                lookupTableRGBImage.ptr<uchar>(i * lookupTalbeChannelBin + j, k)[1] =
                    j * lookupTableInterval + halfInteval;
                lookupTableRGBImage.ptr<uchar>(i * lookupTalbeChannelBin + j, k)[2] =
                    k * lookupTableInterval + halfInteval;
            }
        }
    }
    cv::Mat lookupTableHSVImage;
    cvtColor(lookupTableRGBImage, lookupTableHSVImage, CV_BGR2HSV);

    double dResult;
    cv::Mat samples(1, 3, CV_8UC1);
    for (int i = 0; i < lookupTalbeChannelBin; ++i) {
        FILE_LOG(logDEBUG) << "Lookup table, step 2 :" << i << "/" << lookupTalbeChannelBin;
        for (int j = 0; j < lookupTalbeChannelBin; ++j) {
            for (int k = 0; k < lookupTalbeChannelBin; ++k) {
                samples.at<uchar>(0, 0) =
                    lookupTableHSVImage.ptr<uchar>(i * lookupTalbeChannelBin + j, k)[0];
                samples.at<uchar>(0, 1) =
                    lookupTableHSVImage.ptr<uchar>(i * lookupTalbeChannelBin + j, k)[1];
                samples.at<uchar>(0, 2) =
                    lookupTableHSVImage.ptr<uchar>(i * lookupTalbeChannelBin + j, k)[2];
                //dResult = em.calcLikelihood(samples);
				dResult = em->predict2(samples, cv::noArray())[0];
                int offset = 30;
                int temp = int(dResult + offset + 3);
                temp = temp < 0 ? 0 : temp;
                temp = temp > offset ? offset : temp;

                temp = temp * 255 / offset;
                temp = temp > 255 ? 255 : temp;
                lookupTableData[(i * lookupTalbeChannelBin + j) * lookupTalbeChannelBin + k] = (uchar)(temp);
            }
        }
    }

    // create a lookup table
    FILE_LOG(logINFO) << "Finish registration";

    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of CColorGMM
///  written by Ren Haibing
/// Acknowledge		:
///
/// Function name	: RGB_to_ColorMap
/// Description	    : transfer RGB image to color map
///
/// Argument		:
///
/// Return type		:
///
/// Create Time		: 2015-4-28  10:23
///
///
/// Side Effect		:
///
///////////////////////////////////////////////////////////////////////////////////////////////
void CColorGMM::RGBToColorMap(const cv::Mat &ImageMat, cv::Mat &colorMap)
{
    int imageWidth = ImageMat.cols;
    int imageHeight = ImageMat.rows;
    colorMap = cv::Mat::zeros(imageHeight, imageWidth, CV_8U);
    int B, G, R;

    for (int i = 0; i < imageHeight; ++i) {
        const cv::Vec3b * ptrSrc = ImageMat.ptr<cv::Vec3b>(i);
        uchar * ptr = colorMap.ptr<uchar>(i);
        for (int j = 0; j < imageWidth; ++j) {
            B = ptrSrc[j][0];
            G = ptrSrc[j][1];
            R = ptrSrc[j][2];
            B = int(B * 1.0 / lookupTableInterval);
            G = int(G * 1.0 / lookupTableInterval);
            R = int(R * 1.0 / lookupTableInterval);
            //int idx = ((B * lookupTalbeChannelBin) + G) * lookupTalbeChannelBin + R;
            ptr[j] = lookupTableData[((B * lookupTalbeChannelBin) + G) * lookupTalbeChannelBin + R];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of CColorGMM
///  written by Ren Haibing
/// Acknowledge		:
///
/// Function name	: SaveLookupTable
/// Description	    : save lookup table
///
/// Argument		:
///
/// Return type		:
///
/// Create Time		: 2015-4-28  13:13
///
///
/// Side Effect		:
///
///////////////////////////////////////////////////////////////////////////////////////////////
bool CColorGMM::saveLookupTable(const std::string &sFilename)
{
    FILE *file = fopen(sFilename.c_str(), "wb");
    if (file == nullptr) {
        FILE_LOG(logERROR) << "Can't write file:" << sFilename;
        return false;
    }
    int result = fwrite(lookupTableData, 1, lookupTalbeChannelBin * lookupTalbeChannelBin *
                        lookupTalbeChannelBin, file);
    fclose(file);
    if (result == lookupTalbeChannelBin * lookupTalbeChannelBin * lookupTalbeChannelBin) {
        return true;
    } else {
        return false;
    }
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of CColorGMM
///  written by Ren Haibing
/// Acknowledge		:
///
/// Function name	: LoadLookupTable
/// Description	    : load lookup table
///
/// Argument		:
///
/// Return type		:
///
/// Create Time		: 2015-4-28  13:23
///
///
/// Side Effect		:
///
///////////////////////////////////////////////////////////////////////////////////////////////
bool CColorGMM::loadLookupTable(const std::string & filename)
{
    FILE *file = fopen(filename.c_str(), "rb");
    if (file == nullptr) {
        FILE_LOG(logERROR) << "Can't find file:" << filename;
        return false;
    }
    int result = fread(lookupTableData, 1, lookupTalbeChannelBin * lookupTalbeChannelBin *
                       lookupTalbeChannelBin, file);
    fclose(file);

    if (result == lookupTalbeChannelBin * lookupTalbeChannelBin * lookupTalbeChannelBin) {
        return true;
    } else {
        return false;
    }
}
} // end namespace robot