/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * Author: Hua Tang, Fei Duan, Haibing Ren, Ziang Li
 * */

#ifndef COLOR_GMM_H
#define COLOR_GMM_H

#include <opencv2/core/core.hpp>
//#include <opencv2/legacy/legacy.hpp>
#include <opencv2/ml.hpp>
#include <string>

#include "mode.h"

namespace robot
{
class CColorGMM
{
public:
	int clusterNum;
	int lookupTableInterval;
	int lookupTalbeChannelBin;
	//CvEM em;
	cv::Ptr<cv::ml::EM> em = cv::ml::EM::create();
	//CvEMParams emParams;
	uchar * lookupTableData;

public:
	int trainLookupTable(const cv::Mat & img_tr);
	int trainLUT(const cv::Mat & samples);
	bool saveLookupTable(const std::string & lut_file);
	bool loadLookupTable(const std::string & lut_file);
	void RGBToColorMap(const cv::Mat & img, cv::Mat & colorMap);

	CColorGMM(void);
	~CColorGMM(void);
};

} // namespace robot

#endif