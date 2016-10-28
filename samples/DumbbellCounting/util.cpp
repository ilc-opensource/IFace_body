/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * Author: Hua Tang, Fei Duan, Haibing Ren, Ziang Li
 * */

#include "util.h"
#include "log.h"

using namespace std;
using namespace robot;

//Calculate distance of face
ushort calcDepthValue(const ushort * ptrDep, const cv::Size & imgSize, int x, int y)
{
	double z{0};
	vector<ushort> deps;
	for (int r = y - 5; r < y + 5; ++r)
	{
		for (int c = x - 5; c < x + 5; ++c)
		{
			z = ptrDep[r * imgSize.width + c];
			if (z >= 500 && z <= 2500)
			{
				deps.push_back((ushort)z);
			}
		}
	}
	//cout << __FILE__ << ":" << __LINE__ << endl;
	std::nth_element(deps.begin(), deps.begin() + deps.size() / 2, deps.end());
	//cout << __FILE__ << ":" << __LINE__ << endl;
	if (deps.empty())
		FILE_LOG(logERROR) << "Depth Image is empty.";
	return deps.empty() ? 0 : deps[deps.size() / 2];
}

//Detect dumbbell
void locateDumbbell(robot::CColorGMM & gmm, cv::Mat & frame, cv::Rect & bbox, int minSize /*= 20*/)
{
	assert(!frame.empty());
	const static int strelSize = 5;
	cv::Mat deciMap, mask;
	gmm.RGBToColorMap(frame, deciMap);

	int threshVal = 90; //60;//130; // 90;//60;// 130; // 30: orange ball,
	cv::threshold(deciMap, mask, threshVal, 255, CV_THRESH_BINARY);
	cv::Mat ker = cv::getStructuringElement(cv::MORPH_ELLIPSE,
	                                        cv::Size(strelSize, strelSize));
	cv::erode(mask, mask, ker);
	cv::dilate(mask, mask, ker);

	robot::conn_comp_t ccMax;
	robot::findMaxCc(mask, ccMax);
	bbox = ccMax.bbox;
	return;
}
