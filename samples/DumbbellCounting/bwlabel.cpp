/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * Author: Hua Tang, Fei Duan, Haibing Ren, Ziang Li
 * */

#include "opencv/cv.h"
#include "opencv/highgui.h"

#include "bwlabel.h"
#include <map>
#include <iterator>

using namespace std;

namespace robot
{

#define NO_OBJECT 0
#define ELEM(img, h, w) (CV_IMAGE_ELEM(img, unsigned char, h, w))
#define ONETWO(L, h, w, width) (L[(h) * (width) + w])

int find(int set[], int x)
{
	int h = x;
	while (set[h] != h)
	{
		h = set[h];
	}
	return h;
}

int bwlabel(IplImage * img, int * labels, int n /* = 8 */)
{
	if (n != 4 && n != 8)
	{
		n = 4;
	}
	int height = img->height;
	int width = img->width;
	int total = height * width;

	// results
	memset(labels, 0, total * sizeof(int));
	int nobj = 0; // number of objects found in image
	
	// label table
	int * ltable = new int[total]; 
	memset(ltable, 0, total * sizeof(int));
	int ntable = 0;
	for (int h = 0; h < height; ++h)
	{
		for (int w = 0; w < width; ++w)
		{
			if (!ELEM(img, h, w)) // A is not an object so leave it
			{
				ONETWO(labels, h, w, width) = NO_OBJECT; 
				continue;
			}

			// get the neighboring pixels B, C, D, and E
			int B = (w == 0) ? 0 : find(ltable, ONETWO(labels, h, w - 1, width));
			int C = (h == 0) ? 0 : find(ltable, ONETWO(labels, h - 1, w, width));
			int D = (h == 0 || w == 0) ? 0 : find(ltable, ONETWO(labels, h - 1, w - 1, width));
			int E = (h == 0 || w == width - 1) ? 0 : find(ltable, ONETWO(labels, h - 1, w + 1, width));
			
			switch (n)
			{
				case 4:
					// apply 4 connectedness
					if (B && C)
					{ // B and C are labeled
						if (B == C)
							ltable[C] = B;
						ONETWO(labels, h, w, width) = B;
					}
					else if (B) // B is object but C is not
						ONETWO(labels, h, w, width) = B;
					else if (C) // C is object but B is not
						ONETWO(labels, h, w, width) = C;
					else
					{ // B, C, D not object - new object
						//   label and put into table
						ntable++;
						ONETWO(labels, h, w, width) = ltable[ntable] = ntable;
					}
					break;
				case 6:
					// apply 6 connected ness
					if (D) // D object, copy label and move on
						ONETWO(labels, h, w, width) = D;
					else if (B && C)
					{ // B and C are labeled
						if (B == C)
						{
							ltable[B] = MIN(B, C);
							ltable[C] = MIN(B, C);
						}
						ONETWO(labels, h, w, width) = MIN(B, C);
					}
					else if (B) // B is object but C is not
						ONETWO(labels, h, w, width) = B;
					else if (C) // C is object but B is not
						ONETWO(labels, h, w, width) = C;
					else
					{ // B, C, D not object - new object
						//   label and put into table
						ntable++;
						ONETWO(labels, h, w, width) = ltable[ntable] = ntable;
					}
					break;
				case 8:
					// apply 8 connectedness
					if (B || C || D || E)
					{
						int tl = B;
						if (B)
							tl = B;
						else if (C)
							tl = C;
						else if (D)
							tl = D;
						else if (E)
							tl = E;
						ONETWO(labels, h, w, width) = tl;
						if (B && B != tl)
							ltable[B] = tl;
						if (C && C != tl)
							ltable[C] = tl;
						if (D && D != tl)
							ltable[D] = tl;
						if (E && E != tl)
							ltable[E] = tl;
					}
					else
					{
						//   label and put into table
						ntable++;
						ONETWO(labels, h, w, width) = ltable[ntable] = ntable;
					}
					break;
				default:
					break;
			}
	
		}
	}
	// consolidate component table
	for (int i = 0; i <= ntable; i++)
	{
		ltable[i] = find(ltable, i);
	}
	// run image through the look-up table
	for (int h = 0; h < height; ++h)
	{
		for (int w = 0; w < width; ++w)
		{
			ONETWO(labels, h, w, width) = ltable[ONETWO(labels, h, w, width)];
		}
	}
	// count up the objects in the image
	for (int i = 0; i <= ntable; i++)
	{
		ltable[i] = 0;
	}
	for (int h = 0; h < height; ++h)
	{
		for (int w = 0; w < width; ++w)
		{
			ltable[ONETWO(labels, h, w, width)]++;
		}
	}
	// number the objects from 1 through n objects
	nobj = 0;
	ltable[0] = 0;
	for (int i = 1; i <= ntable; i++)
	{
		if (ltable[i] > 0)
		{
			ltable[i] = ++nobj;
		}
	}
	// run through the look-up table again
	for (int h = 0; h < height; ++h)
	{
		for (int w = 0; w < width; ++w)
		{
			ONETWO(labels, h, w, width) = ltable[ONETWO(labels, h, w, width)];
		}
	}
	//
	delete[] ltable;
	return nobj;
}

int bwlabel(const cv::Mat & img, cv::Mat & lblMap, int n /* = 8 */)
{
	if (n != 4 && n != 8)
		n = 4;
	int height = img.rows;
	int width = img.cols;
	int total = height * width;
	// results
	uchar * labels = new uchar[total];
	memset(labels, 0, total * sizeof(uchar));
	int nobj = 0; // number of objects found in image
	// other variables
	int * ltable = new int[total]; // label table
	memset(ltable, 0, total * sizeof(int));

	int ntable = 0;
	for (int h = 0; h < height; ++h)
	{
		const uchar * ptr_src = img.ptr<uchar>(h);
		for (int w = 0; w < width; ++w)
		{
			if (ptr_src[w] == 0) // if A is an object
			{
				ONETWO(labels, h, w, width) = NO_OBJECT; // A is not an object so leave it
				continue;
			}

			// get the neighboring pixels B, C, D, and E
			int B = (w == 0) ? 0 : find(ltable, ONETWO(labels, h, w - 1, width));
			int C = (h == 0) ? 0 : find(ltable, ONETWO(labels, h - 1, w, width));	
			int D = (h == 0 || w == 0) ? 0 : find(ltable, ONETWO(labels, h - 1, w - 1, width));
			int E = (h == 0 || w == width - 1) ? 0 : find(ltable, ONETWO(labels, h - 1, w + 1, width));

			switch (n)
			{
			case 4:
				// apply 4 connectedness
				if (B && C)
				{ 
					if (B == C)
						ltable[C] = B;
					ONETWO(labels, h, w, width) = B;
				}
				else if (B) // B is object but C is not
					ONETWO(labels, h, w, width) = B;
				else if (C) // C is object but B is not
					ONETWO(labels, h, w, width) = C;
				else
				{ // B, C, D not object - new object
					//   label and put into table
					ntable++;
					ONETWO(labels, h, w, width) = ltable[ntable] = ntable;
				}
				break;
			case 6:
				// apply 6 connected ness
				if (D) // D object, copy label and move on
					ONETWO(labels, h, w, width) = D;
				else if (B && C)
				{ // B and C are labeled
					if (B == C)
						ONETWO(labels, h, w, width) = B;
					else
					{
						int tl = MIN(B, C);
						ltable[B] = tl;
						ltable[C] = tl;
						ONETWO(labels, h, w, width) = tl;
					}
				}
				else if (B) // B is object but C is not
					ONETWO(labels, h, w, width) = B;
				else if (C) // C is object but B is not
					ONETWO(labels, h, w, width) = C;
				else
				{ // B, C, D not object - new object
					//   label and put into table
					ntable++;
					ONETWO(labels, h, w, width) = ltable[ntable] = ntable;
				}
				break;
			case 8:
				// apply 8 connectedness
				if (B || C || D || E)
				{
					int tl = B;
					if (B)
						tl = B;
					else if (C)
						tl = C;
					else if (D)
						tl = D;
					else if (E)
						tl = E;
					ONETWO(labels, h, w, width) = tl;
					if (B && B != tl)
						ltable[B] = tl;
					if (C && C != tl)
						ltable[C] = tl;
					if (D && D != tl)
						ltable[D] = tl;
					if (E && E != tl)
						ltable[E] = tl;
				}
				else
				{
					//   label and put into table
					ntable++;
					ONETWO(labels, h, w, width) = ltable[ntable] = ntable;
				}
				break;
			default:
				break;
			}
		}
	}

	// consolidate component table 
	for (int i = 0; i <= ntable; ++i)
	{
		ltable[i] = find(ltable, i);
	}
	// run image through the look-up table
	for (int h = 0; h < height; ++h)
	{
		for (int w = 0; w < width; ++w)
		{
			ONETWO(labels, h, w, width) = ltable[ONETWO(labels, h, w, width)];
		}
	}
	// count up the objects in the image
	for (int i = 0; i <= ntable; ++i)
	{
		ltable[i] = 0;
	}
	for (int h = 0; h < height; ++h)
	{
		for (int w = 0; w < width; ++w)
		{
			ltable[ONETWO(labels, h, w, width)]++;
		}
	}
	// number the objects from 1 through n objects
	nobj = 0;
	ltable[0] = 0;
	for(int i = 1; i <= ntable; ++i)
	{
		if (ltable[i] > 0)
		{
			ltable[i] = ++nobj;
		}
	}
	// run through the look-up table again
	for (int h = 0; h < height; ++h)
	{
		for (int w = 0; w < width; ++w)
		{
			ONETWO(labels, h, w, width) = ltable[ONETWO(labels, h, w, width)];
		}
	}
	//
	delete[] ltable;

	lblMap = cv::Mat(height, width, cv::DataType<uchar>::type, labels);

	return nobj;
}

int bwlabel(const cv::Mat & img, cv::Mat & lblMap,
            std::vector<conn_comp_t> & cc, int n /*= 8*/)
{
	int n_cc = bwlabel(img, lblMap);

	std::map<int, vector<cv::Point>> _cc;
	for (int h = 0; h < lblMap.rows; ++h)
	{
		uchar * ptr = lblMap.ptr<uchar>(h);
		for (int c = 0; c < lblMap.cols; ++c)
		{
			_cc[ptr[c]].push_back(cv::Point(c, h));
		}
	}

	cc.resize(n_cc);
	int i = 0;
	while(i < n_cc)
	{
		const vector<cv::Point> & pts = _cc[i + 1];
		int min_x{img.cols}, max_x{-1}, min_y{img.rows}, max_y{-1};
		for (const auto & pos : pts)
		{
			if (pos.x < min_x)      min_x = pos.x;
			if (pos.x > max_x)  	max_x = pos.x;
			if (pos.y < min_y)   	min_y = pos.y;
			if (pos.y > max_y)		max_y = pos.y;
		}
		cc[i].label = i + 1;
		cc[i].bbox = cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
		std::copy(pts.begin(), pts.end(),
		          std::back_inserter(cc[i].coords));
		i += 1;
	}
	return n_cc;
}

bool findMaxCc(const cv::Mat & img, conn_comp_t & ccMax)
{
	vector<conn_comp_t> cc;
	cv::Mat lbl_map;
	int n_cc = bwlabel(img, lbl_map, cc);
	if (n_cc == 0)
		return false;

	int maxArea = -1;
	int idxMax = -1;
	for (int i = 0; i < (int)cc.size(); ++i)
	{
		int len = (int)cc[i].coords.size();
		if (len > maxArea)
		{
			maxArea = len;
			idxMax = i;
		}
	}
	ccMax.label = idxMax + 1;
	ccMax.bbox = cc[idxMax].bbox;
	ccMax.coords.resize(cc[idxMax].coords.size());
	std::copy(cc[idxMax].coords.begin(), cc[idxMax].coords.end(),
	          std::back_inserter(ccMax.coords));
	return true;
}

int findMaxCc(const std::vector<conn_comp_t> & cc)
{
	int maxArea = -1;
	int idxMax = -1;
	for (int i = 0; i < (int)cc.size(); ++i)
	{
		int len = (int)cc[i].coords.size();
		if (len > maxArea)
		{
			maxArea = len;
			idxMax = i;
		}
	}
	return idxMax;
}

} // namespace robot