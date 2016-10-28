/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * */

#ifndef  PHOTO_INDEXING_UTILITY_H
#define PHOTO_INDEXING_UTILITY_H


bool PhotoIndexing_Init(string& configure_file);
void PhotoIndexing_Release();
int PhotoIndexing_ImageEvaluation(char *sFilename, int *scene_label);
int PhotoIndex_ImageEvaluation(IplImage* color_image);
#endif