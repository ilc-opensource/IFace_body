/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * */

#pragma once

#include <opencv/cv.h>
#include "basetypes.hpp"

#if WIN32

#if defined(IFace_EXPORTS)
#define IFACEAPI __declspec(dllexport)
#else
#define IFACEAPI __declspec(dllimport)
#endif

#else

#define IFACEAPI

#endif


// declare virtual classes
class CxFaceDetector;
class CxCompDetBase;
class CxCompDetector;
class CxCompDetector7pt;
class CxBoostDetect;
class CxMCBoostDetect;
class CxBoostFaceRecog;
class CxFaceAnalyzer;
class CxAlignFace;
class CNNClassifier;

// drawing face API
IFACEAPI void iFaceDrawCrossPoint(IplImage *img, CvPoint pt, int thickness = 1);

IFACEAPI void iFaceDrawFaceRect(IplImage *img, CvRect rect, CvScalar colors = CV_RGB(0, 255, 0));

IFACEAPI void iFaceDrawCaption(IplImage *img, CvFont *pFont, char *sCaption);

IFACEAPI void iFaceDrawFaceBlob(IplImage *img, CvFont *pFont, int id, CvRect rect, CvPoint2D32f *landmark6,
				float probSmile = 0, int bBlink = 0, int bSmile = 0, int bGender = 0, int nAgeID = 0,
                                char *sFaceName = NULL, char *sCaption = NULL, IplImage *pImgSmileBGR = NULL,
                                IplImage *pImgSmileBGRA = NULL, IplImage *pImgSmileMask = NULL);

IFACEAPI void iFaceAutoFocusFaceImage(IplImage *pSrc, IplImage *pDest, CvRectItem *vFaceRect, int faceNum,
				      float alpha = 0.05);

class IFACEAPI IFaceAlignFace {
  public:
    IFaceAlignFace(int sizeSmallFace = 64, int sizeBigFace = 128);
     IFaceAlignFace(IplImage *pGrayImg, CvRect rect, CvPoint2D32f landmark6[]);
    ~IFaceAlignFace();

    void init(IplImage *pImg, CvRect rect, CvPoint2D32f landmark6[]);
    void clear();

    IplImage *getBigCutFace();
    IplImage *getSmallCutFace();

    int getBigCutFaceSize();
    int getSmallCutFaceSize();

    IplImage *getAlignedFaceThumbnail(IplImage *pColorImg, CvRect rect, CvPoint2D32f landmark6[]);

  private:
     CxAlignFace *m_alignFace;
};

/************************** FaceDetector *****************************/
class IFACEAPI IFaceFaceDetector {
  public:
    IFaceFaceDetector();
    IFaceFaceDetector(EnumViewAngle viewAngle /*= VIEW_ANGLE_FRONTAL*/ , EnumFeaType feaType = FEA_SURF,
                      int nFaceDetectorNo = 0, const char *modelFile = NULL);
    ~IFaceFaceDetector(void);

    // init
    void init(EnumViewAngle viewAngle = VIEW_ANGLE_FRONTAL, EnumFeaType feaType = FEA_SURF, int nFaceDetectorNo = 0,
              const char *modelFile = NULL);

    // configure parameters
    //void config( tagDetectConfig configParam = tagDetectConfig());

    // detect face, return the number and rect of faces
    int detect(IplImage *image, CvRectItem rects[], int nColorFlag);

    void setFaceDetectionROI(IplImage *lpImage, double dCenterRatio);
    void setFaceDetectionROI(CvRect DetecteRegion);

    void setFaceDetectionSizeRange(IplImage *lpImage, int nMinFaceSize = 60);
    void setFaceDetectionSizeRange(int nMinFaceSize, int nMaxFaceSize);

    void clearFaceDetectionRange();
    void clearFaceDetectionROI();
    // get face thumbnail from rect of image
    //IplImage* getThumbnail( IplImage* image, CvRect rect, IplImage* thumbnail = NULL);

  private:
    int m_faceDetectorNo;
};


/************************** FaceTracker *****************************/
class IFACEAPI IFaceFaceTracker {
  public:
    IFaceFaceTracker();
    IFaceFaceTracker(EnumViewAngle viewAngle /*= VIEW_ANGLE_FRONTAL*/ , EnumTrackerType traType = TRA_SURF,
                     const char *xmlfile = NULL);
    ~IFaceFaceTracker(void);

    // init
    void init(EnumViewAngle viewAngle = VIEW_ANGLE_FRONTAL, EnumTrackerType traType = TRA_SURF,
              const char *xmlfile = NULL);

    // configure parameters
    void config(tagDetectConfig configParam = tagDetectConfig(), int level = TR_NLEVEL_3);

    // detect faces in the image
    int detect(IplImage *image, CvRectItem rects[], int count);

    // tracking faces in a video, if pBGRImage != NULL, color face tracker will be enabled
    int track(IplImage *pGreyImage, CvRectItem rects[], int count, IplImage *pBGRImage = NULL);

    // get face thumbnail from rect of image
    IplImage *getThumbnail(IplImage *image, CvRect rect, IplImage *thumbnail = NULL);

  private:
    //      CxTrackerBase* m_tracker;
};

/************************** Face Landmark *****************************/
class IFACEAPI IFaceLandmarkDetector {
  public:
    IFaceLandmarkDetector();
    IFaceLandmarkDetector(EnumLandmarkerType landmarkerType /*= LDM_6PT*/ , const char *xmlEyeLeftCorner = NULL,
			  const char *xmlMthLeftCorner = NULL, const char *xmlNose = NULL);
    ~IFaceLandmarkDetector(void);

    // init
    void init(EnumLandmarkerType landmarkerType = LDM_7PT, const char *xmlEyeLeftCorner = NULL,
	      const char *xmlMthLeftCorner = NULL, const char *xmlNose = NULL);

    // detect 6 points within 'rc_face', output to 'pt_comp'
    bool detect(const IplImage *image, CvRect *rect, CvPoint2D32f points[], float parameters[] = NULL, int angle = 0);
    bool track(const IplImage *image, CvRect *rect, CvPoint2D32f points[], float parameters[] = NULL, int angle = 0);

    // retrieve each comp
    CvPoint2D32f getPoint(int comp) const;

    //align face: pGrayImg must be gray image
    IplImage *alignFace(const IplImage *pGrayImg, const CvPoint2D32f pt6s[], CvRect rc,
			int nDstImgW, int nDstImgH, bool bHistEq = true, float *sclxyud = NULL,
                        IplImage *pCutFace = NULL);

  private:
     CxCompDetBase *m_comp;
};

/************************** Smile Detector *****************************/

class IFACEAPI IFaceSmileDetector {
  public:
    IFaceSmileDetector();
    IFaceSmileDetector(int cutImgSize /*= 64*/ , const char *modelPath = NULL);
    ~IFaceSmileDetector(void);

    // init
    void init(int cutImgSize = 64, const char *modelPath = NULL);

    //threshold of predict, no smile: [0,0.48], smile: (0.48, 1]
    int predict(IFaceAlignFace *pCutFace, float *prob = NULL);
    int predict(IplImage *pCutFace, float *prob = NULL);
    int voteLabel(int faceTrackID, int label, int voteThreshold = 2, int smoothLen = 8);

    float getDefThreshold();
    int getDefRound();

  private:
     CxBoostDetect *m_smileDetector;
};

/************************** Blink Detector *****************************/
class IFACEAPI IFaceBlinkDetector {
  public:
    IFaceBlinkDetector();
    IFaceBlinkDetector(int cutImgSize /*= 64*/ , const char *modelPath = NULL);
    ~IFaceBlinkDetector(void);

    // init
    void init(int cutImgSize = 64, const char *modelPath = NULL);

    //threshold of predict, open: [0,0.5], close: (0.5, 1]
    int predict(IFaceAlignFace *pCutFace, float *prob = NULL);
    int predict(IplImage *pCutFace, float *prob = NULL);
    int voteLabel(int faceTrackID, int label, int voteThreshold = 1, int smoothLen = 2);

    float getDefThreshold();
    int getDefRound();

  private:
     CxBoostDetect *m_blinkDetector;
};

/************************** Gender Detector *****************************/
class IFACEAPI IFaceGenderDetector {
  public:
    IFaceGenderDetector();
    IFaceGenderDetector(int cutImgSize /*= 64*/ , const char *modelPath = NULL);
    ~IFaceGenderDetector(void);

    // init
    void init(int cutImgSize = 64, const char *modelPath = NULL);

    //threshold of predict, male: [0,0.42], female: (0.42, 1]
    int predict(IFaceAlignFace *pCutFace, float *prob = NULL);
    int predict(IplImage *pCutFace, float *prob = NULL);
    int voteLabel(int faceTrackID, int label);

    float getDefThreshold();
    int getDefRound();

  private:
     CxBoostDetect *m_genderDetector;
};

/************************** Age Detector *****************************/
class IFACEAPI IFaceAgeDetector {
  public:
    IFaceAgeDetector();
    IFaceAgeDetector(int cutImgSize /*= 128*/ , const char *modelPath = NULL);
    ~IFaceAgeDetector(void);

    // init
    void init(int cutImgSize = 128, const char *modelPath = NULL);

    // output class-id: 0=>baby, 1=>child, 2=>adult, 3=>senior
    int predict(IFaceAlignFace *pCutFace, float *prob = NULL);
    int predict(IplImage *pCutFace, float *prob = NULL);
    int voteLabel(int faceTrackID, int label);

    int getDefRound();

  private:
     CxMCBoostDetect *m_ageDetector;
};

/************************** Face Recognizer *****************************/
class IFACEAPI IFaceFaceRecognizer {
  public:
    IFaceFaceRecognizer();
    IFaceFaceRecognizer(int cutImgSize /*= 128*/ , int recognierType = RECOGNIZER_CAS_GLOH,
                        const char *modelPath = NULL);
    ~IFaceFaceRecognizer(void);

    // init
    void init(int cutImgSize = 128, int recognierType = RECOGNIZER_CAS_GLOH, const char *modelPath = NULL);

    // load exemplar face set xml file
    int loadFaceModelXML(const char *xmlPath);
    void saveFaceModelXML(const char *sPathXML, vFaceSet *pvecFaceSet = NULL);
    void getMergedFaceSet(vFaceSet &vvClusters, int minWeight = 0);

    int insertEmptyFaceSet(char *name, bool createFolder = true, int nFaceSetID = -1);
    bool tryInsertFace(IplImage *pCutFace, int nFaceSetIdx, bool bForceInsert = false);
    int removeFaceSet(int nFaceSetIdx);
    int removeFace(int nFaceSetIdx, int faceIdx);

    vFaceSet *getFaceSets();
    const char *getFaceName(int nFaceSetID);
    int getFaceSetID(int nFaceSetIdx);
    int getFaceSetIdx(int nFaceSetID);
     std::vector<std::string> *getKeyFacePaths(int faceSetIdx);
    int getFaceSetSize(int nFaceSetIdx = -1);
    const char *getKeyFacePath(int nFaceSetIdx, int nFaceIdx = -1);

    // output face-id, prob > 0.52 is the face-id person, otherwise reject the person.
    int predict(IFaceAlignFace *pCutFace, float *prob = NULL, bool bAutoCluster = false,
                int faceTrackID = -1, int frameID = -1);
    int predict(IplImage *pCutFace, float *prob = NULL, bool bAutoCluster = false,
                int faceTrackID = -1, int frameID = -1);

    int voteLabel(int faceTrackID, int label);

    float getDefThreshold();
    int getDefRound();

    int getFeatureDim();
    int getFeatureType();

    void extFeature(IplImage *pCutFaceImg, float *pFea);
    bool isSimilarFaces(float *pFea1, float *pFea2, float *pProb = NULL);

    ////////////////////////////////////////////////////////////////////
    //cluster faces
    int forwardCluster(float *pFea, int faceID, char *sCutFaceImg, vFaceSet &vvClusters, vFaceSet &vvRepClusters,
		       float fThreshold = -1);

    int clusterHAC(CvMat *pmSim, vFaceSet &vvFaceSet, float fThreshold = -1,
		   int nMinClusterNum = -1, std::vector<int> *pvExemplars = NULL);

    int rankOrderCluster(CvMat *pmSim, vFaceSet &vvClusters, float rankDistThresh = 12, float normDistThresh = 1.02);

    void mergeClusters(vFaceSet &vvClusters, int cA, int cB, vFaceSet *vvRepClusters = NULL);

    CvMat *clacSimMat(std::vector<std::string> vFaceImgList, CvMat *&pmSim);
    CvMat *clacSimMat(std::vector<CvMat *> matFea, CvMat *&pmSim);

  private:
     CxBoostFaceRecog * m_faceRecognizer;
};

/************************** Face Analyzer *****************************/
class IFACEAPI IFaceFaceAnalyzer {
  public:
    IFaceFaceAnalyzer();
    IFaceFaceAnalyzer(EnumViewAngle viewAngle, EnumTrackerType traType = TRA_SURF, int nFaceDetector = 0,
		      const char *strFaceSetXml = NULL, int recognierType = RECOGNIZER_CAS_GLOH,
                      bool bEnableAutoCluster = false, bool bLandmarkRegressor = false,
                      const char *xmlEyeLeftCorner = NULL, const char *xmlMthLeftCorner = NULL,
                      const char *sFaceRecognizerModelPath = NULL);
    ~IFaceFaceAnalyzer(void);

    // init
    void init(EnumViewAngle viewAngle = VIEW_ANGLE_FRONTAL, EnumTrackerType traType = TRA_SURF, int nFaceDetector = 0,
	      const char *str_facesetxml = NULL, int recognierType = RECOGNIZER_CAS_GLOH,
              bool bEnableAutoCluster = false, bool bLandmarkRegressor = false, const char *xmlEyeLeftCorner = NULL,
              const char *xmlMthLeftCorner = NULL, const char *sFaceRecognizerModelPath = NULL);

    void loadFaceModelXML(const char *sPathXML);
    void saveFaceModelXML(const char *sPathXML, vFaceSet *pvecFaceSet = NULL);
    void getMergedFaceSet(vFaceSet &vvClusters, int minWeight = 0);

    int insertEmptyFaceSet(char *name, bool createFolder = true, int nFaceSetID = -1);
    bool tryInsertFace(IplImage *pCutFace, int nFaceSetIdx, bool bForceInsert = false);
    int removeFaceSet(int nFaceSetIdx);
    int removeFace(int nFaceSetIdx, int faceIdx);

    vFaceSet *getFaceSets();
    char *getFaceImgDBPath();
    IplImage *getBigCutFace();
    int getFaceSetID(int nFaceSetIdx);
    int getFaceSetSize(int nFaceSetIdx = -1);
    const char *getKeyFacePath(int nFaceSetIdx, int nFaceIdx = -1);

    //prop_estimate: #fk #fa #fs #mk #ma #ms #smile
    //pStat: %02dperson (likely [%1dMale, %1dFemale]; [%1dkid, %1dadult, %1dsenior]), Prefer
    void detect(IplImage *pImg, int *propEstimate, char *pStat = NULL);
    bool faceDetection(IplImage *pColorImg, int nMinFaceSize, char *ThumbnailImgFilanem);
    //void track(IplImage *pGreyImg, int *prop_estimate, char *pStat = NULL, IplImage *pBGRImg = NULL);

    //predict pCutFace's facesetID and sFaceName
    int predictFaceSet(IplImage *pCutFace, float *prob, char *sFaceName = NULL);

    int getMaxFaceNum() { return 16; }
    int getFaceNum();
    int getFaceTrackID(int idx);
    int *getFaceTrackIDs();
    CvRectItem getFaceRect(int idx);
    CvRectItem *getFaceRects();
    CvPoint2D32f *getFaceLdmks(int idx);

    int getFaceID(int idx);
    float getFaceProb(int idx);
    char *getFaceName(int idx);

  private:
    CxFaceAnalyzer *m_faceAnalyzer;
};

/************************** Iflib end *****************************/
