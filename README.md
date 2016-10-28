

# Tablet Robot ININ Visual Libraries

Tablet Robot ININ was created by Intel Labs China based on Intel® RealSense™ technology.

## Introduction

Tablet Robot ININ Visual Libraries consist of:
- IHumanDetector_RGBD: detect head region in input pair of RGB frame and depth frame from RealSense.
- IHumanOriEstimation: identify head orientation.
- IFace: detect face and recognize it in registered faces.

## Folder Structure

There are several folders under root directory:
- The data folder includes necessary files needed by libraries and sample projects;
- The 3rd_party folder includes additional 3rd party code dependencies;
- The extra folder includes additional code for samples/tests to use;
- The include folder includes head files of the visual libraries;
- The samples folders include code samples.

## Build Introduction

Both Windows and Linux are supported.

Before build the project, please install [OpenCV](http://www.opencv.org) and [librealsense](https://github.com/IntelRealSense/librealsense) first, and if in Windows, setup environment variables OpenCV_DIR and LIBRS pointing to each directory, CMake will find the libraries according to them.

Use CMake to build. For example in Linux:
```
mkdir build
cd build
cmake ..
make
```
There're options user can choose to build the project.

The libraries and executables will be in 'bin' folder under project root directory.

## Sample Usage

In the following examples, the working directory is set to the project root directory and the file paths in configuration files are relative paths according to project root directory.

* DumbbellCounting Usage:

  ```
  DumbbellCounting DumbbellCounting_config_file
  e.g. DumbbellCounting ./data/DumbbellCounting_config.yml
  ```

* HumanDetection Usage:

  HumanDetection is the demo program on human detection and orientation classification. Please connect RealSense R200 first because it's live demo. The image will show the human detection result (big rectangle), orientation classification result and their computation time.

  ```
  HumanDetection HumanDetection_config_file
  e.g. HumanDetection ./data/HumanDetection_config.yml 
  ```

* 3DPoseEstimation Usage:

  ```
  3DPoseEstimation opencv_config_file
  e.g. 3DPoseEstimation ./data/3DPoseEstimation_config.yml
  ```

* IFaceTest Usage:

  ```
  # list01.txt and list02.txt are lists of pictures containing faces, one picture path per line.
  # Register face pictures.
  IFaceTest -c configure_file -r ImgList.txt person_name
  e.g. IFaceTest -c ./data/IFaceTest/iface_configure.yml -r ./data/IFaceTest/list01.txt Alice

  # Detect face in input pictures and output to target folder.
  IFaceTest -c configure_file -i ImgList.txt [ -o output_directory ]
  e.g. IFaceTest -c ./data/IFaceTest/iface_configure.yml -i ./data/IFaceTest/list02.txt
  ```
## Note

The binary libraries are 64bit, and built on:
- Windows 8.1 64bit, Visual Studio 2013, OpenCV 3.1.0
- Ubuntu 14.04 64bit, OpenCV 3.1.0.