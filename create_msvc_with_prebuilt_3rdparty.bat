@echo off
echo Creating Microsoft Visual Studio 2013 x64 solution files...

cmake -G"Visual Studio 12 2013 Win64" -B".\build\Release" --no-warn-unused-cli -DCMAKE_BUILD_TYPE:STRING="Release" -DCMAKE_CONFIGURATION_TYPES:STRING="Release" -DUSE_PREBUILT_LIBRARIES:BOOL="ON"  -DBUILD_TESTS:BOOL="ON" -H"."
cmake -G"Visual Studio 12 2013 Win64" -B".\build\Debug" --no-warn-unused-cli -DCMAKE_BUILD_TYPE:STRING="Debug" -DCMAKE_CONFIGURATION_TYPES:STRING="Debug" -DUSE_PREBUILT_LIBRARIES:BOOL="ON"  -DBUILD_TESTS:BOOL="ON" -H"."

echo Copy 3rdparty prebuilt DLLs to bin

xcopy 3rd_party\librealsense\bin\realsense-d.dll bin\Debug\ /Y
xcopy 3rd_party\opencv\bin\*310d.dll bin\Debug\ /Y
xcopy 3rd_party\openblas\bin\*.dll bin\Debug\ /Y

xcopy 3rd_party\librealsense\bin\realsense.dll bin\Release\ /Y
xcopy 3rd_party\opencv\bin\*310.dll bin\Release\ /Y
xcopy 3rd_party\openblas\bin\*.dll bin\Release\ /Y

xcopy src\face\debug\IFace.dll bin\Debug\ /Y
xcopy src\face3DPoseEstimation\debug\IFace3DPoseEstimation.dll bin\Debug\ /Y
xcopy src\FaceAttributeRecognition\debug\FaceAttributeRecognition.dll bin\Debug\ /Y
xcopy src\humanDetector\debug\IHumanDetector_RGBD.dll bin\Debug\ /Y
xcopy src\humanOriEstimation\debug\IHumanOriEstimation.dll bin\Debug\ /Y 

xcopy src\face\release\IFace.dll bin\Release\ /Y
xcopy src\face3DPoseEstimation\release\IFace3DPoseEstimation.dll bin\Release\ /Y
xcopy src\FaceAttributeRecognition\release\FaceAttributeRecognition.dll bin\Release\ /Y
xcopy src\humanDetector\release\IHumanDetector_RGBD.dll bin\Release\ /Y
xcopy src\humanOriEstimation\release\IHumanOriEstimation.dll bin\Release\ /Y

echo Done.
pause
