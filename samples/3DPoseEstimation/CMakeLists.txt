include_directories("${PROJECT_SOURCE_DIR}/extra/RSWrapper")

file(GLOB 3DPoseEstimation_srcs "*.cpp")
file(GLOB 3DPoseEstimation_hdrs "*.h*")

if(USE_PREBUILT_LIBRARIES)
    include_directories(${CMAKE_SOURCE_DIR}/3rd_party/opencv/include)
    link_directories(${CMAKE_SOURCE_DIR}/3rd_party/opencv/lib)
    link_directories(${CMAKE_SOURCE_DIR}/3rd_party/librealsense/lib)
endif()

if(WIN32)
	if(CMAKE_BUILD_TYPE MATCHES "Debug")
	    link_directories(${CMAKE_SOURCE_DIR}/lib/Windows/Debug)
	else()
	    link_directories(${CMAKE_SOURCE_DIR}/lib/Windows/Release)
	endif()
else()
	if(CMAKE_BUILD_TYPE MATCHES "Debug")
	    link_directories(${CMAKE_SOURCE_DIR}/lib/Linux/Debug)
	else()
	    link_directories(${CMAKE_SOURCE_DIR}/lib/Linux/Release)
	endif()
endif()

add_executable(3DPoseEstimation ${3DPoseEstimation_srcs} ${3DPoseEstimation_hdrs})
target_link_libraries(3DPoseEstimation IFace3DPoseEstimation IFace RSWrapper)


if(USE_PREBUILT_LIBRARIES)
	if(CMAKE_BUILD_TYPE MATCHES "Debug")
		target_link_libraries(3DPoseEstimation debug opencv_core310d debug opencv_imgproc310d debug opencv_highgui310d debug opencv_imgcodecs310d)
	else()
		target_link_libraries(3DPoseEstimation optimized opencv_core310 optimized opencv_imgproc310 optimized opencv_highgui310 optimized opencv_imgcodecs310)
	endif()
else()
    target_link_libraries(3DPoseEstimation ${OpenCV_LIBS})
endif()

if(UNIX)
    target_link_libraries(3DPoseEstimation gomp)
endif()

if(RSWrapper_as_Shared)
    set_target_properties(3DPoseEstimation PROPERTIES COMPILE_DEFINITIONS "-DRSWrapper_SHARED")
endif()

set_target_properties(3DPoseEstimation PROPERTIES FOLDER "Samples")
