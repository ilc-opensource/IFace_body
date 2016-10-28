cmake_minimum_required(VERSION 2.8)
project(poc-sdk)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")


add_definitions(-D__SSE2__ -DUSE_OMP)
if(WIN32)
    add_definitions(/wd4996)
    add_definitions(-DWIN32)
    add_compile_options(/openmp /arch:AVX)
    option(USE_PREBUILT_LIBRARIES "Use prebuilt vs2013 x64 3rdparty libraries on Windows; unset this if you have these libraries installed or build other versions." ON)
else()
    add_compile_options(-fopenmp -mavx)

    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
    endif(NOT CMAKE_BUILD_TYPE)
    add_compile_options(-Wall)

    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
    if(COMPILER_SUPPORTS_CXX11)
	add_compile_options(-std=c++11)
    else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use higher version C++ compiler.")
    endif()
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include_directories(include)

if(NOT USE_PREBUILT_LIBRARIES)
    find_package(OpenCV REQUIRED)
endif()

option(BUILD_SAMPLES "Build samples" ON)
if(BUILD_SAMPLES)
    add_subdirectory(samples)
endif()

if(BUILD_TESTS OR BUILD_SAMPLES)
    add_subdirectory(extra)
endif()

option(ENABLE_DEBUG_MESSAGES "Enable debug messages for core libs" ON)
if(ENABLE_DEBUG_MESSAGES)
    add_definitions(-DCORELIB_DEBUG)
endif()

