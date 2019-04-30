cmake_minimum_required(VERSION 2.8)
project(TSDF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${CUDA_INCLUDE_DIRS}
)

find_package(CUDA REQUIRED)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -use_fast_math)
SET(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
cuda_add_library(${PROJECT_NAME} SHARED src/tsdf.cu include/tsdf.cuh)

find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if (NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif ()
endif ()

find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Example)

add_executable(load_frames src/main.cpp
        src/PointCloudGenerator.cpp
        src/SaveFrame.cpp
        src/safecall.cpp)

target_link_libraries(load_frames 
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
    glog
	glut GL GLU GLEW)

