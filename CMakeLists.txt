cmake_minimum_required(VERSION 2.8)
project(TSDF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")

find_package(CUDA REQUIRED)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -use_fast_math)
include_directories(${CUDA_INCLUDE_DIRS})


find_package(OpenCV 3.3.1 QUIET)
if (NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if (NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif ()
endif ()

#find_package(CUDA REQUIRED)
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -use_fast_math)
SET(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
cuda_add_library(${PROJECT_NAME} SHARED tsdf.cu tsdf.cuh)

include_directories(${CUDA_INCLUDE_DIRS})

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
)

add_executable(demo main.cpp)

target_link_libraries(demo 
	${PROJECT_NAME}
	${OpenCV_LIBS} 
	${CUDA_LIBRARIES} 
	glut GL GLU GLEW)