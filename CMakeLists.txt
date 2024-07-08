cmake_minimum_required(VERSION 3.5.1)
project(mnn_yolov5)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-std=c++14 -o3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fopenmp")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} -fopenmp")
set(OpenCV_DIR "${PROJECT_SOURCE_DIR}/opencv-3.4.3/share/OpenCV")

find_package(OpenCV REQUIRED)

MESSAGE(STATUS "Project: ${PROJECT_NAME}")
MESSAGE(STATUS "OpenCV library status:")
MESSAGE(STATUS "    version: ${OpenCV_VERSION}")
MESSAGE(STATUS "    libraries: ${OpenCV_LIBS}")
MESSAGE(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${PROJECT_SOURCE_DIR}/include/)
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/lib)

add_executable(mnn_yolov5 src/main.cpp src/utils.cpp)

target_link_libraries (mnn_yolov5 ${OpenCV_LIBS} libMNN.so libMNN_Express.so libMNNOpenCV.so -lpthread)
