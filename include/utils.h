/**
 * 
 */

#pragma once

#include <MNN/MNNDefined.h>
#include <MNN/MNNForwardType.h>
#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>
#include <opencv2/opencv.hpp>

typedef struct {
  float x1;
  float x2;
  float y1;
  float y2;
  float score;
  int label;
} BoxInfo;

typedef struct {
  int inp_size;
  int max_side;
  int pad_w;
  int pad_h;
  float ratio;
} MatInfo;

cv::Mat preprocess(cv::Mat &image, MatInfo &mat_info);

void nms(std::vector<BoxInfo> &result, float nms_threshold);

void draw_box(cv::Mat &image, std::vector<BoxInfo> &boxes, MatInfo &mat_info);
