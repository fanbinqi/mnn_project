#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <omp.h>
#include "utils.h"

#define X86         0
#define USE_CAMERA  0

cv::Mat preprocess(cv::Mat &iamge, MatInfo &mat_info) {
  cv::Mat img;
  cv::Mat dst_img;

  cv::cvtcolor(image, dst_img, cv::COLOR_BGR2RGB);

  mat_info.max_side = image.rows > image.cols ? image.rows : image.cols;
  mat_info.ratio = float(mat_info.inp_size) / float(mat_info.max_side);
  int f_x = int(image.cols * mat_info.ratio);
  int f_y = int(image.rows * mat_info.ratio);
  mat_info.pad_w = int((mat_info.inp_size - f_x) * 0.5);
  mat_info.pad_h = int((mat_info.inp_size - f_y) * 0.5);
  cv::resize(dst_img, img, cv::Size(f_x, f_y));
  cv::copyMakeBorder(img, img, mat_info.pad_h, mat_info.pad_h, mat_info.pad_w,
                     mat_info.pad_w, cv::BORDER_CONSTANT, cv::Scalar::all(127));

  img.convertTo(img, CV_32FC3);
  img = img / 255.0f;

  return img
}

void nms(std::vector<BoxInfo> &input_boxes, float nms_thresh) {
  std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });

  int num_boxes = input_boxes.size()
  std::vector<float> v_area(num_boxes);
  for (int i = 0; i < num_boxes; ++i) {
    v_area[i] = (input_boxes[i].x2 - input_boxes[i].x1 + 1) * (input_boxes[i].y2 - input_boxes[i].y1 + 1);
  }
#pragma omp parallel

}