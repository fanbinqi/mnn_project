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

cv::Mat preprocess(cv::Mat &image, MatInfo &mat_info) {
  cv::Mat img;
  cv::Mat dst_img;

  cv::cvtColor(image, dst_img, cv::COLOR_BGR2RGB);

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

  return img;
}

void nms(std::vector<BoxInfo> &input_boxes, float nms_thresh) {
  std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });

  int num_boxes = input_boxes.size();
  std::vector<float> v_area(num_boxes);
  for (int i = 0; i < num_boxes; ++i) {
    v_area[i] = (input_boxes[i].x2 - input_boxes[i].x1 + 1) * (input_boxes[i].y2 - input_boxes[i].y1 + 1);
  }
#pragma omp parallel num_threads(2)
#pragma omp parallel for
  for (int i = 0; i < input_boxes.size(); ++i) {
    for (int j = i + 1; j < input_boxes.size(); ) {
      float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
      float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
      float xx2 = std::max(input_boxes[i].x2, input_boxes[j].x2);
      float yy2 = std::max(input_boxes[i].y2, input_boxes[j].y2);
      float w = std::max(float(0), xx2 - xx1 + 1);
      float h = std::max(float(0), yy2 - yy1 + 1);
      float inter = w * h;
      float ovr = inter / (v_area[i] + v_area[j] - inter);
      if (ovr >= nms_thresh) {
        input_boxes.erase(input_boxes.begin() + j);
        v_area.erase(v_area.begin() + j);
      } else {
        j++;
      }
    }
  }
}

void draw_box(cv::Mat &image, std::vector<BoxInfo> &boxes, MatInfo &mat_info) {
  static const char *class_names[] = {"0", "1"};

  int x1, y1, objw, objh;
  char text[256];
  printf("bboxes num is: %d \n", boxes.size());
  for (auto box : boxes) {
    x1 = int(box.x1 / mat_info.ratio) - int(mat_info.pad_w / mat_info.ratio);
    y1 = int(box.y1 / mat_info.ratio) - int(mat_info.pad_h / mat_info.ratio);
    objw = int((box.x2 - box.x1) / mat_info.ratio);
    objh = int((box.y2 - box.y1) / mat_info.ratio);
    cv::Point pos = cv::Point(x1, y1 - 5);
    printf("%f, %f, %f, %f, %f\n", box.x1, box.y1, box.x2, box.y2, box.score);
    cv::Rect rect = cv::Rect(x1, y1, objw, objh);
    cv::rectangle(image, rect, cv::Scalar(0, 255, 0));
    printf("-------%d \n", __LINE__);
    sprintf(text, "%s %.1f%%", class_names[box.label], box.score * 100);
    cv::putText(image, text, pos, cv::FONT_HERSHEY_SIMPLEX, (float)objh / (float)mat_info.max_side,
                cv::Scalar(0, 0, 255), 1);
  }
#if use_camera
  cv::imshow("Fourcc", image);
  cv::waitKey(1);
#else
printf("-------%d \n", __LINE__);
  cv::imwrite("./yolov5_result.jpg", image);
#endif
}
