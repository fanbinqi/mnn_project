#include <iostream>
#include <string>
#include <ctime>
#include <stdio.h>
#include <omp.h>

#include <MNN/MNNDefine.h>
#include <MNN/MNNForwardType.h>
#include <MNN/Interpreter.hpp>

#include "utils.h"

#define USE_CAMERA 0

std::vector<BoxInfo> decode(cv::Mat &image, std::shared_ptr<MNN::Interpreter> &net,
                            MNN::Session *session, int input_size) {
  std::vector<int> dims{1, input_size, input_size, 3};
  auto nhwc_tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
  auto nhwc_data = nhwc_tensor->host<float>();
  auto nhwc_size = nhwc_tensor->size();
  std::memcpy(nhwc_data, image.data, nhwc_size);

  auto input_tensor = net->getSessionInput(session, nullptr);
  std::vector<int> shape = input_tensor->shape();
  int input_h = shape[1];
  int input_w = shape[2];

  std::cout << "----" << input_h << "-----" << input_w << std::endl;

  input_tensor->copyFromHostTensor(nhwc_tensor);

  net->runSession(session);

  MNN::Tensor *tensor_scores = net->getSessionOutput(session, nullptr);
  MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
  tensor_scores->copyToHostTensor(&tensor_scores_host);
  auto pred_dims = tensor_scores_host.shape();

  const unsigned int num_proposals = pred_dims.at(1);
  const unsigned int num_classes = pred_dims.at(2) - 5;
  std::vector<BoxInfo> bbox_collection;

  for (int i = 0; i < num_proposals; ++i) {
    const float *offset_obj_cls_ptr = tensor_scores_host.host<float>() + (i * (num_classes + 5));
    float obj_conf = offset_obj_cls_ptr[4];
    if (obj_conf < 0.5) continue;

    float cls_conf = offset_obj_cls_ptr[5];
    unsigned int label = 0;
    for (unsigned int j = 0; j < num_classes; ++j) {
      float tmp_conf = offset_obj_cls_ptr[j + 5];
      if (tmp_conf > cls_conf) {
        cls_conf = tmp_conf;
        label = j;
      }
    }

    float conf = obj_conf * cls_conf;
    if (conf < 0.50)
      continue;

    float cx = offset_obj_cls_ptr[0];
    float cy = offset_obj_cls_ptr[1];
    float w = offset_obj_cls_ptr[2];
    float h = offset_obj_cls_ptr[3];

    float x1 = (cx - w / 2.f);
    float y1 = (cy - h / 2.f);
    float x2 = (cx + w / 2.f);
    float y2 = (cy + h / 2.f);

    BoxInfo box;
    box.x1 = std::max(0.f, x1);
    box.y1 = std::max(0.f, y1);
    box.x2 = std::min(x2, (float)input_size - 1.f);
    box.y2 = std::min(y2, (float)input_size - 1.f);
    box.score = conf;
    box.label = label;
    bbox_collection.push_back(box);
  }

  delete nhwc_tensor;
  return bbox_collection;
}

int main(int argc, char **argv) {
  std::string model_name = "/workspace/user/fanbinqi/models/yolov5n.mnn";
  std::shared_ptr<MNN::Interpreter> net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_name.c_str()));
  
  if (net == nullptr) return 0;
  MNN::ScheduleConfig config;
  config.numThread = 4;
  config.type = MNN_FORWARD_AUTO;
  MNN::BackendConfig backendConfig;
  backendConfig.precision = MNN::BackendConfig::Precision_High;
  config.backendConfig = &backendConfig;

  auto session = net->createSession(config);
  std::vector<BoxInfo> bbox_collection;

  cv::Mat image;
  MatInfo mmat_objection;
  mmat_objection.inp_size = 640;

  for (size_t i = 0; i < 100; i++) {
    bbox_collection.clear();

    struct timespec begin, end;
    long time;
    clock_gettime(CLOCK_MONOTONIC, &begin);

    std::string image_name = "../images/000000001000.jpg";
    cv::Mat raw_image = cv::imread(image_name.c_str());

    cv::Mat pimg = preprocess(raw_image, mmat_objection);

    bbox_collection = decode(pimg, net, session, mmat_objection.inp_size);

    nms(bbox_collection, 0.50);

    draw_box(raw_image, bbox_collection, mmat_objection);

    clock_gettime(CLOCK_MONOTONIC, &end);
    time = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec);
    if (time > 0)
      printf(">> Time : %lf ms\n", (double)time / 1000000);
  }

  return 0;
}
