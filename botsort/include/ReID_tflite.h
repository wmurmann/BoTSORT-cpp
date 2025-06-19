#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "DataType.h"                     // FeatureVector alias (1×D Eigen row)
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

/** optional run-time pre-processing flags */
struct ReIDPreprocParams {
  bool swap_rb      = true;   ///< convert BGR→RGB before normalising
  bool l2_normalise = true;   ///< L2-normalise descriptor after inference
};

class ReIDModel {
public:
  ReIDModel(const std::string& tflite_path,
            const ReIDPreprocParams& pp = {});

  /** extract a 1×D feature (D = 512 for osnet-x0.25) from a BGR crop */
  FeatureVector extract(const cv::Mat& bgr_patch);

  int input_width()  const { return in_w_;  }
  int input_height() const { return in_h_;  }
  int feat_dim()     const { return out_dim_; }

private:
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter>     interpreter_;
  ReIDPreprocParams prep_params_;

  int input_idx_{-1}, output_idx_{-1};
  int in_h_{0}, in_w_{0}, out_dim_{0};
};