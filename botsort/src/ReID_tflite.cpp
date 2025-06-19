#include "ReID_tflite.h"

#include <opencv2/imgproc.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include <iostream>
#include <memory>
#include <stdexcept>

namespace {

// Convenience: check TfLiteStatus and throw on failure
inline void TfLiteCheck(TfLiteStatus s, const char* where) {
  if (s != kTfLiteOk) {
    throw std::runtime_error(std::string("TFLite error at ") + where);
  }
}

}  // namespace

/* -------------------------------------------------------------------------- */
/*                         ReID_tflite public methods                         */
/* -------------------------------------------------------------------------- */

ReID_tflite::ReID_tflite(const std::string& model_path,
                         const ReIDPreprocParams& pp)
    : prep_params_(pp) {
  // Load flat‑buffer model
  model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  if (!model_) {
    throw std::runtime_error("Failed to mmap TFLite model: " + model_path);
  }

  // Build interpreter with NNAPI / XNNPACK delegates disabled (portable)
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);
  if (!interpreter_) {
    throw std::runtime_error("Failed to construct TFLite interpreter");
  }

  // Use single thread; caller may adjust
  interpreter_->SetNumThreads(1);

  TfLiteCheck(interpreter_->AllocateTensors(), "AllocateTensors");

  // Cache input/output indices and shape
  input_idx_  = interpreter_->inputs()[0];
  output_idx_ = interpreter_->outputs()[0];

  const TfLiteTensor* in_t = interpreter_->tensor(input_idx_);
  if (in_t->dims->size != 4 || in_t->type != kTfLiteFloat32) {
    throw std::runtime_error("Unexpected input tensor shape/type");
  }
  in_h_ = in_t->dims->data[1];
  in_w_ = in_t->dims->data[2];

  const TfLiteTensor* out_t = interpreter_->tensor(output_idx_);
  if (out_t->dims->size != 2 || out_t->type != kTfLiteFloat32) {
    throw std::runtime_error("Unexpected output tensor shape/type");
  }
  out_dim_ = out_t->dims->data[1];
}

FeatureVector ReID_tflite::extract(const cv::Mat& bgr_patch) {
  if (bgr_patch.empty())
    throw std::invalid_argument("Empty patch for Re‑ID extraction");

  // -------- preprocessing ----------
  cv::Mat resized, rgb_f32;
  cv::resize(bgr_patch, resized, {in_w_, in_h_});

  if (prep_params_.swap_rb) {
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
  }

  resized.convertTo(rgb_f32, CV_32F, 1.0f / 255.0f);

  // Copy to input tensor (NHWC)
  float* in_data = interpreter_->typed_tensor<float>(input_idx_);
  std::memcpy(in_data, rgb_f32.data,
              in_h_ * in_w_ * 3 * sizeof(float));

  // -------- inference --------------
  TfLiteCheck(interpreter_->Invoke(), "Invoke");

  const float* out_data =
      interpreter_->typed_output_tensor<float>(0);

  FeatureVector feat(1, out_dim_);
  for (int i = 0; i < out_dim_; ++i) feat(0, i) = out_data[i];

  // -------- L2 normalise (optional) --------
  if (prep_params_.l2_normalise) {
    float norm = std::sqrt(feat.squaredNorm());
    if (norm > 1e-6f) feat /= norm;
  }
  return feat;
}
