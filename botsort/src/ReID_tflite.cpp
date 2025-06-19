#include "ReID_tflite.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {

// helper: throw on non-OK
inline void TfLiteCheck(TfLiteStatus s, const char* where) {
  if (s != kTfLiteOk)
    throw std::runtime_error(std::string("TFLite error at ") + where);
}

} // namespace

/* ---------------- constructor ---------------- */

ReIDModel::ReIDModel(const std::string& model_path,
                     const ReIDPreprocParams& pp)
    : prep_params_(pp) {
  model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  if (!model_) throw std::runtime_error("Cannot mmap TFLite model: " + model_path);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);
  if (!interpreter_) throw std::runtime_error("Interpreter build failed");

  interpreter_->SetNumThreads(1);
  TfLiteCheck(interpreter_->AllocateTensors(), "AllocateTensors");

  input_idx_  = interpreter_->inputs()[0];
  output_idx_ = interpreter_->outputs()[0];

  const TfLiteTensor* in_t = interpreter_->tensor(input_idx_);
  if (in_t->dims->size != 4 || in_t->type != kTfLiteFloat32)
    throw std::runtime_error("ReID input tensor shape/type mismatch");
  in_h_ = in_t->dims->data[1];
  in_w_ = in_t->dims->data[2];

  const TfLiteTensor* out_t = interpreter_->tensor(output_idx_);
  if (out_t->dims->size != 2 || out_t->type != kTfLiteFloat32)
    throw std::runtime_error("ReID output tensor shape/type mismatch");
  out_dim_ = out_t->dims->data[1];
}

/* ---------------- feature extraction ---------------- */

FeatureVector ReIDModel::extract(const cv::Mat& bgr_patch) {
  if (bgr_patch.empty())
    throw std::invalid_argument("ReID: empty crop passed in");

  cv::Mat resized, tmp32;
  cv::resize(bgr_patch, resized, {in_w_, in_h_});

  if (prep_params_.swap_rb)
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

  resized.convertTo(tmp32, CV_32F, 1.f / 255.f);

  float* in_data = interpreter_->typed_tensor<float>(input_idx_);
  std::memcpy(in_data, tmp32.data, in_h_ * in_w_ * 3 * sizeof(float));

  TfLiteCheck(interpreter_->Invoke(), "Invoke");

  const float* out_data = interpreter_->typed_output_tensor<float>(0);
  FeatureVector vec(1, out_dim_);
  std::memcpy(vec.data(), out_data, out_dim_ * sizeof(float));

  if (prep_params_.l2_normalise) {
    float n = std::sqrt(vec.squaredNorm());
    if (n > 1e-6f) vec /= n;
  }
  return vec;
}
