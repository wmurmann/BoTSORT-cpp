// ReID_tflite_osnet.h
#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include "DataType.h"                     // FeatureVector alias (1×D Eigen row)
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

class ReIDModel {
public:
    explicit ReIDModel(const std::string& tflite_path) {
        model_ = tflite::FlatBufferModel::BuildFromFile(tflite_path.c_str());
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);
        interpreter_->SetNumThreads(1);
        interpreter_->AllocateTensors();

        // read dynamic input dims
        auto* in_t = interpreter_->tensor(interpreter_->inputs()[0]);
        if (in_t->dims->size != 4 || in_t->type != kTfLiteFloat32)
            throw std::runtime_error("Unexpected input tensor");
        batch_ = in_t->dims->data[0];      // usually 1
        in_h_  = in_t->dims->data[1];
        in_w_  = in_t->dims->data[2];
        in_c_  = in_t->dims->data[3];

        // read dynamic output dim
        auto* out_t = interpreter_->tensor(interpreter_->outputs()[0]);
        if (out_t->dims->size != 2 || out_t->type != kTfLiteFloat32)
            throw std::runtime_error("Unexpected output tensor");
        out_dim_ = out_t->dims->data[1];
    }

    /** crop must be CV_8UC3 (BGR) */
    FeatureVector extract(const cv::Mat& bgr) {
        if (bgr.empty()) throw std::invalid_argument("Empty input");
        cv::Mat tmp, rgb;
        cv::resize(bgr, tmp, {in_w_, in_h_});
        cv::cvtColor(tmp, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1/255.0f);

        float* in_data = interpreter_->typed_tensor<float>(interpreter_->inputs()[0]);
        std::memcpy(in_data, rgb.data, batch_*in_h_*in_w_*in_c_*sizeof(float));

        if (interpreter_->Invoke() != kTfLiteOk)
            throw std::runtime_error("TFLite Invoke failed");

        const float* out_data = interpreter_->typed_output_tensor<float>(0);
        FeatureVector f(1, out_dim_);
        std::memcpy(f.data(), out_data, out_dim_*sizeof(float));
        f.normalize();
        return f;
    }

private:
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter>      interpreter_;
    int batch_, in_h_, in_w_, in_c_, out_dim_;
};