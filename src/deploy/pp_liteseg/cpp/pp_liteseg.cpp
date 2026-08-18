/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pp_liteseg.h"

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <onnxruntime_cxx_api.h>  // NOLINT(build/include_order)
#include <yaml-cpp/yaml.h>  // NOLINT(build/include_order)

#include "vision_model_config.h"
#include "vision_model_factory.h"
#include "operators/image_preprocess/cpu_image_preprocessor.h"
#include "operators/image_preprocess/image_preprocess_geometry.h"

namespace vision_deploy {

namespace {

int positive_dim(int64_t d, int fallback) {
    if (d > 0) {
        return static_cast<int>(d);
    }
    return fallback;
}

vision_operators::ImagePreprocessSpec make_pp_liteseg_preprocess_spec(
    int input_width,
    int input_height,
    float mean,
    float standard_deviation)
{
    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = input_width;
    spec.output_height = input_height;
    spec.resize_mode =
        vision_operators::PreprocessResizeMode::kFitTopLeft;
    spec.output_rgb = true;
    spec.mean = {mean * 255.0F, mean * 255.0F, mean * 255.0F};
    spec.scale = {
        1.0F / (255.0F * standard_deviation),
        1.0F / (255.0F * standard_deviation),
        1.0F / (255.0F * standard_deviation)};
    return spec;
}

vision_operators::CpuChannelTransform make_pp_liteseg_cpu_transform(
    float mean,
    float standard_deviation)
{
    vision_operators::CpuChannelTransform transform;
    transform.input_scale = {
        1.0F / 255.0F,
        1.0F / 255.0F,
        1.0F / 255.0F};
    transform.mean = {mean, mean, mean};
    transform.output_scale = {
        1.0F / standard_deviation,
        1.0F / standard_deviation,
        1.0F / standard_deviation};
    return transform;
}

}  // namespace

std::unique_ptr<vision_core::BaseModel> PPLiteSeg::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for PPLiteSeg");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for PPLiteSeg");
    }

    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    int num_classes = vision_core::yaml_utils::getInt(default_params, "num_classes", 19);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<PPLiteSeg>(model_path, num_threads, num_classes, lazy_load, provider);
}

PPLiteSeg::PPLiteSeg(const std::string& model_path,
                        int num_threads,
                        int num_classes,
                        bool lazy_load,
                        const std::string& provider)
    : BaseModel(model_path, lazy_load),
        num_threads_(num_threads),
        num_classes_(num_classes),
        provider_(provider),
        mean_val_(0.5f),
        std_val_(0.5f) {
    if (!lazy_load) {
        load_model();
    }
}

void PPLiteSeg::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);

    if (output_num_ >= 1) {
        Ort::TypeInfo out_info = session_->GetOutputTypeInfo(0);
        auto out_shape = out_info.GetTensorTypeAndShapeInfo().GetShape();
        if (out_shape.size() == 4 && out_shape[1] > 0) {
            num_classes_ = static_cast<int>(out_shape[1]);
        }
    }

    model_loaded_ = true;
}

cv::Mat PPLiteSeg::preprocess(const cv::Mat& image, int& valid_h, int& valid_w) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    ensure_model_loaded();

    int in_h = positive_dim(input_shape_[2], 512);
    int in_w = positive_dim(input_shape_[3], 1024);
    if (in_h <= 0 || in_w <= 0) {
        throw std::runtime_error("PPLiteSeg: invalid input spatial dims in ONNX (expected positive H,W)");
    }

    // Compute scale and valid region on original uint8 image
    const int h = image.rows;
    const int w = image.cols;
    const vision_operators::FitResizeDimensions dimensions =
        vision_operators::calculate_fit_resize_dimensions(
            static_cast<float>(w), static_cast<float>(h),
            in_w, in_h);
    const int new_h = dimensions.height;
    const int new_w = dimensions.width;
    valid_h = new_h;
    valid_w = new_w;

    return vision_operators::preprocess_bgr_to_nchw(
        image,
        make_pp_liteseg_preprocess_spec(
            in_w, in_h, mean_val_, std_val_),
        make_pp_liteseg_cpu_transform(
            mean_val_, std_val_));
}

cv::Mat PPLiteSeg::postprocess_to_label_map(std::vector<Ort::Value>& outputs,
                                                int origin_h,
                                                int origin_w,
                                                int valid_h,
                                                int valid_w) {
    if (outputs.empty()) {
        throw std::runtime_error("PPLiteSeg: empty model output");
    }

    Ort::Value& out0 = outputs[0];
    auto info = out0.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = info.GetShape();
    ONNXTensorElementDataType et = info.GetElementType();
    cv::Mat pred_small;
    if (et != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        throw std::runtime_error("PPLiteSeg: expected int32 output tensor");
    }

    const int32_t* data = out0.GetTensorData<int32_t>();

    int H = 0;
    int W = 0;
    if (shape.size() == 4) {
        // Python: raw_output = raw_output[0], then argmax(axis=0) for 3D logits.
        const int N = static_cast<int>(shape[0]);
        const int C = static_cast<int>(shape[1]);
        H = static_cast<int>(shape[2]);
        W = static_cast<int>(shape[3]);
        if (N != 1 || C <= 0 || H <= 0 || W <= 0) {
            throw std::runtime_error("PPLiteSeg: invalid int32 output shape [N,C,H,W]");
        }
        pred_small.create(H, W, CV_8U);
        const size_t hw = static_cast<size_t>(H) * static_cast<size_t>(W);
        for (int y = 0; y < H; ++y) {
            uint8_t* row = pred_small.ptr<uint8_t>(y);
            for (int x = 0; x < W; ++x) {
                const size_t base = static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x);
                int best_idx = 0;
                int32_t best_val = data[base];
                for (int c = 1; c < C; ++c) {
                    const int32_t v = data[static_cast<size_t>(c) * hw + base];
                    if (v > best_val) {
                        best_val = v;
                        best_idx = c;
                    }
                }
                row[x] = static_cast<uint8_t>(best_idx);
            }
        }
    } else if (shape.size() == 3) {
        const int d0 = static_cast<int>(shape[0]);
        const int d1 = static_cast<int>(shape[1]);
        const int d2 = static_cast<int>(shape[2]);
        if (d0 == 1) {
            // Python: if ndim==3 and shape[0]==1 => squeeze to [H,W], no argmax.
            H = d1;
            W = d2;
            if (H <= 0 || W <= 0) {
                throw std::runtime_error("PPLiteSeg: invalid int32 output shape [1,H,W]");
            }
            pred_small.create(H, W, CV_8U);
            for (int y = 0; y < H; ++y) {
                uint8_t* row = pred_small.ptr<uint8_t>(y);
                for (int x = 0; x < W; ++x) {
                    const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x);
                    row[x] = static_cast<uint8_t>(data[idx]);
                }
            }
        } else {
            // 3D logits [C,H,W] -> argmax over C.
            const int C = d0;
            H = d1;
            W = d2;
            if (C <= 0 || H <= 0 || W <= 0) {
                throw std::runtime_error("PPLiteSeg: invalid int32 output shape [C,H,W]");
            }
            pred_small.create(H, W, CV_8U);
            const size_t hw = static_cast<size_t>(H) * static_cast<size_t>(W);
            for (int y = 0; y < H; ++y) {
                uint8_t* row = pred_small.ptr<uint8_t>(y);
                for (int x = 0; x < W; ++x) {
                    const size_t base = static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x);
                    int best_idx = 0;
                    int32_t best_val = data[base];
                    for (int c = 1; c < C; ++c) {
                        const int32_t v = data[static_cast<size_t>(c) * hw + base];
                        if (v > best_val) {
                            best_val = v;
                            best_idx = c;
                        }
                    }
                    row[x] = static_cast<uint8_t>(best_idx);
                }
            }
        }
    } else if (shape.size() == 2) {
        H = static_cast<int>(shape[0]);
        W = static_cast<int>(shape[1]);
        if (H <= 0 || W <= 0) {
            throw std::runtime_error("PPLiteSeg: invalid int32 output shape [H,W]");
        }
        pred_small.create(H, W, CV_8U);
        for (int y = 0; y < H; ++y) {
            uint8_t* row = pred_small.ptr<uint8_t>(y);
            for (int x = 0; x < W; ++x) {
                const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x);
                row[x] = static_cast<uint8_t>(data[idx]);
            }
        }
    } else {
        throw std::runtime_error("PPLiteSeg: unsupported int32 output rank");
    }

    cv::Mat cropped = pred_small(cv::Rect(0, 0, valid_w, valid_h)).clone();
    cv::Mat pred_origin;
    cv::resize(cropped, pred_origin, cv::Size(origin_w, origin_h), 0, 0, cv::INTER_NEAREST);
    return pred_origin;
}

vision_common::SegmentationResultList PPLiteSeg::split_semantic_masks(const cv::Mat& label_u8) {
    vision_common::SegmentationResultList out;
    if (label_u8.empty() || label_u8.type() != CV_8U) {
        return out;
    }

    for (int cid = 1; cid < num_classes_; ++cid) {
        cv::Mat bin;
        cv::compare(label_u8, cid, bin, cv::CMP_EQ);
        if (cv::countNonZero(bin) == 0) {
            continue;
        }
        cv::Mat m255;
        bin.convertTo(m255, CV_8U, 255.0);

        vision_common::SegmentationResult r;
        r.bbox = vision_common::BoundingBox{0, 0, static_cast<float>(bin.cols), static_cast<float>(bin.rows)};
        r.label = cid;
        r.score = 1.0f;
        r.mask = std::make_shared<cv::Mat>(m255);
        out.push_back(std::move(r));
    }
    return out;
}

vision_common::SegmentationResultList PPLiteSeg::segment(
    const cv::Mat& image,
    float /*conf_threshold*/,
    float /*iou_threshold*/) {
    vision_core::ImageInput input;
    input.image = image;
    return segment_input(input);
}

vision_common::SegmentationResultList PPLiteSeg::segment_input(
    const vision_core::ImageInput& input) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    const int origin_h =
        input.format == vision_core::ImagePixelFormat::kNv12
            ? input.image.rows * 2 / 3
            : input.image.rows;
    const int origin_w = input.image.cols;

    const int in_h = positive_dim(input_shape_[2], 512);
    const int in_w = positive_dim(input_shape_[3], 1024);
    const vision_operators::FitResizeDimensions valid_dimensions =
        vision_operators::calculate_fit_resize_dimensions(
            static_cast<float>(origin_w),
            static_cast<float>(origin_h),
            in_w, in_h);
    int valid_h = valid_dimensions.height;
    int valid_w = valid_dimensions.width;
    const auto t_pre0 = std::chrono::steady_clock::now();
    const vision_operators::ImagePreprocessSpec spec =
        make_pp_liteseg_preprocess_spec(
            in_w, in_h, mean_val_, std_val_);
    auto prepared = prepare_image(
        input, spec,
        [this, &valid_h, &valid_w](const cv::Mat& bgr) {
            return preprocess(bgr, valid_h, valid_w);
        });
    const auto t_pre1 = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

    const auto t_infer0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs =
        run_session(prepared.tensor());
    const auto t_infer1 = std::chrono::steady_clock::now();
    set_runtime_model_infer_ms(std::chrono::duration<double, std::milli>(t_infer1 - t_infer0).count());

    const auto t_post0 = std::chrono::steady_clock::now();
    cv::Mat label_map = postprocess_to_label_map(outputs, origin_h, origin_w, valid_h, valid_w);
    vision_common::SegmentationResultList results = split_semantic_masks(label_map);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());

    return results;
}


std::vector<vision_core::InferIntent> PPLiteSeg::supported_intents() const {
    return {vision_core::InferIntent::kSegment};
}

vision_core::InferResponse PPLiteSeg::Run(const vision_core::InferRequest& request) {
    if (request.intent != vision_core::InferIntent::kSegment) {
        return unsupported_intent_response(request.intent);
    }
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "PPLiteSeg expects ImageInput";
        return response;
    }

    vision_common::SegmentationResultList task_results =
        segment_input(*image_input);
    vision_core::InferResponse response;
    response.results.reserve(task_results.size());
    for (auto& item : task_results) {
        response.results.emplace_back(std::move(item));
    }
    return response;
}

std::vector<vision_core::ModelCapability> PPLiteSeg::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

static vision_core::ModelRegistrar<PPLiteSeg> registrar("PPLiteSeg");

}  // namespace vision_deploy
