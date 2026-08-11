/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "superpoint_extractor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "operators/image_preprocess/cpu_image_preprocessor.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

namespace {

float bilinear_descriptor(
    const float* descriptor_map,
    int channel,
    int descriptor_height,
    int descriptor_width,
    float x,
    float y) {
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int x0c = std::clamp(x0, 0, descriptor_width - 1);
    const int x1c = std::clamp(x1, 0, descriptor_width - 1);
    const int y0c = std::clamp(y0, 0, descriptor_height - 1);
    const int y1c = std::clamp(y1, 0, descriptor_height - 1);
    const float wx = x - x0;
    const float wy = y - y0;
    const size_t plane =
        static_cast<size_t>(descriptor_height) * descriptor_width;
    const float* values =
        descriptor_map + static_cast<size_t>(channel) * plane;
    return values[y0c * descriptor_width + x0c] *
            (1.0f - wx) * (1.0f - wy) +
        values[y0c * descriptor_width + x1c] *
            wx * (1.0f - wy) +
        values[y1c * descriptor_width + x0c] *
            (1.0f - wx) * wy +
        values[y1c * descriptor_width + x1c] *
            wx * wy;
}

}  // namespace

vision::LocalFeatures build_superpoint_features(
    const float* scores,
    const float* descriptor_map,
    int image_height,
    int image_width,
    int descriptor_channels,
    int descriptor_height,
    int descriptor_width,
    int num_keypoints,
    int nms_radius,
    int remove_borders,
    int original_width,
    int original_height,
    const std::string& feature_type) {
    if (scores == nullptr || descriptor_map == nullptr ||
        image_height <= 0 || image_width <= 0 ||
        descriptor_channels <= 0 ||
        descriptor_height <= 0 || descriptor_width <= 0 ||
        num_keypoints <= 0 ||
        nms_radius < 0 || remove_borders < 0 ||
        original_width <= 0 || original_height <= 0) {
        throw std::invalid_argument(
            "SuperPoint postprocess dimensions must be valid");
    }

    std::vector<int> order;
    order.reserve(static_cast<size_t>(image_height) * image_width);
    for (int y = remove_borders;
        y < image_height - remove_borders;
        ++y) {
        for (int x = remove_borders;
            x < image_width - remove_borders;
            ++x) {
            const int index = y * image_width + x;
            if (std::isfinite(scores[index]) && scores[index] > 0.0f) {
                order.push_back(index);
            }
        }
    }
    std::sort(
        order.begin(),
        order.end(),
        [scores](int lhs, int rhs) {
            return scores[lhs] > scores[rhs];
        });

    std::vector<uint8_t> suppressed(
        static_cast<size_t>(image_height) * image_width,
        0);
    std::vector<int> selected;
    selected.reserve(num_keypoints);
    for (const int index : order) {
        if (suppressed[index] != 0) {
            continue;
        }
        selected.push_back(index);
        const int x = index % image_width;
        const int y = index / image_width;
        for (int yy = std::max(0, y - nms_radius);
            yy <= std::min(image_height - 1, y + nms_radius);
            ++yy) {
            for (int xx = std::max(0, x - nms_radius);
                xx <= std::min(image_width - 1, x + nms_radius);
                ++xx) {
                suppressed[yy * image_width + xx] = 1;
            }
        }
        if (static_cast<int>(selected.size()) == num_keypoints) {
            break;
        }
    }

    vision::LocalFeatures output;
    output.keypoints.resize(num_keypoints);
    output.descriptors.assign(
        static_cast<size_t>(num_keypoints) * descriptor_channels,
        0.0f);
    output.descriptor_dim = descriptor_channels;
    output.image_width = original_width;
    output.image_height = original_height;
    output.feature_type = feature_type;

    const float coordinate_scale_x =
        static_cast<float>(original_width) / image_width;
    const float coordinate_scale_y =
        static_cast<float>(original_height) / image_height;
    const float descriptor_scale_x =
        static_cast<float>(descriptor_width - 1) / image_width;
    const float descriptor_scale_y =
        static_cast<float>(descriptor_height - 1) / image_height;

    for (size_t i = 0; i < selected.size(); ++i) {
        const int index = selected[i];
        const int model_x = index % image_width;
        const int model_y = index / image_width;
        output.keypoints[i] = {
            model_x * coordinate_scale_x,
            model_y * coordinate_scale_y,
            scores[index]};
        float norm_squared = 0.0f;
        for (int channel = 0;
            channel < descriptor_channels;
            ++channel) {
            const float value = bilinear_descriptor(
                descriptor_map,
                channel,
                descriptor_height,
                descriptor_width,
                model_x * descriptor_scale_x,
                model_y * descriptor_scale_y);
            output.descriptors[
                i * descriptor_channels + channel] = value;
            norm_squared += value * value;
        }
        const float norm = std::sqrt(norm_squared);
        if (norm > 1e-12f) {
            for (int channel = 0;
                channel < descriptor_channels;
                ++channel) {
                output.descriptors[
                    i * descriptor_channels + channel] /= norm;
            }
        }
    }
    return output;
}

SuperPointExtractor::SuperPointExtractor(
    const std::string& model_path,
    int num_keypoints,
    int nms_radius,
    int remove_borders,
    std::string feature_type,
    int num_threads,
    bool lazy_load,
    std::string provider)
    : BaseModel(model_path, lazy_load),
        num_keypoints_(num_keypoints),
        nms_radius_(nms_radius),
        remove_borders_(remove_borders),
        feature_type_(std::move(feature_type)),
        num_threads_(num_threads),
        provider_(std::move(provider)) {
    if (!lazy_load) {
        load_model();
    }
}

std::unique_ptr<vision_core::BaseModel> SuperPointExtractor::create(
    const YAML::Node& config,
    bool lazy_load) {
    const std::string model_path =
        vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error(
            "model_path not found in config for SuperPointExtractor");
    }
    const YAML::Node params = config["default_params"];
    return std::make_unique<SuperPointExtractor>(
        model_path,
        vision_core::yaml_utils::getInt(params, "num_keypoints", 512),
        vision_core::yaml_utils::getInt(params, "nms_radius", 4),
        vision_core::yaml_utils::getInt(params, "remove_borders", 4),
        vision_core::yaml_utils::getString(
            params, "feature_type", "superpoint"),
        vision_core::yaml_utils::getInt(params, "num_threads", 4),
        lazy_load,
        vision_core::yaml_utils::getProvider(config));
}

void SuperPointExtractor::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    if (input_shape_.size() != 4 ||
        input_shape_[0] != 1 ||
        input_shape_[1] != 1 ||
        input_shape_[2] <= 0 ||
        input_shape_[3] <= 0 ||
        output_num_ != 2) {
        throw std::runtime_error(
            "SuperPointExtractor expects [1,1,H,W] input and two outputs");
    }
    model_loaded_ = true;
}

vision::LocalFeatures SuperPointExtractor::extract(
    const vision_core::ImageInput& input) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto total_begin = std::chrono::steady_clock::now();
    const auto preprocess_begin = std::chrono::steady_clock::now();
    const cv::Mat bgr =
        vision_operators::image_input_to_bgr_cpu(input);
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::invalid_argument(
            "SuperPointExtractor expects BGR8 image data");
    }
    const int input_height = static_cast<int>(input_shape_[2]);
    const int input_width = static_cast<int>(input_shape_[3]);
    vision_operators::ImagePreprocessSpec preprocess_spec;
    preprocess_spec.output_width = input_width;
    preprocess_spec.output_height = input_height;
    vision_operators::CpuGrayscaleTransform grayscale_transform;
    grayscale_transform.input_scale = 1.0F / 255.0F;
    cv::Mat tensor = vision_operators::preprocess_bgr_to_gray_nchw(
        bgr, preprocess_spec, grayscale_transform);
    const auto preprocess_end = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(
        std::chrono::duration<double, std::milli>(
            preprocess_end - preprocess_begin).count());

    const auto infer_begin = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = run_session(tensor);
    const auto infer_end = std::chrono::steady_clock::now();
    const double infer_ms =
        std::chrono::duration<double, std::milli>(
            infer_end - infer_begin).count();
    set_runtime_model_infer_ms(infer_ms);
    add_runtime_component_timing("superpoint.infer", infer_ms);

    const auto postprocess_begin = std::chrono::steady_clock::now();
    if (outputs.size() != 2 ||
        !outputs[0].IsTensor() ||
        !outputs[1].IsTensor()) {
        throw std::runtime_error(
            "SuperPointExtractor returned invalid outputs");
    }
    const auto score_info =
        outputs[0].GetTensorTypeAndShapeInfo();
    const auto descriptor_info =
        outputs[1].GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> score_shape = score_info.GetShape();
    const std::vector<int64_t> descriptor_shape =
        descriptor_info.GetShape();
    if (score_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        descriptor_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        score_shape.size() != 3 ||
        score_shape[0] != 1 ||
        score_shape[1] != input_height ||
        score_shape[2] != input_width ||
        descriptor_shape.size() != 4 ||
        descriptor_shape[0] != 1 ||
        descriptor_shape[1] <= 0 ||
        descriptor_shape[2] <= 0 ||
        descriptor_shape[3] <= 0) {
        throw std::runtime_error(
            "SuperPointExtractor output shapes are incompatible");
    }
    vision::LocalFeatures result = build_superpoint_features(
        outputs[0].GetTensorData<float>(),
        outputs[1].GetTensorData<float>(),
        input_height,
        input_width,
        static_cast<int>(descriptor_shape[1]),
        static_cast<int>(descriptor_shape[2]),
        static_cast<int>(descriptor_shape[3]),
        num_keypoints_,
        nms_radius_,
        remove_borders_,
        bgr.cols,
        bgr.rows,
        feature_type_);
    const auto postprocess_end = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(
        std::chrono::duration<double, std::milli>(
            postprocess_end - postprocess_begin).count());
    set_runtime_total_ms(
        std::chrono::duration<double, std::milli>(
            postprocess_end - total_begin).count());
    return result;
}

vision_core::InferResponse SuperPointExtractor::Run(
    const vision_core::InferRequest& request) {
    vision_core::InferResponse response;
    if (request.intent !=
        vision_core::InferIntent::kExtractLocalFeatures) {
        response.ok = false;
        response.error_message =
            "SuperPointExtractor only supports kExtractLocalFeatures";
        return response;
    }
    const auto* image =
        std::get_if<vision_core::ImageInput>(&request.input);
    if (image == nullptr) {
        response.ok = false;
        response.error_message =
            "SuperPointExtractor expects ImageInput";
        return response;
    }
    response.results.emplace_back(extract(*image));
    return response;
}

std::vector<vision_core::InferIntent>
SuperPointExtractor::supported_intents() const {
    return {vision_core::InferIntent::kExtractLocalFeatures};
}

std::vector<vision_core::ModelCapability>
SuperPointExtractor::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

static vision_core::ModelRegistrar<SuperPointExtractor> registrar(
    "SuperPointExtractor");

}  // namespace vision_deploy
