/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "core/cpp/vision_model_base.h"
#include "core/cpp/vision_model_config.h"
#include "core/cpp/vision_model_factory.h"
#include "operators/image_preprocess/cpu_image_preprocessor.h"

namespace vision_deploy {

class MobileSAMSegmentor final : public vision_core::BaseModel {
public:
    MobileSAMSegmentor(const YAML::Node& config, bool lazy)
        : BaseModel(config["model_path"].as<std::string>(), lazy) {
        const auto params = config["default_params"];
        decoder_path_ = params["decoder_model_path"].as<std::string>();
        threads_ = vision_core::yaml_utils::getInt(params, "num_threads", 4);
        provider_ = vision_core::yaml_utils::getProvider(config);
        if (threads_ <= 0 || decoder_path_.empty()) {
            throw std::invalid_argument("MobileSAM invalid configuration");
        }
        if (!lazy) load_model();
    }

    static std::unique_ptr<vision_core::BaseModel> create(
        const YAML::Node& config, bool lazy) {
        return std::make_unique<MobileSAMSegmentor>(config, lazy);
    }

    void load_model() override {
        if (model_loaded_) return;
        init_session(threads_, provider_);
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(threads_);
        if (provider_ == "SpaceMITExecutionProvider") {
            Ort::Status status = Ort::SessionOptionsSpaceMITEnvInit(options);
            if (!status.IsOK())
                throw std::runtime_error(status.GetErrorMessage());
        }
        decoder_ = std::make_unique<Ort::Session>(
            vision_core::shared_ort_env(), decoder_path_.c_str(), options);
        if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1 ||
            decoder_->GetInputCount() != 5 || decoder_->GetOutputCount() != 2) {
            throw std::runtime_error("MobileSAM tensor counts mismatch");
        }
        // Encoder names vary between exports. BaseModel reads the actual
        // names from the session; only type and shape are part of its contract.
        check(*session_, true, 0, "", {1, 3, 448, 448});
        check(*session_, false, 0, "", {1, 256, 28, 28});
        check(*decoder_, true, 0, "image_embeddings", {1, 256, 28, 28});
        check(*decoder_, true, 1, "point_coords", {1, -1, 2});
        check(*decoder_, true, 2, "point_labels", {1, -1});
        check(*decoder_, true, 3, "mask_input", {1, 1, 112, 112});
        check(*decoder_, true, 4, "has_mask_input", {1});
        model_loaded_ = true;
    }

    void release() override {
        decoder_.reset();
        BaseModel::release();
    }

    std::vector<vision_core::InferIntent> supported_intents() const override {
        return {vision_core::InferIntent::kSegment};
    }

    std::vector<vision_core::ModelCapability> get_capabilities()
        const override {
        return {vision_core::ModelCapability::kDraw};
    }

    vision_core::InferResponse Run(
        const vision_core::InferRequest& request) override {
        if (request.intent != vision_core::InferIntent::kSegment) {
            return unsupported_intent_response(request.intent);
        }
        const auto* input =
            std::get_if<vision_core::ImageInput>(&request.input);
        if (!input || input->image.empty() ||
            input->format != vision_core::ImagePixelFormat::kBgr8 ||
            input->image.type() != CV_8UC3) {
            throw std::invalid_argument("MobileSAM requires a BGR8 image");
        }
        const auto& points = request.params.point_coords;
        const auto& labels = request.params.point_labels;
        if (points.empty() || points.size() != labels.size()) {
            throw std::invalid_argument(
                "MobileSAM requires matching points and labels");
        }
        const cv::Mat& image = input->image;
        const float scale = 448.0F / std::max(image.rows, image.cols);
        const int width =
            std::max(1, static_cast<int>(image.cols * scale + 0.5F));
        const int height =
            std::max(1, static_cast<int>(image.rows * scale + 0.5F));
        std::vector<float> coords;
        std::vector<float> point_labels;
        int corners = 0;
        for (size_t i = 0; i < points.size(); ++i) {
            if (!std::isfinite(points[i].x) || !std::isfinite(points[i].y) ||
                labels[i] < -1 || labels[i] > 3) {
                throw std::invalid_argument("MobileSAM invalid point or label");
            }
            if (labels[i] == 2) {
                if (i + 1 >= labels.size() || labels[i + 1] != 3 ||
                    points[i].x >= points[i + 1].x ||
                    points[i].y >= points[i + 1].y) {
                    throw std::invalid_argument(
                        "MobileSAM invalid box corners");
                }
                ++corners;
            }
            if (labels[i] == 3 && (i == 0 || labels[i - 1] != 2)) {
                throw std::invalid_argument(
                    "MobileSAM missing top-left box corner");
            }
            coords.push_back(points[i].x * width / image.cols);
            coords.push_back(points[i].y * height / image.rows);
            point_labels.push_back(static_cast<float>(labels[i]));
        }
        if (corners == 0 &&
            std::find(labels.begin(), labels.end(), -1) == labels.end()) {
            coords.insert(coords.end(), {0.0F, 0.0F});
            point_labels.push_back(-1.0F);
        }
        ensure_model_loaded();
        reset_runtime_profile();
        const auto started = std::chrono::steady_clock::now();
        // Reuse the exact dimensions used for prompt mapping above, including
        // its rounding. The shared packer only pads and packs this resized ROI.
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
        vision_operators::ImagePreprocessSpec spec;
        spec.output_width = 448;
        spec.output_height = 448;
        spec.resize_mode = vision_operators::PreprocessResizeMode::kFitTopLeft;
        spec.mean = {123.675F, 116.28F, 103.53F};
        spec.scale = {1.0F / 58.395F, 1.0F / 57.12F, 1.0F / 57.375F};
        cv::Mat blob = vision_operators::preprocess_bgr_to_nchw(resized, spec);
        const auto preprocessed = std::chrono::steady_clock::now();
        auto embedding = run_session(blob);
        auto memory =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> tensors;
        tensors.push_back(std::move(embedding[0]));
        const int64_t count = static_cast<int64_t>(point_labels.size());
        const std::vector<int64_t> coord_shape = {1, count, 2};
        const std::vector<int64_t> label_shape = {1, count};
        const std::vector<int64_t> mask_shape = {1, 1, 112, 112};
        const std::vector<int64_t> flag_shape = {1};
        std::vector<float> mask(112 * 112, 0.0F);
        float has_mask = 0.0F;
        const auto append = [&](float* data, size_t size,
                                const std::vector<int64_t>& shape) {
            tensors.push_back(Ort::Value::CreateTensor<float>(
                memory, data, size, shape.data(), shape.size()));
        };
        append(coords.data(), coords.size(), coord_shape);
        append(point_labels.data(), point_labels.size(), label_shape);
        append(mask.data(), mask.size(), mask_shape);
        append(&has_mask, 1, flag_shape);
        const char* names[] = {"image_embeddings", "point_coords",
            "point_labels", "mask_input", "has_mask_input"};
        const char* outputs[] = {"iou_predictions", "low_res_masks"};
        auto decoded = decoder_->Run(Ort::RunOptions{nullptr}, names,
            tensors.data(), 5, outputs, 2);
        const auto inferred = std::chrono::steady_clock::now();
        const auto scores_info = decoded[0].GetTensorTypeAndShapeInfo();
        const auto masks_info = decoded[1].GetTensorTypeAndShapeInfo();
        if (scores_info.GetElementType() !=
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            masks_info.GetElementType() !=
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            scores_info.GetShape() != std::vector<int64_t>{1, 4} ||
            masks_info.GetShape() != std::vector<int64_t>{1, 4, 112, 112}) {
            throw std::runtime_error(
                "MobileSAM decoder output contract mismatch");
        }
        const float* scores = decoded[0].GetTensorData<float>();
        for (int i = 0; i < 4; ++i) {
            if (!std::isfinite(scores[i])) {
                throw std::runtime_error("MobileSAM returned non-finite score");
            }
        }
        const size_t best = std::max_element(scores, scores + 4) - scores;
        cv::Mat low(112, 112, CV_32F,
            const_cast<float*>(decoded[1].GetTensorData<float>() +
                best * 112 * 112));
        cv::Mat logits;
        cv::resize(low, logits, cv::Size(448, 448), 0, 0, cv::INTER_LINEAR);
        cv::Mat restored;
        cv::resize(logits(cv::Rect(0, 0, width, height)), restored,
            image.size(), 0, 0, cv::INTER_LINEAR);
        vision::Segmentation result;
        result.bbox = {-1.0F, -1.0F, -1.0F, -1.0F};
        result.label = 0;
        result.score = scores[best];
        result.mask = std::make_shared<cv::Mat>();
        cv::compare(restored, 0.0, *result.mask, cv::CMP_GT);
        vision_core::InferResponse response;
        response.results.emplace_back(std::move(result));
        const auto finished = std::chrono::steady_clock::now();
        const auto ms = [](auto begin, auto end) {
            return std::chrono::duration<double, std::milli>(
                end - begin).count();
        };
        set_runtime_preprocess_ms(ms(started, preprocessed));
        set_runtime_model_infer_ms(ms(preprocessed, inferred));
        set_runtime_postprocess_ms(ms(inferred, finished));
        set_runtime_total_ms(ms(started, finished));
        return response;
    }

private:
    void check(Ort::Session& session, bool input, size_t index,
        const std::string& name, const std::vector<int64_t>& shape) {
        auto actual_name =
            input ? session.GetInputNameAllocated(index, allocator_)
                : session.GetOutputNameAllocated(index, allocator_);
        auto type = input ? session.GetInputTypeInfo(index)
            : session.GetOutputTypeInfo(index);
        auto info = type.GetTensorTypeAndShapeInfo();
        // An empty expected name means index/type/shape validation only.
        if ((!name.empty() && name != actual_name.get()) ||
            info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            info.GetShape() != shape) {
            throw std::runtime_error(
                std::string("MobileSAM tensor contract mismatch: ") +
                actual_name.get());
        }
    }

    std::string decoder_path_;
    std::string provider_;
    int threads_ = 4;
    std::unique_ptr<Ort::Session> decoder_;
};

static vision_core::ModelRegistrar<MobileSAMSegmentor> registrar(
    "MobileSAMSegmentor");

}  // namespace vision_deploy
