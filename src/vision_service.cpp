/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vision_service.h"

#include <cstring>
#include <exception>
#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <new>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>  // NOLINT(build/include_order)
#include <yaml-cpp/yaml.h>  // NOLINT(build/include_order)

#include "common/cpp/datatype.h"
#include "common/cpp/drawing.h"
#include "common/cpp/embedding_utils.h"
#include "common/cpp/image_processing.h"
#include "core/cpp/vision_model_base.h"
#include "core/cpp/vision_model_factory.h"
#include "core/cpp/vision_task_interfaces.h"

struct VisionService::Impl {
    std::unique_ptr<vision_core::BaseModel> model;
    std::string config_path;
    std::vector<vision_common::Result> last_raw_results;
    std::vector<std::string> labels;
    std::string default_image_path;
    std::string last_config_path_value;
};

thread_local std::string VisionService::g_last_error_;

namespace {

std::vector<std::string> loadLabelsForConfig(const YAML::Node& config, const std::string& config_file) {
    try {
        if (!config["label_file_path"]) {
            return {};
        }
        std::string label_file = config["label_file_path"].as<std::string>();
        label_file = vision_core::resolveResourcePath(label_file, config_file);
        return vision_common::load_labels(label_file);
    } catch (...) {
        return {};
    }
}

void ConvertResults(const std::vector<vision_common::Result>& raw_results,
                    std::vector<VisionServiceResult>* out_results) {
    out_results->clear();
    out_results->reserve(raw_results.size());
    for (const auto& src : raw_results) {
        VisionServiceResult dst{};
        dst.x1 = src.x1;
        dst.y1 = src.y1;
        dst.x2 = src.x2;
        dst.y2 = src.y2;
        dst.score = src.score;
        dst.label = src.label;
        dst.track_id = src.track_id;
        out_results->push_back(dst);
    }
}

}  // namespace

VisionService::~VisionService() = default;

std::unique_ptr<VisionService> VisionService::Create(const std::string& config_path,
                                                       const std::string& model_path_override,
                                                       bool lazy_load) {
    g_last_error_.clear();
    if (config_path.empty()) {
        g_last_error_ = "config_path is empty";
        return nullptr;
    }

    std::unique_ptr<VisionService> service(new (std::nothrow) VisionService());
    if (!service) {
        g_last_error_ = "Failed to allocate VisionService";
        return nullptr;
    }
    service->impl_.reset(new (std::nothrow) Impl());
    if (!service->impl_) {
        g_last_error_ = "Failed to allocate VisionService::Impl";
        return nullptr;
    }

    try {
        std::filesystem::path config_file = std::filesystem::absolute(config_path);
        if (!std::filesystem::exists(config_file)) {
            g_last_error_ = "Config file not found: " + config_file.string();
            service->last_error_ = g_last_error_;
            return nullptr;
        }
        service->impl_->config_path = config_file.string();
        YAML::Node config = YAML::LoadFile(service->impl_->config_path);
        service->impl_->labels = loadLabelsForConfig(config, service->impl_->config_path);
        service->impl_->model = vision_core::createModelFromConfigPath(
            service->impl_->config_path, model_path_override, lazy_load, &config);
        service->last_error_.clear();
        g_last_error_.clear();
        return service;
    } catch (const std::exception& e) {
        g_last_error_ = e.what();
        service->last_error_ = e.what();
        return nullptr;
    } catch (...) {
        g_last_error_ = "Unknown error while creating model service";
        service->last_error_ = g_last_error_;
        return nullptr;
    }
}

const std::string& VisionService::LastCreateError() {
    return g_last_error_;
}

VisionServiceStatus VisionService::SetError(VisionServiceStatus code, const std::string& message) {
    g_last_error_ = message;
    last_error_ = message;
    return code;
}

const std::string& VisionService::LastError() const {
    if (!last_error_.empty()) {
        return last_error_;
    }
    return g_last_error_;
}

VisionServiceStatus VisionService::InferImage(const std::string& image_path,
                                               std::vector<VisionServiceResult>* out_results) {
    if (out_results == nullptr) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "out_results must not be null");
    }
    if (image_path.empty()) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "image_path must not be empty");
    }

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        return SetError(VISION_SERVICE_IO_FAILED, std::string("Failed to read image: ") + image_path);
    }

    return InferImage(image, out_results);
}

VisionServiceStatus VisionService::InferImage(const cv::Mat& image,
                                               std::vector<VisionServiceResult>* out_results) {
    if (out_results == nullptr) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "out_results must not be null");
    }
    out_results->clear();

    if (impl_ == nullptr || impl_->model == nullptr) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "service/model must not be null");
    }
    if (image.empty()) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "image must not be empty");
    }
    if (image.channels() != 3) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "image must be 3-channel BGR");
    }

    try {
        vision_core::BaseModel* model = impl_->model.get();
        if (model->supports_capability(vision_core::ModelCapability::kTrackUpdate)) {
            auto* tracking_model = dynamic_cast<vision_core::ITrackingModel*>(model);
            if (tracking_model != nullptr) {
                impl_->last_raw_results = tracking_model->track(image);
            } else {
                return SetError(VISION_SERVICE_INFER_FAILED, "Model advertises tracking but ITrackingModel is missing");
            }
        } else if (auto* pose_model = dynamic_cast<vision_core::IPoseModel*>(model);
                   pose_model != nullptr) {
            impl_->last_raw_results = pose_model->estimate_pose(image);
        } else if (auto* seg_model = dynamic_cast<vision_core::ISegmentationModel*>(model);
                   seg_model != nullptr) {
            impl_->last_raw_results = seg_model->segment(image);
        } else if (auto* det_model = dynamic_cast<vision_core::IDetectionModel*>(model);
                   det_model != nullptr) {
            impl_->last_raw_results = det_model->detect(image);
        } else if (auto* cls_model = dynamic_cast<vision_core::IClassificationModel*>(model);
                   cls_model != nullptr) {
            impl_->last_raw_results = cls_model->classify(image);
        } else {
            return SetError(VISION_SERVICE_INFER_FAILED, "Model does not provide a supported image task interface");
        }

        if (impl_->last_raw_results.empty()) {
            last_error_.clear();
            return VISION_SERVICE_OK;
        }
        ConvertResults(impl_->last_raw_results, out_results);
        last_error_.clear();
        return VISION_SERVICE_OK;
    } catch (const std::exception& e) {
        return SetError(VISION_SERVICE_INFER_FAILED, e.what());
    } catch (...) {
        return SetError(VISION_SERVICE_INFER_FAILED, "Unknown error during inference");
    }
}

VisionServiceStatus VisionService::InferEmbedding(const std::string& image_path,
                                                   std::vector<float>* out_embedding) {
    if (impl_ == nullptr || impl_->model == nullptr || out_embedding == nullptr) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "invalid argument");
    }
    out_embedding->clear();
    if (image_path.empty()) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "image_path must not be empty");
    }
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        return SetError(VISION_SERVICE_IO_FAILED, std::string("Failed to read image: ") + image_path);
    }
    return InferEmbedding(image, out_embedding);
}

VisionServiceStatus VisionService::InferEmbedding(const cv::Mat& image,
                                                   std::vector<float>* out_embedding) {
    if (impl_ == nullptr || impl_->model == nullptr || out_embedding == nullptr) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "invalid argument");
    }
    out_embedding->clear();
    if (image.empty()) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "image must not be empty");
    }
    if (image.channels() != 3) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "image must be 3-channel BGR");
    }
    try {
        vision_core::BaseModel* model = impl_->model.get();
        if (!model->supports_capability(vision_core::ModelCapability::kEmbedding)) {
            return SetError(VISION_SERVICE_INFER_FAILED, "current model does not support embedding");
        }
        auto* embedding_model = dynamic_cast<vision_core::IEmbeddingModel*>(model);
        if (embedding_model == nullptr) {
            return SetError(VISION_SERVICE_INFER_FAILED, "Model advertises embedding but IEmbeddingModel is missing");
        }
        *out_embedding = embedding_model->infer_embedding(image);
        if (out_embedding->empty()) {
            return SetError(VISION_SERVICE_INFER_FAILED, "Embedding output is empty");
        }
        last_error_.clear();
        return VISION_SERVICE_OK;
    } catch (const std::exception& e) {
        return SetError(VISION_SERVICE_INFER_FAILED, e.what());
    } catch (...) {
        return SetError(VISION_SERVICE_INFER_FAILED, "Unknown error during embedding inference");
    }
}

float VisionService::EmbeddingSimilarity(const std::vector<float>& embedding_a,
                                         const std::vector<float>& embedding_b) {
    if (embedding_a.empty() || embedding_b.empty() || embedding_a.size() != embedding_b.size()) {
        return 0.0f;
    }
    return vision_common::compute_similarity(embedding_a, embedding_b);
}

VisionServiceStatus VisionService::InferSequence(const float* pts, int image_width, int image_height,
                                                 std::vector<float>* out_scores) {
    if (impl_ == nullptr || impl_->model == nullptr || out_scores == nullptr) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "invalid argument");
    }
    out_scores->clear();
    if (pts == nullptr) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "pts must not be null");
    }
    try {
        vision_core::BaseModel* model = impl_->model.get();
        if (!model->supports_capability(vision_core::ModelCapability::kSequenceInput)) {
            return SetError(VISION_SERVICE_INFER_FAILED, "current model does not support sequence inference");
        }
        auto* seq_model = dynamic_cast<vision_core::ISequenceActionModel*>(model);
        if (seq_model == nullptr) {
            return SetError(VISION_SERVICE_INFER_FAILED,
                "Model advertises sequence but ISequenceActionModel is missing");
        }
        *out_scores = seq_model->infer_sequence(pts, image_width, image_height);
        last_error_.clear();
        return VISION_SERVICE_OK;
    } catch (const std::exception& e) {
        return SetError(VISION_SERVICE_INFER_FAILED, e.what());
    } catch (...) {
        return SetError(VISION_SERVICE_INFER_FAILED, "Unknown error during sequence inference");
    }
}

std::vector<std::string> VisionService::GetSequenceClassNames() {
    if (impl_ == nullptr || impl_->model == nullptr) {
        return {};
    }
    auto* seq_model = dynamic_cast<vision_core::ISequenceActionModel*>(impl_->model.get());
    if (seq_model == nullptr) return {};
    return seq_model->get_class_names();
}

int VisionService::GetFallDownClassIndex() {
    if (impl_ == nullptr || impl_->model == nullptr) {
        return -1;
    }
    auto* seq_model = dynamic_cast<vision_core::ISequenceActionModel*>(impl_->model.get());
    if (seq_model == nullptr) return -1;
    return seq_model->get_fall_down_class_index();
}

VisionServiceStatus VisionService::Draw(const cv::Mat& image, cv::Mat* out_image) {
    if (impl_ == nullptr || impl_->model == nullptr) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "service/model must not be null");
    }
    if (out_image == nullptr) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "out_image must not be null");
    }
    if (image.empty()) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "image must not be empty");
    }
    if (image.channels() != 3) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "image must be 3-channel BGR");
    }
    if (!impl_->model->supports_capability(vision_core::ModelCapability::kDraw)) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "current model does not support draw");
    }

    const std::vector<vision_common::Result>& results = impl_->last_raw_results;
    if (results.empty()) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "No results to draw; run inference first");
    }

    try {
        *out_image = image.clone();
        bool has_keypoints = false;
        bool has_mask = false;
        bool has_track_id = false;
        for (const auto& r : results) {
            if (!r.keypoints.empty()) has_keypoints = true;
            if (r.mask != nullptr && !r.mask->empty()) has_mask = true;
            if (r.track_id >= 0) has_track_id = true;
        }

        if (has_keypoints) {
            vision_common::draw_keypoints(*out_image, results);
        } else if (has_mask) {
            vision_common::draw_segmentation(*out_image, results, impl_->labels);
        } else if (has_track_id) {
            vision_common::draw_tracking_results(*out_image, results, impl_->labels);
        } else {
            vision_common::draw_detections(*out_image, results, impl_->labels);
        }

        last_error_.clear();
        return VISION_SERVICE_OK;
    } catch (const std::exception& e) {
        return SetError(VISION_SERVICE_INFER_FAILED, e.what());
    } catch (...) {
        return SetError(VISION_SERVICE_INFER_FAILED, "Unknown error during draw");
    }
}

VisionServiceStatus VisionService::GetLastKeypoints(int result_index,
                                                    std::vector<VisionServiceKeypoint>* out_keypoints) {
    if (impl_ == nullptr || out_keypoints == nullptr) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "service/out_keypoints must not be null");
    }
    out_keypoints->clear();
    const std::vector<vision_common::Result>& results = impl_->last_raw_results;
    if (result_index < 0 || static_cast<size_t>(result_index) >= results.size()) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "result_index out of range");
    }
    const std::vector<vision_common::KeyPoint>& kps = results[static_cast<size_t>(result_index)].keypoints;
    if (kps.empty()) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "no keypoints in result");
    }
    out_keypoints->resize(kps.size());
    for (size_t i = 0; i < kps.size(); ++i) {
        (*out_keypoints)[i].x = kps[i].x;
        (*out_keypoints)[i].y = kps[i].y;
        (*out_keypoints)[i].visibility = kps[i].visibility;
    }
    last_error_.clear();
    return VISION_SERVICE_OK;
}

std::string VisionService::GetDefaultImage() {
    if (impl_ == nullptr || impl_->model == nullptr) {
        SetError(VISION_SERVICE_INVALID_ARGUMENT, "service/model must not be null");
        return {};
    }

    if (!impl_->default_image_path.empty()) {
        return impl_->default_image_path;
    }

    try {
        if (impl_->config_path.empty() || !std::filesystem::exists(impl_->config_path)) {
            return {};
        }
        YAML::Node config = YAML::LoadFile(impl_->config_path);
        if (!config["test_image"]) {
            return {};
        }
        const std::string raw_path = config["test_image"].as<std::string>();
        impl_->default_image_path = vision_core::resolveResourcePath(raw_path, impl_->config_path);
        return impl_->default_image_path;
    } catch (...) {
        return {};
    }
}

std::string VisionService::GetConfigPathValue(const std::string& config_key) {
    if (impl_ == nullptr || impl_->model == nullptr || config_key.empty()) {
        SetError(VISION_SERVICE_INVALID_ARGUMENT, "service/model or config_key invalid");
        return {};
    }

    try {
        if (impl_->config_path.empty() || !std::filesystem::exists(impl_->config_path)) {
            return {};
        }
        YAML::Node config = YAML::LoadFile(impl_->config_path);
        if (!config[config_key]) {
            return {};
        }
        const std::string raw_path = config[config_key].as<std::string>();
        impl_->last_config_path_value = vision_core::resolveResourcePath(raw_path, impl_->config_path);
        return impl_->last_config_path_value;
    } catch (...) {
        return {};
    }
}
