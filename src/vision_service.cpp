/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vision_service.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>  // NOLINT(build/c++17)
#include <iostream>
#include <memory>
#include <new>
#include <string>
#include <utility>
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
    VisionServiceTimingOptions timing_options;
    VisionServiceTiming last_timing;
    uint64_t timed_tracking_frame_count = 0;
    uint64_t timed_tracking_object_sum = 0;
};

thread_local std::string VisionService::g_last_error_;

namespace {

double ToMs(const std::chrono::steady_clock::duration& duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

void ResetAllTiming(VisionServiceTiming* timing) {
    if (timing == nullptr) {
        return;
    }
    timing->preprocess_ms = 0.0;
    timing->model_infer_ms = 0.0;
    timing->postprocess_ms = 0.0;
    timing->detect_ms = 0.0;
    timing->track_ms = 0.0;
    timing->infer_ms = 0.0;
    timing->draw_ms = 0.0;
    timing->embedding_ms = 0.0;
    timing->sequence_ms = 0.0;
}

void ResetImageTiming(VisionServiceTiming* timing) {
    ResetAllTiming(timing);
}

void ResetDrawTiming(VisionServiceTiming* timing) {
    if (timing == nullptr) {
        return;
    }
    timing->draw_ms = 0.0;
}

void FillTimingFromRuntimeProfile(const vision_core::RuntimeProfile& profile,
                                    bool is_tracking,
                                    VisionServiceTiming* timing) {
    if (timing == nullptr) {
        return;
    }
    if (is_tracking) {
        timing->detect_ms = profile.detect_ms;
        timing->track_ms = profile.track_ms;
        timing->infer_ms = profile.total_ms;
    } else {
        timing->preprocess_ms = profile.preprocess_ms;
        timing->model_infer_ms = profile.model_infer_ms;
        timing->postprocess_ms = profile.postprocess_ms;
        timing->infer_ms = profile.total_ms;
    }
}

void MaybePrintImageTiming(const VisionServiceTimingOptions& options,
                            const VisionServiceTiming& timing,
                            bool is_tracking,
                            int tracked_count = -1,
                            double avg_tracked_count = 0.0) {
    if (!options.enabled || !options.print_to_stdout) {
        return;
    }

    if (is_tracking) {
        std::cout << "[VisionService][Timing][Image][tracking] "
                    << "detect=" << timing.detect_ms << "ms, "
                    << "track=" << timing.track_ms << "ms, "
                    << "tracked=" << tracked_count << ", "
                    << "avg_tracked=" << avg_tracked_count << ", "
                    << "total=" << timing.infer_ms << "ms"
                    << std::endl;
    } else {
        std::cout << "[VisionService][Timing][Image][generic] "
                    << "preprocess=" << timing.preprocess_ms << "ms, "
                    << "model_infer=" << timing.model_infer_ms << "ms, "
                    << "postprocess=" << timing.postprocess_ms << "ms, "
                    << "total=" << timing.infer_ms << "ms"
                    << std::endl;
    }
}

void MaybePrintEmbeddingTiming(const VisionServiceTimingOptions& options,
                                const VisionServiceTiming& timing) {
    if (!options.enabled || !options.print_to_stdout) {
        return;
    }
    std::cout << "[VisionService][Timing][Embedding] "
                << "preprocess=" << timing.preprocess_ms << "ms, "
                << "model_infer=" << timing.model_infer_ms << "ms, "
                << "postprocess=" << timing.postprocess_ms << "ms, "
                << "total=" << timing.embedding_ms << "ms"
                << std::endl;
}

void MaybePrintSequenceTiming(const VisionServiceTimingOptions& options,
                                const VisionServiceTiming& timing) {
    if (!options.enabled || !options.print_to_stdout) {
        return;
    }
    std::cout << "[VisionService][Timing][Sequence] "
                << "preprocess=" << timing.preprocess_ms << "ms, "
                << "model_infer=" << timing.model_infer_ms << "ms, "
                << "postprocess=" << timing.postprocess_ms << "ms, "
                << "total=" << timing.sequence_ms << "ms"
                << std::endl;
}

void MaybePrintDrawTiming(const VisionServiceTimingOptions& options,
                            const VisionServiceTiming& timing) {
    if (!options.enabled || !options.print_to_stdout) {
        return;
    }
    std::cout << "[VisionService][Timing][Draw] "
                << "total=" << timing.draw_ms << "ms"
                << std::endl;
}

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
        if (!src.keypoints.empty()) {
            dst.keypoints.resize(src.keypoints.size());
            for (size_t i = 0; i < src.keypoints.size(); ++i) {
                dst.keypoints[i].x = src.keypoints[i].x;
                dst.keypoints[i].y = src.keypoints[i].y;
                dst.keypoints[i].visibility = src.keypoints[i].visibility;
            }
        }
        if (src.mask != nullptr && !src.mask->empty()) {
            dst.mask = src.mask->clone();
        }
        out_results->push_back(std::move(dst));
    }
}

}  // namespace

VisionService::~VisionService() = default;

void VisionService::Release() {
    if (impl_ == nullptr) {
        return;
    }
    impl_->model.reset();
    impl_->last_raw_results.clear();
    impl_->labels.clear();
    impl_->default_image_path.clear();
    impl_->last_config_path_value.clear();
    ResetAllTiming(&impl_->last_timing);
    impl_->timed_tracking_frame_count = 0;
    impl_->timed_tracking_object_sum = 0;
}

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
        const bool timing_enabled = (impl_ != nullptr) && impl_->timing_options.enabled;
        std::chrono::steady_clock::time_point t0;
        if (timing_enabled) {
            ResetImageTiming(&impl_->last_timing);
            t0 = std::chrono::steady_clock::now();
        }
        vision_core::BaseModel* model = impl_->model.get();
        const bool is_tracking = model->supports_capability(vision_core::ModelCapability::kTrackUpdate);
        if (is_tracking) {
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

        if (timing_enabled) {
            const auto t1 = std::chrono::steady_clock::now();
            const auto profile = model->get_runtime_profile();
            FillTimingFromRuntimeProfile(profile, is_tracking, &impl_->last_timing);
            if (impl_->last_timing.infer_ms <= 0.0) {
                impl_->last_timing.infer_ms = ToMs(t1 - t0);
            }
            if (is_tracking) {
                int tracked_count = 0;
                for (const auto& r : impl_->last_raw_results) {
                    if (r.track_id >= 0) {
                        ++tracked_count;
                    }
                }
                ++impl_->timed_tracking_frame_count;
                impl_->timed_tracking_object_sum += static_cast<uint64_t>(tracked_count);
                const double avg_tracked_count =
                    (impl_->timed_tracking_frame_count > 0)
                        ? (static_cast<double>(impl_->timed_tracking_object_sum) /
                            static_cast<double>(impl_->timed_tracking_frame_count))
                        : 0.0;
                MaybePrintImageTiming(
                    impl_->timing_options, impl_->last_timing, is_tracking,
                    tracked_count, avg_tracked_count);
            } else {
                MaybePrintImageTiming(impl_->timing_options, impl_->last_timing, is_tracking);
            }
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
        const bool timing_enabled = (impl_ != nullptr) && impl_->timing_options.enabled;
        std::chrono::steady_clock::time_point t0;
        if (timing_enabled) {
            ResetAllTiming(&impl_->last_timing);
            t0 = std::chrono::steady_clock::now();
        }
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
        if (timing_enabled) {
            const auto t1 = std::chrono::steady_clock::now();
            const auto profile = model->get_runtime_profile();
            impl_->last_timing.preprocess_ms = profile.preprocess_ms;
            impl_->last_timing.model_infer_ms = profile.model_infer_ms;
            impl_->last_timing.postprocess_ms = profile.postprocess_ms;
            impl_->last_timing.embedding_ms =
                (profile.total_ms > 0.0) ? profile.total_ms : ToMs(t1 - t0);
            MaybePrintEmbeddingTiming(impl_->timing_options, impl_->last_timing);
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
        const bool timing_enabled = (impl_ != nullptr) && impl_->timing_options.enabled;
        std::chrono::steady_clock::time_point t0;
        if (timing_enabled) {
            ResetAllTiming(&impl_->last_timing);
            t0 = std::chrono::steady_clock::now();
        }
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
        if (timing_enabled) {
            const auto t1 = std::chrono::steady_clock::now();
            const auto profile = model->get_runtime_profile();
            impl_->last_timing.preprocess_ms = profile.preprocess_ms;
            impl_->last_timing.model_infer_ms = profile.model_infer_ms;
            impl_->last_timing.postprocess_ms = profile.postprocess_ms;
            impl_->last_timing.sequence_ms =
                (profile.total_ms > 0.0) ? profile.total_ms : ToMs(t1 - t0);
            MaybePrintSequenceTiming(impl_->timing_options, impl_->last_timing);
        }
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
        const bool timing_enabled = (impl_ != nullptr) && impl_->timing_options.enabled;
        if (timing_enabled) {
            ResetDrawTiming(&impl_->last_timing);
        }
        const auto t0 = timing_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
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

        if (timing_enabled) {
            const auto t1 = std::chrono::steady_clock::now();
            impl_->last_timing.draw_ms = ToMs(t1 - t0);
            MaybePrintDrawTiming(impl_->timing_options, impl_->last_timing);
        }

        last_error_.clear();
        return VISION_SERVICE_OK;
    } catch (const std::exception& e) {
        return SetError(VISION_SERVICE_INFER_FAILED, e.what());
    } catch (...) {
        return SetError(VISION_SERVICE_INFER_FAILED, "Unknown error during draw");
    }
}

bool VisionService::SupportsDraw() const {
    return impl_ != nullptr && impl_->model != nullptr &&
            impl_->model->supports_capability(vision_core::ModelCapability::kDraw);
}

void VisionService::SetTimingOptions(const VisionServiceTimingOptions& options) {
    if (impl_ == nullptr) {
        return;
    }
    const bool was_enabled = impl_->timing_options.enabled;
    impl_->timing_options = options;
    if (!options.enabled) {
        ResetAllTiming(&impl_->last_timing);
        impl_->timed_tracking_frame_count = 0;
        impl_->timed_tracking_object_sum = 0;
    } else if (!was_enabled) {
        ResetAllTiming(&impl_->last_timing);
        impl_->timed_tracking_frame_count = 0;
        impl_->timed_tracking_object_sum = 0;
    }
}

VisionServiceTiming VisionService::GetLastTiming() const {
    if (impl_ == nullptr) {
        return VisionServiceTiming{};
    }
    return impl_->last_timing;
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
