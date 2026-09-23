/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark_scenario.h"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <yaml-cpp/yaml.h>

namespace vision_benchmark {
namespace {

bool SetError(std::string *error, const std::string &message) {
    if (error != nullptr) {
        *error = message;
    }
    return false;
}

std::string ConfigPath(VisionService *service, const std::string &key) {
    return service == nullptr ? std::string{}
                                : service->GetConfigPathValue(key);
}

bool LoadImage(const std::string &path, const char *label, cv::Mat *image,
                std::string *error) {
    if (path.empty()) {
        return SetError(error, std::string("no ") + label + " path configured");
    }
    *image = cv::imread(path);
    if (image->empty()) {
        return SetError(error,
                        std::string("could not read ") + label + ": " + path);
    }
    return true;
}

std::string PairDescription(const std::string &first,
                            const std::string &second) {
    return first + " | " + second;
}

class ImageScenario final : public BenchmarkScenario {
public:
    ImageScenario(std::string path, cv::Mat image,
                    std::vector<cv::Point2f> point_coords,
                    std::vector<int> point_labels)
        : path_(std::move(path)), image_(std::move(image)),
            point_coords_(std::move(point_coords)),
            point_labels_(std::move(point_labels)) {}

    const char *name() const override {
        return point_coords_.empty() ? "image" : "prompt_segmentation";
    }

    std::string input_description() const override { return path_; }

    const char *measured_scope() const override {
        return point_coords_.empty()
                    ? "one complete image Infer(); image decoding excluded"
                    : "prompted encoder+decoder Infer(); image decoding "
                    "excluded";
    }

    bool Initialize(VisionService *, std::string *) override { return true; }

    bool NextRequest(VisionServiceRequest *request,
                    std::string *error) override {
        if (request == nullptr) {
            return SetError(error, "request must not be null");
        }
        *request = VisionServiceRequest{};
        request->image = image_;
        request->point_coords = point_coords_;
        request->point_labels = point_labels_;
        return true;
    }

    const cv::Mat &draw_image() const override { return image_; }

private:
    std::string path_;
    cv::Mat image_;
    std::vector<cv::Point2f> point_coords_;
    std::vector<int> point_labels_;
};

class StereoScenario final : public BenchmarkScenario {
public:
    StereoScenario(std::string left_path, std::string right_path, cv::Mat left,
                    cv::Mat right)
        : left_path_(std::move(left_path)), right_path_(std::move(right_path)),
            left_(std::move(left)), right_(std::move(right)) {}

    const char *name() const override { return "stereo"; }

    std::string input_description() const override {
        return PairDescription(left_path_, right_path_);
    }

    const char *measured_scope() const override {
        return "one complete stereo Infer(); image decoding excluded";
    }

    bool Initialize(VisionService *, std::string *) override { return true; }

    bool NextRequest(VisionServiceRequest *request,
                    std::string *error) override {
        if (request == nullptr) {
            return SetError(error, "request must not be null");
        }
        *request = VisionServiceRequest{};
        request->image = left_;
        request->image2 = right_;
        return true;
    }

    const cv::Mat &draw_image() const override { return left_; }

private:
    std::string left_path_;
    std::string right_path_;
    cv::Mat left_;
    cv::Mat right_;
};

class FeatureMatchingScenario final : public BenchmarkScenario {
public:
    FeatureMatchingScenario(std::string image_path1, std::string image_path2,
                            vision::LocalFeatures features1,
                            vision::LocalFeatures features2)
        : image_path1_(std::move(image_path1)),
            image_path2_(std::move(image_path2)),
            features1_(std::move(features1)), features2_(std::move(features2)) {}

    const char *name() const override { return "feature_matching"; }

    std::string input_description() const override {
        return PairDescription(image_path1_, image_path2_);
    }

    const char *measured_scope() const override {
        return "matcher Infer() only; feature extraction and image decoding "
                "excluded";
    }

    bool Initialize(VisionService *, std::string *) override { return true; }

    bool NextRequest(VisionServiceRequest *request,
                    std::string *error) override {
        if (request == nullptr) {
            return SetError(error, "request must not be null");
        }
        *request = VisionServiceRequest{};
        request->local_features0 = &features1_;
        request->local_features1 = &features2_;
        return true;
    }

    const cv::Mat &draw_image() const override { return empty_; }

private:
    std::string image_path1_;
    std::string image_path2_;
    vision::LocalFeatures features1_;
    vision::LocalFeatures features2_;
    cv::Mat empty_;
};

class VideoTrackingScenario final : public BenchmarkScenario {
public:
    VideoTrackingScenario(std::string path, bool has_initial_bbox,
                            vision::BoundingBox initial_bbox)
        : path_(std::move(path)), capture_(path_),
            has_initial_bbox_(has_initial_bbox), initial_bbox_(initial_bbox) {}

    bool is_open() const { return capture_.isOpened(); }

    const char *name() const override { return "video_tracking"; }

    std::string input_description() const override { return path_; }

    const char *measured_scope() const override {
        return "per-frame Infer() on sequential frames; video decode and "
                "tracker "
                "initialization excluded";
    }

    bool Initialize(VisionService *service, std::string *error) override {
        if (!has_initial_bbox_) {
            return true;
        }
        if (service == nullptr) {
            return SetError(error, "service must not be null");
        }
        if (!capture_.read(frame_) || frame_.empty()) {
            return SetError(
                error, "could not read tracker initialization frame: " + path_);
        }
        VisionServiceRequest request;
        request.image = frame_;
        request.has_initial_bbox = true;
        request.initial_bbox = initial_bbox_;
        VisionServiceResponse response;
        if (service->Infer(request, &response) != VISION_SERVICE_OK) {
            return SetError(error, "tracker initialization failed: " +
                                        service->LastError());
        }
        return true;
    }

    bool NextRequest(VisionServiceRequest *request,
                    std::string *error) override {
        if (request == nullptr) {
            return SetError(error, "request must not be null");
        }
        if (!capture_.read(frame_) || frame_.empty()) {
            return SetError(error, "video ended before all warmup and measured "
                                    "runs completed: " +
                                        path_);
        }
        *request = VisionServiceRequest{};
        request->image = frame_;
        return true;
    }

    const cv::Mat &draw_image() const override { return frame_; }

private:
    std::string path_;
    cv::VideoCapture capture_;
    bool has_initial_bbox_ = false;
    vision::BoundingBox initial_bbox_;
    cv::Mat frame_;
};

bool ExtractFeatures(VisionService *extractor, const cv::Mat &image,
                    vision::LocalFeatures *features, std::string *error) {
    VisionServiceResponse response;
    if (extractor->Infer(image, &response) != VISION_SERVICE_OK) {
        return SetError(error,
                        "feature extraction failed: " + extractor->LastError());
    }
    if (response.results.size() != 1) {
        return SetError(
            error, "feature extractor returned an unexpected result count");
    }
    const auto *value =
        std::get_if<vision::LocalFeatures>(&response.results.front());
    if (value == nullptr) {
        return SetError(error,
                        "feature extractor did not return LocalFeatures");
    }
    *features = *value;
    return true;
}

bool ParsePrompt(const YAML::Node &benchmark, std::vector<cv::Point2f> *coords,
                std::vector<int> *labels, std::string *error) {
    const YAML::Node coord_nodes = benchmark["point_coords"];
    const YAML::Node label_nodes = benchmark["point_labels"];
    if (!coord_nodes || !coord_nodes.IsSequence() || !label_nodes ||
        !label_nodes.IsSequence() || coord_nodes.size() == 0 ||
        coord_nodes.size() != label_nodes.size()) {
        return SetError(error, "prompt_segmentation requires equally sized "
                                "benchmark.point_coords "
                                "and benchmark.point_labels");
    }
    for (std::size_t index = 0; index < coord_nodes.size(); ++index) {
        const YAML::Node point = coord_nodes[index];
        if (!point.IsSequence() || point.size() != 2) {
            return SetError(error,
                            "each benchmark.point_coords item must be [x, y]");
        }
        coords->emplace_back(point[0].as<float>(), point[1].as<float>());
        labels->push_back(label_nodes[index].as<int>());
    }
    return true;
}

bool ParseInitialBox(const YAML::Node &config, bool *has_box,
                    vision::BoundingBox *box, std::string *error) {
    const YAML::Node node = config["initial_bbox"];
    if (!node) {
        *has_box = false;
        return true;
    }
    try {
        const float x = node["x"].as<float>();
        const float y = node["y"].as<float>();
        const float width = node["w"].as<float>();
        const float height = node["h"].as<float>();
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(width) ||
            !std::isfinite(height) || width <= 0.0F || height <= 0.0F) {
            return SetError(
                error, "initial_bbox must contain a positive finite xywh box");
        }
        *has_box = true;
        *box = {x, y, x + width, y + height};
        return true;
    } catch (const std::exception &exception) {
        return SetError(error, std::string("invalid initial_bbox: ") +
                                    exception.what());
    }
}

} // namespace

std::unique_ptr<BenchmarkScenario>
CreateBenchmarkScenario(const std::string &config_path,
                        const ScenarioOverrides &overrides,
                        VisionService *service, std::string *error) {
    if (service == nullptr) {
        SetError(error, "service must not be null");
        return nullptr;
    }

    try {
        const YAML::Node config = YAML::LoadFile(config_path);
        const YAML::Node benchmark = config["benchmark"];
        std::string scenario = "image";
        if (benchmark && benchmark["scenario"]) {
            scenario = benchmark["scenario"].as<std::string>();
        }

        if (scenario == "image" || scenario == "prompt_segmentation") {
            std::string path = overrides.input_path;
            if (path.empty()) {
                path = service->GetDefaultImage();
            }
            if (path.empty()) {
                path = ConfigPath(service, "test_image1");
            }
            cv::Mat image;
            if (!LoadImage(path, "input image", &image, error)) {
                return nullptr;
            }
            std::vector<cv::Point2f> coords;
            std::vector<int> labels;
            if (scenario == "prompt_segmentation" &&
                !ParsePrompt(benchmark, &coords, &labels, error)) {
                return nullptr;
            }
            return std::make_unique<ImageScenario>(
                std::move(path), std::move(image), std::move(coords),
                std::move(labels));
        }

        if (scenario == "stereo") {
            const std::string left_path =
                overrides.input_path.empty()
                    ? ConfigPath(service, "test_image1")
                    : overrides.input_path;
            const std::string right_path =
                overrides.input_path2.empty()
                    ? ConfigPath(service, "test_image2")
                    : overrides.input_path2;
            cv::Mat left;
            cv::Mat right;
            if (!LoadImage(left_path, "left image", &left, error) ||
                !LoadImage(right_path, "right image", &right, error)) {
                return nullptr;
            }
            return std::make_unique<StereoScenario>(
                left_path, right_path, std::move(left), std::move(right));
        }

        if (scenario == "feature_matching") {
            const std::string image_path1 =
                overrides.input_path.empty()
                    ? ConfigPath(service, "test_image1")
                    : overrides.input_path;
            const std::string image_path2 =
                overrides.input_path2.empty()
                    ? ConfigPath(service, "test_image2")
                    : overrides.input_path2;
            cv::Mat image1;
            cv::Mat image2;
            if (!LoadImage(image_path1, "first feature image", &image1,
                            error) ||
                !LoadImage(image_path2, "second feature image", &image2,
                            error)) {
                return nullptr;
            }
            const std::string extractor_config =
                ConfigPath(service, "superpoint_config_path");
            if (extractor_config.empty()) {
                SetError(error,
                        "feature_matching requires superpoint_config_path");
                return nullptr;
            }
            auto extractor = VisionService::Create(extractor_config, "", false);
            if (!extractor) {
                SetError(error, "could not create feature extractor: " +
                                    VisionService::LastCreateError());
                return nullptr;
            }
            vision::LocalFeatures features1;
            vision::LocalFeatures features2;
            if (!ExtractFeatures(extractor.get(), image1, &features1, error) ||
                !ExtractFeatures(extractor.get(), image2, &features2, error)) {
                return nullptr;
            }
            return std::make_unique<FeatureMatchingScenario>(
                image_path1, image_path2, std::move(features1),
                std::move(features2));
        }

        if (scenario == "video_tracking") {
            const std::string path = overrides.input_path.empty()
                                        ? ConfigPath(service, "test_video")
                                        : overrides.input_path;
            if (path.empty()) {
                SetError(error, "video_tracking requires test_video or --image "
                                "<video>");
                return nullptr;
            }
            bool has_initial_bbox = false;
            vision::BoundingBox initial_bbox;
            if (!ParseInitialBox(config, &has_initial_bbox, &initial_bbox,
                                error)) {
                return nullptr;
            }
            auto video = std::make_unique<VideoTrackingScenario>(
                path, has_initial_bbox, initial_bbox);
            if (!video->is_open()) {
                SetError(error, "could not open input video: " + path);
                return nullptr;
            }
            return video;
        }

        SetError(error, "unsupported benchmark.scenario: " + scenario);
        return nullptr;
    } catch (const std::exception &exception) {
        SetError(error, std::string("failed to prepare benchmark scenario: ") +
                            exception.what());
        return nullptr;
    }
}

} // namespace vision_benchmark
