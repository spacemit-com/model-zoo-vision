/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include "vision_model_base.h"
#include "vision_model_factory.h"
#include "vision_service.h"

namespace {

int g_failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        ++g_failures;
    }
}

class ExtendedProbeModel final : public vision_core::BaseModel {
public:
    explicit ExtendedProbeModel(bool lazy_load)
        : BaseModel("/tmp/unused-extended-probe.onnx", lazy_load) {}

    static std::unique_ptr<vision_core::BaseModel> create(
        const YAML::Node&, bool lazy_load) {
        return std::make_unique<ExtendedProbeModel>(lazy_load);
    }

    void load_model() override {
        model_loaded_ = true;
    }

    std::vector<vision_core::InferIntent> supported_intents() const override {
        return {
            vision_core::InferIntent::kStereoDepth,
            vision_core::InferIntent::kMatchLocalFeatures,
            vision_core::InferIntent::kTrack,
        };
    }

    std::vector<vision_core::ModelCapability> get_capabilities() const override {
        return {vision_core::ModelCapability::kDraw};
    }

    vision_core::InferResponse Run(
        const vision_core::InferRequest& request) override {
        bool supported = false;
        for (const auto intent : supported_intents()) {
            if (request.intent == intent) {
                supported = true;
                break;
            }
        }
        if (!supported) {
            return unsupported_intent_response(request.intent);
        }
        vision_core::InferResponse response;
        if (request.intent == vision_core::InferIntent::kMatchLocalFeatures) {
            const auto* pair =
                std::get_if<vision_core::LocalFeaturePairInput>(&request.input);
            if (pair == nullptr) {
                response.ok = false;
                response.error_message = "expected LocalFeaturePairInput";
                return response;
            }
            vision::FeatureMatch match;
            match.query_index = 0;
            match.train_index = 0;
            match.query_point = pair->query.keypoints.front();
            match.train_point = pair->train.keypoints.front();
            match.score = 0.75f;
            response.results.emplace_back(std::move(match));
            return response;
        }
        if (request.intent == vision_core::InferIntent::kStereoDepth) {
            const auto* stereo =
                std::get_if<vision_core::StereoImageInput>(&request.input);
            if (stereo == nullptr) {
                response.ok = false;
                response.error_message = "expected StereoImageInput";
                return response;
            }
            auto map = std::make_shared<cv::Mat>(
                stereo->left.image.rows,
                stereo->left.image.cols,
                CV_32FC1);
            for (int y = 0; y < map->rows; ++y) {
                for (int x = 0; x < map->cols; ++x) {
                    map->at<float>(y, x) =
                        static_cast<float>(y * map->cols + x);
                }
            }
            vision::Disparity disparity;
            disparity.map = std::move(map);
            response.results.emplace_back(std::move(disparity));
            return response;
        }

        const auto* image =
            std::get_if<vision_core::ImageInput>(&request.input);
        if (image == nullptr) {
            response.ok = false;
            response.error_message = "expected ImageInput";
            return response;
        }
        vision::Tracking tracking;
        tracking.bbox = image->initial_bbox;
        tracking.track_id = 0;
        response.results.emplace_back(std::move(tracking));
        return response;
    }
};

vision_core::ModelRegistrar<ExtendedProbeModel> registrar(
    "ExtendedProbeModel");

std::string write_config() {
    const std::string path = "/tmp/vision_extended_probe.yaml";
    std::ofstream output(path);
    output << "class: deploy.test.ExtendedProbeModel\n"
        << "model_path: /tmp/unused-extended-probe.onnx\n"
        << "default_params: {}\n";
    return path;
}

vision::LocalFeatures make_features() {
    vision::LocalFeatures features;
    features.image_width = 640;
    features.image_height = 480;
    features.descriptor_dim = 2;
    features.feature_type = "unit";
    features.keypoints = {{10.0f, 20.0f, 0.9f}};
    features.descriptors = {1.0f, 2.0f};
    return features;
}

}  // namespace

int main() {
    ExtendedProbeModel probe(true);
    vision_core::InferRequest unsupported_request{};
    unsupported_request.intent = vision_core::InferIntent::kOcr;
    const vision_core::InferResponse unsupported_response =
        probe.Run(unsupported_request);
    check(
        !unsupported_response.ok,
        "unsupported intent should return a non-fatal error response");
    check(
        unsupported_response.error_message.find("kOcr") != std::string::npos &&
            unsupported_response.error_message.find("kTrack") != std::string::npos,
        "unsupported intent response should identify requested and supported intents");

    const std::string config_path = write_config();
    std::unique_ptr<VisionService> service =
        VisionService::Create(config_path, "", true);
    check(service != nullptr, "probe service should be created");
    if (!service) {
        std::remove(config_path.c_str());
        return 1;
    }

    vision::LocalFeatures query = make_features();
    vision::LocalFeatures train = make_features();
    VisionServiceRequest match_request{};
    match_request.local_features0 = &query;
    match_request.local_features1 = &train;
    VisionServiceResponse response;
    check(
        service->Infer(match_request, &response) == VISION_SERVICE_OK,
        "feature-pair request should route to matching intent");
    check(response.results.size() == 1, "matching should return one result");
    check(
        !response.results.empty() &&
            std::holds_alternative<vision::FeatureMatch>(
                response.results.front()),
        "matching should return FeatureMatch");

    VisionServiceRequest ambiguous = match_request;
    ambiguous.image = cv::Mat::zeros(4, 4, CV_8UC3);
    check(
        service->Infer(ambiguous, &response) ==
            VISION_SERVICE_INVALID_ARGUMENT,
        "image plus local features should be rejected");

    VisionServiceRequest stereo{};
    stereo.image = cv::Mat::zeros(2, 3, CV_8UC3);
    stereo.image2 = cv::Mat::zeros(2, 3, CV_8UC3);
    check(
        service->Infer(stereo, &response) == VISION_SERVICE_OK,
        "two images should route to stereo intent");
    check(
        !response.results.empty() &&
            std::holds_alternative<vision::Disparity>(
                response.results.front()),
        "stereo inference should return Disparity");
    cv::Mat drawn;
    check(
        service->Draw(stereo.image, response, &drawn) == VISION_SERVICE_OK,
        "disparity result should support Draw");
    check(
        drawn.type() == CV_8UC3 && drawn.size() == stereo.image.size(),
        "drawn disparity should be BGR and keep input size");

    VisionServiceResponse depth_response;
    vision::DepthMap depth;
    depth.map = std::make_shared<cv::Mat>(2, 3, CV_32FC1);
    for (int y = 0; y < depth.map->rows; ++y) {
        for (int x = 0; x < depth.map->cols; ++x) {
            depth.map->at<float>(y, x) =
                1.0f + static_cast<float>(y * depth.map->cols + x);
        }
    }
    depth_response.results.emplace_back(std::move(depth));
    check(
        service->Draw(stereo.image, depth_response, &drawn) ==
            VISION_SERVICE_OK,
        "metric depth result should support Draw");
    check(
        drawn.type() == CV_8UC3 && drawn.size() == stereo.image.size(),
        "drawn metric depth should be BGR and keep input size");

    VisionServiceResponse wrong_size_depth_response;
    vision::DepthMap wrong_size_depth;
    wrong_size_depth.map =
        std::make_shared<cv::Mat>(1, 1, CV_32FC1, cv::Scalar(1.0f));
    wrong_size_depth_response.results.emplace_back(
        std::move(wrong_size_depth));
    check(
        service->Draw(
            stereo.image,
            wrong_size_depth_response,
            &drawn) == VISION_SERVICE_INFER_FAILED,
        "metric depth draw should reject a map with the wrong size");

    VisionServiceResponse wrong_size_disparity_response;
    vision::Disparity wrong_size_disparity;
    wrong_size_disparity.map =
        std::make_shared<cv::Mat>(1, 1, CV_32FC1, cv::Scalar(1.0f));
    wrong_size_disparity_response.results.emplace_back(
        std::move(wrong_size_disparity));
    check(
        service->Draw(
            stereo.image,
            wrong_size_disparity_response,
            &drawn) == VISION_SERVICE_INFER_FAILED,
        "disparity draw should reject a map with the wrong size");

    VisionServiceRequest bad_stereo = stereo;
    bad_stereo.image2 = cv::Mat::zeros(2, 3, CV_32FC1);
    check(
        service->Infer(bad_stereo, &response) ==
            VISION_SERVICE_INVALID_ARGUMENT,
        "invalid second BGR image type should be rejected");

    VisionServiceRequest stereo_with_tracking_box = stereo;
    stereo_with_tracking_box.has_initial_bbox = true;
    stereo_with_tracking_box.initial_bbox = {0.0f, 0.0f, 2.0f, 2.0f};
    check(
        service->Infer(stereo_with_tracking_box, &response) ==
            VISION_SERVICE_INVALID_ARGUMENT,
        "tracking initialization must not be ignored by stereo inference");

    VisionServiceRequest bad_tracking{};
    bad_tracking.image = cv::Mat::zeros(20, 30, CV_8UC3);
    bad_tracking.has_initial_bbox = true;
    bad_tracking.initial_bbox = {4.0f, 5.0f, 4.0f, 10.0f};
    check(
        service->Infer(bad_tracking, &response) ==
            VISION_SERVICE_INVALID_ARGUMENT,
        "zero-width tracking initialization should be rejected");

    const vision::Result result = vision::FeatureMatch{};
    check(vision::get_label(result) == -1, "FeatureMatch label should default to -1");
    check(vision::get_bbox(result).x1 == 0.0f, "FeatureMatch bbox should be empty");

    std::remove(config_path.c_str());
    if (g_failures != 0) {
        std::cerr << g_failures << " assertion(s) failed" << std::endl;
        return 1;
    }
    std::cout << "PASS: extended inference contract" << std::endl;
    return 0;
}
