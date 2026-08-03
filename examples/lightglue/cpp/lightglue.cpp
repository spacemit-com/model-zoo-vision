/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "vision_service.h"

namespace {

bool extract(
    VisionService* service,
    const cv::Mat& image,
    vision::LocalFeatures* features) {
    VisionServiceResponse response;
    if (service->Infer(image, &response) != VISION_SERVICE_OK ||
        response.results.size() != 1) {
        return false;
    }
    const auto* value =
        std::get_if<vision::LocalFeatures>(&response.results.front());
    if (value == nullptr) {
        return false;
    }
    *features = *value;
    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
        << " <lightglue.yaml> [image1 image2]\n";
        return 1;
    }
    auto matcher = VisionService::Create(argv[1], "", true);
    if (!matcher) {
        std::cerr << VisionService::LastCreateError() << '\n';
        return 1;
    }
    const std::string extractor_config =
        matcher->GetConfigPathValue("superpoint_config_path");
    auto extractor =
        VisionService::Create(extractor_config, "", true);
    if (!extractor) {
        std::cerr << VisionService::LastCreateError() << '\n';
        return 1;
    }
    std::string image1_path =
        matcher->GetConfigPathValue("test_image1");
    std::string image2_path =
        matcher->GetConfigPathValue("test_image2");
    if (argc >= 4) {
        image1_path = argv[2];
        image2_path = argv[3];
    }
    const cv::Mat image1 = cv::imread(image1_path);
    const cv::Mat image2 = cv::imread(image2_path);
    if (image1.empty() || image2.empty()) {
        std::cerr << "Failed to load image pair\n";
        return 1;
    }

    vision::LocalFeatures features1;
    vision::LocalFeatures features2;
    if (!extract(extractor.get(), image1, &features1) ||
        !extract(extractor.get(), image2, &features2)) {
        std::cerr << "SuperPoint extraction failed: "
        << extractor->LastError() << '\n';
        return 1;
    }
    VisionServiceRequest request;
    request.local_features0 = &features1;
    request.local_features1 = &features2;
    VisionServiceResponse response;
    if (matcher->Infer(request, &response) != VISION_SERVICE_OK) {
        std::cerr << "LightGlue matching failed: "
        << matcher->LastError() << '\n';
        return 1;
    }

    const int canvas_height =
        std::max(image1.rows, image2.rows);
    cv::Mat canvas = cv::Mat::zeros(
        canvas_height,
        image1.cols + image2.cols,
        CV_8UC3);
    image1.copyTo(canvas(cv::Rect(0, 0, image1.cols, image1.rows)));
    image2.copyTo(canvas(
        cv::Rect(image1.cols, 0, image2.cols, image2.rows)));
    for (const auto& result : response.results) {
        const auto* match =
            std::get_if<vision::FeatureMatch>(&result);
        if (match == nullptr) {
            continue;
        }
        cv::line(
            canvas,
            cv::Point(
                cvRound(match->query_point.x),
                cvRound(match->query_point.y)),
            cv::Point(
                image1.cols + cvRound(match->train_point.x),
                cvRound(match->train_point.y)),
            cv::Scalar(0, 255, 0),
            1,
            cv::LINE_AA);
    }
    if (!cv::imwrite("lightglue_matches.jpg", canvas)) {
        std::cerr << "Failed to write match visualization\n";
        return 1;
    }
    std::cout << "Matches: " << response.results.size()
            << ", output: lightglue_matches.jpg\n";
    return 0;
}
