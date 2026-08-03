/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <memory>
#include <string>

#include <opencv2/imgcodecs.hpp>

#include "vision_service.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
        << " <config.yaml> [image]\n";
        return 1;
    }
    auto service = VisionService::Create(argv[1], "", true);
    if (!service) {
        std::cerr << VisionService::LastCreateError() << '\n';
        return 1;
    }
    const std::string image_path =
        argc >= 3 ? argv[2] : service->GetDefaultImage();
    const cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << '\n';
        return 1;
    }
    VisionServiceResponse response;
    if (service->Infer(image, &response) != VISION_SERVICE_OK ||
        response.results.size() != 1) {
        std::cerr << "Inference failed: "
        << service->LastError() << '\n';
        return 1;
    }
    const auto* features =
        std::get_if<vision::LocalFeatures>(&response.results.front());
    if (features == nullptr) {
        std::cerr << "SuperPoint returned an unexpected result type\n";
        return 1;
    }
    cv::Mat output;
    if (service->Draw(image, response, &output) != VISION_SERVICE_OK ||
        !cv::imwrite("superpoint_keypoints.jpg", output)) {
        std::cerr << "Failed to draw SuperPoint keypoints\n";
        return 1;
    }
    std::cout << "Features: " << features->keypoints.size()
            << ", descriptor dim: " << features->descriptor_dim
            << ", output: superpoint_keypoints.jpg\n";
    return 0;
}
