/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <filesystem>  // NOLINT(build/c++17)
#include <iostream>
#include <memory>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "vision_service.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
        << " <config.yaml> [left_image right_image]\n";
        return 1;
    }
    auto service = VisionService::Create(argv[1], "", true);
    if (!service) {
        std::cerr << VisionService::LastCreateError() << '\n';
        return 1;
    }
    std::string left_path =
        service->GetConfigPathValue("test_image1");
    std::string right_path =
        service->GetConfigPathValue("test_image2");
    if (argc >= 4) {
        left_path = argv[2];
        right_path = argv[3];
    }
    const cv::Mat left = cv::imread(left_path);
    const cv::Mat right = cv::imread(right_path);
    if (left.empty() || right.empty()) {
        std::cerr << "Failed to load stereo images\n";
        return 1;
    }

    VisionServiceRequest request;
    request.image = left;
    request.image2 = right;
    VisionServiceResponse response;
    if (service->Infer(request, &response) != VISION_SERVICE_OK ||
        response.results.size() != 1) {
        std::cerr << "Inference failed: "
        << service->LastError() << '\n';
        return 1;
    }
    const auto* disparity =
        std::get_if<vision::Disparity>(&response.results.front());
    if (disparity == nullptr || !disparity->map ||
        disparity->map->empty()) {
        std::cerr << "BANet2D returned no disparity map\n";
        return 1;
    }
    cv::Mat normalized;
    cv::normalize(
        *disparity->map, normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::Mat color;
    cv::applyColorMap(normalized, color, cv::COLORMAP_TURBO);
    const std::string output = "banet2d_disparity.png";
    if (!cv::imwrite(output, color)) {
        std::cerr << "Failed to write " << output << '\n';
        return 1;
    }
    std::cout << "Disparity " << disparity->map->cols << 'x'
            << disparity->map->rows << ", output: " << output << '\n';
    return 0;
}
