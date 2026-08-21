/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <variant>

#include <opencv2/opencv.hpp>

#include "vision_service.h"

namespace {

void print_usage(const char* program_name) {
    std::cout
        << "Usage: " << program_name
        << " <config_yaml> [options]\n"
        << "  --model-path <path>   Override model_path in yaml\n"
        << "  --image <path>        Input image path\n"
        << "  --output <path>       Output visualization path"
        << " (default: yolo26_depth_result.jpg)\n"
        << "  --help                Show this help message\n";
}

bool depth_statistics(
    const cv::Mat& depth,
    float* min_depth,
    float* max_depth,
    double* mean_depth,
    size_t* valid_count) {
    *min_depth = std::numeric_limits<float>::infinity();
    *max_depth = -std::numeric_limits<float>::infinity();
    *mean_depth = 0.0;
    *valid_count = 0;
    for (int y = 0; y < depth.rows; ++y) {
        const float* row = depth.ptr<float>(y);
        for (int x = 0; x < depth.cols; ++x) {
            const float value = row[x];
            if (!std::isfinite(value) || value <= 0.0F) {
                continue;
            }
            *min_depth = std::min(*min_depth, value);
            *max_depth = std::max(*max_depth, value);
            *mean_depth += value;
            ++*valid_count;
        }
    }
    if (*valid_count == 0) {
        return false;
    }
    *mean_depth /= static_cast<double>(*valid_count);
    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string config_path = argv[1];
    std::string image_path;
    std::string output_path = "yolo26_depth_result.jpg";
    std::string model_path_override;
    for (int i = 2; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        if (argument == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (argument == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (argument == "--model-path" && i + 1 < argc) {
            model_path_override = argv[++i];
        } else {
            std::cerr << "Error: Unknown or incomplete option: "
                << argument << '\n';
            print_usage(argv[0]);
            return 1;
        }
    }

    std::unique_ptr<VisionService> service = VisionService::Create(
        config_path,
        model_path_override,
        true);
    if (!service) {
        std::cerr << "Error: "
            << VisionService::LastCreateError() << '\n';
        return 1;
    }
    if (image_path.empty()) {
        image_path = service->GetDefaultImage();
    }
    if (image_path.empty()) {
        std::cerr << "Error: No input image. Use --image <path> "
            << "or set test_image in config.\n";
        return 1;
    }

    const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not load image: "
            << image_path << '\n';
        return 1;
    }

    VisionServiceResponse response;
    if (service->Infer(image, &response) != VISION_SERVICE_OK) {
        std::cerr << "Error: " << service->LastError() << '\n';
        return 1;
    }
    if (response.results.size() != 1) {
        std::cerr << "Error: Model did not return one depth map.\n";
        return 1;
    }
    const auto* result = std::get_if<vision::DepthMap>(
        &response.results.front());
    if (result == nullptr || result->map == nullptr ||
        result->map->empty() || result->map->type() != CV_32FC1 ||
        result->map->size() != image.size()) {
        std::cerr << "Error: Invalid metric depth result.\n";
        return 1;
    }

    float min_depth = 0.0F;
    float max_depth = 0.0F;
    double mean_depth = 0.0;
    size_t valid_count = 0;
    if (!depth_statistics(
            *result->map,
            &min_depth,
            &max_depth,
            &mean_depth,
            &valid_count)) {
        std::cerr << "Error: Depth map has no valid values.\n";
        return 1;
    }

    cv::Mat visualization;
    if (!service->SupportsDraw() ||
        service->Draw(image, response, &visualization) !=
            VISION_SERVICE_OK ||
        visualization.empty()) {
        std::cerr << "Error: Draw failed: "
            << service->LastError() << '\n';
        return 1;
    }
    if (!cv::imwrite(output_path, visualization)) {
        std::cerr << "Error: Could not save output: "
            << output_path << '\n';
        return 1;
    }

    std::cout << "Depth: min=" << min_depth
        << " m, max=" << max_depth
        << " m, mean=" << mean_depth
        << " m, valid=" << valid_count << '\n'
        << "Saved: " << output_path << '\n';
    return 0;
}
