/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <memory>
#include <string>
#include <variant>

#include <opencv2/opencv.hpp>

#include "vision_service.h"

namespace {

void print_usage(const char* program) {
    std::cout << "Usage: " << program
        << " <config.yaml> [--model-path path] [--image path]"
        << " [--output path]\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    const std::string config_path = argv[1];
    std::string model_path;
    std::string image_path;
    std::string output_path = "yolo26_seg_result.jpg";
    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--model-path" && index + 1 < argc) {
            model_path = argv[++index];
        } else if (argument == "--image" && index + 1 < argc) {
            image_path = argv[++index];
        } else if (argument == "--output" && index + 1 < argc) {
            output_path = argv[++index];
        } else if (argument == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown or incomplete option: " << argument << '\n';
            return 1;
        }
    }

    std::unique_ptr<VisionService> service =
        VisionService::Create(config_path, model_path, true);
    if (!service) {
        std::cerr << "Create failed: "
            << VisionService::LastCreateError() << '\n';
        return 1;
    }
    if (image_path.empty()) {
        image_path = service->GetDefaultImage();
    }
    const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Cannot read image: " << image_path << '\n';
        return 1;
    }

    std::cout << "Running instance segmentation...\n";
    VisionServiceResponse response;
    if (service->Infer(image, &response) != VISION_SERVICE_OK) {
        std::cerr << "Infer failed: " << service->LastError() << '\n';
        return 1;
    }
    for (const auto& item : response.results) {
        const auto* instance = std::get_if<vision::Segmentation>(&item);
        if (instance == nullptr || instance->mask == nullptr ||
            instance->mask->type() != CV_8UC1 ||
            instance->mask->size() != image.size()) {
            std::cerr << "Invalid instance mask\n";
            return 1;
        }
    }
    if (response.results.empty()) {
        std::cout << "No objects detected\n";
    } else {
        std::cout << "Detected " << response.results.size()
            << " objects with segmentation masks\n";
    }
    cv::Mat visualization;
    if (service->Draw(image, response, &visualization) !=
            VISION_SERVICE_OK ||
        visualization.empty() ||
        !cv::imwrite(output_path, visualization)) {
        std::cerr << "Cannot save visualization: "
            << service->LastError() << '\n';
        return 1;
    }
    std::cout << "Result image saved to: " << output_path << '\n';
    return 0;
}
