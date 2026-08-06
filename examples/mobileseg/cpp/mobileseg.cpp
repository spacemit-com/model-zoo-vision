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

void print_usage(const char* program_name) {
    std::cout
        << "Usage: " << program_name
        << " <config_yaml> [options]\n"
        << "  --model-path <path>   Override model_path in yaml\n"
        << "  --image <path>        Input image path\n"
        << "  --output <path>       Output image path"
        << " (default: mobileseg_result.jpg)\n"
        << "  --help                Show this help message\n"
        << "\nExample:\n"
        << "  " << program_name
        << " examples/mobileseg/config/mobileseg.yaml\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string config_path = argv[1];
    std::string image_path;
    std::string output_path = "mobileseg_result.jpg";
    std::string model_path_override;

    for (int i = 2; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        if (argument == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (
            argument == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (
            argument == "--model-path" && i + 1 < argc) {
            model_path_override = argv[++i];
        } else {
            std::cerr << "Error: Unknown or incomplete option: "
            << argument << '\n';
            print_usage(argv[0]);
            return 1;
        }
    }

    std::unique_ptr<VisionService> service =
        VisionService::Create(
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
        std::cerr
            << "Error: No input image. Use --image <path> "
            << "or set test_image in config.\n";
        return 1;
    }

    const cv::Mat image = cv::imread(
        image_path,
        cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not load image: "
            << image_path << '\n';
        return 1;
    }

    VisionServiceResponse response;
    const VisionServiceStatus infer_status =
        service->Infer(image, &response);
    if (infer_status != VISION_SERVICE_OK) {
        std::cerr << "Error: " << service->LastError()
            << '\n';
        return 1;
    }
    if (response.results.empty()) {
        std::cerr
            << "Error: Model returned no semantic masks.\n";
        return 1;
    }

    for (const vision::Result& item : response.results) {
        const auto* segmentation =
            std::get_if<vision::Segmentation>(&item);
        if (segmentation == nullptr ||
            segmentation->mask == nullptr ||
            segmentation->mask->empty() ||
            segmentation->mask->type() != CV_8UC1 ||
            segmentation->mask->size() != image.size()) {
            std::cerr
                << "Error: Invalid segmentation result.\n";
            return 1;
        }
    }

    if (!service->SupportsDraw()) {
        std::cerr
            << "Error: Model does not support Draw().\n";
        return 1;
    }
    cv::Mat visualization;
    const VisionServiceStatus draw_status =
        service->Draw(
            image,
            response,
            &visualization);
    if (draw_status != VISION_SERVICE_OK ||
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

    std::cout << "Saved: " << output_path
            << " (" << response.results.size()
            << " semantic class mask(s), "
            << image.cols << 'x' << image.rows << ")\n";
    return 0;
}
