/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <variant>

#include <opencv2/imgcodecs.hpp>

#include "vision_service.h"  // NOLINT(build/include_subdir)

namespace {
void usage(const char* program) {
    std::cout
        << "Usage: " << program << " <config.yaml> [options]\n"
        << "  --model-path <path>  Override model_path\n"
        << "  --image <path>       Default: test_image from YAML\n"
        << "  --output <path>      Default: scrfd_result.jpg\n"
        << "  --help               Show help\n";
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) == "--help") {
        usage(argv[0]);
        return argc < 2 ? 1 : 0;
    }
    std::string model_path, image_path, output_path = "scrfd_result.jpg";
    for (int i = 2; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--help") {
            usage(argv[0]);
            return 0;
        }
        if (option == "--model-path" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (option == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (option == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else {
            std::cerr << "Unknown or incomplete option: " << option << '\n';
            return 1;
        }
    }
    try {
        auto service = VisionService::Create(argv[1], model_path, false);
        if (!service) {
            std::cerr << VisionService::LastCreateError() << '\n';
            return 1;
        }
        if (image_path.empty()) image_path = service->GetDefaultImage();
        if (image_path.empty()) {
            std::cerr << "No image; use --image or set test_image in YAML\n";
            return 1;
        }
        const auto image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Could not read image: " << image_path << '\n';
            return 1;
        }
        VisionServiceResponse response;
        if (service->Infer(image, &response) != VISION_SERVICE_OK ||
            !response.ok) {
            std::cerr << "Infer failed: " << service->LastError() << '\n';
            return 1;
        }
        std::cout << "Faces: " << response.results.size() << '\n';
        for (const auto& item : response.results) {
            const auto* face = std::get_if<vision::Pose>(&item);
            if (!face || face->keypoints.size() != 5) {
                std::cerr << "Expected a face with five landmarks\n";
                return 1;
            }
            std::cout << "  score: " << std::fixed << std::setprecision(4)
                << face->score << ", box: [" << face->bbox.x1 << ", "
                << face->bbox.y1 << ", " << face->bbox.x2 << ", "
                << face->bbox.y2 << "], landmarks: 5\n";
        }
        cv::Mat output = image.clone();
        if (!response.results.empty() &&
            (service->Draw(image, response, &output) != VISION_SERVICE_OK ||
            output.empty())) {
            std::cerr << "Draw failed: " << service->LastError() << '\n';
            return 1;
        }
        if (!cv::imwrite(output_path, output)) {
            std::cerr << "Could not write output: " << output_path << '\n';
            return 1;
        }
        std::cout << "Saved: " << output_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
