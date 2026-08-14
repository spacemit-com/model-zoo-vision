/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include "vision_service.h"

namespace {

void usage(const char* program) {
    std::cout
        << "Usage: " << program << " <config.yaml> [options]\n"
        << "  --model-path <path>  Override model_path\n"
        << "  --image <path>       Override test_image\n"
        << "  --output <path>      Output image (default: yolop_result.jpg)\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }
    const std::string config_path = argv[1];
    std::string model_path;
    std::string image_path;
    std::string output_path = "yolop_result.jpg";
    for (int i = 2; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--model-path" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (argument == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (argument == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (argument == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown or incomplete argument: " << argument << std::endl;
            return 1;
        }
    }

    std::unique_ptr<VisionService> service =
        VisionService::Create(config_path, model_path, false);
    if (!service) {
        std::cerr << "Create failed: " << VisionService::LastCreateError() << std::endl;
        return 1;
    }
    if (image_path.empty()) image_path = service->GetDefaultImage();
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Could not read image: " << image_path << std::endl;
        return 1;
    }

    VisionServiceResponse response;
    const VisionServiceStatus status = service->Infer(image, &response);
    if (status != VISION_SERVICE_OK || !response.ok) {
        std::cerr << "Infer failed: " << service->LastError() << std::endl;
        return 1;
    }
    size_t detections = 0;
    size_t masks = 0;
    for (const vision::Result& result : response.results) {
        detections += std::holds_alternative<vision::Detection>(result) ? 1 : 0;
        masks += std::holds_alternative<vision::Segmentation>(result) ? 1 : 0;
    }
    if (masks != 2) {
        std::cerr << "Expected two semantic masks, got " << masks << std::endl;
        return 1;
    }
    std::cout << "Detections: " << detections << ", masks: " << masks << std::endl;
    if (!service->SupportsDraw()) {
        std::cerr << "YOLOP must support Draw" << std::endl;
        return 1;
    }
    cv::Mat output;
    if (service->Draw(image, response, &output) != VISION_SERVICE_OK || output.empty()) {
        std::cerr << "Draw failed: " << service->LastError() << std::endl;
        return 1;
    }
    if (!cv::imwrite(output_path, output)) {
        std::cerr << "Could not write output: " << output_path << std::endl;
        return 1;
    }
    std::cout << "Saved: " << output_path << std::endl;
    return 0;
}
